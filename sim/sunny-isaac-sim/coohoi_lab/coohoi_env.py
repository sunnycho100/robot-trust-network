"""CooHOI's box carry in Isaac Lab 3.0 (Isaac Sim 6.2 PhysX): a DirectMARLEnv with two HUMANOID_28_CFG humanoids and
one box ("pair", for TwoAgent.pth), or one humanoid and a cube ("single", for SingleAgent.pth). No training: the
policy, observations and action mapping are CooHOI's (policy.py, obs.py, joint_map.py).
Usage: cfg = make_cfg("pair", num_envs=16); then inside isaaclab.app.launch_simulation(cfg, args): env = CoohoiEnv(cfg).
play.py and compare_reference.py do exactly that. Nothing here has run yet (no GPU while writing); see README.md.

Layout, rebuilt from share_humanoid_amp_carryobject.py: CooHOI puts each humanoid of a pair in its own Isaac Gym env,
the left one (offset env) 10 m from the right one (base env) along x, the box between them. The two humanoids and the
base env's box share a collision group, and the offset env's box is a ghost in its own group whose observed state
is overwritten with the base env's box (+10 m). So in training both humanoids touched one physical box, as they do
here in one env. unverified: that Isaac Gym collision groups really span the two envs (CooHOI's comment says so).
"""
from __future__ import annotations

import math
import os

import torch
from isaaclab_assets import HUMANOID_28_CFG
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sim.schemas import PhysxCollisionCfg, PhysxRigidBodyCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
from isaaclab.utils import configclass, replace

import joint_map as jm
import obs as ob

HOME_X = {"left": -5.0, "right": 5.0, "single": 0.0}  # humanoid start x around the layout centre (CooHOI: 10 m apart)
ROOT_H = 0.89  # CooHOI start_pose height of the pelvis
FRICTION = RigidBodyMaterialBaseCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0)  # CooHOI plane, MJCF geoms
CONTACT = PhysxCollisionCfg(contact_offset=0.02, rest_offset=0.0)  # CooHOI sim.physx contact/rest offset, all shapes


def _humanoid(agent, effort):
    """HUMANOID_28_CFG with CooHOI's MJCF gains. effort: 'gear' caps each joint at its MJCF motor gear, 'unlimited'
    sets 1e9, 'usd' keeps the USD's limit. unverified: which one Isaac Gym used; compare_reference.py prints Gym's."""
    spawn = HUMANOID_28_CFG.spawn
    # Isaac Lab's Humanoid28 USD is rooted at the torso; CooHOI's MJCF (and its policy) at the pelvis. Point this at
    # amp_humanoid.xml converted with scripts/tools/convert_mjcf.py to keep CooHOI's own tree (see README).
    if os.environ.get("COOHOI_HUMANOID_USD"):
        spawn = replace(spawn, usd_path=os.environ["COOHOI_HUMANOID_USD"])
    return replace(
        HUMANOID_28_CFG,
        prim_path=f"{{ENV_REGEX_NS}}/Robot_{agent}",
        # CooHOI AssetOptions: angular_damping 0.01 (Isaac Sim's default is 0.05); max_angular_velocity 100 rad/s is
        # already Isaac Sim's default (5729.58 deg/s)
        spawn=replace(spawn, rigid_props=replace(spawn.rigid_props, angular_damping=0.01), collision_props=CONTACT),
        init_state=replace(HUMANOID_28_CFG.init_state, pos=(HOME_X[agent], 0.0, ROOT_H)),
        actuators={"body": ImplicitActuatorCfg(
            joint_names_expr=[".*"], stiffness=jm.gains("stiffness"), damping=jm.gains("damping"),
            armature=jm.gains("armature"), joint_effort_limit={"gear": jm.gains("gear"), "unlimited": 1e9, "usd": None}[effort],
            joint_velocity_limit=100.0)},  # as HUMANOID_28_CFG; unverified: Isaac Gym's MJCF dof velocity limit
    )


def _box(size, density):
    """CooHOI's box: create_box with AssetOptions(density=...); the other AssetOptions stay at Isaac Gym's defaults.
    unverified: those defaults (angular_damping 0.5, max_angular_velocity 64 rad/s) and the box friction (1.0)."""
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Box",
        spawn=sim_utils.CuboidCfg(
            size=size,
            rigid_props=[sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=False),
                         PhysxRigidBodyCfg(angular_damping=0.5, max_angular_velocity=math.degrees(64.0),
                                           max_depenetration_velocity=10.0, solver_position_iteration_count=4,
                                           solver_velocity_iteration_count=0)],
            mass_props=sim_utils.MassCfg(density=density),
            collision_props=[sim_utils.UsdPhysicsCollisionCfg(), CONTACT],
            physics_material=FRICTION,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.55, 0.25)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.6)),
    )


@configclass
class CoohoiSceneCfg(InteractiveSceneCfg):
    ground: AssetBaseCfg | None = AssetBaseCfg(
        prim_path="/World/ground", collision_group=-1, spawn=sim_utils.GroundPlaneCfg(physics_material=FRICTION))
    stage: AssetBaseCfg | None = None  # an external scene USD (our reconstructions) replaces the ground plane
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)))
    robot_left: ArticulationCfg | None = None
    robot_right: ArticulationCfg | None = None
    robot_single: ArticulationCfg | None = None
    box: RigidObjectCfg | None = None


@configclass
class CoohoiEnvCfg(DirectMARLEnvCfg):
    # CooHOI: sim dt 1/60 with physx substeps 2 (so physics at 120 Hz), controlFrequencyInv 2 (policy at 30 Hz).
    # Isaac Lab has no substeps; its migration guide maps that to dt 1/120 and twice the decimation.
    decimation = 4
    episode_length_s = 20.0  # episodeLength 600 control steps
    possible_agents = ["left", "right"]
    action_spaces = {"left": 28, "right": 28}
    observation_spaces = {"left": 299, "right": 299}
    state_space = 0
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120, render_interval=4, physics_material=FRICTION,
        physics=PhysxCfg(solver_type=1, bounce_threshold_velocity=0.2,
                         gpu_found_lost_pairs_capacity=2**23, gpu_total_aggregate_pairs_capacity=2**23))
    scene: CoohoiSceneCfg = CoohoiSceneCfg(num_envs=16, env_spacing=15.0, replicate_physics=True)
    mode: str = "pair"
    box_scale: float = 1.0  # CooHOI draws it per env: [0.9, 1] for the pair, [0.8, 1] single; one value here
    layout_origin: tuple = (0.0, 0.0, 0.0)  # where the task layout sits inside each env (e.g. a spot in a scan)


def make_cfg(mode="pair", num_envs=16, box_scale=1.0, physics_dt=1 / 120, decimation=4, effort="gear",
             scene_usd=None, origin=(0.0, 0.0, 0.0), seed=0):
    cfg = CoohoiEnvCfg()
    agents = ["left", "right"] if mode == "pair" else ["single"]
    cfg.mode, cfg.possible_agents, cfg.box_scale, cfg.layout_origin, cfg.seed = mode, agents, box_scale, tuple(origin), seed
    cfg.action_spaces, cfg.observation_spaces = {a: 28 for a in agents}, {a: 299 for a in agents}
    cfg.sim.dt, cfg.decimation, cfg.sim.render_interval = physics_dt, decimation, decimation
    cfg.scene.num_envs = num_envs
    for a in agents:
        setattr(cfg.scene, f"robot_{a}", _humanoid(a, effort))
    s = box_scale
    if mode == "pair":  # create_box(2.0, 0.5, 0.5), density 50, then set_actor_scale(s)
        cfg.scene.box = _box((2.0 * s, 0.5 * s, 0.5 * s), 50.0)  # unverified: that set_actor_scale also scales mass
    else:  # a 0.5 m cube scaled by s, density 0.8 * 50 / s^3, so always 5 kg
        cfg.scene.box = _box((0.5 * s,) * 3, 0.8 * 50.0 / s**3)
    if scene_usd:
        cfg.scene.ground = None
        cfg.scene.stage = AssetBaseCfg(prim_path="/World/Stage", collision_group=-1,
                                       spawn=sim_utils.UsdFileCfg(usd_path=scene_usd))
    return cfg


class CoohoiEnv(DirectMARLEnv):
    cfg: CoohoiEnvCfg

    def __init__(self, cfg: CoohoiEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.robots = {a: self.scene[f"robot_{a}"] for a in self.possible_agents}
        self.box = self.scene["box"]
        r = self.robots[self.possible_agents[0]]
        self.jmap = jm.JointMap(r.joint_names, r.body_names)
        print("[coohoi] " + self.jmap.report().replace("\n", "\n[coohoi] "))
        if r.body_names[0] != "pelvis":
            raise RuntimeError(f"root link is {r.body_names[0]!r}, not the MJCF root 'pelvis': root writes would be wrong")
        d, n, s = self.device, self.num_envs, cfg.box_scale
        self.body_ids = torch.tensor(self.jmap.bodies_gym_from_lab, device=d)
        self.lab_from_gym = torch.tensor(self.jmap.joints_lab_from_gym, device=d)
        off, sc = jm.pd_action_offset_scale()
        self.pd_offset, self.pd_scale = torch.tensor(off, device=d), torch.tensor(sc, device=d)
        full = lambda v: torch.full((n,), v, device=d)  # noqa: E731
        if cfg.mode == "pair":
            self.width, self.length, self.height = full(0.5 * s), full(2.0 * s), full(0.5 * s)
            self.geo = {a: ob.pair_geometry(self.width, self.length, a) for a in self.possible_agents}
            self.density = full(2 * 50.0)  # CooHOI observes asset_density * 2
        else:
            self.width = self.length = self.height = full(0.5 * s)
            self.geo = {"single": ob.single_geometry(self.width, self.length, self.height)}
            self.density = full(0.8 * 50.0 / s**3)
        # terminations (compute_humanoid_reset): fall = a body other than hands and feet below 0.2 m (head 0.3 m)
        self.term_h = torch.full((15,), 0.2, device=d)
        self.term_h[jm.BODIES.index("head")] = 0.3
        self.check_h = torch.ones(15, dtype=torch.bool, device=d)
        self.check_h[[jm.BODIES.index(b) for b in ("right_foot", "left_foot", "right_hand", "left_hand")]] = False
        self.hand_ids = [jm.BODIES.index("left_hand"), jm.BODIES.index("right_hand")]
        self.origin = self.scene.env_origins + torch.tensor(cfg.layout_origin, device=d)
        self.tar_pos, self.tar_rot = torch.zeros(n, 3, device=d), torch.zeros(n, 4, device=d)
        self.tar_rot[:, 3] = 1.0
        self.prev_box_pos = torch.zeros(n, 3, device=d)
        self.targets = {a: torch.zeros(n, 28, device=d) for a in self.possible_agents}
        self.box_max_h, self.ep_dist, self.ep_max_h = torch.zeros(n, device=d), torch.zeros(n, device=d), torch.zeros(n, device=d)

    # --- state in the task frame (env origin + layout_origin at 0; MJCF body order; xyzw quaternions) ---

    def bodies(self, a):
        dat, i = self.robots[a].data, self.body_ids
        return (dat.body_pos_w.torch[:, i] - self.origin.unsqueeze(1), ob.to_xyzw(dat.body_quat_w.torch[:, i]),
                dat.body_lin_vel_w.torch[:, i], dat.body_ang_vel_w.torch[:, i])

    def box_state(self):
        dat = self.box.data
        return torch.cat([dat.root_pos_w.torch - self.origin, ob.to_xyzw(dat.root_quat_w.torch),
                          dat.root_lin_vel_w.torch, dat.root_ang_vel_w.torch], -1)

    def write_state(self, env_ids, roots, dof_pos, dof_vel, box, tar_pos, tar_rot):
        """Set the full task state (compare_reference.py). roots {agent: (k, 13)} pelvis pos, xyzw quat, lin and ang
        vel; dof_pos and dof_vel {agent: (k, 28)} in MJCF order; box (k, 13); all positions in the task frame."""
        o = self.origin[env_ids]
        for a in self.possible_agents:
            r = self.robots[a]
            r.write_root_link_pose_to_sim_index(root_pose=torch.cat([roots[a][:, :3] + o, ob.from_xyzw(roots[a][:, 3:7])], -1), env_ids=env_ids)
            r.write_root_com_velocity_to_sim_index(root_velocity=roots[a][:, 7:13], env_ids=env_ids)  # unverified: Gym's root velocity is the COM's
            r.write_joint_position_to_sim_index(position=dof_pos[a][:, self.lab_from_gym], env_ids=env_ids)
            r.write_joint_velocity_to_sim_index(velocity=dof_vel[a][:, self.lab_from_gym], env_ids=env_ids)
        self.box.write_root_pose_to_sim_index(root_pose=torch.cat([box[:, :3] + o, ob.from_xyzw(box[:, 3:7])], -1), env_ids=env_ids)
        self.box.write_root_velocity_to_sim_index(root_velocity=box[:, 7:13], env_ids=env_ids)
        self.tar_pos[env_ids], self.tar_rot[env_ids] = tar_pos, tar_rot
        self.prev_box_pos[env_ids] = box[:, :3]

    # --- DirectMARLEnv hooks ---

    def _pre_physics_step(self, actions):
        self.prev_box_pos[:] = self.box.data.root_pos_w.torch - self.origin
        for a in self.possible_agents:
            pd = self.pd_offset + self.pd_scale * actions[a].clamp(-1.0, 1.0)  # rl_games and VecTask clip to +-1
            self.targets[a] = pd[:, self.lab_from_gym]

    def _apply_action(self):
        for a in self.possible_agents:
            self.robots[a].set_joint_position_target_index(target=self.targets[a])

    def _get_observations(self):
        box, out = self.box_state(), {}
        for a in self.possible_agents:
            p, q, v, w = self.bodies(a)
            root = torch.cat([p[:, 0], q[:, 0]], -1)
            task = (ob.pair_task_obs if self.cfg.mode == "pair" else ob.single_task_obs)(
                root, box, self.tar_pos, self.tar_rot, self.geo[a], self.density)
            out[a] = torch.cat([ob.humanoid_obs(p, q, v, w), task], -1)
        return out

    def _get_states(self):
        return None

    def _get_rewards(self):
        return {a: torch.zeros(self.num_envs, device=self.device) for a in self.possible_agents}  # play only

    def _get_dones(self):
        box_pos = self.box.data.root_pos_w.torch - self.origin
        box_speed = ((box_pos - self.prev_box_pos)[:, :2] / self.step_dt).norm(dim=-1)
        failed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for a in self.possible_agents:
            z = self.bodies(a)[0][..., 2]
            fallen = ((z < self.term_h) & self.check_h).any(-1)
            # ponytail: height only; CooHOI also needs a contact force on that body. Add a ContactSensor if it matters.
            kicked = (box_speed > 1.0) & (box_pos[:, 2] < 0.3) & (z[:, self.hand_ids].mean(-1) > 0.5)
            failed |= fallen | kicked  # a pair resets together (compute_humanoid_reset syncs base and offset env)
        failed &= self.episode_length_buf > 1
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        self.box_max_h = torch.maximum(self.box_max_h, box_pos[:, 2])
        self.ep_dist, self.ep_max_h = (box_pos - self.tar_pos).norm(dim=-1), self.box_max_h.clone()  # read by play.py
        return {a: failed for a in self.possible_agents}, {a: time_out for a in self.possible_agents}

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        k, d = len(env_ids), self.device
        rand = lambda lo, hi: lo + (hi - lo) * torch.rand(k, device=d)  # noqa: E731
        zeros = torch.zeros(k, 28, device=d)
        roots = {}
        for a in self.possible_agents:  # stateInit Default: start pose, zero joints, no velocity
            roots[a] = torch.zeros(k, 13, device=d)
            roots[a][:, 0], roots[a][:, 2], roots[a][:, 6] = HOME_X[a], ROOT_H, 1.0
        # _reset_box / _reset_target, randomize=True, around the layout centre (pair: between the two humanoids)
        pair = self.cfg.mode == "pair"
        dist, theta = rand(0.5, 2.0 if pair else 10.0), rand(0.0, 2 * math.pi)
        box = torch.zeros(k, 13, device=d)
        box[:, 0], box[:, 1] = dist * torch.cos(theta), dist * torch.sin(theta)
        box[:, 2] = 0.6 if pair else self.height[env_ids] / 2.0
        box[:, 3:7] = ob.yaw_quat(0.25 * math.pi * (torch.rand(k, device=d) - 0.5) if pair else rand(0.0, 2 * math.pi))
        dist, theta = rand(1.5, 4.0 if pair else 10.0), rand(0.0, 2 * math.pi)
        tar = torch.stack([dist * torch.cos(theta) + box[:, 0], dist * torch.sin(theta) + box[:, 1],
                           (self.width if pair else self.height)[env_ids] / 2.0], -1)
        tar_rot = ob.yaw_quat(0.25 * math.pi * (torch.rand(k, device=d) - 0.5) if pair else rand(0.0, 2 * math.pi))
        self.write_state(env_ids, roots, {a: zeros for a in self.possible_agents}, {a: zeros for a in self.possible_agents}, box, tar, tar_rot)
        self.box_max_h[env_ids] = 0.0
