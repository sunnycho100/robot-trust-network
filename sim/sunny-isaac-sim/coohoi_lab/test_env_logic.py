"""python test_env_logic.py: runs coohoi_env.py's own logic on CPU against a fake Isaac Lab (no Isaac Sim): config
building, the joint and body permutation on a breadth-first articulation, reset sampling, actions to PD targets,
observations, terminations, and compare_reference.lab_state's frame mapping. The fakes only stand in for the Isaac
Lab calls coohoi_env.py makes (zero-joint-angle kinematics, no physics), so this says nothing about Isaac Lab itself."""
import copy
import math
import sys
import types
import xml.etree.ElementTree as ET
from collections import deque

import numpy as np
import torch

import joint_map as jm
import obs as ob

torch.manual_seed(0)


class Cfg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def configclass(cls):
    def __init__(self, **kw):
        for k in dir(cls):
            v = getattr(cls, k)
            if not k.startswith("__") and not callable(v):
                setattr(self, k, copy.deepcopy(v))
        self.__dict__.update(kw)

    cls.__init__ = __init__
    return cls


def replace(o, **kw):
    o = copy.deepcopy(o)
    o.__dict__.update(kw)
    return o


# MJCF tree at zero joint angles: body offsets from the pelvis, and a breadth-first order like a PhysX articulation
root_el = ET.parse(jm.mjcf_path()).getroot().find("worldbody").find("body")
OFFSET, BFS, q = {}, [], deque([(root_el, np.zeros(3))])
while q:
    b, p = q.popleft()
    BFS.append(b)
    OFFSET[b.get("name")] = p
    q.extend((c, p + np.array(c.get("pos", "0 0 0").split(), float)) for c in b.findall("body"))
LAB_BODIES = [b.get("name") for b in BFS] + ["dummy_link"]
LAB_JOINTS = [j.get("name") for b in BFS for j in b.findall("joint")]


class P:  # ProxyArray stand-in
    def __init__(self, t):
        self.torch = t


class FakeRobot:
    def __init__(self, n):
        self.joint_names, self.body_names, self.n = LAB_JOINTS, LAB_BODIES, n
        self.root = torch.zeros(n, 7)
        self.root[:, 6] = 1.0
        self.target = None
        self.offs = torch.tensor(np.array([OFFSET.get(b, np.zeros(3)) for b in LAB_BODIES]), dtype=torch.float32)

    @property
    def data(self):
        k = len(LAB_BODIES)
        q = self.root[:, 3:7].unsqueeze(1).expand(-1, k, 4)
        pos = self.root[:, None, :3] + ob.quat_rotate(q, self.offs.expand(self.n, k, 3))
        return Cfg(body_pos_w=P(pos), body_quat_w=P(q.clone()), body_lin_vel_w=P(torch.zeros(self.n, k, 3)),
                   body_ang_vel_w=P(torch.zeros(self.n, k, 3)))

    def write_root_link_pose_to_sim_index(self, root_pose, env_ids):
        self.root[env_ids] = root_pose

    def write_root_com_velocity_to_sim_index(self, root_velocity, env_ids):
        pass

    def write_joint_position_to_sim_index(self, position, env_ids):
        assert position.shape[1] == 28

    def write_joint_velocity_to_sim_index(self, velocity, env_ids):
        assert velocity.shape[1] == 28

    def set_joint_position_target_index(self, target):
        self.target = target


class FakeBox:
    def __init__(self, n):
        self.state = torch.zeros(n, 13)

    @property
    def data(self):
        s = self.state
        return Cfg(root_pos_w=P(s[:, :3].clone()), root_quat_w=P(s[:, 3:7].clone()), root_lin_vel_w=P(s[:, 7:10]),
                   root_ang_vel_w=P(s[:, 10:13]))

    def write_root_pose_to_sim_index(self, root_pose, env_ids):
        self.state[env_ids, :7] = root_pose

    def write_root_velocity_to_sim_index(self, root_velocity, env_ids):
        self.state[env_ids, 7:] = root_velocity


class DirectMARLEnv:
    def __init__(self, cfg, render_mode=None, **kw):
        n = cfg.scene.num_envs
        self.cfg, self.device, self.num_envs, self.possible_agents = cfg, "cpu", n, cfg.possible_agents
        ents = {f"robot_{a}": FakeRobot(n) for a in cfg.possible_agents}
        ents["box"] = FakeBox(n)
        self.scene = Cfg(env_origins=torch.arange(n, dtype=torch.float32).unsqueeze(1) * torch.tensor([15.0, 0, 0]))
        self.scene.__class__ = type("S", (), {"__getitem__": lambda s, k: ents[k]})
        self.scene.__dict__["env_origins"] = torch.arange(n, dtype=torch.float32).unsqueeze(1) * torch.tensor([15.0, 0, 0])
        self.episode_length_buf = torch.zeros(n, dtype=torch.long)
        self.step_dt = cfg.sim.dt * cfg.decimation
        self.max_episode_length = math.ceil(cfg.episode_length_s / self.step_dt)

    def _reset_idx(self, env_ids):
        self.episode_length_buf[env_ids] = 0


def module(name, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    sys.modules[name] = m
    return m


hum = Cfg(prim_path="{ENV_REGEX_NS}/Robot", spawn=Cfg(rigid_props=Cfg(max_depenetration_velocity=10.0)),
          init_state=Cfg(pos=(0, 0, 0.8)), actuators={})
module("isaaclab_assets", HUMANOID_28_CFG=hum)
module("isaaclab_physx")
module("isaaclab_physx.physics", PhysxCfg=Cfg)
module("isaaclab_physx.sim")
module("isaaclab_physx.sim.schemas", PhysxCollisionCfg=Cfg, PhysxRigidBodyCfg=Cfg)
module("isaaclab")
sim = module("isaaclab.sim", SimulationCfg=Cfg, **{k: Cfg for k in (
    "CuboidCfg", "UsdPhysicsRigidBodyCfg", "MassCfg", "UsdPhysicsCollisionCfg", "PreviewSurfaceCfg", "GroundPlaneCfg",
    "DomeLightCfg", "UsdFileCfg")})
sys.modules["isaaclab"].sim = sim
module("isaaclab.sim.spawners")
module("isaaclab.sim.spawners.materials", RigidBodyMaterialBaseCfg=Cfg)
module("isaaclab.actuators", ImplicitActuatorCfg=Cfg)
module("isaaclab.assets", ArticulationCfg=Cfg, AssetBaseCfg=Cfg, RigidObjectCfg=type("R", (Cfg,), {"InitialStateCfg": Cfg}))
module("isaaclab.envs", DirectMARLEnv=DirectMARLEnv, DirectMARLEnvCfg=type("E", (), {}))
module("isaaclab.scene", InteractiveSceneCfg=type("S", (), {}))
module("isaaclab.utils", configclass=configclass, replace=replace)
module("isaaclab.app", add_launcher_args=lambda p: None, launch_simulation=None)

from coohoi_env import CoohoiEnv, make_cfg  # noqa: E402

# ---- configs ----
c = make_cfg("pair", 4, 0.95)
assert c.possible_agents == ["left", "right"] and c.scene.robot_single is None and c.scene.robot_left.prim_path.endswith("Robot_left")
assert c.scene.box.spawn.size == (1.9, 0.475, 0.475) and c.sim.dt == 1 / 120 and c.decimation == 4
assert c.scene.robot_right.init_state.pos == (5.0, 0.0, 0.89) and c.scene.robot_left.actuators["body"].stiffness["abdomen_x"] == 1000
s = make_cfg("single", 3, 1.0, 1 / 60, 2, effort="usd", scene_usd="/x.usda", origin=(1, 2, 0.5))
assert s.scene.robot_left is None and s.scene.ground is None and s.scene.stage.spawn.usd_path == "/x.usda"
assert s.scene.box.spawn.mass_props.density == 40.0 and s.scene.robot_single.actuators["body"].joint_effort_limit is None
assert make_cfg("pair").scene.robot_single is None, "configs must not leak between calls"

# ---- pair env ----
env = CoohoiEnv(make_cfg("pair", 64, 1.0))
assert env.jmap.joints_lab_from_gym != list(range(28)), "the fake articulation is breadth first"
names = [LAB_BODIES[i] for i in env.jmap.bodies_gym_from_lab]
assert names == jm.BODIES
ids = torch.arange(64)
env._reset_idx(ids)
o = env._get_observations()
assert set(o) == {"left", "right"} and all(v.shape == (64, 299) for v in o.values())
assert torch.allclose(o["left"][:, 0], torch.tensor(0.89)) and torch.allclose(o["right"][:, 223], torch.tensor(100.0))
p_l, p_r = env.bodies("left")[0], env.bodies("right")[0]
assert torch.allclose(p_l[:, 0, 0], torch.tensor(-5.0)) and torch.allclose(p_r[:, 0, 0], torch.tensor(5.0))
assert (p_l[:, jm.BODIES.index("left_thigh"), 1] > p_l[:, 0, 1]).all(), "play.py's quaternion-order check holds"
box = env.box_state()
dist = box[:, :2].norm(dim=-1)
assert (dist >= 0.5).all() and (dist <= 2.0).all() and torch.allclose(box[:, 2], torch.tensor(0.6))
assert (ob.calc_heading(box[:, 3:7]).abs() <= math.pi / 8 + 1e-6).all()
td = (env.tar_pos[:, :2] - box[:, :2]).norm(dim=-1)
assert (td >= 1.5).all() and (td <= 4.0).all() and torch.allclose(env.tar_pos[:, 2], torch.tensor(0.25))
# the obs equals obs.py on the same gym-order state (permutation applied in the right direction)
p, q, v, w = env.bodies("right")
want = torch.cat([ob.humanoid_obs(p, q, v, w), ob.pair_task_obs(torch.cat([p[:, 0], q[:, 0]], -1), box, env.tar_pos,
                                                                  env.tar_rot, env.geo["right"], env.density)], -1)
assert torch.allclose(o["right"], want)
# actions -> PD targets in Isaac Lab joint order
env._pre_physics_step({"left": torch.zeros(64, 28), "right": torch.full((64, 28), 3.0)})
env._apply_action()
t_l, t_r = env.robots["left"].target, env.robots["right"].target
off, sc = jm.pd_action_offset_scale()
e = LAB_JOINTS.index("right_elbow")
assert abs(t_l[0, e] - off[jm.JOINT_NAMES.index("right_elbow")]) < 1e-6, "zero action -> offset"
assert abs(t_r[0, e] - (off[9] + sc[9])) < 1e-6, "action clipped to +1"
# terminations: nothing at reset; a fallen right humanoid resets the pair after step 1
env.episode_length_buf[:] = 5
term, trunc = env._get_dones()
assert not term["left"].any() and not trunc["left"].any()
env.robots["right"].root[:3, 2] = 0.1
term, _ = env._get_dones()
assert term["left"][:3].all() and term["right"][:3].all() and not term["left"][3:].any()
env.episode_length_buf[:] = env.max_episode_length - 1
assert env.max_episode_length == 600 and env._get_dones()[1]["right"].all()
print("pair env logic ok (BFS permutation, reset ranges, obs, PD targets, terminations)")

# ---- single env ----
env1 = CoohoiEnv(make_cfg("single", 32, 0.8, origin=(3.0, -2.0, 0.4)))
env1._reset_idx(torch.arange(32))
o1 = env1._get_observations()["single"]
assert torch.allclose(o1[:, 0], torch.tensor(0.89)) and torch.allclose(o1[:, 223], torch.tensor(40.0 / 0.8**3))
b1 = env1.box_state()
assert torch.allclose(b1[:, 2], torch.tensor(0.2)) and (b1[:, :2].norm(dim=-1) <= 10.0).all()
assert torch.allclose(env1.robots["single"].root[:, 2], torch.tensor(0.89 + 0.4)), "layout origin z lifts the start"
print("single env logic ok")

# ---- compare_reference.lab_state: Gym's two env frames -> one shared frame ----
sys.argv = ["compare_reference.py", "x.npz"]
import compare_reference as cr  # noqa: E402

T, shared_box = 2, np.array([[2.0, 1.0, 0.3], [2.2, 1.1, 0.4]])
ref = {"base_env_ids": np.array([1]), "offset_env_ids": np.array([0]),
       "root_states": np.zeros((T, 2, 13)), "dof_pos": np.zeros((T, 2, 28)), "dof_vel": np.zeros((T, 2, 28)),
       "box_states": np.zeros((T, 2, 13)), "tar_pos": np.zeros((T, 2, 3)), "tar_rot": np.tile([0, 0, 0, 1.0], (T, 2, 1))}
for t in range(T):  # right env frame = shared; left env frame = shared + 10 m in x (CooHOI's copy)
    ref["box_states"][t, 1, :3], ref["box_states"][t, 0, :3] = shared_box[t], shared_box[t] + [10, 0, 0]
    ref["root_states"][t, 1, :3], ref["root_states"][t, 0, :3] = [7, 1, 0.9], [-3 + 10, 1, 0.9]
    ref["tar_pos"][t, :, :] = [4, 3, 0.25]
    ref["tar_pos"][t, 0, 0] += 10
roots, _, _, box, tar, _ = cr.lab_state(ref, 1, cr.agents_of(ref))
for a, x in (("left", -3.0), ("right", 7.0)):  # positions relative to the box are what the agents saw
    assert np.allclose((roots[a][0, :3] - box[0, :3]).numpy(), np.array([x, 1, 0.9]) - shared_box[1]), a
assert np.allclose((tar[0] - box[0, :3]).numpy(), np.array([4, 3, 0.25]) - shared_box[1])
print("compare_reference frame mapping ok")
