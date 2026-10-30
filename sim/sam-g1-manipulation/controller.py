"""G1 arm IK and jerk-limited commands over the frozen BiGym balance controller.

This is a manipulation bring-up adapter, not a collision-aware grasp planner.
All live motion goes through env.step; no robot state is teleported.
"""

from __future__ import annotations

import mujoco
import numpy as np
from bigym.loco import make
from bigym.loco.agent.cli import EnvToolsConfig
from bigym.loco.agent.envtools import EnvTools
from ruckig import InputParameter, OutputParameter, Result, Ruckig, Synchronization


def collision_aware_ik(env):
    """Robot-only Mink solver with the live robot's physical collision shapes.

    The upstream teleoperation solver disables every collision mask. Restore
    masks by body name and within-body geom order, including unnamed geoms.
    No scene geometry or object state enters this stripped kinematic model.
    """
    import mink

    from robot_kinematics import RobotArmIK

    ik = RobotArmIK(env)
    model, live = ik.model, env.model
    limit_to_commandable_range(ik, live)
    groups = {"left": [], "right": [], "trunk": []}
    for body in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body)
        if name is None:
            raise RuntimeError("IK collision model contains an unnamed robot body")
        live_body = mujoco.mj_name2id(live, mujoco.mjtObj.mjOBJ_BODY, name)
        if live_body < 0:
            raise RuntimeError(f"IK collision body has no live counterpart: {name}")
        own = range(
            model.body_geomadr[body],
            model.body_geomadr[body] + model.body_geomnum[body],
        )
        original = range(
            live.body_geomadr[live_body],
            live.body_geomadr[live_body] + live.body_geomnum[live_body],
        )
        for geom, source in zip(own, original, strict=True):
            if model.geom_type[geom] != live.geom_type[source] or any(
                not np.allclose(
                    getattr(model, attr)[geom], getattr(live, attr)[source], atol=1e-8
                )
                for attr in ("geom_size", "geom_pos", "geom_quat")
            ):
                raise RuntimeError(f"IK collision geometry mapping differs for {name}")
            model.geom_contype[geom] = live.geom_contype[source]
            model.geom_conaffinity[geom] = live.geom_conaffinity[source]
            if not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
                continue
            short = name.split("/")[-1]
            if any(part in short for part in ("shoulder", "elbow", "wrist", "dex1")):
                side = next(
                    (s for s in ("left", "right") if short.startswith(s + "_")), None
                )
                if side is None:
                    raise RuntimeError(f"Unclassified arm collision body: {name}")
                groups[side].append(geom)
            elif any(
                part in short
                for part in ("torso", "waist", "pelvis", "hip", "knee", "ankle")
            ):
                groups["trunk"].append(geom)
    if not all(groups.values()):
        raise RuntimeError("IK collision model is missing an arm or the trunk")
    # Legs, waist and fingers are measured, not controlled by arm IK. Lock
    # them inside the QP so it cannot invent motion to manufacture clearance.
    ik._limits.append(
        mink.VelocityLimit(
            model, {env.robot.namespace + name: 1e-8 for name in ik.fixed_joint_names}
        )
    )
    ik._collision_limits = {
        active: mink.CollisionAvoidanceLimit(
            model,
            [(groups[side], groups["trunk"]) for side in active]
            + [(groups["left"], groups["right"])],
            # Mink bounds the approach rate; the IK QP integrates 50 ms
            # displacement steps. A small gain prevents stepping through the
            # nonlinear mesh boundary before the next distance evaluation.
            gain=0.04,
            minimum_distance_from_collisions=0.012,
            collision_detection_distance=0.05,
        )
        for active in (("left",), ("right",), ("left", "right"))
    }
    ik._limits.append(ik._collision_limits[("left", "right")])
    return ik


COMMAND_MARGIN = 0.03  # SmoothTargets keeps commands this far inside actuator bounds


def limit_to_commandable_range(ik, live):
    """Solve only for arm angles the position actuators can actually command.

    The G1 asset gives the right shoulder roll the left arm's actuator range
    (ctrlrange [-1.59, 2.25] against a joint range of [-2.25, 1.59]). An IK that
    plans past an actuator bound is clipped at the motor; it then believes the
    arm is somewhere it is not and steers into contorted poses.
    """
    import mink

    arm_ids = [ik.model.bind(joint).id for joint in ik.arm_joints]
    for joint in arm_ids:
        name = mujoco.mj_id2name(ik.model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        live_joint = mujoco.mj_name2id(live, mujoco.mjtObj.mjOBJ_JOINT, name)
        actuators = np.flatnonzero(
            (live.actuator_trnid[:, 0] == live_joint)
            & (live.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT)
        )
        low, high = live.jnt_range[live_joint]
        for actuator in actuators:
            if live.actuator_ctrllimited[actuator]:
                low = max(low, live.actuator_ctrlrange[actuator, 0])
                high = min(high, live.actuator_ctrlrange[actuator, 1])
        ik.model.jnt_range[joint] = (low + COMMAND_MARGIN, high - COMMAND_MARGIN)
    ik._limits[0] = mink.ConfigurationLimit(ik.model)
    ik._limits_low = ik.model.jnt_range[arm_ids, 0].copy()
    ik._limits_high = ik.model.jnt_range[arm_ids, 1].copy()


def checked_pelvis_pose(pose):
    pose = np.asarray(pose, dtype=float)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError("Planning pelvis pose requires seven finite coordinates")
    if abs(np.linalg.norm(pose[3:]) - 1) > 1e-4:
        raise ValueError("Planning pelvis quaternion must be normalized")
    return pose.copy()


def checked_balance_positions(robot, positions):
    """Planning may predict only the balance-owned joints, within their limits."""
    if not isinstance(positions, dict):
        raise TypeError("Planning balance positions must be a named mapping")
    result = {}
    for name, value in positions.items():
        joint = robot.joints.get(name)
        if joint is None or joint["owner"] != "balance":
            raise ValueError("Planning posture may override only balance joints")
        if not np.isfinite(value):
            raise ValueError("Planning balance positions must be finite")
        low, high = joint["limits"]
        result[name] = float(np.clip(value, low, high))
    return result


LIVE_LM_DAMPING = 1.5
LIVE_POSITION_SCALE = 3.0


class SmoothTargets:
    """Online, synchronized position targets with bounded derivatives."""

    def __init__(self, initial: np.ndarray, dt: float = 0.02, limits=None, bounds=None):
        self.dof = len(initial)
        self.otg = Ruckig(self.dof, dt)
        self.input = InputParameter(self.dof)
        self.output = OutputParameter(self.dof)
        self.input.current_position = list(initial)
        self.input.current_velocity = [0.0] * self.dof
        self.input.current_acceleration = [0.0] * self.dof
        self.input.target_velocity = [0.0] * self.dof
        self.input.target_acceleration = [0.0] * self.dof
        # A streaming filter re-targeted every tick: channels need not arrive
        # together, and time-synchronizing all 16 can fail at a mid-motion reversal.
        self.input.synchronization = Synchronization.No
        # Arms in rad/s; the final two channels are normalized gripper closure/s.
        if limits is None:
            limits = (
                [1.3] * (self.dof - 2) + [1.0, 1.0],
                [3.5] * self.dof,
                [18.0] * self.dof,
            )
        self.input.max_velocity, self.input.max_acceleration, self.input.max_jerk = (
            limits
        )
        self.velocity_limits = np.asarray(limits[0], dtype=float)
        self.bounds = (
            None
            if bounds is None
            else tuple(np.asarray(x, dtype=float) for x in bounds)
        )

    def step(self, target: np.ndarray) -> np.ndarray:
        target = np.asarray(target, dtype=float)
        if target.shape != (self.dof,) or not np.isfinite(target).all():
            raise ValueError(
                "Target must be a finite vector matching the actuator layout"
            )
        if self.bounds is not None:
            low, high = self.bounds
            margin = np.minimum(0.03, (high - low) * 0.04)
            target = np.clip(target, low + margin, high - margin)
            position = np.asarray(self.input.current_position)
            acceleration = np.asarray(self.input.max_acceleration)
            jerk = np.asarray(self.input.max_jerk)

            def stopping_speed(distance):
                distance = np.maximum(distance, 0)
                return np.maximum(
                    np.minimum(
                        self.velocity_limits,
                        np.minimum(
                            np.sqrt(acceleration * distance),
                            np.cbrt(jerk * distance**2),
                        ),
                    ),
                    0.001,
                )

            # Direction-specific braking envelope; retargeting near a hard stop
            # must account for residual velocity, not just clip the new goal.
            self.input.max_velocity = stopping_speed(high - margin - position).tolist()
            self.input.min_velocity = (
                -stopping_speed(position - low - margin)
            ).tolist()
        self.input.target_position = target.tolist()
        result = self.otg.update(self.input, self.output)
        if result not in (Result.Working, Result.Finished):
            raise RuntimeError(f"Trajectory generator failed: {result}")
        self.output.pass_to_input(self.input)
        return np.asarray(self.output.new_position)


class ManipulationRuntime:
    """World-coordinate hand targets, two grippers, and a live balance policy."""

    def __init__(
        self,
        seed: int = 7,
        scene: str = "reach",
        episode_length: int = 100000,
        joint_lab: bool = False,
    ):
        # No per-step camera rendering: control/physics work headlessly on macOS.
        # This is a research configuration, not a benchmark submission.
        env = make(
            {
                "pinch": "lab_scene:PINCH",
                "calibration": "lab_scene:PINCH",
                "sorting": "lab_scene:SORTING",
                "clutter": "lab_scene:CLUTTER",
                "arena": "lab_scene:ARENA",
                "reach": "reach_target_single",
            }[scene],
            camera_keys=(),
            episode_length=episode_length,
            # Keep serving hand commands after entering the benchmark's 5cm
            # target sphere. Our own measured-error checks evaluate precision.
            success_hold_seconds=1000,
        )
        self.tools = EnvTools(
            "reach_target_single",
            EnvToolsConfig(interface="tools", tier="privileged"),
            env,
        )
        self.tools.reset(seed)
        self.dt = self.tools.control_dt
        self.start = self.tools.n_base
        self.arm_slice = slice(self.start, self.start + 14)
        if len(self.tools.limb_names) != 14 or self.tools.action_dim != self.start + 16:
            self.close()
            raise RuntimeError(
                "Unexpected robot layout: expected two 7-DoF arms and two grippers"
            )
        self.raw = self.tools.hold_action().astype(float)
        self.raw[self.arm_slice] = np.concatenate(
            (self.tools.arm_qpos("left"), self.tools.arm_qpos("right"))
        )
        self.goal = self.raw[self.start :].copy()
        self.smoother = SmoothTargets(
            self.goal,
            self.dt,
            bounds=(
                self.tools.raw_low[self.start :],
                self.tools.raw_high[self.start :],
            ),
        )
        self.height_goal = self.raw[self.tools.base_slot["height"]]
        self.height_smoother = SmoothTargets(
            np.array([self.height_goal]), self.dt, ([0.12], [0.3], [1.5])
        )
        # GR00T-WBC owns the waist; its torso-pitch command is the only lean input.
        self.pitch_slot = self.tools.base_slot.get("pitch")
        self.pitch_goal = (
            float(self.raw[self.pitch_slot]) if self.pitch_slot is not None else 0.0
        )
        self.pitch_smoother = SmoothTargets(
            np.array([self.pitch_goal]), self.dt, ([0.5], [1.5], [6.0])
        )
        self.targets: dict[str, np.ndarray] = {}
        self.orientations: dict[str, np.ndarray] = {}
        self.feedback_offsets: dict[str, np.ndarray] = {}
        self.arm_feedback_period = 5
        # Outer-loop correction of steady IK/tracking bias (1/s); skills raise it
        # while hovering for precision, the default keeps free motion gentle.
        self.feedback_gain = 0.5
        self.ticks = 0
        self.trace: list[dict] = []
        self.contact_mode = False
        self.suppress_downward_bias = False
        self.joint_lab = joint_lab
        self.motion_goal = np.zeros(3)
        self.motion_smoother = SmoothTargets(
            np.zeros(3), self.dt, ([0.35, 0.35, 0.5], [0.7, 0.7, 1], [3, 3, 4])
        )
        self.joints = {}
        lower = self.tools.outer.lowerbody.controller
        lower_names = {
            n.split("/")[-1]: i for i, n in enumerate(lower.controlled_joints)
        }
        for j in range(self.tools.model.njnt):
            full_name = (
                mujoco.mj_id2name(self.tools.model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
            )
            name = full_name.split("/")[-1]
            if name not in self.tools.limb_names and name not in lower_names:
                continue
            self.joints[name] = {
                "id": j,
                "qpos": int(self.tools.model.jnt_qposadr[j]),
                "dof": int(self.tools.model.jnt_dofadr[j]),
                "limits": self.tools.model.jnt_range[j].tolist(),
                "owner": "arm" if name in self.tools.limb_names else "balance",
                "index": self.tools.limb_names.index(name)
                if name in self.tools.limb_names
                else lower_names[name],
            }
        if len(self.joints) != 29:
            raise RuntimeError(f"Expected 29 body joints, found {len(self.joints)}")
        from arm_gravity import ArmGravity

        self.arm_gravity = ArmGravity(self.tools)
        self.gravity_offsets = np.zeros(14)
        self.lower_proxy = JointOverrideController(lower, self.dt, supported=joint_lab)
        self.tools.outer.lowerbody.controller = self.lower_proxy
        if joint_lab:
            inner = self.tools.inner
            body = self.tools.data.bind(self.tools.robot.pelvis)
            support_id = self.tools.model.bind(inner.support_body).id
            mocap_id = self.tools.model.body_mocapid[support_id]
            self.tools.data.mocap_pos[mocap_id] = body.xpos.copy()
            self.tools.data.mocap_quat[mocap_id] = body.xquat.copy()
            self.tools.data.eq_active[self.tools.model.bind(inner.support_weld).id] = 1

    def joint_state(self):
        return {
            name: {
                "position_rad": float(self.tools.data.qpos[j["qpos"]]),
                "velocity_rad_s": float(self.tools.data.qvel[j["dof"]]),
                "limits_rad": j["limits"],
                "actuator_force_nm": float(self.tools.data.qfrc_actuator[j["dof"]]),
                "distance_to_limit_rad": float(
                    min(
                        self.tools.data.qpos[j["qpos"]] - j["limits"][0],
                        j["limits"][1] - self.tools.data.qpos[j["qpos"]],
                    )
                ),
                "owner": "direct_joint"
                if j["index"] in self.lower_proxy.overrides and j["owner"] == "balance"
                else j["owner"],
                "direct_command_available": j["owner"] == "arm" or self.joint_lab,
            }
            for name, j in self.joints.items()
        }

    def set_joint_targets(self, targets):
        """Atomic validation; arm IK and direct targets never own an arm together."""
        if not isinstance(targets, dict) or not targets:
            raise ValueError("Provide a nonempty mapping of joint names to radians")
        for name, value in targets.items():
            if name not in self.joints:
                raise ValueError(
                    f"Unknown body joint: {name}; read joint inventory first"
                )
            joint = self.joints[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
            ):
                raise ValueError("Joint targets must be finite radians")
            if not joint["limits"][0] <= value <= joint["limits"][1]:
                raise ValueError(f"{name} outside joint limits {joint['limits']}")
            if joint["owner"] == "balance" and not self.joint_lab:
                raise ValueError(
                    "Leg/waist direct targets require Joint experiment mode; normal mode uses balance/walking control"
                )
            if joint["owner"] == "balance" and np.linalg.norm(self.motion_goal) > 1e-6:
                raise ValueError("Stop walking before overriding balance joints")
        for name, value in targets.items():
            joint = self.joints[name]
            if joint["owner"] == "arm":
                side = "left" if name.startswith("left") else "right"
                self.targets.pop(side, None)
                self.orientations.pop(side, None)
                self.feedback_offsets.pop(side, None)
                self.goal[joint["index"]] = value
            else:
                self.lower_proxy.target(
                    joint["index"], value, float(self.tools.data.qpos[joint["qpos"]])
                )

    def velocity(self, vx, vy, wz):
        vector = np.array([vx, vy, wz], dtype=float)
        if not np.isfinite(vector).all() or np.any(np.abs(vector) > [0.35, 0.25, 0.6]):
            raise ValueError(
                "Walking limits: |vx| <= .35 m/s, |vy| <= .25 m/s, |yaw rate| <= .6 rad/s"
            )
        if self.lower_proxy.overrides and np.linalg.norm(vector) > 0:
            raise ValueError(
                "Reset to release experimental lower-body overrides before walking"
            )
        self.motion_goal = vector

    def hold(self):
        self.targets.clear()
        self.orientations.clear()
        self.feedback_offsets.clear()
        self.goal = self.raw[self.start :].copy()
        self.goal[:14] -= self.gravity_offsets
        self.height_goal = self.raw[self.tools.base_slot["height"]]
        if self.pitch_slot is not None:
            self.pitch_goal = float(self.raw[self.pitch_slot])
        self.motion_goal[:] = 0
        self.lower_proxy.hold()
        self.contact_mode = False
        self.suppress_downward_bias = False

    def state(self) -> dict:
        obs = self.tools.observation()
        fingers = {}
        for side in ("left", "right"):
            hand = self.tools.robot.grippers[self.tools._HandSide[side.upper()]]
            fingers[side] = [
                float(self.tools.data.bind(joint).qpos[0])
                for joint in hand.actuated_joints
            ]
        return {
            "time_s": self.ticks * self.dt,
            "fell": bool(obs["fell"]),
            "base_pos": obs["base_pos"].tolist(),
            "left_hand_pos": obs["left_hand_pos"].tolist(),
            "right_hand_pos": obs["right_hand_pos"].tolist(),
            "finger_positions_m": fingers,
            "gripper_closure": {
                side: float(obs[f"{side}_gripper"]) for side in ("left", "right")
            },
            "target_pos": obs["target_pos"].tolist(),
            "target_errors_m": {
                side: float(np.linalg.norm(obs[f"{side}_hand_pos"] - target))
                for side, target in self.targets.items()
            },
        }

    def reach(self, side: str, position, quaternion=None) -> None:
        if side not in ("left", "right"):
            raise ValueError("side must be left or right")
        position = np.asarray(position, dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("position must be three finite world-coordinate metres")
        # Reject obviously unreachable requests before disturbing the robot.
        pelvis = np.asarray(self.state()["base_pos"])
        if np.linalg.norm(position - pelvis) > 0.85:
            raise ValueError(
                "Target outside this stationary reach experiment's workspace"
            )
        if quaternion is not None:
            quaternion = np.asarray(quaternion, dtype=float)
            if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
                raise ValueError("quaternion must be finite w,x,y,z")
            norm = np.linalg.norm(quaternion)
            if norm < 1e-8:
                raise ValueError("quaternion must have nonzero norm")
            self.orientations[side] = quaternion / norm
        else:
            self.orientations.pop(side, None)
        self.targets[side] = position.copy()
        self.feedback_offsets.setdefault(side, np.zeros(3))

    def arm_in_contact(self, side, ignore_pad_contacts=False):
        """Local contact feedback for Cartesian bias correction, no object map."""
        for contact in self.tools.data.contact:
            for geom_id in (contact.geom1, contact.geom2):
                name = (
                    mujoco.mj_id2name(
                        self.tools.model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id)
                    )
                    or ""
                )
                if ignore_pad_contacts and f"{side}_dex1_pad_" in name:
                    continue
                body = (
                    mujoco.mj_id2name(
                        self.tools.model,
                        mujoco.mjtObj.mjOBJ_BODY,
                        int(self.tools.model.geom_bodyid[geom_id]),
                    )
                    or ""
                )
                short = body.split("/")[-1]
                if short.startswith(side + "_") and any(
                    part in short for part in ("dex1", "wrist", "elbow", "shoulder")
                ):
                    return True
        return False

    def grip(self, side: str, closure: float) -> None:
        if side not in ("left", "right"):
            raise ValueError("side must be left or right")
        if not np.isfinite(closure) or not 0 <= closure <= 1:
            raise ValueError("closure must be between 0 (open) and 1 (closed)")
        self.goal[14 + (side == "right")] = closure

    def contact_links(self, side):
        """Own contacting link names and loads; no contacted object identity."""
        result = []
        for i, contact in enumerate(self.tools.data.contact):
            for geom in (contact.geom1, contact.geom2):
                name = (
                    mujoco.mj_id2name(
                        self.tools.model, mujoco.mjtObj.mjOBJ_GEOM, int(geom)
                    )
                    or ""
                )
                body = (
                    mujoco.mj_id2name(
                        self.tools.model,
                        mujoco.mjtObj.mjOBJ_BODY,
                        int(self.tools.model.geom_bodyid[geom]),
                    )
                    or ""
                )
                short = body.split("/")[-1]
                if short.startswith(side + "_") and any(
                    part in short for part in ("dex1", "wrist", "elbow", "shoulder")
                ):
                    force = np.zeros(6)
                    mujoco.mj_contactForce(self.tools.model, self.tools.data, i, force)
                    result.append(
                        {
                            "robot_link": name or body,
                            "normal_force_n": float(abs(force[0])),
                        }
                    )
        return result

    def contact_in_payload(self, side, position, rotationally_uncertain=False):
        """Touch is inside the camera-tracked held volume, without object IDs."""
        payload = getattr(self, "contact_payloads", {}).get(side)
        if not payload:
            return False
        site = self.tools.data.bind(self.tools._hand(side).wrist_site)
        rotation = np.asarray(site.xmat).reshape(3, 3)
        center = np.asarray(site.xpos) + rotation @ np.asarray(
            payload["grasp_offset_local_m"]
        )
        relative = rotation @ np.asarray(payload["grasp_rotation_world"]).T
        local = relative.T @ (np.asarray(position) - center)
        if rotationally_uncertain:
            # A rigid prop can rotate inside the jaws. Its circumscribed sphere
            # covers that rotation; a fixed AABB cannot. Carry association only:
            # support/obstacle checks during placement keep the tighter bounds.
            return bool(
                np.linalg.norm(local) < np.linalg.norm(payload["half_size_m"]) + 0.008
            )
        return bool(np.all(np.abs(local) < np.asarray(payload["half_size_m"]) + 0.008))

    def hand_velocity(self, side):
        """World linear velocity of the robot's own tool site (proprioception)."""
        site = int(self.tools.model.bind(self.tools._hand(side).wrist_site).id)
        twist = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.tools.model, self.tools.data, mujoco.mjtObj.mjOBJ_SITE, site, twist, 0
        )
        return twist[3:]

    def contact_normals(self, side):
        """Touch normals for projecting out inward bias, including finger pads.

        A palm resting on a support must still be able to correct upward. The
        previous all-axis freeze trapped the arm below a collision-free target.
        Normals are local touch sensing, without contacted-object identity.
        """
        normals = []
        model = self.tools.model
        prefix = self.tools.robot.namespace + side + "_"
        for contact in self.tools.data.contact:
            bodies = [
                mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])
                )
                or ""
                for g in (contact.geom1, contact.geom2)
            ]
            if all(n.startswith(self.tools.robot.namespace) for n in bodies):
                continue
            for index, geom in enumerate((contact.geom1, contact.geom2)):
                body = bodies[index]
                if body.startswith(prefix) and any(
                    p in body.split("/")[-1]
                    for p in ("dex1", "wrist", "elbow", "shoulder")
                ):
                    if "dex1" in body and self.contact_in_payload(side, contact.pos):
                        continue
                    normals.append(
                        np.asarray(contact.frame).reshape(3, 3)[0]
                        * (1 if index == 1 else -1)
                    )
        return np.asarray(normals).reshape(-1, 3)

    def separating_contact_bias(self, side, correction):
        normals = self.contact_normals(side)
        projected = np.array(correction, dtype=float)
        for _ in range(4):
            for normal in normals:
                projected -= min(0, float(normal @ projected)) * normal
        return (
            projected
            if not len(normals) or np.all(normals @ projected >= -1e-9)
            else np.zeros(3)
        )

    def posture(self, height: float, pitch: float | None = None) -> None:
        """Command a smooth crouch and, when available, a torso lean via the WBC."""
        if self.joint_lab:
            raise ValueError(
                "Supported joint rig has a fixed pelvis; use direct joint targets"
            )
        if not np.isfinite(height) or not 0.45 <= height <= 0.80:
            raise ValueError("Pelvis height must be in [0.45, 0.80] m")
        if pitch is not None:
            if self.pitch_slot is None:
                raise ValueError("This scene has no torso-pitch command")
            if not np.isfinite(pitch) or not -0.2 <= pitch <= 0.8:
                raise ValueError("Torso pitch must be in [-0.2, 0.8] rad")
            self.pitch_goal = float(pitch)
        self.height_goal = float(height)

    def tick(self) -> dict:
        if self.tools.outer.episode_fell():
            raise RuntimeError("Robot fell; reset required")
        if self.tools.last_timestep.last():
            raise RuntimeError("Episode ended; reset required")
        if self.targets and self.ticks % self.arm_feedback_period == 0:
            obs = self.tools.observation()
            kwargs = {}
            for side, pos in self.targets.items():
                error = pos - obs[f"{side}_hand_pos"]
                # Small bounded integral correction for loaded joint-tracking bias.
                # Integrate only near the goal; this is a free-space servo, not
                # contact/force control and must not be used to push into objects.
                if np.linalg.norm(error) < 0.10:
                    correction = self.feedback_gain * (self.arm_feedback_period * self.dt) * error
                    if self.contact_mode:
                        normals = self.contact_normals(side)
                        # Remove already accumulated inward bias as well as
                        # blocking new inward increments. Keep tangential bias.
                        for _ in range(2):
                            for normal in normals:
                                self.feedback_offsets[side] -= (
                                    min(0, float(normal @ self.feedback_offsets[side]))
                                    * normal
                                )
                        if len(normals) and np.any(
                            normals @ self.feedback_offsets[side] < -1e-6
                        ):
                            self.feedback_offsets[side][:] = 0
                        correction = self.separating_contact_bias(side, correction)
                    if self.suppress_downward_bias:
                        correction[2] = max(0, correction[2])
                    self.feedback_offsets[side] += correction
                    self.feedback_offsets[side] = np.clip(
                        self.feedback_offsets[side], -0.06, 0.06
                    )
                kwargs[f"{side}_pos"] = pos + self.feedback_offsets[side]
            kwargs.update(
                {f"{side}_quat": quat for side, quat in self.orientations.items()}
            )
            for attempt in range(4):
                try:
                    solution = self.solve_ik(**kwargs, iters=2)
                    break
                except ValueError as exc:
                    if "self-collision" not in str(exc) or attempt == 3:
                        raise
                    # Integral correction cannot demand a pose beyond the
                    # collision boundary. Unwind it before changing the task.
                    for side, pos in self.targets.items():
                        self.feedback_offsets[side] *= 0.5 if attempt < 2 else 0
                        kwargs[f"{side}_pos"] = pos + self.feedback_offsets[side]
            for side in self.targets:
                arm = slice(0, 7) if side == "left" else slice(7, 14)
                low = self.tools.raw_low[self.arm_slice][arm]
                high = self.tools.raw_high[self.arm_slice][arm]
                # Leave a small command margin at hard stops when online IK
                # retargets a trajectory that still has residual velocity.
                self.goal[arm] = np.clip(solution[arm], low + 0.02, high - 0.02)
                # The next solve must start from what was commanded, never from
                # an answer the motors could not take.
                self.sync_arm_seed(side)
        self.gravity_offsets = self.arm_gravity.offsets(self.tools)
        motor_goal = self.goal.copy()
        motor_goal[:14] += self.gravity_offsets
        motor_goal = np.clip(
            motor_goal,
            self.tools.raw_low[self.start :],
            self.tools.raw_high[self.start :],
        )
        self.raw[self.start :] = self.smoother.step(motor_goal)
        self.raw[self.tools.base_slot["height"]] = self.height_smoother.step(
            np.array([self.height_goal])
        )[0]
        if self.pitch_slot is not None:
            self.raw[self.pitch_slot] = self.pitch_smoother.step(
                np.array([self.pitch_goal])
            )[0]
        velocity = self.motion_smoother.step(self.motion_goal)
        for i, key in enumerate(("vx", "vy", "wz")):
            self.raw[self.tools.base_slot[key]] = velocity[i]
        lo, hi = self.tools.raw_low, self.tools.raw_high
        if np.any(self.raw < lo - 1e-5) or np.any(self.raw > hi + 1e-5):
            indices = np.flatnonzero((self.raw < lo - 1e-5) | (self.raw > hi + 1e-5))
            raise RuntimeError(
                "Generated trajectory exceeded actuator bounds: "
                + str(
                    [
                        (int(i), float(self.raw[i]), float(lo[i]), float(hi[i]))
                        for i in indices
                    ]
                )
            )
        self.tools.step(self.raw)
        self.ticks += 1
        state = self.state()
        if not np.isfinite(self.tools.data.qpos).all() or state["fell"]:
            raise RuntimeError("Unstable physics or fall; reset required")
        self.trace.append(
            {
                **state,
                "command": self.raw.tolist(),
                "command_velocity": list(self.smoother.output.new_velocity),
                "command_acceleration": list(self.smoother.output.new_acceleration),
            }
        )
        return state

    def solve_ik(
        self,
        left_pos=None,
        left_quat=None,
        right_pos=None,
        right_quat=None,
        iters=10,
        planning=False,
        seed_q=None,
        pelvis_pose=None,
        fixed_joint_positions=None,
    ):
        """Seed waist, leg and finger joints from measured kinematics.

        EnvTools' teleoperation helper zeroes waist yaw/roll. Retaining measured
        angles keeps planning, servo feedback and live collision geometry aligned.
        The extracted IK model contains robot kinematics only.
        """
        from bigym.vr.ik.mink_upper_body_ik import Pose
        from pyquaternion import Quaternion

        if fixed_joint_positions is not None:
            if not planning:
                raise ValueError(
                    "A posture override is only permitted for private planning"
                )
            fixed_joint_positions = checked_balance_positions(
                self, fixed_joint_positions
            )
        if pelvis_pose is not None:
            if not planning:
                raise ValueError(
                    "A pelvis override is only permitted for private planning"
                )
            pelvis_pose = checked_pelvis_pose(pelvis_pose)
        if seed_q is not None:
            if not planning:
                raise ValueError(
                    "An explicit IK seed is only permitted for private planning"
                )
            seed_q = np.asarray(seed_q, dtype=float)
            if seed_q.shape != (14,) or not np.isfinite(seed_q).all():
                raise ValueError(
                    "Planning seed must contain 14 finite arm joint angles"
                )
        if planning:
            ik = self.planning_solver()
            ik.reset_seed()
            if seed_q is not None:
                ik._last_solution = np.clip(seed_q, ik._limits_low, ik._limits_high)
        else:
            if self.tools._ik is None:
                self.tools._ik = collision_aware_ik(self.tools.inner)
                # Targets here are smooth planned paths, not noisy VR hands, so
                # the live servo can follow the planner's converged solutions
                # with lighter damping and a weaker pull toward the rest pose.
                for task in (self.tools._ik._left_task, self.tools._ik._right_task):
                    task.lm_damping = LIVE_LM_DAMPING
                    task.cost[:3] *= LIVE_POSITION_SCALE
            ik = self.tools._ik
        position, quaternion = self.tools.pelvis_pose()
        if pelvis_pose is not None:
            position, quaternion = pelvis_pose[:3], pelvis_pose[3:]
        pelvis = Pose(position, Quaternion(quaternion))
        fixed = self.tools.data.qpos[ik.live_fixed_qpos_adr].copy()
        for name, value in (fixed_joint_positions or {}).items():
            fixed[ik.fixed_joint_names.index(name)] = value
        q_left, q_right = self.tools.arm_qpos("left"), self.tools.arm_qpos("right")
        import mink

        velocities = {
            self.tools.robot.namespace + n: 1e-8 for n in ik.fixed_joint_names
        }
        # Live servo steps stay gentle; private planning may take larger steps so
        # a far reconfiguration converges in tens of iterations, not hundreds.
        arm_speed = 8.0 if planning else 2.0
        for side, desired, measured, arm in (
            ("left", left_pos, q_left, slice(0, 7)),
            ("right", right_pos, q_right, slice(7, 14)),
        ):
            velocities.update(
                {
                    self.tools.robot.namespace + n: 1e-8 if desired is None else arm_speed
                    for n in self.tools.limb_names
                    if n.startswith(side)
                }
            )
            if desired is None and ik._last_solution is not None:
                ik._last_solution[arm] = measured
        # Uncommanded arms must remain measured obstacles, not hypothetical
        # moving arms that the optimizer can relocate to manufacture clearance.
        ik._limits[-2] = mink.VelocityLimit(ik.model, velocities)
        active = tuple(
            s for s, p in (("left", left_pos), ("right", right_pos)) if p is not None
        )
        ik._limits[-1] = ik._collision_limits[active or ("left", "right")]
        ik.seed(pelvis, np.concatenate((q_left, q_right)).astype(np.float64), fixed)
        wanted = {"left": (left_pos, left_quat), "right": (right_pos, right_quat)}
        for side in wanted:
            ik.set_orientation_cost(side, 0 if wanted[side][1] is None else 2)
        for _ in range(max(1, int(iters))):
            targets = {}
            for side, (pos, quat) in wanted.items():
                targets[side] = Pose(
                    self.tools._hand(side).wrist_position
                    if pos is None
                    else np.asarray(pos),
                    ik.wrist_pose(side).orientation
                    if quat is None
                    else Quaternion(quat),
                )
            try:
                solution = ik.solve(
                    pelvis_pose=pelvis,
                    qpos_arm_left=q_left,
                    qpos_arm_right=q_right,
                    target_pose_left=targets["left"],
                    target_pose_right=targets["right"],
                    fixed_qpos=fixed,
                )
            except mink.NoSolutionFound as exc:
                ik.reset_seed()
                raise ValueError(
                    "IK constraints have no feasible solution; choose another approach or base stance"
                ) from exc
        # QP distance constraints are locally linearized. Check the resulting
        # nonlinear geometry too: an unreachable torso target must never become
        # an actuator command merely because the optimizer returned a vector.
        clearance, geom_a, geom_b = min(
            (
                mujoco.mj_geomDistance(
                    ik.model, ik._configuration.data, a, b, 0.02, None
                ),
                a,
                b,
            )
            for a, b in ik._limits[-1].geom_id_pairs
        )
        if clearance < 0.004:
            ik.reset_seed()
            links = [
                mujoco.mj_id2name(
                    ik.model, mujoco.mjtObj.mjOBJ_BODY, int(ik.model.geom_bodyid[g])
                ).split("/")[-1]
                for g in (geom_a, geom_b)
            ]
            raise ValueError(
                f"IK self-collision clearance is insufficient ({clearance:.4f} m, {links}); choose another approach or base stance"
            )
        return np.asarray(solution, dtype=np.float32)

    def planning_solver(self):
        """The private, converged IK used to plan and to choose arm configurations."""
        if getattr(self, "planning_ik", None) is None:
            self.planning_ik = collision_aware_ik(self.tools.inner)
            # BiGym's teleop tuning (heavy LM damping) keeps VR following smooth
            # but leaves centimetre residuals far from the rest pose. Planning
            # needs converged answers, so this private solver damps less and
            # iterates more. The rest-pose pull stays: weakening it leaves arm
            # joints nearly unconstrained and the QP solver fails at random.
            for task in (self.planning_ik._left_task, self.planning_ik._right_task):
                task.lm_damping = 0.2
            self.planning_ik._solve_iters = 8
        return self.planning_ik

    def sync_arm_seed(self, side=None):
        """Continue live IK from the commanded arm, e.g. after a joint-space move."""
        ik = self.tools._ik
        if ik is None or ik._last_solution is None:
            return
        for name in (side,) if side else ("left", "right"):
            arm = slice(0, 7) if name == "left" else slice(7, 14)
            ik._last_solution[arm] = np.clip(
                self.goal[arm], ik._limits_low[arm], ik._limits_high[arm]
            )

    def close(self) -> None:
        self.tools.close()


class JointOverrideController:
    """Policy passthrough OR supported joint rig, selected only during reset.

    In rig mode the pelvis is explicitly supported and the balance policy is
    not stepped. Every lower-body target is owned here; no fighting controllers.
    Rig motions do not demonstrate free-standing balance, running or jumping.
    """

    def __init__(self, wrapped, dt, supported=False):
        self.wrapped = wrapped
        self.dt = dt
        self.overrides = {}
        self.supported = supported
        if supported:
            for i, value in enumerate(wrapped.last_targets):
                self.target(i, float(value), float(value))

    def __getattr__(self, name):
        return getattr(self.wrapped, name)

    def target(self, index, value, measured):
        if index not in self.overrides:
            anchor = (
                float(self.wrapped.last_targets[index])
                if hasattr(self.wrapped, "last_targets")
                else measured
            )
            self.overrides[index] = [
                SmoothTargets(np.array([anchor]), self.dt, ([0.3], [0.6], [3.0])),
                value,
            ]
        self.overrides[index][1] = value

    def hold(self):
        for override in self.overrides.values():
            override[1] = override[0].input.current_position[0]

    def step(self):
        output = (
            np.zeros(len(self.wrapped.controlled_joints))
            if self.supported
            else self.wrapped.step().copy()
        )
        for index, (smoother, target) in self.overrides.items():
            output[index] = smoother.step(np.array([target]))[0]
        return output
