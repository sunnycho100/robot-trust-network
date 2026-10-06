"""Unitree Go1 walk using the published MuJoCo Playground ONNX policy.

The observation layout and position targets follow
google-deepmind/mujoco_playground ``play_go1_joystick.py``. This demo does not
train the policy. ``scripts/fetch_assets`` downloads the weights and the model.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from .g1_policy import yaw_from_quaternion


THIRD_PARTY = Path(__file__).resolve().parents[2] / "third_party"
POLICY_PATH = THIRD_PARTY / "go1_policy.onnx"
GO1_XML = THIRD_PARTY / "mujoco_menagerie" / "unitree_go1" / "go1.xml"
# The published policy was trained with position kp of 35.
POLICY_KP = 35.0
ACTION_SCALE = np.float32(0.5)
# Playground "home" keyframe. The policy's residual actions are added to this pose.
DEFAULT_ANGLES = np.array(
    [0.1, 0.9, -1.8, -0.1, 0.9, -1.8, 0.1, 0.9, -1.8, -0.1, 0.9, -1.8],
    dtype=np.float32,
)
JOINT_NAMES = (
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
)
ACTUATOR_NAMES = (
    "FR_hip", "FR_thigh", "FR_calf",
    "FL_hip", "FL_thigh", "FL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
)


def go1_observation(linvel, gyro, imu_xmat, joint_position, joint_velocity, last_action, command) -> np.ndarray:
    """48-D observation expected by go1_policy.onnx."""
    rotation = np.asarray(imu_xmat, dtype=np.float64).reshape(3, 3)
    gravity = rotation.T @ np.array([0.0, 0.0, -1.0])
    observation = np.concatenate(
        (
            np.asarray(linvel, dtype=np.float32).reshape(3),
            np.asarray(gyro, dtype=np.float32).reshape(3),
            gravity.astype(np.float32),
            np.asarray(joint_position, dtype=np.float32).reshape(12) - DEFAULT_ANGLES,
            np.asarray(joint_velocity, dtype=np.float32).reshape(12),
            np.asarray(last_action, dtype=np.float32).reshape(12),
            np.asarray(command, dtype=np.float32).reshape(3),
        )
    )
    if observation.shape != (48,):
        raise ValueError(f"Go1 observation must have 48 values, got {observation.shape}")
    return observation


def load_go1_spec() -> mujoco.MjSpec:
    """Menagerie Go1 plus the gyro and velocimeter the policy observes."""
    if not GO1_XML.is_file():
        raise RuntimeError(f"Go1 model is missing: {GO1_XML}. Run ./scripts/fetch_assets")
    spec = mujoco.MjSpec.from_file(str(GO1_XML))
    for name, kind in (("gyro", mujoco.mjtSensor.mjSENS_GYRO), ("local_linvel", mujoco.mjtSensor.mjSENS_VELOCIMETER)):
        sensor = spec.add_sensor()
        sensor.name = name
        sensor.type = kind
        sensor.objtype = mujoco.mjtObj.mjOBJ_SITE
        sensor.objname = "imu"
    return spec


def set_policy_gains(model, prefix: str) -> None:
    for actuator_id in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id) or ""
        if name.startswith(prefix):
            model.actuator_gainprm[actuator_id, 0] = POLICY_KP
            model.actuator_biasprm[actuator_id, 1] = -POLICY_KP


class Go1Controller:
    def __init__(self, model, prefix: str) -> None:
        self.prefix = prefix
        self.action = np.zeros(12, dtype=np.float32)
        self.target_q = DEFAULT_ANGLES.copy()
        imu_name = f"{prefix}imu"
        self.imu_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, imu_name)
        self.gyro_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, f"{prefix}gyro")
        self.linvel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, f"{prefix}local_linvel")
        missing = [
            name for name, index in (
                (imu_name, self.imu_site),
                (f"{prefix}gyro", self.gyro_id),
                (f"{prefix}local_linvel", self.linvel_id),
            ) if index < 0
        ]
        if missing:
            raise RuntimeError(f"Go1 model is missing {', '.join(missing)}")
        self.gyro_adr = int(model.sensor_adr[self.gyro_id])
        self.linvel_adr = int(model.sensor_adr[self.linvel_id])
        joint_ids = np.array([
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + name) for name in JOINT_NAMES
        ])
        actuator_ids = np.array([
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, prefix + name) for name in ACTUATOR_NAMES
        ])
        if np.any(joint_ids < 0) or np.any(actuator_ids < 0):
            raise RuntimeError("Go1 joint or actuator names do not match the Playground policy")
        self.joint_qpos = model.jnt_qposadr[joint_ids].astype(int)
        self.joint_dof = model.jnt_dofadr[joint_ids].astype(int)
        self.actuators = actuator_ids
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}trunk")
        self.body_id = body_id
        self.base_qpos = None
        self.base_dof = None
        for joint_id in range(model.njnt):
            if int(model.jnt_bodyid[joint_id]) == body_id and model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
                self.base_qpos = int(model.jnt_qposadr[joint_id])
                self.base_dof = int(model.jnt_dofadr[joint_id])
                break
        if self.base_qpos is None:
            raise RuntimeError("Go1 trunk has no free joint")
        self._spawn = model.qpos0[self.base_qpos:self.base_qpos + 7].copy()
        self._session = None

    def load_policy(self):
        if self._session is not None:
            return self._session
        if not POLICY_PATH.is_file():
            raise RuntimeError(f"Missing {POLICY_PATH}. Run ./scripts/fetch_assets")
        try:
            import onnxruntime as rt
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is missing. Run: python -m pip install onnxruntime"
            ) from exc
        self._session = rt.InferenceSession(str(POLICY_PATH), providers=["CPUExecutionProvider"])
        return self._session

    def reset_pose(self, data) -> None:
        # qpos0 already holds the spawn frame's world position and heading.
        data.qpos[self.base_qpos:self.base_qpos + 7] = self._spawn
        data.qpos[self.base_qpos + 2] = 0.30
        data.qpos[self.joint_qpos] = DEFAULT_ANGLES
        data.qvel[self.base_dof:self.base_dof + 6] = 0.0
        data.qvel[self.joint_dof] = 0.0
        self.action[:] = 0.0
        self.target_q = DEFAULT_ANGLES.copy()
        data.ctrl[self.actuators] = self.target_q

    def position_xy(self, data) -> np.ndarray:
        return data.qpos[self.base_qpos:self.base_qpos + 2].copy()

    def yaw(self, data) -> float:
        return yaw_from_quaternion(data.qpos[self.base_qpos + 3:self.base_qpos + 7])

    def apply(self, data) -> None:
        data.ctrl[self.actuators] = self.target_q

    def update_policy(self, data, command, _phase=None) -> None:
        session = self.load_policy()
        linvel = data.sensordata[self.linvel_adr:self.linvel_adr + 3]
        gyro = data.sensordata[self.gyro_adr:self.gyro_adr + 3]
        observation = go1_observation(
            linvel, gyro, data.site_xmat[self.imu_site],
            data.qpos[self.joint_qpos], data.qvel[self.joint_dof],
            self.action, command,
        )
        self.action = session.run(["continuous_actions"], {"obs": observation.reshape(1, -1)})[0][0].astype(np.float32)
        self.target_q = self.action * ACTION_SCALE + DEFAULT_ANGLES
