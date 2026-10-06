"""Unitree G1 walk using the published ``unitree_rl_gym`` motion.pt policy.

The observation layout, PD loop, gains, action scale, robot XML, and TorchScript
weights come from unitreerobotics/unitree_rl_gym. ``scripts/fetch_assets``
downloads them; nothing is trained here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np


UPSTREAM_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "unitree_rl_gym"
LEG_JOINT_NAMES = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
)


def gravity_orientation(quaternion_wxyz: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = quaternion_wxyz
    return np.array(
        [2 * (-qz * qx + qw * qy), -2 * (qz * qy + qw * qx), 1 - 2 * (qw * qw + qz * qz)],
        dtype=np.float32,
    )


def yaw_from_quaternion(quat_wxyz: np.ndarray) -> float:
    w, x, y, z = quat_wxyz
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


@dataclass(frozen=True)
class G1Config:
    policy_path: Path
    xml_path: Path
    simulation_dt: float
    control_decimation: int
    kps: np.ndarray
    kds: np.ndarray
    default_angles: np.ndarray
    ang_vel_scale: float
    dof_pos_scale: float
    dof_vel_scale: float
    action_scale: float
    cmd_scale: np.ndarray
    num_actions: int
    num_obs: int

    @property
    def robot_xml(self) -> Path:
        """The 12-DoF G1 alone, for scenes that place it themselves."""
        return self.xml_path.parent / "g1_12dof.xml"


def load_g1_config() -> G1Config:
    import yaml

    path = UPSTREAM_ROOT / "deploy" / "deploy_mujoco" / "configs" / "g1.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Unitree G1 config not found: {path}. Run ./scripts/fetch_assets")
    raw = yaml.safe_load(path.read_text())

    def resolve(value: str) -> Path:
        return Path(value.replace("{LEGGED_GYM_ROOT_DIR}", str(UPSTREAM_ROOT)))

    return G1Config(
        policy_path=resolve(raw["policy_path"]),
        xml_path=resolve(raw["xml_path"]),
        simulation_dt=float(raw["simulation_dt"]),
        control_decimation=int(raw["control_decimation"]),
        kps=np.asarray(raw["kps"], dtype=np.float32),
        kds=np.asarray(raw["kds"], dtype=np.float32),
        default_angles=np.asarray(raw["default_angles"], dtype=np.float32),
        ang_vel_scale=float(raw["ang_vel_scale"]),
        dof_pos_scale=float(raw["dof_pos_scale"]),
        dof_vel_scale=float(raw["dof_vel_scale"]),
        action_scale=float(raw["action_scale"]),
        cmd_scale=np.asarray(raw["cmd_scale"], dtype=np.float32),
        num_actions=int(raw["num_actions"]),
        num_obs=int(raw["num_obs"]),
    )


class G1Controller:
    """One copy of motion.pt driving the G1 whose names start with ``prefix``.

    motion.pt keeps recurrent state for a batch of one, so every robot loads its own copy.
    """

    def __init__(self, model, cfg: G1Config, prefix: str = "") -> None:
        import torch

        self.torch = torch
        self.cfg = cfg
        self.policy = torch.jit.load(str(cfg.policy_path), map_location="cpu").eval()
        self.action = np.zeros(cfg.num_actions, dtype=np.float32)
        self.target_q = cfg.default_angles.copy()
        self.observation = np.zeros(cfg.num_obs, dtype=np.float32)
        base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}floating_base_joint")
        self.body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}pelvis")
        self.base_qpos = int(model.jnt_qposadr[base_id])
        self.base_dof = int(model.jnt_dofadr[base_id])
        joint_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + n) for n in LEG_JOINT_NAMES])
        self.joint_qpos = model.jnt_qposadr[joint_ids].astype(int)
        self.joint_dof = model.jnt_dofadr[joint_ids].astype(int)
        self.actuators = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, prefix + n) for n in LEG_JOINT_NAMES])

    def reset(self, data) -> None:
        data.qpos[self.joint_qpos] = self.cfg.default_angles

    def position_xy(self, data) -> np.ndarray:
        return data.qpos[self.base_qpos:self.base_qpos + 2].copy()

    def quaternion(self, data) -> np.ndarray:
        return data.qpos[self.base_qpos + 3:self.base_qpos + 7]

    def yaw(self, data) -> float:
        return yaw_from_quaternion(self.quaternion(data))

    def apply(self, data) -> None:
        q, dq = data.qpos[self.joint_qpos], data.qvel[self.joint_dof]
        data.ctrl[self.actuators] = (self.target_q - q) * self.cfg.kps - dq * self.cfg.kds

    def update_policy(self, data, command, phase: float) -> None:
        cfg, obs = self.cfg, self.observation
        n = cfg.num_actions
        obs[:3] = data.qvel[self.base_dof + 3:self.base_dof + 6] * cfg.ang_vel_scale
        obs[3:6] = gravity_orientation(self.quaternion(data))
        obs[6:9] = command * cfg.cmd_scale
        obs[9:9 + n] = (data.qpos[self.joint_qpos] - cfg.default_angles) * cfg.dof_pos_scale
        obs[9 + n:9 + 2 * n] = data.qvel[self.joint_dof] * cfg.dof_vel_scale
        obs[9 + 2 * n:9 + 3 * n] = self.action
        obs[-2:] = math.sin(2 * math.pi * phase), math.cos(2 * math.pi * phase)
        with self.torch.inference_mode():
            self.action = self.policy(self.torch.from_numpy(obs).unsqueeze(0)).numpy().squeeze()
        self.target_q = self.action * cfg.action_scale + cfg.default_angles


def gait_phase(data) -> float:
    return (data.time % 0.8) / 0.8
