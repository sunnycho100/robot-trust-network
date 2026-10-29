"""Robot-only gravity feedforward through existing bounded position actuators.

No scene bodies, object poses, contacts or external force injection are inputs.
The position-servo equilibrium is q_command = q_desired + gravity_torque / kp.
The caller smooths the combined command through its normal actuator trajectory.
"""

import mujoco
import numpy as np

from robot_kinematics import RobotArmIK


class ArmGravity:
    def __init__(self, tools):
        self.robot_model = RobotArmIK(tools.inner)
        self.model = self.robot_model.model
        self.data = mujoco.MjData(self.model)
        self.pelvis_id = self.model.body(tools.robot.pelvis.name).id
        self.live_qpos = []
        for i in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            self.live_qpos.append(int(tools.model.joint(name).qposadr[0]))
        self.arm_dofs, self.gains = [], []
        for name in tools.limb_names:
            full = tools.robot.namespace + name
            joint = tools.model.joint(full)
            actuator = np.flatnonzero(tools.model.actuator_trnid[:, 0] == joint.id)
            if len(actuator) != 1:
                raise ValueError(
                    "Gravity feedforward requires one position actuator per arm joint"
                )
            a = int(actuator[0])
            kp = float(tools.model.actuator_gainprm[a, 0])
            if (
                kp <= 0
                or not np.isclose(tools.model.actuator_biasprm[a, 1], -kp)
                or not np.allclose(tools.model.actuator_gear[a], [1, 0, 0, 0, 0, 0])
            ):
                raise ValueError(
                    "Unsupported arm actuator transmission for gravity feedforward"
                )
            self.gains.append(kp)
            self.arm_dofs.append(int(self.model.joint(full).dofadr[0]))
        self.gains = np.asarray(self.gains)

    def offsets(self, tools):
        position, quaternion = tools.pelvis_pose()
        self.model.body_pos[self.pelvis_id] = position
        self.model.body_quat[self.pelvis_id] = quaternion
        self.data.qpos[:] = tools.data.qpos[self.live_qpos]
        self.data.qvel[:] = 0  # gravity only, not unmodelled contact/load compensation
        mujoco.mj_forward(self.model, self.data)
        return np.clip(self.data.qfrc_bias[self.arm_dofs] / self.gains, -0.1, 0.1)
