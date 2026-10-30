"""Arm IK with the rest of the robot retained at measured joint positions.

The private model contains no scene objects. Unlike an arm-only extraction,
keeping leg/finger joints lets collision checking use the actual crouch and jaw
opening. Those joints are measured constraints, never arm-IK control outputs.

Model extraction follows BiGym2's G1UpperBodyIK (Apache-2.0), commit
d71501359b215b9d78e27dcb70b220412d474dfa. See third_party/BiGym2-LICENSE.txt.
"""

import mujoco
import numpy as np
from bigym import scene
from bigym.const import HandSide
from bigym.robots.configs.g1 import (
    G1_DEX1_LEFT_HAND_JOINT_NAMES,
    G1_DEX1_RIGHT_HAND_JOINT_NAMES,
    G1_LEFT_ARM_JOINT_NAMES,
    G1_LEG_JOINT_NAMES,
    G1_RIGHT_ARM_JOINT_NAMES,
    G1_WAIST_JOINT_NAMES,
)
from bigym.vr.ik.g1_upper_body_ik import (
    POSTURE_LM_DAMPING,
    POSTURE_WEIGHT,
    REST_POSE_LEFT,
    REST_POSE_RIGHT,
    SOLVE_ITERS,
)
from bigym.vr.ik.mink_upper_body_ik import MinkUpperBodyIK


class RobotArmIK(MinkUpperBodyIK):
    def __init__(self, env):
        robot = env.robot
        spec = env.spec.copy()
        frame = spec.body(robot.body.name)
        for element in (
            *spec.keys,
            *spec.actuators,
            *spec.sensors,
            *spec.tendons,
            *spec.equalities,
            *spec.excludes,
            *spec.pairs,
        ):
            spec.delete(element)
        world = spec.worldbody
        for element in (*world.geoms, *world.sites, *world.cameras, *world.lights):
            spec.delete(element)
        for body in list(world.bodies):
            if body.name != frame.name:
                spec.delete(body)

        arms = (*G1_LEFT_ARM_JOINT_NAMES, *G1_RIGHT_ARM_JOINT_NAMES)
        self.fixed_joint_names = (
            *G1_WAIST_JOINT_NAMES,
            *G1_LEG_JOINT_NAMES,
            *G1_DEX1_LEFT_HAND_JOINT_NAMES,
            *G1_DEX1_RIGHT_HAND_JOINT_NAMES,
        )
        kept = {robot.namespace + n for n in (*arms, *self.fixed_joint_names)}
        for joint in scene.descendants(frame, "joints"):
            if joint.name not in kept:
                spec.delete(joint)
        for geom in spec.geoms:
            geom.contype = 0
            geom.conaffinity = 0

        self.pelvis = spec.body(robot.pelvis.name)
        self.arm_joints = [spec.joint(robot.namespace + n) for n in arms]
        self.left_arm_site = spec.site(
            robot.namespace + robot.config.arms[HandSide.LEFT].site
        )
        self.right_arm_site = spec.site(
            robot.namespace + robot.config.arms[HandSide.RIGHT].site
        )
        fixed_joints = [spec.joint(robot.namespace + n) for n in self.fixed_joint_names]
        # These are measured obstacles, not optimization variables. MuJoCo's
        # soft physical stops can leave an encoder slightly beyond its nominal
        # limit. Arm IK must preserve that measured pose, not demand that an
        # immovable finger/leg correct its limit error. Live limits are intact.
        for joint in fixed_joints:
            joint.limited = False
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        # EnvTools' limb accessor excludes the coupled finger joints. Read
        # their encoders through explicitly named robot joints, not limb slots.
        self.live_fixed_qpos_adr = np.array(
            [
                env.model.joint(robot.namespace + name).qposadr[0]
                for name in self.fixed_joint_names
            ],
            dtype=int,
        )
        self._init_solver(
            rest_pose=np.concatenate((REST_POSE_LEFT, REST_POSE_RIGHT)),
            posture_weight=np.tile(POSTURE_WEIGHT, 2),
            posture_lm_damping=POSTURE_LM_DAMPING,
            solve_iters=SOLVE_ITERS,
            measured_resync_threshold=None,
            fixed_joints=fixed_joints,
        )
