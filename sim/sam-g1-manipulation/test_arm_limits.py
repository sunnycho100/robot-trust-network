"""The arm IK plans only angles the motors can command, and the servo stays in sync."""

import numpy as np

from controller import COMMAND_MARGIN, ManipulationRuntime


def test_ik_ranges_are_the_commandable_ranges():
    robot = ManipulationRuntime(scene="sorting", seed=7)
    try:
        low = robot.tools.raw_low[robot.arm_slice]
        high = robot.tools.raw_high[robot.arm_slice]
        index = robot.tools.limb_names.index("right_shoulder_roll_joint")
        # The asset's actuator range for this joint is the left arm's, mirrored wrongly.
        assert robot.joints["right_shoulder_roll_joint"]["limits"][0] < low[index] - 0.5
        ik = robot.planning_solver()
        model_low = np.array([robot.joints[name]["limits"][0] for name in robot.tools.limb_names])
        model_high = np.array([robot.joints[name]["limits"][1] for name in robot.tools.limb_names])
        np.testing.assert_allclose(ik._limits_low, np.maximum(low, model_low) + COMMAND_MARGIN, atol=1e-6)
        np.testing.assert_allclose(ik._limits_high, np.minimum(high, model_high) - COMMAND_MARGIN, atol=1e-6)
    finally:
        robot.close()


def test_live_ik_continues_from_what_was_commanded():
    robot = ManipulationRuntime(scene="sorting", seed=7)
    try:
        for _ in range(20):
            robot.tick()
        hand = np.asarray(robot.state()["right_hand_pos"])
        robot.reach("right", hand + [0.05, 0.0, 0.05])
        for _ in range(30):
            robot.tick()
        np.testing.assert_allclose(robot.tools._ik._last_solution[7:14], robot.goal[7:14], atol=1e-6)
        # A joint-space move elsewhere, then a sync: the next solve starts from it.
        robot.goal[7:14] += 0.1
        robot.sync_arm_seed("right")
        np.testing.assert_allclose(robot.tools._ik._last_solution[7:14], robot.goal[7:14], atol=1e-6)
    finally:
        robot.close()
