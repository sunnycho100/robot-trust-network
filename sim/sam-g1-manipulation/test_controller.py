"""Trajectory tests, including changing a goal during motion."""

import numpy as np
import pytest

from controller import SmoothTargets


def test_contact_projection_preserves_free_and_tangential_motion():
    from controller import ManipulationRuntime

    robot = ManipulationRuntime.__new__(ManipulationRuntime)
    robot.contact_normals = lambda side: np.empty((0, 3))
    np.testing.assert_allclose(
        robot.separating_contact_bias("left", [1, 2, 3]), [1, 2, 3]
    )
    robot.contact_normals = lambda side: np.array([[0, 0, 1], [0, 0, -1]])
    np.testing.assert_allclose(
        robot.separating_contact_bias("left", [1, 2, -3]), [1, 2, 0]
    )


def test_robot_only_gravity_compensation_removes_static_arm_droop_and_hold_does_not_double_it():
    from controller import ManipulationRuntime

    errors = []
    for enabled in (False, True):
        robot = ManipulationRuntime(scene="sorting", seed=23)
        try:
            assert robot.arm_gravity.model.njnt == robot.arm_gravity.model.nq
            assert robot.arm_gravity.model.nu == 0
            if not enabled:
                robot.arm_gravity.offsets = lambda tools: np.zeros(14)
            for _ in range(60):
                robot.tick()
            q = robot.solve_ik(
                right_pos=[0.4, -0.23, 0.92],
                right_quat=[1, 0, 0, 0],
                iters=40,
                planning=True,
            )
            robot.goal[7:14] = q[7:14]
            for _ in range(350):
                robot.tick()
            measured = robot.tools.arm_qpos("right")
            errors.append(float(np.max(np.abs(measured - q[7:14]))))
            if enabled:
                # Repeated holds must not accumulate gravity offsets.
                before = measured.copy()
                for _ in range(4):
                    robot.hold()
                    for _ in range(40):
                        robot.tick()
                np.testing.assert_allclose(
                    robot.tools.arm_qpos("right"), before, atol=0.001
                )
            assert not robot.state()["fell"]
        finally:
            robot.close()
    assert errors[0] > 0.02
    assert errors[1] < 0.001


@pytest.mark.parametrize("robot_first", (False, True))
def test_contact_bias_can_escape_but_never_push_into_support(monkeypatch, robot_first):
    from types import SimpleNamespace

    import mujoco

    from controller import ManipulationRuntime

    def own_names(model, kind, index):
        if kind == mujoco.mjtObj.mjOBJ_BODY:
            return "robot/left_dex1_base_link" if index == 1 else "world"
        return None

    monkeypatch.setattr(mujoco, "mj_id2name", own_names)
    robot = ManipulationRuntime.__new__(ManipulationRuntime)
    frame = np.array([[0, 0, -1 if robot_first else 1], [1, 0, 0], [0, 1, 0]])
    contact = SimpleNamespace(
        geom1=1 if robot_first else 0,
        geom2=0 if robot_first else 1,
        frame=frame,
        pos=np.zeros(3),
    )
    robot.tools = SimpleNamespace(
        model=SimpleNamespace(geom_bodyid=np.array([0, 1])),
        data=SimpleNamespace(contact=[contact]),
        robot=SimpleNamespace(namespace="robot/"),
    )
    np.testing.assert_allclose(
        robot.separating_contact_bias("left", [1, 2, -3]), [1, 2, 0]
    )
    np.testing.assert_allclose(
        robot.separating_contact_bias("left", [1, 2, 3]), [1, 2, 3]
    )


def test_planning_does_not_mutate_servo_seed_or_live_physics():
    from controller import ManipulationRuntime

    robot = ManipulationRuntime(scene="sorting")
    try:
        target = np.asarray(robot.state()["right_hand_pos"])
        robot.solve_ik(right_pos=target)
        before = robot.tools.data.qpos.copy()
        seed = robot.tools._ik._last_solution.copy()
        planned = robot.solve_ik(right_pos=target + [0.04, 0, 0.03], planning=True)
        assert robot.planning_ik is not robot.tools._ik
        wanted = target + [0.05, 0, 0.03]
        first = robot.solve_ik(right_pos=wanted, planning=True, seed_q=planned, iters=1)
        robot.solve_ik(right_pos=target + [-0.03, 0, 0.04], planning=True)
        second = robot.solve_ik(right_pos=wanted, planning=True, seed_q=planned, iters=1)
        np.testing.assert_allclose(first, second, atol=1e-7)
        # The uncommanded arm stays where it is measured.
        np.testing.assert_allclose(second[:7], robot.tools.arm_qpos("left"), atol=2e-6)
        np.testing.assert_array_equal(robot.tools.data.qpos, before)
        np.testing.assert_array_equal(robot.tools._ik._last_solution, seed)
        for bad in (np.zeros(7), np.full(14, np.nan)):
            with pytest.raises(ValueError, match="Planning seed"):
                robot.solve_ik(right_pos=target, planning=True, seed_q=bad)
        for override in ({"seed_q": planned}, {"pelvis_pose": np.r_[target, 1, 0, 0, 0]}):
            with pytest.raises(ValueError, match="only permitted for private planning"):
                robot.solve_ik(right_pos=target, **override)
        for bad in ({"right_shoulder_pitch_joint": 0.0}, {"waist_pitch_joint": np.nan}):
            with pytest.raises(ValueError, match="Planning"):
                robot.solve_ik(right_pos=target, planning=True, fixed_joint_positions=bad)
    finally:
        robot.close()


def test_infeasible_ik_is_recoverable_and_does_not_command_joints(monkeypatch):
    import mink

    from controller import ManipulationRuntime

    robot = ManipulationRuntime(scene="sorting")
    try:
        before = robot.tools.data.qpos.copy()
        command = robot.goal.copy()

        def infeasible(*args, **kwargs):
            raise mink.NoSolutionFound("daqp")

        monkeypatch.setattr(mink, "solve_ik", infeasible)
        with pytest.raises(ValueError, match="no feasible solution"):
            robot.solve_ik(right_pos=robot.state()["right_hand_pos"], planning=True)
        assert robot.planning_ik._last_solution is None
        np.testing.assert_array_equal(robot.tools.data.qpos, before)
        np.testing.assert_array_equal(robot.goal, command)
    finally:
        robot.close()


def test_crouched_leg_and_finger_geometry_matches_measured_robot():
    import mujoco

    from controller import ManipulationRuntime

    robot = ManipulationRuntime(scene="sorting")
    try:
        model, data = robot.tools.model, robot.tools.data
        namespace = robot.tools.robot.namespace
        for side in ("left", "right"):
            for suffix, value in (
                ("hip_pitch_joint", -0.55),
                ("knee_joint", 1.0),
                ("ankle_pitch_joint", -0.45),
            ):
                joint = model.joint(namespace + side + "_" + suffix)
                data.qpos[joint.qposadr[0]] = value
        # A coupled finger at a soft physical stop must remain measured, not
        # make arm IK demand an impossible correction of this fixed coordinate.
        finger = model.joint(namespace + "left_dex1_finger_joint_1")
        data.qpos[finger.qposadr[0]] = finger.range[1] + 0.0002
        mujoco.mj_forward(model, data)
        before = data.qpos.copy()
        robot.solve_ik(right_pos=robot.state()["right_hand_pos"], planning=True)
        ik = robot.planning_ik
        private = ik._configuration.data
        for name in ik.fixed_joint_names:
            live_joint = model.joint(namespace + name)
            private_joint = ik.model.joint(namespace + name)
            np.testing.assert_allclose(
                private.qpos[private_joint.qposadr],
                data.qpos[live_joint.qposadr],
                atol=1e-8,
            )
        # Leg collision shapes must follow the bent knee, including their
        # orientations. This fails for the previous zeroed-leg extraction.
        for body_name in ("left_knee_link", "right_ankle_roll_link"):
            live_body = model.body(namespace + body_name).id
            private_body = ik.model.body(namespace + body_name).id
            np.testing.assert_allclose(
                private.xpos[private_body], data.xpos[live_body], atol=1e-7
            )
            np.testing.assert_allclose(
                private.xmat[private_body], data.xmat[live_body], atol=1e-7
            )
        pairs = ik._limits[-1].geom_id_pairs
        assert any(
            "knee" in ik.model.body(int(ik.model.geom_bodyid[g])).name
            for pair in pairs
            for g in pair
        )
        np.testing.assert_array_equal(data.qpos, before)
    finally:
        robot.close()


def test_unnamed_arm_torso_collision_is_sensed_and_constrained():
    import mujoco

    from controller import ManipulationRuntime

    robot = ManipulationRuntime(scene="sorting")
    try:
        for name, value in (
            ("right_shoulder_pitch_joint", -1.23),
            ("right_shoulder_roll_joint", 0.52),
        ):
            robot.tools.data.qpos[robot.joints[name]["qpos"]] = value
        mujoco.mj_forward(robot.tools.model, robot.tools.data)
        assert robot.arm_in_contact("right", ignore_pad_contacts=True)
        assert any(
            "right_shoulder" in row["robot_link"]
            for row in robot.contact_links("right")
        )
        try:
            robot.solve_ik(right_pos=robot.state()["right_hand_pos"], planning=True)
        except ValueError as exc:
            assert "self-collision" in str(exc)
        ik = robot.planning_ik
        pairs = ik._limits[-1].geom_id_pairs
        body_pairs = [
            [
                mujoco.mj_id2name(
                    ik.model, mujoco.mjtObj.mjOBJ_BODY, int(ik.model.geom_bodyid[g])
                )
                for g in pair
            ]
            for pair in pairs
        ]
        assert any(
            any("right_shoulder_roll" in n for n in names)
            and any("torso" in n for n in names)
            for names in body_pairs
        )
    finally:
        robot.close()


def test_inside_torso_goal_cannot_produce_colliding_actuator_target():
    import mujoco

    from controller import ManipulationRuntime

    robot = ManipulationRuntime(scene="sorting")
    try:
        before = robot.tools.data.qpos.copy()
        try:
            robot.solve_ik(
                right_pos=np.array(robot.state()["base_pos"]) + [0.03, -0.03, 0.25],
                iters=30,
                planning=True,
            )
        except ValueError as exc:
            assert "self-collision" in str(exc)
        else:
            ik = robot.planning_ik
            assert (
                min(
                    mujoco.mj_geomDistance(
                        ik.model, ik._configuration.data, a, b, 0.02, None
                    )
                    for a, b in ik._limits[-1].geom_id_pairs
                )
                >= 0.004
            )
        np.testing.assert_array_equal(robot.tools.data.qpos, before)
    finally:
        robot.close()


def test_retarget_preserves_derivative_limits_and_converges():
    smoother = SmoothTargets(np.zeros(16))
    target = np.full(16, 0.4)
    acceleration = []
    velocity = []
    for tick in range(400):
        if tick == 35:
            target[:14] = -0.2
        position = smoother.step(target)
        acceleration.append(smoother.output.new_acceleration)
        velocity.append(smoother.output.new_velocity)
    np.testing.assert_allclose(position, target, atol=1e-7)
    assert np.max(np.abs(np.asarray(velocity)[:, :14])) <= 1.300001
    assert np.max(np.abs(acceleration)) <= 3.500001
    assert np.max(np.abs(np.diff(acceleration, axis=0) / 0.02)) <= 18.000001


@pytest.mark.parametrize(
    "bad", [np.zeros(15), np.full(16, np.nan), np.full(16, np.inf)]
)
def test_bad_targets_are_rejected_without_poisoning_trajectory(bad):
    smoother = SmoothTargets(np.zeros(16))
    with pytest.raises(ValueError):
        smoother.step(bad)
    assert np.isfinite(smoother.step(np.ones(16) * 0.1)).all()


def test_joint_limit_braking_survives_mid_motion_reversals():
    low, high = np.full(16, -0.5), np.full(16, 0.5)
    smooth = SmoothTargets(np.zeros(16), bounds=(low, high))
    velocities, accelerations = [], []
    for tick in range(1200):
        target = np.full(16, 0.5 if (tick // 43) % 2 == 0 else -0.5)
        actual = smooth.step(target)
        assert np.all(actual >= low) and np.all(actual <= high)
        velocities.append(smooth.output.new_velocity)
        accelerations.append(smooth.output.new_acceleration)
    assert np.max(np.abs(velocities)) <= 1.30001
    assert np.max(np.abs(accelerations)) <= 3.50001
    assert np.max(np.abs(np.diff(accelerations, axis=0) / 0.02)) <= 18.0001
