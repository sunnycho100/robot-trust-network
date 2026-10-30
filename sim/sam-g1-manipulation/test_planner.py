"""Stance, route and motion-order planning on simple scenes."""

import numpy as np
import pytest

from body import minimum_jerk, motion_schedule, posture_order, standing, wrap
from planner import (
    BODY_MARGIN_M,
    Navigator,
    PlanningError,
    Scene,
    body_clearance,
    hand_yaw_options,
    natural_hand_yaw,
    turn_clear,
)


def wall(x, y_from, y_to, z_from=0.05, z_to=1.0, step=0.02):
    ys, zs = np.meshgrid(np.arange(y_from, y_to, step), np.arange(z_from, z_to, step))
    return np.column_stack((np.full(ys.size, x), ys.ravel(), zs.ravel()))


def test_body_clearance_sees_a_wall_in_front_only_when_close():
    scene = Scene(wall(0.6, -1.0, 1.0), focus=np.zeros(2))
    far = body_clearance(scene, standing(), np.array([0.0, 0.0, 0.0]))
    near = body_clearance(scene, standing(), np.array([0.45, 0.0, 0.0]))
    assert far > 0.4
    assert near < 0.1
    assert body_clearance(Scene(), standing(), np.zeros(3)) == np.inf


def test_turning_beside_a_wall_needs_room():
    scene = Scene(wall(0.30, -1.0, 1.0), focus=np.zeros(2))
    assert not turn_clear(scene, np.array([0.12, 0.0]), 0.0, np.pi)
    assert turn_clear(scene, np.array([-0.5, 0.0]), 0.0, np.pi)


def test_route_goes_around_a_wall_and_prefers_open_entries():
    scene = Scene(np.vstack((wall(1.0, -0.6, 0.6),)), focus=np.zeros(2))
    navigator = Navigator(scene, np.zeros(3))
    goal = np.array([1.8, 0.0, np.pi])  # behind the wall, facing back toward it
    route = navigator.route(goal)
    waypoints = np.asarray(route["waypoints"])
    # Every step stays out of the wall's band.
    assert all(abs(p[1]) > 0.6 for p in waypoints[:-1] if 0.7 < p[0] < 1.3)
    np.testing.assert_allclose(waypoints[-1], goal[:2])


def test_unreachable_goal_is_reported():
    box = np.vstack(
        [wall(x, -0.5, 0.5) for x in (0.9, 1.9)]
        + [np.column_stack((np.arange(0.9, 1.9, 0.02), np.full(50, y), np.full(50, 0.5))) for y in (-0.5, 0.5)]
    )
    navigator = Navigator(Scene(box, focus=np.zeros(2)), np.zeros(3))
    with pytest.raises(PlanningError):
        navigator.route(np.array([1.4, 0.0, 0.0]))


def test_gripper_fits_only_narrow_sides():
    cube = {"half_size_m": [0.025, 0.025, 0.025], "yaw_rad": 0.3}
    assert len(hand_yaw_options(cube)) == 4
    long_box = {"half_size_m": [0.08, 0.03, 0.05], "yaw_rad": 0.0}
    options = hand_yaw_options(long_box)
    assert all(abs(np.sin(o)) < 1e-9 for o in options)  # fingers along the long side
    assert hand_yaw_options({"half_size_m": [0.03, 0.03, 0.05], "shape": "round"}) is None
    with pytest.raises(PlanningError):
        hand_yaw_options({"half_size_m": [0.08, 0.06, 0.05], "yaw_rad": 0.0})


def test_natural_hand_yaw_points_from_the_shoulder():
    yaw = natural_hand_yaw(np.array([0.0, 0.0, 0.0]), "right", np.array([0.4, -0.15]))
    assert abs(wrap(yaw)) < 0.05


def test_reaching_down_moves_the_arm_first_and_rising_moves_the_body_first():
    assert posture_order(0.74, 0.0, 0.62, 0.4) == "arm_first"
    assert posture_order(0.62, 0.4, 0.74, 0.0) == "body_first"
    for order in ("arm_first", "body_first"):
        assert motion_schedule(order, 0.0) == (0.0, 0.0)
        assert motion_schedule(order, 1.0) == (1.0, 1.0)
    arm, body = motion_schedule("arm_first", 0.4)
    assert arm > 0.6 and body < 0.2
    arm, body = motion_schedule("body_first", 0.4)
    assert body > 0.6 and arm < 0.2
    assert minimum_jerk(0.5) == pytest.approx(0.5)


def test_standing_stance_margin_is_physical():
    assert 0.03 <= BODY_MARGIN_M <= 0.08
