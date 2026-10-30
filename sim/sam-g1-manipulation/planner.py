"""Where to stand, how to hold the body, and how to move a hand there.

One body model is used everywhere: the measured leg/torso surface for each
GR00T-WBC posture (posture_table.json) checked against camera points. Stances
face the target, postures may crouch and lean, routes avoid what was seen, and
hands rise before reaching over a surface. Feasibility uses physical tolerances
(about a centimetre), not sub-millimetre gates the arm cannot execute anyway.
"""

from __future__ import annotations

import heapq
import time

import mujoco
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from body import (
    gait,
    motion_schedule,
    postures,
    rotation_z,
    standing,
    wrap,
    yaw_quaternion,
)

IK_TOLERANCE_M = 0.006
REACH_MARGIN_M = 0.03  # spare reach beyond the deepest hand target
ARM_POSE_MARGIN_M = 0.05  # arm at its approach pose vs. seen points (it tracks within a few cm)
HELD_POSE_MARGIN_M = 0.02  # a held object at its hover pose vs. seen points
HELD_ALLOWANCE_M = 0.015  # while swinging, a held object may pass this much closer than the arm
IK_TOLERANCE_RAD = 0.2
BODY_MARGIN_M = 0.05  # standing at a stance (static body); a stepping leg reaches a few cm further
POSTURE_MARGIN_M = 0.03  # crouching/leaning in place there, which is slow and precise
ROOMY_CLEARANCE_M = 0.12  # stances with less clearance than this cost extra
TRANSIT_RADIUS_M = 0.34
GRIPPER_OPENING_M = 0.083  # inner gap between open Dex1 pads, less a small margin


class PlanningError(RuntimeError):
    def __init__(self, stage, reason, **details):
        super().__init__(reason)
        self.stage = stage
        self.details = details


class Scene:
    """Camera points of everything above the floor, gathered during one skill."""

    def __init__(self, points=None, focus=None):
        self.points = np.zeros((0, 3)) if points is None else np.asarray(points)
        self.focus = focus
        self._tree = None

    def add(self, points, voxel=0.015):
        merged = np.vstack((self.points, points))
        keys = np.floor(merged / voxel).astype(np.int64)
        _, index = np.unique(keys, axis=0, return_index=True)
        self.points = merged[np.sort(index)]
        self._tree = None

    def without_box(self, center, half_size, yaw, margin=0.02):
        c, s = np.cos(yaw), np.sin(yaw)
        local = self.points - np.asarray(center)
        a = local[:, 0] * c + local[:, 1] * s
        b = -local[:, 0] * s + local[:, 1] * c
        inside = (
            (np.abs(a) <= half_size[0] + margin)
            & (np.abs(b) <= half_size[1] + margin)
            & (np.abs(local[:, 2]) <= half_size[2] + margin)
        )
        return Scene(self.points[~inside], self.focus)

    def without_sphere(self, center, radius):
        keep = np.linalg.norm(self.points - np.asarray(center), axis=1) > radius
        return Scene(self.points[keep], self.focus)

    @property
    def tree(self):
        if self._tree is None:
            self._tree = cKDTree(self.points) if len(self.points) else None
        return self._tree

    def distance_field(self, center, half_extent=3.2, resolution=0.025, z_max=1.6):
        """3D distance (m) from each 2 cm cell to the nearest seen point."""
        key = (tuple(np.round(center[:2], 1)), half_extent, resolution)
        if getattr(self, "_field_key", None) == key:
            return self._field
        lo = np.r_[np.asarray(center[:2]) - half_extent, 0.0]
        shape = np.r_[np.full(2, int(2 * half_extent / resolution) + 1), int(z_max / resolution) + 1]
        occupied = np.zeros(shape, dtype=bool)
        if len(self.points):
            cells = np.floor((self.points - lo) / resolution).astype(int)
            keep = np.all((cells >= 0) & (cells < shape), axis=1)
            occupied[tuple(cells[keep].T)] = True
        field = ndimage.distance_transform_edt(~occupied) * resolution
        self._field_key, self._field = key, (lo, resolution, field)
        return self._field

    def clearance(self, query):
        query = np.atleast_2d(query)
        if self.tree is None:
            return np.full(len(query), np.inf)
        return self.tree.query(query)[0]


def body_clearance(scene, posture, stance):
    """Smallest distance between the robot's leg/torso surface and seen points.

    A coarse 3D distance field answers instantly far from obstacles; body
    points within 8 cm of something get an exact point-to-point distance.
    """
    if not len(scene.points):
        return np.inf
    focus = scene.focus if scene.focus is not None else stance[:2]
    lo, resolution, field = scene.distance_field(focus)
    surface = posture.surface_world(stance)
    cells = np.floor((surface - lo) / resolution).astype(int)
    inside = np.all((cells >= 0) & (cells < field.shape), axis=1)
    coarse = np.full(len(surface), np.inf)
    coarse[inside] = field[tuple(cells[inside].T)]
    near = coarse < 0.08 + resolution * 1.8
    if not near.any():
        return float(coarse.min())
    exact = scene.tree.query(surface[near], distance_upper_bound=0.12)[0]
    return float(min(exact.min(), coarse[~near].min() if (~near).any() else np.inf))


def _solve_chain(robot, side, pelvis, balance, targets, seed):
    """Solve hand poses in order, each starting from the previous solution.

    Returns (worst position error, worst angle error, one configuration per target).
    """
    worst_position, worst_angle, solution, solutions = 0.0, 0.0, seed, []
    for position, quaternion in targets:
        result = None
        wanted = {f"{side}_pos": np.asarray(position)}
        if quaternion is not None:  # None leaves the hand's orientation free
            wanted[f"{side}_quat"] = np.asarray(quaternion)
        for iterations in (20, 32):
            try:
                result = robot.solve_ik(
                    **wanted,
                    iters=iterations,
                    planning=True,
                    pelvis_pose=pelvis,
                    fixed_joint_positions=balance,
                    seed_q=solution,
                )
                break
            except ValueError:
                continue  # ended too close to the torso; a different stop point may not
        if result is None:
            return np.inf, np.inf, None
        solution = np.asarray(result, dtype=float)
        solutions.append(solution)
        achieved = robot.planning_ik.wrist_pose(side)
        error = float(np.linalg.norm(np.asarray(achieved.position) - position))
        worst_position = max(worst_position, error)
        if quaternion is None:
            continue
        goal = np.empty(9)
        mujoco.mju_quat2Mat(goal, np.asarray(quaternion, dtype=float))
        actual = np.empty(9)
        mujoco.mju_quat2Mat(actual, np.asarray(achieved.orientation.elements))
        cosine = (np.trace(actual.reshape(3, 3).T @ goal.reshape(3, 3)) - 1) / 2
        worst_angle = max(worst_angle, float(np.arccos(np.clip(cosine, -1, 1))))
    return worst_position, worst_angle, solutions


def _seeds(robot, side, extra=()):
    """The commanded arm, a neutral forearm-forward start, then any given seeds.

    A folded arm (e.g. carrying close to the chest) can trap the solver in a
    local minimum, so more than one start is tried.
    """
    current = np.asarray(robot.goal[:14], dtype=float).copy()
    neutral = current.copy()
    neutral[slice(0, 7) if side == "left" else slice(7, 14)] = 0.0
    return [np.asarray(s, dtype=float) for s in extra if s is not None] + [current, neutral]


def ik_errors(robot, side, posture, stance, targets, seeds=()):
    """Plan arm IK at a predicted posture and stance.

    Returns (worst position error, worst angle error, configurations per target).
    """
    pelvis = posture.pelvis_pose(stance)
    best = (np.inf, np.inf, None)
    for seed in _seeds(robot, side, seeds):
        result = _solve_chain(robot, side, pelvis, posture.balance_joints, targets, seed)
        if (result[0], result[1]) < (best[0], best[1]):
            best = result
        if best[0] <= IK_TOLERANCE_M and best[1] <= IK_TOLERANCE_RAD:
            break
    return best


def check_reach(robot, scene, side, posture, stance, targets, held=()):
    """Judge reaching `targets` from a stance and posture, the way it will be done.

    Returns (verdict, position error, angle error, configurations). The verdict
    is "ok" when every pose is reachable with REACH_MARGIN_M to spare and the
    arm (and anything it holds) keeps ARM_POSE_MARGIN_M from seen points at the
    first, approach pose; "edge" when reachable only at the limit of the arm;
    "blocked" when the arm would lie in or over something; "fail" otherwise.
    """
    error, angle, solutions = ik_errors(robot, side, posture, stance, targets)
    if not (error <= IK_TOLERANCE_M and angle <= IK_TOLERANCE_RAD):
        return "fail", error, angle, solutions
    body = (posture.pelvis_pose(stance), posture.balance_joints)
    if len(scene.points):
        points, pinch = arm_points(robot, side, body, solutions[0])
        clearance = scene.clearance(points)
        gripper = gripper_mask(robot, side)
        if clearance[~gripper].min() < ARM_POSE_MARGIN_M or clearance[gripper].min() < HELD_POSE_MARGIN_M:
            return "blocked", error, angle, solutions
        # What the hand holds is lowered straight down from here, precisely: it
        # may come closer to a neighbour than the arm does.
        for offset, radius in held:
            if float(scene.clearance(pinch + np.asarray(offset))[0]) - radius < HELD_POSE_MARGIN_M:
                return "blocked", error, angle, solutions
    # The real body settles a few centimetres from the measured posture (the
    # table has no arm load): keep some reach in reserve beyond the deepest pose.
    deepest, quaternion = targets[-1]
    away = np.asarray(deepest[:2]) - np.asarray(stance[:2])
    stretched = np.r_[deepest[:2] + REACH_MARGIN_M * away / max(np.linalg.norm(away), 1e-6), deepest[2] - 0.01]
    spare, spare_angle, _ = _solve_chain(robot, side, body[0], body[1], [(stretched, quaternion)], solutions[-1])
    if spare <= IK_TOLERANCE_M and spare_angle <= IK_TOLERANCE_RAD:
        return "ok", error, angle, solutions
    return "edge", error, angle, solutions


def measured_body(robot):
    """The pelvis pose (xyz + wxyz) and balance joint angles right now."""
    position, quaternion = robot.tools.pelvis_pose()
    balance = {
        name: float(robot.tools.data.qpos[joint["qpos"]])
        for name, joint in robot.joints.items()
        if joint["owner"] == "balance"
    }
    return np.r_[position, quaternion], balance


def solve_arm(robot, side, body, target, seeds=()):
    """Arm configuration (14 angles) putting the hand at `target` for a body pose.

    `body` is (pelvis pose, balance joints), predicted or measured. Returns
    (configuration, position error m, angle error rad).
    """
    pelvis, balance = body
    best = (None, np.inf, np.inf)
    for seed in _seeds(robot, side, seeds):
        error, angle, solutions = _solve_chain(robot, side, pelvis, balance, [target], seed)
        if (error, angle) < (best[1], best[2]):
            best = (solutions[-1], error, angle)
        if best[1] <= 0.004 and best[2] <= 0.05:
            break
    return best


# Links whose collision surface is checked against the scene when the arm swings.
ARM_LINKS = (
    "shoulder_yaw_link",
    "elbow_link",
    "wrist_roll_link",
    "wrist_pitch_link",
    "wrist_yaw_link",
    "dex1_base_link",
    "dex1_finger_link_1",
    "dex1_finger_link_2",
)


def _arm_surface(robot, side):
    """Points on one arm's collision geometry, each in its link's frame (cached)."""
    cache = robot.__dict__.setdefault("_arm_surface", {})
    if side not in cache:
        model = robot.planning_solver().model
        namespace = robot.tools.robot.namespace
        grid = np.stack(np.meshgrid(*[np.linspace(-1, 1, 4)] * 3), -1).reshape(-1, 3)
        bodies, surfaces = [], []
        for link in ARM_LINKS:
            body = model.body(namespace + f"{side}_{link}").id
            local = []
            for geom in range(model.body_geomadr[body], model.body_geomadr[body] + model.body_geomnum[body]):
                if not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
                    continue
                if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
                    mesh = model.geom_dataid[geom]
                    start = model.mesh_vertadr[mesh]
                    points = model.mesh_vert[start : start + model.mesh_vertnum[mesh]]
                else:
                    size = model.geom_size[geom]
                    half = {
                        mujoco.mjtGeom.mjGEOM_BOX: size,
                        mujoco.mjtGeom.mjGEOM_SPHERE: [size[0]] * 3,
                        mujoco.mjtGeom.mjGEOM_CYLINDER: [size[0], size[0], size[1]],
                        mujoco.mjtGeom.mjGEOM_CAPSULE: [size[0], size[0], size[1] + size[0]],
                    }[model.geom_type[geom]]
                    points = grid * np.asarray(half)
                rotation = np.empty(9)
                mujoco.mju_quat2Mat(rotation, model.geom_quat[geom])
                local.append(model.geom_pos[geom] + points @ rotation.reshape(3, 3).T)
            local = np.vstack(local)
            keys = np.floor(local / 0.012).astype(np.int64)
            _, keep = np.unique(keys, axis=0, return_index=True)
            bodies.append(body)
            surfaces.append(local[np.sort(keep)])
        cache[side] = (bodies, surfaces)
    return cache[side]


GRIPPER_LINKS = ("dex1_base_link", "dex1_finger_link_1", "dex1_finger_link_2")


def gripper_mask(robot, side):
    """Which of arm_points' points belong to the gripper (cached)."""
    cache = robot.__dict__.setdefault("_gripper_mask", {})
    if side not in cache:
        _, surfaces = _arm_surface(robot, side)
        cache[side] = np.concatenate(
            [np.full(len(local), link in GRIPPER_LINKS) for link, local in zip(ARM_LINKS, surfaces)]
        )
    return cache[side]


def arm_points(robot, side, body, configuration):
    """World points on one arm's collision surface, and its pinch point.

    `body` is (pelvis pose, balance joints), predicted or measured; finger
    joints are the measured ones.
    """
    from bigym.vr.ik.mink_upper_body_ik import Pose
    from pyquaternion import Quaternion

    ik = robot.planning_solver()
    if getattr(robot, "_fk_data", None) is None:
        robot._fk_data = mujoco.MjData(ik.model)
        robot._fk_sites = {
            name: ik.model.site(task.frame_name).id
            for name, task in (("left", ik._left_task), ("right", ik._right_task))
        }
    pelvis, balance = body
    data = robot._fk_data
    ik.set_pelvis_pose(Pose(np.asarray(pelvis[:3]), Quaternion(np.asarray(pelvis[3:]))))
    fixed = robot.tools.data.qpos[ik.live_fixed_qpos_adr].copy()
    for name, value in balance.items():
        fixed[ik.fixed_joint_names.index(name)] = value
    data.qpos[:] = 0.0
    data.qpos[ik._qpos_adr] = configuration
    data.qpos[ik._fixed_qpos_adr] = fixed
    mujoco.mj_kinematics(ik.model, data)
    bodies, surfaces = _arm_surface(robot, side)
    points = np.vstack(
        [data.xpos[b] + local @ data.xmat[b].reshape(3, 3).T for b, local in zip(bodies, surfaces)]
    )
    return points, data.site_xpos[robot._fk_sites[side]].copy()


def _blend_body(start, end, fraction):
    pelvis = (1 - fraction) * np.asarray(start[0]) + fraction * np.asarray(end[0])
    pelvis[3:] /= np.linalg.norm(pelvis[3:])
    balance = {
        name: (1 - fraction) * start[1].get(name, value) + fraction * value
        for name, value in end[1].items()
    }
    return pelvis, balance


def swept_clearance(robot, scene, side, start, end, held=(), extra=(), samples=12, order=None):
    """Smallest distance from the moving arm (and what it holds) to seen points.

    The gripper and a held object may come HELD_ALLOWANCE_M closer than the arm.

    `start`/`end` are (body, configuration) pairs; body pose and arm angles are
    blended in the order Body.move_arm uses (`order`: arm_first, body_first or
    None for together). `held` lists
    (offset from the pinch point, radius) spheres covering an object in the
    hand, and `extra` lists (centre, radius) spheres of things the cameras have
    not seen yet, such as an object just put down.
    """
    (body_a, q_a), (body_b, q_b) = start, end
    worst = np.inf
    for fraction in np.linspace(0.0, 1.0, samples):
        arm, posture = motion_schedule(order, fraction)
        body = _blend_body(body_a, body_b, posture)
        q = (1 - arm) * np.asarray(q_a) + arm * np.asarray(q_b)
        points, pinch = arm_points(robot, side, body, q)
        # The gripper and what it holds go precisely where planned and may pass
        # HELD_ALLOWANCE_M closer than the rest of the arm.
        radii = np.where(gripper_mask(robot, side), -HELD_ALLOWANCE_M, 0.0)
        if held:
            points = np.vstack((points, [pinch + np.asarray(offset) for offset, _ in held]))
            # The held object may pass HELD_ALLOWANCE_M closer than the arm.
            radii = np.r_[radii, [radius - HELD_ALLOWANCE_M for _, radius in held]]
        if len(scene.points):
            worst = min(worst, float(np.min(scene.clearance(points) - radii)))
        for centre, radius in extra:
            distance = np.linalg.norm(points - np.asarray(centre), axis=1) - radii - radius
            worst = min(worst, float(distance.min()))
    return worst


def hand_yaw_options(geometry):
    """Finger directions that close the pads across a graspable width."""
    half = np.asarray(geometry["half_size_m"])
    yaw = geometry.get("yaw_rad", 0.0)
    if geometry.get("shape") == "round":
        return None  # any finger direction works for a round object
    options = []
    if 2 * half[1] <= GRIPPER_OPENING_M:
        options += [yaw, yaw + np.pi]  # fingers along the long side, pads across the short side
    if 2 * half[0] <= GRIPPER_OPENING_M:
        options += [yaw + np.pi / 2, yaw - np.pi / 2]
    if not options:
        raise PlanningError(
            "grasp",
            f"Too wide for the two-finger gripper: narrowest side {2 * half[1] * 100:.1f} cm, "
            f"the open fingers fit {GRIPPER_OPENING_M * 100:.1f} cm",
        )
    return [wrap(o) for o in options]


def best_hand_yaw(options, heading):
    if options is None:
        return heading
    return min(options, key=lambda option: abs(wrap(option - heading)))


# Shoulder (pitch joint) to pinch point, fully stretched, is 0.56 m on the G1;
# keep REACH_MARGIN_M in hand so a stance is never planned at full stretch.
REACH_LIMIT_M = 0.53
POSTURES_PER_STANCE = 8


def shoulder_offsets(robot, posture):
    """Both shoulder joints for a posture, in the frame of feet at (0, 0, 0)."""
    cache = robot.__dict__.setdefault("_shoulders", {})
    key = (posture.height, posture.pitch)
    if key not in cache:
        _, _ = arm_points(robot, "left", (posture.pelvis_pose(np.zeros(3)), posture.balance_joints), robot.goal[:14])
        model, data = robot.planning_solver().model, robot._fk_data
        namespace = robot.tools.robot.namespace
        cache[key] = {
            side: data.xpos[model.body(namespace + f"{side}_shoulder_pitch_link").id].copy()
            for side in ("left", "right")
        }
    return cache[key]


def within_reach(robot, side, posture, stance, targets):
    """Cheap test before any IK: are all targets inside the arm's stretch?"""
    offset = shoulder_offsets(robot, posture)[side]
    shoulder = np.r_[np.asarray(stance[:2]) + rotation_z(stance[2])[:2, :2] @ offset[:2], offset[2]]
    return max(np.linalg.norm(np.asarray(position) - shoulder) for position, _ in targets) <= REACH_LIMIT_M


def plan_reach(
    robot,
    scene,
    side,
    focus,
    target_fn,
    current_stance,
    navigator,
    *,
    reaches=(0.40, 0.36, 0.44, 0.32, 0.48, 0.52, 0.28),
    laterals=(0.16, 0.10, 0.22, 0.04),
    max_turn_deg=180,
    max_evaluations=90,
    trace=None,
    held=(),
    choices=3,
    avoid=(),
):
    """Choose a stance (feet x, y, yaw) and posture from which the hand reaches.

    `target_fn(stance)` returns [(position, quaternion), ...] hand poses to
    check, e.g. above-object and grasp poses. The robot faces the focus point.
    Candidates are ranked by real walking distance (around obstacles), comfort
    of reach and room around the body; the best of the first few that pass
    `check_reach` is chosen.
    """
    started = time.monotonic()
    focus = np.asarray(focus, dtype=float)
    sign = 1 if side == "left" else -1
    direction_to_robot = np.arctan2(current_stance[1] - focus[1], current_stance[0] - focus[0])
    walking, _ = navigator._costs()

    def walk_cost(stance):
        """Walking distance to just outside the stance, or a shuffle from here."""
        best = np.inf
        for angle in np.radians(np.arange(0, 360, 45)):
            point = stance[:2] + 0.35 * np.array([np.cos(angle), np.sin(angle)])
            best = min(best, walking[navigator.cell(point)] + 0.35)
        here = np.linalg.norm(stance[:2] - current_stance[:2])
        return min(best, 1.5 * here) if here <= SHUFFLE_RANGE_M else best

    stand = standing()
    rejected = {"body_clearance": 0, "unreachable": 0, "reach": 0}
    candidates = []
    for turn in np.radians(np.arange(0, max_turn_deg + 1, 15)):
        for angle in {direction_to_robot + turn, direction_to_robot - turn}:
            heading = wrap(angle + np.pi)  # face the focus point
            for reach in reaches:
                for lateral in laterals:
                    offset = rotation_z(heading)[:2, :2] @ np.array([reach, sign * lateral])
                    stance = np.array([focus[0] - offset[0], focus[1] - offset[1], heading])
                    clearance = body_clearance(scene, stand, stance)
                    if clearance < BODY_MARGIN_M or body_clearance(scene, gait(), stance) < GAIT_MARGIN_M:
                        rejected["body_clearance"] += 1
                        continue
                    walk = walk_cost(stance)
                    if not np.isfinite(walk):
                        rejected["unreachable"] += 1
                        continue
                    cost = (
                        walk
                        + 0.25 * abs(wrap(heading - current_stance[2]))
                        + 1.5 * abs(reach - 0.40)
                        + 1.0 * abs(lateral - 0.16)
                        # GR00T-WBC lands within ~5-10 cm of a goal: a stance
                        # wedged against furniture costs like a longer walk.
                        + 5.0 * max(0.0, ROOMY_CLEARANCE_M - clearance)
                    )
                    candidates.append((cost, tuple(stance), reach, lateral))
    candidates.sort(key=lambda row: row[0])
    ordered_postures = sorted(postures(), key=lambda p: p.cost)
    evaluated = 0
    best_failure = None
    fallback = None
    good = []
    blocked_stances = []
    for cost, stance, reach, lateral in candidates:
        # A fixed amount of work, not wall time: the same scene gives the same plan.
        if evaluated >= max_evaluations or len(good) >= choices or time.monotonic() - started > 60:
            break
        if good and cost > min(row[0] for row in good):
            break  # nothing later can beat what was found
        stance = np.asarray(stance)
        if any(
            np.linalg.norm(stance[:2] - np.asarray(other)[:2]) < 0.15 and abs(wrap(stance[2] - other[2])) < 0.5
            for other in avoid
        ):
            continue  # already tried from there
        if any(
            np.linalg.norm(stance[:2] - other[:2]) < 0.12 and abs(wrap(stance[2] - other[2])) < 0.3
            for other in blocked_stances
        ):
            rejected["arm_blocked"] = rejected.get("arm_blocked", 0) + 1
            continue  # a neighbour already showed the arm would lie in or over something
        entry = navigator.entry_for(stance)
        if entry is None:
            rejected["unreachable"] += 1
            continue
        targets = target_fn(stance)
        tried = blocked = 0
        for posture in ordered_postures:
            if tried >= POSTURES_PER_STANCE:
                break
            if blocked >= 3 and blocked * 2 >= tried:
                blocked_stances.append(stance)
                break
            if not within_reach(robot, side, posture, stance, targets):
                continue
            if posture is not stand and body_clearance(scene, posture, stance) < POSTURE_MARGIN_M:
                continue
            tried += 1
            evaluated += 1
            verdict, position_error, angle_error, solutions = check_reach(
                robot, scene, side, posture, stance, targets, held
            )
            if trace is not None:
                trace.append(
                    {
                        "stance": np.round(stance, 3).tolist(),
                        "posture": [posture.height, posture.pitch],
                        "verdict": verdict,
                        "position_error_m": round(float(position_error), 4),
                        "angle_error_rad": round(float(angle_error), 3),
                    }
                )
            if verdict in ("ok", "edge"):
                found = {
                    "stance": stance,
                    "posture": posture,
                    "reach_m": reach,
                    "lateral_m": lateral,
                    "approach": "shuffle" if entry[1] else "walk in",
                    "ik_error_m": position_error,
                    "ik_angle_rad": angle_error,
                    "arm_solution": solutions[-1],
                    "evaluated": evaluated,
                    "rejected": rejected,
                }
                if verdict == "ok":
                    # Total effort: walking, bending, and side-stepping into place.
                    good.append((cost + posture.cost + (0.3 if entry[1] else 0.0), found))
                    break
                fallback = fallback or found
            elif verdict == "blocked":
                blocked += 1
                rejected["arm_blocked"] = rejected.get("arm_blocked", 0) + 1
            if best_failure is None or position_error < best_failure[0]:
                best_failure = (position_error, stance.tolist(), posture.height, posture.pitch)
        else:
            rejected["reach"] += 1
    if good:
        return min(good, key=lambda row: row[0])[1]
    if fallback is not None:
        return fallback  # reachable, but only at the edge of the arm's reach
    details = {"evaluated_postures": evaluated, "rejected_stances": rejected}
    if best_failure is not None:
        details["closest_miss_m"] = round(best_failure[0], 4)
        details["closest_stance"] = best_failure[1]
    raise PlanningError(
        "stance",
        "No standing spot lets the arm reach that point without the body hitting what the cameras see",
        **details,
    )


TURN_MARGIN_M = 0.06  # stepping envelope while turning on the spot
SHUFFLE_RANGE_M = 0.40  # turn on the spot, then step sideways/forward into place
SHUFFLE_MARGIN_M = 0.04
WALK_MARGIN_M = 0.04
GAIT_MARGIN_M = 0.015  # stepping envelope over the last steps into a stance


def turn_clear(scene, xy, yaw_from, yaw_to, margin=TURN_MARGIN_M):
    """Can the robot turn in place at xy from one heading to another?

    Turning steps sweep the gait envelope and drift 3-11 cm (measured), so the
    envelope keeps `margin` from everything all the way round.
    """
    stepping = gait()
    turn = yaw_to - yaw_from
    for fraction in np.linspace(0, 1, max(2, int(abs(turn) / 0.15) + 1)):
        if body_clearance(scene, stepping, np.r_[xy, yaw_from + fraction * turn]) < margin:
            return False
    return True


class Navigator:
    """Walkable floor seen by the cameras, and which of it the robot can reach.

    The same body model decides stances and routes: transit keeps a radius from
    every seen point, and the last step in to a stance is checked with the exact
    measured leg/torso surface, so the robot can stand close to a table.
    """

    def __init__(self, scene, start, resolution=0.05, radius=TRANSIT_RADIUS_M, extent=4.0):
        self.scene = scene
        self.start = np.asarray(start, dtype=float)
        self.resolution = resolution
        self.lo = self.start[:2] - extent
        self.shape = np.full(2, int(np.ceil(2 * extent / resolution)) + 1)
        points = scene.points
        occupied = np.zeros(self.shape, dtype=bool)
        if len(points):
            band = points[(points[:, 2] > 0.03) & (points[:, 2] < 1.6)]
            cells = np.floor((band[:, :2] - self.lo) / resolution).astype(int)
            keep = np.all((cells >= 0) & (cells < self.shape), axis=1)
            occupied[cells[keep, 0], cells[keep, 1]] = True
        self.distance = ndimage.distance_transform_edt(~occupied) * resolution
        free = self.distance >= radius
        start_cell = self.cell(self.start[:2])
        # The robot is where it is: let it leave a tight spot it already stands in.
        free[
            max(0, start_cell[0] - 3) : start_cell[0] + 4,
            max(0, start_cell[1] - 3) : start_cell[1] + 4,
        ] = True
        labels, _ = ndimage.label(free, structure=np.ones((3, 3)))
        self.free = free
        self.reachable = labels == labels[start_cell]
        self._entries = {}
        self._walk = None

    def cell(self, xy):
        index = np.round((np.asarray(xy, dtype=float) - self.lo) / self.resolution).astype(int)
        return tuple(np.clip(index, 0, self.shape - 1))

    def world(self, index):
        return self.lo + np.asarray(index) * self.resolution

    def _costs(self):
        """Walking cost (m) from the start to every reachable cell, and the path tree."""
        if self._walk is None:
            cost = np.full(self.shape, np.inf)
            parent = {}
            start = self.cell(self.start[:2])
            cost[start] = 0.0
            queue = [(0.0, start)]
            steps = [(dx, dy, self.resolution * np.hypot(dx, dy)) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy]
            while queue:
                here, current = heapq.heappop(queue)
                if here > cost[current]:
                    continue
                for dx, dy, step in steps:
                    nxt = (current[0] + dx, current[1] + dy)
                    if not (0 <= nxt[0] < self.shape[0] and 0 <= nxt[1] < self.shape[1]) or not self.reachable[nxt]:
                        continue
                    # Prefer the middle of free space so the robot keeps its distance.
                    new = here + step + self.resolution * max(0.0, 0.5 - self.distance[nxt]) * 2
                    if new < cost[nxt]:
                        cost[nxt] = new
                        parent[nxt] = current
                        heapq.heappush(queue, (new, nxt))
            self._walk = (cost, parent)
        return self._walk

    def segment_clear(self, a, goal):
        """Walk from a to the goal stance facing the travel direction, then turn.

        The heading wobbles while walking, so the stepping envelope keeps
        WALK_MARGIN_M at +/-0.25 rad, and GAIT_MARGIN_M over the last 10 cm.
        """
        stepping = gait()
        segment = goal[:2] - a
        length = np.linalg.norm(segment)
        heading = np.arctan2(segment[1], segment[0])
        for t in np.linspace(0, 1, max(2, int(length / 0.06) + 1)):
            margin = WALK_MARGIN_M if (1 - t) * length > 0.10 else GAIT_MARGIN_M
            for delta in (-0.25, 0.0, 0.25):
                if body_clearance(self.scene, stepping, np.r_[a + t * segment, heading + delta]) < margin:
                    return False
        # The turn in place at the goal sweeps the hips and knees too.
        turn = wrap(goal[2] - heading)
        return abs(turn) <= 0.2 or turn_clear(self.scene, goal[:2], heading, heading + turn)

    def shuffle_clear(self, a, goal):
        """Short sideways/forward steps from a to the goal, already facing its heading.

        The stepping envelope keeps SHUFFLE_MARGIN_M on the way (the gait lands
        within a few centimetres) and GAIT_MARGIN_M over the last 10 cm.
        """
        stepping = gait()
        segment = goal[:2] - a
        length = np.linalg.norm(segment)
        for t in np.linspace(0, 1, max(2, int(length / 0.04) + 1)):
            margin = SHUFFLE_MARGIN_M if (1 - t) * length > 0.10 else GAIT_MARGIN_M
            if body_clearance(self.scene, stepping, np.r_[a + t * segment, goal[2]]) < margin:
                return False
        return True

    def entry_for(self, goal):
        """How to step into `goal`: (entry point, shuffle) or None.

        Options are ranked by walking distance from where the robot stands plus
        turning, as a person picks a way in:
        - walk to a point around the goal and step straight in, facing it,
          turning at the goal only where the whole sweep is clear;
        - walk to a point near the goal (or stay, if already near), turn on the
          spot to the final heading there, then shuffle sideways/forward into
          place. This fits tight gaps, e.g. between two tables.
        """
        goal = np.asarray(goal, dtype=float)
        key = tuple(np.round(goal, 3))
        if key in self._entries:
            return self._entries[key]
        cost, _ = self._costs()
        here = self.start
        options = []
        gap = float(np.linalg.norm(goal[:2] - here[:2]))
        if gap <= SHUFFLE_RANGE_M:
            options.append((1.5 * gap + 0.1 * abs(wrap(goal[2] - here[2])), here[:2].copy(), True, here[2]))
        for distance in (0.30, 0.45):
            for angle in np.radians(np.arange(0, 360, 30)):
                point = goal[:2] + distance * np.array([np.cos(angle), np.sin(angle)])
                walk = cost[self.cell(point)]
                if not np.isfinite(walk):
                    continue
                options.append((walk + distance + 0.1 * abs(wrap(goal[2] - (angle + np.pi))), point, False, None))
                if distance <= SHUFFLE_RANGE_M:
                    arriving = np.arctan2(point[1] - here[1], point[0] - here[0])
                    turn = abs(wrap(goal[2] - arriving))
                    options.append((walk + 1.5 * distance + 0.1 * turn + 0.05, point, True, arriving))
        entry = None
        for _, point, shuffle, heading in sorted(options, key=lambda row: row[0])[:16]:
            if shuffle:
                clear = turn_clear(self.scene, point, heading, heading + wrap(goal[2] - heading)) and self.shuffle_clear(
                    point, goal
                )
            else:
                clear = self.segment_clear(point, goal)
            if clear:
                entry = (point, shuffle)
                break
        self._entries[key] = entry
        return entry

    def route(self, goal):
        """Waypoints to the goal stance and whether to turn first and shuffle in."""
        goal = np.asarray(goal, dtype=float)
        approach = self.entry_for(goal)
        if approach is None:
            raise PlanningError(
                "route",
                "That standing spot cannot be reached: the gaps around it are narrower than the robot",
                goal_pose=np.round(goal, 3).tolist(),
            )
        entry, shuffle = approach
        if shuffle and np.linalg.norm(entry - self.start[:2]) < 1e-6:
            return {"waypoints": [goal[:2]], "turn_first": True}
        _, parent = self._costs()
        path = [self.cell(entry)]
        while path[-1] in parent:
            path.append(parent[path[-1]])
        path = [self.world(c) for c in reversed(path)]
        simplified = [path[0]]
        anchor = 0
        for index in range(2, len(path)):
            a, b = path[anchor], path[index]
            samples = np.linspace(0, 1, max(2, int(np.linalg.norm(b - a) / (self.resolution / 2))))
            if not all(self.reachable[self.cell(a + t * (b - a))] for t in samples):
                simplified.append(path[index - 1])
                anchor = index - 1
        simplified.append(path[-1])
        waypoints = [p for p in simplified[1:] if np.linalg.norm(p - self.start[:2]) > 0.08]
        if shuffle and (not waypoints or np.linalg.norm(waypoints[-1] - entry) > 0.03):
            waypoints.append(entry)
        waypoints.append(goal[:2])
        return {"waypoints": waypoints, "turn_first": shuffle}


def hand_waypoints(start, goal, scene, safe_z, exclude_radius=0.0, target=None):
    """Raise, travel above, then descend, so the hand never sweeps through a surface."""
    start, goal = np.asarray(start, dtype=float), np.asarray(goal, dtype=float)
    path = []
    travel_z = max(safe_z, start[2], goal[2])
    if start[2] < travel_z - 0.02:
        path.append(np.r_[start[:2], travel_z])
    if np.linalg.norm(goal[:2] - start[:2]) > 0.02:
        path.append(np.r_[goal[:2], travel_z])
    path.append(goal)
    return path


def natural_hand_yaw(stance, side, target_xy):
    """Point the fingers from the shoulder toward the target, as a person would."""
    sign = 1 if side == "left" else -1
    shoulder = np.asarray(stance[:2]) + rotation_z(stance[2])[:2, :2] @ [-0.04, sign * 0.15]
    direction = np.asarray(target_xy[:2]) - shoulder
    return float(np.arctan2(direction[1], direction[0]))


def hand_quaternion(yaw):
    """Fingers horizontal, pointing along `yaw`, closing sideways."""
    return yaw_quaternion(yaw)
