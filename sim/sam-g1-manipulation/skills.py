"""Pick, place, release, walk and look: whole-body skills the agent can call.

Every skill sees through the robot's cameras, plans where to stand and how to
hold the body, moves, and reports what it measured in plain language. A user's
direct request to let go is always obeyed. No simulator object poses are read.
"""

from __future__ import annotations

import numpy as np

from body import (
    STAND_HEIGHT,
    Body,
    MotionError,
    feet_pose,
    gait,
    pelvis_pose2d,
    postures,
    rotation_z,
    standing,
    wrap,
)
from perception import (
    capture,
    object_geometry,
    obstacle_points,
    point_cloud,
    project,
    surface_at,
    surface_free,
)
from planner import (
    BODY_MARGIN_M,
    GAIT_MARGIN_M,
    IK_TOLERANCE_RAD,
    POSTURE_MARGIN_M,
    Navigator,
    PlanningError,
    Scene,
    arm_points,
    best_hand_yaw,
    body_clearance,
    check_reach,
    hand_quaternion,
    hand_yaw_options,
    measured_body,
    natural_hand_yaw,
    plan_reach,
    solve_arm,
    swept_clearance,
    turn_clear,
)
from scene_memory import memory_for

SIDES = ("left", "right")
NEXT_TO_GAP_M = 0.06  # space left between an object and the one it is put next to
# The open Dex1 pads reach ~4 cm below the pinch point: hover this far above a top.
PREGRASP_CLEARANCE_M = 0.07
SWEEP_MARGIN_M = 0.035  # arm surface and held object vs. seen points while swinging
# Hand points relative to the standing pelvis (forward, outward, up), fingers forward.
READY_POINT = (0.20, 0.20, 0.18)
CARRY_POINT = (0.28, 0.17)
# Where a free hand waits while the body bends (forward, outward, up from the
# pelvis in the coming posture): beside the hip, tucked in front of the belly or
# chest, high at the side, or behind the hip, whichever stays clear.
PARK_POINTS = (
    (0.05, 0.28, 0.22),
    (0.0, 0.30, 0.12),
    (0.18, 0.10, 0.14),
    (0.15, 0.12, 0.30),
    (0.12, 0.26, 0.30),
    (0.0, 0.26, 0.38),
    (-0.10, 0.22, 0.05),
    (-0.05, 0.30, 0.05),
)


class SkillError(RuntimeError):
    def __init__(self, stage, reason, suggestion=None, **details):
        super().__init__(reason)
        self.stage = stage
        self.suggestion = suggestion
        self.details = details


def describe_size(geometry):
    half = np.asarray(geometry["half_size_m"]) * 200
    return f"{half[0]:.0f}×{half[1]:.0f}×{half[2]:.0f} cm"


def object_spheres(centre, half, round_shape=False):
    """Spheres stacked up an upright object's axis (for sweep checks)."""
    radius = (float(np.max(half[:2])) if round_shape else float(np.linalg.norm(half[:2]))) + 0.008
    low, high = centre[2] - half[2] + radius, centre[2] + half[2] - radius
    heights = np.linspace(low, high, max(1, int(np.ceil((high - low) / radius)) + 1)) if high > low else [centre[2]]
    return [(np.r_[centre[:2], z], radius) for z in heights]


def grasp_height(geometry):
    """Pads are 26 mm tall: keep them clear of the support and below the top."""
    top, bottom = geometry["top_z_m"], geometry["bottom_z_m"]
    centre = (top + bottom) / 2 if geometry.get("bottom_certain", True) else top - 0.02
    low, high = bottom + 0.016, top - 0.010
    return float(np.clip(centre, low, high)) if low <= high else float((top + bottom) / 2)


class Skills:
    def __init__(self, console, epoch):
        self.console = console
        self.epoch = epoch
        self.body = Body(console, epoch, report=self.note)
        if not hasattr(console, "held"):
            console.held = {}

    @property
    def robot(self):
        return self.console.robot

    def note(self, text, **details):
        self.console.message("skill", text, details=details or None)

    # ---- seeing ----------------------------------------------------------
    def frame(self, frame_id):
        frames = getattr(self.console, "frames", {})
        if frame_id not in frames:
            raise SkillError(
                "perception",
                f"Camera frame {frame_id} is no longer available",
                "Use the frame_id of the latest image",
            )
        return frames[frame_id]

    def look(self, camera="head"):
        return capture(self.console, camera)

    def held_spheres(self):
        """Spheres around objects in the robot's hands, left out of measurements."""
        spheres = []
        for side, held in self.console.held.items():
            hand, _ = self.body.hand(side)
            half = np.asarray(held["geometry"]["half_size_m"])
            centre = hand - [0, 0, held["hand_above_bottom_m"] - half[2]]
            spheres.append((centre, float(np.linalg.norm(half)) + 0.03))
        return spheres

    def scene(self, *sensors):
        scene = Scene(focus=feet_pose(self.robot)[:2])
        for sensor in sensors:
            scene.add(obstacle_points(sensor))
        return scene

    def measure_near(self, geometry, camera="head", radius=0.0, chroma=None):
        """Measure an object again where `geometry` says it is, from one camera.

        Returns (measurement or None, sensor). `radius` widens the image box,
        e.g. to look for something that may have shifted or tipped over.
        """
        sensor = self.look(camera)
        center = np.asarray(geometry["center_world_m"])
        half = np.asarray(geometry["half_size_m"]) + radius
        corners = np.array(
            [
                center + rotation_z(geometry.get("yaw_rad", 0.0)) @ (half * [sx, sy, sz])
                for sx in (-1, 1)
                for sy in (-1, 1)
                for sz in (-1, 1)
            ]
        )
        pixels = project(sensor, corners)
        h, w = sensor["depth"].shape
        if not np.isfinite(pixels).all():
            return None, sensor
        (u0, v0), (u1, v1) = pixels.min(axis=0), pixels.max(axis=0)
        if u1 < 0 or v1 < 0 or u0 > w or v0 > h:
            return None, sensor
        pad_u, pad_v = (u1 - u0) * 0.1 + 3, (v1 - v0) * 0.1 + 3
        box = np.clip(
            [(v0 - pad_v) / h * 1000, (u0 - pad_u) / w * 1000, (v1 + pad_v) / h * 1000, (u1 + pad_u) / w * 1000],
            0,
            1000,
        )
        try:
            return object_geometry(sensor, box, ignore=self.held_spheres(), chroma=chroma), sensor
        except ValueError:
            return None, sensor

    def reacquire(self, geometry):
        """Measure the same object again from the head camera after moving."""
        center = np.asarray(geometry["center_world_m"])
        half = np.asarray(geometry["half_size_m"])
        fresh, sensor = self.measure_near(geometry, chroma=geometry.get("chroma"))
        if fresh is None:
            return geometry, sensor
        if np.linalg.norm(np.subtract(fresh["center_world_m"][:2], center[:2])) > 0.04:
            return geometry, sensor  # a different object; keep the earlier measurement
        if np.max(np.abs(np.subtract(fresh["half_size_m"][:2], half[:2]))) > 0.006:
            # Part of it is hidden (e.g. behind the robot's own hand) in this view;
            # a partial outline would shift the centre, so keep the full view.
            return geometry, sensor
        if not fresh.get("bottom_certain", True) and geometry.get("bottom_certain", True):
            fresh["bottom_z_m"] = geometry["bottom_z_m"]
            fresh["bottom_certain"] = True
            fresh["center_world_m"][2] = (fresh["top_z_m"] + fresh["bottom_z_m"]) / 2
            fresh["half_size_m"][2] = (fresh["top_z_m"] - fresh["bottom_z_m"]) / 2
        return fresh, sensor

    # ---- getting there -------------------------------------------------------
    def step_back_if_tight(self, scene, carrying_side=None, force=False):
        """Leave a tight spot (e.g. close to a table) by stepping back before turning."""
        stand, stepping = standing(), gait()
        feet = feet_pose(self.robot)
        if not force and body_clearance(scene, stand, feet) >= 0.10:
            return False
        here = body_clearance(scene, stepping, feet)
        best = None
        for back, side_step in ((-0.20, 0.0), (-0.15, 0.10), (-0.15, -0.10), (-0.30, 0.0), (-0.25, 0.15), (-0.25, -0.15)):
            xy = feet[:2] + rotation_z(feet[2])[:2, :2] @ [back, side_step]
            # The stepping legs must not come any closer to things on the way out,
            # and must end up clear.
            on_the_way = [
                body_clearance(scene, stepping, np.r_[feet[:2] + t * (xy - feet[:2]), feet[2]])
                for t in (0.25, 0.5, 0.75, 1.0)
            ]
            if min(on_the_way) < min(here, GAIT_MARGIN_M) - 0.003 or on_the_way[-1] < GAIT_MARGIN_M:
                continue
            # Prefer a spot where turning around is possible.
            room = on_the_way[-1] + (0.05 if turn_clear(scene, xy, feet[2], feet[2] + 2 * np.pi) else 0.0)
            if best is None or room > best[0]:
                best = (room, xy)
        if best is None:
            return False
        self.note("Stepping back from the table before turning")
        offset = rotation_z(feet[2])[:2, :2] @ np.asarray(stand.pelvis_offset)
        self.body.freeze_arms()
        self.console.motion_kind = "walk"
        self.console.carrying_side = carrying_side
        self.body.guard = lambda pose: body_clearance(scene, stepping, pose)
        try:
            self.body._dock(best[1] + offset, carrying_side, timeout=5.0)
        except MotionError:
            pass  # stopped short of something: wherever it is now is the open spot
        finally:
            self.console.motion_kind = None
            self.body.guard = None
        return True

    def walk_to_stance(self, stance, navigator, carrying_side=None):
        feet = feet_pose(self.robot)
        if np.linalg.norm(stance[:2] - feet[:2]) < 0.04 and abs(wrap(stance[2] - feet[2])) < 0.08:
            return {"walked_m": 0.0}
        scene = navigator.scene
        stepping = gait()

        def guard(feet):
            return body_clearance(scene, stepping, feet)

        walked = 0.0
        for attempt in range(3):
            route = self._route(stance, scene, carrying_side)
            offset = rotation_z(stance[2])[:2, :2] @ np.asarray(standing().pelvis_offset)
            path = [np.asarray(w) for w in route["waypoints"][:-1]] + [stance[:2] + offset]
            distance = sum(
                np.linalg.norm(b - a) for a, b in zip([pelvis_pose2d(self.robot)[:2]] + path[:-1], path)
            )
            self.note(
                (
                    f"Walking {distance:.1f} m, then turning to face the spot and stepping in"
                    if len(path) > 1
                    else "Turning to face the spot, then stepping into place"
                )
                if route["turn_first"]
                else f"Walking {distance:.1f} m, facing where I go, to stand facing the target",
                route=[np.round(w, 2).tolist() for w in path],
                stance=np.round(stance, 3).tolist(),
            )
            start = pelvis_pose2d(self.robot)[:2]
            try:
                arrival = self.body.walk_path(
                    path,
                    final_yaw=stance[2],
                    carrying_side=carrying_side,
                    guard=guard,
                    turn_first=route["turn_first"],
                )
                arrival["walked_m"] = round(float(walked + distance), 2)
                return arrival
            except RuntimeError as exc:
                if self.epoch != self.console.epoch:
                    raise  # replaced by a newer instruction, not blocked
                walked += float(np.linalg.norm(pelvis_pose2d(self.robot)[:2] - start))
                if attempt == 2:
                    raise SkillError(
                        "walking",
                        f"Walking stopped: {exc}",
                        "Look again and pick a clearer approach",
                    ) from exc
                # Like a person brushing a table: step away, then find another way.
                self.note("Too close to something on the way; stepping away and finding another way")
                self.body.guard = guard
                try:
                    self.body._back_off(carrying_side, distance=0.08)
                finally:
                    self.body.guard = None
        raise AssertionError("unreachable")

    def _route(self, stance, scene, carrying_side):
        """Plan from where the robot stands, stepping into the open first if needed."""
        # Close to furniture, step back into the open before turning or walking off.
        self.step_back_if_tight(scene, carrying_side)
        navigator = Navigator(scene, feet_pose(self.robot))
        feet = feet_pose(self.robot)
        try:
            route = navigator.route(stance)
        except PlanningError:
            route = None
        # Walking starts by turning toward the first waypoint; if that turn would
        # sweep into something here, step back into the open first and plan again.
        needs_room = route is None
        if route is not None and (len(route["waypoints"]) > 1 or not route["turn_first"]):
            first = np.asarray(route["waypoints"][0])
            heading = np.arctan2(first[1] - feet[1], first[0] - feet[0])
            needs_room = np.linalg.norm(first - feet[:2]) > 0.25 and not turn_clear(
                scene, feet[:2], feet[2], feet[2] + wrap(heading - feet[2])
            )
        if needs_room and self.step_back_if_tight(scene, carrying_side, force=True):
            navigator = Navigator(scene, feet_pose(self.robot))
        try:
            return navigator.route(stance)
        except PlanningError as exc:
            raise SkillError(
                "route", str(exc), "Try another approach side, or move obstacles", **exc.details
            ) from exc

    # ---- arm configurations -------------------------------------------------
    def body_pose_configuration(self, side, forward, outward, up):
        """Arm angles for a hand point fixed to the torso (cached per robot).

        Solved once for the standing posture; the arm then keeps these angles
        and moves with the body, like a person's relaxed or carrying arm.
        """
        cache = self.robot.__dict__.setdefault("_pose_configurations", {})
        key = (side, round(forward, 3), round(outward, 3), round(up, 3))
        if key not in cache:
            stand = standing()
            pelvis = stand.pelvis_pose(np.zeros(3))
            sign = 1 if side == "left" else -1
            target = np.r_[pelvis[0] + forward, pelvis[1] + sign * outward, pelvis[2] + up]
            configuration, error, angle = solve_arm(
                self.robot, side, (pelvis, stand.balance_joints), (target, hand_quaternion(0.0))
            )
            if configuration is None or error > 0.02 or angle > IK_TOLERANCE_RAD:
                raise SkillError("arm", f"No comfortable arm pose for the {side} hand at {key[1:]}")
            cache[key] = configuration
        return cache[key]

    def swing_arm(self, side, configuration, posture=None, scene=None, held=(), extra=()):
        """Joint-space swing (with an optional posture change) if the sweep is clear.

        The other arm rides along with the body; when bending would carry it into
        something, it is parked beside the torso in the same motion. Returns
        False without moving when no clear motion was found.
        """
        robot = self.robot
        arm = slice(0, 7) if side == "left" else slice(7, 14)
        start_q = robot.goal[:14].copy()
        goal = start_q.copy()
        goal[arm] = np.asarray(configuration)[arm]
        start = measured_body(robot)
        end = (posture.pelvis_pose(feet_pose(robot)), posture.balance_joints) if posture is not None else start
        order = None
        if posture is not None:
            # Reaching down, the arm goes out first; rising, the body goes first.
            _, hand_now = arm_points(robot, side, start, start_q)
            _, hand_bent = arm_points(robot, side, end, start_q)
            order = "arm_first" if hand_bent[2] < hand_now[2] - 0.01 else "body_first"
        sides = [side]
        if scene is not None or extra:
            scene = scene or Scene()
            if swept_clearance(robot, scene, side, (start, start_q), (end, goal), held, extra, order=order) < SWEEP_MARGIN_M:
                return False
            other = "right" if side == "left" else "left"
            if posture is not None and (
                swept_clearance(
                    robot, scene, other, (start, start_q), (end, goal), self.held_shape(other), extra, order=order
                )
                < SWEEP_MARGIN_M
            ):
                parked = self._park(other, (start, start_q), end, goal, scene, extra, order)
                if parked is None:
                    return False
                goal = parked
                sides.append(other)
        self.body.move_arm(
            sides,
            goal,
            None if posture is None else posture.height,
            None if posture is None else posture.pitch,
            order=order,
        )
        return True

    def _park(self, side, start, end_body, goal, scene, extra, order):
        """Arm angles that keep one hand clear beside the torso in the coming posture."""
        robot = self.robot
        arm = slice(0, 7) if side == "left" else slice(7, 14)
        feet = feet_pose(robot)
        sign = 1 if side == "left" else -1
        pelvis = end_body[0]
        for forward, outward, up in PARK_POINTS:
            target = np.r_[pelvis[:2] + rotation_z(feet[2])[:2, :2] @ [forward, sign * outward], pelvis[2] + up]
            configuration, error, _ = solve_arm(robot, side, end_body, (target, None), seeds=[goal])
            if configuration is None or error > 0.02:
                continue
            trial = goal.copy()
            trial[arm] = configuration[arm]
            if (
                swept_clearance(robot, scene, side, start, (end_body, trial), self.held_shape(side), extra, order=order)
                >= SWEEP_MARGIN_M
            ):
                return trial
        return None

    def held_shape(self, side):
        """Spheres (offset from the pinch point, radius) covering the held object."""
        held = self.console.held.get(side)
        if held is None:
            return ()
        half = np.asarray(held["geometry"]["half_size_m"])
        centre = np.array([0.0, 0.0, half[2] - held["hand_above_bottom_m"]])
        return object_spheres(centre, half, held["geometry"].get("shape") == "round")

    def ready_hand(self, side, scene=None, posture=None, extra=()):
        """Free hand up in front of the chest, elbow bent, moving with the body."""
        configuration = self.body_pose_configuration(side, *READY_POINT)
        if self.swing_arm(side, configuration, posture, scene, extra=extra):
            return
        # Something is in the way: rise straight up first, then try again.
        hand, _ = self.body.hand(side)
        self.body.move_hand(side, hand + [0, 0, 0.12], speed=0.25, tolerance=0.03)
        if not self.swing_arm(side, configuration, posture, scene, extra=extra):
            self.note(f"Keeping the {side} hand where it is; swinging it in would brush what is nearby")
            if posture is not None:
                self.body.set_posture(posture.height, posture.pitch)

    def reach_any(self, postures_to_try, side, target, quaternion, scene, held=(), tolerance=0.006):
        """reach_pose with the first posture whose swing is clear (a blocked one does not move)."""
        failure = None
        for posture in postures_to_try:
            try:
                return self.reach_pose(side, target, quaternion, posture, scene, held=held, tolerance=tolerance)
            except SkillError as exc:
                if exc.stage != "reach":
                    raise
                failure = exc
        raise failure

    def reach_pose(self, side, target, quaternion, posture, scene, held=(), extra=(), tolerance=0.006):
        """Bend into `posture` while the arm swings to the configuration that puts
        the hand at `target`; then servo the last few millimetres.

        Returns the remaining hand position error (m).
        """
        robot = self.robot
        goal_body = (posture.pelvis_pose(feet_pose(robot)), posture.balance_joints)
        configuration, error, angle = solve_arm(robot, side, goal_body, (target, quaternion))
        if configuration is None or error > 0.015 or angle > IK_TOLERANCE_RAD:
            raise SkillError(
                "reach",
                "From where I stand my arm cannot reach that pose",
                "Look again; the target may have moved",
                planned_error_m=None if configuration is None else round(float(error), 4),
            )
        if not self.swing_arm(side, configuration, posture, scene, held, extra):
            # Come in over the top: first above the target, then straight down.
            high = target + [0.0, 0.0, 0.12]
            above, error_high, _ = solve_arm(robot, side, goal_body, (high, quaternion), seeds=[configuration])
            if above is None or error_high > 0.02 or not self.swing_arm(side, above, posture, scene, held, extra):
                raise SkillError(
                    "reach",
                    "Swinging the arm there would hit the surface or something on it",
                    "Choose a spot with more free space around it",
                )
            return self.body.move_hand(side, target, quaternion, speed=0.08, tolerance=tolerance, settle_timeout=2.5)
        # The body settles a little differently from the prediction: solve again
        # at the measured pose, then let the hand servo remove what remains.
        settled, error_settled, _ = solve_arm(robot, side, measured_body(robot), (target, quaternion), seeds=[configuration])
        if settled is not None and error_settled < 0.01:
            self.body.move_arm(side, settled, speed=0.5)
        return self.body.move_hand(side, target, quaternion, speed=0.1, tolerance=tolerance, settle_timeout=2.5)

    def settle_reach(self, side, target_fn, plan, scene, held=()):
        """After walking, confirm the hand can reach from where the robot stands.

        Walking lands within a few centimetres; near the edge of the arm's reach
        that matters. Judge the planned and nearby postures here with the same
        test the planner used, and shuffle toward the planned spot if none works.
        Returns up to three workable postures, best first, and the stance.
        """
        planned = np.asarray(plan["stance"], dtype=float)
        planned_posture = plan["posture"]
        ordered = sorted(
            postures(),
            key=lambda p: abs(p.height - planned_posture.height) + 0.3 * abs(p.pitch - planned_posture.pitch),
        )
        stand = standing()
        verdicts = {}
        for attempt in range(3):
            stance = feet_pose(self.robot)
            targets = target_fn(stance)
            good, edge = [], []
            for posture in ordered[:10]:
                if body_clearance(scene, posture, stance) < POSTURE_MARGIN_M:
                    verdicts["body_too_close"] = verdicts.get("body_too_close", 0) + 1
                    continue
                verdict = check_reach(self.robot, scene, side, posture, stance, targets, held)[0]
                verdicts[verdict] = verdicts.get(verdict, 0) + 1
                if verdict == "ok":
                    good.append(posture)
                    if len(good) == 3:
                        break
                elif verdict == "edge":
                    edge.append(posture)
            if good or edge:
                # Best first; the caller tries the next if swinging there is blocked.
                return (good + edge)[:3], stance
            # Not reachable from here: step toward the planned spot if there is room.
            focus = np.asarray(targets[-1][0][:2])
            toward = planned[:2] + 0.03 * attempt * (focus - planned[:2]) / max(np.linalg.norm(focus - planned[:2]), 1e-6)
            if body_clearance(scene, stand, np.r_[toward, planned[2]]) < BODY_MARGIN_M:
                break
            offset = rotation_z(planned[2])[:2, :2] @ np.asarray(stand.pelvis_offset)
            self.note("Adjusting my footing to reach comfortably", shortfall_attempt=attempt + 1)
            self.body.freeze_arms()
            self.console.motion_kind = "walk"
            self.body.guard = lambda feet: body_clearance(scene, gait(), feet)
            try:
                self.body._dock(toward + offset, None)
                self.body.turn_to(planned[2])
            except MotionError:
                self.body._back_off(None)
            finally:
                self.console.motion_kind = None
                self.body.guard = None
        raise SkillError(
            "reach",
            "From where I can stand, my arm does not reach that spot precisely enough",
            "Choose a spot nearer the edge of the surface or a different surface",
            verdicts=verdicts,
            planned_stance=np.round(planned, 3).tolist(),
            actual_stance=np.round(feet_pose(self.robot), 3).tolist(),
        )

    def _another_stance(self, side, scene, plan, focus, target_fn, held=()):
        """The arm cannot reach from where the robot ended up: plan another spot."""
        self.note("I can't reach it well from here; trying another place to stand")
        tried = [plan["stance"], feet_pose(self.robot)]
        navigator = Navigator(scene, feet_pose(self.robot))
        try:
            plan = plan_reach(
                self.robot, scene, side, focus, target_fn, feet_pose(self.robot), navigator, held=held, avoid=tried
            )
        except PlanningError as exc:
            raise SkillError(
                "reach",
                "From where I can stand, my arm does not reach that spot precisely enough",
                "Choose a spot nearer the edge of the surface or a different surface",
                **exc.details,
            ) from exc
        return plan, navigator

    def free_sides(self):
        return [s for s in SIDES if s not in self.console.held]

    # ---- skills --------------------------------------------------------------
    def pick_up(self, frame_id, object_box, side="auto", label=None):
        label = label or "the object"
        free = self.free_sides()
        if not free:
            raise SkillError("pick", "Both hands are already holding objects", "Place or release one first")
        sensor = self.frame(frame_id)
        try:
            geometry = object_geometry(sensor, object_box)
        except ValueError as exc:
            raise SkillError("perception", str(exc), "Observe from the head camera and select the object again") from exc
        self.note(
            f"Measured {label}: {describe_size(geometry)}, top at {geometry['top_z_m']:.2f} m",
            geometry=geometry,
        )
        options = hand_yaw_options(geometry)
        # The arm must swing around the object itself; only the final grasp
        # (fingers either side of it) and the stance search may ignore it.
        surroundings = self.scene(sensor, self.look("overview"), self.look("head"))
        scene = surroundings.without_box(
            geometry["center_world_m"], geometry["half_size_m"], geometry.get("yaw_rad", 0.0), margin=0.03
        )
        navigator = Navigator(scene, feet_pose(self.robot))
        sides = [side] if side in SIDES else self._preferred_sides(geometry["center_world_m"], free)
        sides = [s for s in sides if s in free] or free
        plan, plan_side, failures = None, None, {}
        for candidate in sides:
            def targets(stance, geometry=geometry, candidate=candidate):
                natural = natural_hand_yaw(stance, candidate, geometry["center_world_m"])
                quaternion = hand_quaternion(best_hand_yaw(options, natural))
                grasp = np.r_[geometry["center_world_m"][:2], grasp_height(geometry)]
                above = grasp + [0, 0, max(0.05, geometry["top_z_m"] - grasp[2] + PREGRASP_CLEARANCE_M)]
                return [(above, quaternion), (grasp, quaternion)]

            try:
                plan = plan_reach(
                    self.robot,
                    scene,
                    candidate,
                    geometry["center_world_m"],
                    targets,
                    feet_pose(self.robot),
                    navigator,
                )
                plan_side = candidate
                break
            except PlanningError as exc:
                failures[candidate] = {"reason": str(exc), **exc.details}
        if plan is None:
            raise SkillError(
                "stance",
                "No reachable way to stand at the object with the free hand(s)",
                "Check the object is on an open surface; try the other hand or a clearer view",
                attempts=failures,
            )
        side = plan_side
        self.note(
            f"Using the {side} hand; standing {plan['reach_m']:.2f} m from it"
            + (
                f", crouching to {plan['posture'].height:.2f} m"
                if plan["posture"].height < STAND_HEIGHT - 0.01
                else ""
            )
            + (f", leaning {plan['posture'].pitch:.1f} rad" if plan["posture"].pitch > 0 else ""),
            stance=plan["stance"].tolist(),
            ik_error_m=plan["ik_error_m"],
        )
        for hand in self.free_sides():
            self.ready_hand(hand, surroundings)
        for attempt in range(2):
            arrival = self.walk_to_stance(plan["stance"], navigator)
            geometry, _ = self.reacquire(geometry)

            def grasp_targets(stance, geometry=geometry):
                natural = natural_hand_yaw(stance, side, geometry["center_world_m"])
                quaternion = hand_quaternion(best_hand_yaw(options, natural))
                grasp = np.r_[geometry["center_world_m"][:2], grasp_height(geometry)]
                above = grasp + [0, 0, max(0.05, geometry["top_z_m"] - grasp[2] + PREGRASP_CLEARANCE_M)]
                return [(above, quaternion), (grasp, quaternion)]

            try:
                choices, stance = self.settle_reach(side, grasp_targets, plan, scene)
                break
            except SkillError as exc:
                if exc.stage != "reach" or attempt == 1:
                    raise
                plan, navigator = self._another_stance(side, scene, plan, geometry["center_world_m"], grasp_targets)
        (above, quaternion), (grasp, _) = grasp_targets(stance)
        self.robot.grip(side, 0.0)  # opens while the arm swings over
        error_above = self.reach_any(choices, side, above, quaternion, surroundings, tolerance=0.005)
        for attempt in range(2):
            if attempt:
                error_above = self.body.move_hand(
                    side, above, quaternion, speed=0.25, tolerance=0.005, settle_timeout=3.0
                )
            error_grasp = self.body.move_hand(side, grasp, quaternion, speed=0.10, tolerance=0.008)
            self.body.grip(side, 1.0, seconds=1.0)
            forces = self.body.pad_forces(side)
            gap = self.body.finger_gap(side)
            if min(forces) >= 0.05 and gap >= 0.012:
                break
            # Missed or bumped it: open, back off above, look again, and re-centre.
            self.body.grip(side, 0.0, seconds=0.5)
            self.body.move_hand(side, above + [0, 0, 0.03], quaternion, speed=0.2, tolerance=0.02)
            if attempt == 0:
                self.note(
                    f"The grasp missed (fingers {forces[0]:.1f}/{forces[1]:.1f} N); looking again and re-centring"
                )
                geometry, _ = self.reacquire(geometry)
                grasp = np.r_[geometry["center_world_m"][:2], grasp_height(geometry)]
                above = grasp + [0, 0, max(0.05, geometry["top_z_m"] - grasp[2] + PREGRASP_CLEARANCE_M)]
        else:
            raise SkillError(
                "grasp",
                f"The fingers closed on nothing twice (pad forces {forces[0]:.2f}/{forces[1]:.2f} N, "
                f"gap {gap * 1000:.0f} mm)",
                "Look again from the head camera; the object may have moved or be too wide",
                hand_error_above_m=error_above,
                hand_error_grasp_m=error_grasp,
                arrival=arrival,
            )
        lift = grasp + [0, 0, 0.10]
        self.body.move_hand(side, lift, quaternion, speed=0.15, tolerance=0.02)
        forces_lifted = self.body.pad_forces(side)
        if min(forces_lifted) < 0.03:
            self.body.grip(side, 0.0, seconds=0.5)
            raise SkillError(
                "lift",
                "The object slipped out while lifting",
                "Try again; the object may need re-centring between the fingers",
                forces_before=forces,
                forces_after=forces_lifted,
            )
        memory = memory_for(self.console)
        memory_id = memory.remember(
            geometry, label, geometry.get("frame_id"), getattr(self.console, "current_request", "")
        )
        memory.mark(memory_id, "lift_observed")
        self.console.held[side] = {
            "label": label,
            "geometry": geometry,
            "hand_above_bottom_m": float(grasp[2] - geometry["bottom_z_m"]),
            "memory_id": memory_id,
        }
        self.carry(side, scene)
        return {
            "ok": True,
            "summary": f"Picked up {label} with the {side} hand; both fingers feel it "
            f"({forces_lifted[0]:.1f} N / {forces_lifted[1]:.1f} N)",
            "side": side,
            "memory_id": memory_id,
            "object_size_m": (2 * np.asarray(geometry["half_size_m"])).round(3).tolist(),
            "walked_m": arrival.get("walked_m", 0.0),
        }

    def carry(self, side, scene=None):
        """Bring the object in to the chest while standing up, both at once."""
        held = self.console.held[side]
        # Keep the object's bottom high enough to pass over tabletops and what
        # stands on them, while the hand stays in a comfortable chest-high pose.
        up = max(0.95, 0.90 + held["hand_above_bottom_m"]) - standing().pelvis_z
        configuration = self.body_pose_configuration(side, *CARRY_POINT, up)
        for _ in range(2):
            if self.swing_arm(side, configuration, standing(), scene, self.held_shape(side)):
                return
            hand, _ = self.body.hand(side)
            self.body.move_hand(side, hand + [0, 0, 0.10], speed=0.15, tolerance=0.02)
        # Still too close to the table to bring the arm in: stand up as it is,
        # step back into the open, then bring the object to the chest there.
        self.note("Standing up and stepping back from the table before bringing the object in")
        self.body.set_posture(STAND_HEIGHT, 0.0)
        self.step_back_if_tight(scene, side, force=True)
        if not self.swing_arm(side, configuration, None, scene, self.held_shape(side)):
            self.note("Keeping the object where it is in my hand; bringing it in would brush something")

    def place(self, relation, frame_id=None, reference_box=None, point=None, memory_id=None, side="auto", label=None):
        if side not in SIDES:
            holding = [s for s in SIDES if s in self.console.held]
            if not holding:
                raise SkillError("place", "Not holding anything", "Pick something up first")
            side = holding[0]
        if side not in self.console.held:
            raise SkillError("place", f"The {side} hand is not holding anything")
        held = self.console.held[side]
        half = np.asarray(held["geometry"]["half_size_m"])
        sensor = self.frame(frame_id) if frame_id is not None else self.look("head")
        views = [sensor, self.look("overview"), self.look("head")]
        destination, support_z, describe = self._destination(relation, views, reference_box, point, memory_id, half, label)
        scene = self.scene(*views)
        hand_now, _ = self.body.hand(side)
        scene = scene.without_sphere(hand_now, np.linalg.norm(half) + 0.08)
        # A frame from before the pick still shows the object where it was.
        before = held["geometry"]
        scene = scene.without_box(before["center_world_m"], before["half_size_m"], before.get("yaw_rad", 0.0), margin=0.02)
        release_z = support_z + held["hand_above_bottom_m"] + 0.008
        hover = np.r_[destination, release_z + 0.08]
        release = np.r_[destination, release_z]

        def targets(stance):
            quaternion = hand_quaternion(natural_hand_yaw(stance, side, destination))
            return [(hover, quaternion), (release, quaternion)]

        navigator = Navigator(scene, feet_pose(self.robot))
        try:
            plan = plan_reach(
                self.robot,
                scene,
                side,
                np.r_[destination, support_z],
                targets,
                feet_pose(self.robot),
                navigator,
                held=self.held_shape(side),
            )
        except PlanningError as exc:
            raise SkillError(
                "stance",
                f"Cannot reach {describe}: {exc}",
                "Choose a spot nearer an edge of the surface, or another surface",
                **exc.details,
            ) from exc
        self.note(
            f"Placing {held['label']} {describe}; standing {plan['reach_m']:.2f} m away"
            + (f", crouching to {plan['posture'].height:.2f} m" if plan["posture"].height < STAND_HEIGHT - 0.01 else "")
            + (f", leaning {plan['posture'].pitch:.1f} rad" if plan["posture"].pitch > 0 else ""),
            stance=plan["stance"].tolist(),
        )
        for hand in self.free_sides():
            self.ready_hand(hand, scene)
        for attempt in range(2):
            arrival = self.walk_to_stance(plan["stance"], navigator, carrying_side=side)
            if relation in ("on_top", "next_to") and reference_box is not None:
                refreshed = self._refresh_destination(relation, half)
                if refreshed is not None:
                    destination, support_z = refreshed
                    release_z = support_z + held["hand_above_bottom_m"] + 0.008
                    hover = np.r_[destination, release_z + 0.08]
                    release = np.r_[destination, release_z]

            def place_targets(stance, hover=hover, release=release, destination=destination):
                quaternion = hand_quaternion(natural_hand_yaw(stance, side, destination))
                return [(hover, quaternion), (release, quaternion)]

            try:
                choices, stance = self.settle_reach(side, place_targets, plan, scene, self.held_shape(side))
                break
            except SkillError as exc:
                if exc.stage != "reach" or attempt == 1:
                    raise
                plan, navigator = self._another_stance(
                    side, scene, plan, np.r_[destination, support_z], place_targets, self.held_shape(side)
                )
        quaternion = place_targets(stance)[0][1]
        # Bend and bring the object over the spot in one coordinated motion.
        hover_error = self.reach_any(choices, side, hover, quaternion, scene, held=self.held_shape(side))
        if hover_error > 0.02:
            raise SkillError(
                "place",
                f"Could not hold {held['label']} steadily over the spot ({hover_error * 100:.1f} cm off); "
                "still holding it",
                "Try again, or choose a spot nearer the edge of the surface",
                hover_error_m=round(hover_error, 4),
            )
        contact = self._guarded_descent(side, release, quaternion)
        contact["hover_error_m"] = round(hover_error, 4)
        contact["target_xy_m"] = np.round(destination, 4).tolist()
        contact["support_z_m"] = round(float(support_z), 4)
        contact["hand_xy_m"] = np.round(self.body.hand(side)[0][:2], 4).tolist()
        self.robot.grip(side, 0.35)  # loosen first so the object settles, then open
        self.body.wait(0.4)
        self.body.grip(side, 0.0, seconds=0.6)
        # Lift straight up until the fingers clear the object's top before moving away.
        height = 2 * half[2]
        clear = max(0.08, height - held["hand_above_bottom_m"] + 0.06)
        hand, _ = self.body.hand(side)
        self.body.move_hand(side, hand + [0, 0, clear], quaternion, speed=0.15, tolerance=0.02)
        self.console.held.pop(side, None)
        memory = memory_for(self.console)
        if held.get("memory_id"):
            memory.mark(held["memory_id"], "release_executed")
        bottom = contact["contact_height_m"] - held["hand_above_bottom_m"]
        placed = object_spheres(np.r_[hand[:2], bottom + half[2]], half, held["geometry"].get("shape") == "round")
        self.ready_hand(side, scene, standing(), extra=placed)
        check = self._verify_placement(held, destination, support_z)
        result = {
            "ok": True,
            "summary": f"Placed {held['label']} {describe}; contact felt at {contact['contact_height_m']:.3f} m"
            + f"; I can see it {check.get('offset_m', 0) * 100:.1f} cm from the target spot",
            "contact": contact,
            "placement_check": check,
            "walked_m": arrival.get("walked_m", 0.0),
        }
        if not check["seen_by"]:
            result.update(
                ok=False,
                stage="verify",
                summary=f"Let go of {held['label']} {describe}, but I cannot see it there now; "
                "it may have fallen or be hidden from this view",
                suggestion="Look with the overview camera to find it",
            )
        elif not check["upright"]:
            result.update(
                ok=False,
                stage="verify",
                summary=f"Let go of {held['label']} {describe}, but it is not standing as expected "
                f"(top at {check['top_m']:.2f} m); it may have tipped over",
                suggestion="Look again before deciding what to do",
            )
        return result

    def _destination(self, relation, views, reference_box, point, memory_id, half, label):
        sensor = views[0]  # the frame the boxes and points were drawn on
        if relation in ("on_top", "next_to"):
            if reference_box is None:
                raise SkillError("place", f"{relation} needs reference_box around the reference object")
            try:
                reference = object_geometry(sensor, reference_box, ignore=self.held_spheres())
            except ValueError as exc:
                raise SkillError("perception", str(exc), "Select the reference object again in a clear view") from exc
            self.last_reference = reference
            if relation == "on_top":
                return np.asarray(reference["top_center_m"][:2]), reference["top_z_m"], "on top of the reference object"
            spot = self._free_spot_beside(reference, half, views)
            return spot, reference["support_z_m"], "next to the reference object"
        if relation == "at_point":
            if point is None:
                raise SkillError("place", "at_point needs a [y, x] point on a surface")
            try:
                surface = surface_at(sensor, point)
            except ValueError as exc:
                raise SkillError("perception", str(exc), "Pick a point on a visible flat surface") from exc
            self.last_reference = None
            return surface[:2], float(surface[2]), "at the chosen spot"
        if relation == "back_where_it_was":
            memory = memory_for(self.console)
            try:
                row = memory.get(memory_id) if memory_id else memory.moves[-1]
            except (ValueError, IndexError) as exc:
                raise SkillError("place", "No remembered earlier location for that object") from exc
            before = row["before"]
            self.last_reference = None
            return np.asarray(before["center_world_m"][:2]), float(before["bottom_z_m"]), "back where it was"
        raise SkillError("place", f"Unknown placement relation {relation}")

    def _free_spot_beside(self, reference, half, views):
        """A spot on the reference's own surface beside it, judged from every view:
        the robot's body or the held object can hide part of the table in one."""
        points = np.vstack([point_cloud(view, stride=2)[0] for view in views])
        for centre, radius in self.held_spheres():
            points = points[np.linalg.norm(points - centre, axis=1) > radius]
        ref_center = np.asarray(reference["center_world_m"][:2])
        ref_half = np.asarray(reference["half_size_m"][:2])
        robot_xy = pelvis_pose2d(self.robot)[:2]
        toward_robot = np.arctan2(*(robot_xy - ref_center)[::-1])
        best = None
        for turn in np.radians([0, 45, -45, 90, -90, 135, -135, 180]):
            angle = toward_robot + turn
            direction = np.array([np.cos(angle), np.sin(angle)])
            reach = np.max(np.abs(rotation_z(reference.get("yaw_rad", 0.0))[:2, :2].T @ direction) * ref_half) + np.max(half[:2]) + NEXT_TO_GAP_M
            spot = ref_center + direction * reach
            # The whole footprint must rest on the surface, not hang over its edge.
            coverage, blocking = surface_free(points, spot, half[:2] + 0.005, 0.0, reference["support_z_m"])
            if coverage >= 0.85 and blocking == 0:
                return spot
            if best is None and coverage >= 0.7 and blocking <= 2:
                best = spot
        if best is not None:
            return best
        raise SkillError(
            "place",
            "No clear spot on the same surface right next to the reference object",
            "Choose a spot with at_point, or another surface",
        )

    def _refresh_destination(self, relation, half):
        """Re-measure the reference up close from the head camera at the stance.

        The close, top-down view is better than the first one (which may have
        been far and oblique), so it replaces it when it finds the same-coloured
        object nearby.
        """
        reference = getattr(self, "last_reference", None)
        if reference is None:
            return None
        fresh, sensor = self.measure_near(reference, "head", radius=0.03, chroma=reference.get("chroma"))
        if fresh is None or np.linalg.norm(np.subtract(fresh["center_world_m"][:2], reference["center_world_m"][:2])) > 0.10:
            return None
        if relation == "on_top":
            return np.asarray(fresh["top_center_m"][:2]), fresh["top_z_m"]
        try:
            return self._free_spot_beside(fresh, half, [sensor, self.look("overview")]), fresh["support_z_m"]
        except SkillError:
            return None

    def _guarded_descent(self, side, release, quaternion, speed=0.035):
        """Lower until the support carries the object, then stop (no gate to pass).

        Contact is felt two ways, like a person setting something down: the
        fingers stop carrying the object's weight, or the hand stops moving down.
        """
        robot = self.robot
        start, _ = self.body.hand(side)
        self.body.wait(0.2)
        baseline = self.body.pad_vertical_load(side)
        bottom_z = release[2] - 0.06  # search past the expected touch: the grip may have slipped
        z = start[2]
        dt = robot.dt
        felt, how = False, None
        stall = unloaded = crouches = 0
        elapsed, previous, velocity = 0.0, start[2], 0.0
        # Keep the arm's integral correction on (the held object always touches
        # the pads), but never let it push the object down into the surface.
        robot.suppress_downward_bias = True
        robot.feedback_gain = 1.5
        try:
            for _ in range(round((start[2] - bottom_z) / speed / dt) + 50):
                z = max(bottom_z, z - speed * dt)
                robot.reach(side, np.r_[release[:2], z], quaternion)
                self.body.step()
                elapsed += dt
                actual, _ = self.body.hand(side)
                # Downward hand speed, low-pass filtered against servo jitter.
                velocity += 0.3 * ((previous - actual[2]) / dt - velocity)
                previous = actual[2]
                # The servo lags a moving target by ~1 cm; a real stop is a hand
                # well behind its target that has nearly stopped descending.
                blocked = elapsed > 0.4 and actual[2] - z > 0.02 and velocity < 0.25 * speed
                stall = stall + 1 if blocked else 0
                load = self.body.pad_vertical_load(side)
                unloaded = unloaded + 1 if baseline > 0.25 and load < 0.4 * baseline else 0
                if unloaded * dt >= 0.06:
                    felt, how = True, "weight moved from the fingers to the surface"
                    break
                if stall * dt >= 0.15:
                    if baseline > 0.25 and load > 0.7 * baseline:
                        # Still carrying the weight: the arm ran out of reach, not
                        # into the surface. Crouch a little and keep lowering.
                        if crouches >= 2 or robot.height_goal - 0.04 < 0.50:
                            raise SkillError(
                                "place",
                                "The arm cannot lower the object any further from here; it is still held",
                                "Place it on a higher surface or nearer the edge",
                                hand_height_m=round(float(actual[2]), 3),
                            )
                        crouches += 1
                        robot.reach(side, np.r_[release[:2], actual[2]], quaternion)
                        self.body.set_posture(
                            robot.height_goal - 0.04, getattr(robot, "pitch_goal", 0.0), hold_hands_in_place=True
                        )
                        z = self.body.hand(side)[0][2]
                        stall = 0
                        continue
                    felt, how = True, "hand stopped moving down"
                    break
                if z <= bottom_z and abs(actual[2] - bottom_z) < 0.005:
                    break
            if not felt:
                actual, _ = self.body.hand(side)
                robot.reach(side, actual + [0, 0, 0.02], quaternion)
                self.body.wait(0.3)
                raise SkillError(
                    "place",
                    "Lowered past where the surface should be without feeling it; still holding the object",
                    "Look again: the surface may be lower, farther away or not where it seemed",
                    lowest_hand_height_m=round(float(actual[2]), 3),
                )
            # Seat it: press a few millimetres more so it stands flat before letting go.
            actual, _ = self.body.hand(side)
            for _ in range(round(0.25 / dt)):
                robot.reach(side, np.r_[release[:2], actual[2] - 0.006], quaternion)
                self.body.step()
        finally:
            robot.suppress_downward_bias = False
            robot.feedback_gain = 0.5
        actual, _ = self.body.hand(side)
        robot.reach(side, actual, quaternion)
        self.body.wait(0.15)
        return {
            "felt": felt,
            "how": how,
            "contact_height_m": float(actual[2]),
            "planned_release_height_m": float(release[2]),
            "finger_load_before_n": round(float(baseline), 3),
        }

    def _verify_placement(self, held, destination, support_z):
        """Look (head camera, then overview) for the object where it was let go.

        Success needs the object seen at the spot, with its top where an upright
        object resting on the support would have it.
        """
        expected = dict(held["geometry"])
        height = 2 * float(expected["half_size_m"][2])
        expected["center_world_m"] = [destination[0], destination[1], support_z + height / 2]
        for camera in ("head", "overview"):
            fresh, _ = self.measure_near(expected, camera, radius=0.03, chroma=held["geometry"]["chroma"])
            if fresh is None:
                continue
            offset = float(np.linalg.norm(np.subtract(fresh["center_world_m"][:2], destination)))
            if offset > 0.06:
                continue
            return {
                "seen_by": camera,
                "measured_center_m": np.round(fresh["center_world_m"], 3).tolist(),
                "offset_m": round(offset, 4),
                "top_m": round(float(fresh["top_z_m"]), 3),
                "upright": bool(abs(fresh["top_z_m"] - (support_z + height)) < 0.02),
            }
        return {"seen_by": None}

    def release(self, side="auto"):
        sides = [side] if side in SIDES else [s for s in SIDES if s in self.console.held] or list(SIDES)
        notes = []
        for hand in sides:
            position, _ = self.body.hand(hand)
            held = self.console.held.pop(hand, None)
            self.robot.grip(hand, 0.0)
            notes.append(
                f"opened the {hand} gripper"
                + (f" and let go of {held['label']} from {position[2]:.2f} m" if held else "")
            )
        self.body.wait(0.8)
        return {"ok": True, "summary": "Released: " + "; ".join(notes)}

    def grip(self, side, closure):
        if side not in SIDES:
            raise SkillError("grip", "side must be left or right")
        if closure < 0.5 and side in self.console.held:
            return self.release(side)
        measured = self.body.grip(side, float(np.clip(closure, 0, 1)))
        forces = self.body.pad_forces(side)
        return {
            "ok": True,
            "summary": f"{side} gripper at {measured:.2f} closure; pads {forces[0]:.1f}/{forces[1]:.1f} N",
        }

    def go_to(self, frame_id, box=None, point=None, distance=0.6):
        sensor = self.frame(frame_id)
        if box is not None:
            target = np.asarray(object_geometry(sensor, box)["center_world_m"])
        elif point is not None:
            target = surface_at(sensor, point)
        else:
            raise SkillError("go_to", "Give an object box or a [y, x] point to walk to")
        scene = self.scene(sensor, self.look("overview"), self.look("head"))
        feet = feet_pose(self.robot)
        navigator = Navigator(scene, feet)
        bearing = np.arctan2(target[1] - feet[1], target[0] - feet[0])
        stand = standing()
        for turn in np.radians([0, 20, -20, 40, -40, 60, -60, 90, -90, 120, -120, 180]):
            heading = wrap(bearing + turn)
            stance = np.r_[target[:2] - distance * np.array([np.cos(heading), np.sin(heading)]), heading]
            if body_clearance(scene, stand, stance) < BODY_MARGIN_M or navigator.entry_for(stance) is None:
                continue
            arrival = self.walk_to_stance(stance, navigator)
            return {
                "ok": True,
                "summary": f"Walked {arrival.get('walked_m', 0):.1f} m and now face the target "
                f"{distance:.2f} m away",
            }
        raise SkillError("route", "No walkable way to stand in front of that spot", "Choose another spot")

    def turn(self, degrees):
        yaw = pelvis_pose2d(self.robot)[2] + np.radians(degrees)
        error = self.body.turn_to(wrap(yaw))
        return {"ok": True, "summary": f"Turned {degrees:.0f}° (heading error {np.degrees(error):.1f}°)"}

    def walk(self, forward_m, left_m=0.0):
        pose = pelvis_pose2d(self.robot)
        if abs(forward_m) > 3 or abs(left_m) > 1:
            raise SkillError("walk", "Walk at most 3 m forward/back and 1 m sideways per step")
        goal = pose[:2] + rotation_z(pose[2])[:2, :2] @ [forward_m, left_m]
        holding = next(iter(self.console.held), None)
        arrival = self.body.walk_path([goal], final_yaw=pose[2], carrying_side=holding)
        return {"ok": True, "summary": f"Walked to within {arrival['position_error_m'] * 100:.0f} cm of the goal"}

    def set_posture(self, height_m, torso_pitch_rad=0.0):
        self.body.set_posture(float(height_m), float(torso_pitch_rad))
        z = self.robot.tools.pelvis_pose()[0][2]
        return {"ok": True, "summary": f"Pelvis at {z:.2f} m, torso pitch command {torso_pitch_rad:.2f} rad"}

    def move_hand(self, side, forward_m, left_m, up_m):
        if side not in SIDES:
            raise SkillError("move_hand", "side must be left or right")
        if max(abs(forward_m), abs(left_m), abs(up_m)) > 0.15:
            raise SkillError("move_hand", "Nudge the hand at most 15 cm per axis at a time")
        pose = pelvis_pose2d(self.robot)
        hand, _ = self.body.hand(side)
        goal = hand + np.r_[rotation_z(pose[2])[:2, :2] @ [forward_m, left_m], up_m]
        error = self.body.move_hand(side, goal, speed=0.15, tolerance=0.01)
        return {"ok": True, "summary": f"Moved the {side} hand; {error * 100:.1f} cm from the requested spot"}

    def _preferred_sides(self, target, free):
        pose = pelvis_pose2d(self.robot)
        local = rotation_z(pose[2])[:2, :2].T @ (np.asarray(target[:2]) - pose[:2])
        order = ["left", "right"] if local[1] > 0 else ["right", "left"]
        return [s for s in order if s in free]


def forget_dropped(console):
    """Holding is what the fingers feel: forget objects no pad touches any more."""
    body = Body(console, console.epoch)
    for side in list(getattr(console, "held", {})):
        if max(body.pad_forces(side)) < 0.05:
            console.held.pop(side)


def execute(console, name, args, epoch):
    """Run one agent tool on the physics thread; failures become readable results."""
    skills = Skills(console, epoch)
    try:
        if name == "pick_up":
            return skills.pick_up(args["frame_id"], args["object_box"], args.get("side", "auto"), args.get("object_name"))
        if name == "place":
            return skills.place(
                args["relation"],
                args.get("frame_id"),
                args.get("reference_box"),
                args.get("point"),
                args.get("memory_id"),
                args.get("side", "auto"),
            )
        if name == "move_object":
            picked = skills.pick_up(args["frame_id"], args["object_box"], args.get("side", "auto"), args.get("object_name"))
            # The pick changed the robot's pose: the reference is measured in the
            # same (earlier) frame, which is still valid world geometry.
            placed = skills.place(
                args["relation"],
                args["frame_id"],
                args.get("reference_box"),
                args.get("point"),
                args.get("memory_id"),
                picked["side"],
            )
            result = {
                "ok": placed.get("ok") is True,
                "summary": picked["summary"] + ". " + placed["summary"],
                "pick": picked,
                "place": placed,
            }
            if not result["ok"]:
                result["stage"] = placed.get("stage", "place")
                if "suggestion" in placed:
                    result["suggestion"] = placed["suggestion"]
            return result
        if name == "release":
            return skills.release(args.get("side", "auto"))
        if name == "grip":
            return skills.grip(args["side"], args["closure"])
        if name == "go_to":
            return skills.go_to(args["frame_id"], args.get("box"), args.get("point"), args.get("distance_m", 0.6))
        if name == "turn":
            return skills.turn(args["degrees"])
        if name == "walk":
            return skills.walk(args["forward_m"], args.get("left_m", 0.0))
        if name == "set_posture":
            return skills.set_posture(args["height_m"], args.get("torso_pitch_rad", 0.0))
        if name == "move_hand":
            return skills.move_hand(args["side"], args.get("forward_m", 0.0), args.get("left_m", 0.0), args.get("up_m", 0.0))
        raise SkillError("tool", f"Unknown action {name}")
    except SkillError as exc:
        return {
            "ok": False,
            "stage": exc.stage,
            "summary": f"Could not {name.replace('_', ' ')}: {exc}",
            "reason": str(exc),
            "suggestion": exc.suggestion,
            "details": exc.details,
        }
    except (PlanningError, MotionError) as exc:
        return {
            "ok": False,
            "stage": exc.stage,
            "summary": f"Could not {name.replace('_', ' ')}: {exc}",
            "reason": str(exc),
            "details": exc.details,
        }
    except RuntimeError as exc:
        if epoch != console.epoch:
            raise  # cancelled for a newer instruction: let the server stop it
        # e.g. the contact monitor stopped a step: report it, never crash the skill
        console.robot.motion_goal[:] = 0
        return {"ok": False, "stage": "motion", "summary": f"Could not {name.replace('_', ' ')}: {exc}", "reason": str(exc)}
    finally:
        skills.robot.contact_mode = False
        forget_dropped(console)
