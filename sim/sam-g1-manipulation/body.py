"""Whole-body motion for the G1 over GR00T-WBC: posture, gait and hand motion.

Only the robot's own proprioception and the measured posture table are used.
Walking turns toward where it is going and walks forward; it side-steps only
for the last few centimetres. Hand motions follow smooth minimum-jerk paths.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np

POSTURE_TABLE = Path(__file__).with_name("posture_table.json")
GAIT_ENVELOPE = Path(__file__).with_name("gait_envelope.json")
STAND_HEIGHT = 0.74
# Gait measurements (arena, 4 s commands): forward commands below ~0.07 m/s
# make GR00T-WBC stand still; it tracks ~85% of larger forward commands and
# after a stop travels on ~0.6-0.75 s x the commanded speed. Side-stepping
# tracks only ~50%, so it is reserved for docking.
MIN_FORWARD = 0.09
MIN_SIDEWAYS = 0.12
MAX_FORWARD = 0.30
MAX_TURN = 0.5
BRAKE_SECONDS = 0.75
TURN_IN_PLACE_RAD = 0.35  # larger heading changes stop and turn on the spot
APPROACH_SPEED = 0.12  # m/s over the last APPROACH_DISTANCE_M to a stance
APPROACH_DISTANCE_M = 0.5
FINAL_DOCK_M = 0.25  # the last stretch into a stance is stepped, not walked
# Walking stops when the stepping envelope gets this close to anything seen,
# and steers sideways away from it below STEER_MARGIN_M.
GUARD_MARGIN_M = 0.015
STEER_MARGIN_M = 0.05


class MotionError(RuntimeError):
    """A motion could not be completed; the message says what was measured."""

    def __init__(self, stage, reason, **details):
        super().__init__(reason)
        self.stage = stage
        self.details = details


def wrap(angle):
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def rotation_z(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def yaw_quaternion(yaw):
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def quaternion_multiply(a, b):
    result = np.empty(4)
    mujoco.mju_mulQuat(result, np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    return result


def pelvis_pose2d(robot):
    position, q = robot.tools.pelvis_pose()
    yaw = np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
    return np.array([position[0], position[1], yaw], dtype=float)


def feet_pose(robot):
    """Midpoint and heading of the planted feet: the stance reference."""
    model, data = robot.tools.model, robot.tools.data
    namespace = robot.tools.robot.namespace
    feet = [model.body(namespace + f"{s}_ankle_roll_link").id for s in ("left", "right")]
    midpoint = np.mean(data.xpos[feet], axis=0)
    forward = np.mean(data.xmat[feet].reshape(2, 3, 3)[:, :, 0], axis=0)
    return np.array([midpoint[0], midpoint[1], np.arctan2(forward[1], forward[0])])


@dataclass(frozen=True)
class Posture:
    """One measured GR00T-WBC posture, expressed relative to the planted feet."""

    height: float
    pitch: float
    pelvis_offset: tuple
    pelvis_z: float
    pelvis_quaternion: tuple
    balance_joints: dict
    surface: np.ndarray

    @property
    def cost(self):
        # People prefer standing; a lean is cheaper than a deep crouch.
        return 2.0 * abs(self.height - STAND_HEIGHT) + 0.4 * self.pitch

    def pelvis_pose(self, stance):
        """World pelvis pose (xyz + wxyz) for IK planning at stance (x, y, yaw)."""
        x, y, yaw = stance
        offset = rotation_z(yaw)[:2, :2] @ np.asarray(self.pelvis_offset)
        quaternion = quaternion_multiply(yaw_quaternion(yaw), self.pelvis_quaternion)
        return np.r_[x + offset[0], y + offset[1], self.pelvis_z, quaternion]

    def surface_world(self, stance):
        x, y, yaw = stance
        rotation = rotation_z(yaw)[:2, :2]
        xy = self.surface[:, :2] @ rotation.T + [x, y]
        return np.column_stack((xy, self.surface[:, 2]))


@lru_cache(maxsize=1)
def postures():
    table = json.loads(POSTURE_TABLE.read_text())
    return tuple(
        Posture(
            height=row["height_command"],
            pitch=row["pitch_command"],
            pelvis_offset=tuple(row["pelvis_offset_feet_m"]),
            pelvis_z=row["pelvis_z_m"],
            pelvis_quaternion=tuple(row["pelvis_quaternion_feet_wxyz"]),
            balance_joints=row["balance_joints_rad"],
            surface=np.asarray(row["body_surface_feet_m"], dtype=float),
        )
        for row in table["entries"]
        if row.get("settled", True)
    )


def posture_for(height, pitch=0.0):
    return min(postures(), key=lambda p: abs(p.height - height) + abs(p.pitch - pitch))


def standing():
    return posture_for(STAND_HEIGHT, 0.0)


@lru_cache(maxsize=1)
def gait():
    """The space legs and torso sweep while stepping (gait_envelope.json).

    A stepping hip and thigh reach ~6 cm further forward than when standing;
    every walking, turning and shuffling check uses this envelope.
    """
    surface = np.asarray(json.loads(GAIT_ENVELOPE.read_text())["surface_feet_m"], dtype=float)
    return replace(standing(), surface=surface)


# Share of the leading motion done before the following one starts.
ARM_LEAD = 0.6


def posture_order(height_now, pitch_now, height, pitch):
    """Reaching down: arm first, then bend. Straightening up: body first."""
    return "arm_first" if height < height_now - 0.01 or pitch > pitch_now + 0.05 else "body_first"


def motion_schedule(order, fraction, lead_span=0.6):
    """(arm progress, body progress) at overall progress `fraction` (0..1).

    The leading motion fills the first `lead_span` of the time; the following
    one starts once the leader is ARM_LEAD of its way through.
    """
    if order is None:
        return minimum_jerk(fraction), minimum_jerk(fraction)
    lead = minimum_jerk(fraction / lead_span)
    follow = minimum_jerk((fraction - ARM_LEAD * lead_span) / (1 - ARM_LEAD * lead_span))
    return (lead, follow) if order == "arm_first" else (follow, lead)


def minimum_jerk(fraction):
    s = np.clip(fraction, 0.0, 1.0)
    return s**3 * (10 - 15 * s + 6 * s**2)


class Body:
    """Feedback motion through the console tick, so the viewer and safety stay live."""

    def __init__(self, console, epoch, report=None):
        self.console = console
        self.epoch = epoch
        self.report = report or (lambda text, **details: None)

    @property
    def robot(self):
        return self.console.robot

    def step(self, count=1):
        for _ in range(count):
            try:
                self.console.tick(self.epoch)
            except ValueError as exc:
                # The arm IK could not keep its target without self-collision.
                # Stop the arms where they are rather than forcing the motion.
                self.freeze_arms()
                raise MotionError("arm", f"Arm motion stopped to avoid hitting the body: {exc}") from exc

    def freeze_arms(self):
        """Hold both arms at their current joint angles, moving with the body."""
        robot = self.robot
        robot.targets.clear()
        robot.orientations.clear()
        robot.feedback_offsets.clear()
        goal = robot.raw[robot.start :].copy()
        goal[:14] -= robot.gravity_offsets
        robot.goal[:14] = goal[:14]
        robot.sync_arm_seed()

    def move_arm(self, sides, configuration, height=None, pitch=None, speed=0.9, timeout=10.0, order=None):
        """Swing arms in joint space to a planned configuration (14 angles).

        Differential IK follows a Cartesian path from wherever the arm is and
        can settle in a contorted local solution far from the plan. Large moves
        therefore go straight to the configuration the planner solved, as
        MoveIt's transfer moves do; only short approaches stay Cartesian.

        With a height/pitch the body changes posture in the same motion, in the
        order a person uses: reaching down, the arm goes out first and the body
        bends once the arm is most of the way there; straightening up, the body
        rises first and the arm follows. The arm lags its command by a few
        centimetres while the balance controller bends the torso promptly, so
        doing both at once would dip the hand below its path. Arms not in
        `sides` keep their angles and move with the body.
        """
        robot = self.robot
        sides = (sides,) if isinstance(sides, str) else tuple(sides)
        moving = np.zeros(14, dtype=bool)
        for side in sides:
            moving[slice(0, 7) if side == "left" else slice(7, 14)] = True
        for side in ("left", "right") if height is not None else sides:
            # Arms ride with the body during a posture change, never holding a world point.
            robot.targets.pop(side, None)
            robot.orientations.pop(side, None)
            robot.feedback_offsets.pop(side, None)
        start = robot.goal[:14].copy()
        goal = np.where(moving, np.asarray(configuration, dtype=float), start)
        steps = max(1, round(max(0.5, 1.875 * float(np.max(np.abs(goal - start))) / speed) / robot.dt))
        order = None
        if height is not None:
            pitch = (robot.pitch_goal if pitch is None else pitch) if robot.pitch_slot is not None else 0.0
            order = order or posture_order(robot.height_goal, robot.pitch_goal, height, pitch)
            progress = self._posture_progress(height, pitch)
            if order == "body_first":
                robot.posture(height, pitch if robot.pitch_slot is not None else None)
                for _ in range(round(timeout / robot.dt)):
                    if progress() >= ARM_LEAD:
                        break
                    self.step()
        for tick in range(1, steps + 1):
            robot.goal[:14] = start + (goal - start) * minimum_jerk(tick / steps)
            if order == "arm_first" and tick == max(1, round(ARM_LEAD * steps)):
                robot.posture(height, pitch if robot.pitch_slot is not None else None)
            self.step()
        robot.goal[:14] = goal
        robot.sync_arm_seed()
        if height is not None:
            robot.posture(height, pitch if robot.pitch_slot is not None else None)
            self._settle_posture(height, pitch, timeout=4.0)
        else:
            self.wait(0.25)

    def _posture_progress(self, height, pitch):
        """Fraction of a commanded posture change the body has been told to make."""
        robot = self.robot
        slots = [(robot.tools.base_slot["height"], height)]
        if robot.pitch_slot is not None:
            slots.append((robot.pitch_slot, pitch))
        starts = [float(robot.raw[slot]) for slot, _ in slots]

        def progress():
            parts = [
                (robot.raw[slot] - start) / (goal - start)
                for (slot, goal), start in zip(slots, starts)
                if abs(goal - start) > 1e-3
            ]
            return float(np.clip(min(parts, default=1.0), 0.0, 1.0))

        return progress

    def wait(self, seconds):
        self.step(max(1, round(seconds / self.robot.dt)))

    # ---- posture -------------------------------------------------------
    def current_posture(self):
        robot = self.robot
        pitch = robot.pitch_goal if robot.pitch_slot is not None else 0.0
        return posture_for(robot.height_goal, pitch)

    def set_posture(self, height, pitch=0.0, timeout=6.0, hold_hands_in_place=False):
        """Crouch/lean. Arms move with the body unless told to keep the hands still."""
        robot = self.robot
        if robot.pitch_slot is None:
            pitch = 0.0
        if (
            abs(robot.height_goal - height) < 1e-3
            and abs(getattr(robot, "pitch_goal", 0.0) - pitch) < 1e-3
            and abs(robot.raw[robot.tools.base_slot["height"]] - height) < 2e-3
        ):
            return
        if not hold_hands_in_place:
            self.freeze_arms()
        robot.posture(height, pitch if robot.pitch_slot is not None else None)
        self._settle_posture(height, pitch, timeout)

    def _settle_posture(self, height, pitch, timeout):
        robot = self.robot
        previous = np.asarray(robot.tools.pelvis_pose()[0])
        quiet = 0.0
        for step in range(round(timeout / robot.dt)):
            self.step()
            position = np.asarray(robot.tools.pelvis_pose()[0])
            speed = np.linalg.norm(position - previous) / robot.dt
            previous = position
            commanded = abs(robot.raw[robot.tools.base_slot["height"]] - height) < 2e-3
            if robot.pitch_slot is not None:
                commanded &= abs(robot.raw[robot.pitch_slot] - pitch) < 2e-3
            quiet = quiet + robot.dt if commanded and speed < 0.02 else 0.0
            if quiet >= 0.3:
                return
        self.report(
            "Posture still settling after the time limit; continuing",
            height=height,
            pitch=pitch,
        )

    # ---- gait ----------------------------------------------------------
    def _drive(self, vx, vy, wz):
        self.robot.velocity(
            float(np.clip(vx, -0.2, MAX_FORWARD)),
            float(np.clip(vy, -0.2, 0.2)),
            float(np.clip(wz, -MAX_TURN, MAX_TURN)),
        )

    def stop(self, settle=0.8):
        self.robot.motion_goal[:] = 0
        self.wait(settle)

    def come_to_rest(self, timeout=2.0):
        """Stop and let the gait settle; turning while still rolling drifts sideways."""
        robot = self.robot
        robot.motion_goal[:] = 0
        quiet = 0.0
        for _ in range(round(timeout / robot.dt)):
            lin, ang, _ = robot.lower_proxy.get_base_obs()
            quiet = quiet + robot.dt if np.linalg.norm(lin[:2]) < 0.04 and abs(ang[2]) < 0.1 else 0.0
            if quiet >= 0.2:
                return
            self.step()
            if robot.ticks % 5 == 0:
                self._check_guard()

    def turn_to(self, yaw, tolerance=0.06, timeout=15.0):
        """Turn on the spot from rest; a second, slower pass removes the coast-out error."""
        robot = self.robot
        for attempt in range(2):
            if abs(wrap(yaw - pelvis_pose2d(robot)[2])) < tolerance:
                break
            self.come_to_rest()
            for _ in range(round(timeout / robot.dt)):
                error = wrap(yaw - pelvis_pose2d(robot)[2])
                if abs(error) < tolerance:
                    break
                rate = np.sign(error) * max(0.18, min(MAX_TURN / (attempt + 1), 1.6 * abs(error)))
                self._drive(0, 0, rate)
                self.step()
                if self.robot.ticks % 5 == 0:
                    self._check_guard()
            self.stop(0.5)
        return wrap(yaw - pelvis_pose2d(robot)[2])

    def walk_path(self, waypoints, final_yaw=None, tolerance=0.04, carrying_side=None, guard=None, turn_first=False):
        """Turn toward each waypoint, walk forward with heading feedback, dock.

        Waypoints are pelvis xy positions in the world frame. With turn_first the
        robot turns on the spot to the final heading and shuffles straight into
        place instead (for short moves next to furniture). `guard(pose)` returns
        the body's clearance to what the cameras saw; motion stops before contact
        if it gets too close. The console's contact monitor stays active as a
        last resort.
        """
        robot = self.robot
        console = self.console
        self.freeze_arms()
        console.motion_kind = "walk"
        console.carrying_side = carrying_side
        self.guard = guard
        points = [np.asarray(w, dtype=float)[:2] for w in waypoints]
        try:
            try:
                if turn_first and final_yaw is not None:
                    # Walk near, turn to face the spot, then shuffle into place.
                    for index, goal in enumerate(points[:-1]):
                        following = points[index + 1] if index + 2 < len(points) else None
                        self._walk_to_point(goal, following, tolerance, carrying_side)
                    self.robot.motion_goal[:] = 0
                    self.turn_to(final_yaw)
                    if len(points) > 1 and np.linalg.norm(points[-2] - pelvis_pose2d(robot)[:2]) > 0.05:
                        # Braking and turning drifted off the planned entry; get back on it.
                        self._dock(points[-2], carrying_side)
                        self.turn_to(final_yaw)
                    self._dock(points[-1], carrying_side, timeout=8.0)
                else:
                    for index, goal in enumerate(points):
                        following = points[index + 1] if index + 1 < len(points) else None
                        self._walk_to_point(goal, following, tolerance, carrying_side)
                self.robot.motion_goal[:] = 0
                self._settle_walk(carrying_side)
                if final_yaw is not None:
                    self.turn_to(final_yaw)
                for _ in range(3):
                    pose = pelvis_pose2d(robot)
                    residual = points[-1] - pose[:2]
                    if np.linalg.norm(residual) <= tolerance * 1.5:
                        break
                    self._dock(points[-1], carrying_side)
                    if final_yaw is not None:
                        self.turn_to(final_yaw)
            except MotionError:
                # Right at the spot, furniture closer than planned: stop a little
                # short instead of giving up; the reach check decides from here.
                if guard is None or np.linalg.norm(points[-1] - pelvis_pose2d(robot)[:2]) > 0.15:
                    raise
                self._back_off(carrying_side)
                self.report("Stopped a few centimetres short of the spot to keep clear of the furniture")
        finally:
            robot.motion_goal[:] = 0
            console.motion_kind = None
            self.guard = None
        pose = pelvis_pose2d(robot)
        return {
            "arrived_xy_m": pose[:2].tolist(),
            "position_error_m": float(np.linalg.norm(pose[:2] - points[-1])),
            "heading_error_rad": float(wrap((final_yaw or pose[2]) - pose[2])),
        }

    def _back_off(self, carrying_side, distance=0.06):
        """Step a few centimetres away from whatever the guard saw."""
        feet = feet_pose(self.robot)
        best = None
        for angle in np.radians(np.arange(0, 360, 45)):
            direction = np.array([np.cos(angle), np.sin(angle)])
            clearance = self.guard(np.r_[feet[:2] + distance * direction, feet[2]])
            if best is None or clearance > best[0]:
                best = (clearance, direction)
        guard, self.guard = self.guard, None
        try:
            # A timed step: a docking move this short would stop before it starts.
            local = rotation_z(pelvis_pose2d(self.robot)[2])[:2, :2].T @ best[1]
            speed = 0.12
            # Side-steps track about half of what is commanded.
            self._drive(speed * local[0], 1.5 * speed * local[1], 0.0)
            self.wait(max(0.4, distance / (0.6 * speed)))
            self.come_to_rest()
        finally:
            self.guard = guard

    def _stopping_distance(self):
        """How far the body still travels after a stop command (measured: ~0.75 s x speed).

        Uses the smoothed commanded velocity; the measured base speed swings with
        every step and is no guide to momentum.
        """
        slots = self.robot.tools.base_slot
        commanded = np.hypot(self.robot.raw[slots["vx"]], self.robot.raw[slots["vy"]])
        return float(commanded * BRAKE_SECONDS)

    def _steer_clear(self):
        """Sideways velocity away from something seen close by while walking (or 0).

        A reactive correction on top of the planned route: when the stepping
        envelope comes within STEER_MARGIN_M of a seen point, step toward the
        side with more room instead of walking on into it.
        """
        guard = getattr(self, "guard", None)
        if guard is None:
            return 0.0
        feet = feet_pose(self.robot)
        here = guard(feet)
        if here >= STEER_MARGIN_M:
            return 0.0
        left = rotation_z(feet[2])[:2, :2] @ [0.0, 0.04]
        gain_left = guard(np.r_[feet[:2] + left, feet[2]]) - here
        gain_right = guard(np.r_[feet[:2] - left, feet[2]]) - here
        if max(gain_left, gain_right) <= 0.0:
            return 0.0
        return MIN_SIDEWAYS if gain_left > gain_right else -MIN_SIDEWAYS

    def _check_guard(self):
        guard = getattr(self, "guard", None)
        if guard is None:
            return
        clearance = guard(feet_pose(self.robot))
        if clearance < GUARD_MARGIN_M:
            self.robot.motion_goal[:] = 0
            self.wait(0.6)
            raise MotionError(
                "walking",
                f"Stopped walking {clearance * 100:.1f} cm from something the cameras saw",
                clearance_m=round(float(clearance), 4),
            )

    def _walk_to_point(self, goal, following, tolerance, carrying_side):
        robot = self.robot
        last = following is None
        pose = pelvis_pose2d(robot)
        distance = np.linalg.norm(goal - pose[:2])
        if distance < (tolerance if last else 0.12):
            return
        bearing = np.arctan2(goal[1] - pose[1], goal[0] - pose[0])
        if distance > 0.25 and abs(wrap(bearing - pose[2])) > TURN_IN_PLACE_RAD:
            self.turn_to(bearing, tolerance=0.12)
        elif distance <= 0.25:
            self._dock(goal, carrying_side)
            return
        # A corner that needs a turn on the spot needs a slow, nearly stopped arrival.
        sharp = False
        if not last:
            segment = np.arctan2(*(goal - pose[:2])[::-1])
            onward = np.arctan2(*(following - goal)[::-1])
            sharp = abs(wrap(onward - segment)) > TURN_IN_PLACE_RAD
        top = MAX_FORWARD if not carrying_side else 0.22
        limit = round((distance / 0.08 + 8.0) / robot.dt)
        sidestep = 0.0
        for tick in range(limit):
            pose = pelvis_pose2d(robot)
            offset = goal - pose[:2]
            distance = np.linalg.norm(offset)
            bearing = np.arctan2(offset[1], offset[0])
            error = wrap(bearing - pose[2])
            stopping = self._stopping_distance()
            if last and distance <= FINAL_DOCK_M + stopping:
                break  # the last stretch is stepped precisely (see below)
            if sharp and distance <= max(0.06, stopping):
                break  # slow arrival at a corner that needs a turn on the spot
            if not (last or sharp) and distance < 0.15:
                break  # passing through: the next segment takes over
            if abs(error) > 1.2:
                break  # passed the point; the next segment or dock corrects it
            speed = np.clip(0.7 * distance, MIN_FORWARD, top)
            if sharp:
                speed = min(speed, max(MIN_FORWARD, 0.5 * distance))
            if last:
                # Arrive slowly, as a person does near furniture: the gait coasts
                # ~0.8 s x speed after stopping, and less speed means less coast.
                speed = min(speed, APPROACH_SPEED + 0.5 * max(0.0, distance - APPROACH_DISTANCE_M))
            if tick % 5 == 0:
                self._check_guard()
                sidestep = self._steer_clear()
            self._drive(speed * max(0.3, np.cos(error)) * (0.7 if sidestep else 1.0), sidestep, 1.8 * error)
            self.step()
        if last:
            # Come to rest short of the spot, then step the last stretch: walking
            # on into it overshoots by the gait's coasting distance.
            self.come_to_rest()
            self._dock(goal, carrying_side, timeout=8.0)
        elif sharp:
            robot.motion_goal[:] = 0
            self.wait(0.5)

    def _dock(self, goal, carrying_side, timeout=6.0):
        """Short holonomic steps that stop early: the gait coasts after a stop."""
        robot = self.robot
        self.robot.motion_goal[:] = 0
        for _ in range(round(timeout / robot.dt)):
            pose = pelvis_pose2d(robot)
            local = rotation_z(pose[2])[:2, :2].T @ (goal - pose[:2])
            coasting = self._stopping_distance()
            if np.linalg.norm(local) < max(0.03, coasting):
                break
            vx = np.sign(local[0]) * max(MIN_FORWARD, min(0.12, 1.0 * abs(local[0])))
            vy = np.sign(local[1]) * max(MIN_SIDEWAYS, min(0.18, 1.6 * abs(local[1])))
            self._drive(
                vx if abs(local[0]) > 0.02 else 0.0,
                vy if abs(local[1]) > 0.02 else 0.0,
                0.0,
            )
            self.step()
            if self.robot.ticks % 5 == 0:
                self._check_guard()
        self.stop(0.6)

    def _settle_walk(self, carrying_side, seconds=1.5):
        robot = self.robot
        quiet = 0.0
        for _ in range(round(seconds / robot.dt)):
            self.step()
            lin, ang, _ = robot.lower_proxy.get_base_obs()
            quiet = quiet + robot.dt if np.linalg.norm(lin[:2]) < 0.04 and abs(ang[2]) < 0.08 else 0.0
            if quiet > 0.4:
                return

    # ---- hands ---------------------------------------------------------
    def hand(self, side):
        site = self.robot.tools.data.bind(self.robot.tools._hand(side).wrist_site)
        return np.asarray(site.xpos).copy(), np.asarray(site.xmat).reshape(3, 3).copy()

    def move_hand(
        self,
        side,
        position,
        quaternion=None,
        speed=0.25,
        tolerance=0.012,
        settle_timeout=2.0,
        contact_ok=False,
    ):
        """Minimum-jerk Cartesian motion of the pinch point between the pads."""
        robot = self.robot
        start, rotation = self.hand(side)
        goal = np.asarray(position, dtype=float)
        if quaternion is None:
            quaternion = np.empty(4)
            mujoco.mju_mat2Quat(quaternion, rotation.ravel())
        duration = max(0.4, np.linalg.norm(goal - start) / speed)
        steps = round(duration / robot.dt)
        if side not in robot.targets:
            robot.sync_arm_seed(side)  # taking over from a joint-space hold
        robot.contact_mode = contact_ok
        for index in range(1, steps + 1):
            point = start + (goal - start) * minimum_jerk(index / steps)
            robot.reach(side, point, quaternion)
            self.step()
        error = np.linalg.norm(self.hand(side)[0] - goal)
        # Settle with a stronger outer-loop correction of steady tracking bias.
        robot.feedback_gain = 1.5
        try:
            for _ in range(round(settle_timeout / robot.dt)):
                if error < tolerance:
                    break
                self.step()
                error = np.linalg.norm(self.hand(side)[0] - goal)
        finally:
            robot.feedback_gain = 0.5
            robot.contact_mode = False
        return float(error)

    def grip(self, side, closure, seconds=0.9):
        self.robot.grip(side, float(closure))
        self.wait(seconds)
        return self.robot.state()["gripper_closure"][side]

    def pad_forces(self, side):
        """Normal force on each Dex1 finger pad (N): the robot's touch sensors."""
        model, data = self.robot.tools.model, self.robot.tools.data
        forces = [0.0, 0.0]
        for index, contact in enumerate(data.contact):
            for geom in (contact.geom1, contact.geom2):
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom)) or ""
                for pad in (1, 2):
                    if f"{side}_dex1_pad_{pad}" in name:
                        wrench = np.zeros(6)
                        mujoco.mj_contactForce(model, data, index, wrench)
                        forces[pad - 1] += abs(float(wrench[0]))
        return forces

    def pad_vertical_load(self, side):
        """Upward force the finger pads exert on what they hold (N).

        While an object hangs in the grip this is about its weight; when the
        object rests on a surface it drops toward zero.
        """
        model, data = self.robot.tools.model, self.robot.tools.data
        total = 0.0
        for index, contact in enumerate(data.contact):
            for geom in (contact.geom1, contact.geom2):
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom)) or ""
                if f"{side}_dex1_pad_" not in name:
                    continue
                wrench = np.zeros(6)
                mujoco.mj_contactForce(model, data, index, wrench)
                world = np.asarray(contact.frame).reshape(3, 3).T @ wrench[:3]
                # mj_contactForce gives the force on geom2 from geom1; flip so the
                # result is the force the pad applies to the other body.
                total += world[2] if int(geom) == int(contact.geom1) else -world[2]
        return float(total)

    def finger_gap(self, side):
        """Distance between the two pads (m), measured from finger joints."""
        fingers = self.robot.state()["finger_positions_m"][side]
        # Pad centres are 89.1 mm apart open (q=0.0228) and 6.4 mm closed (q=-0.0186).
        return float(0.0064 + (np.mean(fingers) + 0.0186) * (0.0891 - 0.0064) / 0.0414)
