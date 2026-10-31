"""Goal assignment and velocity commands for several robots on their stock walking policies.

Pure numpy, so it runs and is tested on the Mac (test_fleet.py); multi_robot_scene.py --goal imports it.
Each robot's policy takes a body-frame command (vx, vy, wz). The fleet layer picks which robot goes to which
goal, then turns each robot toward its goal and walks it there. Robots yield by priority: a robot
stops while another robot is close in front of it and steps away from higher-priority ones (longest trip first).
"""
import itertools
import math

import numpy as np


def assign(starts, goals):
    """Goal index per robot, minimising the total straight-line distance.
    ponytail: brute force over permutations, fine to 8 robots; scipy.optimize.linear_sum_assignment beyond."""
    starts, goals = np.asarray(starts, float), np.asarray(goals, float)
    cost = np.linalg.norm(starts[:, None, :2] - goals[None, :, :2], axis=2)
    perms = itertools.permutations(range(len(goals)), len(starts))
    return list(min(perms, key=lambda p: cost[np.arange(len(starts)), list(p)].sum()))


def priority(starts, goals):
    """Rank per robot (0 = highest), longest trip first: robots parked early on a goal sit in the way of the
    ones still travelling, so the travellers keep right of way and the parked ones step aside."""
    trip = [math.dist(a[:2], b[:2]) for a, b in zip(starts, goals)]
    order = sorted(range(len(trip)), key=lambda i: -trip[i])
    return [order.index(i) for i in range(len(trip))]


def command(pose, goal, higher, vmax, lower=(), wmax=0.8, reach=0.3, clearance=1.2, vaway=0.4, holonomic=True):
    """Body-frame (vx, vy, wz) for a robot at pose (x, y, yaw) heading to goal (x, y).
    higher and lower: (x, y) of the other robots, by priority. No robot walks forward into another within
    `clearance` in front of it. While a higher-priority robot is that close, this one backs and sidesteps away
    from it (the walking policies take vy), with a bias to its right so two robots never mirror each other,
    even when parked at its goal. The top-priority robot never moves aside, so the fleet cannot all stall.
    holonomic=False (H1): no vy, no reversing. Trained command ranges in the 6.2 assets' env yaml files:
    H1 vx 0..1, vy 0, wz -1..1; Go2 and ANYmal -1..1 on all three; Spot vx -2..3, vy -1.5..1.5, wz -2..2.
    The defaults here (vmax 0.6, vaway 0.4, wmax 0.8) sit inside all of them."""
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    blocked, away = False, np.zeros(2)
    for k, (ox, oy) in enumerate(list(higher) + list(lower)):
        rx, ry = ox - x, oy - y
        d = math.hypot(rx, ry)
        if d < clearance:
            ahead, left = c * rx + s * ry, -s * rx + c * ry  # body frame
            blocked |= ahead > 0.3 * d  # inside a ~145 degree cone ahead
            if k < len(higher):
                away -= np.array([ahead, left]) / max(d, 1e-6) + np.array([0.0, 0.5])
    dx, dy = goal[0] - x, goal[1] - y
    dist = math.hypot(dx, dy)
    err = math.atan2(dy, dx) - yaw
    err = math.atan2(math.sin(err), math.cos(err))
    wz = 0.0 if dist < reach else max(-wmax, min(wmax, 1.5 * err))
    # turn in place when facing away; slow near the goal but not below what a walking policy tracks
    vx = 0.0 if blocked or dist < reach else vmax * max(0.0, math.cos(err)) * min(1.0, max(dist, 0.4))
    vy = 0.0
    if away.any():
        vx, vy = vaway * away / np.linalg.norm(away)
        if blocked:
            vx = min(vx, 0.0)  # stepping away must not walk into whoever is in front
    if not holonomic:
        vx, vy = max(vx, 0.0), 0.0
    return np.asarray([vx, vy, wz], np.float32)


def smooth(prev, new, alpha=0.2):
    """Low-pass the command at the 50 Hz fleet rate (about 0.1 s), so the stop and go at the clearance edge
    does not chatter. ponytail: no hysteresis band; add one if gaits still stutter."""
    return (1 - alpha) * np.asarray(prev, np.float32) + alpha * np.asarray(new, np.float32)


def check_goals(goals, min_spacing=1.5):
    """Goals closer than the clearance make parked robots push each other forever."""
    for i in range(len(goals)):
        for j in range(i):
            d = math.dist(goals[i][:2], goals[j][:2])
            assert d >= min_spacing, f"goals {goals[j]} and {goals[i]} are {d:.2f} m apart, need {min_spacing} m"


def yaw_from_wxyz(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
