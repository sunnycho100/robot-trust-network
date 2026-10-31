"""python test_fleet.py: checks fleet.py with unicycle robots standing in for the walking policies."""
import math

import numpy as np

from fleet import assign, check_goals, command, priority, smooth, yaw_from_wxyz


def simulate(starts, goals, order=None, vmax=0.6, dt=0.02, steps=3000, holonomic=(True,) * 4):
    """Same ranking, command and smoothing as multi_robot_scene.py --goal."""
    order = order or assign([s[:2] for s in starts], goals)
    rank = priority(starts, [goals[g] for g in order])
    poses = [list(s) for s in starts]
    cmds = [np.zeros(3, np.float32) for _ in starts]
    min_gap = np.inf
    for _ in range(steps):
        cmds = [smooth(cmds[i], command(p, goals[order[i]], [q[:2] for j, q in enumerate(poses) if rank[j] < rank[i]], vmax,
                                        [q[:2] for j, q in enumerate(poses) if rank[j] > rank[i]], holonomic=holonomic[i]))
                for i, p in enumerate(poses)]
        for p, (vx, vy, wz) in zip(poses, cmds):
            p[2] += wz * dt
            p[0] += (vx * math.cos(p[2]) - vy * math.sin(p[2])) * dt
            p[1] += (vx * math.sin(p[2]) + vy * math.cos(p[2])) * dt
        for i in range(len(poses)):
            for j in range(i):
                min_gap = min(min_gap, math.dist(poses[i][:2], poses[j][:2]))
    return order, poses, min_gap


# longest trip first, ties by list order
assert priority([(0, 0), (0, 1), (5, 5)], [(1, 0), (0, 4), (5, 6)]) == [1, 0, 2]
# nearest goals win: robot 0 at the origin should take the goal next to it
assert assign([[0, 0], [10, 0]], [[9, 0], [1, 0]]) == [1, 0]
# yaw from an Isaac wxyz quaternion
assert abs(yaw_from_wxyz([math.cos(0.4), 0, 0, math.sin(0.4)]) - 0.8) < 1e-9
# a robot behind the goal turns around before walking
assert command((0, 0, 0), (-3, 0), [], 0.6)[0] == 0
# assignment would leave each robot on the goal it stands on
assert assign([[0, 0], [6, 0]], [[6, 0], [0, 0]]) == [1, 0]
# forced head-on swap: both arrive, and they never come closer than about half the clearance
order, poses, gap = simulate([(0, 0, 0), (6, 0.2, math.pi)], [(6, 0), (0, 0)], order=[0, 1])
for i, p in enumerate(poses):
    g = [(6, 0), (0, 0)][order[i]]
    assert math.dist(p[:2], g) < 0.35, (i, p, g)
assert gap > 0.5, gap
# four robots crossing the square diagonally and sideways (forced), all through the middle
starts = [(0, 0, 0), (0, 4, 0), (4, 0, math.pi / 2), (4, 4, math.pi)]
goals = [(4, 4), (4, 0), (0, 4), (0, 0)]
order, poses, gap = simulate(starts, goals, order=[0, 1, 2, 3])
assert all(math.dist(p[:2], goals[order[i]]) < 0.35 for i, p in enumerate(poses)), poses
assert gap > 0.5, gap
print(f"fleet ok, closest approach {gap:.2f} m")
# the Colab demo layout (overnight/colab/fleet_run.sh): four robots converge on a goal cluster, own assignment
starts = [(-4, -3, 0), (-4, -1, 0), (-4, 1, 0), (-4, 3, 0)]
goals = [(3, -1.6), (3, 0), (3, 1.6), (4.6, 0)]
order, poses, gap = simulate(starts, goals)
assert all(math.dist(p[:2], goals[order[i]]) < 0.35 for i, p in enumerate(poses)), poses
assert gap > 0.5, gap
print(f"demo layout ok, assignment {order}, closest approach {gap:.2f} m")
# same layout with both H1s forward-only (no vy, no reversing), as the scene runs them
order, poses, gap = simulate(starts, goals, holonomic=(False, True, False, True))
assert all(math.dist(p[:2], goals[order[i]]) < 0.35 for i, p in enumerate(poses)), poses
assert gap > 0.5, gap
print(f"forward-only H1 ok, closest approach {gap:.2f} m")
check_goals(goals)
try:
    check_goals([(0, 0), (1, 0)])
    raise SystemExit("check_goals missed close goals")
except AssertionError:
    pass
