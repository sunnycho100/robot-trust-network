"""python test_navmap.py: checks navmap.py on synthetic points (a wall with one gap, a floor, a ceiling)."""
import math

import numpy as np

from navmap import NavMap

rng = np.random.default_rng(0)
floor = np.c_[rng.uniform(-5, 5, (20000, 2)), np.zeros(20000)]
ceiling = np.c_[rng.uniform(-5, 5, (5000, 2)), np.full(5000, 4.0)]  # above robot height: must not block
wall = np.c_[np.zeros(4000), rng.uniform(-5, 5, 4000), rng.uniform(0, 3, 4000)]  # wall along x = 0
wall = wall[np.abs(wall[:, 1] - 3.0) > 0.8]  # with a 1.6 m gap at y = 3
pts = np.concatenate([floor, ceiling, wall])

m = NavMap(pts, radius=0.35)
assert abs(m.floor_z) < 0.05, m.floor_z
assert m.free(-2, 0) and not m.free(0, 0) and m.free(0, 3), "wall blocks, gap is open"
path = m.path((-3, 0), (3, 0))
assert path[0] == (-3, 0) and path[-1] == (3, 0), path
assert any(abs(x) < 0.3 and abs(y - 3) < 0.6 for x, y in path), f"path should go through the gap: {path}"
# every leg of the simplified path stays in free space
for a, b in zip(path, path[1:]):
    n = int(math.dist(a, b) / 0.05) + 2
    assert all(m.free(x, y) for x, y in np.linspace(a, b, n)), (a, b)
length = sum(math.dist(a, b) for a, b in zip(path, path[1:]))
assert length < 1.2 * (2 * math.hypot(3, 3)), length  # close to the shortest way round
assert len(path) <= 4, path  # straight runs collapse to corners
assert m.path((-3, 0), (-1, 0)) == [(-3, 0), (-1, 0)]  # clear line: straight
assert m.clear((-3, 0), (-1, 0)) and not m.clear((-3, 0), (3, 0))

# goal inside an obstacle or no way through
sealed = NavMap(np.concatenate([floor, np.c_[np.zeros(4000), rng.uniform(-5, 5, 4000), rng.uniform(0, 3, 4000)]]), radius=0.35)
assert sealed.path((-3, 0), (3, 0)) is None
# a reconstruction collider with a 1.2 m hole in the floor (no ground plane underneath): with support=True the hole is
# blocked and the path goes round it; the floor's own sampling gaps (about 1 in 7 empty 10 cm cells here) stay free
holed = floor[np.hypot(floor[:, 0], floor[:, 1]) > 0.6]
assert NavMap(holed, radius=0.35, floor_z=0.0).path((-3, 0), (3, 0)) == [(-3, 0), (3, 0)]  # without: straight over it
s = NavMap(holed, radius=0.35, floor_z=0.0, support=True)
assert not s.free(0, 0) and s.free(-3, 0) and s.free(3, 0)
route = s.path((-3, 0), (3, 0))
assert route and len(route) > 2, route
for a, b in zip(route, route[1:]):
    t = np.linspace(0, 1, 200)[:, None]
    assert (np.hypot(*(np.asarray(a) + t * (np.subtract(b, a))).T) > 0.6 + 0.3).all(), "path crosses the hole"
inner = (np.abs(s.lo[0] + (np.arange(s.shape[0]) + 0.5) * s.res) < 4)[:, None] & \
        (np.abs(s.lo[1] + (np.arange(s.shape[1]) + 0.5) * s.res) < 4)[None, :]
plain = NavMap(holed, radius=0.35, floor_z=0.0)
assert (s.blocked & ~plain.blocked & inner).sum() * s.res ** 2 < 4.0, "support mask blocks far more than the hole"

# one H1 stand-in follows the path with fleet.command, advancing waypoints at 0.5 m as multi_robot_scene.py --plan does.
# Following cuts corners by about 0.15 m, so the plan inflates by body radius (0.3) plus 0.2 m, as multi_robot_scene does.
from fleet import command, smooth

m2 = NavMap(pts, radius=0.3 + 0.2)
route = m2.path((-3, 0), (3, 0))
pose, cmd, k, gap = [-3.0, 0.0, 0.0], np.zeros(3, np.float32), 1, np.inf
for _ in range(3000):
    if k < len(route) - 1 and math.dist(pose[:2], route[k]) < 0.5:
        k += 1
    cmd = smooth(cmd, command(pose, route[k], [], 0.6, holonomic=False))
    pose[2] += cmd[2] * 0.02
    pose[0] += cmd[0] * math.cos(pose[2]) * 0.02
    pose[1] += cmd[0] * math.sin(pose[2]) * 0.02
    gap = min(gap, np.hypot(wall[:, 0] - pose[0], wall[:, 1] - pose[1]).min())
assert k == len(route) - 1 and math.dist(pose[:2], route[-1]) < 0.35, pose
assert gap > 0.3, f"centre came {gap:.2f} m from the wall, inside the body radius"
print("navmap ok, path", [(round(x, 2), round(y, 2)) for x, y in path], f"length {length:.2f} m; followed with {gap:.2f} m wall clearance")
