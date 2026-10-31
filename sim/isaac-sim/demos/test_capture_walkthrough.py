# Checks the pose math in capture_walkthrough.py with numpy only (no Isaac Sim). Usage: python test_capture_walkthrough.py
import os
import tempfile

import numpy as np
import capture_walkthrough as cw


def project(q, t, fx, cx, cy, p):
    w, x, y, z = q
    r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                  [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                  [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])  # COLMAP qvec2rotmat
    c = r @ p + t
    return c[0] / c[2] * fx + cx, c[1] / c[2] * fx + cy, c[2]


fx, cx, cy = 500.0, 640.0, 360.0
eye, target = np.array([2.0, -3.0, 1.6]), np.array([5.0, 1.0, 1.2])
c2w = cw.look_at_c2w(eye, target)
q, t = cw.colmap_pose(c2w)
u, v, z = project(q, t, fx, cx, cy, target)
assert abs(u - cx) < 1e-6 and abs(v - cy) < 1e-6 and z > 0, (u, v, z)       # target lands at the centre, in front
u2, v2, _ = project(q, t, fx, cx, cy, target + [0, 0, 0.5])
assert v2 < cy - 1, v2                                                     # world up is image up (smaller v)
fwd = (target - eye) / np.linalg.norm(target - eye)
right = np.cross(fwd, [0, 0, 1])
u3, _, _ = project(q, t, fx, cx, cy, target + 0.5 * right)
assert u3 > cx + 1, u3                                                     # camera right is image right
assert abs(np.linalg.norm(q) - 1) < 1e-9

# backprojecting a flat depth image and reprojecting gives the same pixels
depth = np.full((720, 1280), 4.0, np.float32)
pts, vv, uu = cw.backproject(depth, fx, fx, cx, cy, c2w, 50, np.random.default_rng(0))
for p, pv, pu in zip(pts, vv, uu):
    a, b, d = project(q, t, fx, cx, cy, p)
    assert abs(a - (pu + 0.5)) < 1e-6 and abs(b - (pv + 0.5)) < 1e-6 and abs(d - 4.0) < 1e-6

# the path stays inside the box, at the requested height, with unit-length horizontal look directions
lo, hi = np.array([-10.0, -20.0, 0.0]), np.array([10.0, 20.0, 8.0])
path = cw.ring_path(lo, hi, 40, 1.6)
assert len(path) == 40 and all((e[:2] > lo[:2]).all() and (e[:2] < hi[:2]).all() and e[2] == 1.6 for e, _ in path)
assert all(abs(np.degrees(np.arctan2(e[2] - t[2], np.linalg.norm(t[:2] - e[:2]))) - 15) < 1e-6 for e, t in path)  # pitched down 15 degrees

with tempfile.TemporaryDirectory() as d:
    cw.write_colmap(d, 1280, 720, fx, fx, cx, cy, ["a.png"], [(q, t)], pts[:2], np.zeros((2, 3), int))
    lines = open(os.path.join(d, "images.txt")).read().split("\n")
    assert lines[0].split()[-1] == "a.png" and lines[1] == ""
    cw.write_ply(os.path.join(d, "m.ply"), np.zeros((3, 3)), np.array([[0, 1, 2]]))
    assert os.path.getsize(os.path.join(d, "m.ply")) > 3 * 12 + 13
print("walkthrough math ok")
