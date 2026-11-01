"""Register the CarDA WS30 Xsens BVH (cycle 003) into the plant frame and check it against both station cameras.

Xsens zero-velocity updates make the mocap frame ride with the belt while the worker stands on it (robot_consult_demo
s1), so plant_xy(t) = Rz(yaw) m_xy(t) + t0 + (v max(t - t_on, 0), 0). The worker steps back onto the belt at about
t_on = 12 s (after the trip to the line-side rack). yaw and t0 are fitted to the worker's feet picked by hand in the
real frames (PICKS, projected to the floor with the CarDA calibration); the fit residuals are printed.

  python register_bvh.py BVH IN_MP4 OUT_MP4 CAMERAS_JSON OUT_DIR

Writes OUT_DIR/worker_plant.npz (joints per frame in the plant frame, metres, z up, 60 fps) and register_check.jpg.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../robot_consult_demo/src"))
from rcd.common.bvh import _parse, load_bvh  # noqa: E402

# (video time s, camera, u, v): midpoint between the worker's shoe soles, cycle 003
PICKS = [(1, "out", 383, 382), (9, "out", 530, 330), (15, "in", 1230, 542), (22, "in", 1220, 557),
         (36, "in", 1070, 545), (45, "in", 1028, 548), (52, "in", 1005, 537)]

ap = argparse.ArgumentParser()
for a in ("bvh", "in_mp4", "out_mp4", "cams", "out"):
    ap.add_argument(a)
ap.add_argument("--v", type=float, default=0.0239, help="m/s belt speed, WS30 (s1 phase correlation)")
ap.add_argument("--t_on", type=float, default=12.0, help="s, worker back on the belt for good")
ap.add_argument("--times", default="1,9,22,36,52")
a = ap.parse_args()

cams = json.load(open(a.cams))["cams"]


def floor(side, u, v):
    K, R, T = (np.array(cams[side][k]) for k in "KRT")
    C = -R.T @ T
    d = R.T @ np.linalg.solve(K, [u, v, 1.0])
    return (C + (-C[2] / d[2]) * d)[:2]


b = load_bvh(a.bvh)
names, fps, J = b["names"], b["fps"], b["joints"]
parents = [j.parent for j in _parse(a.bvh)[0]]
feet = [names.index("RightAnkle"), names.index("LeftAnkle")]
ts = np.array([p[0] for p in PICKS], float)
drift = a.v * np.maximum(ts - a.t_on, 0)
Q = np.array([floor(s, u, v) for _, s, u, v in PICKS]) - np.c_[drift, 0 * ts]
M = np.array([J[int(t * fps), feet, :2].mean(0) for t in ts])
best = None
for yaw in np.arange(0, 360, 0.5):
    c, s = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
    Rm = M @ np.array([[c, -s], [s, c]]).T
    t0 = (Q - Rm).mean(0)
    r = np.linalg.norm(Q - Rm - t0, axis=1)
    if best is None or r.mean() < best[0]:
        best = (r.mean(), yaw, t0, r)
_, yaw, t0, res = best
print(f"fit: yaw {yaw:.1f} deg, t0 {np.round(t0, 3).tolist()}, feet residuals m {np.round(res, 2).tolist()}")

c, s = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
tt = np.arange(len(J)) / fps
plant = J @ Rz.T + [t0[0], t0[1], 0]
plant[..., 0] += (a.v * np.maximum(tt - a.t_on, 0))[:, None]
os.makedirs(a.out, exist_ok=True)
np.savez_compressed(os.path.join(a.out, "worker_plant.npz"), joints=plant.astype(np.float32), names=np.array(names),
                    parents=np.array(parents), fps=np.float32(fps), yaw_deg=np.float32(yaw), t0=np.float32(t0),
                    v=np.float32(a.v), t_on=np.float32(a.t_on), residual_m=np.float32(res))

tiles = []
for t in [float(x) for x in a.times.split(",")]:
    P = plant[min(int(t * fps), len(J) - 1)]
    row = []
    for side, mp4 in (("in", a.in_mp4), ("out", a.out_mp4)):
        cap = cv2.VideoCapture(mp4)
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        _, img = cap.read()
        K, R, T = (np.array(cams[side][k]) for k in "KRT")
        x = P @ R.T + T
        uv = (x @ K.T)[:, :2] / x[:, 2:3]
        for j, p in enumerate(parents):
            if p >= 0:
                cv2.line(img, tuple(map(int, uv[p])), tuple(map(int, uv[j])), (0, 255, 255), 3)
        cv2.putText(img, f"{side} t={t:.0f}s", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        row.append(cv2.resize(img, (640, 360)))
    tiles.append(np.hstack(row))
cv2.imwrite(os.path.join(a.out, "register_check.jpg"), np.vstack(tiles))
print("wrote", os.path.join(a.out, "register_check.jpg"))
