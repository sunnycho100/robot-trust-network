"""Monocular metric depth (colab_depth.py) -> calibrated 3D points -> a bird's-eye height map of the WS30 station.

Each depth map is rescaled so the clear belt floor matches the CarDA calibration (ray hits z = 0), then every pixel
is lifted to the plant frame. Only object placement and size are read from this; no mesh is made.

  python depth_layout.py DEPTH_DIR FRAMES_DIR CAMERAS_JSON OUT_DIR

Writes OUT_DIR/depth_scale.json, bev_height.png (max height per 4 cm cell, grid every 1 m) and points_<frame>.npz.
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np

ap = argparse.ArgumentParser()
for a in ("depth", "frames", "cams", "out"):
    ap.add_argument(a)
a = ap.parse_args()
cams = json.load(open(a.cams))["cams"]
X0, X1, Y0, Y1, CELL = 10.5, 20.0, 0.0, 6.0, 0.04
CLEAR = (15.4, 18.6, 2.05, 3.75)   # belt patch downstream of the trolley, empty in every sampled frame (x0 x1 y0 y1)
H, W = 720, 1280
uu, vv = np.meshgrid(np.arange(W), np.arange(H))
bev = np.zeros((int((Y1 - Y0) / CELL), int((X1 - X0) / CELL)), np.float32)
scales = {}
for f in sorted(glob.glob(os.path.join(a.depth, "*.npy"))):
    name = os.path.basename(f)[:-4]
    side = name.split("_")[0]
    K, R, T = (np.array(cams[side][k]) for k in "KRT")
    C = -R.T @ T
    d = np.load(f).astype(np.float32)
    rays = np.stack([uu, vv, np.ones_like(uu)], -1).reshape(-1, 3) @ np.linalg.inv(K).T     # z = 1 in the camera
    dirs = rays @ R                                                                          # world, per unit z
    s_floor = -C[2] / dirs[:, 2]                                                             # camera z-depth to z = 0
    fp = C + s_floor[:, None] * dirs
    m = (fp[:, 0] > CLEAR[0]) & (fp[:, 0] < CLEAR[1]) & (fp[:, 1] > CLEAR[2]) & (fp[:, 1] < CLEAR[3]) & (s_floor > 0)
    k = float(np.median(s_floor[m] / d.reshape(-1)[m]))
    P = C + (k * d.reshape(-1))[:, None] * dirs
    floor_err = np.abs(P[m, 2])
    scales[name] = dict(scale=round(k, 4), clear_floor_px=int(m.sum()), floor_z_abs_median_m=round(float(np.median(floor_err)), 3),
                        floor_z_abs_p90_m=round(float(np.percentile(floor_err, 90)), 3))
    np.savez_compressed(os.path.join(a.out, f"points_{name}.npz"), P=P.reshape(H, W, 3).astype(np.float16))
    ok = (P[:, 0] > X0) & (P[:, 0] < X1 - CELL) & (P[:, 1] > Y0) & (P[:, 1] < Y1 - CELL) & (P[:, 2] < 3.0)
    ix = ((P[ok, 0] - X0) / CELL).astype(int)
    iy = ((P[ok, 1] - Y0) / CELL).astype(int)
    np.maximum.at(bev, (iy, ix), P[ok, 2])
    print(name, scales[name])
json.dump(dict(model="Depth-Anything-V2-Metric-Indoor-Large-hf", clear_floor_patch_xy=CLEAR, frames=scales),
          open(os.path.join(a.out, "depth_scale.json"), "w"), indent=1)
img = cv2.applyColorMap((np.clip(bev / 2.2, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
img[bev == 0] = 30
img = cv2.resize(img[::-1], None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)           # +y up in the image
sc = 2 / CELL
for x in np.arange(np.ceil(X0), X1):
    cv2.line(img, (int((x - X0) * sc), 0), (int((x - X0) * sc), img.shape[0]), (255, 255, 255), 1)
    cv2.putText(img, f"x{x:.0f}", (int((x - X0) * sc) + 3, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
for y in np.arange(np.ceil(Y0), Y1):
    yy = img.shape[0] - int((y - Y0) * sc)
    cv2.line(img, (0, yy), (img.shape[1], yy), (255, 255, 255), 1)
    cv2.putText(img, f"y{y:.0f}", (3, yy - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
cv2.imwrite(os.path.join(a.out, "bev_height.png"), img)
print("wrote", os.path.join(a.out, "bev_height.png"), "(colour: max height 0..2.2 m, turbo)")
