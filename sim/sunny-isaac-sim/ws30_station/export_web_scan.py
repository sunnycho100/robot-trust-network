"""Export the CarDA WS30 station as a point cloud for the site hero, in the site/scan.js format (see site/tools/export_scan.py).

The dots come from the real CarDA frames (CC BY 4.0): Depth Anything V2 Metric depth, rescaled to the calibrated floor and
lifted to the plant frame by depth_layout.py (points_<frame>.npz, one 720x1280x3 point map per frame). One time step per
camera, so the moving door carrier and worker are not smeared. Shade is the grey level of the same real frame (dark belt,
bright edge strips and lane lines). Station boxes (hulls) come from layout.json.

Cropped to the station footprint (belt, both walkways, trolley lane). Points are ordered for a ground-first reveal:
pts[0 .. ground_count-1] is the floor (z = 0, random order so a reveal fills evenly), the rest is a low band up to 1.6 m,
mainly the trolley lane, thinned with height and sorted by z ascending.

  python export_web_scan.py PTS_DIR FRAMES_DIR CAMERAS_JSON LAYOUT_JSON OUT_JS [--t 25] [--points 13500]

Units in the output are centimetres in the CarDA plant frame (x = belt travel, y towards the IN camera, z up, floor z = 0).
Writes OUT_JS and <OUT_JS stem>_readout.json next to it.
"""
import argparse
import json
import os

import cv2
import numpy as np

ap = argparse.ArgumentParser()
for a in ("pts", "frames", "cams", "layout", "out"):
    ap.add_argument(a)
ap.add_argument("--t", type=int, default=25, help="cycle second of the frames to use (1, 25 or 40)")
ap.add_argument("--points", type=int, default=13500)
a = ap.parse_args()
cams = json.load(open(a.cams))["cams"]
L = json.load(open(a.layout))
pf = L["platform"]
BOX = (12.0, pf["x"][1] + 0.1, pf["y"][0] - 0.1, pf["y"][1] + 0.1)   # station footprint x0 x1 y0 y1: belt, both walkways, trolley lane
GROUND_Z, TOP_Z = 0.06, 1.6   # depth points below GROUND_Z are floor; above-floor band stops at TOP_Z
cx = L["carrier"]["pose_t0"]["x"] + L["conveyor"]["speed_mps"] * a.t   # the carrier rides the belt
FOCUS = (cx - 1.35, cx + 1.0, 2.2, 4.0)   # carrier, door, trim panel and both humanoid stands (layout automation_anchors)
H, W = 720, 1280
uu, vv = np.meshgrid(np.arange(W), np.arange(H))

gp, gg, up, ug = [], [], [], []
for side in ("in", "out"):
    P = np.load(os.path.join(a.pts, f"points_{side}_{a.t}.npz"))["P"].astype(np.float32)
    K, R = np.array(cams[side]["K"]), np.array(cams[side]["R"])
    C = np.array(cams[side]["C"], np.float32)
    r = np.linalg.norm(P - C, axis=2)
    # Monocular depth smears points across object edges: drop pixels whose range jumps > 3 % to a neighbour.
    jump = np.zeros_like(r)
    jump[1:-1, 1:-1] = np.max([np.abs(r[1:-1, 1:-1] - r[:-2, 1:-1]), np.abs(r[1:-1, 1:-1] - r[2:, 1:-1]),
                               np.abs(r[1:-1, 1:-1] - r[1:-1, :-2]), np.abs(r[1:-1, 1:-1] - r[1:-1, 2:])], axis=0)
    ok = (jump < 0.03 * r).reshape(-1)
    g = cv2.imread(os.path.join(a.frames, f"{side}_{a.t}.png"), cv2.IMREAD_GRAYSCALE).astype(np.float32).reshape(-1) / 255
    P = P.reshape(-1, 3)
    # Floor pixels sit where the calibrated ray meets z = 0 (depth only decides which pixels are floor), so the
    # ground is flat and complete even where the monocular depth is a few cm off.
    dirs = (np.stack([uu, vv, np.ones_like(uu)], -1).reshape(-1, 3) @ np.linalg.inv(K).T) @ R
    F = C + (-C[2] / dirs[:, 2])[:, None] * dirs
    fl = ok & (P[:, 2] < GROUND_Z) & (dirs[:, 2] < 0)
    inb = lambda q: (q[:, 0] > BOX[0]) & (q[:, 0] < BOX[1]) & (q[:, 1] > BOX[2]) & (q[:, 1] < BOX[3])
    m = fl & inb(F)
    gp.append(F[m]); gg.append(g[m])
    m = ok & ~fl & inb(P) & (P[:, 2] <= TOP_Z)
    up.append(P[m]); ug.append(g[m])
rng = np.random.default_rng(0)
def cat(ps, gs):   # shuffled, so the voxel survivor is random and the floor does not show the voxel grid
    p, g = np.concatenate(ps), np.concatenate(gs)
    o = rng.permutation(len(p))
    return p[o], g[o]
gp, gg = cat(gp, gg)
up, ug = cat(up, ug)
gp[:, 2] = 0
foc = (up[:, 0] > FOCUS[0]) & (up[:, 0] < FOCUS[1]) & (up[:, 1] > FOCUS[2]) & (up[:, 1] < FOCUS[3])
keep_p = np.where(foc, 1.0, 0.3) * (1 - 0.65 * up[:, 2] / TOP_Z)   # mainly the trolley lane, thinner with height
coin = rng.random(len(up))

# Voxel downsample both layers on one grid; grow the voxel until the total fits --points.
v = 0.02
while True:
    _, gi = np.unique(np.floor(gp[:, :2] / v).astype(np.int64), axis=0, return_index=True)
    _, ui = np.unique(np.floor(up / v).astype(np.int64), axis=0, return_index=True)
    ui = ui[coin[ui] < keep_p[ui]]
    if len(gi) + len(ui) <= a.points:
        break
    v *= 1.05
ui = ui[np.argsort(up[ui, 2], kind="stable")]   # ground first (random order, even fill), then rising by height
p = np.concatenate([gp[gi], up[ui]])
shade = np.clip(np.round((np.concatenate([gg[gi], ug[ui]]) - 0.08) / 0.7 * 9), 0, 9)  # frame grey level, 0 = black belt and door
pts = np.column_stack([np.round(p * 100), shade]).astype(int)
ground_count = len(gi)
cam_rows = [[*np.round(np.array(cams[s]["C"]) * 100), *np.round(np.array(cams[s]["R"])[2] * 100)] for s in ("in", "out")]

# Station objects as boxes [cx, cy, cz, hx, hy, hz, yaw] in cm, carrier placed at time t (it rides the belt).
cr, ls = L["carrier"], L["line_side"]
cy = cr["pose_t0"]["y"]
def b(x, y, z, sx, sy, sz):
    return [round(x * 100), round(y * 100), round(z * 100), round(sx * 50), round(sy * 50), round(sz * 50), 0]
f, d = cr["frame_size_local"], L["door"]["local"]
hulls = [b(cx, cy, f["z_top"] / 2, f["x"][1] - f["x"][0], f["y"][1] - f["y"][0], f["z_top"]),
         b(cx + (d["x"][0] + d["x"][1]) / 2, cy + d["y_center"], (d["z_bottom"] + d["z_top"]) / 2,
           d["x"][1] - d["x"][0], d["thickness"], d["z_top"] - d["z_bottom"])]
for k_ in cr["klt_bins_local"]:
    hulls.append(b(cx + k_["center"][0], cy + k_["center"][1], k_["center"][2], *k_["size"]))
tp = cr["trim_panel_local"]
hulls.append(b(cx + tp["center"][0], cy + tp["center"][1], tp["center"][2], *tp["size"]))   # ponytail: 30 deg tilt ignored
for key in ("screwdriver_rack", "desk_in_side", "desks_out_side", "monitor_cart_out_side", "electrical_cabinet"):
    o = ls[key]
    hulls.append(b(*o["center"], o["size"][2] / 2, *o["size"]))
cb = ls["checkerboard_stand"]
hulls.append(b(*cb["center"], sum(cb["board_z"]) / 2, cb["board_size"][0], 0.04, cb["board_z"][1] - cb["board_z"][0]))
for px, py in ls["gantry_posts"]:
    hulls.append(b(px, py, ls["gantry_post_height"] / 2, 0.2, 0.2, ls["gantry_post_height"]))   # post section is a guess
hulls = [h for h in hulls if BOX[0] * 100 < h[0] < BOX[1] * 100 and BOX[2] * 100 < h[1] < BOX[3] * 100]   # inside the footprint

by = L["conveyor"]["belt_y"]
data = {"voxel_cm": round(v * 100, 1), "pts": pts.ravel().tolist(), "cams": np.array(cam_rows).astype(int).ravel().tolist(),
        "ground_count": ground_count, "walls": [], "hulls": hulls, "spawn": [1700, round(50 * (by[0] + by[1]))]}   # spawn: clear belt downstream of the carrier
with open(a.out, "w") as out:
    out.write("/* Generated by sim/sunny-isaac-sim/ws30_station/export_web_scan.py from CarDA WS30 cycle 003 at t = %d s "
              "(real IN and OUT frames, Depth Anything V2 Metric rescaled to the calibrated floor; CarDA CC BY 4.0). "
              "Points [x,y,z,shade] in cm (shade = frame grey level 0..9), cams [x,y,z,fx,fy,fz]. No walls: the station is an open floor. "
              "Order: the first ground_count points are the floor (z = 0, random order), the rest rise by z up to 160 cm. */\n" % a.t)
    out.write("window.SCAN=" + json.dumps(data, separators=(",", ":")) + ";\n")

lo, hi = p.min(0), p.max(0)
readout = {"name": "CarDA WS30", "points": len(pts), "voxel_cm": data["voxel_cm"], "cameras": len(cam_rows),
           "station_size_m": [round(float(hi[0] - lo[0]), 1), round(float(hi[1] - lo[1]), 1), round(float(hi[2] - lo[2]), 1)],
           "size_note": "x (belt) by y (across) by z, bounding box of the points, as engine.js measures the room",
           "method": "real CarDA video frames back-projected with the calibrated IN and OUT cameras and Depth Anything V2 "
                     "Metric depth rescaled to the calibrated floor (cycle 003, t = %d s); shade = grey level of the same frame. Cropped to the "
                     "station footprint; floor pixels (depth z < 6 cm) placed where their calibrated ray meets z = 0; above-floor "
                     "band 6 to 160 cm, mainly the trolley lane, thinned with height" % a.t,
           "ground_count": ground_count, "above_count": len(pts) - ground_count, "max_z_cm": int(pts[:, 2].max()),
           "order": "pts[0 .. ground_count-1] floor (z = 0, random order); pts[ground_count ..] sorted by z ascending",
           "hulls": len(hulls), "walls": 0, "credit": "CarDA dataset, CC BY 4.0 (Zenodo 10.5281/zenodo.14644367)"}
json.dump(readout, open(os.path.splitext(a.out)[0] + "_readout.json", "w"), indent=1)

assert 0 < len(pts) <= a.points and len(data["pts"]) % 4 == 0 and len(data["cams"]) == 12
assert pts[:, 3].min() >= 0 and pts[:, 3].max() <= 9
assert (pts[:ground_count, 2] == 0).all() and (np.diff(pts[ground_count:, 2]) >= 0).all()
assert pts[ground_count:, 2].min() >= 5 and pts[:, 2].max() <= 160
print(f"{len(pts)} points (voxel {v*100:.1f} cm), {len(cam_rows)} cameras, {len(hulls)} hulls, "
      f"size {readout['station_size_m']} m, ground {ground_count}, shade counts {np.bincount(pts[:, 3], minlength=10).tolist()} -> {a.out}")
