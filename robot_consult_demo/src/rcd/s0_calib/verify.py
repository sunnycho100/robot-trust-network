"""Verify a camera's calibration against targets detected in the CarDA mp4 (left ZED image, 1280x720).

* Checkerboard (OUT views): cv2.findChessboardCornersSB over several pattern sizes. The board and the
  camera are both static, so one board pose is fitted to the corners of *all* sampled frames
  (solvePnP, square size 1 unit -> RMSE is scale-free). Scale-free extrinsic check: the board stands
  on a vertical stand, so its plane normal expressed in the world frame (R_cal^T R_board) should be
  horizontal.
* ArUco (IN views; marker on the cart): per-frame IPPE_SQUARE pose of a unit marker -> corner RMSE
  (only 4 points per frame: weak check). Scale-free extrinsic check: the marker rides on the conveyor,
  so its displacement direction in the world frame should be horizontal and along the line (x axis).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

PATTERNS = [(6, 8), (8, 6), (7, 9), (9, 7), (5, 7), (7, 5), (6, 9), (9, 6), (5, 8), (8, 5), (6, 7), (7, 6),
            (7, 10), (10, 7), (4, 6), (6, 4)]
ARUCO_DICTS = ["DICT_4X4_50", "DICT_5X5_100", "DICT_6X6_250", "DICT_ARUCO_ORIGINAL", "DICT_APRILTAG_36h11"]


def sample_frames(path: str | Path, n: int, gray: bool = True) -> list[tuple[int, np.ndarray]]:
    """n frames evenly spaced over the clip (seek; targets are static or slow, exact frame index not critical)."""
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    for k in np.linspace(0, max(total - 2, 0), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(k))
        ok, im = cap.read()
        if ok:
            out.append((int(k), cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) if gray else im))
    cap.release()
    return out


def _rmse(obj, img, rvec, tvec, K, dist) -> np.ndarray:
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    return np.linalg.norm(proj.reshape(-1, 2) - img.reshape(-1, 2), axis=1)


def checkerboard(frames, K, dist, R_cal) -> dict | None:
    pattern, dets = None, []
    for n_tried, (k, g) in enumerate(frames):
        if pattern is None and n_tried >= 2:
            return None  # no board in the first frames: the board is static, so it is not in this view
        pats = [pattern] if pattern else PATTERNS
        for p in pats:
            ok, c = cv2.findChessboardCornersSB(g, p, flags=cv2.CALIB_CB_ACCURACY | (
                cv2.CALIB_CB_EXHAUSTIVE if pattern else 0))
            if ok:
                pattern = p
                dets.append((k, c.reshape(-1, 2)))
                break
    if not dets:
        return None
    cols, rows = pattern
    obj = np.zeros((cols * rows, 3), np.float64)
    obj[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)  # unit squares
    # per-frame poses, then one joint pose for the static board
    per = []
    for _, c in dets:
        ok, rv, tv = cv2.solvePnP(obj, c, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        per.append(float(np.sqrt(np.mean(_rmse(obj, c, rv, tv, K, dist) ** 2))))
    allobj = np.vstack([obj] * len(dets))
    allimg = np.vstack([c for _, c in dets])
    ok, rv, tv = cv2.solvePnP(allobj, allimg, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    err = _rmse(allobj, allimg, rv, tv, K, dist)
    Rb, _ = cv2.Rodrigues(rv)
    Rw = R_cal.T @ Rb  # board axes in the world frame
    elev = [float(np.degrees(np.arcsin(min(1.0, abs(Rw[2, j]))))) for j in range(3)]
    return {
        "target": "checkerboard",
        "pattern_inner_corners": list(pattern),
        "frames_detected": len(dets),
        "frames_sampled": len(frames),
        "rmse_px_joint_static_pose": float(np.sqrt(np.mean(err ** 2))),
        "rmse_px_per_frame_median": float(np.median(per)),
        # scale-free extrinsic check: a board on a stand may lean back (normal elevation != 0) but its
        # horizontal edge stays level, so min(edge elevations) ~ 0 if R_cal is consistent with gravity
        "board_edge_elevation_deg": min(elev[0], elev[1]),
        "board_normal_elevation_deg": elev[2],
    }


def aruco(frames, K, dist, R_cal) -> dict | None:
    best = None
    for dn in ARUCO_DICTS:
        det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dn)),
                                      cv2.aruco.DetectorParameters())
        hits = []
        for k, g in frames:
            corners, ids, _ = det.detectMarkers(g)
            if ids is not None:
                for c, i in zip(corners, ids.ravel()):
                    hits.append((k, int(i), c.reshape(4, 2).astype(np.float64)))
        if hits and (best is None or len(hits) > len(best[1])):
            best = (dn, hits)
    if not best:
        return None
    dn, hits = best
    obj = np.array([[-0.5, 0.5, 0], [0.5, 0.5, 0], [0.5, -0.5, 0], [-0.5, -0.5, 0]], np.float64)
    errs, track = [], {}
    for k, i, c in hits:
        ok, rv, tv = cv2.solvePnP(obj, c, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        errs.append(_rmse(obj, c, rv, tv, K, dist))
        track.setdefault(i, []).append((k, tv.ravel()))
    res = {
        "target": "aruco",
        "dictionary": dn,
        "marker_ids": sorted(track),
        "detections": len(hits),
        "frames_sampled": len(frames),
        "rmse_px": float(np.sqrt(np.mean(np.concatenate(errs) ** 2))),
    }
    # displacement direction of the most-seen marker (scale-free): first vs last detection
    i, tr = max(track.items(), key=lambda kv: len(kv[1]))
    tr.sort(key=lambda x: x[0])
    if len(tr) >= 2:
        d = R_cal.T @ (tr[-1][1] - tr[0][1])
        if np.linalg.norm(d) > 0:
            d = d / np.linalg.norm(d)
            res["marker_motion_dir_world"] = [float(x) for x in d]
            res["marker_motion_angle_from_x_deg"] = float(np.degrees(np.arccos(min(1.0, abs(d[0])))))
    return res
