"""Camera calibration for customer footage when no calibration file is delivered (intake `cameras.calibration`).

Two field procedures that a site visit can realistically provide, both giving K, R, t in the pipeline convention
x_cam = R @ X_world + t with a z-up world whose floor is z = 0:

* `calibrate_checkerboard`: a printed checkerboard of known square size is filmed by the camera. If the board is
  moved around, intrinsics come from Zhang's method over all views; the extrinsics come from one view in which the
  board lies flat on the floor (the board then defines the world frame: origin at its first inner corner, x along
  its rows, z up).
* `calibrate_reference_length`: >= 4 floor points with known metric coordinates (tape-measured corners of a floor
  marking, a pallet, a tile grid, ...) clicked in one image. The floor homography gives the focal length
  (square pixels, principal point at the image centre: one unknown, solved from the orthogonality of the rotation
  columns) and then R, t by decomposition. With exactly two points of known distance only a scale would be
  observable, so the intake lists points with coordinates; `reference_lengths_m` entries are
  {"image_px": [[u, v], ...], "world_xy_m": [[x, y], ...]} per camera.
"""
from __future__ import annotations

import cv2
import numpy as np


def _proper(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(R)
    Ro = U @ Vt
    if np.linalg.det(Ro) < 0:
        U[:, -1] *= -1
        Ro = U @ Vt
    return Ro


def calibrate_checkerboard(frames: list[np.ndarray], pattern: tuple[int, int], square_m: float,
                           floor_frame: int | None = None, K: np.ndarray | None = None) -> dict | None:
    """Intrinsics (if `K` is None) + floor-referenced extrinsics from checkerboard views.

    frames: grayscale images of one static camera; pattern: inner corners (cols, rows); square_m: square size.
    floor_frame: index (into the frames where the board was found) of the view with the board lying on the floor;
    default = the view whose board is most nearly parallel to the image's lower half (largest board area).
    Returns {K, dist, R, t, rmse_px, frames_used} or None if fewer than 1 (K given) / 3 (K unknown) detections."""
    cols, rows = pattern
    obj = np.zeros((cols * rows, 3), np.float32)
    obj[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_m
    pts, used = [], []
    for i, g in enumerate(frames):
        ok, c = cv2.findChessboardCornersSB(g, pattern, flags=cv2.CALIB_CB_ACCURACY)
        if ok:
            pts.append(c.reshape(-1, 1, 2).astype(np.float32))
            used.append(i)
    if len(pts) < (1 if K is not None else 3):
        return None
    h, w = frames[0].shape[:2]
    if K is None:
        rms, K, dist, rvecs, tvecs = cv2.calibrateCamera([obj] * len(pts), pts, (w, h), None, None)
    else:
        dist = np.zeros(5)
        rvecs, tvecs, errs = [], [], []
        for p in pts:
            ok, rv, tv = cv2.solvePnP(obj, p, K, dist)
            rvecs.append(rv)
            tvecs.append(tv)
            pr, _ = cv2.projectPoints(obj, rv, tv, K, dist)
            errs.append(np.linalg.norm(pr.reshape(-1, 2) - p.reshape(-1, 2), axis=1))
        rms = float(np.sqrt(np.mean(np.concatenate(errs) ** 2)))
    if floor_frame is None:
        floor_frame = int(np.argmax([cv2.contourArea(cv2.convexHull(p)) for p in pts]))
    Rb, _ = cv2.Rodrigues(rvecs[floor_frame])
    tb = np.asarray(tvecs[floor_frame], float).reshape(3)
    # board frame: x along rows, y along columns, z into the board (OpenCV) -> flip y,z so world z is up
    F = np.diag([1.0, -1.0, -1.0])
    R = Rb @ F
    if (R.T @ -tb)[2] < 0:  # camera must be above the floor
        F = np.diag([-1.0, 1.0, -1.0])
        R = Rb @ F
    return {"K": np.asarray(K, float), "dist": np.asarray(dist, float).ravel()[:5], "R": _proper(R), "t": tb,
            "rmse_px": float(rms), "frames_used": used, "method": "checkerboard (Zhang + floor-lying board pose)"}


def calibrate_reference_length(image_px, world_xy_m, image_size: tuple[int, int]) -> dict:
    """K, R, t from >= 4 floor points with known metric floor coordinates (z = 0) in one image.

    Model: square pixels, zero skew, principal point at the image centre, no distortion -> f is the single
    intrinsic unknown, solved in closed form from h1.T w h2 = 0 and |K^-1 h1| = |K^-1 h2| of the floor
    homography H = K [r1 r2 t]; then refined with solvePnP. Returns {K, dist, R, t, rmse_px, focal_px}."""
    img = np.asarray(image_px, np.float64).reshape(-1, 2)
    wxy = np.asarray(world_xy_m, np.float64).reshape(-1, 2)
    if len(img) < 4:
        raise ValueError("need >= 4 floor points")
    W, Hh = image_size
    cx, cy = W / 2.0, Hh / 2.0
    H, _ = cv2.findHomography(wxy, img, 0)
    # shift the principal point to the origin: H' = T H, then K' = diag(f, f, 1)
    T = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1.0]])
    Hp = T @ H
    h1, h2 = Hp[:, 0], Hp[:, 1]
    # orthogonality: (h11 h21 + h12 h22)/f^2 + h13 h23 = 0 ; equal norms: (h11^2+h12^2-h21^2-h22^2)/f^2 + h13^2-h23^2 = 0
    cands = []
    if abs(h1[2] * h2[2]) > 1e-12:
        cands.append(-(h1[0] * h2[0] + h1[1] * h2[1]) / (h1[2] * h2[2]))
    d = h1[2] ** 2 - h2[2] ** 2
    if abs(d) > 1e-12:
        cands.append(-(h1[0] ** 2 + h1[1] ** 2 - h2[0] ** 2 - h2[1] ** 2) / d)
    f2 = [c for c in cands if c > 0]
    f = float(np.sqrt(np.median(f2))) if f2 else float(max(W, Hh))
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    obj = np.c_[wxy, np.zeros(len(wxy))]
    if len(img) >= 5:  # refine f jointly with the pose (principal point, aspect and distortion held fixed)
        fl = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO |
              cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3)
        _, K2, _, _, _ = cv2.calibrateCamera([obj.astype(np.float32)], [img.astype(np.float32).reshape(-1, 1, 2)],
                                             (W, Hh), K.copy(), np.zeros(5), flags=fl)
        if np.isfinite(K2).all() and K2[0, 0] > 0:
            K, f = K2, float(K2[0, 0])
    ok, rv, tv = cv2.solvePnP(obj, img, K, np.zeros(5), flags=cv2.SOLVEPNP_ITERATIVE if len(img) > 4 else cv2.SOLVEPNP_IPPE)
    R, _ = cv2.Rodrigues(rv)
    t = tv.reshape(3)
    if (R.T @ -t)[2] < 0:
        raise ValueError("solution places the camera below the floor; check point order")
    pr, _ = cv2.projectPoints(obj, rv, tv, K, np.zeros(5))
    rmse = float(np.sqrt(np.mean(np.sum((pr.reshape(-1, 2) - img) ** 2, 1))))
    return {"K": K, "dist": np.zeros(5), "R": _proper(R), "t": t, "rmse_px": rmse, "focal_px": f,
            "method": "floor reference points (homography focal + PnP)"}


def _background(path, n: int = 9) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fr = []
    for k in np.linspace(0, max(total - 2, 0), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(k))
        ok, im = cap.read()
        if ok:
            fr.append(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
    cap.release()
    return np.median(np.stack(fr), axis=0).astype(np.uint8)


def scene_shift(ref_video, cur_video, K: np.ndarray, stale_px: float = 3.0) -> dict | None:
    """Has the camera moved since the calibration session? Compare the static background (median frame) of a clip
    recorded with the calibration against the current clip: ORB matches + RANSAC homography; for a camera that was
    bumped (rotated about its centre) the background maps by H = K R_d K^-1. Returns the median displacement of
    the inlier matches, and when it exceeds `stale_px` the rotation R_d to update the extrinsics
    (R_new = R_d R, t_new = R_d t: same centre, new orientation)."""
    a, b = _background(ref_video), _background(cur_video)
    orb = cv2.ORB_create(4000)
    ka, da = orb.detectAndCompute(a, None)
    kb, db = orb.detectAndCompute(b, None)
    if da is None or db is None:
        return None
    m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, db)
    if len(m) < 20:
        return None
    pa = np.float32([ka[x.queryIdx].pt for x in m])
    pb = np.float32([kb[x.trainIdx].pt for x in m])
    H, inl = cv2.findHomography(pa, pb, cv2.RANSAC, 3.0)
    if H is None:
        return None
    inl = inl.ravel() > 0
    disp = float(np.median(np.linalg.norm(pb[inl] - pa[inl], axis=1)))
    out = {"median_displacement_px": round(disp, 2), "inlier_matches": int(inl.sum()), "stale": bool(disp > stale_px)}
    if disp > stale_px:
        Rd = np.linalg.inv(K) @ H @ K
        Rd = _proper(Rd / np.cbrt(np.linalg.det(Rd)))
        out["R_delta"] = Rd
        out["rotation_deg"] = round(float(np.degrees(np.arccos(np.clip((np.trace(Rd) - 1) / 2, -1, 1)))), 3)
    return out
