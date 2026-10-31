"""Video measurements on the floor plane (z = 0 of the s0 world frame).

Each calibrated camera gives a homography floor(x, y) -> image, so a frame can be resampled into a metric
bird's-eye view (BEV). Two measurements use it:

* belt_edges: the conveyor is bordered by two yellow toothed strips; their y positions come from the
  row profile of a yellow mask in the BEV of a background (median) frame.
* belt_speed: the conveyor surface (and the carts on it) translates along x. Phase correlation between
  BEV crops of the belt interior taken `dt` seconds apart gives the shift in pixels = metres * 1/res.
"""
from __future__ import annotations

import cv2
import numpy as np

RES = 0.005  # m per BEV pixel


def floor_to_image(K, R, t, z0: float = 0.0) -> np.ndarray:
    P = np.asarray(K) @ np.hstack([np.asarray(R), np.asarray(t).reshape(3, 1)])
    H = P[:, [0, 1, 3]].copy()
    H[:, 2] += P[:, 2] * z0
    return H


def bev_map(cam: dict, x0: float, x1: float, y0: float, y1: float, res: float = RES):
    """Homography BEV pixel -> image pixel (for warpPerspective with WARP_INVERSE_MAP) and BEV size.
    BEV column u -> x = x0 + u*res ; row v -> y = y1 - v*res."""
    A = np.array([[res, 0, x0], [0, -res, y1], [0, 0, 1.0]])
    H = floor_to_image(cam["K"], cam["R"], cam["t"]) @ A
    return H, (int(round((x1 - x0) / res)), int(round((y1 - y0) / res)))


def _peaks(profile: np.ndarray, min_sep: int, n: int = 2) -> list[int]:
    p = profile.copy()
    out = []
    for _ in range(n):
        i = int(np.argmax(p))
        if p[i] <= 0:
            break
        out.append(i)
        p[max(0, i - min_sep):i + min_sep] = 0
    return sorted(out)


def belt_edges(bg: np.ndarray, cam: dict, xc: float, y_range=(0.6, 5.2)) -> dict | None:
    """y of the two yellow strips bordering the conveyor (world metres), from a background frame."""
    res = 0.01
    x0, x1 = xc - 2.5, xc + 2.5
    H, size = bev_map(cam, x0, x1, y_range[0], y_range[1], res)
    b = cv2.warpPerspective(bg, H, size, flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    hsv = cv2.cvtColor(b, cv2.COLOR_BGR2HSV)
    yellow = ((hsv[..., 0] >= 15) & (hsv[..., 0] <= 40) & (hsv[..., 1] >= 90) & (hsv[..., 2] >= 90))
    valid = b.sum(2) > 0
    prof = yellow.sum(1) / np.maximum(valid.sum(1), 1)
    prof = np.convolve(prof, np.ones(5) / 5, mode="same")
    pk = _peaks(prof, min_sep=int(0.8 / res))
    if len(pk) < 2 or min(prof[pk]) < 0.15:
        return None
    ys = sorted(y_range[1] - np.array(pk) * res)
    return {"y_low": float(ys[0]), "y_high": float(ys[1]), "strength": [float(prof[i]) for i in pk]}


def background(path, n: int = 15) -> np.ndarray:
    """Median of n frames spread over the clip (removes the worker and, mostly, the slow cart)."""
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fr = []
    for k in np.linspace(0, max(total - 2, 0), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(k))
        ok, im = cap.read()
        if ok:
            fr.append(im)
    cap.release()
    return np.median(np.stack(fr), axis=0).astype(np.uint8)


def _orange_fg(small: list[np.ndarray]) -> float:
    """Median count (per 320x180 frame) of saturated-orange pixels that differ from the clip background.
    The Xsens suit has orange straps; the camera on the operator's side sees them larger/unoccluded."""
    med = np.median(np.stack(small), axis=0).astype(np.int16)
    cnt = []
    for im in small:
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        orange = (hsv[..., 0] >= 3) & (hsv[..., 0] <= 18) & (hsv[..., 1] >= 150) & (hsv[..., 2] >= 120)
        fg = np.abs(im.astype(np.int16) - med).sum(2) > 60
        cnt.append(int((orange & fg).sum()))
    return float(np.median(cnt))


def scan(path, cam: dict, xc: float, band: tuple[float, float], max_samples: int = 240,
         dt_target: float = 2.0, min_resp: float = 0.2) -> dict:
    """One sequential pass: belt speed (phase correlation of BEV belt crops dt_target apart) + orange score."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(15, int(np.ceil(total / max_samples)))
    gap = max(1, int(round(dt_target * fps / step)))
    x0, x1 = xc - 2.0, xc + 2.0
    H, size = bev_map(cam, x0, x1, band[0], band[1])
    bev, small, k = [], [], 0
    while True:
        if not cap.grab():
            break
        if k % step == 0:
            ok, im = cap.retrieve()
            if ok:
                bev.append(cv2.cvtColor(cv2.warpPerspective(im, H, size, flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR),
                                        cv2.COLOR_BGR2GRAY))
                if (k // step) % max(1, int(round(fps / step))) == 0:
                    small.append(cv2.resize(im, (320, 180), interpolation=cv2.INTER_AREA))
                shape = im.shape[:2]
        k += 1
    cap.release()
    out = {"frames_read": k, "fps": fps, "orange_fg_px": _orange_fg(small) if small else None}
    valid = cv2.warpPerspective(np.ones(shape, np.uint8), H, size, flags=cv2.WARP_INVERSE_MAP) > 0
    cols = np.where(valid.mean(0) > 0.98)[0]
    if len(cols) < 100 or len(bev) <= gap:
        out["belt"] = None
        return out
    sl = slice(cols[0], cols[-1] + 1)
    win = cv2.createHanningWindow((sl.stop - sl.start, size[1]), cv2.CV_32F)
    dt = gap * step / fps
    shifts = []
    for i in range(len(bev) - gap):
        (sx, sy), r = cv2.phaseCorrelate(bev[i][:, sl].astype(np.float32), bev[i + gap][:, sl].astype(np.float32), win)
        if r >= min_resp and abs(sy) < 3:
            shifts.append(sx)
    if len(shifts) < 3:
        out["belt"] = None
        return out
    v = np.array(shifts) * RES / dt * 1000.0  # mm/s along +x world
    med = float(np.median(v))
    out["belt"] = {
        "speed_mm_s": abs(med),
        "direction_x_sign": int(np.sign(med)),
        "pairs_used": len(v),
        "pairs_total": len(bev) - gap,
        "iqr_mm_s": [float(np.percentile(np.abs(v), 25)), float(np.percentile(np.abs(v), 75))],
        "dt_s": dt,
        "band_y_m": [float(band[0]), float(band[1])],
        "x_range_m": [float(x0 + sl.start * RES), float(x0 + sl.stop * RES)],
    }
    return out
