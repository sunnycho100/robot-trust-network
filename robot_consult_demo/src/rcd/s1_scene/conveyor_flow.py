"""Conveyor found from motion only: coherent translation of the floor plane in a metric bird's-eye view.

No appearance rule (colour, markings) is used, so it works for any belt / slat / floor-plate conveyor that
carries texture. Per calibrated camera clip:

1. Floor bird's-eye view (BEV): the s0 calibration gives the floor homography z=0 -> image, so frames are
   resampled into a metric top view (2 cm/px) over the floor area the camera sees within `MAX_RANGE_M`.
2. Dense optical flow (Farneback) between BEV frames `DT_S` apart. People are masked out (their floor
   footprint from the pose stage, when given); a per-pixel temporal median over all frame pairs removes
   the remaining transients (walking people, hands, lighting flicker).
3. Belt region: pixels whose median displacement is non-zero, consistent over time and aligned with the
   dominant motion direction (the travel direction); the belt band is the contiguous run of such rows across
   the line, walked outward from the camera (see `motion_map`). Cameras on both sides of the line each
   contribute their reliable near edge (`combine_edges`).
4. Speed: phase correlation of BEV crops of the belt interior (rotated so the travel direction is the image
   x axis, 5 mm/px), median over frame pairs with a clear correlation peak. Only the floor-level surface moves
   at the true speed (objects riding on the belt project faster in a floor BEV), and the belt texture
   dominates the correlation of the belt interior.

If the intake gives `conveyor.polygon_xy` / `direction_xy`, they replace steps 2-3 (speed is still measured).
"""
from __future__ import annotations

import cv2
import numpy as np

MAX_RANGE_M = 6.0     # floor considered visible/usable within this horizontal distance from the camera
FLOW_RES = 0.02       # m/px for the motion map
SPEED_RES = 0.005     # m/px for the speed measurement
DT_S = 2.0            # frame-pair spacing (slow belts move only a few cm per second)
MIN_SPEED = 0.004     # m/s: below this a pixel counts as static
MAX_SPEED = 0.6       # m/s: above this it is not a conveyor surface (people, carts pushed by hand)
NEAR_ALONG_M = 2.5    # speed measured on the belt within this along-line distance of the camera


def floor_to_image(K, R, t) -> np.ndarray:
    """Homography from floor coordinates (x, y, 1) at z = 0 to image pixels."""
    P = np.asarray(K) @ np.hstack([np.asarray(R), np.asarray(t).reshape(3, 1)])
    return P[:, [0, 1, 3]].copy()


def image_to_floor(cam: dict, uv: np.ndarray) -> np.ndarray:
    """Intersect pixel rays with the floor z = 0 (NaN for rays that do not hit the floor in front)."""
    K, R, t = cam["K"], cam["R"], cam["t"]
    C = -R.T @ t
    uv = np.atleast_2d(uv).astype(float)
    d = (R.T @ np.linalg.solve(K, np.c_[uv, np.ones(len(uv))].T)).T
    s = -C[2] / d[:, 2]
    X = C + s[:, None] * d
    X[(s <= 0) | ~np.isfinite(s)] = np.nan
    return X


class BEV:
    """Metric top view with axes (a, b): a = along `axis` (unit xy), b = 90 deg to its left.
    Pixel column u -> a = a0 + u*res, row v -> b = b1 - v*res; world = a*axis + b*perp."""

    def __init__(self, cam: dict, axis, a0, a1, b0, b1, res):
        self.axis = np.asarray(axis, float) / np.linalg.norm(axis)
        self.perp = np.array([-self.axis[1], self.axis[0]])
        self.a0, self.a1, self.b0, self.b1, self.res = a0, a1, b0, b1, res
        A = np.array([[self.axis[0] * res, -self.perp[0] * res, a0 * self.axis[0] + b1 * self.perp[0]],
                      [self.axis[1] * res, -self.perp[1] * res, a0 * self.axis[1] + b1 * self.perp[1]],
                      [0, 0, 1.0]])
        self.H = floor_to_image(cam["K"], cam["R"], cam["t"]) @ A
        self.size = (int(round((a1 - a0) / res)), int(round((b1 - b0) / res)))

    def warp(self, im):
        return cv2.warpPerspective(im, self.H, self.size, flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)

    def to_world(self, u, v):
        a = self.a0 + np.asarray(u, float) * self.res
        b = self.b1 - np.asarray(v, float) * self.res
        return a[..., None] * self.axis + b[..., None] * self.perp

    def to_px(self, xy):
        xy = np.atleast_2d(xy)
        return np.stack([(xy @ self.axis - self.a0) / self.res, (self.b1 - xy @ self.perp) / self.res], 1)


def visible_floor(cam: dict, image_size, max_range: float = MAX_RANGE_M) -> np.ndarray:
    """Floor points (N, 2) seen by the camera within `max_range` m of its foot point."""
    W, H = image_size
    uu, vv = np.meshgrid(np.linspace(0, W - 1, 64), np.linspace(0, H - 1, 36))
    X = image_to_floor(cam, np.c_[uu.ravel(), vv.ravel()])
    C = -cam["R"].T @ cam["t"]
    ok = np.isfinite(X[:, 0]) & (np.linalg.norm(X[:, :2] - C[:2], axis=1) < max_range)
    return X[ok, :2]


def _read_bev(path, bevs: list[BEV], step: int):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out = [[] for _ in bevs]
    idx, k, shape = [], 0, None
    while True:
        if not cap.grab():
            break
        if k % step == 0:
            ok, im = cap.retrieve()
            if ok:
                g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                shape = g.shape
                for b, o in zip(bevs, out):
                    o.append(b.warp(g))
                idx.append(k)
        k += 1
    cap.release()
    return out, np.array(idx), fps, k, shape


def _flow_field(path, bev: BEV, people, max_samples: int):
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    step = max(15, int(np.ceil(total / max_samples)))
    (frames,), idx, fps, n_read, shape = _read_bev(path, [bev], step)
    gap = max(1, int(round(DT_S * fps / step)))
    if len(frames) <= gap:
        return None
    valid = bev.warp(np.full(shape, 255, np.uint8)) > 0
    flows = []
    for i in range(0, len(frames) - gap):
        f = cv2.calcOpticalFlowFarneback(frames[i], frames[i + gap], None, 0.5, 4, 21, 3, 5, 1.1, 0)
        f[~valid] = np.nan
        if people is not None:  # mask every detected person's floor footprint in both frames of the pair
            pf, pxy = people
            for k in (idx[i], idx[i + gap]):
                j = int(np.argmin(np.abs(pf - k)))
                if abs(pf[j] - k) > 15:
                    continue
                for pk in pxy[j]:
                    if np.isfinite(pk).all():
                        u, v = bev.to_px(pk)[0]
                        if -1e4 < u < 1e4 and -1e4 < v < 1e4:
                            cv2.circle(f, (int(u), int(v)), int(0.6 / bev.res), (np.nan, np.nan), -1)
        flows.append(f.astype(np.float16))
    dt = gap * step / fps
    F = np.stack(flows).astype(np.float32) * bev.res / dt          # (P, H, W, 2) m/s in BEV pixel axes
    with np.errstate(all="ignore"):
        med = np.nanmedian(F, 0)
        dev = np.linalg.norm(F - med[None], axis=-1)
        sp = np.linalg.norm(med, axis=-1)
        cons = np.nanmean(dev < np.maximum(0.5 * sp[None], 0.002), 0)
    # BEV pixel axes -> (along, across) world components: columns = +axis, rows = -perp
    va, vb = med[..., 0], -med[..., 1]
    return {"va": va, "vb": vb, "cons": cons, "valid": valid, "pairs": len(flows), "dt": dt}


def motion_map(path, cam: dict, image_size, people=None, max_samples: int = 120, ratio_tol: float = 0.2,
               max_gap_m: float = 2.5, min_frac: float = 0.5) -> dict | None:
    """Belt band, travel direction and a coarse flow speed for one camera clip (steps 1-3).

    people: optional (frame indices (M,), floor xy (M,P,2)) of detected people, masked out of the flow.
    Rows of the BEV across the line are "strong" when most of their visible length moves along the travel
    direction (a conveyor surface is continuous along the line; people and loads are local). Walking outward from
    the camera, the belt starts at the first strong run and absorbs further strong runs that move at the same
    speed (within `ratio_tol`) across static-looking gaps of up to `max_gap_m` (untextured belt areas give no
    flow; two co-moving bands with nothing static-but-textured between them are one conveyor). It ends at static
    floor or at a run with a different apparent speed -- e.g. the floor-BEV projection of loads riding on the
    belt, which moves faster by Cz/(Cz-h) for a point at height h. The edge nearest the camera is the reliable
    one; the far edge may be cut short by such loads (reported in `far_edge_stop`)."""
    pts = visible_floor(cam, image_size)
    if len(pts) < 50:
        return None
    lo, hi = np.percentile(pts, 1, 0), np.percentile(pts, 99, 0)
    axis = np.array([1.0, 0.0])
    for _ in range(2):
        a_ = pts @ axis
        b_ = pts @ np.array([-axis[1], axis[0]])
        bev = BEV(cam, axis, a_.min(), a_.max(), b_.min(), b_.max(), FLOW_RES)
        ff = _flow_field(path, bev, people, max_samples)
        if ff is None:
            return None
        sp = np.hypot(ff["va"], ff["vb"])
        moving = (np.nan_to_num(sp) > MIN_SPEED) & (np.nan_to_num(sp) < MAX_SPEED) & (np.nan_to_num(ff["cons"]) > 0.5)
        if moving.sum() < 200:
            return None
        dv = np.array([np.nanmedian(ff["va"][moving]), np.nanmedian(ff["vb"][moving])])
        if abs(np.degrees(np.arctan2(dv[1], dv[0]))) < 5 or _ == 1:
            break
        axis = dv[0] * bev.axis + dv[1] * bev.perp       # re-align the BEV with the travel direction
        axis /= np.linalg.norm(axis)
    sign = 1.0 if dv[0] >= 0 else -1.0
    d = sign * bev.axis
    va = sign * ff["va"]
    along_ok = moving & (np.nan_to_num(va) > MIN_SPEED) & (np.abs(np.nan_to_num(ff["vb"])) < 0.5 * np.abs(np.nan_to_num(va)))
    nrow = va.shape[0]
    frac = along_ok.sum(1) / np.maximum(ff["valid"].sum(1), 1)
    row_v = np.array([np.median(va[r][along_ok[r]]) if along_ok[r].sum() >= 5 else np.nan for r in range(nrow)])
    row_b = bev.b1 - (np.arange(nrow) + 0.5) * bev.res                 # across coordinate of each row
    C = -cam["R"].T @ cam["t"]
    cb = float(C[:2] @ bev.perp)
    order = np.argsort(np.abs(row_b - cb))                             # rows from near to far
    # "strong" rows move along most of their visible length: a conveyor surface is continuous along the line,
    # whereas people and loads riding on it are local (low fraction)
    strong = (frac[order] > min_frac) & np.isfinite(row_v[order])
    runs, i = [], 0
    while i < nrow:
        if strong[i]:
            j = i
            while j < nrow and strong[j]:
                j += 1
            if (j - i) * bev.res >= 0.06:
                runs.append((i, j - 1, float(np.median(row_v[order[i:j]]))))
            i = j
        else:
            i += 1
    if not runs:
        return None
    first, last, v_near = runs[0]
    stop = "end of visible floor"
    for (i0, i1, v) in runs[1:]:
        if (i0 - last) * bev.res > max_gap_m:
            stop = "static floor"
            break
        if abs(v / v_near - 1) > ratio_tol:
            stop = "surface moving at a different apparent speed (objects riding on the belt / other conveyor)"
            break
        last = i1
    band_rows = order[first:last + 1]
    bb = row_b[band_rows]
    near_edge = float(row_b[order[first]])
    far_edge = float(row_b[order[last]])
    cols = np.where(along_ok[band_rows].any(0))[0]
    aa = bev.a0 + (cols + 0.5) * bev.res
    a_lo, a_hi = (float(np.percentile(aa, 2)), float(np.percentile(aa, 98))) if len(aa) else (bev.a0, bev.a1)
    across = [float(bb.min()), float(bb.max())]
    if sign < 0:  # express in the frame of d (perp flips with the axis)
        across = [-across[1], -across[0]]
        near_edge, far_edge = -near_edge, -far_edge
        a_lo, a_hi = -a_hi, -a_lo
    return {"direction_xy": [float(x) for x in d], "across_m": across, "along_m": [a_lo, a_hi],
            "near_edge_m": near_edge, "far_edge_m": far_edge, "far_edge_stop": stop,
            "flow_speed_mm_s": v_near * 1000, "pairs": ff["pairs"], "dt_s": ff["dt"],
            "camera_across_m": cb * sign}


def combine_edges(maps: list[dict]) -> tuple[list[float], str]:
    """Belt band across the line from several camera clips: every camera's near edge is reliable; a far edge is
    used only where no camera looks from that side."""
    lo_near = [m["near_edge_m"] for m in maps if m["near_edge_m"] <= m["far_edge_m"]]
    hi_near = [m["near_edge_m"] for m in maps if m["near_edge_m"] > m["far_edge_m"]]
    lo = float(np.median(lo_near)) if lo_near else float(np.median([m["far_edge_m"] for m in maps if m["near_edge_m"] > m["far_edge_m"]]))
    hi = float(np.median(hi_near)) if hi_near else float(np.median([m["far_edge_m"] for m in maps if m["near_edge_m"] <= m["far_edge_m"]]))
    how = ("near edges of cameras on both sides" if lo_near and hi_near else
           "near edge of the camera side + far edge where moving floor ends (single viewing side)")
    return [lo, hi], how


def belt_speed(path, cam: dict, direction_xy, across, along, max_samples: int = 240, min_resp: float = 0.2,
               shrink: float = 0.25) -> dict | None:
    """Phase-correlation speed of the belt interior in a BEV rotated so that travel = +columns (step 4)."""
    d = np.asarray(direction_xy, float)
    d = d / np.linalg.norm(d)
    b0, b1 = across[0] + shrink, across[1] - shrink
    if b1 - b0 < 0.2:
        b0, b1 = across
    a0, a1 = along
    ca = float((-cam["R"].T @ cam["t"])[:2] @ d)       # use the belt within NEAR_ALONG_M of the camera (best resolution)
    a0, a1 = max(a0, ca - NEAR_ALONG_M), min(a1, ca + NEAR_ALONG_M)
    bev = BEV(cam, d, a0, a1, b0, b1, SPEED_RES)
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    step = max(15, int(np.ceil(total / max_samples)))
    (frames,), idx, fps, n_read, shape = _read_bev(path, [bev], step)
    gap = max(1, int(round(DT_S * fps / step)))
    valid = bev.warp(np.full(shape, 255, np.uint8)) > 0
    cols = np.where(valid.mean(0) > 0.98)[0]
    if len(cols) < 100 or len(frames) <= gap:
        return None
    sl = slice(cols[0], cols[-1] + 1)
    win = cv2.createHanningWindow((sl.stop - sl.start, bev.size[1]), cv2.CV_32F)
    dt = gap * step / fps
    shifts = []
    for i in range(len(frames) - gap):
        (sx, sy), r = cv2.phaseCorrelate(frames[i][:, sl].astype(np.float32), frames[i + gap][:, sl].astype(np.float32), win)
        if r >= min_resp and abs(sy) < 3:
            shifts.append(sx)
    if len(shifts) < 3:
        return None
    v = np.array(shifts) * SPEED_RES / dt * 1000.0
    med = float(np.median(v))
    return {"speed_mm_s": abs(med), "direction_sign": int(np.sign(med)), "pairs_used": len(v),
            "pairs_total": len(frames) - gap, "iqr_mm_s": [float(np.percentile(np.abs(v), 25)), float(np.percentile(np.abs(v), 75))],
            "dt_s": dt, "frames_read": int(n_read), "fps": float(fps)}


def polygon(direction_xy, across, along) -> list[list[float]]:
    d = np.asarray(direction_xy, float)
    d = d / np.linalg.norm(d)
    p = np.array([-d[1], d[0]])
    pts = [a * d + b * p for a, b in [(along[0], across[0]), (along[1], across[0]), (along[1], across[1]), (along[0], across[1])]]
    return [[round(float(x), 3), round(float(y), 3)] for x, y in pts]


def from_polygon(poly_xy, direction_xy=None) -> tuple[np.ndarray, list[float], list[float]]:
    """(direction, across, along) of an intake-given belt polygon; direction defaults to its long axis."""
    P = np.asarray(poly_xy, float)
    if direction_xy is None:
        (cx, cy), (w, h), ang = cv2.minAreaRect(P.astype(np.float32))
        a = np.radians(ang if w >= h else ang + 90)
        direction_xy = [np.cos(a), np.sin(a)]
    d = np.asarray(direction_xy, float)
    d = d / np.linalg.norm(d)
    p = np.array([-d[1], d[0]])
    return d, [float((P @ p).min()), float((P @ p).max())], [float((P @ d).min()), float((P @ d).max())]
