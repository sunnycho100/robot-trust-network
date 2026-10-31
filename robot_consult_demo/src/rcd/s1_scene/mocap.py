"""Mocap (Xsens BVH) analysis: dwell segmentation, work zone, line side, contact points, world registration.

All thresholds are kinematic conventions for 60 Hz full-body mocap, not tuned per cycle:
* pelvis "stationary" = horizontal speed < 0.20 m/s (normal walking is 1.0-1.5 m/s)
* hand "manipulation dwell" = wrist speed < 0.12 m/s for >= 0.4 s while the worker stands in the work zone
"""
from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.ndimage import uniform_filter1d
from scipy.spatial import ConvexHull

PELVIS_STILL = 0.20     # m/s
LINESIDE_STILL = 0.30   # m/s, looser: picking at the line side is often brief
HAND_STILL = 0.12       # m/s
MIN_STAND_S = 1.0
MIN_HAND_S = 0.4
WORK_RADIUS = 0.6       # m around the work-zone mode
CONTACT_CLUSTER = 0.12  # m


def _segments(mask: np.ndarray, min_len: int, fill: int = 0) -> list[tuple[int, int]]:
    m = mask.astype(bool).copy()
    if fill:  # close short gaps
        idx = np.where(m)[0]
        for a, b in zip(idx[:-1], idx[1:]):
            if 1 < b - a <= fill:
                m[a:b] = True
    out, i, n = [], 0, len(m)
    while i < n:
        if m[i]:
            j = i
            while j < n and m[j]:
                j += 1
            if j - i >= min_len:
                out.append((i, j))
            i = j
        else:
            i += 1
    return out


def hull_polygon(pts: np.ndarray, buffer: float = 0.1) -> list[list[float]]:
    """Convex hull of xy points, grown by `buffer` (circle samples) so even 1-2 points give an area."""
    pts = np.asarray(pts, float).reshape(-1, 2)
    ang = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    ring = np.stack([np.cos(ang), np.sin(ang)], 1) * buffer
    P = (pts[:, None, :] + ring[None]).reshape(-1, 2)
    h = ConvexHull(P)
    return [[round(float(x), 3), round(float(y), 3)] for x, y in P[h.vertices]]


def analyse(J: np.ndarray, names: list[str], fps: float) -> dict:
    """Segment one cycle in the raw mocap frame."""
    ix = {n: i for i, n in enumerate(names)}
    T = len(J)
    w = max(1, int(round(0.25 * fps)))
    Js = uniform_filter1d(J, size=w, axis=0, mode="nearest")
    vel = np.gradient(Js, 1.0 / fps, axis=0)
    pel = Js[:, ix["Hips"]]
    sp = np.linalg.norm(vel[:, ix["Hips"], :2], axis=1)

    # facing direction: z x (right hip - left hip), sign validated against walking direction
    right = Js[:, ix["RightHip"], :2] - Js[:, ix["LeftHip"], :2]
    right += Js[:, ix["RightShoulder"], :2] - Js[:, ix["LeftShoulder"], :2]
    fwd = np.stack([-right[:, 1], right[:, 0]], 1)
    fwd /= np.linalg.norm(fwd, axis=1, keepdims=True) + 1e-9
    walking = sp > 0.6
    sign_check = float(np.mean(np.sum(fwd[walking] * vel[walking, ix["Hips"], :2], 1) > 0)) if walking.any() else None
    if sign_check is not None and sign_check < 0.5:
        fwd = -fwd

    still = sp < PELVIS_STILL
    stands = _segments(still, int(MIN_STAND_S * fps), fill=int(0.3 * fps))
    # work-zone centre = time-weighted mode of stationary pelvis positions
    P = pel[still, :2]
    lo = P.min(0) - 0.5
    H, xe, ye = np.histogram2d(P[:, 0], P[:, 1], bins=[np.arange(lo[0], P[:, 0].max() + 0.5, 0.05),
                                                      np.arange(lo[1], P[:, 1].max() + 0.5, 0.05)])
    from scipy.ndimage import gaussian_filter
    Hs = gaussian_filter(H, 3)  # 0.15 m kernel
    i, j = np.unravel_index(np.argmax(Hs), Hs.shape)
    centre = np.array([xe[i] + 0.025, ye[j] + 0.025])
    near = np.linalg.norm(pel[:, :2] - centre, axis=1) < WORK_RADIUS
    work = still & near
    work_segs = _segments(work, int(MIN_STAND_S * fps), fill=int(0.3 * fps))
    work = np.zeros(T, bool)
    for a, b in work_segs:
        work[a:b] = True

    # line side = farthest place (from the work zone) where the worker stops, however briefly
    cand = _segments(sp < LINESIDE_STILL, int(0.3 * fps), fill=int(0.2 * fps))
    best, far = None, 0.0
    for a, b in cand:
        d = np.linalg.norm(np.median(pel[a:b, :2], 0) - centre)
        if d > far:
            best, far = (a, b), d
    line = np.zeros(T, bool)
    line_seg = None
    if best is not None and far > 0.8:
        c = np.median(pel[best[0]:best[1], :2], 0)
        for a, b in cand:  # every stop near that place
            if np.linalg.norm(np.median(pel[a:b, :2], 0) - c) < 0.5:
                line[a:b] = True
        line_seg = best

    # manipulation dwells of each hand while standing in the work zone
    contacts = []
    for hand in ("RightWrist", "LeftWrist"):
        hs = np.linalg.norm(vel[:, ix[hand]], axis=1)
        for a, b in _segments((hs < HAND_STILL) & work, int(MIN_HAND_S * fps), fill=int(0.1 * fps)):
            contacts.append({"hand": hand, "t0": a / fps, "t1": b / fps,
                             "p": np.median(Js[a:b, ix[hand]], 0)})
    supply = []
    if line.any():
        for hand in ("RightWrist", "LeftWrist"):
            hs = np.linalg.norm(vel[:, ix[hand]], axis=1)
            segs = _segments((hs < 0.25) & line, int(0.2 * fps), fill=int(0.1 * fps))
            if not segs:  # fall back to the hand position at the slowest pelvis moment of the stop
                k = int(np.argmin(np.where(line, sp, np.inf)))
                segs = [(k, k + 1)]
            for a, b in segs:
                supply.append({"hand": hand, "t0": a / fps, "t1": b / fps, "p": np.median(Js[a:b, ix[hand]], 0)})

    feet = [ix[n] for n in ("LeftAnkle", "RightAnkle", "LeftToe", "RightToe") if n in ix]
    return {
        "T": T, "fps": fps, "Js": Js, "pelvis": pel, "speed": sp, "fwd": fwd, "facing_sign_check": sign_check,
        "work": work, "line": line, "line_seg": line_seg, "line_dist_m": far, "work_centre": centre,
        "contacts": contacts, "supply": supply, "feet_idx": feet,
        "stands": stands,
    }


def cluster_points(items: list[dict], radius: float = CONTACT_CLUSTER) -> list[dict]:
    if not items:
        return []
    P = np.array([c["p"] for c in items])
    lab = fcluster(linkage(P, "average"), radius, "distance") if len(P) > 1 else np.array([1])
    out = []
    for l in np.unique(lab):
        m = lab == l
        dw = np.array([items[k]["t1"] - items[k]["t0"] for k in np.where(m)[0]])
        out.append({
            "p": (P[m] * dw[:, None]).sum(0) / dw.sum(),
            "dwell_s": float(dw.sum()),
            "n": int(m.sum()),
            "hands": sorted({items[k]["hand"] for k in np.where(m)[0]}),
            "first_t_s": float(min(items[k]["t0"] for k in np.where(m)[0])),
        })
    return sorted(out, key=lambda c: -c["dwell_s"])


def yaw_matrix(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def register(an: dict, target_facing: np.ndarray, anchor_xy: np.ndarray) -> np.ndarray:
    """4x4 world_from_mocap: yaw so the mean work-zone facing equals `target_facing`, then translate so the
    dwell-weighted contact centroid (or the point 0.5 m in front of the work-zone pelvis) is at `anchor_xy`."""
    f = an["fwd"][an["work"]].mean(0) if an["work"].any() else an["fwd"].mean(0)
    theta = np.arctan2(target_facing[1], target_facing[0]) - np.arctan2(f[1], f[0])
    Rz = yaw_matrix(theta)
    cl = cluster_points(an["contacts"])
    if cl:
        w = np.array([c["dwell_s"] for c in cl])
        a = (np.array([c["p"] for c in cl]) * w[:, None]).sum(0) / w.sum()
    else:
        a = np.r_[an["work_centre"] + 0.5 * f / (np.linalg.norm(f) + 1e-9), 1.0]
    M = np.eye(4)
    M[:3, :3] = Rz
    M[:2, 3] = anchor_xy - (Rz @ a)[:2]
    return M


def apply(M: np.ndarray, X: np.ndarray) -> np.ndarray:
    return X @ M[:3, :3].T + M[:3, 3]
