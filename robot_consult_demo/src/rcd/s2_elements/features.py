"""Kinematic feature extraction from full-body joints (metres, z-up) resampled to an analysis rate.

Signals (all derived explicitly from joint positions):
  pelvis_v      horizontal pelvis speed  [m/s]   (walking vs. at a station)
  wrist_v[R/L]  wrist speed              [m/s]   (manipulation activity / dwell)
  reach[R/L]    horizontal wrist distance from pelvis [m]  (reach distance class, MTM-UAS)
  wrist_z[R/L]  wrist height above floor [m]
  hand_dist     distance between wrists  [m]     (bimanual handling, loading a screw on the bit)
  trunk_flex    angle Hips->Neck vs vertical [deg]
  trunk_twist   shoulder line vs hip line, horizontal plane [deg]
  trunk_flex_sag  Hips->Neck flexion projected on the pelvis sagittal plane (+ forward) [deg]
  lateral_bend  Hips->Neck inclination in the pelvis frontal plane [deg]
  upper_arm_elev[R/L] shoulder->elbow vs downward vertical [deg]
  elbow_above_shoulder[R/L], hand_above_head[R/L]  booleans
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d

RATE_HZ = 10.0          # analysis rate (60 fps mocap averaged in blocks of 6 frames)
SMOOTH_S = 0.3          # moving-average window before differentiation


def _ang(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    c = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def extract(J: np.ndarray, names: list[str], fps: float, rate: float = RATE_HZ) -> dict:
    I = {n: i for i, n in enumerate(names)}
    w = max(1, int(round(SMOOTH_S * fps)))
    Js = uniform_filter1d(J, w, axis=0, mode="nearest")
    step = max(1, int(round(fps / rate)))
    T = (len(Js) // step) * step
    D = Js[:T].reshape(T // step, step, *Js.shape[1:]).mean(1)       # block average -> rate
    dt = step / fps
    P = lambda k: D[:, I[k]]
    hips, neck = P("Hips"), P("Neck")
    up = np.array([0.0, 0.0, 1.0])
    f: dict = {"t": (np.arange(len(D)) + 0.5) * dt, "dt": dt, "pelvis_xy": hips[:, :2].copy(), "pelvis_z": hips[:, 2].copy()}
    f["pelvis_v"] = np.linalg.norm(np.gradient(hips[:, :2], dt, axis=0), axis=1)
    trunk = neck - hips
    f["trunk_flex"] = _ang(trunk, up)
    hip_lr = P("LeftHip") - P("RightHip")
    sh_lr = P("LeftShoulder") - P("RightShoulder")
    h2, s2 = hip_lr.copy(), sh_lr.copy()
    h2[:, 2] = 0
    s2[:, 2] = 0
    f["trunk_twist"] = np.degrees(np.arctan2(np.cross(h2, s2)[:, 2], (h2 * s2).sum(1)))
    lr = h2 / (np.linalg.norm(h2, axis=1, keepdims=True) + 1e-12)
    fwd = np.cross(lr, up)                                   # pelvis forward axis (horizontal)
    # EAWS / REBA define trunk flexion in the sagittal plane and lateral bending in the frontal plane:
    f["trunk_flex_sag"] = np.degrees(np.arctan2((trunk * fwd).sum(1), trunk[:, 2]))
    f["lateral_bend"] = np.degrees(np.arctan2((trunk * lr).sum(1), trunk[:, 2]))
    feet_z = np.minimum(P("LeftAnkle")[:, 2], P("RightAnkle")[:, 2]) if "LeftAnkle" in I else np.zeros(len(D))
    f["floor_z"] = float(np.percentile(feet_z, 5)) - 0.08   # ankle joint is ~8 cm above the sole
    for s in ("R", "L"):
        side = "Right" if s == "R" else "Left"
        wr, el, sh = P(f"{side}Wrist"), P(f"{side}Elbow"), P(f"{side}Shoulder")
        f[f"wrist_{s}"] = wr.copy()
        f[f"wrist_v{s}"] = np.linalg.norm(np.gradient(wr, dt, axis=0), axis=1)
        rel = wr - hips
        f[f"reach{s}"] = np.linalg.norm(rel[:, :2], axis=1)
        f[f"wrist_z{s}"] = wr[:, 2] - f["floor_z"]
        f[f"upper_arm_elev{s}"] = _ang(el - sh, -up)
        f[f"elbow_above_shoulder{s}"] = el[:, 2] >= sh[:, 2]
        f[f"hand_above_head{s}"] = wr[:, 2] >= P("Head")[:, 2]
    f["hand_dist"] = np.linalg.norm(f["wrist_R"] - f["wrist_L"], axis=1)
    f["hand_dz"] = np.abs(f["wrist_R"][:, 2] - f["wrist_L"][:, 2])
    f["act"] = np.maximum(f["wrist_vR"], f["wrist_vL"])
    f["dwell"] = np.minimum(f["wrist_vR"], f["wrist_vL"])
    f["reach_max"] = np.maximum(f["reachR"], f["reachL"])
    f["reach_min"] = np.minimum(f["reachR"], f["reachL"])
    f["n"] = len(D)
    return f


def work_anchor(f: dict, v_station: float = 0.20, min_dwell_s: float = 4.0) -> np.ndarray:
    """Time-local work anchor (T,2): centroid of the nearest-in-time *long dwell* (stationary bout >= 4 s).

    Assembly at the door happens in long dwells; picks at the cart / tool rack are short stops between walks.
    Using the nearest long dwell (not one global point) follows the carrier when the conveyor moves it and the
    operator re-positions. Falls back to the median stationary position if the clip has no long dwell."""
    st = f["pelvis_v"] < v_station
    n = f["n"]
    bouts = []
    i = 0
    while i < n:
        if st[i]:
            j = i
            while j < n and st[j]:
                j += 1
            bouts.append([i, j])
            i = j
        else:
            i += 1
    merged = []
    for b in bouts:   # bridge sub-0.5 s interruptions (weight shifts)
        if merged and b[0] - merged[-1][1] <= int(0.5 * RATE_HZ):
            merged[-1][1] = b[1]
        else:
            merged.append(b)
    longs = [b for b in merged if (b[1] - b[0]) >= min_dwell_s * RATE_HZ]
    f["dwells"] = [(round(a / RATE_HZ, 1), round(b / RATE_HZ, 1)) for a, b in longs]
    if not longs:
        return np.tile(np.median(f["pelvis_xy"][st] if st.any() else f["pelvis_xy"], axis=0), (n, 1))
    cents = np.array([np.median(f["pelvis_xy"][a:b], axis=0) for a, b in longs])
    t = np.arange(n)
    dist_t = np.stack([np.where(t < a, a - t, np.where(t >= b, t - b + 1, 0)) for a, b in longs])
    return cents[np.argmin(dist_t, axis=0)]


def path_length(xy: np.ndarray, rate: float = RATE_HZ, min_step_speed: float = 0.1) -> float:
    """Pelvis travel [m] ignoring postural sway (steps slower than min_step_speed are dropped)."""
    if len(xy) < 2:
        return 0.0
    d = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    return float(d[d * rate > min_step_speed].sum())
