"""Work-element segmentation from mocap.

Two decoders share one set of frame-level evidence scores (log-likelihood-like, from features.py):

1. Standard-work forced alignment (inside an annotated subgoal span). The CarDA activity list gives the
   ordered element script (taxonomy.script_from_activities); optional WALK states are inserted between
   steps. A left-to-right hidden semi-Markov model with explicit minimum durations (each state expanded to a
   chain of `min_frames` sub-states, Rabiner 1989 sec. IV-D) is decoded with the Viterbi algorithm, so the
   element order comes from the written standard work and every boundary time comes from the motion.
2. Ergodic HMM (spans with no script): all element codes reachable from each other with a switching penalty
   (sticky transitions = hysteresis), then a minimum-segment-length merge.

Evidence rules (global constants, never tuned per cycle):
  WALK            pelvis speed > 0.30 m/s
  station states  pelvis speed < 0.30 m/s, plus
  GET_PART/TOOL   pelvis slowing (< 0.45 m/s) away from the work anchor (> 0.60 m) -> line-side zone
                  (cart, tool rack); the anchor is the nearest-in-time long dwell (features.work_anchor)
  INSERT          at the work anchor, hands close together and level (bimanual handling of one part)
  REACH_FASTENER  at the work anchor, a hand close to the body (pouch) or hands together (screw onto bit), moving
  FASTEN          at the work anchor, one hand dwelling (< 0.08 m/s) with extended reach (> 0.33 m)
  INSPECT         at the work anchor, slow hands, upright, short reach
  IDLE            both hands still (< 0.05 m/s)
"""
from __future__ import annotations

import numpy as np

from .features import RATE_HZ, path_length

V_WALK = 0.30
D_AWAY = 0.60
MIN_DUR_S = {"WALK": 0.6, "GET_PART": 0.8, "GET_TOOL": 0.6, "INSERT": 1.0, "REACH_FASTENER": 0.6,
             "FASTEN": 0.8, "INSPECT": 0.8, "IDLE": 1.0, "SETUP": 1.0}
WALK_ENTRY_PENALTY = 1.5      # log-score cost of inserting an optional walk
SKIP_PENALTY = 6.0            # log-score cost of skipping one scripted step (template step not performed)
MAX_SKIP = 3
ERGODIC_SWITCH_PENALTY = 4.0


def _ls(x):  # log-sigmoid, numerically safe
    return -np.logaddexp(0.0, -x)


def hi(x, thr, w):
    return _ls((np.asarray(x) - thr) / w)


def lo(x, thr, w):
    return _ls((thr - np.asarray(x)) / w)


def evidence(f: dict, anchor: np.ndarray) -> dict[str, np.ndarray]:
    d_anchor = np.linalg.norm(f["pelvis_xy"] - anchor, axis=1)   # anchor: (T,2) time-local
    f["d_anchor"] = d_anchor
    walk = hi(f["pelvis_v"], V_WALK, 0.06)
    stay = lo(f["pelvis_v"], V_WALK, 0.06)
    away = hi(d_anchor, D_AWAY, 0.12)
    near = lo(d_anchor, D_AWAY, 0.12)
    moving = hi(f["act"], 0.12, 0.04)
    slow = lo(f["pelvis_v"], 0.45, 0.08)          # picks happen while the operator slows down at the cart
    sym = lo(f["hand_dz"], 0.15, 0.05)             # both hands at the same height (bimanual part handling)
    E = {
        "WALK": walk,
        "GET_PART": slow + away + 0.5 * hi(f["reach_max"], 0.30, 0.06),
        "GET_TOOL": np.logaddexp(slow + away, stay + near + hi(f["reach_max"], 0.40, 0.06)) - np.log(2),
        "INSERT": stay + near + lo(f["hand_dist"], 0.40, 0.08) + sym,
        "REACH_FASTENER": stay + near + np.logaddexp(lo(f["reach_min"], 0.22, 0.04), lo(f["hand_dist"], 0.22, 0.05))
                          - np.log(2) + moving,
        "FASTEN": stay + near + lo(f["dwell"], 0.08, 0.03) + hi(f["reach_max"], 0.33, 0.05) + 0.5 * (-sym),
        "INSPECT": stay + near + lo(f["act"], 0.12, 0.04) + lo(f["trunk_flex"], 20, 5) + lo(f["reach_max"], 0.33, 0.05),
        "IDLE": stay + lo(f["act"], 0.05, 0.02) + lo(f["reach_max"], 0.30, 0.05),
        "SETUP": np.logaddexp(slow + hi(f["reach_max"], 0.30, 0.06), stay + moving) - np.log(2),
    }
    return E


def posterior(E: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    codes = list(E)
    M = np.stack([E[c] for c in codes])
    M = M - M.max(0)
    P = np.exp(M)
    P /= P.sum(0)
    return {c: P[i] for i, c in enumerate(codes)}


# ----------------------------------------------------------------------------------------------- decoders
def forced_align(E: dict[str, np.ndarray], script: list[dict], t0: int, t1: int) -> list[tuple[dict, int, int]]:
    """Viterbi alignment of the standard-work script to frames [t0, t1).

    Expanded segments: W0 S1 W1 S2 ... Sn Wn, where W are optional walks and S the scripted steps. A step can be
    skipped (template step not performed in this cycle) at a cost of SKIP_PENALTY per skipped step, up to
    MAX_SKIP consecutive steps (profile-HMM deletion states). Returns [(segment, start_frame, end_frame)]."""
    segs = [{"code": "WALK", "optional": True, "step": None}]
    for s in script:
        segs.append({"code": s["code"], "optional": False, "step": s})
        segs.append({"code": "WALK", "optional": True, "step": None})
    T = t1 - t0
    need = sum(int(np.ceil(MIN_DUR_S[g["code"]] * RATE_HZ)) for g in segs if not g["optional"])
    scale = min(1.0, 0.8 * T / max(1, need))
    for g in segs:
        g["m"] = max(1, int(np.floor(MIN_DUR_S[g["code"]] * RATE_HZ * scale)))
    first_of, last_of, st_seg = [], [], []
    for j, g in enumerate(segs):
        first_of.append(len(st_seg))
        st_seg += [j] * g["m"]
        last_of.append(len(st_seg) - 1)
    S = len(st_seg)
    st_seg = np.array(st_seg)
    em = np.stack([E[g["code"]][t0:t1] for g in segs])[st_seg]          # (S, T)
    NEG = -1e18
    self_ok = np.zeros(S, bool)
    self_ok[last_of] = True
    entries: list[list[tuple[int, float]]] = [[] for _ in range(S)]
    steps = [j for j, g in enumerate(segs) if not g["optional"]]           # 1, 3, 5, ...

    def preds_of_step(j: int) -> list[tuple[int, float]]:
        """Predecessor last-sub-states for entering step segment j (skipping k earlier steps costs k*SKIP)."""
        out = []
        for k in range(0, MAX_SKIP + 1):
            jj = j - 2 * k                     # step index after skipping k steps
            if jj - 1 >= 0:
                out.append((last_of[jj - 1], -SKIP_PENALTY * k))          # from walk W before step jj
            if jj - 2 >= 0:
                out.append((last_of[jj - 2], -SKIP_PENALTY * k))          # from step jj-1 directly
        return out

    for j, g in enumerate(segs):
        fs = first_of[j]
        if g["optional"]:
            if j >= 1:
                entries[fs].append((last_of[j - 1], -WALK_ENTRY_PENALTY))
        else:
            entries[fs] += preds_of_step(j)
        for k in range(1, g["m"]):
            entries[fs + k].append((fs + k - 1, 0.0))
    M = max(len(e) for e in entries)
    Pm = np.full((M, S), -1)
    Cm = np.zeros((M, S))
    for s_, lst in enumerate(entries):
        for m_, (p, c) in enumerate(lst):
            Pm[m_, s_], Cm[m_, s_] = p, c
    delta = np.full(S, NEG)
    delta[first_of[0]] = em[first_of[0], 0]
    for k, j in enumerate(steps[:MAX_SKIP + 1]):
        delta[first_of[j]] = em[first_of[j], 0] - SKIP_PENALTY * k
    back = np.zeros((T, S), np.int32)
    idx = np.arange(S)
    valid = Pm >= 0
    for t in range(1, T):
        best = np.where(self_ok, delta, NEG)
        arg = idx.copy()
        cand = np.where(valid, delta[np.maximum(Pm, 0)] + Cm, NEG)
        mi = np.argmax(cand, axis=0)
        cb = cand[mi, idx]
        better = cb > best
        best = np.where(better, cb, best)
        arg = np.where(better, Pm[mi, idx], arg)
        delta = best + em[:, t]
        back[t] = arg
    # end: last step or its trailing walk, or skip up to MAX_SKIP final steps
    end_c = {last_of[-1]: 0.0}
    for k in range(0, MAX_SKIP + 1):
        jj = len(segs) - 2 - 2 * k
        if jj >= 1:
            end_c.setdefault(last_of[jj], -SKIP_PENALTY * k)
            end_c.setdefault(last_of[jj + 1], -SKIP_PENALTY * k)
    s = max(end_c, key=lambda e: delta[e] + end_c[e])
    path = np.zeros(T, np.int32)
    for t in range(T - 1, -1, -1):
        path[t] = s
        s = back[t, s]
    seg_path = st_seg[path]
    out = []
    a = 0
    for t in range(1, T + 1):
        if t == T or seg_path[t] != seg_path[a]:
            out.append((segs[int(seg_path[a])], t0 + a, t0 + t))
            a = t
    return out


def ergodic(E: dict[str, np.ndarray], t0: int, t1: int, codes: list[str] | None = None) -> list[tuple[str, int, int]]:
    codes = codes or ["WALK", "GET_PART", "INSERT", "REACH_FASTENER", "FASTEN", "INSPECT", "IDLE"]
    M = np.stack([E[c][t0:t1] for c in codes])
    K, T = M.shape
    delta = M[:, 0].copy()
    back = np.zeros((T, K), np.int32)
    for t in range(1, T):
        stay = delta
        j = int(np.argmax(delta))
        sw = delta[j] - ERGODIC_SWITCH_PENALTY
        arg = np.where(stay >= sw, np.arange(K), j)
        delta = np.maximum(stay, sw) + M[:, t]
        back[t] = arg
    k = int(np.argmax(delta))
    path = np.zeros(T, np.int32)
    for t in range(T - 1, -1, -1):
        path[t] = k
        k = back[t, k]
    segs = []
    a = 0
    for t in range(1, T + 1):
        if t == T or path[t] != path[a]:
            segs.append([codes[path[a]], t0 + a, t0 + t])
            a = t
    # minimum segment length: merge too-short runs into the longer neighbour
    changed = True
    while changed and len(segs) > 1:
        changed = False
        for i, (c, a, b) in enumerate(segs):
            if (b - a) < MIN_DUR_S[c] * RATE_HZ * 0.6:
                if i == 0:
                    j = 1
                elif i == len(segs) - 1:
                    j = i - 1
                else:
                    j = i - 1 if (segs[i - 1][2] - segs[i - 1][1]) >= (segs[i + 1][2] - segs[i + 1][1]) else i + 1
                lo_, hi_ = min(segs[i][1], segs[j][1]), max(segs[i][2], segs[j][2])
                segs[j] = [segs[j][0], lo_, hi_]
                segs.pop(i)
                changed = True
                break
        # merge equal neighbours
        k = 1
        while k < len(segs):
            if segs[k][0] == segs[k - 1][0]:
                segs[k - 1][2] = segs[k][2]
                segs.pop(k)
            else:
                k += 1
    return [tuple(s) for s in segs]


# ----------------------------------------------------------------------------------------------- helpers
def refine_boundary(f: dict, t_s: float, window_s: float = 1.5) -> float:
    """Snap an annotated subgoal boundary to the nearest natural pause (min of hand + pelvis speed)."""
    k = int(round(t_s * RATE_HZ))
    w = int(window_s * RATE_HZ)
    a, b = max(1, k - w), min(f["n"] - 1, k + w + 1)
    if b <= a:
        return t_s
    cost = f["act"][a:b] + f["pelvis_v"][a:b] + 0.02 * np.abs(np.arange(a, b) - k) / RATE_HZ
    return float((a + int(np.argmin(cost))) / RATE_HZ)


def carve_idle(f: dict, code: str, a: int, b: int, min_s: float = 3.0) -> list[tuple[str, int, int]]:
    """Split waiting out of a long scripted element: runs >= 3 s with both hands still (< 0.05 m/s) and the pelvis
    still become IDLE (therblig UD/AD), e.g. waiting for the assistive robot's hand-over."""
    if code in ("IDLE", "WALK") or (b - a) < 2 * min_s * RATE_HZ:
        return [(code, a, b)]
    m = (f["act"][a:b] < 0.05) & (f["pelvis_v"][a:b] < 0.10)
    runs, i = [], 0
    while i < len(m):
        if m[i]:
            j = i
            while j < len(m) and m[j]:
                j += 1
            if j - i >= min_s * RATE_HZ:
                runs.append((a + i, a + j))
            i = j
        else:
            i += 1
    out, cur = [], a
    for x, y in runs:
        if x - cur >= 1:
            out.append((code, cur, x))
        out.append(("IDLE", x, y))
        cur = y
    if b - cur >= 1:
        out.append((code, cur, b))
    return out


def count_screw_cycles(f: dict, a: int, b: int) -> int:
    """Independent count of fastening dwells: runs (>=0.5 s, gaps < 0.4 s merged) of one hand still on an
    extended arm while the operator stands at the work anchor."""
    m = (f["dwell"][a:b] < 0.08) & (f["reach_max"][a:b] > 0.33) & (f["pelvis_v"][a:b] < V_WALK)
    runs = []
    i = 0
    while i < len(m):
        if m[i]:
            j = i
            while j < len(m) and m[j]:
                j += 1
            runs.append([i, j])
            i = j
        else:
            i += 1
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] < 0.4 * RATE_HZ:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return sum(1 for r in merged if r[1] - r[0] >= 0.5 * RATE_HZ)


def element_attrs(f: dict, code: str, a: int, b: int) -> dict:
    sl = slice(a, max(a + 1, b))
    vR = float(np.mean(f["wrist_vR"][sl]))
    vL = float(np.mean(f["wrist_vL"][sl]))
    if max(vR, vL) < 0.04:
        hands = "none"
    elif min(vR, vL) > 0.6 * max(vR, vL):
        hands = "both"
    else:
        hands = "right" if vR > vL else "left"
    side = "R" if vR >= vL else "L"
    wr = f[f"wrist_{side}"][sl]
    wpath = float(np.linalg.norm(np.diff(wr, axis=0), axis=1).sum()) if len(wr) > 1 else 0.0
    tail = wr[int(len(wr) * 0.6):] if len(wr) > 2 else wr
    disp = float(np.max(np.linalg.norm(wr - wr[0], axis=1))) if len(wr) > 1 else 0.0
    return {
        "hands": hands,
        "travel_m": round(path_length(f["pelvis_xy"][sl]), 3),
        "trunk_flex_deg_mean": round(float(np.mean(f["trunk_flex"][sl])), 2),
        "trunk_flex_deg_max": round(float(np.max(f["trunk_flex"][sl])), 2),
        "reach_m_p95": round(float(np.percentile(f[f"reach{side}"][sl], 95)), 3),
        "wrist_path_m": round(wpath, 3),
        "wrist_disp_m": round(disp, 3),
        "hand_height_m": round(float(np.median(f[f"wrist_z{side}"][sl])), 3),
        "work_point_m": [round(float(x), 3) for x in np.median(tail, axis=0)],
        "dist_from_anchor_m": round(float(np.median(f["d_anchor"][sl])), 3),
    }
