"""Posture assessment from mocap: EAWS section-1 posture durations, RULA/REBA trunk & upper-arm scores, and
validation against the CarDA EAWS expert annotation.

EAWS (Ergonomic Assessment Worksheet, Schaub et al. 2013, Int. J. Human Factors Modelling and Simulation 3(4))
standing postures used by CarDA (arXiv 2409.17356, Table 3):
  P1 standing & walking (total), P3 bent forward 20-60 deg, P4 strongly bent forward > 60 deg (trunk Hips->Neck
  flexion measured in the pelvis sagittal plane, as EAWS defines it),
  P5 upright with elbow at/above shoulder level, P6 hands above head level, plus the asymmetric postures
  trunk rotation (TR) and lateral bending (LB).
Thresholds: trunk bands from config.ergonomics (20/60 deg); TR (shoulder line vs hip line, horizontal plane) and
LB (frontal-plane trunk inclination) > 10 deg (ISO 11226:2000: trunk axial rotation / lateral flexion beyond
10 deg is "not recommended" for static postures).
Duration rule: CarDA annotators only counted postures held >= 4 s (paper sec. 3); we apply the same rule
(bouts separated by < 1 s are bridged first). The thresholds are global; nothing is tuned per cycle.

RULA (McAtamney & Corlett 1993) upper-arm score: 1 (<=20 deg), 2 (20-45), 3 (45-90), 4 (>90).
REBA (Hignett & McAtamney 2000) trunk score: 1 upright (<5 deg), 2 (5-20), 3 (20-60), 4 (>60), +1 if twisted or
side-bent (> 10 deg).
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy.stats import spearmanr

from .features import RATE_HZ

POSTURES = ["P1_standing_walking", "P3_bent_20_60", "P4_strongly_bent_gt60", "P5_elbow_at_above_shoulder",
            "P6_hands_above_head", "TR_trunk_rotation", "LB_lateral_bending"]
EAWS_ORDER = POSTURES[:5] + ["P6_hands_above_head"]   # left-to-right order of the EAWS standing block
ASYM_DEG = 10.0
MIN_BOUT_S = 4.0
BRIDGE_S = 1.0


def bouts_filter(mask: np.ndarray, min_s: float = MIN_BOUT_S, bridge_s: float = BRIDGE_S) -> np.ndarray:
    m = mask.astype(bool).copy()
    n = len(m)
    # bridge short gaps
    i = 0
    while i < n:
        if not m[i]:
            j = i
            while j < n and not m[j]:
                j += 1
            if 0 < i and j < n and (j - i) < bridge_s * RATE_HZ:
                m[i:j] = True
            i = j
        else:
            i += 1
    out = np.zeros(n, bool)
    i = 0
    while i < n:
        if m[i]:
            j = i
            while j < n and m[j]:
                j += 1
            if (j - i) >= min_s * RATE_HZ:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


def posture_masks(f: dict, cfg: dict) -> dict[str, np.ndarray]:
    lo_, hi_ = cfg["ergonomics"]["bent_forward_deg"]
    strong = cfg["ergonomics"]["strongly_bent_deg"]
    fl = f["trunk_flex_sag"]
    raw = {
        "P1_standing_walking": np.ones(f["n"], bool),
        "P3_bent_20_60": (fl >= lo_) & (fl < hi_),
        "P4_strongly_bent_gt60": fl >= strong,
        "P5_elbow_at_above_shoulder": f["elbow_above_shoulderR"] | f["elbow_above_shoulderL"],
        "P6_hands_above_head": f["hand_above_headR"] | f["hand_above_headL"],
        "TR_trunk_rotation": np.abs(f["trunk_twist"]) > ASYM_DEG,
        "LB_lateral_bending": np.abs(f["lateral_bend"]) > ASYM_DEG,
    }
    return raw


def durations(masks: dict[str, np.ndarray], a: int, b: int, eaws_rule: bool = True,
              min_s: float = MIN_BOUT_S) -> dict[str, float]:
    out = {}
    for k, m in masks.items():
        mm = bouts_filter(m[a:b], min_s=min_s) if (eaws_rule and k != "P1_standing_walking") else m[a:b]
        out[k] = round(float(mm.sum()) / RATE_HZ, 2)
    return out


def rula_reba(f: dict, a: int = 0, b: int | None = None) -> dict:
    sl = slice(a, b)
    fl = np.abs(f["trunk_flex_sag"][sl])
    asym = (np.abs(f["trunk_twist"][sl]) > ASYM_DEG) | (np.abs(f["lateral_bend"][sl]) > ASYM_DEG)
    reba_trunk = np.select([fl < 5, fl < 20, fl < 60], [1, 2, 3], 4) + asym.astype(int)
    ua = np.maximum(f["upper_arm_elevR"][sl], f["upper_arm_elevL"][sl])
    rula_ua = np.select([ua <= 20, ua <= 45, ua <= 90], [1, 2, 3], 4)

    def dist(x, levels):
        return {str(l): round(float(np.mean(x == l)), 3) for l in levels}
    return {"reba_trunk": {"share_by_score": dist(reba_trunk, range(1, 6)), "p95": float(np.percentile(reba_trunk, 95)),
                           "mean": round(float(reba_trunk.mean()), 3)},
            "rula_upper_arm": {"share_by_score": dist(rula_ua, range(1, 5)), "p95": float(np.percentile(rula_ua, 95)),
                               "mean": round(float(rula_ua.mean()), 3)}}


# ------------------------------------------------------------------------------ annotation column mapping
def infer_column_mapping(records: list[dict]) -> dict:
    """Empirically map the unlabeled CarDA EAWS columns to posture classes.

    records: per (cycle, subgoal) {"pred": {posture: s}, "ann": {col: s or None}}.
    Structure assumed from the EAWS form / CarDA Table 3: the single duration columns G..N follow the EAWS
    standing-posture order (P1, P3, P4, P5, P6; unknown extra columns allowed), and the two (duration, intensity)
    pairs O/P, Q/R are the asymmetric postures {TR, LB}. Among all order-preserving assignments we pick the one
    with minimum total mean-absolute error between mocap-predicted and annotated durations (blank = 0 s). A column
    may stay *unmapped*; its cost is then the MAE of the no-information predictor (0 s), so a column is mapped only
    if the mocap estimate beats predicting "posture absent" (skill criterion)."""
    single = ["G", "H", "I", "J", "K", "L", "M", "N"]
    used = [c for c in single if any((r["ann"].get(c) or 0) > 0 for r in records)]
    order = ["P1_standing_walking", "P3_bent_20_60", "P4_strongly_bent_gt60", "P5_elbow_at_above_shoulder",
             "P6_hands_above_head"]

    def ann(col):
        return np.array([r["ann"].get(col) or 0.0 for r in records])

    def mae(col, post):
        return float(np.mean(np.abs(ann(col) - np.array([r["pred"][post] for r in records]))))

    fit = {c: {p: round(mae(c, p), 2) for p in order} for c in used}
    null = {c: round(float(np.mean(ann(c))), 2) for c in used + ["O", "Q"]}
    best = (np.inf, {})

    def rec(ci: int, pi: int, cost: float, cur: dict):
        nonlocal best
        if ci == len(used):
            if cost < best[0] - 1e-9:
                best = (cost, dict(cur))
            return
        c = used[ci]
        rec(ci + 1, pi, cost + null[c], cur)                     # leave unmapped
        for k in range(pi, len(order)):
            if fit[c][order[k]] < null[c]:
                cur[c] = order[k]
                rec(ci + 1, k + 1, cost + fit[c][order[k]], cur)
                del cur[c]
    rec(0, 0, 0.0, {})
    pair_fit = {col: {p: round(mae(col, p), 2) for p in ("TR_trunk_rotation", "LB_lateral_bending")} for col in ("O", "Q")}
    a1 = pair_fit["O"]["TR_trunk_rotation"] + pair_fit["Q"]["LB_lateral_bending"]
    a2 = pair_fit["O"]["LB_lateral_bending"] + pair_fit["Q"]["TR_trunk_rotation"]
    mapping = dict(best[1])
    mapping.update({"O": "TR_trunk_rotation", "Q": "LB_lateral_bending"} if a1 <= a2 else
                   {"O": "LB_lateral_bending", "Q": "TR_trunk_rotation"})
    return {"mapping": mapping, "unmapped_columns_with_data": [c for c in used if c not in mapping],
            "mae_by_column_and_posture_s": {**fit, **pair_fit}, "null_mae_s": null,
            "rule": "order-preserving assignment of G..N to EAWS standing postures (P1,P3,P4,P5,P6) and {O,Q}->{TR,LB}; "
                    "minimum total MAE; a column is mapped only if mocap beats the 0-s predictor"}


TRIVIAL = ("P1_standing_walking",)   # equals the subgoal duration for a standing operator -> trivially correct


def validate(records: list[dict], mapping: dict) -> dict:
    """Agreement of predicted vs annotated posture durations per (cycle, subgoal, posture).
    Headline metric excludes trivial postures (P1 standing/walking = whole subgoal)."""
    rows = []
    for r in records:
        for col, post in mapping.items():
            a = r["ann"].get(col) or 0.0
            rows.append({"station": r["station"], "cycle_id": r["cycle_id"], "subgoal": r["subgoal"], "column": col,
                         "posture": post, "predicted_s": r["pred"][post], "annotated_s": a,
                         "trivial": post in TRIVIAL})

    def metrics(sel):
        P = np.array([[x["predicted_s"], x["annotated_s"]] for x in sel], float)
        if len(P) < 3:
            return P, None, None
        rho = float(spearmanr(P[:, 0], P[:, 1]).statistic) if np.ptp(P[:, 0]) > 0 and np.ptp(P[:, 1]) > 0 else None
        return P, rho, float(np.mean(np.abs(P[:, 0] - P[:, 1])))
    nt = [x for x in rows if not x["trivial"]]
    P, rho, mae = metrics(nt)
    Pall, rho_all, mae_all = metrics(rows)
    per_post = {}
    for post in sorted(set(mapping.values())):
        S, r_, m_ = metrics([x for x in rows if x["posture"] == post])
        per_post[post] = {"n": len(S), "mae_s": round(m_, 2) if m_ is not None else None,
                          "mean_annotated_s": round(float(S[:, 1].mean()), 2) if len(S) else None,
                          "spearman": round(r_, 3) if r_ is not None else None, "trivial": post in TRIVIAL}
    return {"metric": "spearman_rho(predicted_s, annotated_s) over non-trivial (cycle, subgoal, posture) pairs",
            "excludes_trivial": True, "excluded_postures": list(TRIVIAL),
            "value": round(rho, 3), "n": len(P), "pairs": P.round(2).tolist(),
            "mae_s": round(mae, 2), "mae_rel": round(mae / max(1e-9, float(P[:, 1].mean())), 3),
            "including_trivial": {"spearman": round(rho_all, 3), "mae_s": round(mae_all, 2), "n": len(Pall)},
            "by_posture": per_post, "rows": rows}
