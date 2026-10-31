"""Learned frame evidence for noisy pose sources (hybrid HMM): a frame classifier trained against a reference
instrument replaces the threshold rules of segment.evidence.

Input: multi-scale windowed statistics (0.5 / 1.5 / 4 s rolling mean, std, p10, p90) of the speed, reach, hand
distance and wrist height signals, plus trunk flexion and the distances to the work anchor and to the line side.
Windowing restores the temporal structure that pose jitter destroys frame by frame.
Output as evidence: log P(code | x) - log P(code) (scaled likelihoods, Bourlard & Morgan 1994), so the HSMM / SOP
structure is unchanged. Models are trained strictly leave-one-station-out for evaluation and persisted with their
provenance in models/pose_calib/.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.ndimage import percentile_filter, uniform_filter1d

from rcd.common.io import ROOT

from .features import RATE_HZ

BASE = ["pelvis_v", "wrist_vR", "wrist_vL", "act", "dwell", "reachR", "reachL", "reach_max", "reach_min",
        "hand_dist", "hand_dz", "wrist_zR", "wrist_zL"]
WINDOWS_S = [0.5, 1.5, 4.0]
MODEL_DIR = ROOT / "models" / "pose_calib"
LOGP_FLOOR = -8.0


def frame_features(f: dict, anchor_xy: np.ndarray, line_side_xy: np.ndarray | None) -> tuple[np.ndarray, list[str]]:
    cols, names = [], []
    for k in BASE:
        x = np.asarray(f[k], float)
        cols.append(x)
        names.append(k)
        for w_s in WINDOWS_S:
            w = max(2, int(round(w_s * RATE_HZ)))
            m = uniform_filter1d(x, w, mode="nearest")
            sd = np.sqrt(np.maximum(0.0, uniform_filter1d(x * x, w, mode="nearest") - m * m))
            cols += [m, sd, percentile_filter(x, 10, size=w, mode="nearest"), percentile_filter(x, 90, size=w, mode="nearest")]
            names += [f"{k}_m{w_s}", f"{k}_sd{w_s}", f"{k}_p10_{w_s}", f"{k}_p90_{w_s}"]
    cols.append(np.asarray(f["trunk_flex"], float))
    names.append("trunk_flex")
    cols.append(uniform_filter1d(np.asarray(f["trunk_flex"], float), int(1.5 * RATE_HZ), mode="nearest"))
    names.append("trunk_flex_m1.5")
    d_anchor = np.linalg.norm(f["pelvis_xy"] - anchor_xy, axis=1)
    cols.append(d_anchor)
    names.append("d_anchor")
    if line_side_xy is not None:
        cols.append(np.linalg.norm(f["pelvis_xy"] - np.asarray(line_side_xy)[None, :], axis=1))
    else:
        cols.append(np.full(f["n"], np.nan))
    names.append("d_line_side")
    return np.stack(cols, 1), names


def evidence(model: dict, X: np.ndarray, codes: list[str]) -> dict[str, np.ndarray]:
    """Scaled log-likelihoods log P(c|x) - log P(c) for every code; codes unseen in training get LOGP_FLOOR."""
    P = model["clf"].predict_proba(X)
    classes = list(model["clf"].classes_)
    prior = np.array([model["priors"][c] for c in classes])
    L = np.log(np.maximum(P, 1e-6)) - np.log(prior)[None, :]
    return {c: (L[:, classes.index(c)] if c in classes else np.full(len(X), LOGP_FLOOR)) for c in codes}


def load(name: str) -> dict | None:
    import joblib
    p = MODEL_DIR / f"elements_clf_{name}.joblib"
    return joblib.load(p) if p.exists() else None


def model_key(meta: dict, station: str, cycle_key: str) -> str:
    """Leave-one-station-out model for a station of the training set, else the model trained on all stations."""
    if cycle_key in meta.get("pairs", []) and f"exclude_{station}" in meta.get("models", []):
        return f"exclude_{station}"
    return "all"


def load_meta(variant: str) -> dict | None:
    p = MODEL_DIR / f"elements_clf_{variant}.json"
    return json.loads(p.read_text()) if p.exists() else None
