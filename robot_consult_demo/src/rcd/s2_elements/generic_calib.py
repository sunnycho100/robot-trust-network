"""Pose-source calibration: monotone quantile mapping of kinematic features from a test instrument (video pose)
onto a reference instrument (marker-based / inertial mocap), like a calibration curve in a gauge study.

Video pose jitter inflates every speed-type feature several-fold while geometric features (reach) stay close, so
the absolute thresholds of the element evidence rules misfire on video. For each mapped feature the transfer
function y = Q_ref(F_test(x)) (empirical quantile matching on paired recordings of the same work) is fitted once
per pose model / camera type -- not per customer -- and stored with its provenance in models/pose_calib/.
Fitting is leave-one-station-out for evaluation (the map applied to station S never saw S); a map fitted on all
reference stations is used for unseen lines.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rcd.common.io import ROOT

# feature groups: a group shares one map (left/right wrists are the same instrument)
GROUPS = {"pelvis_v": ["pelvis_v"], "wrist_v": ["wrist_vR", "wrist_vL"], "hand_dist": ["hand_dist"],
          "hand_dz": ["hand_dz"]}
QUANTILES = np.linspace(0.0, 1.0, 201)
CALIB_DIR = ROOT / "models" / "pose_calib"


def _values(f: dict, group: str, valid: np.ndarray | None) -> np.ndarray:
    x = np.concatenate([np.asarray(f[k]) for k in GROUPS[group]])
    if valid is not None:
        x = x[np.concatenate([valid] * len(GROUPS[group]))]
    return x[np.isfinite(x)]


def fit(pairs: list[tuple[dict, dict, np.ndarray | None]]) -> dict:
    """pairs: [(test_features, ref_features, test_valid_mask_at_feature_rate)] of the same recordings."""
    maps = {}
    for g in GROUPS:
        xt = np.concatenate([_values(ft, g, v[: ft["n"]] if v is not None else None) for ft, fr, v in pairs])
        xr = np.concatenate([_values(fr, g, None) for ft, fr, v in pairs])
        qt, qr = np.quantile(xt, QUANTILES), np.quantile(xr, QUANTILES)
        qt = np.maximum.accumulate(qt + np.arange(len(qt)) * 1e-9)      # strictly increasing knots
        maps[g] = {"test_q": [round(float(v), 6) for v in qt], "ref_q": [round(float(v), 6) for v in qr],
                   "test_median": round(float(np.median(xt)), 4), "ref_median": round(float(np.median(xr)), 4),
                   "n_test": int(len(xt)), "n_ref": int(len(xr))}
    return maps


def apply(f: dict, maps: dict) -> dict:
    """Map the calibrated features in place and recompute the derived ones (act, dwell)."""
    for g, m in maps.items():
        for k in GROUPS.get(g, []):
            f[k] = np.interp(f[k], m["test_q"], m["ref_q"])
    f["act"] = np.maximum(f["wrist_vR"], f["wrist_vL"])
    f["dwell"] = np.minimum(f["wrist_vR"], f["wrist_vL"])
    return f


def load(source: str = "video") -> dict | None:
    p = CALIB_DIR / f"{source}.json"
    return json.loads(p.read_text()) if p.exists() else None


def maps_for(calib: dict, station: str, in_fit_data: bool) -> tuple[dict, str]:
    """Leave-one-station-out map for a station of the calibration data set, else the map fitted on all stations."""
    key = f"exclude:{station}"
    if in_fit_data and key in calib["maps"]:
        return calib["maps"][key], key
    return calib["maps"]["all"], "all"
