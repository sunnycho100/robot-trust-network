"""Joint input for the generic (video / SOP-only) mode: cycles and human joints from the s1 scene graph.

Joints come from `<out_root>/s1/human/<ST>_<cycle>.npz` as listed in `<out_root>/s1/scene_graph.json`
(`human.joints_npz`, `human.joint_names`). Video pose may carry a per-frame `valid` mask and a different joint
set; names are canonicalised (COCO / H36M / SMPL / mocap spellings) and missing joints are synthesised from their
neighbours (pelvis = mid-hips, neck = mid-shoulders, head = neck + 0.25 m up when no head/nose joint exists).

TEMPORARY FALLBACK (clearly marked): while the video-pose s1 output of this run does not exist yet, joints are read
from the reference pipeline's s1 (`FALLBACK_OUT_ROOT`, mocap) so s2/s3 can be developed; the cycle records
`joint_source` = "fallback:..." so this can never be mistaken for video joints. Remove once s1 video exists.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter

from rcd.common.io import ROOT

FALLBACK_OUT_ROOT = "out_regress"        # TEMPORARY: reference (mocap) s1, used only when this run has no s1 yet

CANON = ["Hips", "Neck", "Head", "LeftHip", "RightHip", "LeftShoulder", "RightShoulder", "LeftElbow", "RightElbow",
         "LeftWrist", "RightWrist", "LeftAnkle", "RightAnkle"]
_PART = {"hip": "Hip", "upleg": "Hip", "thigh": "Hip", "upperleg": "Hip",
         "shoulder": "Shoulder", "upperarm": "Shoulder", "arm": "Shoulder",
         "elbow": "Elbow", "forearm": "Elbow", "lowerarm": "Elbow",
         "wrist": "Wrist", "hand": "Wrist", "ankle": "Ankle", "foot": "Ankle"}
_CENTRE = {"hips": "Hips", "pelvis": "Hips", "root": "Hips", "midhip": "Hips", "hip": "Hips",
           "neck": "Neck", "thorax": "Neck", "upperneck": "Neck", "neck1": "Neck",
           "head": "Head", "headtop": "Head", "nose": "Head", "head1": "Head"}


def _canon(name: str) -> str | None:
    s = re.sub(r"[^a-z0-9]", "", str(name).lower())
    if s in _CENTRE:
        return _CENTRE[s]
    for side, pre in (("Left", ("left", "l")), ("Right", ("right", "r"))):
        for p in pre:
            for cand in (s[len(p):] if s.startswith(p) else None, s[: -len(p)] if s.endswith(p) else None):
                if cand and cand in _PART:
                    return side + _PART[cand]
    return None


def canonical_joints(J: np.ndarray, names: list[str]) -> tuple[np.ndarray, list[str], list[str]]:
    """(T, J, 3) with arbitrary names -> (T, len(CANON), 3) in CANON order; returns also the synthesised joints."""
    J = np.asarray(J, float)
    idx: dict[str, int] = {}
    for i, n in enumerate(names):
        c = n if n in CANON else _canon(n)
        if c and c not in idx:                      # first match wins (exact names come first in mocap sets)
            idx[c] = i
    for i, n in enumerate(names):                   # exact canonical names override synonyms
        if n in CANON:
            idx[n] = i
    T = J.shape[0]
    out = np.full((T, len(CANON), 3), np.nan)
    for k, c in enumerate(CANON):
        if c in idx:
            out[:, k] = J[:, idx[c]]
    made = []
    P = {c: out[:, k] for k, c in enumerate(CANON)}

    def fill(c, v):
        k = CANON.index(c)
        out[:, k] = v
        P[c] = out[:, k]
        made.append(c)
    if np.isnan(P["Hips"]).all() and not (np.isnan(P["LeftHip"]).all() or np.isnan(P["RightHip"]).all()):
        fill("Hips", 0.5 * (P["LeftHip"] + P["RightHip"]))
    if np.isnan(P["Neck"]).all() and not (np.isnan(P["LeftShoulder"]).all() or np.isnan(P["RightShoulder"]).all()):
        fill("Neck", 0.5 * (P["LeftShoulder"] + P["RightShoulder"]))
    if np.isnan(P["Head"]).all() and not np.isnan(P["Neck"]).all():
        fill("Head", P["Neck"] + np.array([0.0, 0.0, 0.25]))
    for side in ("Left", "Right"):
        if np.isnan(P[f"{side}Hip"]).all() and not np.isnan(P["Hips"]).all():
            other = "Right" if side == "Left" else "Left"
            if not np.isnan(P[f"{other}Hip"]).all():
                fill(f"{side}Hip", 2 * P["Hips"] - P[f"{other}Hip"])
    need = [c for c in CANON if c not in ("LeftAnkle", "RightAnkle")]
    missing = [c for c in need if np.isnan(out[:, CANON.index(c)]).all()]
    if missing:
        raise ValueError(f"pose lacks required joints {missing} (names: {list(names)[:30]})")
    names_out = list(CANON)
    keep = [k for k, c in enumerate(CANON) if not np.isnan(out[:, k]).all()]
    return out[:, keep], [names_out[k] for k in keep], made


def _interp_nan(J: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ok = valid & ~np.isnan(J).any(axis=(1, 2))
    if ok.sum() < 2:
        raise ValueError("fewer than 2 frames with a complete pose")
    t = np.arange(len(J))
    for j in range(J.shape[1]):
        for d in range(3):
            J[~ok, j, d] = np.interp(t[~ok], t[ok], J[ok, j, d])
    return J, ok


def scene_cycles(cfg: dict) -> tuple[list[dict], str]:
    """Cycles of this run from its s1 scene graph; TEMPORARY fallback to the reference s1 when absent."""
    sg_path = ROOT / cfg["out_root"] / "s1" / "scene_graph.json"
    origin = "s1"
    if not sg_path.exists():
        sg_path = ROOT / FALLBACK_OUT_ROOT / "s1" / "scene_graph.json"
        origin = f"fallback:{FALLBACK_OUT_ROOT}"
        print(f"    s2: WARNING no {cfg['out_root']}/s1/scene_graph.json yet - TEMPORARY fallback to {sg_path.relative_to(ROOT)}")
    sg = json.loads(sg_path.read_text())
    stations = set(cfg["data"]["stations"])
    res = []
    for s in sg["stations"]:
        if s["station"] not in stations:
            continue
        ls = next((z for z in s.get("zones", []) if z.get("name") == "line_side" and z.get("polygon_xy")), None)
        ls_xy = np.mean(np.asarray(ls["polygon_xy"], float), axis=0).tolist() if ls else None
        for c in s["cycles"]:
            h = c["human"]
            res.append({"station": s["station"], "cycle_id": c["cycle_id"], "npz": ROOT / h["joints_npz"],
                        "line_side_xy": ls_xy,
                        "names": h.get("joint_names"), "fps": h.get("fps"), "video": c.get("video", {}),
                        "source_desc": h.get("source", ""), "origin": origin})
    return res, origin


def load_cycle_joints(entry: dict) -> dict:
    d = np.load(entry["npz"], allow_pickle=True)
    names = [str(n) for n in (d["names"] if "names" in d.files else entry["names"])]
    fps = float(d["fps"]) if "fps" in d.files else float(entry["fps"])
    J = np.asarray(d["joints"], float)
    valid = d["valid"].astype(bool) if "valid" in d.files else np.ones(len(J), bool)
    Jc, nc, made = canonical_joints(J, names)
    mocap = entry["origin"].startswith(("mocap", "fallback"))
    noise = None
    if not mocap:                      # video pose: noise-adaptive RTS smoothing before differentiation
        Jc, valid0 = _interp_nan(Jc, valid)
        Jc, noise = rts_smooth(Jc, nc, fps, valid)
        valid = valid0
    else:
        Jc, valid = _interp_nan(Jc, valid)
    src = f"{entry['origin']}:{Path(entry['npz']).relative_to(ROOT)}"
    return {"joints": Jc, "names": nc, "fps": fps, "valid": valid, "synthesised": made, "joint_source": src,
            "denoise": not mocap, "pose_noise": noise}


def mocap_cycles(cfg: dict) -> list[dict]:
    """The reference pipeline's mocap joints (for elements_mocap_sop.json: isolates segmentation from pose error)."""
    sg_path = ROOT / FALLBACK_OUT_ROOT / "s1" / "scene_graph.json"
    if not sg_path.exists():
        return []
    sg = json.loads(sg_path.read_text())
    stations = set(cfg["data"]["stations"])
    return [{"station": s["station"], "cycle_id": c["cycle_id"], "npz": ROOT / c["human"]["joints_npz"],
             "names": c["human"].get("joint_names"), "fps": c["human"].get("fps"), "video": c.get("video", {}),
             "source_desc": c["human"].get("source", ""), "origin": f"mocap:{FALLBACK_OUT_ROOT}"}
            for s in sg["stations"] if s["station"] in stations for c in s["cycles"]]


ACCEL_STD = {"Hips": 1.0, "Neck": 1.0, "Head": 1.0, "LeftHip": 1.0, "RightHip": 1.0, "LeftShoulder": 1.0,
             "RightShoulder": 1.0}   # m/s^2 white-acceleration std of the body trunk; limbs ACCEL_LIMB
ACCEL_LIMB = 2.0


def rts_smooth(J: np.ndarray, names: list[str], fps: float, valid: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Noise-adaptive smoothing of 3D joint tracks: constant-velocity Kalman filter + Rauch-Tung-Striebel smoother
    per joint and axis. Measurement noise is estimated from the data (robust MAD of the residual about a 2 s running
    median; video pose noise is temporally correlated, so second differences would underestimate it); process noise is a
    white acceleration of typical human-motion magnitude (ACCEL_STD). Frames without a valid pose are treated as
    missing measurements. For low-noise sources (mocap) the smoother is practically the identity."""
    T = J.shape[0]
    dt = 1.0 / fps
    F = np.array([[1.0, dt], [0.0, 1.0]])
    Gq = np.array([[0.5 * dt * dt], [dt]])
    ok = np.ones(T, bool) if valid is None else valid.astype(bool)
    out = J.copy()
    sig = {}
    for j, n in enumerate(names):
        qa = ACCEL_STD.get(n, ACCEL_LIMB)
        Q = (Gq @ Gq.T) * qa * qa
        for d in range(3):
            z = J[:, j, d]
            zz = z[ok] if ok.sum() > 3 else z
            w = max(3, int(round(2.0 * fps)) | 1)
            res = zz - median_filter(zz, size=w, mode="nearest")
            s = 1.4826 * np.median(np.abs(res))
            R = max(s * s, 1e-8)
            sig[(n, d)] = float(np.sqrt(R))
            xs = np.zeros((T, 2))
            Ps = np.zeros((T, 2, 2))
            xp = np.zeros((T, 2))
            Pp = np.zeros((T, 2, 2))
            x = np.array([z[np.argmax(ok)], 0.0])
            P = np.diag([R, 1.0])
            for t in range(T):
                if t > 0:
                    x = F @ x
                    P = F @ P @ F.T + Q
                xp[t], Pp[t] = x, P
                if ok[t]:
                    S = P[0, 0] + R
                    K = P[:, 0] / S
                    x = x + K * (z[t] - x[0])
                    P = P - np.outer(K, P[0, :])
                xs[t], Ps[t] = x, P
            for t in range(T - 2, -1, -1):
                C = Ps[t] @ F.T @ np.linalg.inv(Pp[t + 1])
                xs[t] = xs[t] + C @ (xs[t + 1] - xp[t + 1])
                Ps[t] = Ps[t] + C @ (Ps[t + 1] - Pp[t + 1]) @ C.T
            out[:, j, d] = xs[:, 0]
    noise = {"median_sigma_m": round(float(np.median(list(sig.values()))), 4),
             "max_sigma_m": round(float(np.max(list(sig.values()))), 4)}
    return out, noise
