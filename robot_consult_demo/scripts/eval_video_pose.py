"""Accuracy of the video-only 3D worker (s1 generic mode) against the mocap reference run (evaluation only).

    python scripts/eval_video_pose.py [--ref config/demo_regress.yaml] [--gen config/carda_generic.yaml]

Uses the same metrics as gate G1 (tests/validate_generic.py): valid-frame coverage, trunk flexion (Hips->Neck vs
vertical) Pearson r and MAE, rigid-aligned pelvis floor path, wrist-speed Spearman. Reported for the pipeline's
own choice (multi-view where >= 2 cameras exist) and for monocular reconstructions of the same two-view cycles
(each camera alone). Also reports a clock-offset diagnostic: lag maximising the pelvis-speed correlation between
video and mocap (the gate assumes both clips start at the same instant). Writes <gen out_root>/s1/pose_eval.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rcd.common.io import load_config  # noqa: E402


def trunk_flex(P, names):
    v = P[:, names.index("Neck")] - P[:, names.index("Hips")]
    return np.degrees(np.arccos(np.clip(v[:, 2] / np.linalg.norm(v, axis=1), -1, 1)))


def kabsch2d(a, b):
    ma, mb = a.mean(0), b.mean(0)
    H = (a - ma).T @ (b - mb)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    return np.linalg.norm((a - ma) @ R.T + mb - b, axis=1)


def metrics(a, b) -> dict | None:
    na, nb = list(a["names"]), list(b["names"])
    ta = np.arange(len(a["joints"])) / float(a["fps"])
    tb = np.arange(len(b["joints"])) / float(b["fps"])
    valid = b["valid"].astype(bool)
    m = valid & (tb <= ta[-1])
    T = tb[m]
    if len(T) < 30:
        return None
    fa = np.interp(T, ta, trunk_flex(a["joints"], na))
    fb = trunk_flex(b["joints"], nb)[m]
    pa = np.stack([np.interp(T, ta, a["joints"][:, na.index("Hips"), k]) for k in (0, 1)], 1)
    pb = b["joints"][m, nb.index("Hips"), :2]
    sp = []
    for side in ("Right", "Left"):
        wa = np.stack([np.interp(T, ta, a["joints"][:, na.index(f"{side}Wrist"), k]) for k in range(3)], 1)
        wb = b["joints"][m, nb.index(f"{side}Wrist")]
        va = np.linalg.norm(np.gradient(wa, T, axis=0), axis=1)
        vb = np.linalg.norm(np.gradient(wb, T, axis=0), axis=1)
        k5 = np.ones(5) / 5
        sp.append(spearmanr(np.convolve(va, k5, "same"), np.convolve(vb, k5, "same")).statistic)
    # clock-offset diagnostic from pelvis horizontal speed
    tt = np.arange(0, ta[-1], 0.1)
    xa = np.stack([np.interp(tt, ta, a["joints"][:, na.index("Hips"), k]) for k in (0, 1)], 1)
    xb = b["joints"][:, nb.index("Hips"), :2]
    va = uniform_filter1d(np.linalg.norm(np.gradient(xa, 0.1, axis=0), axis=1), 5)
    vb = uniform_filter1d(np.linalg.norm(np.gradient(xb, 1 / float(b["fps"]), axis=0), axis=1), 5)
    best = (-2.0, 0.0)
    for lag in np.arange(-5, 5.01, 0.1):
        k = (tt + lag >= 0) & (tt + lag <= tb[-1])
        r = np.corrcoef(va[k], np.interp(tt[k] + lag, tb, vb))[0, 1]
        best = max(best, (r, round(float(lag), 1)))
    return {"valid_frac": round(float(valid.mean()), 4),
            "trunk_r": round(float(np.corrcoef(fa, fb)[0, 1]), 3),
            "trunk_mae_deg": round(float(np.mean(np.abs(fa - fb))), 2),
            "trunk_bias_deg": round(float(np.mean(fb - fa)), 2),
            "trunk_ref_std_deg": round(float(fa.std()), 2),
            "pelvis_path_m": round(float(np.median(kabsch2d(pb, pa))), 3),
            "wrist_speed_spearman": round(float(np.nanmean(sp)), 3),
            "clock_offset_diag": {"best_lag_s": best[1], "pelvis_speed_r_at_best": round(best[0], 3)}}


def summary(rows: list[dict]) -> dict:
    if not rows:
        return {}
    k = ["valid_frac", "trunk_r", "trunk_mae_deg", "pelvis_path_m", "wrist_speed_spearman"]
    out = {f"median_{x}": round(float(np.median([r[x] for r in rows])), 3) for x in k}
    out["mean_valid_frac"] = round(float(np.mean([r["valid_frac"] for r in rows])), 3)
    out["n_cycles"] = len(rows)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="config/demo_regress.yaml")
    ap.add_argument("--gen", default="config/carda_generic.yaml")
    a = ap.parse_args()
    ref, gen = load_config(ROOT / a.ref), load_config(ROOT / a.gen)
    R = {(s["station"], c["cycle_id"]): c for s in json.loads((ROOT / ref["out_root"] / "s1/scene_graph.json").read_text())["stations"] for c in s["cycles"]}
    G = {(s["station"], c["cycle_id"]): c for s in json.loads((ROOT / gen["out_root"] / "s1/scene_graph.json").read_text())["stations"] for c in s["cycles"]}
    res = {"main": {}, "mono": {}}
    for k in sorted(set(R) & set(G)):
        A = np.load(ROOT / R[k]["human"]["joints_npz"])
        B = np.load(ROOT / G[k]["human"]["joints_npz"])
        e = metrics(A, B)
        if e:
            res["main"]["/".join(k)] = {"views": G[k]["human"].get("views"), **e}
        for side, f in (G[k]["human"].get("mono_reconstructions") or {}).items():
            e = metrics(A, np.load(ROOT / f))
            if e:
                res["mono"]["/".join(k) + f"/{side}"] = {"views": "mono", **e}
    main = list(res["main"].values())
    two = [v for kk, v in res["main"].items() if v["views"] == "multi"]
    out = {
        "reference": f"{ref['out_root']}/s1 (mocap pipeline)", "candidate": f"{gen['out_root']}/s1 (video only)",
        "metrics": "as gate G1",
        "summary": {
            "all_cycles_pipeline_choice": summary(main),
            "two_view_cycles_multi": summary(two),
            "two_view_cycles_mono_each_camera": summary(list(res["mono"].values())),
            "two_view_cycles_mono_in_camera": summary([v for kk, v in res["mono"].items() if kk.endswith("/in")]),
            "two_view_cycles_mono_out_camera": summary([v for kk, v in res["mono"].items() if kk.endswith("/out")]),
            "single_view_cycles_mono": summary([v for v in main if v["views"] == "mono"]),
        },
        "cycles": res["main"], "mono_on_two_view_cycles": res["mono"],
    }
    p = ROOT / gen["out_root"] / "s1" / "pose_eval.json"
    p.write_text(json.dumps(out, indent=2))
    for k, v in out["summary"].items():
        print(f"{k:38s} {v}")
    print(f"wrote {p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
