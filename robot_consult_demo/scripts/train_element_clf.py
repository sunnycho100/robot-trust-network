"""Train the learned frame evidence (video features -> reference element codes), leave-one-station-out.

    python scripts/train_element_clf.py --variant learned        # raw (RTS-smoothed) video features
    python scripts/train_element_clf.py --variant learned_qmap   # + pose-source quantile map first

Pairs: cycles in both the test run's s1 (video) and the reference run (mocap s1 + reference s2 elements = labels).
Models: 'all' (every station) and 'exclude_<ST>' (trained without station ST, applied only to ST).
-> models/pose_calib/elements_clf_<variant>_<key>.joblib + elements_clf_<variant>.json (provenance, LOSO scores)
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rcd.common.io import load_config  # noqa: E402
from rcd.s2_elements import generic_calib as GC  # noqa: E402
from rcd.s2_elements import generic_clf as GCL  # noqa: E402
from rcd.s2_elements import generic_io as GIO  # noqa: E402
from rcd.s2_elements.features import RATE_HZ, extract, work_anchor  # noqa: E402


def features_for(en, maps):
    jd = GIO.load_cycle_joints(en)
    f = extract(jd["joints"], jd["names"], jd["fps"])
    if maps:
        f = GC.apply(f, maps)
    step = max(1, int(round(jd["fps"] / RATE_HZ)))
    v = jd["valid"][: f["n"] * step].reshape(-1, step).mean(1) >= 0.5
    X, names = GCL.frame_features(f, work_anchor(f), en.get("line_side_xy"))
    return X, names, v


def main() -> int:
    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="learned", choices=["learned", "learned_qmap"])
    ap.add_argument("--test", default="config/carda_generic.yaml")
    ap.add_argument("--ref", default="config/demo_regress.yaml")
    a = ap.parse_args()
    tcfg, rcfg = load_config(ROOT / a.test), load_config(ROOT / a.ref)
    test, origin = GIO.scene_cycles(tcfg)
    if origin != "s1":
        raise SystemExit("test run has no s1 of its own")
    ref_el = json.loads((ROOT / rcfg["out_root"] / "s2" / "elements.json").read_text())
    labels = {(c["station"], c["cycle_id"]): c["elements"] for c in ref_el["cycles"]}
    cal = GC.load("video") if a.variant == "learned_qmap" else None
    entries = [en for en in test if (en["station"], en["cycle_id"]) in labels]
    stations = sorted({en["station"] for en in entries})
    keys = ["all"] + [f"exclude_{s}" for s in stations]
    report = {}
    for key in keys:
        held = key.split("_", 1)[1] if key != "all" else None
        maps = (cal["maps"]["all"] if key == "all" else cal["maps"][f"exclude:{held}"]) if cal else None
        Xs, ys, Xh, yh = [], [], [], []
        for en in entries:
            X, names, v = features_for(en, maps)
            lab = np.array(["-"] * len(X), object)
            for e in labels[(en["station"], en["cycle_id"])]:
                lab[int(round(e["start_s"] * RATE_HZ)):int(round(e["end_s"] * RATE_HZ))] = e["code"]
            ok = v & (lab != "-")
            (Xh if en["station"] == held else Xs).append(X[ok])
            (yh if en["station"] == held else ys).append(lab[ok])
        Xtr, ytr = np.concatenate(Xs), np.concatenate(ys).astype(str)
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
                                             l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
                                             random_state=0)
        clf.fit(Xtr, ytr)
        cls, cnt = np.unique(ytr, return_counts=True)
        priors = {c: float(n / cnt.sum()) for c, n in zip(cls, cnt)}
        rec = {"n_train_frames": int(len(ytr)), "train_stations": [s for s in stations if s != held],
               "classes": [str(c) for c in clf.classes_], "priors": priors}
        if held:
            Xte, yte = np.concatenate(Xh), np.concatenate(yh).astype(str)
            rec["held_out_station"] = held
            rec["held_out_frame_accuracy"] = round(float(np.mean(clf.predict(Xte) == yte)), 4)
            rec["n_test_frames"] = int(len(yte))
        joblib.dump({"clf": clf, "priors": priors, "feature_names": names, "variant": a.variant, "key": key},
                    GCL.MODEL_DIR / f"elements_clf_{a.variant}_{key}.joblib")
        report[key] = rec
        print(key, {k: v for k, v in rec.items() if k not in ("priors", "classes")})
    sg = ROOT / tcfg["out_root"] / "s1" / "scene_graph.json"
    meta = {"variant": a.variant, "model": "sklearn HistGradientBoostingClassifier (log-loss), evidence = "
                                         "log P(code|x) - log P(code) (hybrid HMM scaled likelihoods)",
            "features": "multi-scale (0.5/1.5/4 s) rolling mean/std/p10/p90 of speed, reach, hand distance and "
                        "wrist-height signals + trunk flexion + distances to work anchor and line side",
            "labels": f"reference element codes ({rcfg['out_root']}/s2/elements.json, mocap + annotated subgoals)",
            "test": f"{tcfg['out_root']}/s1 video joints (RTS-smoothed" + (", quantile-mapped)" if cal else ")"),
            "test_scene_graph_mtime": datetime.datetime.fromtimestamp(sg.stat().st_mtime).isoformat(timespec="seconds"),
            "trained_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "pairs": [f"{en['station']}_{en['cycle_id']}" for en in entries], "models": keys, "report": report,
            "caveat": "leave-one-station-out with 3 stations is a small-sample estimate of transfer to new lines"}
    (GCL.MODEL_DIR / f"elements_clf_{a.variant}.json").write_text(json.dumps(meta, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
