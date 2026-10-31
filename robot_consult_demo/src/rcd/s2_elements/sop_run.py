"""Generic-mode s2 driver (mode.segmentation = sop): joints from the s1 scene graph (video pose or any other
source), SOP steps from the customer intake, SOP-only hierarchical HSMM segmentation (sop_segment.py).

No annotation input of any kind is used here. The downstream time study / ergonomics is shared with the original
mode (run.py).
"""
from __future__ import annotations

import numpy as np

from pathlib import Path

import yaml

from rcd.common.intake import load_intake, mode, station as intake_station
from rcd.common.io import ROOT, save_json

from . import ergonomics as ER
from . import generic_calib as GC
from . import generic_clf as GCL
from . import generic_io as GIO
from .features import RATE_HZ, path_length
from .sop_segment import segment_cycle_sop

REVIEW_FRACTION = 0.20
VIDEO_EVIDENCE = "learned_qmap"   # rules_qmap | learned | learned_qmap (video joints only); chosen in round 4 by G3 top-3, then frame agreement


def video_evidence(cfg: dict) -> str:
    """Evidence model for video joints: mode.video_evidence, env RCD_VIDEO_EVIDENCE (experiments), else default."""
    import os
    return os.environ.get("RCD_VIDEO_EVIDENCE") or mode(cfg, "video_evidence", VIDEO_EVIDENCE)

ALGORITHM = ("SOP-only two-level hierarchical HSMM over the whole clip (no annotation timestamps): top level = the "
             "intake's ordered SOP steps (skippable; clip may start/end mid-SOP; non-cycle time before/after the SOP; "
             "Mallows prior on small order deviations; weak log-normal step-duration priors = pace factor x MTM-UAS "
             "synthetic time of the step's activities; step boundaries preferably at line-side visits or posture/"
             "location change points); bottom level = left-to-right profile HSMM of the step's element script "
             "derived from the activity text by a dataset-neutral lexicon (optional walks/waits, deletions, explicit "
             "minimum durations, motion-only fallback for unclassified text). Evidence: hybrid scaled likelihoods of "
             "the kinematic element rules, tempered by their autocorrelation time; exact DP over a 1 s boundary grid "
             "at 5 Hz, then 10 Hz Viterbi re-alignment of each step; waiting carved out as IDLE")


def _std_dir(cfg: dict) -> Path | None:
    """mode.std_times: none | loo_dir:<path>. Without the key, a leave-one-cycle-out directory generated for this
    run's out_root (scripts/make_std_times_loo.py) is used for the secondary output only."""
    m = mode(cfg, "std_times", "")
    if m.startswith("loo_dir:"):
        return ROOT / m.split(":", 1)[1]
    for meta in sorted((ROOT / "intake" / "std_times_loo").glob("*/meta.yaml")):
        if yaml.safe_load(meta.read_text()).get("out_root") == cfg["out_root"]:
            return meta.parent
    return None


def _load_std(d: Path | None) -> dict:
    res = {}
    if d is None or not d.exists():
        return res
    for p in d.glob("*.yaml"):
        y = yaml.safe_load(p.read_text())
        if y and "cycle_id" in y:
            res[(y["station"], str(y["cycle_id"]))] = y.get("std_time_s") or {}
    return res


def _with_std(sop: list[dict], std: dict | None) -> list[dict]:
    if not std:
        return sop
    return [dict(s, std_time_s=std[s["step"]]) if s["step"] in std else dict(s) for s in sop]


def _segment(cfg: dict, entries: list[dict], sops: dict, element_ergo, std: dict | None = None) -> tuple[list, dict, dict, list]:
    cycles, feats, masks_by, erg_cycles = [], {}, {}, []
    for en in entries:
        sop = sops.get(en["station"])
        if sop and std is not None:
            sop = _with_std(sop, std.get((en["station"], en["cycle_id"])))
        if not sop:
            print(f"    s2: no SOP in the intake for {en['station']} - cycle {en['cycle_id']} skipped")
            continue
        jd = GIO.load_cycle_joints(en)
        cmaps, ckey, clf, mkey = None, None, None, None
        variant = video_evidence(cfg)
        if jd["denoise"] and mode(cfg, "human_source", "mocap") == "video":
            cal = GC.load("video") if variant in ("rules_qmap", "learned_qmap") else None
            if cal:
                fit_data = f"{en['station']}_{en['cycle_id']}" in cal.get("pairs", [])
                cmaps, ckey = GC.maps_for(cal, en["station"], fit_data)
            if variant.startswith("learned"):
                meta = GCL.load_meta(variant)
                if meta:
                    mkey = GCL.model_key(meta, en["station"], f"{en['station']}_{en['cycle_id']}")
                    clf = GCL.load(f"{variant}_{mkey}")
        cyc, f = segment_cycle_sop(en["station"], en["cycle_id"], jd["joints"], jd["names"], jd["fps"], sop,
                                   valid=jd["valid"], joint_source=jd["joint_source"], video=en.get("video"),
                                   takt_s=float(cfg["line"]["takt_s"]), denoise=jd["denoise"], calib=cmaps,
                                   clf=clf, line_side_xy=en.get("line_side_xy"))
        cyc["pose_calibration"] = f"models/pose_calib/video.json:{ckey}" if ckey else None
        cyc["evidence"] = (f"learned:models/pose_calib/elements_clf_{variant}_{mkey}.joblib" if clf is not None
                           else ("rules+quantile_map" if ckey else "rules"))
        cyc["pose_noise_sigma_m"] = jd["pose_noise"]
        cyc["joints_synthesised"] = jd["synthesised"]
        M = ER.posture_masks(f, cfg)
        for e in cyc["elements"]:
            element_ergo(e, M, f)
        a, b = (int(round(x * RATE_HZ)) for x in cyc["cycle_window_s"])
        b = max(b, a + 1)
        cyc["spaghetti_m"] = round(path_length(f["pelvis_xy"][a:b]), 2)
        cyc["walk_distance_m"] = round(sum(e["travel_m"] for e in cyc["elements"] if e["code"] == "WALK"), 2)
        cycles.append(cyc)
        key = (en["station"], en["cycle_id"])
        feats[key] = f
        # ergonomics restricted to the cycle window (time outside the SOP is not work)
        masks_by[key] = {k: v[a:b] for k, v in M.items()}
        f["_masks"] = M                                     # whole-clip masks (validation on annotated spans)
        sub = []
        for sp in cyc["subgoals"]:
            x, y = int(round(sp["refined"][0] * RATE_HZ)), int(round(sp["refined"][1] * RATE_HZ))
            sub.append({"subgoal": sp["name"], "span_s": sp["refined"], "postures_s": ER.durations(M, x, y),
                        "postures_raw_s": ER.durations(M, x, y, eaws_rule=False), "scores": ER.rula_reba(f, x, y)})
        erg_cycles.append({"station": en["station"], "cycle_id": en["cycle_id"], "duration_s": cyc["duration_s"],
                           "postures_s": ER.durations(M, a, b), "postures_raw_s": ER.durations(M, a, b, False),
                           "scores": ER.rula_reba(f, a, b), "subgoals": sub})
    return cycles, feats, masks_by, erg_cycles


def review_queue(cycles: list[dict], frac: float = REVIEW_FRACTION) -> dict:
    """IE review queue: one entry per decoded SOP step (its start/end boundaries); confidence = the smaller boundary
    margin (score gap vs the best alternative boundary within +/-10 s); the lowest-confidence share `frac` and every
    entry without a computable margin are marked needs_review."""
    rows = [dict(r) for c in cycles for r in c.get("review", [])]
    confs = sorted(r["confidence"] for r in rows if r["confidence"] is not None)
    k = int(np.ceil(frac * len(rows)))
    thr = confs[k - 1] if (k and confs) else None
    for r in rows:
        r["needs_review"] = r["confidence"] is None or (thr is not None and r["confidence"] <= thr)
    return {"method": "per decoded SOP step: confidence = min(start, end) boundary margin = top-level score of the "
                      "chosen boundary minus the best alternative boundary >= 2 s away within +/-10 s (tempered "
                      "nats); flags: skipped (SOP steps before it not performed), order_deviation, duration_outlier "
                      "(|z| > 2 vs the step's duration prior), low_margin (< 1 nat)",
            "needs_review_rule": f"lowest-confidence {frac:.0%} of entries (confidence <= {thr})",
            "n_entries": len(rows), "n_needs_review": sum(r["needs_review"] for r in rows), "entries": rows}


def segment_all(cfg: dict, od, element_ergo, taxonomy: list[dict]):
    intake = load_intake(cfg)
    sops = {s: (intake_station(intake, s) or {}).get("sop") for s in cfg["data"]["stations"]}
    entries, origin = GIO.scene_cycles(cfg)
    sdir = _std_dir(cfg)
    std = _load_std(sdir)
    primary_std = mode(cfg, "std_times", "none").startswith("loo_dir:")
    cycles, feats, masks_by, erg_cycles = _segment(cfg, entries, sops, element_ergo, std if primary_std else None)
    meta = {"segmentation": "sop", "used_annotation_times": False, "intake": cfg.get("intake"),
            "video_evidence": video_evidence(cfg),
            "std_times": "loo" if primary_std else "none",
            "algorithm": ALGORITHM, "rate_hz": RATE_HZ, "joints": origin,
            "joints_note": "TEMPORARY fallback: reference (mocap) s1 joints, video s1 not available yet"
                           if origin.startswith("fallback") else "s1 joints of this run"}
    if primary_std:
        meta["std_times_dir"] = str(sdir.relative_to(ROOT))
    save_json(review_queue(cycles), od / "review_queue.json")
    have = {(c["station"], c["cycle_id"]) for c in cycles}
    moc = [m for m in GIO.mocap_cycles(cfg) if (m["station"], m["cycle_id"]) in have]
    for suffix, sd in [("", None)] + ([("_std", std)] if std else []):
        is_std = sd is not None
        meta_v = {**meta, "std_times": "loo" if is_std else "none"}
        if is_std:
            meta_v["std_times_dir"] = str(sdir.relative_to(ROOT))
        else:
            meta_v.pop("std_times_dir", None)
        same_as_primary = is_std == primary_std
        # SOP-only segmentation on the reference pipeline's mocap joints (separates segmentation from pose error)
        if moc:
            mc = cycles if (origin.startswith("fallback") and same_as_primary) else \
                _segment(cfg, moc, sops, element_ergo, sd)[0]
            save_json({"method": {**meta_v, "joints": "mocap (reference s1)"}, "taxonomy": taxonomy, "cycles": mc},
                      od / f"elements_mocap_sop{suffix}.json", contract="elements")
        if is_std:
            vc = cycles if same_as_primary else _segment(cfg, entries, sops, element_ergo, sd)[0]
            save_json({"method": meta_v, "taxonomy": taxonomy, "cycles": vc}, od / "elements_sop_std.json",
                      contract="elements")
            save_json(review_queue(vc), od / "review_queue_sop_std.json")
    return cycles, feats, masks_by, erg_cycles, meta
