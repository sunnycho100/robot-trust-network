"""Fit the pose-source calibration (video pose -> mocap reference) on paired recordings.

    python scripts/fit_pose_calib.py [--test config/carda_generic.yaml] [--ref config/demo_regress.yaml]
    -> models/pose_calib/video.json

Pairs = cycles present in both the test run's s1 (video pose) and the reference run's s1 (mocap). Features are the
ones the SOP-mode s2 uses (generic_io canonicalisation + RTS smoothing, features.extract). Maps are fitted on all
stations ("all") and leave-one-station-out ("exclude:<ST>" = fitted on the other stations).
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
from rcd.s2_elements import generic_io as GIO  # noqa: E402
from rcd.s2_elements.features import RATE_HZ, extract  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="config/carda_generic.yaml")
    ap.add_argument("--ref", default="config/demo_regress.yaml")
    ap.add_argument("--source", default="video")
    a = ap.parse_args()
    tcfg, rcfg = load_config(ROOT / a.test), load_config(ROOT / a.ref)
    test, origin = GIO.scene_cycles(tcfg)
    if origin != "s1":
        raise SystemExit("test run has no s1 of its own")
    ref = {(e["station"], e["cycle_id"]): e for e in GIO.mocap_cycles(rcfg)}
    pairs = {}
    for en in test:
        r = ref.get((en["station"], en["cycle_id"]))
        if not r:
            continue
        jt = GIO.load_cycle_joints(en)
        ft = extract(jt["joints"], jt["names"], jt["fps"])
        step = max(1, int(round(jt["fps"] / RATE_HZ)))
        v = jt["valid"][: ft["n"] * step].reshape(-1, step).mean(1) >= 0.5 if step > 1 else jt["valid"][: ft["n"]]
        jr = GIO.load_cycle_joints(r)
        fr = extract(jr["joints"], jr["names"], jr["fps"])
        pairs[(en["station"], en["cycle_id"])] = (ft, fr, v)
    stations = sorted({k[0] for k in pairs})
    maps = {"all": GC.fit(list(pairs.values()))}
    for st in stations:
        maps[f"exclude:{st}"] = GC.fit([p for k, p in pairs.items() if k[0] != st])
    sg = ROOT / tcfg["out_root"] / "s1" / "scene_graph.json"
    out = {"source": a.source, "method": "monotone empirical quantile mapping test->reference per feature group "
                                        "(201 quantiles), fitted on paired recordings of the same work",
           "groups": GC.GROUPS, "reference": "mocap (reference s1 of " + rcfg["out_root"] + ")",
           "test": f"{tcfg['out_root']}/s1 human joints (s2 generic_io: canonicalised, RTS-smoothed)",
           "test_scene_graph_mtime": datetime.datetime.fromtimestamp(sg.stat().st_mtime).isoformat(timespec="seconds"),
           "fitted_at": datetime.datetime.now().isoformat(timespec="seconds"),
           "pairs": [f"{s}_{c}" for s, c in sorted(pairs)], "stations": stations,
           "note": "once per pose model / camera type, not per customer; 'exclude:<ST>' maps are used to evaluate "
                   "station <ST> without it being in the fit",
           "maps": maps}
    GC.CALIB_DIR.mkdir(parents=True, exist_ok=True)
    p = GC.CALIB_DIR / f"{a.source}.json"
    p.write_text(json.dumps(out, indent=1))
    print(p, len(pairs), "pairs;", {g: (m["test_median"], m["ref_median"]) for g, m in maps["all"].items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
