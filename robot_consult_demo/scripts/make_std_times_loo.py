"""Emulate the customer's standard-work combination sheet (manual time per SOP step) WITHOUT leakage:
leave-one-cycle-out. For every annotated cycle, std_time_s of a step = median annotated duration of that step over
the OTHER cycles of the same station (steps with no other cycle get no std_time_s).

    python scripts/make_std_times_loo.py --config config/carda_generic.yaml
    -> intake/std_times_loo/<config stem>/<STATION>_<cycle_id>.yaml  (+ meta.yaml)

Cycle ids repeat across stations (e.g. 002_1003 in WS10 and WS20), so files are keyed <STATION>_<cycle_id>.
The s2 SOP mode reads them via config `mode.std_times: loo_dir:<path>` (primary) or, when the directory exists
for the run's out_root, as the secondary output elements_sop_std.json.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rcd.common.io import data_path, load_config  # noqa: E402
from rcd.s2_elements.data import parse_annotations  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    cfg = load_config(a.config)
    stem = Path(a.config).stem
    out = ROOT / "intake" / "std_times_loo" / stem
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for st in cfg["data"]["stations"]:
        xl = sorted(data_path(cfg, st.lower()).glob("annotation_data_*.xlsx"))
        if not xl:
            continue
        ann = parse_annotations(xl[0])
        durs = defaultdict(dict)                       # step -> {cycle: duration}
        for cid, sgs in ann.items():
            for sg in sgs:
                durs[sg.name][cid] = float(sg.end - sg.start)
        for cid in ann:
            steps = {}
            for name, d in durs.items():
                others = [v for c, v in d.items() if c != cid]
                if others:
                    steps[name] = round(float(np.median(others)), 2)
            (out / f"{st}_{cid}.yaml").write_text(yaml.safe_dump(
                {"station": st, "cycle_id": cid, "std_time_s": steps,
                 "basis": "leave-one-cycle-out median of the annotated step durations of the other cycles "
                          "(emulated standard-work combination sheet)"}, sort_keys=False))
            n += 1
    (out / "meta.yaml").write_text(yaml.safe_dump({"config": a.config, "out_root": cfg["out_root"],
                                                   "intake": cfg.get("intake"), "cycles": n}, sort_keys=False))
    print(out, n, "cycles")
    return 0


if __name__ == "__main__":
    sys.exit(main())
