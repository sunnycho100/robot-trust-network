"""Run the whole pipeline (or selected stages) in dependency order.

    python scripts/run_all.py                 # every stage
    python scripts/run_all.py --stage s2 s3   # selected stages (inputs must already exist in out/)

Each stage lives in src/rcd/<stage>/run.py and exposes run(cfg) -> None.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rcd.common.io import load_config  # noqa: E402

ORDER = [
    ("s0", "rcd.s0_calib.run"),
    ("s1", "rcd.s1_scene.run"),
    ("s2", "rcd.s2_elements.run"),
    ("s3", "rcd.s3_automatability.run"),
    ("s4", "rcd.s4_catalog.run"),
    ("s6a", "rcd.s6a_robot_sim.run"),
    ("s5", "rcd.s5_optimize.run"),
    ("s6b", "rcd.s6b_des.run"),
    ("s7", "rcd.s7_roi.run"),
    ("s5x", "rcd.s5_optimize.explore"),
    ("s9", "rcd.s9_vla.run"),
    ("s8", "rcd.s8_report.run"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", nargs="*", default=[s for s, _ in ORDER])
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    log = {}
    t_all = time.time()
    for stage, mod in ORDER:
        if stage not in a.stage:
            continue
        t0 = time.time()
        print(f"=== {stage} ({mod})", flush=True)
        importlib.import_module(mod).run(cfg)
        log[stage] = round(time.time() - t0, 2)
        print(f"    done in {log[stage]}s", flush=True)
    log["total_s"] = round(time.time() - t_all, 2)
    out = ROOT / cfg["out_root"]
    out.mkdir(exist_ok=True)
    (out / "run_log.json").write_text(json.dumps(log, indent=2))
    print(json.dumps(log))
    return 0


if __name__ == "__main__":
    sys.exit(main())
