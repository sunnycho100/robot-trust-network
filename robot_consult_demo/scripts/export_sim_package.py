"""Write the simulation hand-off package (out/sim_package/) from the current pipeline outputs.

    python scripts/export_sim_package.py
    cd out/sim_package && python replay.py WS30 --video
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rcd.common.io import load_config  # noqa: E402
from rcd.sim_package import export  # noqa: E402

if __name__ == "__main__":
    export.run(load_config())
