"""Upstream-input resolution for the decision stages (s5, s6b, s7, s8).

Each upstream file is read from `out/<stage>/<file>` when it exists (the real pipeline output). If it does not
exist yet, a contract-valid fixture from `src/rcd/s5_optimize/fixtures/out/<stage>/<file>` is used instead, and
the provenance is recorded so that every downstream artefact says which inputs were real and which were fixtures.
Fixtures are never written into `out/` (they belong to other stage owners).
"""
from __future__ import annotations

from pathlib import Path

from rcd.common.io import ROOT, load_json

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "out"

# (stage, file, contract)
UPSTREAM = {
    "calibration": ("s0", "calibration.json", "calibration"),
    "scene_graph": ("s1", "scene_graph.json", "scene_graph"),
    "elements": ("s2", "elements.json", "elements"),
    "ergonomics": ("s2", "ergonomics.json", "ergonomics"),
    "time_study": ("s2", "time_study.json", "time_study"),
    "automatability": ("s3", "automatability.json", "automatability"),
    "robot_catalog": ("s4", "robot_catalog.json", "robot_catalog"),
    "candidates": ("s4", "candidates.json", "candidates"),
    "robot_task_times": ("s6a", "robot_task_times.json", "robot_task_times"),
    "task_plan": ("s9", "task_plan.json", "task_plan"),
    "allocation": ("s5", "allocation.json", "allocation"),
    "des_results": ("s6b", "des_results.json", "des_results"),
    "roi": ("s7", "roi.json", "roi"),
    "design_space": ("s5x", "design_space.json", "design_space"),
}


def real_path(cfg: dict, name: str) -> Path:
    stage, fn, _ = UPSTREAM[name]
    return ROOT / cfg["out_root"] / stage / fn


def load_input(cfg: dict, name: str, provenance: dict | None = None, required: bool = True):
    """Load an upstream artefact (real if present, else fixture). Records 'real'/'fixture' in provenance."""
    stage, fn, contract = UPSTREAM[name]
    p = real_path(cfg, name)
    src = "real"
    if not p.exists():
        p = FIXTURE_ROOT / stage / fn
        src = "fixture"
        if not p.exists():
            if required:
                raise FileNotFoundError(f"neither out/{stage}/{fn} nor fixture {p} exists")
            if provenance is not None:
                provenance[name] = "missing"
            return None
    try:
        obj = load_json(p, contract=contract)
    except Exception:
        # A real upstream file that violates its contract is still read (the gate flags it for that stage);
        # we do not silently replace real data by a fixture.
        obj = load_json(p)
        src += " (contract violation)"
    if provenance is not None:
        provenance[name] = src
    return obj
