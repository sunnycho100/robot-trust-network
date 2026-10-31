"""Customer intake: the facts a factory gives us that video cannot measure (contract: contracts/intake.schema.json).

The intake replaces every dataset-specific constant in the pipeline (part list, standard-work text, takt, layout hints).
Stages read it via `load_intake(cfg)`; when a config has no `intake` key the original CarDA behaviour is used.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from .io import ROOT, validate


def mode(cfg: dict, key: str, default: str) -> str:
    return cfg.get("mode", {}).get(key, default)


def load_intake(cfg: dict) -> dict | None:
    p = cfg.get("intake")
    if not p:
        return None
    with open(ROOT / p) as f:
        it = yaml.safe_load(f)
    validate(it, "intake")
    return it


def station(intake: dict, name: str) -> dict | None:
    return next((s for s in intake["stations"] if s["name"] == name), None)


def part_for(intake: dict, station_name: str, text: str | None) -> dict:
    """Match a step/subgoal text to the station's declared parts by keyword; fall back to the station default."""
    st = station(intake, station_name) or {}
    t = (text or "").lower()
    for p in st.get("parts", []):
        if any(k.lower() in t for k in p.get("keywords", [p["name"]])):
            return {**p, "source": "intake"}
    d = st.get("default_part") or intake.get("default_part")
    return {**d, "source": "intake"}
