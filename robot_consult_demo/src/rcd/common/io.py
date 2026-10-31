"""Shared paths, config loading, and contract-checked JSON I/O.

Every stage reads its inputs and writes its outputs only through `load_json` / `save_json`,
so a teammate's real implementation can replace a stage as long as it honours the same contract
in `contracts/<name>.schema.json`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema
import yaml

ROOT = Path(__file__).resolve().parents[3]
CONTRACTS = ROOT / "contracts"


def load_config(path: str | Path | None = None) -> dict:
    p = Path(path) if path else ROOT / "config" / "demo.yaml"
    with open(p) as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = str(ROOT)
    return cfg


def out_dir(cfg: dict, stage: str) -> Path:
    d = ROOT / cfg["out_root"] / stage
    d.mkdir(parents=True, exist_ok=True)
    return d


def data_path(cfg: dict, *parts: str) -> Path:
    return ROOT / cfg["data"]["root"] / Path(*parts)


def _schema(name: str) -> dict:
    with open(CONTRACTS / f"{name}.schema.json") as f:
        return json.load(f)


def validate(obj: Any, contract: str) -> None:
    jsonschema.validate(obj, _schema(contract))


def save_json(obj: Any, path: str | Path, contract: str | None = None) -> Path:
    if contract:
        validate(obj, contract)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=float)
    return p


def load_json(path: str | Path, contract: str | None = None) -> Any:
    with open(path) as f:
        obj = json.load(f)
    if contract:
        validate(obj, contract)
    return obj
