"""Parse CarDA `cameras_calibration.xlsx`.

Sheet layout (one block per station):
    WS10
    CAMERA_IN10 | | | | CAMERA_OUT10
    Camera SN | K | R | T | Camera SN | K | R | T
    SN.....   | 'fx,0,cx,0,fy,cy,0,0,1' | 9 numbers row-major | 3 numbers | (same for OUT)
followed by a table "Cameras world (global) positions (X,Y,Z) (m)" with one row per camera.

Convention (checked numerically in run.py): x_cam = R @ X_world + T, so the camera centre is -R^T T,
which reproduces the listed world positions to a few cm. Some labels use Greek capitals (Ι, Ν).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import openpyxl

_GREEK = str.maketrans({"Ι": "I", "Ν": "N", "Ο": "O", "Τ": "T", "Ε": "E", "Α": "A"})


def _norm(v) -> str:
    return str(v).translate(_GREEK).strip().upper() if v is not None else ""


def _nums(s) -> np.ndarray:
    return np.array([float(x) for x in re.split(r"[,\s\[\]]+", str(s)) if x.strip()])


def parse(path: str | Path) -> dict:
    """Return {(station, side): {sn, K, R, T, listed_center}} plus the sheet name used."""
    wb = openpyxl.load_workbook(path, data_only=True)
    res: dict[tuple[str, str], dict] = {}
    listed: dict[tuple[str, str], np.ndarray] = {}
    for ws in wb.worksheets:
        rows = [list(r) for r in ws.iter_rows(values_only=True)]
        station, sides = None, {}
        for r in rows:
            c0 = _norm(r[0]) if r else ""
            m = re.fullmatch(r"WS(\d+)", c0)
            if m:
                station, sides = f"WS{m.group(1)}", {}
                continue
            # camera-name row: map column offset -> side
            names = {j: _norm(v) for j, v in enumerate(r) if v is not None}
            if any(re.fullmatch(r"CAMERA_(IN|OUT)\d+", n) for n in names.values()) and not any(
                    isinstance(v, str) and v.strip().startswith("[") for v in r):
                for j, n in names.items():
                    mm = re.fullmatch(r"CAMERA_(IN|OUT)(\d+)", n)
                    if mm:
                        sides[j] = mm.group(1).lower()
                continue
            if station and c0.startswith("SN") and sides:
                for j, side in sides.items():
                    try:
                        K = _nums(r[j + 1]).reshape(3, 3)
                        R = _nums(r[j + 2]).reshape(3, 3)
                        T = _nums(r[j + 3]).reshape(3)
                    except (ValueError, TypeError, IndexError):
                        continue
                    res[(station, side)] = {"sn": str(r[j]).strip(), "K": K, "R": R, "T": T}
                continue
            # world-position table: "CAMERA_IN10 | [x, y, z]"
            mm = re.fullmatch(r"CAMERA_(IN|OUT)(\d+)", c0)
            if mm and len(r) > 1 and r[1] is not None and "[" in str(r[1]):
                listed[(f"WS{mm.group(2)}", mm.group(1).lower())] = _nums(r[1])
    for k, v in res.items():
        v["listed_center"] = listed.get(k)
    return res
