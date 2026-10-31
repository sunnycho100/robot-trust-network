"""Discovery of CarDA cycles, joint loading and annotation-workbook parsing.

A *cycle* is one mocap recording `data/carda/<st>/xsens/xsens_<id>_*.bvh`. Joints are read from the s1
output `out/s1/human/<ST>_<id>.npz` when it exists (keys joints (T,J,3) m z-up, names, fps), otherwise
from the BVH with `rcd.common.bvh.load_bvh` (same frame convention).

Annotation workbook (one sheet per cycle, CarDA layout):
    col A subgoal name (only on the subgoal's first row), B activity id (1.1, 1.4.1 ...), C activity text,
    D start [s] (subgoal first row), E end [s] (last row of the subgoal block that has a value),
    F duration, G..N  EAWS posture durations (8 unlabeled columns),
    O,P and Q,R  two (duration, intensity 0-5) pairs (asymmetric postures).
The EAWS numbers are recorded once per subgoal (on its first row).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from rcd.common.bvh import load_bvh
from rcd.common.io import ROOT, data_path

EAWS_DUR_COLS = list(range(6, 14))          # G..N
EAWS_PAIR_COLS = [(14, 15), (16, 17)]       # (O,P), (Q,R)
EAWS_COL_LETTERS = {i: chr(ord("A") + i) for i in range(18)}


@dataclass
class Activity:
    aid: str
    text: str
    children: list["Activity"] = field(default_factory=list)


@dataclass
class Subgoal:
    name: str
    start: float
    end: float
    activities: list[Activity]
    eaws: dict[str, float | None]           # column letter -> duration (None = blank)
    eaws_intensity: dict[str, float | None]


@dataclass
class Cycle:
    station: str
    cycle_id: str
    bvh: Path
    joints: np.ndarray = None   # (T,J,3)
    names: list[str] = None
    fps: float = 60.0
    joint_source: str = ""
    subgoals: list[Subgoal] = field(default_factory=list)
    video: dict = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return self.joints.shape[0] / self.fps


CYCLE_RE = re.compile(r"xsens_(\d+)_WS\d+_\d{4}_(\d{2})_(\d{2})")


VIDEO_RE = re.compile(r"WS\d+-SN\d+-\w*?__(\d{2})_(\d{2})_\d{4}_(\d{2})_(\d{2})_(\d{2})")


def cycle_id(name: str) -> str | None:
    """'xsens_002_WS10_2023_09_21_...' -> '002_0921' (CONTRACTS: ids repeat across recording days).
    Video-only recordings 'WS30-SN<serial>-HD720dt__DD_MM_YYYY_HH_MM_SS...' -> 'HHMMSS_MMDD'."""
    m = CYCLE_RE.search(name)
    if m:
        return f"{m.group(1)}_{m.group(2)}{m.group(3)}"
    m = VIDEO_RE.search(name)
    return f"{m.group(3)}{m.group(4)}{m.group(5)}_{m.group(2)}{m.group(1)}" if m else None


def _num(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def _nest(rows: list[tuple[str, str]]) -> list[Activity]:
    """Build an activity tree from (id, text) rows using the dotted ids (1.4 is parent of 1.4.1)."""
    root: list[Activity] = []
    stack: list[tuple[int, Activity]] = []
    for aid, text in rows:
        depth = aid.count(".")
        a = Activity(aid, text)
        while stack and stack[-1][0] >= depth:
            stack.pop()
        (stack[-1][1].children if stack else root).append(a)
        stack.append((depth, a))
    return root


def parse_annotations(xlsx: Path) -> dict[str, list[Subgoal]]:
    import pandas as pd
    res: dict[str, list[Subgoal]] = {}
    for sheet, df in pd.read_excel(xlsx, sheet_name=None, header=None).items():
        cid = cycle_id(sheet) or (cycle_id(str(df.iloc[1, 3])) if df.shape[0] > 1 and df.shape[1] > 3 else None)
        if not cid:
            continue
        blocks = []
        cur = None
        for _, row in df.iloc[3:].iterrows():
            name = row.iloc[0]
            if isinstance(name, str) and name.strip():
                if cur:
                    blocks.append(cur)
                cur = {"name": name.strip(), "start": _num(row.iloc[3]), "end": _num(row.iloc[4]), "rows": [],
                       "eaws": {EAWS_COL_LETTERS[c]: _num(row.iloc[c]) for c in EAWS_DUR_COLS + [p[0] for p in EAWS_PAIR_COLS]
                                if c < len(row)},
                       "inten": {EAWS_COL_LETTERS[d]: _num(row.iloc[i]) for d, i in EAWS_PAIR_COLS if i < len(row)}}
            if cur is None:
                continue
            e = _num(row.iloc[4])
            if e is not None:
                cur["end"] = e
            aid, txt = row.iloc[1], row.iloc[2]
            if isinstance(txt, str) and txt.strip():
                aid = str(aid).strip() if aid is not None and str(aid) != "nan" else f"{len(cur['rows']) + 1}"
                cur["rows"].append((aid, " ".join(txt.split())))
        if cur:
            blocks.append(cur)
        sgs = []
        for b in blocks:
            if b["start"] is None or b["end"] is None:
                continue
            sgs.append(Subgoal(b["name"], b["start"], b["end"], _nest(b["rows"]), b["eaws"], b["inten"]))
        res[cid] = sgs
    return res


def load_joints(cfg: dict, station: str, cycle_id: str, bvh: Path) -> tuple[np.ndarray, list[str], float, str]:
    npz = ROOT / cfg["out_root"] / "s1" / "human" / f"{station}_{cycle_id}.npz"
    if npz.exists():
        d = np.load(npz, allow_pickle=True)
        return np.asarray(d["joints"], float), [str(n) for n in d["names"]], float(d["fps"]), f"s1:{npz.relative_to(ROOT)}"
    d = load_bvh(bvh)
    return d["joints"], d["names"], float(d["fps"]), f"bvh:{bvh.relative_to(ROOT)}"


def discover(cfg: dict) -> list[Cycle]:
    """Every station folder that exists and has BVH files; annotations attached when present."""
    cycles = []
    for st in cfg["data"]["stations"]:
        root = data_path(cfg, st.lower())
        bvhs = sorted((root / "xsens").glob("*.bvh")) if root.exists() else []
        if not bvhs:
            continue
        xl = sorted(root.glob("annotation_data_*.xlsx"))
        ann = parse_annotations(xl[0]) if xl else {}
        for f in bvhs:
            cid = cycle_id(f.name)
            J, names, fps, src = load_joints(cfg, st, cid, f)
            vids = [v for v in sorted((root / "mp4").glob("xsens_*.mp4")) if cycle_id(v.name) == cid] \
                if (root / "mp4").exists() else []
            c = Cycle(st, cid, f, J, names, fps, src, ann.get(cid, []),
                      {("in" if "_in_" in v.name else "out"): str(v.relative_to(ROOT)) for v in vids})
            cycles.append(c)
    return cycles
