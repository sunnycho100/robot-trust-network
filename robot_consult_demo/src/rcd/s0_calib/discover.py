"""Discover CarDA stations / cycles on disk (shared by s0 and s1).

Layout per station: data/carda/<ws>/{mp4,xsens}/. cycle_id = "<NNN>_<MMDD>" from
`xsens_<NNN>_WS<xx>_<YYYY>_<MM>_<DD>_...` (the 3-digit id repeats across recording days in WS10).
Video side comes from the `_in_camera` / `_out_camera` tag; untagged videos (e.g. WS20 004/005) are
assigned to the side whose tagged videos of the same station look most similar (normalised correlation of
a mid-clip frame; static cameras see the same background). Missing station folders / sides are skipped.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import numpy as np

from rcd.common.io import ROOT, data_path

CYCLE_RE = re.compile(r"xsens_(\d+)_WS\d+_(\d{4})_(\d{2})_(\d{2})")


def cycle_id(name: str) -> str | None:
    m = CYCLE_RE.search(name)
    return f"{m.group(1)}_{m.group(3)}{m.group(4)}" if m else None


def rel(p: Path) -> str:
    """Path relative to the project root, without resolving the data symlink."""
    try:
        return str(Path(p).relative_to(ROOT))
    except ValueError:
        return str(p)


def _side_tag(name: str) -> str | None:
    m = re.search(r"_(in|out)_camera", name)
    return m.group(1) if m else None


@lru_cache(maxsize=None)
def _thumb(path: str) -> np.ndarray | None:
    import cv2
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, n // 3))
    ok, im = cap.read()
    cap.release()
    if not ok:
        return None
    g = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (160, 90)).astype(np.float32)
    g = g[:45]  # upper half: background (walls, racks); the conveyor/cart/worker change between clips
    return (g - g.mean()) / (g.std() + 1e-6)


def _infer_side(path: Path, tagged: dict[str, list[Path]]) -> tuple[str | None, dict]:
    t = _thumb(str(path))
    scores = {}
    for side, refs in tagged.items():
        cs = [float(np.mean(t * r)) for r in (_thumb(str(p)) for p in refs) if r is not None]
        if cs:
            scores[side] = max(cs)
    if not scores or t is None:
        return None, scores
    best = max(scores, key=scores.get)
    return (best if scores[best] > 0.5 else None), scores


def discover(cfg: dict) -> dict[str, list[dict]]:
    """{station: [{cycle_id, bvh, video_in, video_out, video_side_inferred}]} for stations present on disk."""
    out: dict[str, list[dict]] = {}
    for st in cfg["data"]["stations"]:
        sdir = data_path(cfg, st.lower())
        xs = sdir / "xsens"
        if not xs.is_dir():
            continue
        mp4s = sorted((sdir / "mp4").glob("*.mp4")) if (sdir / "mp4").is_dir() else []
        tagged: dict[str, list[Path]] = {}
        for v in mp4s:
            s = _side_tag(v.name)
            if s:
                tagged.setdefault(s, []).append(v)
        cycles = []
        for f in sorted(xs.glob("*.bvh")):
            cid = cycle_id(f.name)
            if not cid:
                continue
            prefix = CYCLE_RE.search(f.name).group(0)
            c = {"cycle_id": cid, "bvh": f, "video_in": None, "video_out": None, "video_side_inferred": {}}
            for v in mp4s:
                if not v.name.startswith(prefix + "_"):
                    continue
                side = _side_tag(v.name)
                if side is None:
                    side, sc = _infer_side(v, tagged)
                    c["video_side_inferred"][v.name] = {"side": side, "similarity": sc}
                if side and c[f"video_{side}"] is None:
                    c[f"video_{side}"] = v
            cycles.append(c)
        if cycles:
            out[st] = cycles
    return out



# Clip naming conventions (pattern table): regex -> cycle id and, when the name carries it, the camera serial.
#   'xsens_002_WS10_2023_09_21_in_camera_...'          -> '002_0921'   (side from the _in/_out_camera tag)
#   'WS30-SN22354094-HD720dt__09_05_2024_10_14_50_...' -> '101450_0509' (camera from the serial number)
CLIP_PATTERNS = [
    (re.compile(r"xsens_(\d+)_WS\d+_(\d{4})_(\d{2})_(\d{2})"), lambda m: f"{m.group(1)}_{m.group(3)}{m.group(4)}", None),
    (re.compile(r"WS\d+-SN(\d+)-\w*?__(\d{2})_(\d{2})_\d{4}_(\d{2})_(\d{2})_(\d{2})"),
     lambda m: f"{m.group(4)}{m.group(5)}{m.group(6)}_{m.group(3)}{m.group(2)}", lambda m: f"SN{m.group(1)}"),
]


def clip_info(name: str) -> tuple[str, str | None]:
    """(cycle id, camera serial or None) from a clip file name; unknown patterns -> stem without view tags."""
    for rx, cid, ser in CLIP_PATTERNS:
        m = rx.search(name)
        if m:
            return cid(m), (ser(m) if ser else None)
    stem = Path(name).stem
    stem = re.sub(r"_(in|out)_camera", "", stem)
    return re.sub(r"_(cropped|L_view|R_view)", "", stem), None


def _norm_serial(s) -> str:
    d = re.sub(r"\D", "", str(s or ""))
    return d


def discover_videos(cfg: dict, serials: dict | None = None) -> dict[str, list[dict]]:
    """Like `discover` but enumerates cycles from the video clips only (no mocap files are touched).

    {station: [{cycle_id, bvh: None, video_in, video_out, video_side_inferred}]}; a cycle is one recording seen by
    one or more cameras. Camera side: `_in_camera`/`_out_camera` tag, else the serial number in the name looked up
    in `serials` ({serial: (station, side)}, from the calibration), else inferred from the background similarity
    to tagged clips of the same station."""
    ser = {_norm_serial(k): v for k, v in (serials or {}).items()}
    out: dict[str, list[dict]] = {}
    for st in cfg["data"]["stations"]:
        sdir = data_path(cfg, st.lower())
        mp4s = sorted((sdir / "mp4").glob("*.mp4")) if (sdir / "mp4").is_dir() else []
        if not mp4s:
            continue
        tagged: dict[str, list[Path]] = {}
        sides: dict[Path, str | None] = {}
        for v in mp4s:
            s = _side_tag(v.name)
            if s is None:
                sn = clip_info(v.name)[1]
                hit = ser.get(_norm_serial(sn)) if sn else None
                if hit and hit[0] == st:
                    s = hit[1]
            sides[v] = s
            if s:
                tagged.setdefault(s, []).append(v)
        groups: dict[str, list[Path]] = {}
        for v in mp4s:
            groups.setdefault(clip_info(v.name)[0], []).append(v)
        cycles = []
        for cid in sorted(groups):
            c = {"cycle_id": cid, "bvh": None, "video_in": None, "video_out": None, "video_side_inferred": {}}
            for v in groups[cid]:
                side = sides[v]
                if side is None:
                    side, sc = _infer_side(v, tagged)
                    c["video_side_inferred"][v.name] = {"side": side, "similarity": sc}
                if side and c[f"video_{side}"] is None:
                    c[f"video_{side}"] = v
            cycles.append(c)
        out[st] = cycles
    return out
