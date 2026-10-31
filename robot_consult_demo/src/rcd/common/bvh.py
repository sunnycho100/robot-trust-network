"""Minimal BVH reader + forward kinematics for the CarDA Xsens exports.

CarDA BVH: millimetres, Y-up, per-joint channel order "Yrotation Xrotation Zrotation" (root also has
Xposition Yposition Zposition). The header's `Frames:` count is wrong in the cropped files, so the
number of frames is taken from the data rows. Output is converted to metres, Z-up:
    world = (x, -z, y) / 1000
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class Joint:
    name: str
    parent: int
    offset: np.ndarray
    channels: list[str] = field(default_factory=list)


def _parse(path: Path) -> tuple[list[Joint], float, np.ndarray]:
    lines = Path(path).read_text().splitlines()
    joints: list[Joint] = []
    stack: list[int] = []
    i = 0
    pending_end = False
    while not lines[i].strip().startswith("MOTION"):
        tok = lines[i].split()
        if tok and tok[0] in ("ROOT", "JOINT"):
            joints.append(Joint(tok[1], stack[-1] if stack else -1, np.zeros(3)))
        elif tok and tok[0] == "End":
            pending_end = True
        elif tok and tok[0] == "{":
            stack.append(-2 if pending_end else len(joints) - 1)
        elif tok and tok[0] == "}":
            if stack and stack[-1] == -2:
                pending_end = False
            stack.pop()
        elif tok and tok[0] == "OFFSET" and not pending_end:
            joints[-1].offset = np.array(list(map(float, tok[1:4])))
        elif tok and tok[0] == "CHANNELS":
            joints[-1].channels = tok[2:]
        i += 1
    frame_time = float(lines[i + 2].split(":")[1])
    rows = [l for l in lines[i + 3:] if l.strip()]
    data = np.array([np.fromstring(r, sep=" ") for r in rows])
    return joints, frame_time, data


def load_bvh(path: str | Path) -> dict:
    """Return {'names', 'fps', 'joints' (T,J,3) metres z-up, 'root_mm' (T,3) raw}."""
    joints, ft, data = _parse(Path(path))
    T = data.shape[0]
    col = 0
    glob_R = [None] * len(joints)
    glob_p = np.zeros((T, len(joints), 3))
    for j, jt in enumerate(joints):
        n = len(jt.channels)
        ch = data[:, col:col + n]
        col += n
        pos = np.tile(jt.offset, (T, 1))
        rot_axes, rot_vals = "", []
        for k, c in enumerate(jt.channels):
            if c.endswith("position"):
                pos[:, "XYZ".index(c[0])] = ch[:, k]
            else:
                rot_axes += c[0]
                rot_vals.append(ch[:, k])
        local = Rotation.from_euler(rot_axes, np.stack(rot_vals, 1), degrees=True) if rot_axes else Rotation.identity(T)
        if jt.parent < 0:
            glob_R[j] = local
            glob_p[:, j] = pos
        else:
            glob_R[j] = glob_R[jt.parent] * local
            glob_p[:, j] = glob_p[:, jt.parent] + glob_R[jt.parent].apply(jt.offset)
    zup = np.stack([glob_p[..., 0], -glob_p[..., 2], glob_p[..., 1]], -1) / 1000.0
    return {"names": [j.name for j in joints], "fps": 1.0 / ft, "joints": zup, "root_mm": data[:, :3]}
