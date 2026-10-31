"""Build a kinematic MuJoCo model (MjSpec) from a vendor ROS description transcribed into models/<name>.json.

Each JSON holds the serial chain exactly as in the vendor URDF/xacro: per joint the origin xyz and rpy (URDF fixed-axis
roll-pitch-yaw, R = Rz(yaw) Ry(pitch) Rx(roll)) relative to the parent link, the joint axis in the child frame,
position limits and velocity limits; plus the flange offset in the last link and the tool (flange-normal) axis.
Source URL and license are stored in the JSON. Meshes are NOT used: every link gets a capsule collision primitive
from its origin to the next joint origin (and to the flange), radius LINK_RADIUS_M.

Catalog `sim_model` values of the form "vendor_urdf:<stem>" refer to models/<stem>.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
LINK_RADIUS_M = 0.055   # assumption: capsule radius for 10-16 kg class cobot links (UR10e Menagerie capsules: 0.05-0.075)
PREFIX = "vendor_urdf:"


def load(stem: str) -> dict:
    return json.loads((HERE / f"{stem}.json").read_text())


def _quat(R: Rotation) -> list[float]:
    x, y, z, w = R.as_quat()
    return [w, x, y, z]


def tool_axis(d: dict) -> np.ndarray:
    if d["tool_axis"] != "auto":
        a = np.asarray(d["tool_axis"], float)
        return a / np.linalg.norm(a)
    j = d["joints"][-1]
    a = np.asarray(j["axis"], float)
    # direction of the last link offset, expressed in the last link frame -> the flange faces away from the wrist
    R = Rotation.from_euler("xyz", j["rpy"])
    off = R.inv().apply(np.asarray(j["xyz"], float))
    s = 1.0 if np.dot(a, off) >= 0 else -1.0
    return s * a


def child_spec(stem: str) -> mujoco.MjSpec:
    d = load(stem)
    spec = mujoco.MjSpec()
    spec.compiler.degree = False  # all angles below are radians
    spec.modelname = d["model"]
    body = spec.worldbody.add_body(name="base")
    J = d["joints"]
    body.add_geom(name="base_geom", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.09, J[0]["xyz"][2] / 2, 0],
                  pos=[0, 0, J[0]["xyz"][2] / 2], contype=1, conaffinity=1)
    parent = body
    for i, j in enumerate(J):
        b = parent.add_body(name=f"link_{i + 1}", pos=list(map(float, j["xyz"])),
                            quat=_quat(Rotation.from_euler("xyz", j["rpy"])))
        b.add_joint(name=j["name"], type=mujoco.mjtJoint.mjJNT_HINGE, axis=list(map(float, j["axis"])),
                    range=[np.radians(j["lower_deg"]), np.radians(j["upper_deg"])], limited=True)
        b.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[LINK_RADIUS_M * 0.9, 0, 0], mass=0.5)  # joint housing
        nxt = J[i + 1]["xyz"] if i + 1 < len(J) else d["flange_xyz"]
        if np.linalg.norm(nxt) > 1e-3:
            b.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[0, 0, 0] + list(map(float, nxt)),
                       size=[LINK_RADIUS_M, 0, 0], mass=1.0)
        parent = b
    ta = tool_axis(d)
    rot, _ = Rotation.align_vectors([ta], [[0, 0, 1]])
    parent.add_site(name="attachment_site", pos=list(map(float, d["flange_xyz"])), quat=_quat(rot))
    return spec


def reach_check(stem: str, n: int = 20000, seed: int = 0) -> dict:
    """Max distance from the J2 axis origin to the flange over random configurations vs the datasheet reach."""
    d = load(stem)
    m = child_spec(stem).compile()
    da = mujoco.MjData(m)
    rng = np.random.default_rng(seed)
    site = m.site("attachment_site").id
    b2 = m.body("link_2").id
    best = 0.0
    for _ in range(n):
        da.qpos[:] = rng.uniform(m.jnt_range[:, 0], m.jnt_range[:, 1])
        mujoco.mj_kinematics(m, da)
        best = max(best, float(np.linalg.norm(da.site_xpos[site] - da.xpos[b2])))
    return {"model_reach_mm": round(best * 1000, 1), "datasheet_reach_mm": d["datasheet_reach_mm"],
            "ratio": round(best * 1000 / d["datasheet_reach_mm"], 3),
            "method": "max |flange - J2 origin| over 20000 random joint configurations"}


if __name__ == "__main__":
    for p in sorted(HERE.glob("*.json")):
        print(p.stem, reach_check(p.stem), tool_axis(load(p.stem)).round(3))
