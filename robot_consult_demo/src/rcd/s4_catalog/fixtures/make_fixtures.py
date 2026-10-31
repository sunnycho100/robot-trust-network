"""Development fixtures for s4/s6a/s9 (used only when the real upstream file in out/ does not exist).

Scene: WS30 of a moving car-door line. The worker fixes a car speaker into the door carrier with 4 screws on a
near-vertical inner panel (screw points ~0.15 m apart, 1.0-1.15 m high). Conveyor runs along +x at 40 mm/s, panel
plane ~ y = 0 (tilted 5 deg), worker stands at +y. WS20 has a pick&place group without s1 geometry (exercises the
default-geometry fallback). Written to fixtures/<stage>/<file> mirroring out/.

    .venv/bin/python src/rcd/s4_catalog/fixtures/make_fixtures.py
"""
import json
from pathlib import Path

import numpy as np

H = Path(__file__).resolve().parent
tilt = np.radians(5)
screws = []
for dx, dz in [(-0.075, 0.075), (0.075, 0.075), (0.075, -0.075), (-0.075, -0.075)]:
    z = 1.05 + dz
    screws.append([round(0.10 + dx, 4), round(-np.tan(tilt) * (z - 1.05), 4), round(z, 4)])

cycles = []
for k, cid in enumerate(["002_0921", "003_0921", "004_0921"]):
    jitter = np.array([0.01 * (k - 1), 0.0, 0.0])
    cycles.append({
        "cycle_id": cid, "duration_s": 48.0 + 2 * k,
        "video": {"in": None, "out": None, "fps": 30},
        "human": {"joints_npz": None, "joint_names": [], "fps": 60, "source": "fixture"},
        "conveyor": {"speed_mm_s": 40.0 + k, "method": "fixture", "direction_xy": [1.0, 0.0]},
        "objects": [
            {"name": "workpiece_contact", "positions_m": (np.array(screws) + jitter).round(4).tolist()},
            {"name": "line_side_supply", "positions_m": [[-1.3, 1.5, 0.95]]},
        ],
    })
sg = {"units": "m", "stations": [{
    "station": "WS30",
    "zones": [{"name": "work_zone", "polygon_xy": [[-0.30, 0.35], [0.50, 0.35], [0.50, 1.05], [-0.30, 1.05]]},
              {"name": "line_side", "polygon_xy": [[-1.6, 1.3], [-1.0, 1.3], [-1.0, 1.7], [-1.6, 1.7]]}],
    "cycles": cycles}]}


def task(st, code, cat, alloc, req=None):
    return {"station": st, "code": code, "element_key": f"{st}:{code}", "scores": {}, "total": 0.0,
            "category": cat, "rationale": "fixture", "allocation": alloc, "requirements": req or {}}


au = {"criteria": [], "tasks": [
    task("WS30", "GET_TOOL", "cobot_screwdriving", "robot", {"part_mass_kg": 0.005}),
    task("WS30", "REACH_FASTENER", "cobot_screwdriving", "robot", {"n_fasteners": 4, "precision_mm": 0.5, "conveyor_tracking": True}),
    task("WS30", "FASTEN", "cobot_screwdriving", "robot", {"n_fasteners": 4, "precision_mm": 0.5, "conveyor_tracking": True}),
    task("WS30", "GET_PART", "cobot_pick_place", "human_robot", {"part_mass_kg": 0.9}),
    task("WS30", "INSERT", "dexterous_manual", "human"),
    task("WS30", "UNOBS", "unobserved", "human"),
    task("WS20", "GET_PART", "cobot_pick_place", "human_robot", {"part_mass_kg": 1.2}),
    task("WS20", "INSERT", "dexterous_manual", "human"),
]}

for stage, fn, obj in [("s1", "scene_graph.json", sg), ("s3", "automatability.json", au)]:
    (H / stage).mkdir(exist_ok=True)
    (H / stage / fn).write_text(json.dumps(obj, indent=1))
print("fixtures written")
