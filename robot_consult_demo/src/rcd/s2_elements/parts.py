"""Part / tool assumptions (not measurable from mocap). Each entry states its basis so the report can show it.

Masses of the CarDA door components are not published; values are engineering estimates for typical OEM
front-door parts and are used only for (a) the MTM-UAS weight class (<=1 kg vs >1-8 kg) and (b) robot payload
sizing in s3/s4, where s4 adds the configured payload safety margin.
"""
from __future__ import annotations

FLEXIBLE = ("weather strip", "glass")   # parts with deformable gaskets/strips (hard for robotic insertion)
PARTS = [  # (keyword in the subgoal name, part, mass_kg, precision_mm, basis)
    ("speaker", "door speaker (165 mm woofer)", 0.8, 1.0,
     "estimate: 6.5-in automotive door woofers weigh ~0.5-1.0 kg; slot/screw-hole alignment ~+/-1 mm"),
    ("rearview mirror", "exterior rear-view mirror assembly", 1.3, 1.0,
     "estimate: power-folding door mirror assemblies ~1.0-1.6 kg"),
    ("tie rod", "lock tie rod", 0.15, 1.0, "estimate: steel rod with clips, <0.2 kg"),
    ("plug", "hole plugs", 0.01, 2.0, "estimate: rubber/plastic plugs"),
    ("carrier", "window-regulator / door-module carrier", 2.5, 1.0,
     "estimate: door-module carriers with regulator ~2-3.5 kg"),
    ("glass", "glass run channel / door glass component", 0.6, 2.0,
     "estimate: channel + gasket component, ~0.3-1 kg"),
    ("weather strip", "belt-line weather strip", 0.2, 2.0, "estimate: EPDM strip ~0.1-0.3 kg"),
    ("vertical coverage", "posterior vertical trim cover", 0.3, 1.0, "estimate: plastic trim ~0.2-0.4 kg"),
]
DEFAULT_PART = ("component", 0.5, 2.0, "default estimate when the subgoal names no known part")

SCREWDRIVER_EOAT = {"name": "OnRobot Screwdriver", "mass_kg": 2.5, "torque_nm": [0.15, 5.0], "screws": "M1.6-M6, <=50 mm",
                    "source": "https://onrobot.com/us/products/onrobot-screwdriver"}
GRIPPER_EOAT = {"name": "Robotiq 2F-85 adaptive gripper", "mass_kg": 0.925, "stroke_mm": 85,
                "source": "https://robotiq.com/products/2f85-140-adaptive-robot-gripper"}


def part_for(subgoal: str | None) -> dict:
    s = (subgoal or "").lower()
    for kw, name, kg, prec, basis in PARTS:
        if kw in s:
            return {"part": name, "mass_kg": kg, "precision_mm": prec, "basis": basis, "flexible": kw in FLEXIBLE}
    name, kg, prec, basis = DEFAULT_PART
    return {"part": name, "mass_kg": kg, "precision_mm": prec, "basis": basis, "flexible": False}
