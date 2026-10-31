"""Process knowledge shared by s4 (requirements, placement), s6a (timing) and s9 (task plan).

Everything that is an assumption is a named constant here with its source, so the three stages agree.

Process types are derived from the s3 category string by keyword (screwdriving / pick_place / inspection);
any other robot category is treated as generic pick_place handling and flagged.
"""
from __future__ import annotations

import re

# ------------------------------------------------------------------ end-of-arm tooling (raw mass, no margin)
TOOLS = {
    "screwdriving": {
        "name": "OnRobot Screwdriver (with screw presenter at the pedestal)",
        "mass_kg": 2.5, "tcp_offset_m": 0.30,
        "source": "https://onrobot.com/en/products/onrobot-screwdriver (2.5 kg, 0.15-5 Nm, max 340 rpm, M1.6-M6)",
        "note": "TCP offset (flange to bit tip incl. 25 mm bit) is an assumption from the published drawing scale",
    },
    "pick_place": {
        "name": "Robotiq 2F-85 adaptive gripper",
        "mass_kg": 0.925, "tcp_offset_m": 0.16,
        "source": "https://blog.robotiq.com/hubfs/Product-sheets/Adaptive%20Grippers/Product-sheet-Adaptive-Grippers-EN.pdf "
                  "(85 mm stroke, 20-150 mm/s, 20-235 N, 5 kg form-fit payload)",
        "note": "mass 0.925 kg and fingertip TCP 0.16 m from the product sheet drawing (TCP rounded)",
    },
    "inspection": {
        "name": "wrist-mounted 2D industrial camera + bracket",
        "mass_kg": 0.35, "tcp_offset_m": 0.10,
        "source": "assumption: compact GigE/USB3 industrial camera with lens and bracket <= 0.35 kg",
        "note": "camera optical standoff from the surface is VIEW_STANDOFF_M",
    },
}
ADAPTER_KG = 0.10        # tool flange adapter / quick-changer plate (assumption)

# ------------------------------------------------------------------ process times (all in seconds)
SCREW = {
    "rpm": 340,                 # OnRobot Screwdriver max output speed (datasheet)
    "thread_pitch_mm": 0.8,     # M5 coarse thread (ISO 261); M4 = 0.7
    "engagement_mm": 10.0,      # assumption: 10 mm thread engagement in a plastic carrier / sheet-metal nut
    "torque_up_s": 0.3,         # assumption: final tightening to torque + bit release
    "pick_dwell_s": 0.3,        # assumption: bit engages screw in the presenter (vacuum/magnetic pick)
    "source": "https://onrobot.com/storage/datasheets/datasheet_screwdriver.pdf",
}


def rundown_s() -> float:
    """Run-down time = turns / (rpm/60) + torque-up; 10 mm / 0.8 mm = 12.5 turns at 340 rpm = 2.21 s (+0.3 s)."""
    turns = SCREW["engagement_mm"] / SCREW["thread_pitch_mm"]
    return turns / (SCREW["rpm"] / 60.0) + SCREW["torque_up_s"]


GRIPPER = {"speed_mm_s": 150.0, "stroke_mm": 40.0, "settle_s": 0.10,
           "source": "Robotiq 2F-85 product sheet: closing speed 20-150 mm/s; 40 mm stroke used (part width + clearance)"}


def gripper_s() -> float:
    return GRIPPER["stroke_mm"] / GRIPPER["speed_mm_s"] + GRIPPER["settle_s"]


CAPTURE_S = 0.30          # assumption: exposure + transfer + 2D check per view
VIEW_STANDOFF_M = 0.25    # assumption: camera working distance

TRACK_SYNC_S = 0.30       # Doosan DRL tracking_conveyor(conv_id, time=0.3): default accel/decel time to sync with the belt
TRACK_SYNC_SOURCE = "https://manual.doosanrobotics.com/en/programming-manual/3.3.0/publish/tracking_conveyor-conv_id-time-0-3"

# ------------------------------------------------------------------ motion limits (timing model)
JOINT_ACCEL_RAD_S2 = 1.4  # URScript movej default a=1.4 rad/s^2 (applied to every joint; conservative for all makers)
JOINT_ACCEL_SOURCE = "https://www.universal-robots.com/manuals/EN/HTML/SW5_26/Content/prod-scriptmanual/all_scripts/movej_qa14v105t0r.htm"
COLLAB_TCP_CAP_M_S = 1.0  # free-space moves capped at 1.0 m/s TCP (FANUC CRX collaborative-mode TCP speed, catalog RCRX(E)-02)
LIN_SPEED_M_S = 0.25      # approach / retract linear speed near the part (assumption, = URScript movel default v)
LIN_ACCEL_M_S2 = 1.2      # linear acceleration (assumption, = URScript movel default a)
APPROACH_OFFSET_M = 0.06  # stand-off of approach/retract poses from the contact point along the surface normal
FEEDER_LIFT_M = 0.05      # vertical lift above the screw presenter / part tray

# ------------------------------------------------------------------ geometry assumptions used by the placement search
SHOULDER_OFFSET_M = 0.18  # generic height of the J2 (shoulder) axis above the mounting flange (UR10e 0.181 m)
PEDESTAL_HEIGHTS_M = [0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
BASE_RADIUS_M = 0.20      # pedestal half-width (0.4 m plate) used for clearances
DOOR_CLEARANCE_M = 0.35   # minimum distance of the base centre in front of the panel plane (carrier passes behind)
WORKER_CLEARANCE_M = 0.25 # minimum distance of the pedestal edge from the observed worker zone polygon
ZONE_CLEARANCE_M = 0.10   # pedestal edge clearance from the belt edge and the line-side supply zone
MIN_REACH_M = 0.30        # inner dead zone around the shoulder
TRAY_RADIUS_M = 0.45      # screw presenter / part tray placed at this horizontal radius from the base
TRAY_BELOW_SHOULDER_M = 0.30
# 7th-axis linear track parallel to the conveyor: used when a fixed floor base would need more reach than the median
# usable reach of the collaborative, tracking-capable catalog models (the base then travels with the carrier).
TRACK_RETURN_SPEED_M_S = 0.5   # assumption: collaborative-speed return of the track carriage after the cycle
TRACK_WIDTH_M = 0.6            # assumption: track + carriage width (floor area)

# ------------------------------------------------------------------ cell space (ISO 13855 + Facilities Planning)
ISO13855 = {
    "K_mm_s": 1600.0,
    "scanner_response_s": 0.070,
    "scanner_source": "https://www.sick.com/media/pdf/9/79/979/dataSheet_NANS3-AAAZ30AN1_1100333_en.pdf (nanoScan3 response time 70 ms)",
    "H_mm": 300.0,           # detection-plane height of a floor-mounted scanner (70 mm resolution -> H >= 15(d-50) = 300 mm)
    "default_stop_s": 0.50,  # assumption when the maker does not publish stopping times (UR/Doosan published values are 0.24-0.36 s)
    "source": "ISO 13855 parallel approach (area scanner): S = K*T + C, K = 1600 mm/s, C = 1200 - 0.4H >= 850 mm; "
              "https://www.keyence.com/ss/products/safetyknowledge/caution/scanner/ , "
              "https://int.reersafety.com/academy/industrial-safety-guide/determination-of-the-safety-distance/",
}
PEDESTAL_M2 = 0.6 * 0.6      # floor plate of a standard cobot pedestal (assumption)
PRESENTER_M2 = 0.4 * 0.6     # screw presenter / KLT part tray stand (VDA KLT 600x400)
LARGEST_LOAD_FT2 = 0.6 * 0.4 * 10.7639  # largest load moved through the aisle = KLT tray 600x400 mm = 2.58 ft^2
# Tompkins et al., Facilities Planning, Table 3.3 "Aisle Allowance Estimates" (ISyE 510 lecture07, slide 7)
AISLE_TABLE = [(6.0, (0.05, 0.10)), (12.0, (0.10, 0.20)), (18.0, (0.20, 0.30)), (float("inf"), (0.30, 0.40))]


def aisle_allowance(largest_load_ft2: float = LARGEST_LOAD_FT2) -> tuple[float, str]:
    for ub, (lo, hi) in AISLE_TABLE:
        if largest_load_ft2 < ub:
            return hi, (f"Tompkins Facilities Planning Table 3.3 (ISyE 510 lecture07): largest load "
                        f"{largest_load_ft2:.2f} ft^2 -> {lo:.0%}-{hi:.0%}; upper bound used (conservative)")
    return 0.40, "Table 3.3 > 18 ft^2"


# ------------------------------------------------------------------ categories
def process_type(category: str) -> str:
    c = (category or "").lower()
    if re.search(r"screw|fasten|bolt|nut", c):
        return "screwdriving"
    if re.search(r"inspect|vision|camera|check", c):
        return "inspection"
    return "pick_place"  # pick/place, handling, insertion, kitting, machine tending ...


# Element codes (s2 taxonomy) realised by each segment of a robot program, by process type.
# The first code of a list present in the group is used; otherwise the group's first code.
CODE_PREF = {
    "fetch": ["GET_TOOL", "GET_PART", "REACH_FASTENER"],
    "reach": ["REACH_FASTENER", "GET_PART", "INSERT"],
    "fasten": ["FASTEN"],
    "pick": ["GET_PART"],
    "place": ["INSERT", "GET_PART"],
    "inspect": ["INSPECT"],
}


def code_for(role: str, codes: list[str]) -> str:
    for c in CODE_PREF.get(role, []):
        if c in codes:
            return c
    return codes[0] if codes else role.upper()


def program(ptype: str, n_targets: int, codes: list[str]) -> list[dict]:
    """Abstract robot program (one cycle) for a task group.

    Ops: move_j / move_l (to a named target), process (screw / grip / capture with a role), track_start / track_stop.
    Targets: 'home', 'tray_above', 'tray', 'app_<i>', 'tgt_<i>', 'view_<i>'. 'tray*' are fixed in the world
    (screw presenter / part tray at the pedestal); 'app_*', 'tgt_*', 'view_*' move with the conveyor.
    """
    C = lambda r: code_for(r, codes)  # noqa: E731
    ops: list[dict] = []
    if ptype == "screwdriving":
        for i in range(n_targets):
            ops += [
                {"op": "move_j", "target": "tray_above", "code": C("fetch")},
                {"op": "move_l", "target": "tray", "code": C("fetch")},
                {"op": "process", "kind": "screw_pick", "code": C("fetch")},
                {"op": "move_l", "target": "tray_above", "code": C("fetch")},
                {"op": "move_j", "target": f"app_{i}", "code": C("reach")},
                {"op": "track_start", "code": C("reach")},
                {"op": "move_l", "target": f"tgt_{i}", "code": C("reach")},
                {"op": "process", "kind": "rundown", "target": f"tgt_{i}", "code": C("fasten")},
                {"op": "move_l", "target": f"app_{i}", "code": C("fasten")},
                {"op": "track_stop", "code": C("fasten")},
            ]
    elif ptype == "inspection":
        ops.append({"op": "track_start", "code": C("inspect")})
        for i in range(n_targets):
            ops += [{"op": "move_j", "target": f"view_{i}", "code": C("inspect")},
                    {"op": "process", "kind": "capture", "target": f"view_{i}", "code": C("inspect")}]
        ops.append({"op": "track_stop", "code": C("inspect")})
    else:  # pick_place: one part per cycle from the robot-side tray to the seat on the workpiece
        for i in range(n_targets):
            ops += [
                {"op": "move_j", "target": "tray_above", "code": C("pick")},
                {"op": "process", "kind": "grip_open", "code": C("pick")},
                {"op": "move_l", "target": "tray", "code": C("pick")},
                {"op": "process", "kind": "grip_close", "code": C("pick")},
                {"op": "move_l", "target": "tray_above", "code": C("pick")},
                {"op": "move_j", "target": f"app_{i}", "code": C("place")},
                {"op": "track_start", "code": C("place")},
                {"op": "move_l", "target": f"tgt_{i}", "code": C("place")},
                {"op": "process", "kind": "grip_open", "target": f"tgt_{i}", "code": C("place")},
                {"op": "move_l", "target": f"app_{i}", "code": C("place")},
                {"op": "track_stop", "code": C("place")},
            ]
    ops.append({"op": "move_j", "target": "home", "code": C("reach")})
    return ops


def process_time(kind: str) -> float:
    return {"screw_pick": SCREW["pick_dwell_s"], "rundown": rundown_s(), "grip_open": gripper_s(),
            "grip_close": gripper_s(), "capture": CAPTURE_S}[kind]


BREAKDOWN_KEY = {"move_j": "move_free", "move_l": "approach_retract", "track_start": "tracking_sync",
                 "track_stop": "tracking_sync", "screw_pick": "screw_pick", "rundown": "rundown",
                 "grip_open": "gripper", "grip_close": "gripper", "capture": "image_capture"}


def analytic_cycle_estimate(ops: list[dict]) -> float:
    """Rough pre-estimate (used only to size the conveyor-tracking window before simulation):
    1.2 s per free move, 0.45 s per linear move, plus process and sync times."""
    t = 0.0
    for o in ops:
        if o["op"] == "move_j":
            t += 1.2
        elif o["op"] == "move_l":
            t += 0.45
        elif o["op"] in ("track_start", "track_stop"):
            t += TRACK_SYNC_S
        else:
            t += process_time(o["kind"])
    return t


def assumptions_table() -> list[dict]:
    return [
        {"name": "screw run-down", "value": round(rundown_s(), 3), "unit": "s/screw",
         "basis": f"{SCREW['engagement_mm']} mm / {SCREW['thread_pitch_mm']} mm pitch at {SCREW['rpm']} rpm + {SCREW['torque_up_s']} s torque-up",
         "source": SCREW["source"]},
        {"name": "screw pick at presenter", "value": SCREW["pick_dwell_s"], "unit": "s", "basis": "assumption", "source": None},
        {"name": "gripper actuation", "value": round(gripper_s(), 3), "unit": "s", "basis": GRIPPER["source"], "source": TOOLS["pick_place"]["source"]},
        {"name": "image capture", "value": CAPTURE_S, "unit": "s", "basis": "assumption", "source": None},
        {"name": "conveyor-tracking sync", "value": TRACK_SYNC_S, "unit": "s per start/stop", "basis": "Doosan default", "source": TRACK_SYNC_SOURCE},
        {"name": "joint acceleration", "value": JOINT_ACCEL_RAD_S2, "unit": "rad/s^2", "basis": "URScript movej default", "source": JOINT_ACCEL_SOURCE},
        {"name": "joint speed limits", "value": "datasheet per-joint maxima", "unit": "deg/s", "basis": "s4 catalog", "source": None},
        {"name": "free-move TCP cap", "value": COLLAB_TCP_CAP_M_S, "unit": "m/s", "basis": "FANUC CRX collaborative-mode TCP speed", "source": "https://www.fanuc.com/fmm/my/product/catalog/RCRX(E)-02.pdf"},
        {"name": "linear approach speed / accel", "value": [LIN_SPEED_M_S, LIN_ACCEL_M_S2], "unit": "m/s, m/s^2", "basis": "assumption (URScript movel defaults)", "source": None},
    ]
