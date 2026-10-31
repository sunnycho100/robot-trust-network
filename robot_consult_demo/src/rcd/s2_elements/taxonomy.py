"""Work-element taxonomy grounded in Gilbreth's therbligs and MTM-UAS basic operations.

Each element is a *minimum rational work element* (ISyE 510 lecture06: smallest feasible division of work
that can still be assigned to one resource), i.e. one MTM-UAS basic operation group performed on one
object. Therbligs: Gilbreth & Gilbreth (1921), "Motion Study"; MTM-UAS: MTM Association (1976),
basic-operation groups Get&Place (A*), Place (P*), Handle tool (H*), Operate (B*), Motion cycles (Z*),
Body motions (K*), Visual control (VA).

The activity lexicon maps the free-text CarDA activity list ("Take the speaker from the cart",
"Fix 4 screws ...") to an ordered *standard work sequence* of element codes. That sequence is the
script that the mocap segmenter aligns to the motion (segment.py).
"""
from __future__ import annotations

import re

from .data import Activity

TAXONOMY = [
    {"code": "WALK", "name": "Walk / transport",
     "therblig": ["TE (transport empty)", "TL (transport loaded)"], "mtm_uas": "KA (walk, per metre)",
     "description": "Body displacement between line side (cart, tool rack) and the workpiece on the carrier; "
                    "loaded when a part/tool is carried.", "lean_class": "NVA", "waste": "Motion"},
    {"code": "GET_PART", "name": "Get part",
     "therblig": ["TE", "S (search)", "SE (select)", "G (grasp)", "TL"], "mtm_uas": "AA-AF (get and place)",
     "description": "Reach to a part container/cart at the line side, select and grasp the part.",
     "lean_class": "NNVA", "waste": None},
    {"code": "INSERT", "name": "Position / insert part",
     "therblig": ["P (position)", "A (assemble)", "RL (release load)"], "mtm_uas": "PB/PC (place loose/tight)",
     "description": "Align the part with its mating features on the door/carrier and insert it (clips, slots, plugs).",
     "lean_class": "VA", "waste": None},
    {"code": "GET_TOOL", "name": "Handle tool (get / put aside)",
     "therblig": ["TE", "G", "TL", "RL"], "mtm_uas": "HA-HC (handle auxiliary tool)",
     "description": "Take the power tool from the line side or put it back.", "lean_class": "NNVA", "waste": None},
    {"code": "REACH_FASTENER", "name": "Get fastener and load on tool",
     "therblig": ["TE", "SE", "G", "TL", "PP (pre-position)"], "mtm_uas": "AF (difficult grasp, tight place)",
     "description": "Take one screw from the pouch and place it on the screwdriver bit.",
     "lean_class": "NNVA", "waste": None},
    {"code": "FASTEN", "name": "Fasten (drive screw)",
     "therblig": ["TL", "P", "U (use)"], "mtm_uas": "PC + BA (place tool on screw, actuate)",
     "description": "Bring the loaded tool to the screw point, hold it on axis and run the screw down.",
     "lean_class": "VA", "waste": None},
    {"code": "INSPECT", "name": "Inspect / check",
     "therblig": ["I (inspect)"], "mtm_uas": "VA (visual control)",
     "description": "Visual or tactile check of the assembly.", "lean_class": "NNVA", "waste": None},
    {"code": "SETUP", "name": "Workstation / cart set-up",
     "therblig": ["TL", "P", "PP (pre-position)"], "mtm_uas": "BA/BB (operate) + KA",
     "description": "Bring and lock the dolly cart, adjust the height-adjustable workstation (AWS) to the operator.",
     "lean_class": "NNVA", "waste": None},
    {"code": "IDLE", "name": "Idle / wait",
     "therblig": ["UD (unavoidable delay)", "AD (avoidable delay)", "H (hold)"], "mtm_uas": "- (process/waiting time)",
     "description": "No productive hand motion (waiting for the carrier, hesitation, holding).",
     "lean_class": "NVA", "waste": "Waiting"},
]
CODES = [t["code"] for t in TAXONOMY]
BY_CODE = {t["code"]: t for t in TAXONOMY}
UNOBS = {"code": "UNOBS", "name": "Unobserved remainder of the station cycle", "therblig": [], "mtm_uas": "-",
         "description": "full_cycle_s - observed clip duration (config.observation); manual block, never automated.",
         "lean_class": None, "waste": None}

_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
_TOOL = r"screwdriver|screw driver|driver|tool|gun|wrench|drill|nutrunner|riveter|pliers"
_FASTENER = r"screws?|bolts?|nuts?|rivets?|fasteners?"


def repeat_count(text: str) -> int:
    """'Fix 4 screws' -> 4; 'Fixate two bolts' -> 2; default 1."""
    t = text.lower()
    m = re.search(rf"\b(\d+|{'|'.join(_NUM)})\s+(?:\w+\s+)?(?:{_FASTENER}|clips?|plugs?|connectors?|times)\b", t)
    if not m:
        return 1
    g = m.group(1)
    return int(g) if g.isdigit() else _NUM[g]


def classify_activity(text: str) -> str:
    """Lexicon rules: activity text -> element code (ordered, first match wins)."""
    t = text.lower()
    if re.search(r"\b(cart|dolly)\b.*\b(move|position|lock|place)|\b(move|position|lock|place)\b.*\b(cart|dolly)\b|"
                 r"\block\b.*\bwheel|\badjust\b.*\b(ws|aws|height|workstation)\b", t) and \
            not re.search(r"\b(take|pick|get)\b.*\bfrom the cart\b", t):
        return "SETUP"
    if re.search(r"\bextend robot arm\b|\bwait\b", t):
        return "IDLE"                         # operator waits for the robot / the line
    if re.search(r"^(walk|navigate|go to|move to|move around)\b", t):
        return "WALK"
    if re.search(rf"\b(leave|put back|put down|put away|hang|return|release|place|hand over|take)\b.*\b({_TOOL})\b", t) and \
            not re.search(rf"\b({_FASTENER})\b.*\b(on|onto)\b", t):
        return "GET_TOOL"
    if re.search(rf"\b(pick|get|grab|collect)\b.*\b({_TOOL})\b", t):
        return "GET_TOOL"
    if re.search(rf"\b(take|pick|get|grab|place|put|load)\b.*\b({_FASTENER})\b", t) and \
            not re.search(r"\b(fix\w*|fasten\w*|tight\w*)\b|\bscrew (it|them)\b", t):
        return "REACH_FASTENER"
    if re.search(r"\b(fix\w*|fasten\w*|tight\w*|screw\w*|bolt|rivet|torque)\b|\brotate\b.*\bnut\b", t):
        return "FASTEN"
    if re.search(r"\b(check|inspect|verify|test|control|look)\b", t):
        return "INSPECT"
    if re.search(r"\b(take|pick|get|grab|grasp|collect|remove|unpack)\b", t):
        return "GET_PART"
    return "INSERT"   # insert / place / position / mount / attach / connect / align / push / fit


_GROUP_VERBS = r"^(fix|fixate|fasten|tighten|screw|complete|mount)\b"


def script_from_activities(acts: list[Activity]) -> list[dict]:
    """Expand the activity tree into the ordered standard-work script (CarDA lists are templates; some steps
    may not be performed in a given cycle -> the aligner may skip steps at a cost).

    * A parent with children whose text is a summary ("Fix 4 screws on the speaker") contributes its children
      repeated N times (N parsed from the parent or a child, e.g. "Take N 3 screw"); a parent that is itself an
      action ("Take 3 screws from the pouch") is emitted before its children.
    * Children are not repeated if they already enumerate the N fastenings ("first nut", "second nut").
    * WALK steps are dropped (optional walks are inserted between all steps by the aligner).
    * Adjacent steps with the same code are merged; repeated fastenings keep a `count`.
    """
    out: list[dict] = []
    for a in acts:
        if a.children:
            code = classify_activity(a.text)
            n = max([repeat_count(a.text)] + [repeat_count(c.text) for c in a.children])
            kids = script_from_activities(a.children)
            if not re.search(_GROUP_VERBS, a.text.lower()):
                out.append({"code": code, "activity": a.aid, "text": a.text})
            nf = sum(k["code"] == "FASTEN" for k in kids)
            reps = n if (nf < n and n > 1) else 1
            for k in range(reps):
                for s in kids:
                    s = dict(s)
                    if reps > 1:
                        s["rep"], s["of"] = k + 1, reps
                    out.append(s)
        else:
            out.append({"code": classify_activity(a.text), "activity": a.aid, "text": a.text})
    out = [s for s in out if s["code"] != "WALK"]
    merged: list[dict] = []
    for s in out:
        if merged and merged[-1]["code"] == s["code"] and \
                (merged[-1].get("rep") == s.get("rep") or s["code"] in ("FASTEN", "REACH_FASTENER") and
                 merged[-1].get("rep") != s.get("rep") and merged[-1]["code"] == "FASTEN"):
            m = merged[-1]
            if s["code"] == "FASTEN" and m.get("rep") != s.get("rep"):
                m["count"] = m.get("count", 1) + 1
            if s["activity"] not in m["activity"].split("+"):
                m["activity"] += "+" + s["activity"]
                m["text"] += " / " + s["text"]
        else:
            merged.append(dict(s))
    return merged


def expected_fasteners(script: list[dict]) -> int:
    """Number of fastenings the standard work calls for (FASTEN steps x their count)."""
    return sum(s.get("count", 1) for s in script if s["code"] == "FASTEN")
