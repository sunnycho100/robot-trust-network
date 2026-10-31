"""Dataset-neutral lexicon: free-text standard-work (SOP) activity lines -> work-element codes and step scripts.

Used by the SOP-only segmenter (sop_segment.py). Everything here works on generic shop-floor wording; no line- or
product-specific vocabulary. Rules are ordered, first match wins:

  WALK           walk / go / navigate / move / proceed / step ... to | towards | around | back   (dropped from scripts:
                 optional walks are inserted between all steps by the aligner)
  INSPECT        a line that starts with inspect / check / verify / test / examine / measure / confirm / look
  SETUP          set up / adjust / position / lock / dock a cart, dolly, trolley, fixture, jig, workstation height
  IDLE           wait / await / idle, or a sentence whose subject is the robot ("Extend robot arm ...", "Robot moves ...")
  GET_TOOL       handle a hand tool (take / pick / get / put back / hand over / hang / store ... screwdriver, wrench ...)
  REACH_FASTENER take / place / load / feed a screw, bolt, nut, rivet ... (onto a tool), without driving it
  FASTEN         fasten / tighten / screw / bolt / rivet / torque / fix / clinch / crimp / rotate a nut
  INSPECT        inspect / check / verify / test / examine / measure / confirm / look
  GET_PART       pick / take / get / grab / grasp / fetch / retrieve / collect / remove / unpack
  INSERT         place / insert / mount / fit / attach / install / position / align / push / press / connect / plug ...
  None           nothing matched -> the step is aligned by motion only (code "ANY", relabelled from the kinematics)

A flat SOP activity list (no outline numbering) is turned into an ordered script. Grouping is inferred from the
text: a FASTEN line immediately followed by fastener-acquisition lines is a *group header* ("Fasten 4 bolts ..."),
its body (acquire + drive lines) is repeated N times when the body names fewer drives than the header's count.
"""
from __future__ import annotations

import re

from .taxonomy import repeat_count

ANY = "ANY"     # motion-only fallback code (never written to outputs)

_TOOL = (r"screw ?drivers?|drivers?|tools?|guns?|wrench(?:es)?|spanners?|drills?|nut ?runners?|riveters?|pliers|"
         r"hammers?|mallets?|applicators?|torch(?:es)?|cutters?|keys?|clamps? tool|sealant gun")
_FASTENER = r"screws?|bolts?|nuts?|rivets?|fasteners?|washers?|studs?|pins? and nuts?"
_HOLDER = r"carts?|doll(?:y|ies)|trolley|fixtures?|jigs?|workstations?|work stations?|ws|aws|stations? height|table"

_RULES = [
    ("WALK", re.compile(r"^(walk|navigate|go|move|proceed|step|travel|turn|return|come)\s+(?:back\s+)?"
                        r"(to|towards|toward|around|into|over|across|back)\b")),
    ("INSPECT", re.compile(r"^(visually\s+|manually\s+)?(check|inspect|verify|test|examine|measure|confirm|look)\b")),
    ("SETUP", re.compile(
        rf"(\b(move|position|lock|unlock|place|bring|push|dock|park|roll|secure|clamp|set ?up|setup|prepare)\b.*\b({_HOLDER})\b"
        rf"|\b({_HOLDER})\b.*\b(lock|position|dock|secure)\b|\block\b.*\bwheels?\b|\bset ?up\b"
        rf"|\badjust\b.*\b(height|{_HOLDER}|station|position of the (?:cart|fixture|jig))\b)")),
    ("IDLE", re.compile(r"^(wait|await|idle|pause)\b|\bwait(?:ing)? (?:for|until)\b|^(extend|retract|move|return|park)\s+"
                        r"(?:the\s+)?robot\b|^(?:the\s+)?robot\s+\w+")),
    ("GET_TOOL", re.compile(rf"\b(take|pick|get|grab|collect|fetch|leave|put back|put down|put away|lay down|set down|hang|"
                            rf"return|release|place|hand over|store|holster|replace)\b.*\b({_TOOL})\b")),
    ("REACH_FASTENER", re.compile(rf"\b(take|pick|get|grab|place|put|load|feed|position|set|start)\b.*\b({_FASTENER})\b")),
    ("FASTEN", re.compile(r"\b(fix\w*|fasten\w*|tight\w*|screw\w*|bolt\w*|rivet\w*|torque\w*|clinch\w*|crimp\w*|"
                          r"nut[- ]?run\w*|run down|drive the)\b|\brotate\b.*\bnuts?\b")),
    ("INSPECT", re.compile(r"\b(check\w*|inspect\w*|verif\w*|test\w*|examin\w*|measur\w*|confirm\w*|look\w*|scan\w*|"
                           r"control)\b")),
    ("GET_PART", re.compile(r"\b(take|pick|get|grab|grasp|collect|fetch|retrieve|remove|unpack|unwrap|lift)\b")),
    ("INSERT", re.compile(r"\b(place|insert|mount|fit|attach|install|position|align\w*|push|press|connect|plug|clip|snap|"
                          r"slide|seat|put|assemble|join|rotate|pass|apply|route|lay|hook|engage|pull|tilt|hold|turn|"
                          r"glue|stick|tape|wrap|bend|guide|lower|raise)\b")),
]
# a fastener *on a tool / into a hole* without driving it is still acquisition (not fastening)
_DRIVE = re.compile(r"\b(fix\w*|fasten\w*|tight\w*|torque\w*|riveting)\b|\bscrew (?:it|them|in|down)\b|\bscrew\w* the\b")


def classify_text(text: str) -> str | None:
    """Activity text -> element code, or None when no rule matches (motion-only fallback)."""
    t = " ".join(str(text).lower().split())
    for code, rx in _RULES:
        if not rx.search(t):
            continue
        if code == "SETUP" and re.search(r"\b(take|pick|get|grab)\b.*\bfrom (?:the )?\w+", t) and \
                not re.search(r"\b(lock|dock|adjust|set ?up)\b", t):
            continue                                        # "take X from the cart" is a pick, not a set-up
        if code == "GET_TOOL" and re.search(rf"\b({_FASTENER})\b.*\b(on|onto|into)\b.*\b(tip|bit|socket|{_TOOL})\b", t):
            continue                                        # "place screw on the screwdriver tip" = fastener
        if code == "REACH_FASTENER" and _DRIVE.search(t):
            continue                                        # "place the nut ... and screw it" = fastening
        return code
    return None


def _code(text: str) -> str:
    return classify_text(text) or ANY


_OUTLINE = re.compile(r"^\s*(\d+(?:\.\d+)+)\s+(.*)$")
_GROUP = re.compile(r"^(fix\w*|fasten\w*|tighten\w*|screw\w*|bolt\w*|rivet\w*|complete|mount)\b")


def _tree(lines: list[str]) -> list[dict]:
    """Outline-numbered lines ("3.13.1 Take 1 screw ...") -> nested activity tree."""
    root: list[dict] = []
    stack: list[tuple[int, dict]] = []
    for x in lines:
        m = _OUTLINE.match(str(x))
        aid, text = (m.group(1), m.group(2)) if m else (str(len(root) + 1), str(x))
        node = {"aid": aid, "text": " ".join(text.split()), "children": []}
        depth = aid.count(".")
        while stack and stack[-1][0] >= depth:
            stack.pop()
        (stack[-1][1]["children"] if stack else root).append(node)
        stack.append((depth, node))
    return root


def _expand(nodes: list[dict]) -> list[dict]:
    """Nested activities -> element steps. A parent whose text is a summary of its children (a fastening header
    such as "Fasten 4 bolts") contributes its children, repeated N times when they enumerate fewer than N drives;
    any other parent is emitted before its children."""
    out: list[dict] = []
    for a in nodes:
        code = _code(a["text"])
        if a["children"]:
            kids = _expand(a["children"])
            n = max([repeat_count(a["text"])] + [repeat_count(c["text"]) for c in a["children"]])
            if not _GROUP.search(a["text"].lower()):
                out.append({"code": code, "activity": a["aid"], "text": a["text"]})
            elif not any(k["code"] == "FASTEN" for k in kids):
                kids.append({"code": "FASTEN", "activity": a["aid"], "text": a["text"]})
            nf = sum(k["code"] == "FASTEN" for k in kids)
            reps = n if (nf < n and n > 1) else 1
            for r in range(reps):
                for k in kids:
                    k = dict(k)
                    if reps > 1:
                        k["rep"], k["of"] = r + 1, reps
                        if k["code"] == "FASTEN":
                            k["count"] = 1            # the repetition itself carries the count
                    out.append(k)
        else:
            s = {"code": code, "activity": a["aid"], "text": a["text"]}
            if code == "FASTEN":
                s["count"] = repeat_count(a["text"])
            out.append(s)
    return out


def _merge(out: list[dict]) -> list[dict]:
    out = [s for s in out if s["code"] != "WALK"]
    merged: list[dict] = []
    for s in out:
        if merged and merged[-1]["code"] == s["code"] and s["code"] != ANY and \
                (merged[-1].get("rep") == s.get("rep") or s["code"] == "FASTEN"):
            m = merged[-1]
            if s["code"] == "FASTEN":
                m["count"] = m.get("count", 1) + s.get("count", 1)
            if s["activity"] not in m["activity"].split("+"):
                m["activity"] += "+" + s["activity"]
                m["text"] += " / " + s["text"]
        else:
            merged.append(dict(s))
    return merged


def script_from_lines(lines: list[str], step_id: str = "") -> list[dict]:
    """Ordered activity lines of one SOP step -> element script. Outline-numbered lines are parsed as a tree
    (script_from_tree); a plain flat list falls back to text-inferred grouping (below)."""
    if lines and sum(bool(_OUTLINE.match(str(x))) for x in lines) >= 0.5 * len(lines):
        return _merge(_expand(_tree(lines)))
    return _script_from_flat(lines, step_id)


def _script_from_flat(lines: list[str], step_id: str = "") -> list[dict]:
    """Flat ordered activity lines of one SOP step -> ordered element script [{code, activity, text, count, rep, of}].

    * group header: a FASTEN line followed by REACH_FASTENER lines; its body (the following acquire/drive lines,
      walks inside allowed) is repeated N times (N from the header or body text) when the body has fewer than N
      drives; the header itself is a summary and is not emitted (a body without a drive gets the header as drive).
    * WALK lines are dropped (optional walks are inserted between all steps by the aligner).
    * adjacent steps with the same code are merged; merged drives keep a `count`.
    """
    items = [{"code": _code(x), "activity": f"{step_id}{i + 1}" if step_id else str(i + 1), "text": " ".join(str(x).split())}
             for i, x in enumerate(lines)]
    out: list[dict] = []
    i = 0
    while i < len(items):
        it = items[i]
        nxt = items[i + 1]["code"] if i + 1 < len(items) else None
        if it["code"] == "FASTEN" and nxt == "REACH_FASTENER":
            j = i + 1
            body = []
            while j < len(items):
                c = items[j]["code"]
                after = items[j + 1]["code"] if j + 1 < len(items) else None
                if c in ("REACH_FASTENER", "FASTEN") or (c == "WALK" and after in ("REACH_FASTENER", "FASTEN")):
                    body.append(items[j])
                    j += 1
                else:
                    break
            if not any(b["code"] == "FASTEN" for b in body):
                body.append(dict(it))
            n = max([repeat_count(it["text"])] + [repeat_count(b["text"]) for b in body])
            nf = sum(b["code"] == "FASTEN" for b in body)
            reps = n if (nf < n and n > 1) else 1
            for k in range(reps):
                for b in body:
                    s = dict(b)
                    if reps > 1:
                        s["rep"], s["of"] = k + 1, reps
                    out.append(s)
            i = j
            continue
        s = dict(it)
        if s["code"] == "FASTEN":
            s["count"] = repeat_count(s["text"])
        out.append(s)
        i += 1
    out = [s for s in out if s["code"] != "WALK"]
    merged: list[dict] = []
    for s in out:
        if merged and merged[-1]["code"] == s["code"] and s["code"] != ANY and \
                (merged[-1].get("rep") == s.get("rep") or s["code"] == "FASTEN"):
            m = merged[-1]
            if s["code"] == "FASTEN":
                m["count"] = m.get("count", 1) + s.get("count", 1)
            if s["activity"] not in m["activity"].split("+"):
                m["activity"] += "+" + s["activity"]
                m["text"] += " / " + s["text"]
        else:
            merged.append(dict(s))
    return merged


# ------------------------------------------------------------------------------------------- MTM-UAS nominal times
# Weak duration priors for SOP steps: MTM-UAS basic operations at distance class 2 (20-50 cm), 1 TMU = 0.036 s.
_TMU = {"GET_PART": 45, "INSERT": 40, "GET_TOOL": 45, "REACH_FASTENER": 65, "FASTEN": 65, "INSPECT": 15,
        "SETUP": 45, "IDLE": 0, ANY: 40}
_WALK_TMU = 50          # KA, about 2 m between line side and workpiece per scripted step
TMU_S = 0.036


def mtm_nominal_s(script: list[dict]) -> float:
    """Synthetic (PMTS) time of a step script: sum of UAS blocks per element (x count) plus one walk per element."""
    tot = 0.0
    for s in script:
        n = int(s.get("count", 1)) if s["code"] in ("FASTEN", "REACH_FASTENER") else \
            max(1, len(str(s.get("activity", "")).split("+")))
        tot += n * _TMU.get(s["code"], 40) + _WALK_TMU
    return tot * TMU_S
