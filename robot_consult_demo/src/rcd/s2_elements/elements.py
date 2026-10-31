"""Per-cycle element segmentation: subgoal nesting + forced alignment + ergodic fill."""
from __future__ import annotations

import numpy as np

from . import segment as SG
from .data import Cycle
from .features import RATE_HZ, extract, work_anchor
from .taxonomy import expected_fasteners, script_from_activities


def segment_cycle(c: Cycle) -> tuple[dict, dict]:
    f = extract(c.joints, c.names, c.fps)
    n = f["n"]
    dur = n / RATE_HZ
    anchor = work_anchor(f)
    E = SG.evidence(f, anchor)
    post = SG.posterior(E)
    # subgoal spans (annotated), boundaries snapped to the nearest motion pause, clipped to the clip
    spans = []
    sgs = sorted(c.subgoals, key=lambda s: s.start)
    for k, sg in enumerate(sgs):
        s0 = max(0.0, min(dur, sg.start))
        s1 = max(0.0, min(dur, sg.end))
        if k > 0 and abs(sg.start - sgs[k - 1].end) < 0.5 and 0 < s0 < dur:
            s0 = spans[-1]["t1"]                       # shared boundary already refined
        elif 0 < s0 < dur:
            s0 = SG.refine_boundary(f, s0)
        if k + 1 < len(sgs) and abs(sgs[k + 1].start - sg.end) < 0.5 and 0 < s1 < dur:
            s1 = SG.refine_boundary(f, s1)
        elif s1 < dur - 0.5 and s1 > 0:
            s1 = SG.refine_boundary(f, s1)
        if s1 - s0 >= 1.0:
            spans.append({"sg": sg, "t0": s0, "t1": s1})
    raw = []        # (code, a, b, subgoal, source, script step)
    diag = {"screws": [], "dwells_s": f.get("dwells", [])}
    cur = 0
    for sp in spans:
        a, b = int(round(sp["t0"] * RATE_HZ)), int(round(sp["t1"] * RATE_HZ))
        a = max(a, cur)
        if a - cur >= 1:
            for code, x, y in SG.ergodic(E, cur, a):
                raw.append((code, x, y, None, "mocap_rules", None))
        script = script_from_activities(sp["sg"].activities)
        if script:
            for g, x, y in SG.forced_align(E, script, a, b):
                src = "mocap_rules" if g["code"] == "WALK" else "merged"
                for code, x2, y2 in SG.carve_idle(f, g["code"], x, y):
                    raw.append((code, x2, y2, sp["sg"].name, src if code == g["code"] else "mocap_rules",
                                g["step"] if code == g["code"] else None))
            nexp = expected_fasteners(script)
            if nexp:
                diag["screws"].append({"subgoal": sp["sg"].name, "expected": nexp,
                                       "detected_dwells": SG.count_screw_cycles(f, a, b)})
        else:
            for code, x, y in SG.ergodic(E, a, b):
                raw.append((code, x, y, sp["sg"].name, "mocap_rules", None))
        cur = b
    if n - cur >= 1:
        for code, x, y in SG.ergodic(E, cur, n):
            raw.append((code, x, y, None, "mocap_rules", None))
    els = []
    seen_steps = set()
    for i, (code, a, b, sgname, src, step) in enumerate(raw):
        t0, t1 = a / RATE_HZ, b / RATE_HZ
        conf = float(np.mean(post[code][a:b])) if b > a else 0.0
        e = {"id": f"{c.station}_{c.cycle_id}_{i:03d}", "code": code, "start_s": round(t0, 2), "end_s": round(t1, 2),
             "duration_s": round(t1 - t0, 2), "source": src, "confidence": round(conf, 3), "subgoal": sgname}
        e.update(SG.element_attrs(f, code, a, b))
        if step:
            e["activity"] = step["activity"]
            e["activity_text"] = step["text"]
            key = (sgname, step["activity"], step.get("rep"))
            first_piece = key not in seen_steps          # a step split by carved-out waiting counts once
            seen_steps.add(key)
            e["step_first_piece"] = first_piece
            if e["code"] in ("FASTEN", "REACH_FASTENER"):
                e["fastener_count"] = int(step.get("count", 1)) if first_piece else 0
            if "rep" in step:
                e["repetition"] = f"{step['rep']}/{step['of']}"
        els.append(e)
    # loaded / empty transport for walks (Therblig TL vs TE): loaded if the previous non-walk element got a part/tool
    for i, e in enumerate(els):
        if e["code"] == "WALK":
            prev = next((x for x in reversed(els[:i]) if x["code"] != "WALK"), None)
            e["loaded"] = bool(prev and prev["code"] in ("GET_PART", "GET_TOOL"))
    for e in els:
        e["duration_s"] = round(e["end_s"] - e["start_s"], 2)
    cyc = {"station": c.station, "cycle_id": c.cycle_id, "duration_s": round(dur, 3),
           "joint_source": c.joint_source, "video": c.video, "long_dwells_s": diag["dwells_s"],
           "fastener_check": diag["screws"], "elements": els,
           "subgoals": [{"name": sp["sg"].name, "annotated": [sp["sg"].start, sp["sg"].end],
                         "refined": [round(sp["t0"], 2), round(sp["t1"], 2)]} for sp in spans]}
    return cyc, f
