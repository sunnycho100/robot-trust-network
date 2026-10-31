"""RFQ (request for quotation) markdown per station that receives robot work in the recommended allocation."""
from __future__ import annotations

from collections import Counter

import numpy as np


def _f(x, d=1):
    if isinstance(x, dict):
        x = x.get("fraction_within_usable_reach", x.get("value"))
    if x is None:
        return "-"
    try:
        return f"{float(x):,.{d}f}"
    except (TypeError, ValueError):
        return str(x)


def build_rfqs(D):
    al, cand, cat, rtt = D["allocation"], D["candidates"], D["robot_catalog"], D["robot_task_times"]
    el, sg, tp, au = D["elements"], D.get("scene_graph"), D.get("task_plan"), D["automatability"]
    rec = next(s for s in al["scenarios"] if s.get("recommended"))
    src = rec if rec["robots"] else next((s for s in al["scenarios"] if s["robots"]), None)
    if src is None:
        return {}
    takt = al["params"]["takt_s"]
    rows = {e["element_key"]: e for e in al["elements"]}
    robots = {r["model"]: r for r in cat["robots"]}
    rt = {(e["model"], e["group_id"]): e for e in rtt["entries"]}
    groups = {g["group_id"]: g for g in cand["task_groups"]}
    s3 = {t.get("element_key") or f"{t['station']}:{t['code']}": t for t in au["tasks"]}
    out = {}
    for rb in src["robots"]:
        st = rb["station"]
        stt = src["station_times"][st]
        sub = Counter(e.get("subgoal") for c in el["cycles"] if c["station"] == st for e in c["elements"]
                      if e.get("subgoal"))
        conv = None
        if sg:
            sp = [c["conveyor"]["speed_mm_s"] for s in sg["stations"] if s["station"] == st for c in s["cycles"]]
            conv = float(np.mean(sp)) if sp else None
        L = [f"# RFQ - collaborative robot cell, station {st}", "",
             f"Status: draft generated from allocation scenario `{src['name']}`"
             + (" (recommended)" if src.get("recommended") else " (evaluation only; not the recommended scenario)")
             + f". Inputs: {', '.join(f'{k}={v}' for k, v in al.get('inputs', {}).items())}.", "",
             "## 1. Process description", "",
             f"Paced car-door assembly line (moving conveyor), takt {_f(takt, 0)} s. Station {st} work content "
             f"observed in the CarDA clips (subgoals): " + "; ".join(f"{k} ({v} occurrences)" for k, v in sub.most_common()) + ".",
             f"The robot takes over the elements below; the operator keeps the rest of the cycle "
             f"(including {_f(rows.get(f'{st}:UNOBS', {}).get('time_s'))} s/cycle of work not visible in the clips).", "",
             "## 2. Work elements to automate", "",
             "| element | human time now [s/cycle] | robot time (sim) [s/cycle] | task group | s3 score | s3 category |",
             "|---|---|---|---|---|---|"]
        for a in src["assignment"]:
            if a["station"] == st and a["resource"] != "human":
                t3 = s3.get(a["element_key"], {})
                L.append(f"| {a['element_key']} | {_f(a['human_time_s'])} | {_f(a['time_s'])} | {a.get('group_id')} | "
                         f"{_f(t3.get('total'), 2)} | {t3.get('category', '-')} |")
        L += ["", "## 3. Task points and requirements", ""]
        for gid in rb["groups"]:
            g = groups.get(gid, {})
            req = g.get("requirements", {})
            L.append(f"- Group `{gid}` (codes {', '.join(g.get('codes', []))}): " +
                     ", ".join(f"{k} = {v}" for k, v in req.items() if not isinstance(v, (list, dict))))
            for k, v in req.items():
                if isinstance(v, list) and v and len(v) <= 12:
                    L.append(f"  - {k}: {v}")
        L += ["", "## 4. Required cycle time", "",
              f"- Station takt: {_f(takt, 0)} s. Operator work after automation: {_f(stt['human_s'])} s; robot: "
              f"{_f(stt['robot_s'])} s; handover allowance: {_f(stt['interaction_s'])} s; resulting station cycle "
              f"{_f(stt['cycle_s'])} s (slack {_f(takt - stt['cycle_s'])} s).",
              f"- The robot may start only after the operator's preceding elements "
              f"({_f(stt.get('robot_start_after_human_s'))} s into the cycle); the supplier must guarantee the robot "
              f"sequence within {_f(takt - (stt.get('robot_start_after_human_s') or 0) - stt['interaction_s'])} s "
              f"including conveyor-tracking synchronisation.", "",
              "## 5. Conveyor tracking", "",
              (f"- Door moves continuously at about {_f(conv)} mm/s (measured from video/mocap in s1); the cell needs "
               f"line tracking (encoder input); door travel during the whole robot time per cycle is about "
               f"{_f(conv * stt['robot_s'] / 1000, 2)} m (tracking windows per task group from s4 below)." if conv else "- Conveyor speed not available; tracking required (moving line)."), ""]
        for gid in rb["groups"]:
            g = groups.get(gid, {})
            pl = g.get("placement", {}) or {}
            L.append(f"- Mounting for `{gid}`: **{g.get('mounting', '-')}** (base {pl.get('base_xyz')}, pedestal "
                     f"{pl.get('pedestal_height_m')} m, tracking window {pl.get('tracking_window_m')} m). "
                     f"{pl.get('mounting_reason', '')}")
            if g.get("mounting") == "linear_track":
                L.append("- Supplier to quote a 7th-axis linear track synchronised to the conveyor encoder (track, "
                         "servo drive, robot-controller integration), mounted off the belt next to the line.")
        L += ["", "## 6. Candidate models (from s4 screening, s6a simulation)", "",
              "| model | maker | payload kg | reach mm | price range USD | sim cycle s | IK success | reachability |",
              "|---|---|---|---|---|---|---|---|"]
        for gid in rb["groups"]:
            for e in groups.get(gid, {}).get("evaluated", []):
                if not e.get("passed"):
                    continue
                r = robots.get(e["model"], {})
                s6 = rt.get((e["model"], gid), {})
                pr = r.get("price_usd_range", [None, None])
                L.append(f"| {e['model']}{' (selected)' if e['model'] == rb['model'] else ''} | {r.get('maker', '-')} | "
                         f"{r.get('payload_kg', '-')} | {r.get('reach_mm', '-')} | {_f(pr[0], 0)} - {_f(pr[-1], 0)} | "
                         f"{_f(s6.get('cycle_time_s'))} | {_f(s6.get('ik_success_rate'), 2)} | {_f(e.get('reachability'), 2)} |")
        L += ["", "## 7. Safety and space", ""]
        for gid in rb["groups"]:
            e = next((e for e in groups.get(gid, {}).get("evaluated", []) if e["model"] == rb["model"]), {})
            cs = e.get("cell_space")
            if cs:
                L.append(f"- {gid}: ISO 13855 separation S = K*T + C = {_f(cs['K_mm_s'], 0)} x {_f(cs['T_s'], 2)} + "
                         f"{_f(cs['C_mm'], 0)} = {_f(cs['separation_mm'], 0)} mm; footprint {_f(cs['footprint_m2'], 2)} m2 "
                         f"+ {cs['aisle_allowance']:.0%} aisle allowance = {_f(cs['total_m2'], 2)} m2.")
        L += ["- Collaborative operation per ISO 10218-1/-2 and ISO/TS 15066 (power and force limiting or speed and "
              "separation monitoring); supplier to deliver risk assessment.", "",
              "## 8. Budget reference (s5 / s7)", "",
              f"- Robot {rb['model']} price midpoint {_f(rb['price_mid_usd'], 0)} USD; installed-cell estimate "
              f"{_f(rb['capex_usd'], 0)} USD: " + ", ".join(f"{k.replace('_', ' ')} {_f(v, 0)}"
                                                       for k, v in rb.get("capex_breakdown_usd", {}).items()) + ".",
              (f"- One arm serves {len(rb['categories'])} task categories ({', '.join(rb['categories'])}): automatic tool "
               f"changer required; {rb.get('tool_changes_per_cycle')} exchanges per cycle "
               f"({_f(rb.get('tool_change_s_per_cycle'))} s/cycle included in the robot time above)."
               if rb.get("tool_changes_per_cycle") else "- Single task category: no tool changer."), "",
              "## 9. Draft robot program", ""]
        plans = [p for p in (tp or {}).get("plans", []) if p["station"] == st]
        if plans:
            p = plans[0]
            L.append(f"- Behaviour tree from s9 (`out/s9/task_plan.json`, station {st}, model {p['model']}), "
                     f"{len(p['steps'])} steps: " + " -> ".join(s["primitive"] for s in p["steps"][:40]))
            L += ["", "```xml", p["bt_xml"], "```"]
        else:
            L.append("- No s9 task plan for this station yet.")
        out[st] = "\n".join(L) + "\n"
    return out
