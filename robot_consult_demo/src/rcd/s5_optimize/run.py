"""s5 - allocation & line balancing (industrial engineering decision stage).

Outputs out/s5/allocation.json with
  * line_balancing: RPW / LCR / COMSOAL heuristics + exact SALBP-1 (CP-SAT) on the observed work elements with the
    per-station UNOBS block kept as one manual task; SALBP-2 (min cycle for the current number of stations);
    an exact n*(c) sweep; and a post-automation re-balancing of the remaining human work (headcount view).
  * scenarios: human-robot allocation MILP (CP-SAT) for baseline_manual, min_cost, ergo_capped_min_cost,
    ergonomics_first, max_automation; exactly one recommended.
  * worker_machine: man-machine (human + cobot) chart per robot station; machine-cluster check (lecture04).
  * rough_cut_capacity: robots and operators required (ISyE 510 RCA lab: ceil(required h / available h),
    90 % capacity-cushion rule).
"""
from __future__ import annotations

import math
import time
from collections import defaultdict

from rcd.common.io import out_dir, save_json
from rcd.s5_optimize import balancing as LB
from rcd.s5_optimize.allocation import AllocModel
from rcd.s5_optimize.data import NON_WORK, TIME_BASIS, build
from rcd.s5_optimize.inputs import load_input
from rcd.s7_roi import assumptions as A

ERGO_REDUCTION = 0.50      # ergonomic target: halve awkward-posture seconds per cycle (assumption)
CUSHION = 0.90             # RCA capacity-cushion rule: keep utilisation < 90 %
TAKT_MARGIN = 0.05         # pacing with margin: plan station cycles <= 95 % of takt so stochastic overruns stay rare
                           # (assumption; ISyE 510 lecture06 line pacing, RCA capacity cushion; checked by s6b DES)

CITATIONS = {
    "lecture06": "ISyE 510 Facilities Planning, lecture06 'Assembly line balancing' (RPW, LCR, efficiency, N_t)",
    "lecture04": "ISyE 510 lecture04 'Single station cell analysis' - machine cluster n' = (a+t)/(a+b)",
    "lecture03": "ISyE 510 lecture03 'Economic evaluation of processes' - financial recovery factor A/P",
    "rca_lab": "ISyE 510 LAB Rough-Cut Analysis (RCA I/II): machines = ROUNDUP(required h / available h), "
               "utilisation < 90 % cushion, dedicated vs cross-trained operators",
    "helgeson_birnie": "Helgeson, W.B. & Birnie, D.P. (1961) Assembly line balancing using the ranked positional "
                       "weight technique. Journal of Industrial Engineering 12(6):394-398",
    "arcus": "Arcus, A.L. (1966) COMSOAL: a computer method of sequencing operations for assembly lines. "
             "International Journal of Production Research 4(4):259-277",
    "weckenborg": "Weckenborg, C., Kieckhaefer, K., Mueller, C., Grunewald, M. & Spengler, T.S. (2020) Balancing "
                  "of assembly lines with collaborative robots. Business Research 13:93-132. "
                  "https://link.springer.com/article/10.1007/s40685-019-0101-y",
}


def _human_rebalance(data, scen, c):
    """Re-balance only the human-assigned work (robots fixed) -> minimum operators at takt (headcount view)."""
    rows = data["rows"]
    hum = [a["element_key"] for a in scen["assignment"] if a["resource"] == "human" and a["code"] not in NON_WORK]
    t = {a["element_key"]: a["time_s"] for a in scen["assignment"] if a["element_key"] in hum}  # incl. layout levers
    edges_all = data["precedence_within"] + data["precedence_chain"]
    # precedence restricted to human tasks through removed (robot) tasks: transitive closure
    succ = defaultdict(set)
    for a, b in edges_all:
        succ[a].add(b)
    edges = set()
    for a in hum:
        stack, seen = list(succ[a]), set()
        while stack:
            b = stack.pop()
            if b in seen:
                continue
            seen.add(b)
            if b in t:
                edges.add((a, b))
            else:
                stack += list(succ[b])
    edges = sorted(edges)
    heur = LB.rpw(hum, t, edges, c)
    ex = LB.exact_salbp1(hum, t, edges, c, heur["n_stations"])
    return {"scenario": scen["name"], "human_work_s": sum(t.values()), "theoretical_min_operators":
            math.ceil(sum(t.values()) / c - 1e-9), "rpw_operators": heur["n_stations"],
            "exact_operators": ex["n_stations"], "exact": ex,
            "note": "Robot work stays at its station; only human tasks are re-balanced (precedence through robot "
                    "tasks kept transitively). A headcount reduction needs exact_operators < number of stations."}


def line_balancing(data, takt):
    rows = data["rows"]
    tasks = [k for k, v in rows.items() if v["code"] not in NON_WORK]
    t = {k: rows[k]["time_s"] for k in tasks}
    edges = data["precedence_within"] + data["precedence_chain"]
    heur, ex, tm = LB.run_all(tasks, t, edges, takt)
    as_is = LB.metrics("AS_IS", [sum(t[k] for k in tasks if rows[k]["station"] == st) for st in data["stations"]],
                       takt, [[k for k in tasks if rows[k]["station"] == st] for st in data["stations"]])
    s2 = LB.salbp2_min_cycle(tasks, t, edges, len(data["stations"]))
    total = sum(t.values())
    sweep = []
    for f in (0.6, 0.75, 0.9, 1.0, 1.25, 1.5):
        c = takt * f
        if c < max(t.values()) - 1e-9:
            continue
        r = LB.rpw(tasks, t, edges, c)
        e = LB.exact_salbp1(tasks, t, edges, c, r["n_stations"], time_limit=5.0)
        sweep.append({"cycle_time_s": c, "theoretical_min": math.ceil(total / c - 1e-9),
                      "rpw_stations": r["n_stations"], "exact_stations": e["n_stations"],
                      "exact_efficiency": e["balance_efficiency"]})
    return {
        "method": "SALBP-1 at c = takt on per-cycle element totals; UNOBS kept as one manual task per station; "
                  "IDLE excluded from work content (it is balance delay).",
        "cycle_time_s": takt, "work_content_s": total, "n_tasks": len(tasks),
        "tasks": [{"element_key": k, "time_s": t[k], "station": rows[k]["station"]} for k in tasks],
        "precedence": {"within_station": [list(e) for e in data["precedence_within"]],
                       "station_chain": [list(e) for e in data["precedence_chain"]],
                       "rule": "A->B inside a station if, in every observed cycle where both occur, every occurrence "
                               "of A ends before the first occurrence of B starts (transitively reduced). Across "
                               "stations the door moves WS10->WS20->WS30, so all work of station k precedes "
                               "station k+1 (conservative). UNOBS has no intra-station precedence."},
        "heuristics": heur, "exact": ex, "theoretical_min_stations": tm, "as_is": as_is,
        "salbp2_current_stations": s2, "cycle_time_sweep": sweep,
        "citations": [CITATIONS[k] for k in ("lecture06", "helgeson_birnie", "arcus", "weckenborg")],
        "interpretation": "Clips are cropped (observed share of the cycle: " + ", ".join(
            f"{st} {v['observed_fraction']:.0%}" for st, v in data["station_info"].items()
            if v.get("observed_fraction") is not None) + "), so re-balancing is an analysis, not a proposal: the "
            f"physical {len(data['stations'])}-station paced line is the base case.",
    }


def worker_machine(data, scen, takt):
    out = []
    for st, v in scen["station_times"].items():
        if not v["robot_model"]:
            continue
        h, r, i = v["human_s"], v["robot_s"], v["interaction_s"]
        cyc = max(h, r) + i
        out.append({"station": st, "model": v["robot_model"], "scenario": scen["name"], "human_s": h, "robot_s": r,
                    "interaction_s": i, "cycle_s": cyc, "takt_s": takt,
                    "human_idle_pct": (cyc - h - i) / cyc, "robot_idle_pct": (cyc - r - i) / cyc,
                    "takt_slack_s": takt - cyc,
                    "note": "human_idle_pct / robot_idle_pct are fractions of the station cycle "
                            "(cycle = max(human, robot) + interaction)"})
    return out


def machine_cluster(wm, takt, stations):
    """lecture04 deterministic machine-cluster model applied with the cobot as the server:
    n' = (Tm + Ts) / (Ts + Tr): Ts = cobot service at a station (robot work + handover), Tm = time the station runs
    without the cobot in one cycle (= takt - Ts on a paced line), Tr = cobot repositioning between stations."""
    rows, by = [], {w["station"]: w for w in wm}
    for w in wm:
        Ts = w["robot_s"] + w["interaction_s"]
        Tm = takt - Ts
        Tr = A.CELL_TRAVEL_S
        n = (Tm + Ts) / (Ts + Tr)
        rows.append({"station": w["station"], "Ts_s": Ts, "Tm_s": Tm, "Tr_s": Tr, "n_prime": n,
                     "max_stations_served": math.floor(n)})
    pairs = []
    for a, b in zip(stations, stations[1:]):
        if a in by and b in by:
            need = sum(by[s]["robot_s"] + by[s]["interaction_s"] + A.CELL_TRAVEL_S for s in (a, b))
            pairs.append({"stations": [a, b], "cobot_busy_s": need, "takt_s": takt, "feasible": need <= takt,
                          "robots_saved": 1 if need <= takt else 0})
    return {"formula": "n' = (Tm + Ts) / (Ts + Tr)  (ISyE 510 lecture04; a = Ts, t = Tm, b = Tr in the study-guide "
                       "notation n' = (a + t)/(a + b))", "travel_assumption_s": A.CELL_TRAVEL_S,
            "per_station": rows, "adjacent_pairs": pairs,
            "caveat": "Sharing needs a 7th-axis rail along the conveyor and both doors inside the reach envelope; "
                      "conveyor tracking makes the effective window shorter than takt."}


def rough_cut(cfg, data, scenarios):
    hours = A.annual_hours(cfg)
    cyc = A.annual_cycles(cfg)
    per_op_h = cfg["line"]["hours_per_shift"] * cfg["line"]["days_per_year"]
    out = {"method": CITATIONS["rca_lab"], "available_h_per_robot": hours * A.ROBOT_AVAILABILITY,
           "availability": A.ROBOT_AVAILABILITY, "efficiency": A.ROBOT_EFFICIENCY, "cushion": CUSHION,
           "annual_units": cyc, "robots": [], "operators": []}
    for s in scenarios:
        for st, v in s["station_times"].items():
            if not v["robot_model"]:
                continue
            req = v.get("robot_only_s", v["robot_s"]) * cyc / 3600.0 / A.ROBOT_EFFICIENCY
            av = hours * A.ROBOT_AVAILABILITY
            frac = req / av
            n = max(1, math.ceil(frac - 1e-12))
            u = req / (n * av)
            n_final = n + 1 if u >= CUSHION else n
            out["robots"].append({"scenario": s["name"], "station": st, "model": v["robot_model"],
                                  "required_h": req, "available_h": av, "fraction": frac, "robots": n,
                                  "utilization": u, "robots_after_cushion": n_final,
                                  "utilization_final": req / (n_final * av)})
        human = sum(v["human_s"] for v in s["station_times"].values())
        req_h = human * cyc / 3600.0
        cross = math.ceil(req_h / per_op_h - 1e-12)
        dedicated = len(s["station_times"]) * cfg["line"]["shifts_per_day"]
        out["operators"].append({"scenario": s["name"], "required_h": req_h, "available_h_per_operator": per_op_h,
                                 "dedicated_operators": dedicated, "dedicated_utilization": req_h / (dedicated * per_op_h),
                                 "cross_trained_operators": cross,
                                 "cross_trained_utilization": req_h / (cross * per_op_h) if cross else None,
                                 "note": "Cross-trained count is a rough-cut lower bound; on a paced line it is only "
                                         "reachable through re-balancing (see line_balancing.post_automation)."})
    return out


def run(cfg: dict) -> None:
    t0 = time.time()
    prov = {}
    ts = load_input(cfg, "time_study", prov)
    el = load_input(cfg, "elements", prov)
    au = load_input(cfg, "automatability", prov)
    cat = load_input(cfg, "robot_catalog", prov)
    cand = load_input(cfg, "candidates", prov)
    rtt = load_input(cfg, "robot_task_times", prov)
    data = build(cfg, ts, el, au, cat, cand, rtt)
    takt = data["takt"]

    am = AllocModel(cfg, data)
    base = am.evaluate_baseline()
    cap = (1 - ERGO_REDUCTION) * base["objective"]["awkward_posture_s_per_cycle"]

    def robot_scenario(name, mode, robots=True, **kw):
        s = am.solve(name, mode, robots=robots, helpers=False, **kw)
        if s["status"] != "OPTIMAL":   # levers alone cannot bring every station within takt: allow relief operator
            s2 = am.solve(name, mode, robots=robots, helpers=True, **kw)
            s2["note"] = f"robots alone {s['status']}; relief operator allowed as fallback"
            return s2
        return s
    scen = [base,
            am.solve("manual_relief", "min_cost", robots=False, helpers=True),
            robot_scenario("lean_layout", "min_cost", robots=False, layout=True),
            robot_scenario("min_cost", "min_cost", layout=True),
            robot_scenario("ergo_capped_min_cost", "min_cost", ergo_cap=cap, layout=True),
            robot_scenario("robust_min_cost", "min_cost", layout=True, cycle_cap=(1 - TAKT_MARGIN) * takt),
            robot_scenario("ergonomics_first", "ergonomics_first", layout=True),
            robot_scenario("max_automation", "max_automation", layout=True)]
    dropped = [s["name"] + ":" + s["status"] for s in scen if s["status"] not in ("OPTIMAL", "EVALUATED")]
    scen = [s for s in scen if s["status"] in ("OPTIMAL", "EVALUATED")]
    relief = next((s for s in scen if s["name"] == "manual_relief"), None)

    # ---------- recommendation: cheapest (equivalent annual cost) scenario that buys >=1 robot, fits takt at standard
    # time without a relief operator, and meets the ergonomic target
    for s in scen:
        s["meets_ergonomic_target"] = s["objective"]["awkward_posture_s_per_cycle"] <= cap + 1e-6
        s["meets_takt_margin"] = s["objective"]["max_station_cycle_s"] <= (1 - TAKT_MARGIN) * takt + 1e-6
        s["delta_annual_cost_vs_baseline_usd"] = s["objective"]["annual_cost_usd"] - base["objective"]["annual_cost_usd"]
        if relief:
            s["delta_annual_cost_vs_manual_relief_usd"] = s["objective"]["annual_cost_usd"] - relief["objective"]["annual_cost_usd"]
        s["recommended"] = False
    pool = [s for s in scen if s["status"] == "OPTIMAL" and not s["objective"]["helper_s_per_cycle"]]
    elig = [s for s in pool if s["meets_ergonomic_target"] and s["meets_takt_margin"]]
    rule = ("lowest equivalent annual cost among scenarios that fit every station within "
            f"{1 - TAKT_MARGIN:.0%} of takt at standard time (pacing margin) without a relief operator, and cut "
            "awkward-posture exposure (trunk bent >= 20 deg or arm "
            f"at/above shoulder) by >= {ERGO_REDUCTION:.0%} vs the as-is line")
    if not elig and [s for s in pool if s["meets_takt_margin"]]:
        elig = [s for s in pool if s["meets_takt_margin"]]
        rule = (f"no scenario met the ergonomic target; lowest equivalent annual cost among scenarios that fit "
                f"{1 - TAKT_MARGIN:.0%} of takt at standard time without a relief operator")
    if not elig:
        elig = pool or [s for s in scen if s["robots"]] or [s for s in scen if s["status"] == "OPTIMAL"]
        rule = ("no scenario met the ergonomic target; lowest equivalent annual cost among scenarios that fit takt "
                "without a relief operator")
    order = {s["name"]: i for i, s in enumerate(scen)}
    rec = min(elig, key=lambda s: (round(s["objective"]["annual_cost_usd"], 2), order[s["name"]]))
    rec["recommended"] = True
    o, ob = rec["objective"], base["objective"]
    buys = "; ".join(f"1 x {r['model']} at {r['station']} for {', '.join(e.split(':')[1] for e in r['elements'])}"
                     for r in rec["robots"]) or "no robot"
    lev = "; ".join(f"{l['lever']} at {l['station']} ({', '.join(e.split(':')[1] for e in l['elements'])}, "
                    f"-{l['reduction']:.0%})" for l in rec.get("layout_changes", []) if l["station"] != "LINE")
    if lev:
        buys += "; plus layout levers: " + lev
    ov = [f"{st} {v['overload_pct']:.1%} ({v['overload_basis']})" for st, v in base["station_overload"].items()
          if v["overload_pct"] > 0]
    vs_relief = ""
    if relief:
        d = rec["delta_annual_cost_vs_manual_relief_usd"]
        vs_relief = (f" The manual alternative (manual_relief: {relief['objective']['n_helpers_per_shift']} relief operator(s) "
                     f"per shift) costs {relief['objective']['annual_cost_usd']:,.0f} USD/yr; the recommended option is "
                     f"{'cheaper' if d < 0 else 'dearer'} by {abs(d):,.0f} USD/yr.")
    rec["justification"] = (
        f"Rule: {rule}. As-is at standard time: " + (", ".join(ov) if ov else "no station over takt") +
        f". {rec['name']} buys {buys}; max station cycle {ob['max_station_cycle_s']:.1f} -> {o['max_station_cycle_s']:.1f} s "
        f"(takt {takt:.0f} s). Awkward-posture exposure {ob['awkward_posture_s_per_cycle']:.1f} -> "
        f"{o['awkward_posture_s_per_cycle']:.1f} s/cycle; operator work {ob['human_s_per_cycle']:.1f} -> "
        f"{o['human_s_per_cycle']:.1f} s/cycle; equivalent annual cost {o['annual_cost_usd']:,.0f} USD/yr "
        f"(as-is labour {ob['annual_cost_usd']:,.0f})." + vs_relief)

    lb = line_balancing(data, takt)
    lb["post_automation"] = _human_rebalance(data, rec, takt)
    wm_src = rec if any(v["robot_model"] for v in rec["station_times"].values()) else \
        next((s for s in scen if s["robots"]), rec)
    wm = worker_machine(data, wm_src, takt)
    out = {
        "solver": "OR-Tools CP-SAT (exact MILP over binary assignment variables)",
        "inputs": prov,
        "time_basis": TIME_BASIS,
        "params": {"takt_s": takt, "annual_cycles": A.annual_cycles(cfg), "loaded_wage_usd_h": A.loaded_wage(cfg),
                   "crf": {"i": am.i, "n": am.n, "value": am.crf}, "handover_s": A.HANDOVER_S,
                   "integration_mult": A.INTEGRATION_MULT, "safety_usd": A.SAFETY_USD, "maint_pct": A.MAINT_PCT,
                   "ergonomic_reduction_target": ERGO_REDUCTION, "ergonomic_cap_s": cap,
                   "helper_walk_s": A.HELPER_WALK_S, "helper_annual_usd": am.helper_annual(),
                   "takt_margin": TAKT_MARGIN,
                   "bent_forward_threshold_deg": cfg["ergonomics"]["bent_forward_deg"][0]},
        "model": __import__("rcd.s5_optimize.allocation", fromlist=["x"]).__doc__,
        "as_is": {"overload_pct": base["overload_pct"], "stations": base["station_overload"],
                  "headline_stations": [st for st, v in base["station_overload"].items()
                                        if v["overload_pct"] > 0 and v["overload_basis"] == "measured"],
                  "note": "Overload at standard time (observed x rating x (1+PFD) + UNOBS). Only stations whose "
                          "cycle is mostly observed ('measured') support a headline claim; 'assumption' stations "
                          "rest on the unobserved block."},
        "elements": [{k: v for k, v in r.items()} for r in data["rows"].values()],
        "groups": list(data["groups"].values()),
        "scenarios": scen,
        "dropped_scenarios": dropped,
        "recommendation_rule": rule,
        "line_balancing": lb,
        "worker_machine": wm,
        "machine_cluster": machine_cluster(wm, takt, data["stations"]),
        "rough_cut_capacity": rough_cut(cfg, data, scen),
        "citations": CITATIONS,
        "notes": data["notes"] + [
            f"Element times use time_study '{TIME_BASIS}' (IE line-design basis: observed x rating x (1+PFD); UNOBS "
            "as given by s2). baseline_manual is the as-is line evaluated at that basis (it may exceed takt); "
            "manual_relief is the cheapest all-manual fix (roaming relief operator, ISyE 510 lecture06 'a roaming helper').",
            "UNOBS (unobserved part of the cycle) is one manual, non-automatable task per station.",
            "Labour savings are valued as freed hours x loaded wage (fractional FTE); a headcount reduction only "
            "materialises if line_balancing.post_automation shows fewer operators than stations.",
        ],
        "runtime_s": round(time.time() - t0, 2),
    }
    save_json(out, out_dir(cfg, "s5") / "allocation.json", contract="allocation")
    print(f"  s5: {len(scen)} scenarios, recommended={rec['name']} robots={len(rec['robots'])}; "
          f"LB exact {lb['exact']['n_stations']} st (Nt={lb['theoretical_min_stations']}); inputs={prov}")
