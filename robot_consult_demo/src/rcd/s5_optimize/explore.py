"""s5x - design-space exploration: every robot combination the line admits, not one answer.

The client chooses; we show that the whole space was searched and how the alternatives trade off.

1. Full-factorial enumeration (design of experiments over the decision factors)
     arm at each station in {none} U {models that passed s4 and were simulated in s6a for that station}
   x layout levers {off, on}
   x task split {min_cost, max_automation, ergonomics_first}   (what the arm does = how it moves, s5 MILP stages)
   x pacing {plan to takt, plan to 95 % of takt}
   Each combination fixes y (the arms) and the s5 MILP decides the task split; a combination that cannot meet takt is
   retried with a relief operator. Identical resulting designs (same element -> resource map) are merged.
2. Screening: Pareto filter (non-dominated set) on NPV (max), installed cost (min), awkward-posture s/door (min),
   max station cycle (min), number of arms (min). Every dominated design keeps the id of one design that beats it on
   every criterion ("we looked at it; this one is better on everything").
3. Simulation of the shortlist (screen Pareto set + the s5 named scenarios) with the s6b DES, common random numbers
   (same seeds for every design, so differences between two designs are paired comparisons - variance reduction).
   The final Pareto set replaces max cycle by the simulated takt-violation rate.
4. Preference analysis without asking the client for weights up front: SMAA-2 (Lahdelma & Salminen 2001) with
   weights uniform on the simplex -> rank acceptability (share of all possible weightings for which a design ranks
   1st/2nd/3rd) and central weight vector (the typical preference of a client who would pick it). Plus named
   stakeholder profiles (CFO, EHS, plant manager, pilot-first) and pairwise P(A preferred to B).
Values: linear value functions scaled to [0, 1] between the worst and best design of the simulated shortlist.
"""
from __future__ import annotations

import itertools
import math
import time
from collections import defaultdict

import numpy as np
from scipy import stats

from rcd.common.io import out_dir, save_json
from rcd.s5_optimize.allocation import AllocModel
from rcd.s5_optimize.data import build
from rcd.s5_optimize.inputs import load_input
from rcd.s5_optimize.run import TAKT_MARGIN
from rcd.s6b_des.run import REPS, SEED, replicate, station_specs
from rcd.s7_roi import assumptions as A
from rcd.s7_roi.run import econ

SPLITS = ("min_cost", "max_automation", "ergonomics_first")
SCREEN = [("npv_usd", "max"), ("capex_usd", "min"), ("awkward_s", "min"), ("max_cycle_s", "min"), ("n_robots", "min")]
CRITERIA = [  # final (simulated) criteria used for preference analysis
    {"key": "npv_usd", "label": "NPV", "sense": "max", "unit": "USD"},
    {"key": "capex_usd", "label": "Installed cost", "sense": "min", "unit": "USD"},
    {"key": "awkward_s", "label": "Awkward posture", "sense": "min", "unit": "s/door"},
    {"key": "takt_violation_pct", "label": "Takt overruns (DES)", "sense": "min", "unit": "% of station-cycles"},
    {"key": "n_robots", "label": "Arms to integrate", "sense": "min", "unit": "count"},
]
PERSONAS = [
    {"name": "Finance first (CFO)", "weights": {"npv_usd": .45, "capex_usd": .30, "awkward_s": .05,
                                                "takt_violation_pct": .15, "n_robots": .05}},
    {"name": "Safety & ergonomics (EHS)", "weights": {"npv_usd": .10, "capex_usd": .10, "awkward_s": .50,
                                                      "takt_violation_pct": .20, "n_robots": .10}},
    {"name": "Delivery & stability (plant manager)", "weights": {"npv_usd": .15, "capex_usd": .10, "awkward_s": .10,
                                                                 "takt_violation_pct": .55, "n_robots": .10}},
    {"name": "Low-risk pilot first", "weights": {"npv_usd": .15, "capex_usd": .35, "awkward_s": .05,
                                                 "takt_violation_pct": .10, "n_robots": .35}},
    {"name": "Balanced (equal weights)", "weights": {c["key"]: 0.2 for c in CRITERIA}},
]
N_WEIGHTS = 20000
MAX_SIM = 120
HEADLINE_K = 3            # options shown to the client up front
CITATIONS = {
    "smaa2": "Lahdelma, R. & Salminen, P. (2001) SMAA-2: Stochastic multicriteria acceptability analysis for group "
             "decision making. Operations Research 49(3):444-454",
    "pareto": "Deb, K. (2001) Multi-Objective Optimization using Evolutionary Algorithms, ch. 2 (non-domination)",
    "crn": "Law, A.M. (2015) Simulation Modeling and Analysis, 5th ed., sec. 11.2 common random numbers",
    "doe": "Montgomery, D.C. (2017) Design and Analysis of Experiments, 9th ed., ch. 5 factorial designs",
}


def _signature(s):
    return (tuple(sorted((a["element_key"], a["resource"], a.get("group_id") or "", a.get("lever") or "")
                         for a in s["assignment"])), s["objective"].get("n_helpers_per_shift", 0))


def _dominates(a, b, crit):
    ge = all((a[k] >= b[k] - 1e-9) if s == "max" else (a[k] <= b[k] + 1e-9) for k, s in crit)
    gt = any((a[k] > b[k] + 1e-9) if s == "max" else (a[k] < b[k] - 1e-9) for k, s in crit)
    return ge and gt


def pareto(items, crit):
    """Non-dominated subset (O(n^2), n <= a few thousand) and, for each dominated item, one dominating item."""
    front, by = [], {}
    for x in items:
        dom = [y for y in items if y is not x and _dominates(y["metrics"], x["metrics"], crit)]
        if not dom:
            front.append(x)
        else:   # report the dominator with the largest total normalised margin
            by[x["id"]] = max(dom, key=lambda y: sum(
                abs(y["metrics"][k] - x["metrics"][k]) / (abs(x["metrics"][k]) + 1.0) for k, _ in crit))["id"]
    return front, by


def values(items, criteria):
    """Linear value functions on [0, 1] (1 = best in the set) - the scale used for SMAA and the report sliders."""
    V = np.zeros((len(items), len(criteria)))
    rng = {}
    for j, c in enumerate(criteria):
        x = np.array([it["metrics"][c["key"]] for it in items], float)
        lo, hi = float(x.min()), float(x.max())
        rng[c["key"]] = [lo, hi]
        if hi - lo < 1e-12:
            V[:, j] = 1.0
        else:
            V[:, j] = (x - lo) / (hi - lo) if c["sense"] == "max" else (hi - x) / (hi - lo)
    return V, rng


def smaa(V, n=N_WEIGHTS, seed=SEED):
    rng = np.random.default_rng(seed)
    W = rng.dirichlet(np.ones(V.shape[1]), size=n)          # uniform on the weight simplex
    U = W @ V.T                                              # n x m utilities
    order = np.argsort(-U, axis=1)
    m = V.shape[0]
    ranks = np.empty_like(order)
    ranks[np.arange(n)[:, None], order] = np.arange(m)[None, :]
    b = np.stack([(ranks == r).mean(axis=0) for r in range(min(3, m))], axis=1)   # m x 3
    cw = []
    for i in range(m):
        sel = ranks[:, i] == 0
        cw.append(W[sel].mean(axis=0).tolist() if sel.any() else None)
    return b, cw, U


def pick_headline(front, U, idx, personas, rec_id, compare, k=HEADLINE_K):
    """The few options shown to the client up front (the rest stay in the appendix / explorer).

    1. Practical options: final-front designs that need no extra relief operator and have NPV > 0 (a design that only
       survives by hiring a person, or that loses money, is evidence that we looked, not a proposal).
    2. Archetypes = number of arms (0 = layout only, 1, 2, 3 ...): near-identical variants collapse into one card;
       the representative of an archetype is its design with the highest expected utility over all weightings.
    3. Choose the k archetypes that maximise preference coverage: the share of all weightings (SMAA samples) for which
       the best practical design belongs to one of the shown archetypes. Ties: keep the s5 recommendation, then cheaper.
    """
    prac = [d for d in front if not d["relief_operators_per_shift"] and d["metrics"]["npv_usd"] > 0]
    rule = "final Pareto, no relief operator, NPV > 0"
    if len(prac) < k:
        prac = [d for d in front if not d["relief_operators_per_shift"]] or list(front)
        rule = "final Pareto, no relief operator" if prac and not prac[0]["relief_operators_per_shift"] else "final Pareto"
    cols = [idx[d["id"]] for d in prac]
    arche = defaultdict(list)
    for d in prac:
        arche[d["metrics"]["n_robots"]].append(d)
    mean_u = U.mean(axis=0)
    rep = {a: max(ds, key=lambda d: mean_u[idx[d["id"]]]) for a, ds in arche.items()}
    best = np.array([prac[j]["metrics"]["n_robots"] for j in np.argmax(U[:, cols], axis=1)])
    rec_a = next((d["metrics"]["n_robots"] for d in prac if d["id"] == rec_id), None)
    pick = max(itertools.combinations(sorted(arche), min(k, len(arche))),
               key=lambda S: (round(float(np.isin(best, S).mean()), 6), rec_a in S,
                              -sum(rep[a]["metrics"]["capex_usd"] for a in S)))
    cover = float(np.isin(best, pick).mean())
    pers = defaultdict(list)
    for p in personas:
        u = p["utility"]
        b_ = max(prac, key=lambda d: u[d["id"]])
        pers[b_["metrics"]["n_robots"]].append(p["name"])
    keys = [c["key"] for c in CRITERIA]
    opts = []
    for a in pick:
        d = rep[a]
        others = [rep[x] for x in pick if x != a]
        strong = [c["key"] for c in CRITERIA if all(
            (d["metrics"][c["key"]] > o["metrics"][c["key"]]) if c["sense"] == "max" else
            (d["metrics"][c["key"]] < o["metrics"][c["key"]]) for o in others)]
        weak = [c["key"] for c in CRITERIA if all(
            (d["metrics"][c["key"]] < o["metrics"][c["key"]]) if c["sense"] == "max" else
            (d["metrics"][c["key"]] > o["metrics"][c["key"]]) for o in others)]
        title = "Layout only (no robot)" if a == 0 else \
            f"{a} robot{'s' if a > 1 else ''}: " + ", ".join(f"{r['model']} @ {r['station']}" for r in d["robots"])
        opts.append({"design": d["id"], "archetype_robots": a, "title": title,
                     "share_of_weightings": float((best == a).mean()),
                     "variants_in_archetype": [x["id"] for x in arche[a]],
                     "personas": pers.get(a, []), "is_recommended": d["id"] == rec_id,
                     "strengths": strong, "weaknesses": weak,
                     "metrics": {kk: d["metrics"][kk] for kk in keys + ["payback_years", "annual_savings_usd",
                                                                         "throughput_per_hr", "human_s"]}})
    pairs = [compare(rep[x], rep[y]) for x, y in itertools.combinations(pick, 2)]
    return {"method": pick_headline.__doc__, "k": k, "practical_rule": rule,
            "n_final_front": len(front), "n_practical": len(prac), "archetypes": sorted(arche),
            "coverage": cover, "options": opts, "pairs": pairs,
            "not_shown": {"relief_operator": sum(1 for d in front if d["relief_operators_per_shift"]),
                          "negative_npv": sum(1 for d in front if not d["relief_operators_per_shift"]
                                              and d["metrics"]["npv_usd"] <= 0),
                          "same_archetype_variants": len(prac) - len(pick),
                          "archetypes_not_shown": [a for a in arche if a not in pick]}}


def _label(cfgm, layout, split, pacing, relief, stations):
    arms = " · ".join(f"{st} {cfgm.get(st) or '-'}" for st in stations)
    extra = [x for x in (("layout" if layout else ""), ("" if split == "min_cost" else split),
                         ("95%-takt" if pacing else ""), ("+relief op" if relief else "")) if x]
    return arms + ((" | " + ", ".join(extra)) if extra else "")


def run(cfg: dict) -> None:
    t0 = time.time()
    prov = {}
    ins = [load_input(cfg, n, prov) for n in ("time_study", "elements", "automatability", "robot_catalog",
                                              "candidates", "robot_task_times")]
    alloc = load_input(cfg, "allocation", prov)
    el = ins[1]
    data = build(cfg, *ins)
    takt = data["takt"]
    stations = data["stations"]
    am = AllocModel(cfg, data)
    base = am.evaluate_baseline()
    ec = cfg["economics"]
    P0 = {"wage": A.hourly_wage(), "price_pos": 0.5, "integ": A.INTEGRATION_MULT, "maint": A.MAINT_PCT,
          "i": ec["discount_rate"], "n": ec["horizon_years"], "shifts": cfg["line"]["shifts_per_day"],
          "track": A.TRACK_USD, "changer": A.TOOL_CHANGER_USD}
    arms_at = defaultdict(set)
    for k in am.keys:
        for o in am.opts[k]:
            arms_at[data["rows"][k]["station"]].add(o["model"])
    levels = {st: [None] + sorted(arms_at.get(st, ())) for st in stations}

    # ------------------------------------------------------------------ 1. full-factorial enumeration
    designs, by_sig, runs = [], {}, []
    n_solves = 0
    for combo in itertools.product(*[levels[st] for st in stations]):
        fm = dict(zip(stations, combo))
        for layout, split, pacing in itertools.product((False, True), SPLITS, (False, True)):
            if split != "min_cost" and not any(combo):
                continue   # no arm: the task-split objective has nothing to decide
            cap = (1 - TAKT_MARGIN) * takt if pacing else None
            s = am.solve("x", split, layout=layout, cycle_cap=cap, fix_models=fm, time_limit=10.0)
            n_solves += 1
            relief = False
            if s["status"] not in ("OPTIMAL", "FEASIBLE"):
                s = am.solve("x", split, layout=layout, cycle_cap=cap, fix_models=fm, helpers=True, time_limit=10.0)
                n_solves += 1
                relief = True
            run_rec = {"arms": fm, "layout": layout, "split": split, "pacing_95": pacing, "status": s["status"],
                       "relief": relief}
            runs.append(run_rec)
            if s["status"] not in ("OPTIMAL", "FEASIBLE") or not all(s["checks"].values()):
                run_rec["feasible"] = False
                continue
            run_rec["feasible"] = True
            sig = _signature(s)
            variant = {"layout": layout, "split": split, "pacing_95": pacing, "relief": relief}
            if sig in by_sig:
                by_sig[sig]["variants"].append(variant)
                run_rec["design"] = by_sig[sig]["id"]
                continue
            e = econ(cfg, s, base, None, P0)
            o = s["objective"]
            makers = sorted({data["robots"][r["model"]].get("maker", "?") for r in s["robots"]})
            d = {"id": f"D{len(designs) + 1:03d}",
                 "label": _label(fm, bool(s["layout_changes"]), split, pacing, relief or bool(o["n_helpers_per_shift"]),
                                 stations),
                 "arms": {st: next((r["model"] for r in s["robots"] if r["station"] == st), None) for st in stations},
                 "layout_levers": [f"{l['lever']}@{l['station']}" for l in s["layout_changes"]],
                 "relief_operators_per_shift": o["n_helpers_per_shift"],
                 "makers": makers,
                 "robots": [{"model": r["model"], "station": r["station"], "maker": data["robots"][r["model"]].get("maker"),
                             "tasks": [x.split(":")[1] for x in r["elements"]], "categories": r["categories"],
                             "robot_s": s["station_times"][r["station"]]["robot_only_s"],
                             "starts_after_human_s": s["station_times"][r["station"]]["robot_start_after_human_s"],
                             "tool_changes_per_cycle": r["tool_changes_per_cycle"], "capex_usd": r["capex_usd"]}
                            for r in s["robots"]],
                 "station_cycle_s": {st: v["cycle_s"] for st, v in s["station_times"].items()},
                 "station_human_s": {st: v["human_s"] for st, v in s["station_times"].items()},
                 "metrics": {"npv_usd": e["npv_usd"], "capex_usd": e["capex_usd"],
                             "annual_savings_usd": e["annual_savings_usd"], "payback_years": e["payback_years"],
                             "irr": e["irr"], "annual_cost_usd": o["annual_cost_usd"],
                             "awkward_s": o["awkward_posture_s_per_cycle"], "max_cycle_s": o["max_station_cycle_s"],
                             "human_s": o["human_s_per_cycle"], "n_robots": len(s["robots"]),
                             "n_makers": len(makers)},
                 "variants": [variant], "_scen": s}
            by_sig[sig] = d
            run_rec["design"] = d["id"]
            designs.append(d)

    # named s5 scenarios -> design ids (so the report can say where the classic answers sit in the space)
    named = {}
    for s in alloc["scenarios"]:
        if s["name"] == "baseline_manual":
            continue
        sig = _signature(s)
        if sig not in by_sig:   # e.g. a scenario outside the enumerated factors: add it as its own design
            e = econ(cfg, s, base, None, P0)
            o = s["objective"]
            makers = sorted({data["robots"][r["model"]].get("maker", "?") for r in s["robots"]})
            fm = {st: next((r["model"] for r in s["robots"] if r["station"] == st), None) for st in stations}
            d = {"id": f"D{len(designs) + 1:03d}", "label": _label(fm, bool(s.get("layout_changes")), "min_cost", False,
                                                                   bool(o.get("n_helpers_per_shift")), stations),
                 "arms": fm, "layout_levers": [f"{l['lever']}@{l['station']}" for l in s.get("layout_changes", [])],
                 "relief_operators_per_shift": o.get("n_helpers_per_shift", 0), "makers": makers,
                 "robots": [{"model": r["model"], "station": r["station"],
                             "maker": data["robots"][r["model"]].get("maker"),
                             "tasks": [x.split(":")[1] for x in r["elements"]], "categories": r["categories"],
                             "robot_s": s["station_times"][r["station"]].get("robot_only_s", s["station_times"][r["station"]]["robot_s"]),
                             "starts_after_human_s": s["station_times"][r["station"]].get("robot_start_after_human_s"),
                             "tool_changes_per_cycle": r.get("tool_changes_per_cycle", 0), "capex_usd": r["capex_usd"]}
                            for r in s["robots"]],
                 "station_cycle_s": {st: v["cycle_s"] for st, v in s["station_times"].items()},
                 "station_human_s": {st: v["human_s"] for st, v in s["station_times"].items()},
                 "metrics": {"npv_usd": e["npv_usd"], "capex_usd": e["capex_usd"],
                             "annual_savings_usd": e["annual_savings_usd"], "payback_years": e["payback_years"],
                             "irr": e["irr"], "annual_cost_usd": o["annual_cost_usd"],
                             "awkward_s": o["awkward_posture_s_per_cycle"], "max_cycle_s": o["max_station_cycle_s"],
                             "human_s": o["human_s_per_cycle"], "n_robots": len(s["robots"]), "n_makers": len(makers)},
                 "variants": [{"named": s["name"]}], "_scen": s}
            by_sig[sig] = d
            designs.append(d)
        named.setdefault(by_sig[sig]["id"], []).append(s["name"])
    for d in designs:
        d["named_scenarios"] = named.get(d["id"], [])
    rec_name = next(s["name"] for s in alloc["scenarios"] if s.get("recommended"))
    rec_id = next((i for i, ns in named.items() if rec_name in ns), None)

    # ------------------------------------------------------------------ 2. screening Pareto (analytic criteria)
    front, dom_by = pareto(designs, SCREEN)
    fset = {d["id"] for d in front}
    for d in designs:
        d["pareto_screen"] = d["id"] in fset
        d["dominated_by"] = dom_by.get(d["id"])

    # ------------------------------------------------------------------ 3. DES on the shortlist, common random numbers
    short = [d for d in designs if d["pareto_screen"] or d["named_scenarios"]]
    if len(short) > MAX_SIM:   # keep named scenarios, then the best-NPV screen-Pareto designs
        keep = [d for d in short if d["named_scenarios"]]
        keep += sorted([d for d in short if not d["named_scenarios"]], key=lambda d: -d["metrics"]["npv_usd"])
        short = keep[:MAX_SIM]
    rows, cv, _ = station_specs(alloc, el, takt)
    base_des = replicate("baseline_manual", base, rows, cv, takt, stations, "paced", SEED)
    for d in short:
        r = replicate(d["id"], d["_scen"], rows, cv, takt, stations, "paced", SEED)   # same seed = CRN
        d["des"] = {"throughput_per_hr": r["throughput_per_hr"], "ci95_half_width": r["ci95_half_width"],
                    "takt_violations_pct": r["takt_violations_pct"], "line_stop_pct": r["line_stop_pct"],
                    "reps": r["throughput_reps_per_hr"]}
        d["metrics"]["takt_violation_pct"] = r["takt_violations_pct"]
        d["metrics"]["throughput_per_hr"] = r["throughput_per_hr"]
        d["simulated"] = True
    fin_crit = [(c["key"], c["sense"]) for c in CRITERIA]
    ffront, fdom = pareto(short, fin_crit)
    ffset = {d["id"] for d in ffront}
    for d in short:
        d["pareto_final"] = d["id"] in ffset
        d["dominated_by_final"] = fdom.get(d["id"])

    # ------------------------------------------------------------------ 4. preference analysis on the final front
    Vs, vrange = values(short, CRITERIA)   # scaled on the simulated shortlist (same scale as the report explorer)
    V = Vs[[short.index(d) for d in ffront]]
    b, cw, U = smaa(V)
    for i, d in enumerate(ffront):
        d["smaa"] = {"rank1": float(b[i, 0]), "rank2": float(b[i, 1]) if b.shape[1] > 1 else 0.0,
                     "rank3": float(b[i, 2]) if b.shape[1] > 2 else 0.0,
                     "central_weights": dict(zip([c["key"] for c in CRITERIA], cw[i])) if cw[i] else None,
                     "values": dict(zip([c["key"] for c in CRITERIA], V[i].tolist()))}
    idx = {d["id"]: i for i, d in enumerate(ffront)}
    personas = []
    for p in PERSONAS:
        w = np.array([p["weights"][c["key"]] for c in CRITERIA])
        u = V @ w
        top = [ffront[i]["id"] for i in np.argsort(-u)[:3]]
        personas.append({**p, "best": top[0], "top3": top, "utility": {ffront[i]["id"]: float(u[i]) for i in range(len(ffront))}})

    def compare(a, b_):
        """A vs B card: per-criterion winner, dominance, P(A preferred) over all weightings, paired DES difference."""
        rows_ = []
        for c in CRITERIA:
            x, y = a["metrics"][c["key"]], b_["metrics"][c["key"]]
            better = (x > y) if c["sense"] == "max" else (x < y)
            same = abs(x - y) <= 1e-9 * max(1.0, abs(x), abs(y))
            rows_.append({"criterion": c["key"], "a": x, "b": y, "delta": x - y,
                          "winner": "tie" if same else ("a" if better else "b")})
        pa = None
        if a["id"] in idx and b_["id"] in idx:
            pa = float((U[:, idx[a["id"]]] > U[:, idx[b_["id"]]]).mean())
        diff = np.array(a["des"]["reps"]) - np.array(b_["des"]["reps"])    # paired (CRN)
        hw = float(stats.t.ppf(0.975, REPS - 1) * diff.std(ddof=1) / math.sqrt(REPS)) if diff.std() > 0 else 0.0
        da = _dominates(a["metrics"], b_["metrics"], fin_crit)
        db = _dominates(b_["metrics"], a["metrics"], fin_crit)
        wa = [r["criterion"] for r in rows_ if r["winner"] == "a"]
        wb = [r["criterion"] for r in rows_ if r["winner"] == "b"]
        lab = {c["key"]: c["label"] for c in CRITERIA}
        verdict = (f"{a['id']} dominates {b_['id']} (better or equal on every criterion)" if da else
                   f"{b_['id']} dominates {a['id']}" if db else
                   f"{a['id']} wins on {', '.join(lab[k] for k in wa) or 'nothing'}; {b_['id']} wins on "
                   f"{', '.join(lab[k] for k in wb) or 'nothing'}"
                   + (f"; {a['id']} is preferred for {pa:.0%} of all possible weightings" if pa is not None else ""))
        trade = None
        dc = a["metrics"]["capex_usd"] - b_["metrics"]["capex_usd"]
        de = b_["metrics"]["awkward_s"] - a["metrics"]["awkward_s"]
        if abs(dc) > 1 and de * dc > 0:
            trade = (f"{'extra' if dc > 0 else 'saved'} installed cost per awkward-posture second "
                     f"{'removed' if dc > 0 else 'kept'} per door: {abs(dc) / abs(de):,.0f} USD")
        return {"a": a["id"], "b": b_["id"], "rows": rows_, "a_dominates": da, "b_dominates": db,
                "p_a_preferred": pa, "throughput_diff_per_hr": float(diff.mean()), "throughput_diff_ci95": hw,
                "verdict": verdict, "tradeoff": trade}

    by_id = {d["id"]: d for d in designs}
    anchor = by_id.get(rec_id) or max(ffront, key=lambda d: d["smaa"]["rank1"])
    featured, seen = [], set()
    others = [by_id[p["best"]] for p in personas] + [
        min(ffront, key=lambda d: d["metrics"]["capex_usd"]), max(ffront, key=lambda d: d["metrics"]["npv_usd"]),
        min(ffront, key=lambda d: d["metrics"]["awkward_s"]), max(ffront, key=lambda d: d["smaa"]["rank1"])]
    if rec_id and not by_id[rec_id].get("pareto_final"):
        others.append(by_id[by_id[rec_id]["dominated_by_final"]])
    for o_ in others:
        if o_["id"] != anchor["id"] and o_["id"] not in seen:
            seen.add(o_["id"])
            featured.append(compare(anchor, o_))

    headline = pick_headline(ffront, U, idx, personas, rec_id, compare)

    # main effects: best attainable outcome given one factor level (what if you insist on model X at station S)
    feas = [d for d in designs]
    effects = []
    for st in stations:
        for lv in levels[st]:
            sub = [d for d in feas if d["arms"][st] == lv]
            if not sub:
                effects.append({"station": st, "arm": lv, "n_designs": 0})
                continue
            bn = max(sub, key=lambda d: d["metrics"]["npv_usd"])
            effects.append({"station": st, "arm": lv, "n_designs": len(sub), "best_npv_usd": bn["metrics"]["npv_usd"],
                            "best_npv_design": bn["id"], "min_awkward_s": min(d["metrics"]["awkward_s"] for d in sub),
                            "min_capex_usd": min(d["metrics"]["capex_usd"] for d in sub),
                            "on_final_front": sum(1 for d in sub if d.get("pareto_final"))})

    n_feas = sum(1 for r in runs if r["feasible"])
    out = {
        "method": __doc__,
        "inputs": prov,
        "citations": CITATIONS,
        "factors": {"arm_per_station": {st: [x or "none" for x in levels[st]] for st in stations},
                    "layout_levers": ["off", "on"], "task_split": list(SPLITS),
                    "pacing": ["takt", f"{1 - TAKT_MARGIN:.0%} of takt"],
                    "fallback": "relief operator when the arms cannot meet takt"},
        "criteria_screen": [{"key": k, "sense": s} for k, s in SCREEN],
        "criteria": CRITERIA,
        "value_ranges": vrange,
        "counts": {"arm_combinations": int(np.prod([len(levels[st]) for st in stations])),
                   "factor_runs": len(runs), "milp_solves": n_solves, "feasible_runs": n_feas,
                   "infeasible_runs": len(runs) - n_feas, "unique_designs": len(designs),
                   "screen_pareto": len(front), "simulated": len(short), "final_pareto": len(ffront),
                   "weight_samples": N_WEIGHTS},
        "baseline": {"awkward_s": base["objective"]["awkward_posture_s_per_cycle"],
                     "max_cycle_s": base["objective"]["max_station_cycle_s"],
                     "human_s": base["objective"]["human_s_per_cycle"],
                     "throughput_per_hr": base_des["throughput_per_hr"],
                     "takt_violation_pct": base_des["takt_violations_pct"]},
        "takt_s": takt, "stations": stations,
        "recommended_scenario": rec_name, "recommended_design": rec_id,
        "anchor_design": anchor["id"],
        "personas": personas,
        "headline": headline,
        "featured_comparisons": featured,
        "main_effects": effects,
        "designs": [{k: v for k, v in d.items() if k != "_scen"} for d in designs],
        "runs": runs,
        "runtime_s": 0.0,
    }
    for d in out["designs"]:
        if "des" in d:
            d["des"] = {k: v for k, v in d["des"].items() if k != "reps"}
    out["runtime_s"] = round(time.time() - t0, 2)
    save_json(out, out_dir(cfg, "s5x") / "design_space.json", contract="design_space")
    c = out["counts"]
    print(f"  s5x: {c['arm_combinations']} arm combos x levers -> {c['factor_runs']} runs ({c['milp_solves']} MILP), "
          f"{c['feasible_runs']} feasible, {c['unique_designs']} unique designs; screen Pareto {c['screen_pareto']}, "
          f"simulated {c['simulated']}, final Pareto {c['final_pareto']}; anchor {anchor['id']} "
          f"(recommended {rec_name} = {rec_id}); personas: "
          + "; ".join(f"{p['name'].split(' (')[0]} -> {p['best']}" for p in personas)
          + f"; {out['runtime_s']} s")
