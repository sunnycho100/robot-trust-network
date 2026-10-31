"""Build the s5 work-element table (element_key -> times, ergonomics, robot options) and the precedence graph."""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from rcd.s7_roi import assumptions as A

TIME_BASIS = "standard_s"   # IE line-design basis: observed x rating x (1 + PFD allowance); UNOBS as given by s2
ROBOT_ALLOWED = ("robot", "human_robot")
NON_WORK = ("IDLE",)    # waiting is balance delay, not work content (excluded from line balancing only)


def _stations(cfg, ts):
    have = {r["station"] for r in ts["element_stats"]}
    order = [s for s in cfg["data"]["stations"] if s in have]
    return order + sorted(have - set(order))


def _task_index(au):
    idx = {}
    for t in au["tasks"]:
        key = t.get("element_key") or f"{t['station']}:{t['code']}"
        idx[key] = t
    return idx


def _price_mid(cat_robot) -> float:
    lo, hi = cat_robot["price_usd_range"][0], cat_robot["price_usd_range"][-1]
    return 0.5 * (float(lo) + float(hi))


def _speed(r) -> float:
    v = r.get("joint_speed_deg_s") or 180.0
    return float(np.mean(v)) if isinstance(v, (list, tuple)) else float(v)


def build(cfg, ts, el, au, cat, cand, rtt):
    takt = float(cfg["line"]["takt_s"])
    full = float(cfg.get("observation", {}).get("full_cycle_s", takt))
    bent_lo = float(cfg["ergonomics"]["bent_forward_deg"][0])
    stations = _stations(cfg, ts)
    notes = []

    # ---------------- times per element_key (per-cycle totals from s2)
    rows = {}
    for r in ts["element_stats"]:
        key = f"{r['station']}:{r['code']}"
        rows[key] = {"element_key": key, "station": r["station"], "code": r["code"],
                     "time_s": float(r[TIME_BASIS]), "mean_s": float(r["mean_s"]), "std_s": float(r.get("std_s") or 0),
                     "standard_s": float(r["standard_s"]), "n": r.get("n")}
    for v in rows.values():
        v["non_work"] = v["code"] in NON_WORK
    for st in stations:
        obs = sum(v["time_s"] for v in rows.values() if v["station"] == st and v["code"] != "UNOBS" and not v["non_work"])
        k = f"{st}:UNOBS"
        if k not in rows:
            u = max(0.0, full - obs)
            rows[k] = {"element_key": k, "station": st, "code": "UNOBS", "time_s": u, "mean_s": u, "std_s": 0.0,
                       "standard_s": u, "n": None}
            notes.append(f"{st}: UNOBS not in time_study; set to full_cycle_s - observed = {u:.2f} s")
        if rows[k]["time_s"] < 0:
            notes.append(f"{st}: UNOBS from time_study is negative ({rows[k]['time_s']:.2f} s, observed clip longer "
                         f"than full_cycle_s); clamped to 0")
            rows[k]["time_s"] = 0.0
        tot = obs + rows[k]["time_s"]
        idle = sum(v["time_s"] for v in rows.values() if v["station"] == st and v["non_work"])
        if idle > 0:
            notes.append(f"{st}: {idle:.2f} s/cycle of IDLE (waiting) treated as balance delay, not as work content; "
                         f"work content observed + UNOBS = {tot:.2f} s vs takt {takt:.0f} s")
    # as-is station overload at standard time and how much of it is measured (observed fraction of the cycle)
    stinfo = {}
    sst = {r["station"]: r for r in ts.get("station_stats", [])}
    for st in stations:
        work = sum(v["time_s"] for v in rows.values() if v["station"] == st and not v["non_work"])
        r = sst.get(st, {})
        of = r.get("observed_fraction")
        if of is None:
            om = r.get("observed_mean_s")
            of = min(1.0, float(om) / full) if om is not None else None
        basis = r.get("overload_basis") or (("measured" if of >= 0.8 else "assumption") if of is not None else "unknown")
        stinfo[st] = {"station": st, "standard_work_s": work, "takt_s": takt, "utilization": work / takt,
                      "overload_s": max(0.0, work - takt), "overload_pct": max(0.0, work / takt - 1),
                      "observed_fraction": of, "overload_basis": basis,
                      "unobs_s": rows.get(f"{st}:UNOBS", {}).get("time_s")}
        if work > takt + 1e-9:
            notes.append(f"{st}: standard work content {work:.1f} s exceeds takt {takt:.0f} s by {work / takt - 1:.1%} "
                         f"(basis: {basis}; observed fraction {of if of is None else round(of, 2)})")

    # ---------------- ergonomics: awkward-posture seconds per cycle per element_key. If s2 gives the share of each
    # element's frames in a posture class (posture_share: bent = trunk >= 20 deg, overhead = elbow at/above shoulder),
    # awkward = (bent + overhead) x duration (capped at the duration); otherwise trunk_flex_deg_mean >= 20 deg.
    per_cycle = defaultdict(lambda: defaultdict(float))
    comp = defaultdict(lambda: defaultdict(float))
    ncyc = defaultdict(set)
    first, last = defaultdict(dict), defaultdict(dict)
    for c in el["cycles"]:
        st, cid = c["station"], c["cycle_id"]
        ncyc[st].add(cid)
        for e in c["elements"]:
            key = f"{st}:{e['code']}"
            dur = float(e["duration_s"])
            ps = e.get("posture_share")
            if isinstance(ps, dict) and ("bent" in ps or "overhead" in ps):
                b, o = float(ps.get("bent") or 0), float(ps.get("overhead") or 0)
                per_cycle[key][cid] += min(1.0, b + o) * dur
                comp[key]["bent"] += b * dur
                comp[key]["overhead"] += o * dur
            else:
                flex = e.get("trunk_flex_deg_mean")
                if flex is not None and flex >= bent_lo:
                    per_cycle[key][cid] += dur
                    comp[key]["bent"] += dur
            first[(st, cid)][e["code"]] = min(first[(st, cid)].get(e["code"], 1e18), float(e["start_s"]))
            last[(st, cid)][e["code"]] = max(last[(st, cid)].get(e["code"], -1e18), float(e["end_s"]))
    for key, v in rows.items():
        n = max(1, len(ncyc[v["station"]]))
        v["ergo_s"] = float(sum(per_cycle[key].values()) / n) if v["code"] != "UNOBS" else 0.0
        v["ergo_s"] = min(v["ergo_s"], v["time_s"])
        v["bent_trunk_s"] = comp[key]["bent"] / n
        v["overhead_arm_s"] = comp[key]["overhead"] / n
        v["first_start_mean_s"] = float(np.mean([first[(v["station"], c)][v["code"]] for c in ncyc[v["station"]]
                                                 if v["code"] in first[(v["station"], c)]] or [1e9]))

    # ---------------- precedence: A -> B if in every cycle where both occur, A always ends before B starts
    prec = []
    for st in stations:
        codes = sorted({v["code"] for v in rows.values() if v["station"] == st and v["code"] not in ("UNOBS",) + NON_WORK})
        cyc = sorted(ncyc[st])
        for a in codes:
            for b in codes:
                if a == b:
                    continue
                both = [c for c in cyc if a in first[(st, c)] and b in first[(st, c)]]
                if both and all(last[(st, c)][a] <= first[(st, c)][b] + 0.05 for c in both):
                    prec.append((f"{st}:{a}", f"{st}:{b}"))
    # transitive reduction inside stations
    succ = defaultdict(set)
    for a, b in prec:
        succ[a].add(b)

    def reach(a, b, skip):
        stack, seen = [x for x in succ[a] if (a, x) != skip], set()
        while stack:
            x = stack.pop()
            if x == b:
                return True
            if x not in seen:
                seen.add(x)
                stack += list(succ[x])
        return False
    within = [(a, b) for a, b in prec if not reach(a, b, (a, b))]
    # station chain: the door visits WS10 -> WS20 -> WS30; all work of station k precedes station k+1 (conservative)
    chain = []
    for s1, s2 in zip(stations, stations[1:]):
        k1 = [k for k, v in rows.items() if v["station"] == s1 and v["code"] not in NON_WORK]
        k2 = [k for k, v in rows.items() if v["station"] == s2 and v["code"] not in NON_WORK]
        sinks = [k for k in k1 if not any(a == k for a, _ in within)]
        sources = [k for k in k2 if not any(b == k for _, b in within)]
        chain += [(a, b) for a in sinks for b in sources]

    # ---------------- robot options from s3 (allowed) x s4 (passed models) x s6a (simulated time)
    tasks = _task_index(au)
    robots = {r["model"]: dict(r, _price_mid=_price_mid(r)) for r in cat["robots"]}
    rt = {(e["model"], e["group_id"]): e for e in rtt["entries"]}
    options = defaultdict(list)   # element_key -> [ {model, group_id, category, time_s} ]
    groups = {}
    for g in cand["task_groups"]:
        st = g["station"]
        gkeys = [f"{st}:{c}" for c in g["codes"] if f"{st}:{c}" in rows]
        allowed = [k for k in gkeys if tasks.get(k, {}).get("allocation") in ROBOT_ALLOWED and not k.endswith(":UNOBS")]
        groups[g["group_id"]] = {"group_id": g["group_id"], "station": st, "category": g.get("category", ""),
                                 "codes": g["codes"], "allowed_keys": allowed, "models": []}
        mount = g.get("mounting") or (g.get("placement") or {}).get("mounting") or "fixed_floor"
        groups[g["group_id"]]["mounting"] = mount
        groups[g["group_id"]]["required_reach_mm"] = (g.get("requirements") or {}).get("reach_mm")
        for e in g["evaluated"]:
            if not e.get("passed") or e["model"] not in robots or (e["model"], g["group_id"]) not in rt:
                continue
            ent = rt[(e["model"], g["group_id"])]
            if "surrogate" in json.dumps(ent.get("sim", "")).lower():
                continue   # only models simulated with their own kinematics may be bought
            T = float(ent["cycle_time_s"])
            bc = ent.get("breakdown_by_code") or {}
            if bc and set(bc) <= set(g["codes"]):
                split, src = {f"{st}:{c}": float(v) for c, v in bc.items()}, "s6a breakdown_by_code"
            else:   # fallback: group time split over element keys in proportion to the human time they replace
                hum = {k: rows[k]["time_s"] for k in gkeys}
                H = sum(hum.values()) or 1.0
                split, src = {k: T * hum[k] / H for k in gkeys}, "s6a group time split by human time (fallback)"
            groups[g["group_id"]]["models"].append(e["model"])
            for k in allowed:
                options[k].append({"model": e["model"], "group_id": g["group_id"],
                                   "category": g.get("category", ""), "time_s": float(split.get(k, 0.0)),
                                   "group_cycle_s": T, "time_source": src, "mounting": mount,
                                   "price_mid_usd": _price_mid(robots[e["model"]]), "topsis": e.get("topsis")})
    for k, v in rows.items():
        t = tasks.get(k, {})
        v["s3_allocation"] = t.get("allocation", "human")
        v["s3_category"] = t.get("category")
        v["s3_total"] = t.get("total")
        v["robot_options"] = options.get(k, [])
    return {"stations": stations, "rows": rows, "precedence_within": within, "precedence_chain": chain,
            "groups": groups, "robots": robots, "notes": notes, "takt": takt, "station_info": stinfo,
            "n_cycles": {s: len(ncyc[s]) for s in stations}}


def element_order(data, st):
    """Observed order of element keys in a station (by mean first start)."""
    ks = [k for k, v in data["rows"].items() if v["station"] == st]
    return sorted(ks, key=lambda k: data["rows"][k]["first_start_mean_s"])


def capex_parts(model_price, categories, integration_mult=A.INTEGRATION_MULT):
    base = model_price + A.SAFETY_USD + integration_mult * model_price
    eoat = sum(A.eoat_cost(c) for c in categories)
    return base, eoat
