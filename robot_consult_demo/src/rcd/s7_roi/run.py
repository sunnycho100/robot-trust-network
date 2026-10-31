"""s7 - engineering economics of each s5 allocation scenario (vs baseline_manual).

  capex            = sum over robot cells (robot price midpoint + EOAT/feeder + safety + integration multiple x price)
                     + layout levers (kit racks / fixtures per station, shared AMR)
  A/P (CRF)        = i(1+i)^n / ((1+i)^n - 1)          ISyE 510 lecture03 / Exam-1 study guide sec. 4
  annualized capex = capex x A/P
  annual savings   = labour hours freed x loaded wage - relief-operator headcount cost - maintenance
                     (net annual cash flow vs the as-is manual line, all at standard time)
                     labour hours freed = (H_asis - (H_operator + H_relief)) [s/cycle] x cycles/yr / 3600
  vs_manual_relief = the same, incremental to the cheapest all-manual way to meet takt (relief operator)
  payback          = capex / annual savings (simple, undiscounted)
  NPV              = -capex + annual savings x (P/A, i, n);  IRR: NPV(IRR) = 0;  no salvage value
  tornado          = NPV of the recommended scenario at low/high of each uncertain parameter (one at a time)
"""
from __future__ import annotations

import time

from rcd.common.io import out_dir, save_json
from rcd.s5_optimize.inputs import load_input
from rcd.s7_roi import assumptions as A


def econ(cfg, scen, base, robots_cat, p):
    """p: parameter overrides {wage, price_pos (0..1 within range), integ, maint, i, n, shifts}."""
    i, n = p["i"], int(p["n"])
    cycles = A.annual_cycles(cfg, p["shifts"])
    loaded = p["wage"] * cfg["economics"]["labor_fringe_multiplier"]
    capex, maint_base, cells = 0.0, 0.0, []
    for r in scen["robots"]:
        lo, hi = r["price_usd_range"][0], r["price_usd_range"][-1]
        price = lo + p["price_pos"] * (hi - lo)
        eoat = sum(A.eoat_cost(c) for c in r["categories"])
        bd = r.get("capex_breakdown_usd", {})
        track = p["track"] if bd.get("linear_track") else 0.0
        changer = p["changer"] if bd.get("tool_changer") else 0.0
        cx = price + eoat + A.SAFETY_USD + p["integ"] * price + track + changer
        capex += cx
        maint_base += price + eoat + track + changer
        cells.append({"model": r["model"], "station": r["station"], "robot_usd": price, "eoat_usd": eoat,
                      "safety_usd": A.SAFETY_USD, "integration_usd": p["integ"] * price, "linear_track_usd": track,
                      "tool_changer_usd": changer, "capex_usd": cx})
    for lv in scen.get("layout_changes", []):
        capex += lv["capex_usd"]
        maint_base += lv["capex_usd"]
        cells.append({"model": f"layout:{lv['lever']}", "station": lv["station"], "robot_usd": 0.0, "eoat_usd": 0.0,
                      "safety_usd": 0.0, "integration_usd": 0.0, "capex_usd": lv["capex_usd"]})
    ob, o = base["objective"], scen["objective"]
    dh = ob["human_s_per_cycle"] - (o["human_s_per_cycle"] + o.get("helper_s_per_cycle", 0.0))
    hours = dh * cycles / 3600.0
    gross = hours * loaded
    helpers = o.get("n_helpers_per_shift", 0) or 0
    helper_cost = helpers * A.annual_hours(cfg, p["shifts"]) * loaded
    maint = p["maint"] * maint_base
    net = gross - helper_cost - maint
    crf = A.crf(i, n)
    npv = A.npv(capex, net, i, n)
    return {"capex_usd": capex, "cells": cells, "crf": {"i": i, "n": n, "value": crf},
            "annualized_capex_usd": capex * crf, "labour_hours_freed_per_yr": hours,
            "human_s_saved_per_cycle": dh, "gross_labour_savings_usd": gross, "maintenance_usd": maint,
            "relief_operators_per_shift": helpers, "relief_operator_cost_usd": helper_cost,
            "annual_savings_usd": net,
            "payback_years": (capex / net) if net > 0 else None,
            "npv_usd": npv, "irr": A.irr(capex, net, n),
            "equivalent_annual_net_benefit_usd": net - capex * crf,
            "fte_freed": hours / (cfg["line"]["hours_per_shift"] * cfg["line"]["days_per_year"])}


def run(cfg: dict) -> None:
    t0 = time.time()
    prov = {}
    al = load_input(cfg, "allocation", prov)
    des = load_input(cfg, "des_results", prov, required=False)
    cat = load_input(cfg, "robot_catalog", prov)
    ec = cfg["economics"]
    P0 = {"wage": A.hourly_wage(), "price_pos": 0.5, "integ": A.INTEGRATION_MULT, "maint": A.MAINT_PCT,
          "i": ec["discount_rate"], "n": ec["horizon_years"], "shifts": cfg["line"]["shifts_per_day"],
          "track": A.TRACK_USD, "changer": A.TOOL_CHANGER_USD}
    # annual savings are measured against the as-is manual line (baseline_manual) at standard time
    base = next(s for s in al["scenarios"] if s["name"] == "baseline_manual")
    des_by = {s["name"]: s for s in (des or {}).get("scenarios", [])}
    scen_out = []
    for s in al["scenarios"]:
        if s["name"] == "baseline_manual":
            continue
        e = econ(cfg, s, base, cat, P0)
        e.update({"name": s["name"], "recommended": bool(s.get("recommended")), "n_robots": len(s["robots"]),
                  "robots": [f"{r['model']}@{r['station']}" for r in s["robots"]],
                  "layout_levers": [f"{l['lever']}@{l['station']}" for l in s.get("layout_changes", [])],
                  "awkward_posture_s_per_cycle": s["objective"]["awkward_posture_s_per_cycle"],
                  "awkward_posture_reduction_s": base["objective"]["awkward_posture_s_per_cycle"]
                  - s["objective"]["awkward_posture_s_per_cycle"]})
        d, db = des_by.get(s["name"]), des_by.get("baseline_manual")
        if d and db:
            e["des_throughput_gain_per_hr"] = d["throughput_per_hr"] - db["throughput_per_hr"]
            e["des_extra_units_per_yr"] = e["des_throughput_gain_per_hr"] * A.annual_hours(cfg)
            e["des_takt_violation_pct"] = d["takt_violations_pct"]
        scen_out.append(e)

    relief = next((s for s in scen_out if s["name"] == "manual_relief"), None)
    if relief:
        for s in scen_out:
            if s is relief:
                continue
            d = s["annual_savings_usd"] - relief["annual_savings_usd"]
            s["vs_manual_relief"] = {
                "annual_savings_usd": d, "npv_usd": A.npv(s["capex_usd"], d, P0["i"], int(P0["n"])),
                "payback_years": s["capex_usd"] / d if d > 0 else None, "irr": A.irr(s["capex_usd"], d, int(P0["n"])),
                "note": "incremental to the cheapest all-manual way of meeting takt at standard time "
                        "(relief operator); relevant only if the as-is overload must be removed"}
    rec_s = next(s for s in al["scenarios"] if s.get("recommended"))
    sens = []
    if rec_s["name"] != "baseline_manual":
        wl, wh = 0.8 * A.hourly_wage(), A.WAGE_ANNUAL_TRANSPORT_EQUIP / A.BLS_HOURS_PER_YEAR
        grid = [("hourly_wage_usd", wl, wh, "wage", "low: -20 % of BLS median; high: BLS OOH median for "
                 "transportation equipment manufacturing ($50,790/2,080 h)"),
                ("robot_price_usd", 0.0, 1.0, "price_pos", "low/high end of s4 price_usd_range of every recommended "
                 "robot (values = total robot list price)"),
                ("linear_track_usd", A.TRACK_USD_RANGE[0], A.TRACK_USD_RANGE[1], "track",
                 "7th-axis track per robot: track only (Vention) .. 3 m overhead track + drive/sync"),
                ("tool_changer_usd", A.TOOL_CHANGER_USD_RANGE[0], A.TOOL_CHANGER_USD_RANGE[1], "changer",
                 "automatic tool changer + stand"),
                ("integration_multiple", A.INTEGRATION_MULT_RANGE[0], A.INTEGRATION_MULT_RANGE[1], "integ",
                 "0.5x - 2.0x robot price (2x = NIST lower bound for industrial robots)"),
                ("maintenance_pct", A.MAINT_PCT_RANGE[0], A.MAINT_PCT_RANGE[1], "maint", "2 % - 12 % per year"),
                ("discount_rate", 0.05, 0.12, "i", "MARR 5 % - 12 %"),
                ("shifts_per_day", 1, 3, "shifts", "1 - 3 shifts"),
                ("horizon_years", 3, 7, "n", "3 - 7 year life")]
        for name, lo, hi, key, note in grid:
            pl, ph = dict(P0, **{key: lo}), dict(P0, **{key: hi})
            nl = econ(cfg, rec_s, base, cat, pl)["npv_usd"]
            nh = econ(cfg, rec_s, base, cat, ph)["npv_usd"]
            if key == "price_pos":   # express the price swing in dollars (sum over the recommended robots)
                rs = rec_s["robots"]
                tot = lambda q: sum(r["price_usd_range"][0] + q * (r["price_usd_range"][-1] - r["price_usd_range"][0])  # noqa: E731
                                    for r in rs)
                lo_v, hi_v, base_v = tot(0.0), tot(1.0), tot(0.5)
            else:
                lo_v, hi_v, base_v = lo, hi, P0[key]
            if key in ("track", "changer") and abs(nh - nl) < 1e-6:
                continue   # not used by the recommended scenario
            sens.append({"param": name, "base": base_v, "low": lo_v, "high": hi_v, "npv_low": nl, "npv_high": nh,
                         "swing_usd": abs(nh - nl), "note": note})
        sens.sort(key=lambda r: -r["swing_usd"])
    rec = next((s for s in scen_out if s["recommended"]), None)
    # break-even: savings per cycle needed for NPV = 0 at base parameters
    be = None
    if rec:
        pa = 1.0 / A.crf(P0["i"], int(P0["n"]))
        need_net = rec["capex_usd"] / pa
        need_gross = need_net + rec["maintenance_usd"]
        cyc = A.annual_cycles(cfg)
        be = {"required_annual_savings_usd": need_net,
              "required_human_s_saved_per_cycle": need_gross / (A.loaded_wage(cfg) * cyc / 3600.0),
              "actual_human_s_saved_per_cycle": rec["human_s_saved_per_cycle"],
              "breakeven_hourly_wage_usd": (need_gross / (rec["labour_hours_freed_per_yr"] *
                                            ec["labor_fringe_multiplier"])) if rec["labour_hours_freed_per_yr"] > 0 else None}
        hc = al.get("line_balancing", {}).get("post_automation", {})
        nst = len(base["station_times"])
        if hc:
            freed_ops = max(0, nst - hc.get("exact_operators", nst))
            val = freed_ops * cfg["line"]["shifts_per_day"] * cfg["line"]["hours_per_shift"] \
                * cfg["line"]["days_per_year"] * A.loaded_wage(cfg)
            net_hc = val - rec["maintenance_usd"] - rec.get("relief_operator_cost_usd", 0.0)
            be["headcount_view"] = {
                "stations": nst, "operators_after_rebalance": hc.get("exact_operators"),
                "operators_freed_per_shift": freed_ops, "annual_value_usd": val,
                "annual_net_savings_usd": net_hc,
                "npv_usd": A.npv(rec["capex_usd"], net_hc, P0["i"], int(P0["n"])),
                "payback_years": rec["capex_usd"] / net_hc if net_hc > 0 else None,
                "irr": A.irr(rec["capex_usd"], net_hc, int(P0["n"])),
                "fte_freed_fractional_per_shift": rec["human_s_saved_per_cycle"] / cfg["line"]["takt_s"],
                "note": "Integer-headcount view: only whole operators removed after re-balancing the remaining human "
                        "work (s5 line_balancing.post_automation) count as cash savings. Requires operators to "
                        "work across station boundaries (re-balanced line)."}
    out = {"method": __doc__, "inputs": prov, "currency": "USD",
           "assumptions": A.assumption_rows(cfg),
           "baseline": {"human_s_per_cycle": base["objective"]["human_s_per_cycle"],
                        "annual_labour_usd": base["objective"]["labour_usd"]},
           "scenarios": scen_out, "recommended": rec["name"] if rec else None,
           "sensitivity": sens, "sensitivity_scenario": rec_s["name"], "breakeven": be,
           "not_monetised": ["ergonomic / MSD injury cost avoided (no defensible per-second risk model)",
                             "throughput gain from fewer takt overruns (s6b) - margin per door unknown",
                             "quality (torque traceability) benefits of automated screwdriving"],
           "runtime_s": round(time.time() - t0, 2)}
    save_json(out, out_dir(cfg, "s7") / "roi.json", contract="roi")
    print("  s7: " + "; ".join(f"{s['name']} capex {s['capex_usd']:,.0f} sav {s['annual_savings_usd']:,.0f} "
                                 f"NPV {s['npv_usd']:,.0f}" for s in scen_out))
