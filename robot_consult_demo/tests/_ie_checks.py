"""Industrial-engineering method checks. Imported by validate.py.

These force each stage to apply named IE methods and re-derive the textbook formulas from the outputs.
"""
from __future__ import annotations

import math

import numpy as np


def s2_ie(ts: dict, chk) -> None:
    st = "s2"
    # Performance rating (Westinghouse: skill, effort, conditions, consistency) and ILO allowance table
    r = ts.get("rating", {})
    chk(st, r.get("method", "").lower().startswith("westinghouse") and len(r.get("factors", {})) == 4,
        "performance rating by Westinghouse system (4 factors)")
    al = ts.get("allowance_breakdown", {})
    tot = sum(al.values()) if al else -1
    chk(st, al and abs(tot - ts["params"]["allowance_pfd"]) < 1e-6,
        f"ILO allowance breakdown sums to PFD allowance ({tot:.3f})")
    chk(st, al and all(v >= 0 for v in al.values()),
        "no negative/balancing items in the allowance table (allowance must be derived, not forced)")
    for ss in ts["station_stats"]:
        chk(st, ss.get("overload_basis") in ("measured", "assumption"),
            f"{ss['station']}: overload basis declared ({ss.get('overload_basis')}) given observed fraction {ss.get('observed_fraction')}")
    # Predetermined motion time system comparison (MTM-UAS or MOST) vs observed
    pm = ts.get("pmts", {})
    chk(st, pm.get("system") in ("MTM-UAS", "MTM-1", "MOST", "BasicMOST") and len(pm.get("rows", [])) >= 4,
        f"PMTS ({pm.get('system')}) estimates for >=4 element types")
    tmu_bad = sum(abs(row["tmu"] * 0.036 - row["pmts_s"]) > 0.01 * max(1, row["pmts_s"]) for row in pm.get("rows", []))
    chk(st, tmu_bad == 0, f"PMTS TMU -> seconds conversion (1 TMU = 0.036 s) re-derived: {tmu_bad} mismatches")
    # SPC on station cycle times + capability against takt (upper spec)
    for sp in ts.get("spc", []):
        x = np.asarray(sp["observations_s"], float)
        if len(x) < 2:
            continue
        mr = np.abs(np.diff(x)).mean()
        ucl = x.mean() + 2.66 * mr
        chk(st, abs(ucl - sp["ucl_s"]) < 1e-3 * max(1, ucl), f"{sp['station']}: I-MR chart UCL re-derived ({ucl:.1f}s)")
        sigma = mr / 1.128
        cpu = (sp["usl_s"] - x.mean()) / (3 * sigma) if sigma > 0 else float("inf")
        chk(st, (math.isinf(cpu) and sp["cpu"] is None) or abs(cpu - (sp["cpu"] or 0)) < 1e-3 * max(1, abs(cpu)),
            f"{sp['station']}: Cpu vs takt re-derived ({cpu:.2f})")
    want = {ss["station"] for ss in ts["station_stats"]}
    have = {sp["station"] for sp in ts.get("spc", [])}
    chk(st, want and want <= have, f"SPC (I-MR) computed for every station ({len(have & want)}/{len(want)})")
    # Lean: VA / NVA / NNVA per element type and VA ratio
    ln = ts.get("lean", {})
    cls = {row["class"] for row in ln.get("rows", [])}
    chk(st, cls <= {"VA", "NNVA", "NVA"} and len(cls) >= 2, f"lean value classification uses VA/NNVA/NVA ({sorted(cls)})")
    chk(st, all(row.get("waste") is not None for row in ln.get("rows", []) if row["class"] == "NVA"),
        "every NVA element is mapped to one of the 7 wastes (TIMWOOD)")
    chk(st, 0 <= ln.get("va_ratio", -1) <= 1, f"VA ratio reported ({ln.get('va_ratio')})")


def s4_ie(cand: dict, chk) -> None:
    st = "s4"
    rk = cand.get("ranking", {})
    chk(st, rk.get("method") in ("AHP+TOPSIS", "TOPSIS", "AHP"), f"robot ranking by MCDM ({rk.get('method')})")
    if "AHP" in rk.get("method", ""):
        chk(st, rk.get("consistency_ratio", 1) < 0.10, f"AHP consistency ratio {rk.get('consistency_ratio')} (<0.10)")
    # Cell space: ISO 13855 minimum separation S = K*T + C, then footprint + aisle allowance (Facilities Planning)
    bad = 0
    n = 0
    for g in cand["task_groups"]:
        for e in g["evaluated"]:
            cs = e.get("cell_space")
            if not e["passed"]:
                continue
            n += 1
            if not cs:
                bad += 1
                continue
            S = cs["K_mm_s"] * cs["T_s"] + cs["C_mm"]
            bad += abs(S - cs["separation_mm"]) > 1
            area = cs["footprint_m2"] * (1 + cs["aisle_allowance"])
            bad += abs(area - cs["total_m2"]) > 1e-3
    chk(st, n > 0 and bad == 0, f"cell space (ISO 13855 S=K*T+C, aisle allowance) re-derived for {n} passed models: {bad} errors")


def s5_ie(al: dict, takt: float, chk) -> None:
    st = "s5"
    lb = al.get("line_balancing", {})
    hs = {h["name"] for h in lb.get("heuristics", [])}
    chk(st, {"RPW", "LCR"} <= hs, f"line-balancing heuristics run: {sorted(hs)} (need RPW, LCR)")
    for h in lb.get("heuristics", []) + ([lb["exact"]] if "exact" in lb else []):
        t = np.asarray(h["station_loads_s"], float)
        c = h["cycle_time_s"]
        E = t.sum() / (len(t) * c)
        SI = math.sqrt(((t.max() - t) ** 2).sum())
        chk(st, abs(E - h["balance_efficiency"]) < 1e-3 and abs((1 - E) - h["balance_delay"]) < 1e-3,
            f"{h['name']}: balance efficiency {E:.3f} / delay re-derived")
        chk(st, abs(SI - h["smoothness_index"]) < 1e-3 * max(1, SI), f"{h['name']}: smoothness index {SI:.1f} re-derived")
        chk(st, t.max() <= c + 1e-6, f"{h['name']}: no station exceeds cycle time")
    if "exact" in lb and lb.get("heuristics"):
        best_h = min(len(h["station_loads_s"]) for h in lb["heuristics"])
        chk(st, len(lb["exact"]["station_loads_s"]) <= best_h, "exact MILP needs no more stations than the best heuristic")
    tm = lb.get("theoretical_min_stations")
    if tm is not None and "exact" in lb:
        chk(st, tm == math.ceil(sum(lb["exact"]["station_loads_s"]) / lb["exact"]["cycle_time_s"] - 1e-9),
            f"theoretical minimum stations ceil(sum t / c) = {tm}")
    wm = al.get("worker_machine", [])
    chk(st, len(wm) >= 1, "worker-machine (human + cobot) chart analysis present")
    for w in wm:
        cyc = max(w["human_s"], w["robot_s"]) + w.get("interaction_s", 0)
        chk(st, abs(cyc - w["cycle_s"]) < 1e-3, f"{w['station']}: man-machine cycle re-derived ({cyc:.1f}s)")
        chk(st, abs(w["human_idle_pct"] - (cyc - w["human_s"] - w.get("interaction_s", 0)) / cyc) < 1e-3,
            f"{w['station']}: human idle % re-derived")


def s6b_ie(des: dict, chk) -> None:
    st = "s6b"
    for s in des["scenarios"]:
        ll = s.get("littles_law")
        if not ll:
            chk(st, False, f"{s['name']}: Little's law check missing")
            continue
        wip = ll["throughput_per_s"] * ll["flow_time_s"]
        chk(st, abs(wip - ll["avg_wip"]) <= 0.05 * max(1e-9, ll["avg_wip"]),
            f"{s['name']}: Little's law WIP = TH x CT holds ({wip:.2f} vs {ll['avg_wip']:.2f})")


def s7_ie(roi: dict, chk) -> None:
    st = "s7"
    for s in roi["scenarios"]:
        if "crf" not in s:
            chk(st, False, f"{s['name']}: capital recovery factor (A/P) missing")
            continue
        i, n = s["crf"]["i"], s["crf"]["n"]
        crf = i * (1 + i) ** n / ((1 + i) ** n - 1)
        chk(st, abs(crf - s["crf"]["value"]) < 1e-6, f"{s['name']}: A/P = {crf:.5f} re-derived")
        chk(st, abs(s["capex_usd"] * crf - s["annualized_capex_usd"]) < 0.5, f"{s['name']}: annualized capex = capex x A/P")
