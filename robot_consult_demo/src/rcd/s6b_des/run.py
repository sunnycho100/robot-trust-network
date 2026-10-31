"""s6b - discrete-event simulation (SimPy) of the paced car-door line for every s5 allocation scenario.

Model (ISyE 510 lecture06 line pacing):
  * paced (rigid pacing, moving conveyor with andon stop): every station works on one door; the conveyor indexes when
    the takt timer has elapsed AND every station has finished (an overrun stops the line until the late station is
    done). Index interval = max(takt, max_s T_s).
  * paced-with-margin variant for the recommended allocation: doors released every takt, stations decoupled by
    finite buffers (blocking after service), to show what a buffer would buy.
  Station time T_s = max(H_s, R_s) + I_s (worker-machine chart): H_s = sum of human element times, R_s = sum of robot
  element times, I_s = handover.

Stochastic times
  * human element (per-cycle total of an element_key): lognormal with the s5 time (standard time) as mean and the
    occurrence-level CV (station_specs), fitted by moments:
      sigma^2 = ln(1 + cv^2), mu = ln(mean) - sigma^2/2           (positive, right-skewed, mean preserved)
  * UNOBS block: k = UNOBS / dbar further occurrences like the observed ones, CV_UNOBS = cv_occ / sqrt(k).
  * relief operator (manual_relief): same lognormal model, working in parallel with the station operator.
  * robot group time: normal(mean = s6a cycle share, CV = 3 %) truncated at 0 (deterministic controller, small jitter).
Output analysis: R >= 30 independent replications, warm-up deleted, 95 % t-interval on throughput; Little's law
WIP = TH x W checked from time-averaged WIP, throughput and mean flow time measured in the same window.
"""
from __future__ import annotations

import math
import time
from collections import defaultdict

import numpy as np
import simpy
from scipy import stats

from rcd.common.io import out_dir, save_json
from rcd.s5_optimize.inputs import load_input
from rcd.s7_roi import assumptions as A

REPS = 30
RUN_H = 80.0          # simulated production hours per replication (one 2-shift week)
WARMUP_CYCLES = 10
SEED = 2024


def lognorm_params(m, s):
    if m <= 0:
        return None
    if s <= 0:
        return (math.log(m), 0.0)
    s2 = math.log(1 + (s / m) ** 2)
    return (math.log(m) - s2 / 2, math.sqrt(s2))


def station_specs(alloc, el, takt):
    """Per element_key coefficient of variation of its per-cycle total, from occurrence-level data.

    s2's std_s of per-cycle totals is dominated by clip cropping (clips cover different fractions of the cycle and
    observed + UNOBS is ~takt by construction), so it is not process variability. Instead, for each element_key with
    k occurrences per cycle of mean m_o and s.d. s_o:  CV_total = sqrt(k) s_o / (k m_o)  (sum of k independent
    occurrences). UNOBS is treated as k = UNOBS / dbar further occurrences like the station's observed ones:
    CV_UNOBS = cv_occ / sqrt(k), cv_occ = pooled per-occurrence CV, dbar = mean occurrence duration.
    """
    rows = {e["element_key"]: e for e in alloc["elements"]}
    occ = defaultdict(lambda: defaultdict(list))
    ncyc = defaultdict(set)
    for c in el["cycles"]:
        ncyc[c["station"]].add(c["cycle_id"])
        for e in c["elements"]:
            occ[c["station"]][e["code"]].append(float(e["duration_s"]))
    cv, unobs_cv = {}, {}
    for st, d in occ.items():
        rel, durs = [], []
        nc = max(1, len(ncyc[st]))
        for code, xs in d.items():
            m = float(np.mean(xs))
            s_o = float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0
            k = len(xs) / nc
            cv[f"{st}:{code}"] = (math.sqrt(k) * s_o / (k * m)) if m > 0 and k > 0 else 0.0
            if code != "IDLE":
                durs += xs
                rel += [(x - m) / m for x in xs if m > 0]
        cv_occ = float(np.sqrt(np.mean(np.square(rel)))) if rel else 0.2
        dbar = float(np.mean(durs)) if durs else 5.0
        u = rows.get(f"{st}:UNOBS", {}).get("time_s", 0.0)
        k = max(1.0, u / dbar) if u > 0 else 1.0
        unobs_cv[st] = {"cv_occ": cv_occ, "dbar_s": dbar, "k": k, "cv": cv_occ / math.sqrt(k)}
        cv[f"{st}:UNOBS"] = unobs_cv[st]["cv"]
    return rows, cv, unobs_cv


class Line:
    def __init__(self, scen, rows, cv, takt, rng, stations):
        self.takt, self.rng, self.stations = takt, rng, stations
        self.h, self.hp, self.r, self.i = defaultdict(list), defaultdict(list), defaultdict(float), {}
        for a in scen["assignment"]:
            st = a["station"]
            if a["resource"] in ("human", "helper"):
                p = lognorm_params(a["time_s"], cv.get(a["element_key"], 0.0) * a["time_s"])
                if p:
                    (self.h if a["resource"] == "human" else self.hp)[st].append(p)
            else:
                self.r[st] += a["time_s"]
        for st, v in scen["station_times"].items():
            self.i[st] = v["interaction_s"]

    def _ln(self, ps):
        return sum(self.rng.lognormal(mu, sg) if sg > 0 else math.exp(mu) for mu, sg in ps)

    def sample(self, st):
        H = self._ln(self.h[st])
        R = max(0.0, self.rng.normal(self.r[st], A.ROBOT_TIME_CV * self.r[st])) if self.r[st] > 0 else 0.0
        R += self._ln(self.hp[st])      # relief operator works in parallel like the robot (second resource)
        return H, R, self.i[st]


def simulate_paced(line, horizon, warm):
    env = simpy.Environment()
    st_list = line.stations
    n = len(st_list)
    busy = defaultdict(float)
    slots = [None] * n          # door id at each station
    entry, exits = {}, []
    wip = {"n": 0, "area": 0.0, "last": 0.0}
    viol = {"station_cycles": 0, "over": 0, "index": 0, "stops": 0}
    counter = [0]

    def wip_change(dn):
        if env.now > warm:
            wip["area"] += wip["n"] * (env.now - max(wip["last"], warm))
        wip["last"] = env.now
        wip["n"] += dn

    def work(k, st):
        H, R, I = line.sample(st)
        T = max(H, R) + I
        if env.now >= warm:
            busy[f"{st}.human"] += H + I
            if R > 0:
                busy[f"{st}.robot"] += R + I
            viol["station_cycles"] += 1
            viol["over"] += T > line.takt + 1e-9
        yield env.timeout(T)

    def conveyor():
        while True:
            # load a new door into the first station
            counter[0] += 1
            slots[0] = counter[0]
            entry[counter[0]] = env.now
            wip_change(+1)
            t0 = env.now
            procs = [env.process(work(k, st_list[k])) for k in range(n) if slots[k] is not None]
            yield simpy.AllOf(env, procs) & env.timeout(line.takt)
            if env.now >= warm:
                viol["index"] += 1
                viol["stops"] += (env.now - t0) > line.takt + 1e-9
            # index: last station's door leaves, others move one station
            if slots[-1] is not None:
                d = slots[-1]
                wip_change(-1)
                if env.now >= warm:
                    exits.append(env.now - entry[d])
            for k in range(n - 1, 0, -1):
                slots[k] = slots[k - 1]
            slots[0] = None

    env.process(conveyor())
    env.run(until=horizon)
    wip_change(0)
    T = horizon - warm
    th = len(exits) / T
    return {"throughput_per_s": th, "flow_time_s": float(np.mean(exits)) if exits else 0.0,
            "avg_wip": wip["area"] / T, "util": {k: v / T for k, v in busy.items()},
            "takt_violation_frac": viol["over"] / max(1, viol["station_cycles"]),
            "line_stop_frac": viol["stops"] / max(1, viol["index"])}


def simulate_buffered(line, horizon, warm, buf=1):
    """Pacing with margin: release every takt, finite buffers between stations, blocking after service."""
    env = simpy.Environment()
    st_list = line.stations
    S = [simpy.Resource(env, 1) for _ in st_list]
    B = [simpy.Resource(env, buf) for _ in st_list[:-1]]
    busy = defaultdict(float)
    exits = []
    wip = {"n": 0, "area": 0.0, "last": 0.0}
    viol = {"c": 0, "o": 0}

    def wip_change(dn):
        if env.now > warm:
            wip["area"] += wip["n"] * (env.now - max(wip["last"], warm))
        wip["last"] = env.now
        wip["n"] += dn

    def door():
        t_in = env.now
        wip_change(+1)
        hold_b = None
        req = S[0].request()
        yield req
        for k, st in enumerate(st_list):
            if hold_b is not None:
                B[k - 1].release(hold_b)
                hold_b = None
            H, R, I = line.sample(st)
            T = max(H, R) + I
            if env.now >= warm:
                busy[f"{st}.human"] += H + I
                if R > 0:
                    busy[f"{st}.robot"] += R + I
                viol["c"] += 1
                viol["o"] += T > line.takt + 1e-9
            yield env.timeout(T)
            if k + 1 < len(st_list):
                hb = B[k].request()
                yield hb
                S[k].release(req)
                req = S[k + 1].request()
                yield req
                hold_b = hb
            else:
                S[k].release(req)
        wip_change(-1)
        if env.now >= warm:
            exits.append(env.now - t_in)

    def source():
        while True:
            env.process(door())
            yield env.timeout(line.takt)

    env.process(source())
    env.run(until=horizon)
    wip_change(0)
    T = horizon - warm
    th = len(exits) / T
    return {"throughput_per_s": th, "flow_time_s": float(np.mean(exits)) if exits else 0.0,
            "avg_wip": wip["area"] / T, "util": {k: min(1.0, v / T) for k, v in busy.items()},
            "takt_violation_frac": viol["o"] / max(1, viol["c"]), "line_stop_frac": None}


def replicate(name, scen, rows, cv, takt, stations, mode, seed):
    horizon = RUN_H * 3600.0
    warm = WARMUP_CYCLES * takt
    reps = []
    for r in range(REPS):
        rng = np.random.default_rng([seed, r])
        line = Line(scen, rows, cv, takt, rng, stations)
        reps.append(simulate_paced(line, horizon, warm) if mode == "paced" else
                    simulate_buffered(line, horizon, warm))
    th = np.array([x["throughput_per_s"] * 3600 for x in reps])
    hw = stats.t.ppf(0.975, REPS - 1) * th.std(ddof=1) / math.sqrt(REPS)
    # Little's law on the pooled replications (each replication measured over the same window)
    TH = float(np.mean([x["throughput_per_s"] for x in reps]))
    W = float(np.mean([x["flow_time_s"] for x in reps]))
    L = float(np.mean([x["avg_wip"] for x in reps]))
    util = defaultdict(list)
    for x in reps:
        for k, v in x["util"].items():
            util[k].append(v)
    ls = [x["line_stop_frac"] for x in reps if x["line_stop_frac"] is not None]
    return {"name": name, "mode": mode, "source_scenario": scen["name"], "recommended": bool(scen.get("recommended")),
            "throughput_per_hr": float(th.mean()), "ci95_half_width": float(hw),
            "ci95_rel": float(hw / th.mean()) if th.mean() else None,
            "throughput_reps_per_hr": th.tolist(),
            "utilization": {k: float(np.mean(v)) for k, v in sorted(util.items())},
            "takt_violations_pct": 100.0 * float(np.mean([x["takt_violation_frac"] for x in reps])),
            "line_stop_pct": 100.0 * float(np.mean(ls)) if ls else None,
            "littles_law": {"throughput_per_s": TH, "flow_time_s": W, "avg_wip": L,
                            "th_x_w": TH * W, "rel_error": abs(TH * W - L) / L if L else None},
            "analytic_throughput_per_hr": 3600.0 / max(takt, max(v["cycle_s"] for v in scen["station_times"].values())),
            "station_mean_cycle_s": {k: v["cycle_s"] for k, v in scen["station_times"].items()}}


def run(cfg: dict) -> None:
    t0 = time.time()
    prov = {}
    alloc = load_input(cfg, "allocation", prov)
    el = load_input(cfg, "elements", prov)
    takt = float(cfg["line"]["takt_s"])
    stations = [s for s in cfg["data"]["stations"] if any(a["station"] == s for a in alloc["scenarios"][0]["assignment"])]
    rows, cv, unobs_cv = station_specs(alloc, el, takt)
    out_s = []
    for k, s in enumerate(alloc["scenarios"]):
        out_s.append(replicate(s["name"], s, rows, cv, takt, stations, "paced", SEED + k))
    rec = next(s for s in alloc["scenarios"] if s.get("recommended"))
    b = replicate(f"{rec['name']}+buffer1", rec, rows, cv, takt, stations, "buffered", SEED + 99)
    b["recommended"] = False
    b["note"] = "analysis: pacing with margin (1-door buffers, release every takt), same allocation as recommended"
    out_s.append(b)
    base = next(s for s in out_s if s["name"] == "baseline_manual")
    bott = max(v["cycle_s"] for v in next(s for s in alloc["scenarios"] if s["name"] == "baseline_manual")["station_times"].values())
    res = {
        "tool": f"SimPy {simpy.__version__}",
        "model": __doc__,
        "inputs": prov,
        "replications": REPS, "run_length_h": RUN_H, "warmup_s": WARMUP_CYCLES * takt, "seed": SEED,
        "takt_s": takt, "stations": stations,
        "distributions": {"human_element": "lognormal, mean = s5 time (standard time), CV from occurrence-level "
                                           "variability (see station_specs docstring)",
                          "cv_per_element": cv,
                          "unobs": {st: v for st, v in unobs_cv.items()},
                          "robot": f"normal, CV {A.ROBOT_TIME_CV:.0%} (assumption), truncated at 0",
                          "handover": "deterministic (s5 interaction_s)"},
        "scenarios": out_s,
        "validation": {"des_baseline_throughput": base["throughput_per_hr"],
                       "analytic_baseline_throughput": 3600.0 / max(takt, bott),
                       "formula": "3600 / max(takt, bottleneck mean station time)",
                       "rel_error": abs(base["throughput_per_hr"] - 3600.0 / max(takt, bott)) / (3600.0 / max(takt, bott)),
                       "note": "the DES is lower when overruns stop the paced line (E[max] > max E)"},
        "runtime_s": round(time.time() - t0, 2),
    }
    save_json(res, out_dir(cfg, "s6b") / "des_results.json", contract="des_results")
    print("  s6b: " + "; ".join(f"{s['name']} {s['throughput_per_hr']:.2f}/h +-{s['ci95_half_width']:.2f} "
                                  f"viol {s['takt_violations_pct']:.1f}%" for s in out_s) + f"; inputs={prov}")
