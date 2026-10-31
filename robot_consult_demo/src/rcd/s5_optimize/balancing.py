"""Single-model assembly line balancing (SALBP-1: minimise stations for a given cycle time c).

Heuristics as taught in ISyE 510 lecture06 ("Assembly line balancing"):
  * RPW  - Ranked Positional Weight (Helgeson & Birnie 1961): PW_i = t_i + sum of all (transitive) successors' times
  * LCR  - Largest Candidate Rule: tasks in order of decreasing t_i
  * COMSOAL (Arcus 1966): random feasible-candidate sequences, best of N trials
and an exact SALBP-1 model (CP-SAT, binary station assignment; cf. the MILP families reviewed by
Weckenborg et al. 2020, Business Research 13:543-586, https://link.springer.com/article/10.1007/s40685-019-0101-y).

Assignment rule (lecture06): open a station; repeatedly assign the highest-priority task whose predecessors are all
assigned and whose time fits the remaining station time (c - load); when none fits, open the next station.
Metrics: balance efficiency E = sum(t) / (n * c), balance delay d = 1 - E, smoothness index SI = sqrt(sum (Smax-Si)^2),
theoretical minimum stations N_t = ceil(sum(t) / c).
"""
from __future__ import annotations

import math
import random
from collections import defaultdict

from ortools.sat.python import cp_model

EPS = 1e-9
SCALE = 10000  # 0.1 ms units for the integer model (rounding slack of 1 unit per task, verified on floats)


def _preds(tasks, edges):
    p = defaultdict(set)
    for a, b in edges:
        if a in tasks and b in tasks:
            p[b].add(a)
    return p


def _all_successors(tasks, edges):
    s = defaultdict(set)
    for a, b in edges:
        if a in tasks and b in tasks:
            s[a].add(b)
    memo = {}

    def go(x):
        if x in memo:
            return memo[x]
        out = set()
        for y in s[x]:
            out |= {y} | go(y)
        memo[x] = out
        return out
    return {t: go(t) for t in tasks}


def metrics(name, loads, c, stations_tasks, extra=None):
    n = len(loads)
    tot = sum(loads)
    E = tot / (n * c)
    smax = max(loads)
    d = {"name": name, "cycle_time_s": c, "n_stations": n, "station_loads_s": [float(x) for x in loads],
         "balance_efficiency": E, "balance_delay": 1 - E,
         "smoothness_index": math.sqrt(sum((smax - x) ** 2 for x in loads)),
         "idle_time_s": n * c - tot, "stations": stations_tasks}
    if extra:
        d.update(extra)
    return d


def _assign(order_fn, tasks, t, edges, c):
    """Generic lecture06 assignment loop. order_fn(candidates, remaining) -> chosen task or None."""
    pred = _preds(tasks, edges)
    done, stations, loads = set(), [], []
    cur, load = [], 0.0
    while len(done) < len(tasks):
        cand = [x for x in tasks if x not in done and pred[x] <= done and load + t[x] <= c + EPS]
        x = order_fn(cand) if cand else None
        if x is None:
            if not cur:   # a single task longer than c -> infeasible
                raise ValueError("task longer than cycle time")
            stations.append(cur)
            loads.append(load)
            cur, load = [], 0.0
            continue
        cur.append(x)
        load += t[x]
        done.add(x)
    if cur:
        stations.append(cur)
        loads.append(load)
    return stations, loads


def rpw(tasks, t, edges, c):
    succ = _all_successors(tasks, edges)
    pw = {x: t[x] + sum(t[y] for y in succ[x]) for x in tasks}
    rank = sorted(tasks, key=lambda x: (-pw[x], x))
    st, ld = _assign(lambda cand: min(cand, key=rank.index), tasks, t, edges, c)
    return metrics("RPW", ld, c, st, {"rule": "Ranked Positional Weight (Helgeson-Birnie)",
                                      "positional_weights_s": {x: pw[x] for x in rank}})


def lcr(tasks, t, edges, c):
    rank = sorted(tasks, key=lambda x: (-t[x], x))
    st, ld = _assign(lambda cand: min(cand, key=rank.index), tasks, t, edges, c)
    return metrics("LCR", ld, c, st, {"rule": "Largest Candidate Rule", "order": rank})


def comsoal(tasks, t, edges, c, trials=1000, seed=510):
    rng = random.Random(seed)
    best = None
    hist = defaultdict(int)
    for _ in range(trials):
        st, ld = _assign(lambda cand: rng.choice(sorted(cand)), tasks, t, edges, c)
        n = len(ld)
        hist[n] += 1
        smax = max(ld)
        si = math.sqrt(sum((smax - x) ** 2 for x in ld))
        if best is None or (n, si) < (best[0], best[1]):
            best = (n, si, st, ld)
    return metrics("COMSOAL", best[3], c, best[2],
                   {"rule": "COMSOAL (Arcus 1966), random feasible candidate, best of trials", "trials": trials,
                    "seed": seed, "stations_histogram": {str(k): v for k, v in sorted(hist.items())}})


def exact_salbp1(tasks, t, edges, c, ub, time_limit=20.0):
    """min sum_k u_k  s.t. each task in one station, sum_i ceil(t_i) x_ik <= C u_k, k(i) <= k(j) for (i,j) in P."""
    m = cp_model.CpModel()
    K = ub
    ti = {x: int(round(t[x] * SCALE)) for x in tasks}
    C = int(round(c * SCALE)) + len(tasks)
    x = {(i, k): m.NewBoolVar(f"x_{i}_{k}") for i in tasks for k in range(K)}
    u = [m.NewBoolVar(f"u_{k}") for k in range(K)]
    sidx = {i: m.NewIntVar(0, K - 1, f"s_{i}") for i in tasks}
    for i in tasks:
        m.AddExactlyOne(x[i, k] for k in range(K))
        m.Add(sidx[i] == sum(k * x[i, k] for k in range(K)))
    for k in range(K):
        m.Add(sum(ti[i] * x[i, k] for i in tasks) <= C * u[k])
        if k + 1 < K:
            m.Add(u[k] >= u[k + 1])
    for a, b in edges:
        if a in tasks and b in tasks:
            m.Add(sidx[a] <= sidx[b])
    lb = math.ceil(sum(t.values()) / c - 1e-9)
    m.Add(sum(u) >= lb)
    m.Minimize(sum(u))
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = time_limit
    s.parameters.num_workers = 8
    stat = s.Solve(m)
    status = s.StatusName(stat)
    sol = {q: s.Value(v) for q, v in x.items()} if stat in (cp_model.OPTIMAL, cp_model.FEASIBLE) else {}
    if stat == cp_model.OPTIMAL:
        # second stage: with the minimum number of stations fixed, minimise the largest station load (smoothest
        # optimal balance; SALBP-1 then SALBP-2 lexicographically)
        nopt = int(round(s.ObjectiveValue()))
        m.Add(sum(u) == nopt)
        cmax = m.NewIntVar(0, C, "cmax")
        for k in range(K):
            m.Add(sum(ti[i] * x[i, k] for i in tasks) <= cmax)
        m.Minimize(cmax)
        stat2 = s.Solve(m)
        if stat2 in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            sol = {q: s.Value(v) for q, v in x.items()}
            status = "OPTIMAL" + ("" if stat2 == cp_model.OPTIMAL else " (min-max load stage: feasible)")
    stations = [[i for i in tasks if sol.get((i, k))] for k in range(K)]
    stations = [g for g in stations if g]
    # order tasks inside a station topologically by precedence
    pred = _preds(tasks, edges)
    ordered = []
    for g in stations:
        out, rest = [], list(g)
        while rest:
            nxt = next(i for i in rest if not (pred[i] & set(rest)))
            out.append(nxt)
            rest.remove(nxt)
        ordered.append(out)
    loads = [sum(t[i] for i in g) for g in ordered]
    if max(loads) > c + 1e-6:
        status += "_ROUNDING_VIOLATION"
    return metrics("EXACT_CPSAT", loads, c, ordered,
                   {"rule": "SALBP-1 binary assignment model, CP-SAT; ties broken by minimum max station load",
                    "status": status, "objective_stations": len(loads), "lower_bound": lb,
                    "wall_time_s": s.WallTime()})


def salbp2_min_cycle(tasks, t, edges, n, time_limit=20.0):
    """SALBP-2: for a fixed number of stations n, minimise the cycle time (max station load)."""
    m = cp_model.CpModel()
    ti = {x: int(round(t[x] * SCALE)) for x in tasks}
    x = {(i, k): m.NewBoolVar(f"x_{i}_{k}") for i in tasks for k in range(n)}
    sidx = {i: m.NewIntVar(0, n - 1, f"s_{i}") for i in tasks}
    cmax = m.NewIntVar(0, sum(ti.values()), "c")
    for i in tasks:
        m.AddExactlyOne(x[i, k] for k in range(n))
        m.Add(sidx[i] == sum(k * x[i, k] for k in range(n)))
    for k in range(n):
        m.Add(sum(ti[i] * x[i, k] for i in tasks) <= cmax)
    for a, b in edges:
        if a in tasks and b in tasks:
            m.Add(sidx[a] <= sidx[b])
    m.Minimize(cmax)
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = time_limit
    s.parameters.num_workers = 8
    st = s.Solve(m)
    loads = [sum(t[i] for i in tasks if s.Value(x[i, k])) for k in range(n)]
    return {"n_stations": n, "min_cycle_time_s": max(loads), "station_loads_s": loads, "status": s.StatusName(st)}


def run_all(tasks, t, edges, c, comsoal_trials=1000):
    heur = [rpw(tasks, t, edges, c), lcr(tasks, t, edges, c), comsoal(tasks, t, edges, c, comsoal_trials)]
    ub = min(h["n_stations"] for h in heur)
    ex = exact_salbp1(tasks, t, edges, c, ub)
    tm = math.ceil(sum(t[x] for x in tasks) / c - 1e-9)
    return heur, ex, tm
