"""Human-robot task allocation MILP (CP-SAT).

Decision variables
  x[e,h]      = 1 if element_key e stays with the human
  x[e,o]      = 1 if e is done by robot option o = (model m, task group g) at e's station
  y[m,s]      = 1 if one robot of model m is bought for station s           (at most one cobot cell per station)
  z[m,s,cat]  = 1 if that robot carries the end-effector for task category cat
Objective (equivalent annual cost, USD/yr; engineering economics with A/P, ISyE 510 lecture03):
  min  sum_{m,s} y (P_m + safety + k_int P_m) (A/P,i,n) + maint * P_m
     + sum_{m,s,cat} z * EOAT_cat * ((A/P,i,n) + maint)
     + sum_e x[e,h] * t_e * N_cycles/yr * w_loaded / 3600
Constraints per station s (paced line, worker-machine chart):
  H_s = sum t_e x[e,h],  R_s = sum r_eo x[e,o],  I_s = h_o * sum x[e,o]      (h_o = handover per robot element)
  C_s >= H_s + I_s,  C_s >= R_s + I_s,  C_s <= takt                         (C_s = max(H_s, R_s) + I_s)
  robot may start only after its human predecessors:  sum_{p in pred*(e)} t_p x[p,h] + R_s + I_s <= takt  if x[e,o]
  x[e,o] only if s3 allocation in {robot, human_robot} and the model passed s4 and has an s6a cycle time
  optional ergonomic cap: sum_e ergo_e x[e,h] <= cap   (awkward-posture s/cycle: trunk bent >= 20 deg or
                                                     elbow at/above shoulder, from s2 posture shares)
  IDLE (waiting) is not work content: it is the balance delay of the paced station and is not assigned.
  Layout lever (s3 allocation 'layout_change', e.g. WALK -> AMR kitting / point-of-use supply): the element stays with
  the operator at (1 - reduction) of its time, for an annualised station device + shared line device cost.
  Placement hardware: y needs a 7th-axis conveyor-synchronised track (capex) if any of its task groups is
  track-mounted in s4; an arm serving k > 1 task categories needs an automatic tool changer (capex) and k tool
  exchanges per cycle (k x exchange time added to R_s).
  Fixed configuration (design-space exploration, s5x): fix_models = {station: model | None} pins y; a pinned arm must
  take >= 1 element, and the MILP then decides only the task split (what the arm does and in which order).
  Relief operator (manual_relief): roaming helper takes whole elements in parallel; capacity
  sum(helper times) + walk x stations served <= takt x helpers; cost = helpers x shifts x hours x loaded wage.
"""
from __future__ import annotations

import math
from collections import defaultdict

from ortools.sat.python import cp_model

from rcd.s7_roi import assumptions as A

TS = 1000  # integer time unit = 1 ms


class AllocModel:
    def __init__(self, cfg, data):
        self.cfg, self.d = cfg, data
        self.takt = data["takt"]
        ec = cfg["economics"]
        self.i, self.n = ec["discount_rate"], int(ec["horizon_years"])
        self.crf = A.crf(self.i, self.n)
        self.cycles = A.annual_cycles(cfg)
        self.wage = A.loaded_wage(cfg)
        rows = data["rows"]
        self.keys = [k for k in rows if not rows[k].get("non_work")]   # IDLE = balance delay, not assigned
        self.opts = {k: rows[k]["robot_options"] for k in self.keys}
        # transitive predecessors inside the station (observed precedence)
        succ = defaultdict(set)
        for a, b in data["precedence_within"]:
            succ[a].add(b)
        pred = defaultdict(set)
        for a in list(succ):
            stack = list(succ[a])
            seen = set()
            while stack:
                b = stack.pop()
                if b in seen:
                    continue
                seen.add(b)
                pred[b].add(a)
                stack += list(succ[b])
        self.pred = pred

    # ---------------------------------------------------------------- cost helpers (floats)
    def labour_cost(self, t_s):
        return t_s * self.cycles * self.wage / 3600.0

    def robot_annual(self, price):
        base = price + A.SAFETY_USD + A.INTEGRATION_MULT * price
        return base * self.crf + A.MAINT_PCT * price

    def eoat_annual(self, cat):
        return A.eoat_cost(cat) * (self.crf + A.MAINT_PCT)

    def helper_annual(self):
        """One roaming relief operator on every shift (whole headcount, loaded wage)."""
        return A.annual_hours(self.cfg) * self.wage

    def options(self, k, robots=True, helpers=False, layout=False):
        o = list(self.opts[k]) if robots else []
        r = self.d["rows"][k]
        if layout and r["s3_allocation"] == "layout_change":
            cat = r.get("s3_category") or "default"
            lv = A.LAYOUT_LEVERS.get(cat, A.LAYOUT_LEVERS["default"])
            o.append({"model": "human", "group_id": None, "category": cat, "layout": True,
                      "time_s": r["time_s"] * (1 - lv["reduction"]), "reduction": lv["reduction"]})
        if helpers and r["code"] != "UNOBS":
            o.append({"model": "helper", "group_id": None, "category": "relief_operator", "time_s": r["time_s"],
                      "helper": True})
        return o

    # ---------------------------------------------------------------- build + solve
    def solve(self, name, mode, ergo_cap=None, time_limit=30.0, robots=True, helpers=False, layout=False,
              cycle_cap=None, fix_models=None):
        d, rows = self.d, self.d["rows"]
        stations = d["stations"]
        OPT = {k: self.options(k, robots, helpers, layout) for k in self.keys}
        nonarm = lambda o: o.get("helper") or o.get("layout")  # noqa: E731
        m = cp_model.CpModel()
        xh = {k: m.NewBoolVar(f"h_{k}") for k in self.keys}
        xr = {(k, j): m.NewBoolVar(f"r_{k}_{j}") for k in self.keys for j in range(len(OPT[k]))}
        models_at = defaultdict(set)
        cats_at = defaultdict(set)
        mount_of = {}
        for (k, j) in xr:
            o = OPT[k][j]
            if nonarm(o):
                continue
            st = rows[k]["station"]
            models_at[st].add(o["model"])
            cats_at[(st, o["model"])].add(o["category"])
            mount_of[(st, o["category"])] = o.get("mounting", "fixed_floor")
        y = {(mm, st): m.NewBoolVar(f"y_{mm}_{st}") for st in models_at for mm in models_at[st]}
        z = {(mm, st, c): m.NewBoolVar(f"z_{mm}_{st}_{c}") for (st, mm), cs in cats_at.items() for c in cs}
        for k in self.keys:
            m.AddExactlyOne([xh[k]] + [xr[k, j] for j in range(len(OPT[k]))])
        for (k, j), v in xr.items():
            o = OPT[k][j]
            st = rows[k]["station"]
            if nonarm(o):
                continue
            m.AddImplication(v, y[o["model"], st])
            m.AddImplication(v, z[o["model"], st, o["category"]])
        for (mm, st, c), v in z.items():
            m.AddImplication(v, y[mm, st])
        for st in models_at:
            m.Add(sum(y[mm, st] for mm in models_at[st]) <= 1)
        if fix_models is not None:   # design-space exploration: the arm model at every station is given (None = no arm)
            for st in stations:
                want = fix_models.get(st)
                if want is not None and want not in models_at.get(st, ()):
                    return {"name": name, "status": "MODEL_NOT_APPLICABLE"}
                for mm in models_at.get(st, ()):
                    m.Add(y[mm, st] == int(mm == want))
                if want is not None:   # a bought arm must take at least one element
                    m.Add(sum(v for (k, j), v in xr.items() if rows[k]["station"] == st
                              and OPT[k][j].get("model") == want) >= 1)
        # placement-implied hardware: 7th-axis track if any used category is track-mounted (s4); automatic tool
        # changer if one arm serves >1 category, with k exchanges per cycle for k tools (adds robot time)
        trk = {q: m.NewBoolVar(f"trk_{q[0]}_{q[1]}") for q in y}
        tcg = {q: m.NewBoolVar(f"tc_{q[0]}_{q[1]}") for q in y}
        nch = {q: m.NewIntVar(0, 9, f"nch_{q[0]}_{q[1]}") for q in y}
        for (mm, st), _ in y.items():
            zs = [z[mm, st, c] for c in cats_at[(st, mm)]]
            K = max(1, len(zs))
            for c in cats_at[(st, mm)]:
                if mount_of.get((st, c)) == "linear_track":
                    m.AddImplication(z[mm, st, c], trk[mm, st])
            m.AddImplication(trk[mm, st], y[mm, st])
            m.Add(K * tcg[mm, st] >= sum(zs) - 1)
            m.Add(nch[mm, st] >= sum(zs) - K * (1 - tcg[mm, st]))
            m.AddImplication(tcg[mm, st], y[mm, st])
        takt_i = int(round((cycle_cap if cycle_cap is not None else self.takt) * TS))
        H_o = int(round(A.HANDOVER_S * TS))
        for st in stations:
            ks = [k for k in self.keys if rows[k]["station"] == st]
            th = {k: int(round(rows[k]["time_s"] * TS)) for k in ks}
            slack = len(ks)   # rounding slack (1 ms per element), verified on floats after the solve
            # layout-lever options are still operator work (reduced time) -> H; arm and relief operator -> R
            H = sum(th[k] * xh[k] for k in ks) + sum(int(round(OPT[k][j]["time_s"] * TS)) * xr[k, j] for k in ks
                                                     for j in range(len(OPT[k])) if OPT[k][j].get("layout"))
            R = sum(int(round(OPT[k][j]["time_s"] * TS)) * xr[k, j] for k in ks for j in range(len(OPT[k]))
                    if not OPT[k][j].get("layout")) + sum(int(round(A.TOOL_CHANGE_S * TS)) * nch[q] for q in nch
                                                          if q[1] == st)
            I = sum(H_o * xr[k, j] for k in ks for j in range(len(OPT[k])) if not nonarm(OPT[k][j]))
            C = m.NewIntVar(0, takt_i + slack, f"C_{st}")
            m.Add(C >= H + I)
            m.Add(C >= R + I)
            for k in ks:
                for j in range(len(OPT[k])):
                    if nonarm(OPT[k][j]):
                        continue
                    pre = [p for p in self.pred.get(k, ()) if p in th]
                    if pre:   # (conservative: a predecessor's full human time even if a layout lever shortens it)
                        hum_pre = []
                        for p_ in pre:
                            hum_pre.append(th[p_] * xh[p_])
                            hum_pre += [int(round(OPT[p_][q]["time_s"] * TS)) * xr[p_, q]
                                        for q in range(len(OPT[p_])) if OPT[p_][q].get("layout")]
                        m.Add(sum(hum_pre) + R + I <= takt_i + slack).OnlyEnforceIf(xr[k, j])
        # objective pieces (integer USD)
        lab = {k: int(round(self.labour_cost(rows[k]["time_s"]))) for k in self.keys}
        lab_layout = lambda k, o: int(round(self.labour_cost(o["time_s"])))  # noqa: E731
        rob = {(mm, st): int(round(self.robot_annual(d["robots"][mm]["_price_mid"]))) for (mm, st) in y}
        eo = {(mm, st, c): int(round(self.eoat_annual(c))) for (mm, st, c) in z}
        TRK = int(round(A.TRACK_USD * (self.crf + A.MAINT_PCT)))
        TCG = int(round(A.TOOL_CHANGER_USD * (self.crf + A.MAINT_PCT)))
        cost = sum(lab[k] * xh[k] for k in self.keys) + sum(rob[q] * y[q] for q in y) + sum(eo[q] * z[q] for q in z) \
            + sum(TRK * v for v in trk.values()) + sum(TCG * v for v in tcg.values())
        # layout levers: per-station device (kit rack / fixture) and a line-level shared device (AMR), annualised
        lk = [(k, j) for (k, j) in xr if OPT[k][j].get("layout")]
        lst = {}
        lline = {}
        for (k, j) in lk:
            cat = OPT[k][j]["category"]
            st = rows[k]["station"]
            if (st, cat) not in lst:
                lst[st, cat] = m.NewBoolVar(f"lay_{st}_{cat}")
            if cat not in lline:
                lline[cat] = m.NewBoolVar(f"layline_{cat}")
            m.AddImplication(xr[k, j], lst[st, cat])
            m.AddImplication(lst[st, cat], lline[cat])
            cost = cost + lab_layout(k, OPT[k][j]) * xr[k, j]
        LV = lambda c: A.LAYOUT_LEVERS.get(c, A.LAYOUT_LEVERS["default"])  # noqa: E731
        cost = cost + sum(int(round(LV(c)["station_capex_usd"] * (self.crf + A.MAINT_PCT))) * v for (st, c), v in lst.items()) \
            + sum(int(round(LV(c)["line_capex_usd"] * (self.crf + A.MAINT_PCT))) * v for c, v in lline.items())
        hk = [(k, j) for (k, j) in xr if OPT[k][j].get("helper")]
        # helper-assigned time stays in the (fractional) labour cost: moving work to a relief operator saves nothing;
        # the relief operator is extra headcount (HC per helper and shift, below)
        cost = cost + sum(lab[k] * xr[k, j] for (k, j) in hk)
        nH = m.NewIntVar(0, len(stations), "n_helpers")
        if hk:
            served = {st: m.NewBoolVar(f"served_{st}") for st in stations}
            for (k, j) in hk:
                m.AddImplication(xr[k, j], served[rows[k]["station"]])
            walk = int(round(A.HELPER_WALK_S * TS))
            m.Add(sum(int(round(rows[k]["time_s"] * TS)) * xr[k, j] for (k, j) in hk)
                  + walk * sum(served.values()) <= takt_i * nH)
        else:
            m.Add(nH == 0)
        HC = int(round(self.helper_annual()))
        cost = cost + HC * nH
        # awkward-posture seconds still done by people (operator or relief helper); robots remove exposure
        bent = sum(int(round(rows[k]["ergo_s"] * TS)) * xh[k] for k in self.keys) + sum(
            int(round(rows[k]["ergo_s"] * TS)) * xr[k, j] for (k, j) in xr if OPT[k][j].get("helper")) + sum(
            int(round(rows[k]["ergo_s"] * (1 - OPT[k][j]["reduction"]) * TS)) * xr[k, j] for (k, j) in xr
            if OPT[k][j].get("layout"))
        moved = sum(int(round(rows[k]["time_s"] * TS)) * xr[k, j] for (k, j) in xr if not nonarm(OPT[k][j]))
        if ergo_cap is not None:
            m.Add(bent <= int(math.floor(ergo_cap * TS)))
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit
        solver.parameters.num_workers = 8
        stages = []
        if mode == "ergonomics_first":
            m.Minimize(bent)
            st1 = solver.Solve(m)
            stages.append(("min_bent", solver.StatusName(st1), solver.ObjectiveValue()))
            if st1 != cp_model.OPTIMAL:
                return {"name": name, "status": solver.StatusName(st1)}
            m.Add(bent <= int(round(solver.ObjectiveValue())))
        elif mode == "max_automation":
            m.Maximize(moved)
            st1 = solver.Solve(m)
            stages.append(("max_moved", solver.StatusName(st1), solver.ObjectiveValue()))
            if st1 != cp_model.OPTIMAL:
                return {"name": name, "status": solver.StatusName(st1)}
            m.Add(moved >= int(round(solver.ObjectiveValue())))
        m.Minimize(cost)
        stat = solver.Solve(m)
        stages.append(("min_cost", solver.StatusName(stat), solver.ObjectiveValue()))
        if hk and stat == cp_model.OPTIMAL:   # tie-break: give the relief operator no more work than needed
            best = int(round(solver.ObjectiveValue()))
            m.Add(cost <= best)
            m.Minimize(sum(int(round(rows[k]["time_s"] * TS)) * xr[k, j] for (k, j) in hk))
            stat = solver.Solve(m)
            stages.append(("min_helper_time", solver.StatusName(stat), solver.ObjectiveValue()))
            solver_cost = best
        else:
            solver_cost = solver.ObjectiveValue()
        res = {"name": name, "status": solver.StatusName(stat), "solve_stages": stages,
               "solver_objective_usd": solver_cost, "wall_time_s": solver.WallTime()}
        if stat not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return res
        choice = {}
        for k in self.keys:
            if solver.Value(xh[k]):
                choice[k] = None
            for j in range(len(OPT[k])):
                if solver.Value(xr[k, j]):
                    choice[k] = OPT[k][j]
        res.update(self.evaluate(name, choice, ergo_cap, n_helpers=solver.Value(nH)))
        if cycle_cap is not None:
            res["cycle_cap_s"] = cycle_cap
            res["checks"]["station_cycle_le_cycle_cap"] = all(v["cycle_s"] <= cycle_cap + 1e-6
                                                              for v in res["station_times"].values())
        lay_int = sum(lab_layout(k, OPT[k][j]) for (k, j) in lk if solver.Value(xr[k, j])) + \
            sum(int(round(LV(c)["station_capex_usd"] * (self.crf + A.MAINT_PCT))) for (st, c), v in lst.items() if solver.Value(v)) + \
            sum(int(round(LV(c)["line_capex_usd"] * (self.crf + A.MAINT_PCT))) for c, v in lline.items() if solver.Value(v))
        res["checks"]["objective_matches_solver"] = abs(res["_int_cost"](lab, rob, eo) + HC * solver.Value(nH) + lay_int
                                                        - solver_cost) < 1.5 + len(lk)
        res["helpers_allowed"], res["robots_allowed"], res["layout_allowed"] = helpers, robots, layout
        del res["_int_cost"]
        res["status"] = solver.StatusName(stat)
        return res

    # ---------------------------------------------------------------- as-is line (evaluated, not optimised)
    def evaluate_baseline(self):
        res = self.evaluate("baseline_manual", {k: None for k in self.keys}, None, evaluated=True)
        worst = max(v["cycle_s"] for v in res["station_times"].values())
        res["status"] = "EVALUATED"
        res["overload_pct"] = max(0.0, worst / self.takt - 1)
        res["station_overload"] = {st: {**self.d["station_info"][st],
                                        "cycle_s": v["cycle_s"], "overload_pct": max(0.0, v["cycle_s"] / self.takt - 1)}
                                   for st, v in res["station_times"].items()}
        res["checks"]["overload_pct_rederived"] = abs(res["overload_pct"] - max(0.0, worst / self.takt - 1)) < 1e-12
        del res["_int_cost"]
        return res

    # ---------------------------------------------------------------- evaluate a fixed assignment (floats)
    def evaluate(self, name, choice, ergo_cap=None, n_helpers=0, evaluated=False):
        d, rows = self.d, self.d["rows"]
        assignment, st_times = [], {}
        robots = {}
        for k in self.keys:
            r = rows[k]
            o = choice.get(k)
            if o is None:
                assignment.append({"element_key": k, "station": r["station"], "code": r["code"],
                                   "resource": "human", "time_s": r["time_s"], "human_time_s": r["time_s"]})
            elif o.get("layout"):
                assignment.append({"element_key": k, "station": r["station"], "code": r["code"],
                                   "resource": "human", "time_s": o["time_s"], "human_time_s": r["time_s"],
                                   "lever": o["category"], "lever_reduction": o["reduction"]})
            elif o.get("helper"):
                assignment.append({"element_key": k, "station": r["station"], "code": r["code"],
                                   "resource": "helper", "time_s": r["time_s"], "human_time_s": r["time_s"]})
            else:
                assignment.append({"element_key": k, "station": r["station"], "code": r["code"],
                                   "resource": o["model"], "group_id": o["group_id"], "time_s": o["time_s"],
                                   "human_time_s": r["time_s"]})
                rb = robots.setdefault((o["model"], r["station"]), {"model": o["model"], "station": r["station"],
                                                                    "groups": set(), "categories": set(), "elements": [],
                                                                    "mountings": set()})
                rb["groups"].add(o["group_id"])
                rb["categories"].add(o["category"])
                rb["elements"].append(k)
                rb["mountings"].add(o.get("mounting", "fixed_floor"))
        # automatic tool changer: k tools -> k exchanges per cycle, booked on the robot's first element row
        for (mm, st_), rb in robots.items():
            rb["tool_changes"] = len(rb["categories"]) if len(rb["categories"]) > 1 else 0
            if rb["tool_changes"]:
                first = next(a for a in assignment if a["element_key"] == sorted(rb["elements"])[0])
                first["tool_change_s"] = A.TOOL_CHANGE_S * rb["tool_changes"]
                first["time_s"] += first["tool_change_s"]
        isrob = lambda a: a["resource"] not in ("human", "helper")  # noqa: E731
        prec_ok = True
        for st in d["stations"]:
            rs = [a for a in assignment if a["station"] == st]
            h = sum(a["time_s"] for a in rs if a["resource"] == "human")
            ro = sum(a["time_s"] for a in rs if isrob(a))
            hp = sum(a["time_s"] for a in rs if a["resource"] == "helper")
            nr = sum(1 for a in rs if isrob(a))
            inter = A.HANDOVER_S * nr
            rr = ro + hp     # second resource of the station (robot and/or relief operator), as the gate re-derives
            cyc = max(h, rr) + inter
            pre_max = 0.0
            for a in rs:
                if isrob(a):
                    pre = sum(b["time_s"] for b in rs if b["resource"] == "human" and b["element_key"] in self.pred.get(a["element_key"], ()))
                    pre_max = max(pre_max, pre)
            if nr:
                prec_ok &= pre_max + rr + inter <= self.takt + 1e-6
            st_times[st] = {"human_s": h, "robot_s": rr, "robot_only_s": ro, "helper_s": hp, "interaction_s": inter,
                            "cycle_s": cyc,
                            "robot_model": next((a["resource"] for a in rs if isrob(a)), None),
                            "robot_start_after_human_s": pre_max if nr else None,
                            "utilization_human": h / self.takt, "takt_s": self.takt,
                            "overload_pct": max(0.0, cyc / self.takt - 1)}
        rob_list = []
        capex_total = 0.0
        annual_robot = 0.0
        for (mm, st), rb in sorted(robots.items(), key=lambda x: x[0][1]):
            price = d["robots"][mm]["_price_mid"]
            eoat = sum(A.eoat_cost(c) for c in rb["categories"])
            track = A.TRACK_USD if "linear_track" in rb["mountings"] else 0.0
            changer = A.TOOL_CHANGER_USD if rb["tool_changes"] else 0.0
            capex = price + A.SAFETY_USD + A.INTEGRATION_MULT * price + eoat + track + changer
            ann = self.robot_annual(price) + sum(self.eoat_annual(c) for c in rb["categories"]) \
                + (track + changer) * (self.crf + A.MAINT_PCT)
            capex_total += capex
            annual_robot += ann
            rob_list.append({"model": mm, "station": st, "count": 1, "groups": sorted(rb["groups"]),
                             "categories": sorted(rb["categories"]), "elements": sorted(rb["elements"]),
                             "price_usd_range": d["robots"][mm]["price_usd_range"], "price_mid_usd": price,
                             "capex_breakdown_usd": {"robot": price, "eoat": eoat, "safety": A.SAFETY_USD,
                                                     "integration": A.INTEGRATION_MULT * price,
                                                     **({"linear_track": track} if track else {}),
                                                     **({"tool_changer": changer} if changer else {})},
                             "mounting": sorted(rb["mountings"]),
                             "tool_changes_per_cycle": rb["tool_changes"],
                             "tool_change_s_per_cycle": A.TOOL_CHANGE_S * rb["tool_changes"],
                             "capex_usd": capex, "annualized_usd": ann})
        layout = {}
        for a in assignment:
            if a.get("lever"):
                layout.setdefault((a["station"], a["lever"]), []).append(a["element_key"])
        lay_list, lay_capex, lay_ann = [], 0.0, 0.0
        for (st, cat), keys in sorted(layout.items()):
            lv = A.LAYOUT_LEVERS.get(cat, A.LAYOUT_LEVERS["default"])
            lay_list.append({"station": st, "lever": cat, "elements": keys, "reduction": lv["reduction"],
                             "capex_usd": lv["station_capex_usd"], "desc": lv["desc"]})
            lay_capex += lv["station_capex_usd"]
        for cat in sorted({c for (_, c) in layout}):
            lv = A.LAYOUT_LEVERS.get(cat, A.LAYOUT_LEVERS["default"])
            if lv["line_capex_usd"]:
                lay_list.append({"station": "LINE", "lever": cat, "elements": [], "reduction": None,
                                 "capex_usd": lv["line_capex_usd"], "desc": "shared line-level device for " + cat})
                lay_capex += lv["line_capex_usd"]
        lay_ann = lay_capex * (self.crf + A.MAINT_PCT)
        human_total = sum(a["time_s"] for a in assignment if a["resource"] == "human")
        helper_total = sum(a["time_s"] for a in assignment if a["resource"] == "helper")
        labour = self.labour_cost(human_total + helper_total)
        helper_cost = n_helpers * self.helper_annual()
        ef = lambda a: rows[a["element_key"]]["ergo_s"] * (1 - a.get("lever_reduction", 0.0))  # noqa: E731
        ergo = sum(ef(a) for a in assignment if a["resource"] in ("human", "helper"))
        ergo_op = sum(ef(a) for a in assignment if a["resource"] == "human")
        checks = {
            "every_element_assigned_once": len({a["element_key"] for a in assignment}) == len(assignment) == len(self.keys),
            "cycle_ge_max_human_robot": all(v["cycle_s"] >= max(v["human_s"], v["robot_s"]) - 1e-9 for v in st_times.values()),
            "robot_only_where_s3_allows": all(rows[a["element_key"]]["s3_allocation"] in ("robot", "human_robot")
                                              for a in assignment if isrob(a)),
            "unobserved_block_manual": all(a["resource"] == "human" for a in assignment if a["code"] == "UNOBS"),
            "robot_model_passed_s4_and_simulated": all(
                any(o["model"] == a["resource"] and o["group_id"] == a.get("group_id") for o in rows[a["element_key"]]["robot_options"])
                for a in assignment if isrob(a)),
            "at_most_one_robot_per_station": len({st for (_, st) in robots}) == len(robots),
            "robot_starts_after_human_predecessors": prec_ok,
        }
        if not evaluated:
            checks["station_cycle_le_takt"] = all(v["cycle_s"] <= self.takt + 1e-6 for v in st_times.values())
        if helper_total > 0:
            checks["helper_capacity_respected"] = helper_total + A.HELPER_WALK_S * sum(
                1 for v in st_times.values() if v["helper_s"] > 0) <= n_helpers * self.takt + 1e-6
        if ergo_cap is not None:
            checks["ergonomic_cap_respected"] = ergo <= ergo_cap + 1e-3

        def int_cost(lab, rob, eo):
            c = sum(lab[a["element_key"]] for a in assignment if a["resource"] in ("human", "helper") and not a.get("lever"))
            for (mm, st), rb in robots.items():
                c += rob[(mm, st)] + sum(eo[(mm, st, cat)] for cat in rb["categories"])
                c += int(round(A.TRACK_USD * (self.crf + A.MAINT_PCT))) if "linear_track" in rb["mountings"] else 0
                c += int(round(A.TOOL_CHANGER_USD * (self.crf + A.MAINT_PCT))) if rb["tool_changes"] else 0
            return c
        return {"name": name, "objective": {"annual_cost_usd": labour + annual_robot + helper_cost + lay_ann,
                                            "layout_annualized_usd": lay_ann, "layout_capex_usd": lay_capex,
                                            "robot_capex_usd": capex_total,
                                            "labour_usd": labour, "helper_usd": helper_cost, "n_helpers_per_shift": n_helpers,
                                            "robot_annualized_usd": annual_robot, "human_s_per_cycle": human_total,
                                            "helper_s_per_cycle": helper_total,
                                            "awkward_posture_s_per_cycle": ergo, "awkward_posture_operator_s": ergo_op,
                                            "capex_usd": capex_total + lay_capex, "ergonomic_cap_s": ergo_cap,
                                            "max_station_cycle_s": max(v["cycle_s"] for v in st_times.values())},
                "robots": rob_list, "layout_changes": lay_list, "assignment": assignment, "station_times": st_times,
                "checks": checks, "_int_cost": int_cost}
