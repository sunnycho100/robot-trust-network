"""Stage s3: automatability scoring of every element_key "<station>:<code>" (UNOBS excluded: fixed manual).

Method: weighted-sum multi-criteria scoring with AHP-derived weights (Saaty 1980; CR < 0.10), criteria taken
from (i) Fraunhofer IPA Automation Potential Analysis (technical feasibility of handling/joining, tolerances,
quantity/repetition, cycle-time economics), (ii) MTM-based automation suitability of basic motions
(Kuhlang et al. / Eng 2024, 5(2), 53: motion-element level, grasp/place difficulty), (iii) HRC task-allocation
review (Int. J. Computer Integrated Manufacturing 2023: variability, force, ergonomics as allocation drivers).
Criterion values come from s2 measurements (mocap, time study, ergonomics) except force/torque and part
properties, which are explicit assumptions (rcd.s2_elements.parts). Weight robustness: +/-20 % multiplicative
perturbations (Monte-Carlo, renormalised), Spearman rank correlation of the task ranking vs. baseline.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

from rcd.common.io import ROOT, load_json, out_dir, save_json
from rcd.s2_elements.parts import GRIPPER_EOAT, SCREWDRIVER_EOAT, part_for

from .ahp import ahp, matrix_from_judgements
from .generic_parts import resolver as intake_parts

SRC_IPA = "Fraunhofer IPA, Automation Potential Analysis: https://www.ipa.fraunhofer.de/en/current-research/robot-and-assistive-systems/assembly-automation/automation-potential-analysis.html"
SRC_MTM = "MTM-based automation suitability of work elements, Eng 2024, 5(2), 53: https://doi.org/10.3390/eng5020053"
SRC_HRC = "HRC task allocation review, Int. J. Computer Integrated Manufacturing (2023): https://www.tandfonline.com/doi/full/10.1080/0951192X.2023.2204467"

CRITERIA = [
    ("repetitiveness", "Occurrences per cycle and stability of the per-cycle time (1 - CV): repetitive, stable work "
                       "amortises a robot", f"{SRC_IPA}; {SRC_MTM}"),
    ("handling_simplicity", "Ease of robotic grasp/handling for the element type (MTM-UAS case of grasp; rigid vs. "
                            "deformable part; tool/screw presentation automatable)", f"{SRC_MTM}; {SRC_IPA}"),
    ("tolerance_slack", "Positioning tolerance available (precision_mm / 2 mm, capped at 1): looser tolerance is easier",
     SRC_IPA),
    ("force_feasibility", "Required force/torque within collaborative-robot and EOAT limits (ISO/TS 15066 contact "
                          "limits; OnRobot screwdriver 0.15-5 Nm)", f"{SRC_HRC}; {SCREWDRIVER_EOAT['source']}"),
    ("motion_consistency", "Low spatial variability of the hand motion across occurrences (mocap dispersion of reach "
                           "and hand height): structured motions are programmable", f"{SRC_HRC}; {SRC_MTM}"),
    ("ergonomic_burden", "Share of the element in awkward postures (bent, overhead, asymmetric) and REBA trunk score "
                         "from s2 ergonomics: high burden raises the priority to automate", f"{SRC_HRC}; EAWS (Schaub et al. 2013)"),
    ("cycle_time_share", "Pareto share of the element's standard time in the station's observed standard time "
                         "(normalised to the largest element of the station)", SRC_IPA),
    ("transport_content", "Walking/carrying content (pelvis travel per cycle): candidates for AMR kitting or layout change",
     f"{SRC_IPA}; ISyE 510 lecture07 (space requirements / layout)"),
]
KEYS = [k for k, _, _ in CRITERIA]

# AHP pairwise judgements (Saaty 1-9). Rationale: technical feasibility (handling, tolerance, force, consistency)
# is a precondition in the IPA potential analysis, so it dominates; among economics, cycle-time share outranks
# repetitiveness; ergonomics is a secondary driver in the HRC review; transport is addressed by layout/AMR and is
# least decisive for robot-arm automation.
JUDGEMENTS = {
    ("repetitiveness", "handling_simplicity"): 1 / 3, ("repetitiveness", "tolerance_slack"): 1 / 2,
    ("repetitiveness", "force_feasibility"): 1, ("repetitiveness", "motion_consistency"): 1,
    ("repetitiveness", "ergonomic_burden"): 1, ("repetitiveness", "cycle_time_share"): 1 / 2,
    ("repetitiveness", "transport_content"): 3,
    ("handling_simplicity", "tolerance_slack"): 2, ("handling_simplicity", "force_feasibility"): 3,
    ("handling_simplicity", "motion_consistency"): 2, ("handling_simplicity", "ergonomic_burden"): 3,
    ("handling_simplicity", "cycle_time_share"): 1, ("handling_simplicity", "transport_content"): 5,
    ("tolerance_slack", "force_feasibility"): 2, ("tolerance_slack", "motion_consistency"): 1,
    ("tolerance_slack", "ergonomic_burden"): 2, ("tolerance_slack", "cycle_time_share"): 1 / 2,
    ("tolerance_slack", "transport_content"): 3,
    ("force_feasibility", "motion_consistency"): 1 / 2, ("force_feasibility", "ergonomic_burden"): 1,
    ("force_feasibility", "cycle_time_share"): 1 / 3, ("force_feasibility", "transport_content"): 2,
    ("motion_consistency", "ergonomic_burden"): 2, ("motion_consistency", "cycle_time_share"): 1 / 2,
    ("motion_consistency", "transport_content"): 3,
    ("ergonomic_burden", "cycle_time_share"): 1 / 2, ("ergonomic_burden", "transport_content"): 2,
    ("cycle_time_share", "transport_content"): 4,
}

# Element-type properties for a robot (assumptions, cited in the output).
HANDLING = {"FASTEN": 0.85, "REACH_FASTENER": 0.80, "GET_TOOL": 0.90, "GET_PART": 0.70, "INSERT": 0.50,
            "INSPECT": 0.70, "SETUP": 0.30, "WALK": 0.60, "IDLE": 0.0}
HANDLING_NOTE = {"FASTEN": "tool on axis + run-down: standard robotic screwdriving",
                 "REACH_FASTENER": "single screws from a pouch is a difficult grasp for a human, but is eliminated by "
                                   "automatic screw presentation (blow-feed / pick-up screwdriver)",
                 "GET_TOOL": "eliminated when the screwdriver is the robot's permanent end-effector",
                 "GET_PART": "rigid part from a cart/kit: gripper pick (x0.5 if the part is deformable)",
                 "INSERT": "insertion into clips/channels needs compliance/force control (x0.5 if deformable)",
                 "INSPECT": "machine-vision check", "SETUP": "cart pushing / wheel locking / height adjustment",
                 "WALK": "material transport by AMR or eliminated by layout", "IDLE": "waiting: nothing to automate"}
FORCE = {  # (force/torque requirement, feasibility score)
    "FASTEN": ("screw/nut run-down approx. 2-5 Nm (M4-M6 into plastic/sheet metal)", 0.80),
    "REACH_FASTENER": ("< 5 N", 1.0), "GET_TOOL": ("< 30 N", 1.0), "GET_PART": ("part weight < 30 N", 0.9),
    "INSERT": ("clip/channel insertion approx. 50-150 N push", 0.5), "INSPECT": ("none", 1.0),
    "SETUP": ("cart push approx. 100-200 N", 0.4), "WALK": ("carry < 30 N", 0.9), "IDLE": ("none", 1.0)}
FORCE_FLEX = ("gasket/strip seating with manual pressure (hammer) approx. 100-300 N distributed", 0.25)
PRECISION = {"FASTEN": 1.0, "REACH_FASTENER": 1.0, "GET_TOOL": 3.0, "GET_PART": 5.0, "INSPECT": 5.0,
             "SETUP": 20.0, "WALK": 50.0, "IDLE": 50.0}
CATEGORY = {"FASTEN": "cobot_screwdriving", "REACH_FASTENER": "cobot_screwdriving", "GET_TOOL": "cobot_screwdriving",
            "GET_PART": "cobot_pick_place", "INSERT": "cobot_pick_place", "INSPECT": "vision_inspection",
            "SETUP": "fixture_or_layout", "WALK": "amr_kitting", "IDLE": "keep_manual"}
T_ROBOT, T_HRC = 0.60, 0.45
BASE_HEIGHT_M = 0.85   # robot base on a pedestal at operator hip height next to the carrier


def _clip(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def run(cfg: dict) -> None:
    od = out_dir(cfg, "s3")
    s2 = ROOT / cfg["out_root"] / "s2"
    ts = load_json(s2 / "time_study.json", "time_study")
    el = load_json(s2 / "elements.json", "elements")
    sg_path = ROOT / cfg["out_root"] / "s1" / "scene_graph.json"
    scene = load_json(sg_path) if sg_path.exists() else None

    ip = intake_parts(cfg)          # parts=intake: customer intake instead of the built-in part table
    A = matrix_from_judgements(KEYS, JUDGEMENTS)
    res = ahp(A)
    w = dict(zip(KEYS, res["weights"]))
    if res["cr"] >= 0.10:
        raise RuntimeError(f"AHP consistency ratio {res['cr']:.3f} >= 0.10: revise judgements")

    occ = defaultdict(list)
    for c in el["cycles"]:
        for e in c["elements"]:
            occ[(c["station"], e["code"])].append(e)
    stats = {(r["station"], r["code"]): r for r in ts["element_stats"] if r["code"] != "UNOBS"}
    st_total = defaultdict(float)
    for (st, code), r in stats.items():
        st_total[st] = max(st_total[st], r["standard_s"])
    conveyor = {}
    contacts = {}
    if scene:
        for s in scene.get("stations", []):
            v = [c["conveyor"]["speed_mm_s"] for c in s.get("cycles", []) if c.get("conveyor", {}).get("speed_mm_s")]
            conveyor[s["station"]] = float(np.median(v)) if v else None
            pts = [p for c in s.get("cycles", []) for o in c.get("objects", []) if "contact" in o.get("name", "")
                   for p in o.get("positions_m", [])]
            contacts[s["station"]] = pts

    tasks = []
    for (st, code), r in sorted(stats.items()):
        E = occ.get((st, code), [])
        subgoals = sorted({e.get("subgoal") for e in E if e.get("subgoal")})
        if ip is not None:
            parts = [ip(st, s) for s in subgoals] or [ip(st, None)]
        else:
            parts = [part_for(s) for s in subgoals] or [part_for(None)]
        flexible = any(p["flexible"] for p in parts) and code in ("GET_PART", "INSERT")
        # --- criteria
        cv = r["cv"] if r["cv"] is not None else 1.0
        s_rep = 0.5 * _clip(r["occurrences_per_cycle"] / 4.0) + 0.5 * _clip(1.0 - cv)
        s_hand = HANDLING[code] * (0.5 if flexible else 1.0)
        prec = min(p["precision_mm"] for p in parts) if code == "INSERT" else PRECISION[code]
        s_tol = _clip(prec / 2.0)
        force_txt, s_force = FORCE_FLEX if (flexible and code == "INSERT") else FORCE[code]
        hh = np.array([e["hand_height_m"] for e in E]) if E else np.zeros(1)
        rr = np.array([e["reach_m_p95"] for e in E]) if E else np.zeros(1)
        disp = float(hh.std() + rr.std()) if len(E) > 1 else 0.0
        s_cons = float(np.exp(-disp / 0.15))
        dur = np.array([e["duration_s"] for e in E]) if E else np.ones(1)
        ps = [e.get("posture_share", {}) for e in E]
        awk = float(np.average([p.get("bent", 0) + p.get("overhead", 0) + 0.5 * p.get("asymmetric", 0) for p in ps],
                               weights=dur)) if E else 0.0
        reba = float(np.average([e.get("reba_trunk_mean", 1) for e in E], weights=dur)) if E else 1.0
        s_ergo = _clip(0.6 * _clip(awk) + 0.4 * (reba - 1) / 4)
        s_share = _clip(r["standard_s"] / st_total[st]) if st_total[st] > 0 else 0.0
        travel = sum(e["travel_m"] for e in E) / max(1, r["n"])
        s_trans = _clip(travel / 10.0)
        scores = {"repetitiveness": s_rep, "handling_simplicity": s_hand, "tolerance_slack": s_tol,
                  "force_feasibility": s_force, "motion_consistency": s_cons, "ergonomic_burden": s_ergo,
                  "cycle_time_share": s_share, "transport_content": s_trans}
        scores = {k: round(v, 6) for k, v in scores.items()}
        total = sum(w[k] * scores[k] for k in KEYS)
        # --- requirements for s4 (robot selection)
        tool = SCREWDRIVER_EOAT if CATEGORY[code] == "cobot_screwdriving" else GRIPPER_EOAT
        part_kg = max(p["mass_kg"] for p in parts) if code in ("GET_PART", "INSERT") else 0.0
        reach3d = np.sqrt(rr ** 2 + (hh - BASE_HEIGHT_M) ** 2) if E else np.zeros(1)
        v_conv = conveyor.get(st)
        occ_mean_s = float(np.mean(dur)) if E else 0.0
        track_mm = (v_conv * occ_mean_s / 2.0) if v_conv else None   # robot re-synchronises per occurrence
        req = {
            "part": ", ".join(sorted({p["part"] for p in parts})) if code in ("GET_PART", "INSERT") else None,
            "part_mass_kg": part_kg, "part_mass_basis": "; ".join(sorted({p["basis"] for p in parts})),
            "tool": tool["name"], "tool_mass_kg": tool["mass_kg"], "tool_source": tool["source"],
            "payload_kg": round(part_kg + tool["mass_kg"], 3),
            "reach_mm": round(float(np.percentile(reach3d, 95)) * 1000 + (track_mm or 0.0), 0),
            "reach_static_mm": round(float(np.percentile(reach3d, 95)) * 1000, 0),
            "tracking_allowance_mm": None if track_mm is None else round(track_mm, 0),
            "reach_basis": f"p95 over occurrences of the 3D distance from a base at {BASE_HEIGHT_M} m beside the operator's "
                           "standing point to the active wrist (mocap) + half the conveyor travel during one occurrence "
                           "(robot tracks the carrier during each occurrence, s1 conveyor speed)",
            "conveyor_travel_per_cycle_mm": None if not v_conv else round(v_conv * r["standard_s"], 0),
            "work_points": contacts.get(st) or [e["work_point_m"] for e in E[:8]],
            "work_points_frame": "s1 workpiece_contact (scene graph)" if contacts.get(st) else "mocap world frame of each cycle",
            "precision_mm": prec, "force_torque": force_txt,
            "conveyor_tracking": bool((ip.line() if ip is not None else cfg["line"]).get("paced", cfg["line"].get("paced", True))),
            "conveyor_speed_mm_s": v_conv,
            "time_budget_s": r["standard_s"], "occurrences_per_cycle": r["occurrences_per_cycle"],
            "fasteners_per_cycle": round(sum(e.get("fastener_count", 0) for e in E) / max(1, r["n"]), 2)
            if code in ("FASTEN", "REACH_FASTENER") else None,
        }
        if ip is not None:
            req["part_source"] = "intake"
        tasks.append({"station": st, "code": code, "element_key": f"{st}:{code}", "scores": scores, "total": total,
                      "standard_s": r["standard_s"], "mean_s": r["mean_s"], "flexible_part": flexible,
                      "requirements": req, "_notes": {"handling": HANDLING_NOTE[code], "awkward_share": round(awk, 3),
                                                      "reba_trunk_mean": round(reba, 2), "dispersion_m": round(disp, 3),
                                                      "travel_m_per_cycle": round(travel, 2)}})

    # --- allocation rules
    for t in tasks:
        code, tot = t["code"], t["total"]
        cat = CATEGORY[code]
        if code == "WALK":
            loaded = next((r.get("loaded_share") for r in ts["lean"]["rows"] if r["code"] == "WALK"), 0) or 0
            cat = "amr_kitting" if loaded >= 0.3 else "fixture_or_layout"
            alloc = "layout_change"
            why = (f"Walking is NVA (Motion/Transportation), {t['_notes']['travel_m_per_cycle']} m per cycle; remove by "
                   f"point-of-use presentation / kitting (loaded share {loaded:.0%}) rather than by a robot arm.")
        elif code == "SETUP":
            alloc = "layout_change"
            why = "Cart docking and workstation height adjustment: fixture/poka-yoke and stored height presets, not a robot task."
        elif code == "IDLE":
            alloc, cat = "human", "keep_manual"
            why = "Waiting (NVA) is removed by line balancing and by giving the robot the screwdriver, not automated itself."
        elif tot >= T_ROBOT and t["scores"]["handling_simplicity"] >= 0.5 and t["scores"]["force_feasibility"] >= 0.5:
            alloc = "robot"
            why = f"High automation potential ({tot:.2f} >= {T_ROBOT}); {HANDLING_NOTE[code]}."
        elif tot >= T_HRC:
            alloc = "human_robot"
            why = (f"Medium potential ({tot:.2f}); shared task: robot presents/holds, human does the dexterous part "
                   f"({HANDLING_NOTE[code]}).")
        else:
            alloc, cat = "human", "keep_manual"
            why = f"Low potential ({tot:.2f} < {T_HRC}); {HANDLING_NOTE[code]}."
        t["allocation"], t["category"], t["rationale"] = alloc, cat, why
        t["requirements"]["applicable"] = alloc in ("robot", "human_robot")
    # dependency: tool handling and screw presentation disappear when the station's fastening goes to a robot
    for t in tasks:
        if t["code"] in ("GET_TOOL", "REACH_FASTENER"):
            f = next((x for x in tasks if x["station"] == t["station"] and x["code"] == "FASTEN"), None)
            if f and f["allocation"] == "robot":
                t["allocation"], t["category"] = "robot", "cobot_screwdriving"
                t["rationale"] += " Eliminated with robotic screwdriving: tool is the robot EOAT, screws are auto-fed."
            elif f and t["allocation"] == "robot":
                t["allocation"] = "human_robot"
                t["rationale"] += " Fastening stays manual, so this is at most shared work."
            t["requirements"]["applicable"] = t["allocation"] in ("robot", "human_robot")
    for t in tasks:
        sc = t["scores"]
        top = sorted(KEYS, key=lambda k: -w[k] * sc[k])[:3]
        t["rationale"] += " Main drivers: " + ", ".join(f"{k}={sc[k]:.2f}" for k in top) + "."
        t["total"] = sum(w[k] * sc[k] for k in KEYS)

    # --- sensitivity: +/-20% weight perturbations
    rng = np.random.default_rng(int(cfg.get("seed", 7)) if isinstance(cfg.get("seed"), int) else 7)
    W = np.array([w[k] for k in KEYS])
    S = np.array([[t["scores"][k] for k in KEYS] for t in tasks])
    base = S @ W
    rhos, flips, top_hits = [], [], defaultdict(int)
    n_draws = 1000
    top_k = min(5, len(tasks))
    base_top = set(np.argsort(-base)[:top_k])
    for _ in range(n_draws):
        wp = W * rng.uniform(0.8, 1.2, len(W))
        wp /= wp.sum()
        tot = S @ wp
        rhos.append(spearmanr(base, tot).statistic)
        flips.append(np.mean([(b >= T_ROBOT) != (x >= T_ROBOT) or (b >= T_HRC) != (x >= T_HRC) for b, x in zip(base, tot)]))
        for i in np.argsort(-tot)[:top_k]:
            top_hits[int(i)] += 1
    ranking = sorted(range(len(tasks)), key=lambda i: -base[i])
    for r_, i in enumerate(ranking):
        tasks[i]["rank"] = r_ + 1
        tasks[i]["top5_probability"] = round(top_hits[i] / n_draws, 3)
    tasks.sort(key=lambda t: t["rank"])

    groups = defaultdict(list)
    for t in tasks:
        if t["allocation"] in ("robot", "human_robot"):
            groups[(t["station"], t["category"])].append(t)
    group_list = []
    for (st, cat), ts_ in sorted(groups.items()):
        reqs = [t["requirements"] for t in ts_]
        group_list.append({
            "group_id": f"{st}:{cat}", "station": st, "category": cat, "element_keys": [t["element_key"] for t in ts_],
            "codes": [t["code"] for t in ts_], "allocations": {t["element_key"]: t["allocation"] for t in ts_},
            "requirements": {"payload_kg": max(r["payload_kg"] for r in reqs), "reach_mm": max(r["reach_mm"] for r in reqs),
                             "precision_mm": min(r["precision_mm"] for r in reqs),
                             "conveyor_tracking": any(r["conveyor_tracking"] for r in reqs),
                             "conveyor_speed_mm_s": reqs[0]["conveyor_speed_mm_s"],
                             "tool": reqs[0]["tool"], "time_budget_s": round(sum(r["time_budget_s"] for r in reqs), 3),
                             "work_points": reqs[0]["work_points"]}})

    out = {
        "method": "AHP-weighted multi-criteria automation-potential scoring (weighted sum of criterion scores in [0,1])",
        "criteria": [{"key": k, "weight": float(w[k]), "description": d, "source": s} for k, d, s in CRITERIA],
        "ahp": {"pairwise_matrix": A.round(4).tolist(), "order": KEYS, "lambda_max": res["lambda_max"],
                "ci": res["ci"], "ri": 1.41, "consistency_ratio": res["cr"],
                "judgement_rationale": "technical feasibility (IPA precondition) > economics (cycle-time share > "
                                       "repetition) > ergonomics (HRC review) > transport (solved by layout/AMR)"},
        "thresholds": {"robot": T_ROBOT, "human_robot": T_HRC,
                       "rule": "robot if total >= 0.60 and handling, force >= 0.5; human_robot if >= 0.45; else human; "
                               "WALK/SETUP -> layout_change; IDLE -> human (removed by balancing); GET_TOOL and "
                               "REACH_FASTENER follow the station's FASTEN allocation"},
        "assumptions": {"handling_by_code": HANDLING, "force_by_code": {k: v[0] for k, v in FORCE.items()},
                        "precision_mm_by_code": PRECISION, "eoat": [SCREWDRIVER_EOAT, GRIPPER_EOAT],
                        "robot_base_height_m": BASE_HEIGHT_M},
        "excluded": [{"element_key": f"{st}:UNOBS", "reason": "unobserved remainder: fixed manual block"}
                     for st in sorted({t['station'] for t in tasks})],
        "tasks": tasks,
        "groups": group_list,
        "sensitivity": {"method": "Monte-Carlo, each weight x U(0.8, 1.2), renormalised", "draws": n_draws,
                        "rank_stability": float(np.mean(rhos)), "rank_stability_min": float(np.min(rhos)),
                        "allocation_flip_rate": float(np.mean(flips)),
                        "baseline_top5": [tasks[i]["element_key"] for i in range(top_k)]},
    }
    for t in out["tasks"]:
        t["notes"] = t.pop("_notes")
    save_json(out, od / "automatability.json", contract="automatability")
    print(f"    s3: {len(tasks)} element keys, AHP CR={res['cr']:.3f}, rank stability={out['sensitivity']['rank_stability']:.3f}; "
          f"robot/HRC: {[t['element_key'] for t in tasks if t['allocation'] in ('robot', 'human_robot')]}")
