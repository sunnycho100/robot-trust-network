"""Generate contract-valid upstream fixtures for developing s5/s6b/s7/s8 before the real stages exist.

    .venv/bin/python src/rcd/s5_optimize/fixtures/make_fixtures.py

Writes ONLY under src/rcd/s5_optimize/fixtures/out/<stage>/ (never into out/). Every file carries "fixture": true.
Numbers are synthetic but shaped like the CarDA car-door line (3 paced stations, ~240 s takt, 40-60 s observed clips,
rest of the cycle unobserved). They are placeholders, not results.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from rcd.common.io import load_config, save_json  # noqa: E402

OUT = HERE / "out"
RNG = np.random.default_rng(7)
TAKT = 240.0

TAXONOMY = [
    ("WALK", "Walk", "Transport empty (TE)", "Walk between line side and door"),
    ("GET_PART", "Get part", "Grasp (G)", "Pick a part from cart / rack"),
    ("INSERT", "Insert / position", "Position + Assemble (P, A)", "Insert part into door retaining slots"),
    ("GET_TOOL", "Get tool", "Grasp (G)", "Take the screwdriver from the line side"),
    ("REACH_FASTENER", "Reach fastener", "Transport empty + Grasp (TE, G)", "Take a screw, place on bit"),
    ("FASTEN", "Fasten", "Use (U)", "Drive a screw at a designated point"),
    ("INSPECT", "Inspect", "Inspect (I)", "Visual check of the assembly"),
    ("IDLE", "Idle / wait", "Unavoidable delay (UD)", "Waiting for the conveyor / no hand motion"),
]

# per station: ordered list of (code, mean_s, cv, trunk_deg, subgoal); repeated codes interleave
STATIONS = {
    "WS10": [("WALK", 3.0, .25, 8, "1. Fit wiring harness"), ("GET_PART", 4.5, .20, 15, "1. Fit wiring harness"),
             ("INSERT", 11.0, .18, 38, "1. Fit wiring harness"), ("REACH_FASTENER", 1.6, .25, 25, "2. Fix harness"),
             ("FASTEN", 4.0, .15, 42, "2. Fix harness"), ("REACH_FASTENER", 1.6, .25, 25, "2. Fix harness"),
             ("FASTEN", 4.0, .15, 42, "2. Fix harness"), ("INSPECT", 4.0, .30, 30, "2. Fix harness"),
             ("IDLE", 2.0, .50, 5, "2. Fix harness")],
    "WS20": [("WALK", 4.5, .25, 6, "1. Install regulator"), ("GET_PART", 6.5, .20, 22, "1. Install regulator"),
             ("INSERT", 10.0, .18, 35, "1. Install regulator"), ("GET_TOOL", 2.8, .20, 10, "2. Fix regulator"),
             ("REACH_FASTENER", 1.7, .25, 28, "2. Fix regulator"), ("FASTEN", 3.6, .15, 45, "2. Fix regulator"),
             ("REACH_FASTENER", 1.7, .25, 28, "2. Fix regulator"), ("FASTEN", 3.6, .15, 45, "2. Fix regulator"),
             ("REACH_FASTENER", 1.7, .25, 28, "2. Fix regulator"), ("FASTEN", 3.6, .15, 45, "2. Fix regulator"),
             ("INSPECT", 3.0, .30, 25, "2. Fix regulator")],
    "WS30": [("GET_PART", 5.5, .20, 18, "1. Fixate speaker"), ("INSERT", 8.0, .18, 33, "1. Fixate speaker"),
             ("GET_TOOL", 3.0, .20, 12, "1. Fixate speaker"),
             ("REACH_FASTENER", 1.8, .25, 30, "1. Fixate speaker"), ("FASTEN", 4.2, .15, 48, "1. Fixate speaker"),
             ("REACH_FASTENER", 1.8, .25, 30, "1. Fixate speaker"), ("FASTEN", 4.2, .15, 48, "1. Fixate speaker"),
             ("REACH_FASTENER", 1.8, .25, 30, "1. Fixate speaker"), ("FASTEN", 4.2, .15, 48, "1. Fixate speaker"),
             ("REACH_FASTENER", 1.8, .25, 30, "1. Fixate speaker"), ("FASTEN", 4.2, .15, 48, "1. Fixate speaker"),
             ("WALK", 3.5, .25, 6, "1. Fixate speaker"), ("IDLE", 2.5, .50, 4, "1. Fixate speaker")],
}
CYCLE_IDS = ["002", "003", "004", "005"]


def fx(obj):
    obj["fixture"] = True
    return obj


def make_elements():
    cycles, erg = [], []
    for st, seq in STATIONS.items():
        for cid in CYCLE_IDS:
            t, els = 0.0, []
            for k, (code, m, cv, deg, sg) in enumerate(seq):
                d = float(max(0.3, RNG.normal(m, cv * m)))
                els.append({"id": f"{st}_{cid}_{k:02d}", "code": code, "start_s": round(t, 3),
                            "end_s": round(t + d, 3), "duration_s": round(d, 3), "source": "mocap_rules",
                            "confidence": 0.85, "hands": "both", "travel_m": 1.5 if code == "WALK" else 0.2,
                            "trunk_flex_deg_mean": float(deg + RNG.normal(0, 3)), "subgoal": sg})
                t += d
            cycles.append({"station": st, "cycle_id": cid, "duration_s": round(t, 3), "elements": els})
            bent = sum(e["duration_s"] for e in els if 20 <= e["trunk_flex_deg_mean"] < 60)
            erg.append({"station": st, "cycle_id": cid,
                        "postures_s": {"upright": round(t - bent, 2), "bent_forward_20_60": round(bent, 2),
                                       "strongly_bent_60": 0.0, "elbow_at_shoulder": 1.0}})
    elements = fx({"taxonomy": [{"code": c, "name": n, "therblig": th, "description": d} for c, n, th, d in TAXONOMY]
                   + [{"code": "UNOBS", "name": "Unobserved manual work", "therblig": "-",
                       "description": "Part of the 240 s cycle outside the cropped clip (manual block)"}],
                   "cycles": cycles})
    ergonomics = fx({"method": "EAWS-style trunk flexion bands from mocap (fixture)", "cycles": erg,
                     "validation": {"metric": "spearman", "value": 0.7, "n": 8,
                                    "pairs": [[20 + i, 21 + i * 0.9] for i in range(8)]}})
    return elements, ergonomics


def make_time_study(elements, cfg):
    allow = cfg["time_study"]["allowance_pfd"]
    z, a = 1.959963984540054, cfg["time_study"]["accuracy"]
    rows, ststats, spc = [], [], []
    for st in STATIONS:
        cyc = [c for c in elements["cycles"] if c["station"] == st]
        codes = sorted({e["code"] for c in cyc for e in c["elements"]})
        tot_obs = []
        for code in codes:
            per = np.array([sum(e["duration_s"] for e in c["elements"] if e["code"] == code) for c in cyc])
            m, s = float(per.mean()), float(per.std(ddof=1))
            rows.append({"station": st, "code": code, "n": len(per), "mean_s": m, "std_s": s, "normal_s": m,
                         "standard_s": m * (1 + allow),
                         "n_required": int(math.ceil((z * s / (a * m)) ** 2)) if m > 0 else 0})
        obs = np.array([c["duration_s"] for c in cyc])
        unobs = TAKT - float(obs.mean())
        rows.append({"station": st, "code": "UNOBS", "n": len(obs), "mean_s": unobs, "std_s": 0.0,
                     "normal_s": unobs, "standard_s": unobs * (1 + allow), "n_required": 0})
        ststats.append({"station": st, "n_cycles": len(obs), "observed_mean_s": float(obs.mean()),
                        "standard_s": float(obs.mean()) * (1 + allow) + unobs, "takt_s": TAKT,
                        "unobserved_s": unobs})
        mr = np.abs(np.diff(obs)).mean()
        x = obs + unobs
        spc.append({"station": st, "observations_s": x.tolist(), "ucl_s": float(x.mean() + 2.66 * mr),
                    "usl_s": TAKT, "cpu": float((TAKT - x.mean()) / (3 * mr / 1.128))})
    return fx({"params": {"performance_rating": 1.0, "allowance_pfd": allow, "confidence": 0.95, "accuracy": a},
               "element_stats": rows, "station_stats": ststats, "spc": spc,
               "rating": {"method": "Westinghouse", "factors": {"skill": 0.03, "effort": 0.02, "conditions": 0.0,
                                                                "consistency": -0.05}, "value": 1.0},
               "allowance_breakdown": {"personal": 0.05, "basic_fatigue": 0.04, "standing": 0.02,
                                       "abnormal_posture": 0.02, "delay": 0.02},
               "pmts": {"system": "MTM-UAS", "rows": [
                   {"code": c, "tmu": t, "pmts_s": t * 0.036, "observed_s": o}
                   for c, t, o in [("GET_PART", 120, 5.4), ("INSERT", 210, 9.6), ("FASTEN", 105, 4.1),
                                   ("REACH_FASTENER", 45, 1.7), ("GET_TOOL", 70, 2.9)]]},
               "lean": {"rows": [{"code": "INSERT", "class": "VA", "waste": None},
                                 {"code": "FASTEN", "class": "VA", "waste": None},
                                 {"code": "GET_PART", "class": "NNVA", "waste": None},
                                 {"code": "REACH_FASTENER", "class": "NNVA", "waste": None},
                                 {"code": "GET_TOOL", "class": "NNVA", "waste": None},
                                 {"code": "INSPECT", "class": "NNVA", "waste": None},
                                 {"code": "WALK", "class": "NVA", "waste": "Motion"},
                                 {"code": "IDLE", "class": "NVA", "waste": "Waiting"}], "va_ratio": 0.52}})


S3_RULES = {  # code -> (allocation, category)
    "FASTEN": ("robot", "cobot_screwdriving"), "REACH_FASTENER": ("robot", "cobot_screwdriving"),
    "GET_TOOL": ("robot", "cobot_screwdriving"), "GET_PART": ("human_robot", "cobot_pick_place"),
    "INSPECT": ("robot", "vision_inspection"), "INSERT": ("human", "dexterous_manual"),
    "WALK": ("layout_change", "layout"), "IDLE": ("human", "waiting"), "UNOBS": ("human", "unobserved"),
}
CRIT = [("repeatability", 0.22), ("low_dexterity", 0.20), ("part_presentation", 0.15), ("ergonomic_load", 0.15),
        ("cycle_share", 0.13), ("safety", 0.15)]


def make_automatability(ts):
    tasks = []
    w = dict(CRIT)
    for r in ts["element_stats"]:
        alloc, cat = S3_RULES[r["code"]]
        base = {"robot": 0.8, "human_robot": 0.6, "human": 0.3, "layout_change": 0.4}[alloc]
        sc = {k: float(np.clip(base + RNG.normal(0, 0.08), 0, 1)) for k in w}
        if r["code"] in ("UNOBS", "IDLE"):
            sc = {k: 0.0 for k in w}
        tasks.append({"station": r["station"], "code": r["code"], "element_key": f"{r['station']}:{r['code']}",
                      "scores": sc, "total": sum(w[k] * sc[k] for k in w), "category": cat,
                      "rationale": f"fixture rule for {r['code']}", "allocation": alloc})
    return fx({"criteria": [{"key": k, "weight": v, "description": k.replace("_", " "),
                             "source": "Fixture (AHP pairwise weights)"} for k, v in CRIT],
               "ahp": {"consistency_ratio": 0.04}, "tasks": tasks, "sensitivity": {"rank_stability": 0.9}})


ROBOTS = [  # model, maker, payload, reach, price lo, hi, tracking
    ("UR10e", "Universal Robots", 12.5, 1300, 45000, 55000, True),
    ("CRX-10iA/L", "FANUC", 10, 1418, 42000, 52000, True),
    ("GoFa CRB 15000", "ABB", 5, 950, 40000, 55000, True),
    ("LBR iisy 11", "KUKA", 11, 1300, 38000, 48000, True),
    ("M1013", "Doosan Robotics", 10, 1300, 38000, 48000, True),
    ("TM12", "Techman", 12, 1300, 35000, 45000, True),
    ("HC10DTP", "Yaskawa", 10, 1200, 40000, 50000, True),
    ("FR3", "Franka Robotics", 3, 855, 25000, 32000, False),
    ("UR5e", "Universal Robots", 5, 850, 30000, 38000, True),
    ("CRX-20iA/L", "FANUC", 20, 1418, 50000, 62000, True),
]


def make_catalog():
    return fx({"robots": [{"model": m, "maker": mk, "type": "cobot", "payload_kg": p, "reach_mm": r,
                           "repeatability_mm": 0.05, "weight_kg": 30, "dof": 6, "joint_speed_deg_s": 180,
                           "collaborative": True, "conveyor_tracking": tr, "price_usd_range": [lo, hi],
                           "sim_model": "mujoco_menagerie" if m in ("UR10e", "UR5e", "FR3") else "",
                           "source_urls": ["https://example.com/fixture-spec"]}
                          for m, mk, p, r, lo, hi, tr in ROBOTS]})


GROUPS = [("WS10", "cobot_screwdriving", ["REACH_FASTENER", "FASTEN"], 2),
          ("WS10", "vision_inspection", ["INSPECT"], 0),
          ("WS20", "cobot_screwdriving", ["GET_TOOL", "REACH_FASTENER", "FASTEN"], 3),
          ("WS20", "cobot_pick_place", ["GET_PART"], 0),
          ("WS30", "cobot_screwdriving", ["GET_TOOL", "REACH_FASTENER", "FASTEN"], 4)]
PASS = {"cobot_screwdriving": ["UR10e", "CRX-10iA/L", "TM12"], "vision_inspection": ["UR5e", "GoFa CRB 15000"],
        "cobot_pick_place": ["UR10e", "CRX-10iA/L"]}


def make_candidates():
    tgs = []
    for st, cat, codes, nscrew in GROUPS:
        ev = []
        for m, *_ in ROBOTS:
            ok = m in PASS[cat]
            cs = {"K_mm_s": 1600, "T_s": 0.3, "C_mm": 850, "separation_mm": 1600 * 0.3 + 850,
                  "footprint_m2": 4.0, "aisle_allowance": 0.3, "total_m2": 5.2} if ok else None
            ev.append({"model": m, "passed": ok, "reasons": ["payload/reach/tracking ok" if ok else "fails reach"],
                       "reachability": 0.95 if ok else None, "cell_space": cs, "topsis": 0.6 if ok else None})
        tgs.append({"group_id": f"{st}:{cat}", "station": st, "codes": codes, "category": cat,
                    "requirements": {"payload_kg": 4.0, "reach_mm": 900, "conveyor_tracking": True,
                                     "n_screws": nscrew}, "evaluated": ev})
    return fx({"task_groups": tgs, "ranking": {"method": "AHP+TOPSIS", "consistency_ratio": 0.05}})


def make_task_times(cand):
    ent = []
    for g in cand["task_groups"]:
        n = g["requirements"]["n_screws"]
        for e in g["evaluated"]:
            if not e["passed"]:
                continue
            spd = {"UR10e": 1.0, "CRX-10iA/L": 0.95, "TM12": 1.08, "UR5e": 0.9, "GoFa CRB 15000": 0.92}[e["model"]]
            if "screw" in g["category"]:
                br = {"tool_change": 2.0 if "GET_TOOL" in g["codes"] else 0.0, "approach": 1.6 * n,
                      "screw_feed": 1.2 * n, "drive": 2.8 * n, "retract": 0.8 * n}
            elif "inspect" in g["category"]:
                br = {"approach": 2.0, "capture": 1.5, "retract": 1.5}
            else:
                br = {"approach": 2.5, "grasp": 1.0, "transfer": 3.5, "release": 0.8}
            br = {k: v * spd for k, v in br.items()}
            ent.append({"model": e["model"], "group_id": g["group_id"], "station": g["station"],
                        "cycle_time_s": sum(br.values()), "breakdown": br, "ik_success_rate": 0.97,
                        "sim": "MuJoCo IK (fixture)"})
    return fx({"method": "MuJoCo IK + trapezoidal joint profiles (fixture)", "entries": ent})


def make_calibration_scene():
    cams = []
    for st in STATIONS:
        for side in ("in", "out"):
            cams.append({"id": f"{st}_{side}", "station": st, "side": side, "image_size": [1920, 1080],
                         "K": [[1400, 0, 960], [0, 1400, 540], [0, 0, 1]], "dist": [0, 0, 0, 0, 0],
                         "R": np.eye(3).tolist(), "t": [0, 0, 3.0], "source": "fixture",
                         "reproj_rmse_px": 1.2})
    cal = fx({"world_frame": {"units": "m", "up_axis": "z", "origin_desc": "fixture"}, "cameras": cams})
    sg = fx({"units": "m", "stations": [{"station": st, "zones": [{"name": "work", "polygon_xy": [[0, 0], [2, 0], [2, 2]]}],
                                         "cycles": [{"cycle_id": c, "duration_s": 50.0,
                                                     "video": {"in": "", "out": "", "fps": 30},
                                                     "human": {"joints_npz": "", "joint_names": [], "fps": 60,
                                                               "source": "fixture"},
                                                     "conveyor": {"speed_mm_s": 60.0, "method": "fixture"},
                                                     "objects": []} for c in CYCLE_IDS]}
                                        for st in STATIONS]})
    return cal, sg


def make_task_plan():
    plans = []
    for st, cat, codes, n in GROUPS:
        if "screw" not in cat:
            continue
        steps = [{"primitive": "move_to", "covers": []}]
        if "GET_TOOL" in codes:
            steps.append({"primitive": "tool_change", "covers": [f"{st}:GET_TOOL"]})
        for k in range(n):
            steps += [{"primitive": "pick_screw", "covers": [f"{st}:REACH_FASTENER"]},
                      {"primitive": "screw", "covers": [f"{st}:FASTEN"]}]
        xml = "<root><BehaviorTree ID='main'><Sequence>" + "".join(
            f"<Action ID='{s['primitive']}'/>" for s in steps) + "</Sequence></BehaviorTree></root>"
        plans.append({"station": st, "model": "UR10e", "steps": steps, "bt_xml": xml})
    return fx({"primitives": ["move_to", "tool_change", "pick_screw", "screw", "pick", "place", "inspect"],
               "plans": plans})


def main():
    cfg = load_config()
    el, erg = make_elements()
    ts = make_time_study(el, cfg)
    au = make_automatability(ts)
    cat = make_catalog()
    cand = make_candidates()
    rt = make_task_times(cand)
    cal, sg = make_calibration_scene()
    tp = make_task_plan()
    for stage, fn, obj, con in [("s0", "calibration.json", cal, "calibration"), ("s1", "scene_graph.json", sg, "scene_graph"),
                                ("s2", "elements.json", el, "elements"), ("s2", "ergonomics.json", erg, "ergonomics"),
                                ("s2", "time_study.json", ts, "time_study"),
                                ("s3", "automatability.json", au, "automatability"),
                                ("s4", "robot_catalog.json", cat, "robot_catalog"),
                                ("s4", "candidates.json", cand, "candidates"),
                                ("s6a", "robot_task_times.json", rt, "robot_task_times"),
                                ("s9", "task_plan.json", tp, "task_plan")]:
        save_json(obj, OUT / stage / fn, contract=con)
    print("fixtures written to", OUT)


if __name__ == "__main__":
    main()
