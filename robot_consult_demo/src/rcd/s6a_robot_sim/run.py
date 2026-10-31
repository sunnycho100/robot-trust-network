"""s6a: robot cycle times per (passed model, task group) from a MuJoCo kinematic cell simulation with IK.

Inputs: out/s4/candidates.json (groups, requirements, base placement, passed models), out/s4/robot_catalog.json.
Output: out/s6a/robot_task_times.json

Every passed model is simulated on its OWN kinematics:
* sim_model "xxx_mj_description": MuJoCo Menagerie MJCF (robot_descriptions);
* sim_model "vendor_urdf:<stem>": MJCF built by models/build.py from the vendor's public ROS URDF/xacro, transcribed
  into models/<stem>.json (joint origins, axes, limits; source URL + license in the file). A reach check (max
  flange-to-J2 distance over random configurations vs. datasheet reach) is attached to each entry.
s4 does not pass models without a kinematic model, so there are no surrogate or scaled estimates.
See sim.py for the IK and timing model, s4_catalog/process.py for process times and their sources.
"""
from __future__ import annotations

import time

import numpy as np

from rcd.common.io import ROOT, load_json, out_dir, save_json
from rcd.s4_catalog import process as P

from . import sim
from .models import build

METHOD = ("MuJoCo 3 kinematic cell simulation (Menagerie MJCF, or MJCF built from the vendor ROS URDF) with damped-least-squares IK on "
          "MuJoCo site Jacobians (5-D: TCP position + tool axis) for every approach/contact/retract waypoint; "
          "conveyor tracking by re-solving IK at the carrier position at each arrival time; joint-space trapezoidal "
          "profiles with datasheet per-joint speed limits and a = 1.4 rad/s^2, collaborative TCP cap 1.0 m/s; "
          "process times (screw run-down from OnRobot rpm, gripper stroke/speed, tracking sync) from s4 process.py")


def _group_for_sim(g: dict) -> dict:
    geo = g["geometry"]
    return {"group_id": g["group_id"], "geometry": geo, "targets_m": geo["targets_m"], "process_type": g["process_type"],
            "tool": g["tool"], "program_est_s": geo["program_est_s"], "mounting": g.get("mounting", "fixed_floor"),
            "n_targets": len(geo["targets_m"]), "codes": g["codes"]}


def _round_breakdown(br: dict, total: float) -> dict:
    out = {k: round(v, 3) for k, v in br.items()}
    k = max(out, key=out.get)
    out[k] = round(out[k] + (round(total, 3) - sum(out.values())), 3)
    return out


def run(cfg: dict) -> None:
    t0 = time.time()
    s4 = ROOT / cfg["out_root"] / "s4"
    cand = load_json(s4 / "candidates.json", contract="candidates")
    cat = {r["model"]: r for r in load_json(s4 / "robot_catalog.json", contract="robot_catalog")["robots"]}
    takt = cfg["line"]["takt_s"]
    entries = []
    kin_checks: dict = {}
    for g in cand["task_groups"]:
        G = _group_for_sim(g)
        pl = g["placement"]
        for e in g["evaluated"]:
            if not e["passed"]:
                continue
            r = cat[e["model"]]
            desc = r.get("sim_model")
            if not desc:  # s4 never passes such a model; guard anyway
                raise ValueError(f"{r['model']} passed s4 without a kinematic model")
            if desc.startswith(build.PREFIX):
                stem = desc[len(build.PREFIX):]
                kin = build.load(stem)
                if stem not in kin_checks:
                    kin_checks[stem] = build.reach_check(stem)
                sim_txt = (f"MuJoCo IK on MJCF built from the vendor ROS description ({kin['source']}; "
                           f"{kin['license']}); kinematics only, capsule collision primitives; datasheet joint speeds")
                kind = "vendor_urdf"
                kcheck = kin_checks[stem]
            else:
                sim_txt = f"MuJoCo IK on MuJoCo Menagerie model '{desc}' (robot_descriptions); datasheet joint speeds"
                kind = "menagerie"
                kcheck = None
            speeds = r["joint_speed_deg_s"]
            cell = sim.Cell(desc, pl["base_xyz"], pl["yaw_rad"], G, g["tool"]["tcp_offset_m"])
            res = sim.simulate(cell, G, speeds, pl["base_xyz"], t_mid=G["program_est_s"] / 2)
            res = sim.simulate(cell, G, speeds, pl["base_xyz"], t_mid=res["cycle_time_s"] / 2)  # centre the window
            st = res["stats"]
            T = round(res["cycle_time_s"], 3)
            ent = {
                "model": r["model"], "group_id": g["group_id"], "station": g["station"], "cycle_time_s": T,
                "breakdown": _round_breakdown(res["breakdown"], T),
                "breakdown_by_code": _round_breakdown(res["by_code"], T),
                "ik_success_rate": round(st["ok"] / max(1, st["attempts"]), 4),
                "ik_waypoints": st["attempts"],
                "ik_pos_err_mm_max": round(float(max(st["pos_err_mm"])) if st["pos_err_mm"] else 0.0, 3),
                "collision_free_rate": round(1 - st["collisions"] / max(1, st["ok"]), 4),
                "joint_speed_violations": st["speed_violations"],
                "sim": sim_txt, "sim_kind": kind, "kinematics_check": kcheck,
                "mounting": G["mounting"], "base_xyz": pl["base_xyz"], "n_targets": G["n_targets"],
                "joint_speed_deg_s": speeds,
                "human_time_budget_s": (g["requirements"].get("s3_requirements") or {}).get("time_budget_s"),
                "within_takt": T <= takt,
                "timeline": res["timeline"],
            }
            entries.append(ent)
            print(f"    {g['group_id']:<26} {r['model']:<14} {kind:<12} cycle {T:6.1f}s  IK {ent['ik_success_rate']:.0%} "
                  f"({st['attempts']} wp)  coll-free {ent['collision_free_rate']:.0%}")
    out = {"method": METHOD, "entries": entries,
           "assumptions": P.assumptions_table(),
           "notes": ["breakdown keys are motion/process components; breakdown_by_code splits the same cycle by the s2 "
                     "element code each program segment realises (for s5).",
                     "ik_success_rate = converged IK waypoints / all waypoints (TCP < 0.5 mm, tool axis < 0.5 deg, joint "
                     "limits); collision_free_rate checks robot vs panel/tray/pedestal at converged waypoints only.",
                     "Robot payload dynamics, torque limits and PFL speed reductions near the operator are not modelled."],
           "runtime_s": round(time.time() - t0, 2)}
    save_json(out, out_dir(cfg, "s6a") / "robot_task_times.json", contract="robot_task_times")
