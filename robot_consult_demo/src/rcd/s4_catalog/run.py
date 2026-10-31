"""s4: commercial robot catalog, candidate filtering per task group, AHP+TOPSIS ranking, cell space.

Inputs (real out/ when present, otherwise the local fixture in s4_catalog/fixtures/):
  s3/automatability.json  -> task groups = element keys with allocation robot/human_robot, grouped "<station>:<category>"
  s1/scene_graph.json     -> work points (workpiece_contact), worker zone, conveyor speed/direction
Outputs: s4/robot_catalog.json, s4/candidates.json

Requirements per group (raw, the gate applies the margins from config):
  payload_kg = end-effector + adapter + part mass;  reach_mm = min-max shoulder-to-flange distance from the base
  placement search (scene.place_base);  conveyor_tracking = paced moving line (or s3 flag);  precision_mm.
A model passes if payload >= payload_kg*(1+margin), reach*reach_margin >= reach_mm, tracking if required,
collaborative (the station keeps a human), repeatability <= precision, dof >= 6.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from rcd.common.io import out_dir, save_json

from . import mcdm
from . import process as P
from . import scene as S

HERE = Path(__file__).resolve().parent


def load_catalog() -> dict:
    return json.loads((HERE / "catalog_data.json").read_text())


def price_mid(r) -> float:
    return float(np.mean(r["price_usd_range"]))


def cell_space(r: dict, group: dict) -> dict:
    iso = P.ISO13855
    stop = r.get("stop_time_s")
    stop_src = r.get("stop_time_note", "")
    if stop is None:
        stop, stop_src = iso["default_stop_s"], "maker does not publish stop time -> default assumption 0.50 s"
    T = round(stop + iso["scanner_response_s"], 4)
    C = max(850.0, 1200.0 - 0.4 * iso["H_mm"])
    K = iso["K_mm_s"]
    S_mm = K * T + C
    R_env = r["reach_mm"] / 1000.0 + group["tool"]["tcp_offset_m"]
    envelope = 0.5 * math.pi * R_env ** 2  # robot-side half of the swept envelope (other half overhangs the conveyor)
    foot = P.PEDESTAL_M2 + P.PRESENTER_M2 + envelope
    track_m2 = 0.0
    if group.get("mounting") == "linear_track":
        track_m2 = (S.tracking_window_m(group) + P.PEDESTAL_M2 ** 0.5) * P.TRACK_WIDTH_M
        foot += track_m2
    aisle, aisle_src = P.aisle_allowance()
    R_pf = R_env + S_mm / 1000.0
    return {
        "K_mm_s": K, "T_s": T, "C_mm": C, "separation_mm": round(K * T + C, 3),
        "T_parts": {"robot_stop_s": stop, "robot_stop_source": stop_src, "scanner_response_s": iso["scanner_response_s"],
                    "scanner_source": iso["scanner_source"]},
        "C_basis": f"C = 1200 - 0.4*H with H = {iso['H_mm']:.0f} mm (>= 850 mm)",
        "footprint_m2": round(foot, 4),
        "footprint_parts_m2": {"pedestal": P.PEDESTAL_M2, "presenter_or_tray": P.PRESENTER_M2,
                               "envelope_half_disc": round(envelope, 4), "envelope_radius_m": round(R_env, 3),
                               "linear_track": round(track_m2, 4)},
        "aisle_allowance": aisle, "aisle_source": aisle_src,
        "total_m2": round(foot * (1 + aisle), 4),
        "protective_field_radius_m": round(R_pf, 3),
        "protective_field_note": "Scanner protective field = envelope radius + S. It overlaps the shared worker/aisle "
                                 "area (speed & separation monitoring slows/stops the robot), so it is not counted as "
                                 "dedicated floor area.",
        "source": iso["source"],
    }


def evaluate(r: dict, g: dict, req: dict, place: dict, cfg: dict) -> dict:
    m = cfg["robots"]
    reasons, ok = [], True
    need_p = req["payload_kg"] * (1 + m["safety_margin_payload"])
    good = r["payload_kg"] >= need_p - 1e-9
    ok &= good
    reasons.append(f"payload {r['payload_kg']} kg {'>=' if good else '<'} {need_p:.2f} kg "
                   f"(tool+part {req['payload_kg']:.3f} kg x {1 + m['safety_margin_payload']:.2f})")
    usable = r["reach_mm"] * m["reach_margin"]
    good = usable >= req["reach_mm"] - 1e-9
    ok &= good
    reasons.append(f"usable reach {usable:.0f} mm ({r['reach_mm']} x {m['reach_margin']}) {'>=' if good else '<'} "
                   f"required {req['reach_mm']:.0f} mm")
    if req["conveyor_tracking"]:
        good = bool(r["conveyor_tracking"])
        ok &= good
        reasons.append(("conveyor tracking: " + r["conveyor_tracking_feature"]) if good else
                       ("no documented conveyor tracking (" + r["conveyor_tracking_feature"] + ")"))
    if req["collaborative"]:
        good = bool(r["collaborative"])
        ok &= good
        reasons.append("collaborative (PFL) robot" if good else "not certified collaborative; station keeps a human")
    good = r["repeatability_mm"] <= req["precision_mm"]
    ok &= good
    reasons.append(f"repeatability +/-{r['repeatability_mm']} mm {'<=' if good else '>'} required +/-{req['precision_mm']} mm")
    good = r["dof"] >= 6
    ok &= good
    if not good:
        reasons.append(f"{r['dof']} DOF < 6 (tool orientation)")
    if r.get("sim_model"):
        reasons.append(f"kinematic model for simulation: {r['sim_model']}")
    else:
        ok = False
        reasons.append("no verifiable kinematic model in this study (no Menagerie model; vendor description not "
                       "transcribed/verified) -> cannot be simulated, not offered as a purchasable candidate")
    # reachability at the chosen base (geometric sphere test with the generic shoulder offset)
    F = S.flange_points(g, place["base_xyz"])
    sh = np.asarray(place["shoulder_xyz"], float)
    d = np.linalg.norm(F - sh, axis=1)
    frac = float(np.mean((d <= usable / 1000.0 + 1e-9) & (d >= P.MIN_REACH_M)))
    reach = {"base_xyz": place["base_xyz"], "yaw_rad": place["yaw_rad"], "pedestal_height_m": place["pedestal_height_m"],
             "n_poses": int(len(F)), "max_dist_mm": round(float(d.max()) * 1000, 1), "usable_reach_mm": round(usable, 1),
             "margin_mm": round(usable - float(d.max()) * 1000, 1), "fraction_within_usable_reach": round(frac, 4),
             "method": "geometric: flange points (tool offset, tracking window, tray) inside the usable-reach sphere "
                       "around the shoulder; full IK is done in s6a"}
    out = {"model": r["model"], "passed": bool(ok), "reasons": reasons, "reachability": reach,
           "simulate": bool(ok and r.get("sim_model")), "cell_space": cell_space(r, g) if ok else None}
    return out


def run(cfg: dict) -> None:
    prov: dict = {}
    au = S.load_upstream(cfg, "s3", "automatability.json", prov)
    sg = S.load_upstream(cfg, "s1", "scene_graph.json", prov)
    if au is None:
        raise FileNotFoundError("s3/automatability.json missing (no fixture either)")
    cat = load_catalog()
    robots = cat["robots"]
    for r in robots:
        r["ecosystem_score"] = mcdm.ECOSYSTEM.get(r["maker"], 2)
    od = out_dir(cfg, "s4")
    save_json({"robots": robots, "about": cat["_about"],
               "ecosystem_score_note": "analyst ordinal assumption (1-5) used only as an MCDM criterion",
               "process_assumptions": P.assumptions_table()},
              od / "robot_catalog.json", contract="robot_catalog")

    groups = S.build_groups(au, sg, cfg)
    A = mcdm.ahp()
    w = np.array([A["weights"][c[0]] for c in mcdm.CRITERIA])
    benefit = np.array([c[1] == "benefit" for c in mcdm.CRITERIA])
    task_groups, scores = [], {}
    usable = sorted(r["reach_mm"] * cfg["robots"]["reach_margin"] for r in robots
                    if r["collaborative"] and r["conveyor_tracking"])
    reach_cap = float(np.median(usable))
    for g in groups:
        place = S.place_base(g)
        if place["required_reach_mm"] > reach_cap:
            fixed = place
            g["mounting"] = "linear_track"
            place = S.place_base(g)
            place["mounting_reason"] = (
                f"fixed floor base needs {fixed['required_reach_mm']:.0f} mm (carrier travels {fixed['tracking_window_m']:.2f} m "
                f"during the ~{g['program_est_s']:.0f} s program) > {reach_cap:.0f} mm = median usable reach of the "
                f"collaborative tracking-capable catalog models -> 7th-axis linear track synchronised to the conveyor "
                f"(track cost not in robot price; quote in RFQ)")
            place["fixed_floor_alternative"] = {k: fixed[k] for k in ("base_xyz", "required_reach_mm", "tracking_window_m")}
        else:
            place["mounting_reason"] = (f"fixed floor base: required reach {place['required_reach_mm']:.0f} mm <= "
                                        f"{reach_cap:.0f} mm (median usable reach, collaborative + tracking models)")
        req = {
            "payload_kg": g["payload_kg"],
            "payload_basis": g["payload_source"] + "; safety margin applied at the check",
            "reach_mm": place["required_reach_mm"],
            "reach_basis": place["method"],
            "conveyor_tracking": g["conveyor_tracking"],
            "collaborative": True,
            "precision_mm": g["precision_mm"], "precision_basis": g["precision_source"],
            "n_targets": g["n_targets"],
            "s3_requirements": g["s3_requirements"],
            "reach_note": "s4 required reach is from a fixed floor base OFF the belt (outside s1 conveyor/worker/line-side "
                          "zones) incl. tool offset and the conveyor travel during the robot cycle; s3 reach_mm assumed a "
                          "base beside the operator's standing point",
        }
        ev = [evaluate(r, g, req, place, cfg) for r in robots]
        passed = [e for e in ev if e["passed"]]
        if passed:
            by = {r["model"]: r for r in robots}
            X = []
            for e in passed:
                r = by[e["model"]]
                X.append([price_mid(r), float(bool(r["conveyor_tracking"])),
                          (r["reach_mm"] * cfg["robots"]["reach_margin"] - req["reach_mm"]) / req["reach_mm"],
                          float(np.mean(r["joint_speed_deg_s"][:3])), float(r["ecosystem_score"]),
                          r["payload_kg"] / (req["payload_kg"] * (1 + cfg["robots"]["safety_margin_payload"])) - 1,
                          e["cell_space"]["total_m2"], r["repeatability_mm"]])
            X = np.array(X)
            cc = mcdm.topsis(X, w, benefit)
            order = np.argsort(-cc)
            rank = {passed[i]["model"]: k + 1 for k, i in enumerate(order)}
            for e, x, c in zip(passed, X, cc):
                e["topsis"] = round(float(c), 4)
                e["rank"] = rank[e["model"]]
                e["criteria_values"] = {k[0]: round(float(v), 4) for k, v in zip(mcdm.CRITERIA, x)}
            scores[g["group_id"]] = [{"model": passed[i]["model"], "topsis": round(float(cc[i]), 4), "rank": k + 1}
                                     for k, i in enumerate(order)]
        for e in ev:
            e.setdefault("topsis", None)
        task_groups.append({
            "group_id": g["group_id"], "station": g["station"], "codes": g["codes"], "category": g["category"],
            "process_type": g["process_type"], "element_keys": g["element_keys"], "allocations": g["allocations"],
            "requirements": req, "tool": g["tool"], "placement": place, "mounting": g.get("mounting", "fixed_floor"),
            "geometry": {**g["geometry"], "targets_m": g["targets_m"], "program_est_s": g["program_est_s"]},
            "evaluated": ev,
        })
        print(f"    {g['group_id']}: reach {req['reach_mm']:.0f} mm, payload {req['payload_kg']:.2f} kg, "
              f"passed {[e['model'] for e in passed]}")
    out = {"task_groups": task_groups,
           "ranking": {"method": "AHP+TOPSIS", "consistency_ratio": A["consistency_ratio"], "weights": A["weights"],
                       "criteria": A["criteria"], "pairwise_matrix": A["pairwise_matrix"], "lambda_max": A["lambda_max"],
                       "random_index": A["random_index"], "scores": scores,
                       "notes": "AHP: principal-eigenvector weights, CR = CI/RI (Saaty). TOPSIS over passed models of "
                                "each group, vector normalisation; cost criteria: price, footprint, repeatability."},
           "inputs": prov,
           "config_used": {"safety_margin_payload": cfg["robots"]["safety_margin_payload"],
                           "reach_margin": cfg["robots"]["reach_margin"]}}
    save_json(out, od / "candidates.json", contract="candidates")
