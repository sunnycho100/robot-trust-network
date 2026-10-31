"""Simulation hand-off package: turns the recommended scenario into runnable, per-station simulation inputs.

For every station with a robot in the recommended s5 scenario this writes out/sim_package/<ST>/:
  scene.xml               MuJoCo MJCF (belt frame): belt, carrier + door panel, target sites, linear-track carriage
                          with pedestal / tray / robot (vendor kinematics) / tool, worker as 22 mocap capsules, zones
  targets.json            every robot target: id, operation, position (belt frame at t_ref), approach axis, order
  robot_reference.json    robot program timeline (from s6a) + joint-space reference waypoints from IK
  human_asis.npz          observed worker motion (s1, belt frame) of the representative cycle
  human_tobe.npz/.json    to-be worker motion: the observed element segments the worker keeps (s5), spliced in
                          order, WALK time-compressed by the kitting lever; UNOBS held as a documented gap
  motion.json             conveyor / frame conventions, station timing (takt, robot start interlock)
And at the package root: scenario_matrix.yaml, replay.py (stand-alone baseline harness).

Frames: everything is expressed in the BELT frame = plant frame at the cycle's t_ref (s1). The door, the worker
(who stands on the belt) and the tracking robot carriage all move with the belt, so in this frame they are static
relative to each other; world pose at time t is x_world = x_belt + v * d * (t - t_ref).
"""
from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path

import mujoco
import numpy as np
import yaml

from rcd.common.io import ROOT, load_json, out_dir
from rcd.s4_catalog import process as P
from rcd.s4_catalog import scene as S
from rcd.s6a_robot_sim import sim
from rcd.s6a_robot_sim.models import build
from rcd.s6a_robot_sim.run import _group_for_sim

HERE = Path(__file__).resolve().parent
TOOL_CHANGE_S = 5.0  # same value s5 uses (OnRobot "tool change in under 5 seconds")
BONES = [("Hips", "Chest"), ("Chest", "Chest2"), ("Chest2", "Chest3"), ("Chest3", "Chest4"), ("Chest4", "Neck"),
         ("Neck", "Head"), ("Chest4", "RightCollar"), ("RightCollar", "RightShoulder"), ("RightShoulder", "RightElbow"),
         ("RightElbow", "RightWrist"), ("Chest4", "LeftCollar"), ("LeftCollar", "LeftShoulder"),
         ("LeftShoulder", "LeftElbow"), ("LeftElbow", "LeftWrist"), ("Hips", "RightHip"), ("RightHip", "RightKnee"),
         ("RightKnee", "RightAnkle"), ("RightAnkle", "RightToe"), ("Hips", "LeftHip"), ("LeftHip", "LeftKnee"),
         ("LeftKnee", "LeftAnkle"), ("LeftAnkle", "LeftToe")]
BONE_RADIUS = {"Hips": 0.12, "Chest": 0.13, "Chest2": 0.13, "Chest3": 0.13, "Chest4": 0.12, "Neck": 0.06}
DEFAULT_BONE_RADIUS = 0.05
HEAD_RADIUS = 0.11
WALK_REDUCTION_KEY = "reduction"


# ----------------------------------------------------------------------------------------------- inputs
def _inputs(cfg):
    o = ROOT / cfg["out_root"]
    return dict(
        alloc=load_json(o / "s5" / "allocation.json"), cand=load_json(o / "s4" / "candidates.json"),
        cat={r["model"]: r for r in load_json(o / "s4" / "robot_catalog.json")["robots"]},
        rt=load_json(o / "s6a" / "robot_task_times.json"), sg=load_json(o / "s1" / "scene_graph.json"),
        el=load_json(o / "s2" / "elements.json"), ts=load_json(o / "s2" / "time_study.json"),
        plan=load_json(o / "s9" / "task_plan.json"))


def _rep_cycle(group: dict) -> str:
    m = re.search(r"cycle (\d{3}_\d{4})", group["geometry"]["source"])
    return m.group(1)


# ----------------------------------------------------------------------------------------------- robot
def _robot_reference(desc, groups, base, yaw):
    """Concatenate the s6a program timelines of the robot's groups and attach IK joint waypoints."""
    out, t0 = [], 0.0
    q_prev = None
    joint_names = None
    for gi, (G, ent, tool) in enumerate(groups):
        cell = sim.Cell(desc, base, yaw, G, tool["tcp_offset_m"])
        if joint_names is None:
            joint_names = [cell.m.joint(j).name for j in cell.jnts]
        poses = S.task_poses(G, base)
        if q_prev is None:
            p, ax, _ = poses["home"]
            best = None
            rng = np.random.default_rng(0)
            for s in [np.zeros(cell.n)] + [rng.uniform(np.maximum(cell.lo, -np.pi), np.minimum(cell.hi, np.pi)) for _ in range(24)]:
                q, ok, _ = cell.ik(p, ax, s)
                if ok and not cell.collides(q, np.zeros(3)):
                    c = float(np.sum(q ** 2))
                    if best is None or c < best[0]:
                        best = (c, q)
            q_prev = best[1] if best else np.zeros(cell.n)
        for op in ent["timeline"]:
            q = q_prev
            ok = True
            if op["target"] and op["target"] in poses:
                p, ax, _ = poses[op["target"]]
                q, ok, _ = cell.ik(p, ax, q_prev)
                if not ok:
                    for s in np.random.default_rng(1).uniform(np.maximum(cell.lo, -np.pi), np.minimum(cell.hi, np.pi), (16, cell.n)):
                        q2, ok2, _ = cell.ik(p, ax, s)
                        if ok2:
                            q, ok = q2, True
                            break
            out.append({"t_start_s": round(t0 + op["t_start_s"], 3), "dur_s": op["dur_s"], "op": op["op"],
                        "target": op["target"], "code": op["code"], "group_id": G["group_id"],
                        "tool": tool["name"], "q_end_rad": [round(float(v), 5) for v in q], "ik_ok": bool(ok)})
            q_prev = q
        t0 += ent["cycle_time_s"]
        if len(groups) > 1:
            out.append({"t_start_s": round(t0, 3), "dur_s": TOOL_CHANGE_S, "op": "tool_change", "target": None,
                        "code": "GET_TOOL", "group_id": G["group_id"], "tool": "tool changer",
                        "q_end_rad": [round(float(v), 5) for v in q_prev], "ik_ok": True})
            t0 += TOOL_CHANGE_S
    return {"joint_names": joint_names, "cycle_time_s": round(t0, 3), "waypoints": out}


# ----------------------------------------------------------------------------------------------- human
def _segments(el, station, cycle):
    c = [c for c in el["cycles"] if c["station"] == station and c["cycle_id"] == cycle][0]
    return sorted(c["elements"], key=lambda e: e["start_s"]), c["duration_s"]


def _human_tobe(J, fps, elements, human_codes, walk_factor, blend_s=0.3):
    """Splice the observed frames of the elements the worker keeps; compress WALK by the kitting lever."""
    frames, rows, t = [], [], 0.0
    for e in elements:
        if e["code"] not in human_codes:
            continue
        a, b = int(e["start_s"] * fps), max(int(e["start_s"] * fps) + 1, int(e["end_s"] * fps))
        seg = J[a:b]
        if e["code"] == "WALK" and walk_factor < 1:
            n = max(2, int(len(seg) * walk_factor))
            idx = np.linspace(0, len(seg) - 1, n)
            lo = np.floor(idx).astype(int)
            hi = np.minimum(lo + 1, len(seg) - 1)
            w = (idx - lo)[:, None, None]
            seg = seg[lo] * (1 - w) + seg[hi] * w
        if frames:  # short linear blend from the previous segment's last pose (splice seam)
            nb = int(blend_s * fps)
            prev = frames[-1][-1]
            ramp = np.linspace(0, 1, nb)[:, None, None]
            frames.append(prev * (1 - ramp) + seg[0] * ramp)
            t += nb / fps
        rows.append({"code": e["code"], "start_s": round(t, 3), "end_s": round(t + len(seg) / fps, 3),
                     "source_clip_s": [e["start_s"], e["end_s"]], "subgoal": e.get("subgoal")})
        frames.append(seg)
        t += len(seg) / fps
    return np.concatenate(frames, 0).astype(np.float32), rows


# ----------------------------------------------------------------------------------------------- scene
def _box(body, name, center, half, rgba, collide=True):
    body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=[float(v) for v in center],
                  size=[float(v) for v in half], rgba=rgba, contype=1 if collide else 0, conaffinity=1 if collide else 0)


def _zone_box(wb, name, poly, rgba):
    p = np.asarray(poly, float)
    lo, hi = p.min(0), p.max(0)
    _box(wb, name, [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, 0.002], [(hi[0] - lo[0]) / 2, (hi[1] - lo[1]) / 2, 0.002],
         rgba, collide=False)


def _scene(station, desc, base, yaw, groups, geo, targets, tool_len, track_window, bone_len):
    spec = mujoco.MjSpec()
    spec.compiler.degree = False
    spec.modelname = f"carda_{station}_belt_frame"
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 720
    wb = spec.worldbody
    wb.add_light(pos=[float(base[0]), float(base[1]), 6.0], dir=[0, 0, -1], diffuse=[0.8, 0.8, 0.8])
    c = np.asarray(geo["centroid_m"], float)
    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[12, 8, 0.1], pos=[float(c[0]), float(c[1]), 0],
                rgba=[0.75, 0.75, 0.72, 1])
    _zone_box(wb, "belt", geo["conveyor_zone_xy"], [0.25, 0.25, 0.27, 1])
    _zone_box(wb, "worker_zone", geo["worker_zone_xy"], [0.95, 0.6, 0.2, 0.35])
    if geo.get("line_side_zone_xy"):
        _zone_box(wb, "line_side_zone", geo["line_side_zone_xy"], [0.3, 0.6, 0.9, 0.35])
    # carrier: door panel fitted behind the contact points (approximation, see targets.json "geometry_note")
    n = np.asarray(geo["normal"], float)
    d = np.asarray(geo["conveyor_dir"], float)
    pts = np.asarray(geo["points_m"], float)
    along = pts @ d
    zmin, zmax = pts[:, 2].min(), pts[:, 2].max()
    # Contact face along -n (n points from the door to the robot side). The trim side of a door is not planar and the
    # contact clusters scatter +/-5-10 cm in depth, so the box face is put at the DEEPEST robot target: no target is
    # inside the box (no false tool collisions); shallower targets sit in front of it (recorded as depth_offset_m).
    tg = np.asarray([t["pos_m"] for t in targets if t["on"] == "carrier"], float)
    face = float((tg @ (-n)).max())
    th = 0.10
    door_c = d * (along.min() + along.max()) / 2 + (-n) * (face + th / 2)  # horizontal centre; face through the points
    yaw_door = math.atan2(d[1], d[0])
    carrier = wb.add_body(name="carrier", pos=[0, 0, 0])
    half_len = (along.max() - along.min()) / 2 + 0.20
    carrier.add_geom(name="door_panel", type=mujoco.mjtGeom.mjGEOM_BOX,
                     pos=[float(door_c[0]), float(door_c[1]), float((zmin + zmax) / 2 + 0.03)],
                     quat=sim._yaw_quat(yaw_door), size=[half_len, th / 2, (zmax - zmin) / 2 + 0.18],
                     rgba=[0.12, 0.12, 0.14, 1])
    leg_h = max(0.05, zmin - 0.18)
    cart_c = door_c + (-n) * 0.2
    carrier.add_geom(name="cart_frame", type=mujoco.mjtGeom.mjGEOM_BOX,
                     pos=[float(cart_c[0]), float(cart_c[1]), leg_h / 2],
                     quat=sim._yaw_quat(yaw_door), size=[half_len, 0.3, leg_h / 2], rgba=[0.7, 0.7, 0.75, 1])
    for t in targets:
        if t["on"] == "carrier":
            ax = np.asarray(t["approach_axis"], float)
            carrier.add_site(name=t["id"], pos=t["pos_m"], size=[0.012, 0, 0], rgba=[1, 0.1, 0.1, 1],
                             zaxis=[float(v) for v in ax])
    # linear-track carriage with pedestal, tray and robot
    bx, by, bz = [float(v) for v in base]
    if track_window:
        _box(wb, "track_rail", [bx, by, 0.03], [track_window / 2 + 0.6, 0.3, 0.03], [0.4, 0.4, 0.45, 1], collide=False)
    car = wb.add_body(name="carriage", pos=[bx, by, 0])
    if track_window:
        car.add_joint(name="track", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[float(v) for v in d],
                      range=[-(track_window / 2 + 0.5), track_window / 2 + 0.5], limited=True)
    _box(car, "pedestal", [0, 0, bz / 2], [0.2, 0.2, bz / 2 - 0.005], [0.85, 0.85, 0.85, 1])
    tray = S.tray_position(groups[0][0], np.asarray(base, float))
    _box(car, "tray", [float(tray[0] - bx), float(tray[1] - by), float(tray[2] - 0.035)], [0.15, 0.15, 0.03],
         [0.2, 0.5, 0.2, 1])
    for t in targets:
        if t["on"] == "carriage":
            car.add_site(name=t["id"], pos=[t["pos_m"][0] - bx, t["pos_m"][1] - by, t["pos_m"][2]], size=[0.012, 0, 0],
                         rgba=[0.1, 0.9, 0.1, 1], zaxis=[0, 0, -1])
    stem = desc[len(build.PREFIX):] if desc.startswith(build.PREFIX) else None
    child = build.child_spec(stem) if stem else mujoco.MjSpec.from_file(sim._mjcf_path(desc))
    site = child.site("attachment_site")
    flange_body = site.parent
    zax = np.array([0, 0, 1.0])
    from scipy.spatial.transform import Rotation
    R = Rotation.from_quat([site.quat[1], site.quat[2], site.quat[3], site.quat[0]])
    a = R.apply(zax)
    p0 = np.asarray(site.pos, float)
    flange_body.add_geom(name="tool", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                         fromto=list(map(float, p0)) + list(map(float, p0 + a * (tool_len - 0.035))), size=[0.035, 0, 0],
                         rgba=[0.9, 0.75, 0.1, 1])
    frame = car.add_frame(pos=[0, 0, bz], quat=sim._yaw_quat(yaw))
    frame.attach_body(child.worldbody.first_body(), "r_", "")
    # worker: one mocap capsule per bone (+ head sphere), driven by replay.py
    for a_, b_ in BONES:
        hb = wb.add_body(name=f"h_{a_}_{b_}", mocap=True, pos=[0, 0, -5])
        L = bone_len[(a_, b_)]
        hb.add_geom(name=f"hg_{a_}_{b_}", type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[0, 0, -L / 2, 0, 0, L / 2],
                    size=[BONE_RADIUS.get(a_, DEFAULT_BONE_RADIUS), 0, 0], rgba=[0.95, 0.45, 0.1, 0.9],
                    contype=0, conaffinity=0)
    hh = wb.add_body(name="h_head", mocap=True, pos=[0, 0, -5])
    hh.add_geom(name="hg_head", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[HEAD_RADIUS, 0, 0],
                rgba=[0.95, 0.45, 0.1, 0.9], contype=0, conaffinity=0)
    spec.compile()
    return spec.to_xml()


# ----------------------------------------------------------------------------------------------- main
def run(cfg: dict) -> None:
    I = _inputs(cfg)
    rec = [s for s in I["alloc"]["scenarios"] if s.get("recommended")][0]
    root = out_dir(cfg, "sim_package")
    groups_by_id = {g["group_id"]: g for g in I["cand"]["task_groups"]}
    rt = {(e["model"], e["group_id"]): e for e in I["rt"]["entries"]}
    walk_factor = {lc["station"]: 1 - (lc.get(WALK_REDUCTION_KEY) or 0) for lc in rec.get("layout_changes", [])
                   if lc.get("elements")}
    human_keys = {a["element_key"] for a in rec["assignment"] if a["resource"] == "human"}
    index = []
    for rb in rec["robots"]:
        st, model = rb["station"], rb["model"]
        sd = root / st
        sd.mkdir(parents=True, exist_ok=True)
        gs = [groups_by_id[g] for g in sorted(rb["groups"], key=lambda g: 0 if "pick" in g else 1)]
        G = [_group_for_sim(g) for g in gs]
        pl = gs[0]["placement"]
        base, yaw = pl["base_xyz"], pl["yaw_rad"]
        desc = I["cat"][model]["sim_model"]
        geo = gs[0]["geometry"]
        cyc = _rep_cycle(gs[0])
        # targets
        targets, order = [], 0
        for g, Gs in zip(gs, G):
            n = np.asarray(Gs["geometry"]["normal"], float)
            op = {"screwdriving": "screw", "pick_place": "place", "inspection": "view"}.get(Gs["process_type"], Gs["process_type"])
            for i, p in enumerate(Gs["targets_m"]):
                order += 1
                targets.append({"id": f"{g['category']}_{i}", "group_id": g["group_id"], "order": order, "operation": op,
                                "on": "carrier", "pos_m": [round(float(v), 4) for v in p],
                                "approach_axis": [round(float(v), 4) for v in -n],
                                "approach_offset_m": P.APPROACH_OFFSET_M,
                                "tolerance_mm": g["requirements"].get("precision_mm"),
                                "dwell_s_observed": (Gs["geometry"].get("dwell_s") or [None] * 99)[i],
                                "depth_offset_m": round(float((np.asarray(Gs["targets_m"], float) @ (-n)).max() - np.dot(p, -n)), 4)})
            tray = S.tray_position(Gs, np.asarray(base, float))
            targets.append({"id": f"{g['category']}_supply", "group_id": g["group_id"], "order": 0,
                            "operation": "pick_screw" if op == "screw" else "pick_part", "on": "carriage",
                            "pos_m": [round(float(v), 4) for v in tray], "approach_axis": [0, 0, -1],
                            "approach_offset_m": P.FEEDER_LIFT_M, "tolerance_mm": None, "dwell_s_observed": None})
        # robot reference
        ref = _robot_reference(desc, [(Gs, rt[(model, g["group_id"])], g["tool"]) for g, Gs in zip(gs, G)], base, yaw)
        st_t = rec["station_times"][st]
        ref.update({"model": model, "sim_model": desc, "base_xyz": base, "yaw_rad": yaw,
                    "mounting": gs[0].get("mounting"), "pedestal_height_m": pl.get("pedestal_height_m"),
                    "robot_start_after_human_s": st_t.get("robot_start_after_human_s", 0.0),
                    "joint_speed_deg_s": I["cat"][model]["joint_speed_deg_s"],
                    "joint_accel_rad_s2": P.JOINT_ACCEL_RAD_S2, "tcp_cap_m_s": P.COLLAB_TCP_CAP_M_S,
                    "linear_speed_m_s": P.LIN_SPEED_M_S, "tools": [g["tool"] for g in gs],
                    "bt_xml": [p["bt_xml"] for p in I["plan"]["plans"] if p["station"] == st][0]})
        # human
        npz = np.load(ROOT / [c for s in I["sg"]["stations"] if s["station"] == st for c in s["cycles"] if c["cycle_id"] == cyc][0]["human"]["joints_npz"])
        J, names, fps = npz["joints"], [str(x) for x in npz["names"]], float(npz["fps"])
        shutil.copy(ROOT / f"out/s1/human/{st}_{cyc}.npz", sd / "human_asis.npz")
        els, clip = _segments(I["el"], st, cyc)
        hcodes = {k.split(":")[1] for k in human_keys if k.startswith(st + ":")} - {"UNOBS"}
        Jt, rows = _human_tobe(J, fps, els, hcodes, walk_factor.get(st, 1.0))
        np.savez_compressed(sd / "human_tobe.npz", joints=Jt, names=np.array(names), fps=fps)
        unobs = [a["time_s"] for a in rec["assignment"] if a["element_key"] == f"{st}:UNOBS"]
        std = {(r["station"], r["code"]): r for r in I["ts"]["element_stats"]}
        tobe = {"station": st, "source_cycle": cyc, "fps": fps, "spliced_duration_s": round(len(Jt) / fps, 2),
                "segments": rows, "walk_time_factor": walk_factor.get(st, 1.0),
                "unobserved_block_s": unobs[0] if unobs else 0.0,
                "standard_times_s": {c: std[(st, c)]["standard_s"] for c in sorted(hcodes) if (st, c) in std},
                "robot_codes": sorted({k.split(":")[1] for k in rb["elements"]}),
                "note": ("Observed frames of the elements the worker keeps, in observed order, joined with 0.3 s linear "
                         "blends; WALK resampled by walk_time_factor (AMR kitting assumption). The unobserved part of "
                         "the 240 s cycle (unobserved_block_s) has no motion data: replay holds the last pose; treat its "
                         "location as unknown.")}
        (sd / "human_tobe_timeline.json").write_text(json.dumps(tobe, indent=2))
        bone_len = {}
        for a_, b_ in BONES:
            bone_len[(a_, b_)] = float(np.median(np.linalg.norm(J[:, names.index(a_)] - J[:, names.index(b_)], axis=1)))
        # motion / frames
        sgc = [c for s in I["sg"]["stations"] if s["station"] == st for c in s["cycles"] if c["cycle_id"] == cyc][0]
        motion = {"frame": "belt frame = plant frame (s0) at t_ref; z up, metres",
                  "t_ref_s_in_source_clip": sgc.get("t_ref_s"),
                  "conveyor": {"speed_mm_s": geo["conveyor_speed_mm_s"], "direction_world": geo["conveyor_dir"],
                               "world_from_belt": "x_world(t) = x_belt + v*d*(t - t_ref)",
                               "cycle_to_cycle_speed_spread": "about +/-3% between cameras and cycles (s1)"},
                  "takt_s": cfg["line"]["takt_s"], "station_cycle_cap_s": rec.get("cycle_cap_s"),
                  "station_times_recommended": st_t, "representative_cycle": cyc, "clip_duration_s": clip,
                  "belt_width_m": round(float(np.ptp(np.asarray(geo["conveyor_zone_xy"])[:, 1])), 3),
                  "operator_stands_on_belt": True,
                  "tracking_window_m": pl.get("tracking_window_m"),
                  "geometry_note": ("Door panel and cart are boxes fitted to the worker's wrist-dwell contact points "
                                    "(no CAD in CarDA). Target points are contact clusters, accuracy about +/-5 cm; "
                                    "screw positions must come from door CAD for a real cell.")}
        (sd / "motion.json").write_text(json.dumps(motion, indent=2))
        (sd / "targets.json").write_text(json.dumps({"station": st, "frame": motion["frame"], "targets": targets,
                                                     "geometry_note": motion["geometry_note"]}, indent=2))
        (sd / "robot_reference.json").write_text(json.dumps(ref, indent=2))
        xml = _scene(st, desc, base, yaw, [(g, Gs) for g, Gs in zip(G, gs)], geo, targets,
                     max(g["tool"]["tcp_offset_m"] for g in gs), pl.get("tracking_window_m") if gs[0].get("mounting") == "linear_track" else 0.0,
                     bone_len)
        (sd / "scene.xml").write_text(xml)
        index.append({"station": st, "model": model, "dir": str(sd.relative_to(ROOT))})
    shutil.copy(HERE / "replay.py", root / "replay.py")
    (root / "scenario_matrix.yaml").write_text((HERE / "scenario_matrix.yaml").read_text())
    (root / "index.json").write_text(json.dumps(index, indent=2))
    print("    wrote", [i["dir"] for i in index])
