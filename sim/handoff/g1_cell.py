#!/usr/bin/env python3
"""Our G1 in the worker's place in the rebuilt CarDA WS30 cell: a robot-to-robot hand-off of the door.

The G1 (Unitree's published walking policy, as in sim/mujoco_houses) walks up to the door, holds its hands at the
door for 4 s (the worker's INSERT), and walks off. The M0617 runs the same screw program as carda_cell.py and starts
at 11.0 s (the handoff's WS30 interlock). P2: it only moves while every G1 body is at least 1.0 m from the active
target. P1: it moves regardless, which shows the overlap.

  MUJOCO_GL=egl PYTHONPATH=../mujoco_houses/src python g1_cell.py WS30.bvh --policy P2 --video g1_p2.mp4 --out kpi.json
"""
from __future__ import annotations

import argparse
import json
import math
import re

import mujoco
import numpy as np

import carda_cell as cc
from unitree_demo.coop_lift import demo_config
from unitree_demo.g1_policy import G1Controller, G1UpperBodyController, gait_phase

HOLD_S = 4.0


def build(cell, cfg):
    xml = cc.build_xml(cell, []).replace('<default><geom contype="0" conaffinity="0"/></default>', "")
    xml = re.sub(r'name="(belt|rail|tray|tray_leg)"', r'name="\1" contype="0" conaffinity="0"', xml)
    spec = mujoco.MjSpec.from_string(xml)
    n, t, side = cell["n"], cell["t"], cell["side"]
    spot = cell["center"][:2] - n[:2] * 0.62
    start = spot - n[:2] * 2.4 - t[:2] * side * 0.9
    yaw = math.atan2(*(spot - start)[::-1])
    frame = spec.worldbody.add_frame(pos=[*start, 0], quat=[math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
    spec.attach(mujoco.MjSpec.from_file(str(cfg.robot_xml_23dof)), prefix="g1_", frame=frame)
    model = spec.compile()
    model.opt.timestep = cfg.simulation_dt
    for g in range(model.ngeom):
        name = model.geom(g).name
        if name.startswith("r_"):                      # the arm is driven kinematically; distances are measured
            model.geom_contype[g] = model.geom_conaffinity[g] = 0
    return model, spot, start


def heading_cmd(xy, yaw, goal, face=None, v=0.45):
    """Walk command [vx, vy, wz] toward goal; at the goal, turn to face."""
    d = goal - xy
    dist = np.linalg.norm(d)
    if dist > 0.12:
        err = math.remainder(math.atan2(d[1], d[0]) - yaw, math.tau)
        return np.array([v * max(0.0, math.cos(err)) ** 2, 0, np.clip(1.5 * err, -0.7, 0.7)], np.float32), False
    if face is not None:
        err = math.remainder(face - yaw, math.tau)
        if abs(err) > 0.08:
            return np.array([0, 0, np.clip(1.5 * err, -0.6, 0.6)], np.float32), False
    return np.zeros(3, np.float32), True


def run(bvh, policy, video=None, t_max=150.0):
    cfg = demo_config()
    _, _, _, _, _, cell, info = cc.make_cell(bvh)
    model, spot, start = build(cell, cfg)
    data = mujoco.MjData(model)
    legs, upper = G1Controller(model, cfg, "g1_"), G1UpperBodyController(model, "g1_")
    legs.reset(data)
    mujoco.mj_forward(model, data)
    arm = cc.Arm(model)
    prog = cc.Program(arm, cell, np.random.default_rng(0))
    track = model.jnt_qposadr[model.joint("track").id]
    rgeoms = [model.geom(f"r_{g}").id for g in ["base", "tool"] + [f"link{i}" for i in range(1, 7)]]
    g1_bodies = [b for b in range(model.nbody) if model.body(b).name.startswith("g1_")]
    g1_geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] in g1_bodies and model.geom_contype[g]]
    n, t, side = cell["n"], cell["t"], cell["side"]
    face_door = math.atan2(n[1], n[0])
    exit_xy = spot - n[:2] * 1.2 - t[:2] * side * 2.2
    phase, t_hold, tau, wait, rows = "in", None, 0.0, 0.0, []
    fromto = np.zeros(6)
    steps_per_frame = round(1 / (30 * cfg.simulation_dt))
    rend = vid = None
    if video:
        rend, vid = mujoco.Renderer(model, 720, 1280), cc.VideoOut(video, 1280, 720, 30)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = cell["center"] * [1, 1, 0] + [0, 0, 0.8]
        cam.distance, cam.elevation = 5.2, -26
        v = -n - t * side * 0.5
        cam.azimuth = math.degrees(math.atan2(v[1], v[0])) + 180
    step, prev_tcp, q = 0, None, prog.home
    tdof = model.jnt_dofadr[model.joint("track").id]
    while data.time < t_max:
        now = data.time
        xy, yaw = legs.position_xy(data), legs.yaw(data)
        if phase == "in":
            cmd, done = heading_cmd(xy, yaw, spot, face_door)
            if done:
                phase, t_hold = "hold", now
        elif phase == "hold":
            cmd = np.zeros(3, np.float32)
            upper.set_carry_pose(min(1.0, (now - t_hold) / 1.0))
            if now - t_hold > HOLD_S:
                phase = "out"
        else:
            upper.target_q[:] *= 0.98
            cmd, _ = heading_cmd(xy, yaw, exit_xy)
        if step % cfg.control_decimation == 0:
            legs.update_policy(data, cmd, gait_phase(data))
        if step % steps_per_frame == 0:
            q, op, code, k = prog.at(tau)
            data.qpos[arm.qadr], data.qpos[track] = q, 0
            mujoco.mj_kinematics(model, data)
            tcp = data.site_xpos[arm.site].copy()
            v_r = 0.0 if prev_tcp is None else np.linalg.norm(tcp - prev_tcp) * 30
            prev_tcp = tcp
            d = min(mujoco.mj_geomDistance(model, data, a, b, 3.0, fromto) for a in rgeoms for b in g1_geoms)
            sp = cc.V_H * (cc.T_R + cc.T_S) + v_r * cc.T_R + cc.C_INTR
            active = cell["targets"][k] if k >= 0 else (cell["tray"] if code == "GET_TOOL" else None)
            g1_d = 9.9 if active is None else float(np.linalg.norm(data.xpos[g1_bodies] - active, axis=1).min())
            started = now >= cc.START_AFTER and tau < prog.T
            allowed = started and (policy == "P1" or g1_d >= cc.P2_CLEAR)
            wait += (1 / 30) if started and not allowed else 0
            tau += (1 / 30) if allowed else 0
            rows.append(dict(t=round(now, 3), tau=round(tau, 3), code=code, g1_phase=phase, dist=round(d, 4),
                             s_p=round(sp, 3), g1_to_target=round(g1_d, 3), moving=int(allowed),
                             pelvis_z=round(float(data.xpos[legs.body_id][2]), 3)))
            if vid:
                rend.update_scene(data, cam)
                img = cc.hud(rend.render(), now, tau, prog.T, code, d, sp, allowed, policy, "G1")
                vid.write(img)
            if tau >= prog.T and phase == "out" and np.linalg.norm(xy - exit_xy) < 0.3:
                break
        legs.apply(data)
        upper.apply(data)
        data.qpos[arm.qadr], data.qpos[track] = q, 0      # arm and carriage follow the program exactly
        data.qvel[arm.dadr], data.qvel[tdof] = 0, 0
        mujoco.mj_step(model, data)
        step += 1
    if vid:
        vid.close()
    mv = [r for r in rows if r["moving"]]
    run_rows = [r for r in rows if r["t"] >= cc.START_AFTER]
    kpi = dict(policy=policy, robot_program_s=round(prog.T, 2), finished=bool(tau >= prog.T),
               robot_end_s=round(data.time, 2), robot_wait_s=round(wait, 2),
               g1_arrived_s=round(t_hold, 2) if t_hold else None,
               g1_min_pelvis_z=round(min(r["pelvis_z"] for r in rows), 3),
               min_arm_g1_distance_m=round(min(r["dist"] for r in run_rows), 3),
               within_0p5m_frac=round(np.mean([r["dist"] < 0.5 for r in mv]), 3) if mv else None,
               ssm_violation_frac=round(np.mean([r["dist"] < r["s_p"] for r in mv]), 3) if mv else None)
    return rows, dict(cell=info, kpi=kpi)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bvh")
    ap.add_argument("--policy", default="P2", choices=["P1", "P2"])
    ap.add_argument("--video")
    ap.add_argument("--out")
    ap.add_argument("--trace")
    ap.add_argument("--t-max", type=float, default=150.0)
    a = ap.parse_args()
    rows, out = run(a.bvh, a.policy, a.video, a.t_max)
    print(json.dumps(out["kpi"], indent=1))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
    if a.trace:
        import csv
        with open(a.trace, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
