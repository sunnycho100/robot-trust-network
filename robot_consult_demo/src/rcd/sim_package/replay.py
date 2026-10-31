"""Baseline replay harness for one station of the CarDA hand-off package (needs only mujoco + numpy; cv2 for video).

    python replay.py WS30                    # to-be worker motion, KPIs -> WS30/kpi_tobe.json, preview.png
    python replay.py WS30 --human asis       # observed (as-is) worker motion instead
    python replay.py WS30 --video            # also writes WS30/replay_tobe.mp4

What it does, in the BELT frame (door, worker and tracking carriage are static relative to each other):
  * robot: joint-space linear interpolation between the IK reference waypoints of robot_reference.json, starting at
    robot_start_after_human_s (the s5 interlock); this is a kinematic reference, not a controller;
  * worker: 22 capsules driven by human_tobe.npz (or human_asis.npz); after the motion data ends the last pose is held;
  * every step: minimum robot-worker distance (mj_geomDistance over robot link/tool capsules x worker capsules),
    robot TCP speed, robot-door/cart penetration, and the ISO/TS 15066 speed-and-separation check
    S_p = v_h*(T_r+T_s) + v_r*T_r + C + Z_d + Z_r (parameters in scenario_matrix.yaml -> ssm).
The sim team is expected to replace the robot part with their controller (Isaac Sim / cuRobo / vendor offline
programming) and keep the same KPI definitions.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent


def _quat_z_to(v):
    v = v / (np.linalg.norm(v) + 1e-12)
    z = np.array([0, 0, 1.0])
    c = float(np.dot(z, v))
    if c < -0.999999:
        return np.array([0, 1.0, 0, 0])
    axis = np.cross(z, v)
    s = math.sqrt((1 + c) * 2)
    return np.array([s / 2, *(axis / s)])


def _ssm_params():
    try:
        import yaml
        return yaml.safe_load((HERE / "scenario_matrix.yaml").read_text())["ssm"]
    except Exception:
        return {"v_h_m_s": 1.6, "T_r_s": 0.1, "T_s_s": 0.3, "C_m": 0.12, "Z_d_m": 0.05, "Z_r_m": 0.02}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("station")
    ap.add_argument("--human", choices=["tobe", "asis"], default="tobe")
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--after", choices=["hold", "away"], default="hold",
                    help="worker after the motion data ends: hold the last pose (upper bound on conflict) or leave the "
                         "station (lower bound); the unobserved part of the cycle has no location data")
    a = ap.parse_args()
    sd = HERE / a.station
    m = mujoco.MjModel.from_xml_path(str(sd / "scene.xml"))
    d = mujoco.MjData(m)
    ref = json.loads((sd / "robot_reference.json").read_text())
    hz = np.load(sd / f"human_{a.human}.npz")
    J, names, fps = hz["joints"], [str(x) for x in hz["names"]], float(hz["fps"])
    ssm = _ssm_params()

    qadr = np.array([m.jnt_qposadr[m.joint(n).id] for n in ref["joint_names"]])
    wps = ref["waypoints"]
    t_rs = float(ref.get("robot_start_after_human_s") or 0.0)
    q_home = np.array(wps[0]["q_end_rad"])
    knots_t, knots_q = [t_rs], [q_home]
    for w in wps:
        knots_t.append(t_rs + w["t_start_s"] + w["dur_s"])
        knots_q.append(np.array(w["q_end_rad"]))
    knots_t, knots_q = np.array(knots_t), np.array(knots_q)
    t_robot_end = knots_t[-1]

    def q_at(t):
        if t <= knots_t[0]:
            return knots_q[0]
        if t >= knots_t[-1]:
            return knots_q[-1]
        k = np.searchsorted(knots_t, t) - 1
        w = (t - knots_t[k]) / max(1e-9, knots_t[k + 1] - knots_t[k])
        return knots_q[k] * (1 - w) + knots_q[k + 1] * w

    bones = [(b[2:].split("_")[0], b[2:].split("_")[1]) for b in
             [m.body(i).name for i in range(m.nbody)] if b.startswith("h_") and b != "h_head"]
    mocap = {bb: m.body(f"h_{bb[0]}_{bb[1]}").mocapid[0] for bb in bones}
    head = m.body("h_head").mocapid[0]
    robot_g = [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name.startswith("r_")]
    human_g = [g for g in range(m.ngeom) if (m.geom(g).name or "").startswith("hg_")]
    env_g = [m.geom(n).id for n in ("door_panel", "cart_frame")]
    tool_g = m.geom("r_tool").id
    fromto = np.zeros(6)

    T_end = min(240.0, max(t_robot_end, len(J) / fps) + 1.0)
    times = np.arange(0.0, T_end, a.dt)
    rows, tip_prev = [], None
    renderer = None
    frames = []
    if a.video:
        renderer = mujoco.Renderer(m, 540, 960)
        cam = mujoco.MjvCamera()
        tg = [t for t in json.loads((sd / "targets.json").read_text())["targets"] if t["on"] == "carrier"]
        n = -np.asarray(tg[0]["approach_axis"], float)  # from the door towards the robot / worker side
        cam.lookat[:] = np.mean([t["pos_m"] for t in tg], axis=0) + n * 0.7
        # look from the robot/worker side towards the door, 35 deg oblique, so robot, worker and door are all visible
        cam.distance, cam.elevation = 4.0, -24.0
        cam.azimuth = math.degrees(math.atan2(-n[1], -n[0])) + 35.0
        mujoco.mj_forward(m, d)
        renderer.update_scene(d, camera=cam)
        if np.dot(np.asarray(renderer.scene.camera[0].pos) - cam.lookat, n) < 0:  # make sure we look from the robot side
            cam.azimuth += 180.0
    for t in times:
        d.qpos[qadr] = q_at(t)
        k = min(len(J) - 1, int(t * fps))
        P = J[k] if (a.after == "hold" or int(t * fps) < len(J)) else J[k] + np.array([0.0, 0.0, -50.0])
        for (ja, jb), mid in mocap.items():
            pa, pb = P[names.index(ja)], P[names.index(jb)]
            d.mocap_pos[mid] = (pa + pb) / 2
            d.mocap_quat[mid] = _quat_z_to(pb - pa)
        hv = P[names.index("Head")] - P[names.index("Neck")]
        d.mocap_pos[head] = P[names.index("Head")] + 0.08 * hv / (np.linalg.norm(hv) + 1e-9)
        mujoco.mj_forward(m, d)
        dmin = min(mujoco.mj_geomDistance(m, d, gr, gh, 3.0, fromto) for gr in robot_g for gh in human_g)
        dmin = min(dmin, 3.0)
        pen = min(mujoco.mj_geomDistance(m, d, gr, ge, 0.5, fromto) for gr in robot_g if gr != tool_g for ge in env_g)
        pen_tool = min(mujoco.mj_geomDistance(m, d, tool_g, ge, 0.5, fromto) for ge in env_g)
        tip = d.geom_xpos[tool_g].copy()
        v_r = 0.0 if tip_prev is None else float(np.linalg.norm(tip - tip_prev) / a.dt)
        tip_prev = tip
        Sp = ssm["v_h_m_s"] * (ssm["T_r_s"] + ssm["T_s_s"]) + v_r * ssm["T_r_s"] + ssm["C_m"] + ssm["Z_d_m"] + ssm["Z_r_m"]
        active = t_rs <= t <= t_robot_end
        rows.append((t, dmin, pen, v_r, Sp, active, pen_tool))
        if renderer is not None and int(round(t / a.dt)) % max(1, int(round(0.2 / a.dt))) == 0:
            renderer.update_scene(d, camera=cam)
            frames.append(renderer.render().copy())
    R = np.array([(r[0], r[1], r[2], r[3], r[4], float(r[5]), r[6]) for r in rows])
    act = R[:, 5] > 0.5
    dist_act = R[act, 1]
    kpi = {
        "station": a.station, "human_motion": a.human, "after_motion_end": a.after, "dt_s": a.dt,
        "robot_program_s": round(float(t_robot_end - t_rs), 2), "robot_start_s": t_rs,
        "worker_motion_s": round(len(J) / fps, 2),
        "min_robot_worker_distance_m": round(float(R[:, 1].min()), 3),
        "min_distance_while_robot_active_m": round(float(dist_act.min()), 3) if act.any() else None,
        "share_active_time_below_0_3m": round(float((dist_act < 0.3).mean()), 3) if act.any() else None,
        "share_active_time_below_0_5m": round(float((dist_act < 0.5).mean()), 3) if act.any() else None,
        "share_active_time_ssm_violation": round(float((R[act, 1] < R[act, 4]).mean()), 3) if act.any() else None,
        "ssm_params": ssm,
        "max_tcp_speed_m_s": round(float(R[:, 3].max()), 3),
        "arm_link_vs_door_cart_penetration_s": round(float((R[:, 2] < -0.005).sum() * a.dt), 2),
        "tool_vs_door_cart_penetration_s": round(float((R[:, 6] < -0.005).sum() * a.dt), 2),
        "path_note": "waypoints were collision-checked in s6a; the straight joint-space interpolation between them was "
                     "not - penetration here means a planned (collision-aware) path is needed, not that a target is unreachable",
        "ik_ok_waypoints": f"{sum(w['ik_ok'] for w in wps)}/{len(wps)}",
        "note": "Kinematic reference replay. SSM violation share = fraction of robot-active time the robot would have to "
                "slow down or stop; each second of that is potential cycle-time loss the sim team must quantify.",
    }
    tag = a.human + ("" if a.after == "hold" else "_away")
    (sd / f"kpi_{tag}.json").write_text(json.dumps(kpi, indent=2))
    np.savetxt(sd / f"trace_{tag}.csv", R, delimiter=",", header="t_s,min_dist_m,arm_env_dist_m,tcp_speed_m_s,ssm_Sp_m,robot_active,tool_env_dist_m",
               fmt="%.4f")
    if frames:
        try:
            import cv2
            vw = cv2.VideoWriter(str(sd / f"replay_{a.human}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (960, 540))
            for f in frames:
                vw.write(f[:, :, ::-1])
            vw.release()
        except ImportError:
            pass
        mid = frames[len(frames) // 2]
        try:
            import cv2
            cv2.imwrite(str(sd / "preview.png"), mid[:, :, ::-1])
        except ImportError:
            pass
    print(json.dumps(kpi, indent=2))


if __name__ == "__main__":
    main()
