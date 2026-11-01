#!/usr/bin/env python3
"""CarDA v3 option runner (SIM_OPTIONS_ABC.md, sections 3 to 5) on the teammate's real option packages.

Loads robot_consult_demo/out_options/<OPT>/sim_package/<ST>/ (scene.xml, robot_reference.json, targets.json,
human_tobe*.{npz,json}) plus the stage outputs next to it (s2 time study, s4 groups, s5 scenario), then runs
station x policy x worker_after x draw with common random numbers and writes results_<OPT>.csv, summary.json,
robot_task_times.json and seeds.json.

Per run:
  * robot program: their s6a model (rcd.s6a_robot_sim.sim.simulate + rcd.sim_package.export._robot_reference)
    re-run on the draw's joint acceleration, rundown time, conveyor speed and door pose (IK re-solved on the
    shifted door, i.e. the robot is told where the door is);
  * worker: their to-be motion, each element's segments time-warped to the draw's lognormal element time;
    after the motion the worker stays at the door (last pose) or leaves, until the drawn station load is done;
  * every 0.1 s: capsule distance robot x worker, ISO/TS 15066 SSM always on (program clock scaled by
    (d - S_p0) / (T_r v_r), paused below S_p0), plus the policy gate (P1 none, P2 worker >= 1.0 m from the
    active target, P3 robot only after the worker has left the door), robot vs door/cart/tray/pedestal contact.

  python option_runner.py run --rcd RCD --option C --out OUT --jobs 12
  python option_runner.py baseline --rcd RCD --option C          # their section 7 numbers with this engine
  python option_runner.py video --rcd RCD --station WS10 --policy P1 --after stays --draw 0 --out ws10.mp4
  python option_runner.py contrast --rcd RCD --out OUT           # human only (as-is) vs option C, same seeds
  python option_runner.py sidebyside --rcd RCD --station WS10 --policy P2 --out ws10_sbs.mp4
  python option_runner.py selfcheck
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import subprocess
import sys
import zlib
from pathlib import Path

import mujoco
import numpy as np

# ------------------------------------------------------------------ common random numbers (shared by A, B, C)
MASTER_SEED = 20261031
N_DRAWS = 30
SEEDS = [int(s) for s in np.random.SeedSequence(MASTER_SEED).generate_state(N_DRAWS)]


def stream(seed: int, name: str) -> np.random.Generator:
    """Generator for one random variable of one draw. Keyed by name, so a variable gets the same numbers whatever
    else an option draws (draw k of A, B and C share conveyor, door pose, worker element times, ...)."""
    return np.random.default_rng([seed, zlib.crc32(name.encode())])


def draw_params(seed: int) -> dict:
    """Line-level draws (scenario_matrix.yaml randomization). Station-level ones come from stream() on demand."""
    door = stream(seed, "line/door_pose").uniform(-1, 1, 4) * np.array([0.010, 0.010, 0.010, math.radians(1.0)])
    return {"conveyor_rel": float(stream(seed, "line/conveyor_speed").uniform(-0.05, 0.05)),
            "door_dx_m": float(door[0]), "door_dy_m": float(door[1]), "door_dz_m": float(door[2]),
            "door_yaw_rad": float(door[3]),
            "joint_accel": float(stream(seed, "line/joint_accel").uniform(1.0, 2.0))}


def rundown_draw(seed, station):
    return float(stream(seed, f"{station}/rundown").uniform(2.0, 3.5))   # ponytail: one value for all screws of a run


def lognormal(rng, mean, std):
    """Lognormal with the given mean and std (CV capped at 0.5, scenario_matrix.yaml)."""
    if mean <= 0:
        return 0.0
    cv = min(std / mean, 0.5)
    if cv <= 0:
        return float(mean)
    s2 = math.log(1 + cv * cv)
    return float(rng.lognormal(math.log(mean) - s2 / 2, math.sqrt(s2)))


def draw_elements(stats, st, seed, codes, walk_factor):
    """Drawn per-cycle element times {code: (normal_s, standard_s)} from time_study element_stats rows of one
    station; WALK scaled by the walk factor (AMR kitting lever)."""
    out = {}
    for c in codes:
        r = stats.get(c)
        if r is None or c in ("UNOBS", "IDLE") or r["mean_s"] <= 0:
            continue
        x = lognormal(stream(seed, f"{st}/worker/{c}"), r["mean_s"], r["std_s"])
        wf = walk_factor if c == "WALK" else 1.0
        out[c] = (x * r["rating"] * wf, x * r["standard_s"] / r["mean_s"] * wf)
    return out


# ------------------------------------------------------------------ constants from their package
TAKT_CAP = 228.0
P2_CLEAR_M = 1.0
DT = 0.1
T_MAX = 600.0
PEN_M = 0.005          # their replay's penetration threshold


def _quat_z_to(v):
    v = v / (np.linalg.norm(v) + 1e-12)
    c = float(v[2])
    if c < -0.999999:
        return np.array([0, 1.0, 0, 0])
    axis = np.cross([0, 0, 1.0], v)
    s = math.sqrt((1 + c) * 2)
    return np.array([s / 2, *(axis / s)])


def _rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


class Station:
    """One station of one option package plus the stage outputs it was made from."""

    def __init__(self, rcd: Path, option: str, st: str):
        sys.path.insert(0, str(rcd / "src"))
        self.rcd, self.option, self.st = rcd, option, st
        self.stage = rcd / "out_options" / option
        self.pk = self.stage / "sim_package" / st
        L = lambda p: json.loads(Path(p).read_text())  # noqa: E731
        self.ref0 = L(self.pk / "robot_reference.json")
        self.targets = L(self.pk / "targets.json")["targets"]
        self.tl = L(self.pk / "human_tobe_timeline.json")
        self.motion = L(self.pk / "motion.json")
        self.ssm = __import__("yaml").safe_load((self.stage / "sim_package" / "scenario_matrix.yaml").read_text())
        self.line = self.ssm["line"]
        self.ssm = self.ssm["ssm"]
        hz = np.load(self.pk / "human_tobe.npz")
        self.J, self.names, self.fps = hz["joints"], [str(x) for x in hz["names"]], float(hz["fps"])
        ha = np.load(self.pk / "human_asis.npz")
        self.Ja, self.fps_a = ha["joints"], float(ha["fps"])
        alloc = L(self.stage / "s5" / "allocation.json")
        self.rec = [s for s in alloc["scenarios"] if s.get("recommended")][0]
        self.st_times = self.rec["station_times"][st]
        self.stats = {r["code"]: r for r in L(self.stage / "s2" / "time_study.json")["element_stats"] if r["station"] == st}
        self.rb = [r for r in self.rec["robots"] if r["station"] == st][0]
        cand = L(self.stage / "s4" / "candidates.json")
        by = {g["group_id"]: g for g in cand["task_groups"]}
        self.groups = [by[g] for g in sorted(self.rb["groups"], key=lambda g: 0 if "pick" in g else 1)]
        self.cat = {r["model"]: r for r in L(self.stage / "s4" / "robot_catalog.json")["robots"]}
        self.human_codes = sorted(self.tl["standard_times_s"])
        self.robot_codes = sorted(self.tl["robot_codes"])
        self.n_robot_elements = len(self.robot_codes)
        self.handover_s = self.line["handover_s_per_robot_element"] * self.n_robot_elements
        self.unobs_s = float(self.tl["unobserved_block_s"])
        self.walk_factor = float(self.tl["walk_time_factor"])
        self.t_rs = float(self.ref0.get("robot_start_after_human_s") or 0.0)
        S = self.ssm
        self.sp0 = S["v_h_m_s"] * (S["T_r_s"] + S["T_s_s"]) + S["C_m"] + S["Z_d_m"] + S["Z_r_m"]
        # door pivot for the pose draw: centroid of the carrier targets
        self.door_c = np.mean([t["pos_m"] for t in self.targets if t["on"] == "carrier"], axis=0)
        self.d_line = np.asarray(self.motion["conveyor"]["direction_world"], float)
        self.v_belt = float(self.motion["conveyor"]["speed_mm_s"]) / 1000.0

    # -------------------------------------------------------------- draws
    def door_tf(self, p):
        """Rigid door pose offset of a draw: rotation about the target centroid, then translation."""
        R = _rot_z(p["door_yaw_rad"])
        across = np.cross([0, 0, 1.0], self.d_line)
        off = p["door_dx_m"] * self.d_line + p["door_dy_m"] * across + np.array([0, 0, p["door_dz_m"]])
        return R, self.door_c - R @ self.door_c + off

    def worker_draw(self, seed, codes, walk_factor):
        return draw_elements(self.stats, self.st, seed, codes, walk_factor)

    # -------------------------------------------------------------- robot program (their s6a model)
    def program(self, seed, p):
        from rcd.s4_catalog import process as P
        from rcd.s6a_robot_sim import sim
        from rcd.s6a_robot_sim.run import _group_for_sim
        from rcd.sim_package import export as X
        R, t = self.door_tf(p)
        rd = rundown_draw(seed, self.st)
        acc0, pt0 = P.JOINT_ACCEL_RAD_S2, P.process_time
        P.JOINT_ACCEL_RAD_S2 = p["joint_accel"]
        P.process_time = lambda kind: rd if kind == "rundown" else pt0(kind)
        try:
            model = self.rb["model"]
            desc = self.cat[model]["sim_model"]
            speeds = self.cat[model]["joint_speed_deg_s"]
            pl = self.groups[0]["placement"]
            base, yaw = pl["base_xyz"], pl["yaw_rad"]
            ents, Gs = [], []
            for g in self.groups:
                G = copy.deepcopy(_group_for_sim(g))
                geo = G["geometry"]
                G["targets_m"] = [list(R @ np.asarray(q, float) + t) for q in G["targets_m"]]
                geo["targets_m"] = G["targets_m"]
                geo["normal"] = list(R @ np.asarray(geo["normal"], float))
                geo["conveyor_speed_mm_s"] = geo["conveyor_speed_mm_s"] * (1 + p["conveyor_rel"])
                cell = sim.Cell(desc, base, yaw, G, g["tool"]["tcp_offset_m"])
                res = sim.simulate(cell, G, speeds, base, t_mid=G["program_est_s"] / 2)
                res = sim.simulate(cell, G, speeds, base, t_mid=res["cycle_time_s"] / 2)
                res["cell"] = cell
                ents.append(res)
                Gs.append(G)
            ref = X._robot_reference(desc, [(G, e, g["tool"]) for G, e, g in zip(Gs, ents, self.groups)], base, yaw)
            # IK residual at every contact pose (tool tip vs the shifted screw / seat), with the door pose known
            err = []
            for G, e in zip(Gs, ents):
                poses = __import__("rcd.s4_catalog.scene", fromlist=["x"]).task_poses(G, base)
                for w in ref["waypoints"]:
                    if w["group_id"] == G["group_id"] and (w["target"] or "").startswith("tgt_") and w["op"] != "move_l":
                        tcp, _ = e["cell"].fk(np.asarray(w["q_end_rad"]))
                        err.append(float(np.linalg.norm(tcp - poses[w["target"]][0])) * 1000)
            stats = [e["stats"] for e in ents]
            return dict(ref=ref, program_s=float(ref["cycle_time_s"]), rundown_s=rd,
                        ik_failures=sum(s["attempts"] - s["ok"] for s in stats) + sum(not w["ik_ok"] for w in ref["waypoints"]),
                        waypoint_collisions=sum(s["collisions"] for s in stats),
                        tcp_err_mm=max(err) if err else 0.0,
                        breakdown=[e["breakdown"] for e in ents], by_code=[e["by_code"] for e in ents],
                        group_cycles=[e["cycle_time_s"] for e in ents],
                        group_stats=[{k: s[k] for k in ("attempts", "ok", "collisions")} for s in stats])
        finally:
            P.JOINT_ACCEL_RAD_S2, P.process_time = acc0, pt0


class Sim:
    """MuJoCo scene of one station, driven kinematically (their replay conventions)."""

    def __init__(self, S: Station):
        self.S = S
        m = self.m = mujoco.MjModel.from_xml_path(str(S.pk / "scene.xml"))
        self.d = mujoco.MjData(m)
        self.qadr = np.array([m.jnt_qposadr[m.joint(n).id] for n in S.ref0["joint_names"]])
        bones = [(b[2:].split("_")[0], b[2:].split("_")[1]) for b in
                 [m.body(i).name for i in range(m.nbody)] if b.startswith("h_") and b != "h_head"]
        self.mocap = {bb: m.body(f"h_{bb[0]}_{bb[1]}").mocapid[0] for bb in bones}
        self.bidx = {bb: (S.names.index(bb[0]), S.names.index(bb[1])) for bb in bones}
        self.head = m.body("h_head").mocapid[0]
        self.ih, self.inn = S.names.index("Head"), S.names.index("Neck")
        self.robot_g = [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name.startswith("r_")]
        self.human_g = [g for g in range(m.ngeom) if (m.geom(g).name or "").startswith("hg_")]
        self.tool_g = m.geom("r_tool").id
        rb = sorted({m.geom_bodyid[g] for g in self.robot_g})
        mount = set(rb[:2])   # robot base + first link sit on the pedestal (as in s6a)
        env = {n: m.geom(n).id for n in ("door_panel", "cart_frame", "tray", "pedestal")}
        self.env_pairs = [(g, e, n) for g in self.robot_g for n, e in env.items()
                          if not (n == "pedestal" and m.geom_bodyid[g] in mount)]
        self.carrier = m.body("carrier").id
        self.car_pos0 = m.body_pos[self.carrier].copy()
        self.car_quat0 = m.body_quat[self.carrier].copy()
        self.fromto = np.zeros(6)
        self._swept = (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_SPHERE)
        # their scene has one tool capsule sized for the longest tool (screwdriver 0.30 m); during gripper steps
        # (WS20, TCP 0.16 m) it pokes 14 cm past the fingertips into the tray and the door, so it is resized per group
        pos0, size0 = m.geom_pos[self.tool_g].copy(), m.geom_size[self.tool_g].copy()
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, m.geom_quat[self.tool_g])
        ax = R.reshape(3, 3)[:, 2]
        site = m.site("r_attachment_site").id
        flange = m.site_pos[site] if m.site_bodyid[site] == m.geom_bodyid[self.tool_g] else np.zeros(3)
        ends = [pos0 + ax * size0[1], pos0 - ax * size0[1]]
        self.tool_p0 = min(ends, key=lambda e: np.linalg.norm(e - flange))       # flange end of the capsule
        self.tool_u = (pos0 - self.tool_p0) / max(1e-9, np.linalg.norm(pos0 - self.tool_p0))   # towards the tip
        self.tool_r = size0[0]
        self.tool_len0 = 2 * size0[1] + size0[0]
        self.tool_len = self.tool_len0

    def set_tool(self, L):
        """Tool capsule from the flange to L along the tool axis (their fromto convention: segment ends at L - r)."""
        if abs(L - self.tool_len) < 1e-9:
            return
        h = max(0.005, (L - self.tool_r) / 2)
        self.m.geom_size[self.tool_g, 1] = h
        self.m.geom_pos[self.tool_g] = self.tool_p0 + self.tool_u * h
        self.tool_len = L

    def set_door(self, R, t):
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, R.flatten())
        self.m.body_pos[self.carrier] = R @ self.car_pos0 + t
        self.m.body_quat[self.carrier] = q

    def set_human(self, P):
        for bb, mid in self.mocap.items():
            ia, ib = self.bidx[bb]
            pa, pb = P[ia], P[ib]
            self.d.mocap_pos[mid] = (pa + pb) / 2
            self.d.mocap_quat[mid] = _quat_z_to(pb - pa)
        hv = P[self.ih] - P[self.inn]
        self.d.mocap_pos[self.head] = P[self.ih] + 0.08 * hv / (np.linalg.norm(hv) + 1e-9)

    def dist(self):
        return min(3.0, min(mujoco.mj_geomDistance(self.m, self.d, gr, gh, 3.0, self.fromto)
                            for gr in self.robot_g for gh in self.human_g))

    def contacts(self):
        """Robot geoms penetrating the door, cart, tray or pedestal box by more than 5 mm. Capsules are tested as 7
        spheres along their segment against the box SDF: MuJoCo's capsule-box distance misses a capsule that passes
        right through a thin box (it reports only the end cap)."""
        m, d, out = self.m, self.d, set()
        for g, e, n in self.env_pairs:
            if m.geom_type[e] != mujoco.mjtGeom.mjGEOM_BOX or m.geom_type[g] not in self._swept:
                if mujoco.mj_geomDistance(m, d, g, e, 0.1, self.fromto) < -PEN_M:
                    out.add((g, n))
                continue
            r, h = m.geom_size[g, 0], (m.geom_size[g, 1] if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_SPHERE else 0.0)
            pts = d.geom_xpos[g] + np.outer(np.linspace(-h, h, 7), d.geom_xmat[g].reshape(3, 3)[:, 2])
            loc = (pts - d.geom_xpos[e]) @ d.geom_xmat[e].reshape(3, 3)
            q = np.abs(loc) - m.geom_size[e]
            sdf = np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(1), 0)
            if (sdf - r).min() < -PEN_M:
                out.add((g, n))
        return out


class Worker:
    """To-be worker timeline of one draw: warped observed motion, then stays (last pose) or leaves."""

    AWAY = np.array([0.0, 0.0, -50.0])

    def __init__(self, S: Station, seed, after, walk_factor=None, asis=False):
        self.after = after
        wf = S.walk_factor if walk_factor is None else walk_factor
        if asis:  # human only: every element of the station is the worker's, observed motion as recorded
            codes = [c for c in S.stats if c not in ("UNOBS", "IDLE")]
            dr = S.worker_draw(seed, codes, wf)
            self.J, self.fps = S.Ja, S.fps_a
            self.t_motion = len(self.J) / self.fps
            self.map = None
        else:
            dr = S.worker_draw(seed, S.human_codes, wf)
            self.J, self.fps = S.J, S.fps
            seg = S.tl["segments"]
            tot = {}
            for s in seg:
                tot[s["code"]] = tot.get(s["code"], 0.0) + s["end_s"] - s["start_s"]
            # piecewise-linear map new time -> spliced time; blends between segments keep their 0.3 s
            new, old, tn, to = [0.0], [0.0], 0.0, 0.0
            for s in seg:
                if s["start_s"] > to:
                    tn += s["start_s"] - to; to = s["start_s"]
                    new.append(tn); old.append(to)
                f = dr[s["code"]][0] / tot[s["code"]] if s["code"] in dr and tot[s["code"]] > 0 else 1.0
                tn += (s["end_s"] - s["start_s"]) * f; to = s["end_s"]
                new.append(tn); old.append(to)
            self.map = (np.array(new), np.array(old))
            self.t_motion = tn
        self.draw = dr
        self.walk_s = dr.get("WALK", (0, 0))[1]
        work = sum(v[1] for v in dr.values()) + S.unobs_s
        # to-be: the warped splice is part of the work; as-is: the observed clip only supplies the motion
        self.total_s = work if asis else max(self.t_motion, work)

    def pose(self, t):
        """Joint positions at cycle time t, or None once the worker is away from the station."""
        if t >= self.total_s or (t >= self.t_motion and self.after == "leaves"):
            return None
        ts = min(t, self.t_motion)
        if self.map is not None:
            ts = float(np.interp(ts, *self.map))
        return self.J[min(len(self.J) - 1, int(ts * self.fps))]


def run_one(S: Station, sim: Sim, prog, worker: Worker, policy, p, ssm_on=True, frames=None, cam=None, rend=None):
    """One run. policy in P0 (their baseline: no SSM, no gate), P1, P2, P3. Returns the KPI dict."""
    ref = prog["ref"]
    wps = ref["waypoints"]
    R, tr = S.door_tf(p)
    sim.set_door(R, tr)
    tau_k = np.array([0.0] + [w["t_start_s"] + w["dur_s"] for w in wps])
    q_k = np.array([wps[0]["q_end_rad"]] + [w["q_end_rad"] for w in wps])
    T = float(tau_k[-1])

    def q_at(tau):
        """Joint-space linear interpolation between the IK waypoints (their replay), and the step index."""
        if tau >= T:
            return q_k[-1], len(wps)
        k = max(0, int(np.searchsorted(tau_k, tau, "right")) - 1)
        w = (tau - tau_k[k]) / max(1e-9, tau_k[k + 1] - tau_k[k])
        return q_k[k] * (1 - w) + q_k[k + 1] * w, k

    # active target of every program step (P2): last named target, carrier targets moved with the door
    tpos = {}
    for t in S.targets:
        pos = np.asarray(t["pos_m"], float)
        if t["on"] == "carrier":
            pos = R @ pos + tr
        cat_i = t["id"].rsplit("_", 1)
        tpos[(t["group_id"], ("tray" if cat_i[1] == "supply" else int(cat_i[1])))] = pos
    act, last = [], None
    for w in wps:
        nm = w["target"]
        if nm:
            last = None if nm == "home" else (w["group_id"], "tray" if nm.startswith("tray") else int(nm.split("_")[1]))
        act.append(tpos.get(last) if last else None)
    act.append(None)
    tool_of = {g["group_id"]: g["tool"]["tcp_offset_m"] for g in S.groups}
    win_of = {g["group_id"]: float(g["placement"].get("tracking_window_m") or 0.0) for g in S.groups}
    grp = [w["group_id"] for w in wps] + [wps[-1]["group_id"]]
    span = {}

    def tip(q):
        sim.d.qpos[sim.qadr] = q
        mujoco.mj_kinematics(sim.m, sim.d)
        return sim.d.geom_xpos[sim.tool_g].copy()

    Sx = S.ssm
    t, tau, wait, slow = 0.0, 0.0, 0.0, 0.0
    t_start = S.t_rs if policy != "P3" else None
    t_end = None
    dmin, n_act, n_viol, n_stop = 3.0, 0, 0, 0
    coll_prev, episodes, coll_s = set(), 0, 0.0
    t_door_free = worker.t_motion if worker.after == "leaves" else worker.total_s
    state = "idle"
    while t < T_MAX:
        P = worker.pose(t)
        sim.set_human(P if P is not None else worker.J[-1] + Worker.AWAY)
        if t_start is None and policy == "P3" and t >= max(S.t_rs, t_door_free):
            t_start = t
        started = t_start is not None and t >= t_start and tau < T
        q, k = q_at(tau)
        sim.set_tool(tool_of.get(grp[k], sim.tool_len0))
        x0 = tip(q)
        d = sim.dist()
        if started:
            x1 = tip(q_at(tau + DT)[0])
            v_nom = float(np.linalg.norm(x1 - x0) / DT)
            gate = True
            if policy == "P2" and act[k] is not None and P is not None:
                gate = float(np.linalg.norm(P - act[k], axis=1).min()) >= P2_CLEAR_M
            if not ssm_on:
                s = 1.0
            elif d < S.sp0:
                s = 0.0
            else:
                s = 1.0 if v_nom * Sx["T_r_s"] <= d - S.sp0 else (d - S.sp0) / (Sx["T_r_s"] * v_nom)
            n_act += 1
            sp_g = span.setdefault(grp[k], [t, t])
            sp_g[1] = t + DT
            dmin = min(dmin, d)
            n_viol += d < S.sp0 + s * v_nom * Sx["T_r_s"]
            state = ("waiting: worker within 1 m of target" if not gate else "stopped by SSM" if s == 0
                     else "slowed by SSM" if s < 1 else "moving")
            if gate:
                tau = min(T, tau + s * DT)
                slow += (1 - s) * DT
                n_stop += s == 0.0
            else:
                wait += DT
            sim.d.qpos[sim.qadr] = q
            mujoco.mj_kinematics(sim.m, sim.d)
            c = sim.contacts()
            episodes += len(c - coll_prev)
            coll_s += DT * bool(c)
            coll_prev = c
            if tau >= T and t_end is None:
                t_end = t + DT
        if frames is not None:
            sim.d.qpos[sim.qadr] = q
            mujoco.mj_forward(sim.m, sim.d)
            frames(t, tau, T, d, started, state if started else ("done" if tau >= T else "waiting to start"))
        t += DT
        if (t_end is not None or T <= 0) and (t >= worker.total_s or policy == "P0"):
            break
    done = t_end is not None
    t_end = t_end if done else T_MAX
    t_start = t_start if t_start is not None else T_MAX
    rc = t_end - t_start
    # carriage stays synchronised with the belt for a whole group program (s6a model), then returns
    v = S.v_belt * (1 + p["conveyor_rel"])
    trav = {g: v * (b - a_) for g, (a_, b) in span.items()}
    gmax = max(trav, key=trav.get) if trav else grp[0]
    track, track_win = trav.get(gmax, 0.0), win_of[gmax]
    st_cycle = max(worker.total_s, t_end) + S.handover_s
    return dict(station_cycle_s=st_cycle, robot_cycle_s=rc, robot_wait_s=wait, ssm_slowdown_s=slow,
                min_dist_m=dmin, collisions=episodes, collision_s=coll_s, ik_failures=prog["ik_failures"],
                tcp_err_mm=prog["tcp_err_mm"], robot_program_s=T, robot_start_s=t_start, robot_end_s=t_end,
                worker_s=worker.total_s, worker_motion_s=worker.t_motion, walk_s=worker.walk_s,
                ssm_violation_share=n_viol / max(1, n_act), ssm_stop_share=n_stop / max(1, n_act),
                track_travel_m=track, track_window_m=track_win, track_ok=track <= track_win + 1.0, finished=done)


# ------------------------------------------------------------------ matrix
COLS = ["option", "station", "policy", "worker_after", "draw", "seed", "walk_factor", "station_cycle_s",
        "robot_cycle_s", "robot_wait_s", "ssm_slowdown_s", "min_dist_m", "collisions", "ik_failures", "tcp_err_mm"]
EXTRA = ["over_cap", "worker_s", "walk_s", "robot_program_s", "robot_start_s", "robot_end_s", "ssm_violation_share",
         "ssm_stop_share", "collision_s", "track_travel_m", "track_window_m", "track_ok", "door_offset_mm", "conveyor_rel", "joint_accel",
         "rundown_s", "finished"]


def _job(args):
    rcd, option, st, k, policies, afters = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    S = Station(Path(rcd), option, st)
    sim = Sim(S)
    seed = SEEDS[k]
    p = draw_params(seed)
    prog = S.program(seed, p)
    R, tr = S.door_tf(p)
    off = max(float(np.linalg.norm(R @ np.asarray(t["pos_m"]) + tr - np.asarray(t["pos_m"]))) for t in S.targets
              if t["on"] == "carrier") * 1000
    rows = []
    for after in afters:
        w = Worker(S, seed, after)
        for pol in policies:
            r = run_one(S, sim, prog, w, pol, p)
            row = dict(option=option, station=st, policy=pol, worker_after=after, draw=k, seed=seed,
                       walk_factor=S.walk_factor, over_cap=r["station_cycle_s"] > TAKT_CAP, door_offset_mm=off,
                       conveyor_rel=p["conveyor_rel"], joint_accel=p["joint_accel"], rundown_s=prog["rundown_s"], **r)
            rows.append({c: (round(v, 4) if isinstance(v, float) else v) for c, v in row.items() if c in COLS + EXTRA})
    return rows, dict(station=st, draw=k, program_s=prog["program_s"], group_cycles=prog["group_cycles"],
                      breakdown=prog["breakdown"], by_code=prog["by_code"], group_stats=prog["group_stats"])


def pct(x, q):
    return float(np.percentile(np.asarray(x, float), q)) if len(x) else None


def summarize(rows, option):
    cells = {}
    for r in rows:
        cells.setdefault((r["station"], r["policy"], r["worker_after"]), []).append(r)
        cells.setdefault((r["station"], r["policy"], "pooled"), []).append(r)

    def cell(rs):
        c = [r["station_cycle_s"] for r in rs]
        return {"n": len(rs), "station_cycle_p50_s": round(pct(c, 50), 2), "station_cycle_p95_s": round(pct(c, 95), 2),
                "share_above_228_s": round(float(np.mean([x > TAKT_CAP for x in c])), 3),
                "robot_cycle_p50_s": round(pct([r["robot_cycle_s"] for r in rs], 50), 2),
                "robot_cycle_p95_s": round(pct([r["robot_cycle_s"] for r in rs], 95), 2),
                "robot_wait_mean_s": round(float(np.mean([r["robot_wait_s"] for r in rs])), 2),
                "ssm_slowdown_mean_s": round(float(np.mean([r["ssm_slowdown_s"] for r in rs])), 2),
                "min_dist_min_m": round(min(r["min_dist_m"] for r in rs), 3),
                "collisions_total": int(sum(r["collisions"] for r in rs)),
                "ik_failures_total": int(sum(r["ik_failures"] for r in rs)),
                "tcp_err_max_mm": round(max(r["tcp_err_mm"] for r in rs), 3),
                "track_ok_share": round(float(np.mean([bool(r["track_ok"]) for r in rs])), 3)}

    out = {"option": option, "cap_s": TAKT_CAP, "seeds": SEEDS, "cells": {}, "best_policy": {}}
    for (st, pol, wa), rs in sorted(cells.items()):
        out["cells"].setdefault(st, {}).setdefault(pol, {})[wa] = cell(rs)
    for st, pols in out["cells"].items():
        for wa in ("stays", "leaves", "pooled"):
            # fewest robot-environment collisions, then lowest p95 station cycle (0.1 s), then the larger
            # robot-worker distance (ties: when the worker is the bottleneck every policy gives the same cycle)
            cand = [(v[wa]["collisions_total"] > 0, round(v[wa]["station_cycle_p95_s"], 1), -v[wa]["min_dist_min_m"],
                     v[wa]["station_cycle_p95_s"], pol) for pol, v in pols.items() if wa in v]
            if not cand:
                continue
            *_, p95, pol = min(cand)
            c = pols[pol][wa]
            out["best_policy"].setdefault(st, {})[wa] = {
                "policy": pol, "station_cycle_p95_s": p95, "share_above_228_s": c["share_above_228_s"],
                "pass": bool(p95 <= TAKT_CAP and c["collisions_total"] == 0 and c["tcp_err_max_mm"] <= 1.0),
                "pass_cycle": bool(p95 <= TAKT_CAP), "pass_collisions": c["collisions_total"] == 0,
                "pass_tcp": c["tcp_err_max_mm"] <= 1.0}
    return out


def cmd_run(a):
    stations = a.stations or [d.name for d in sorted((Path(a.rcd) / "out_options" / a.option / "sim_package").iterdir())
                              if (d / "robot_reference.json").exists()]
    jobs = [(a.rcd, a.option, st, k, a.policies, a.after) for st in stations for k in range(a.draws)]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.jobs > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(a.jobs) as pool:
            res = []
            for i, r in enumerate(pool.imap_unordered(_job, jobs)):
                res.append(r)
                print(f"{i + 1}/{len(jobs)} {r[1]['station']} draw {r[1]['draw']}", flush=True)
    else:
        res = [_job(j) for j in jobs]
    rows = sorted([r for rs, _ in res for r in rs], key=lambda r: (r["station"], r["policy"], r["worker_after"], r["draw"]))
    progs = [m for _, m in res]
    with open(out / f"results_{a.option}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS + EXTRA)
        w.writeheader()
        w.writerows(rows)
    summ = summarize(rows, a.option)
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    (out / "seeds.json").write_text(json.dumps({
        "master_seed": MASTER_SEED, "draws": SEEDS,
        "derivation": "seed_k = SeedSequence(master_seed).generate_state(30)[k]; each random variable uses "
                      "default_rng([seed_k, crc32(name)]) with name in: line/conveyor_speed, line/door_pose, "
                      "line/joint_accel, <ST>/rundown, <ST>/worker/<CODE>. Use the same list and names for A and B.",
        "distributions": {"line/conveyor_speed": "uniform(-0.05, 0.05) relative", "line/door_pose":
                          "uniform(-1,1)^4 x (10 mm along, 10 mm across, 10 mm vertical, 1 deg yaw)",
                          "line/joint_accel": "uniform(1.0, 2.0) rad/s^2", "<ST>/rundown": "uniform(2.0, 3.5) s",
                          "<ST>/worker/<CODE>": "lognormal, mean and std of time_study element_stats, CV capped 0.5; "
                                                "times standard_s/mean_s; WALK times the option's walk factor"}}, indent=1))
    write_task_times(progs, summ, Path(a.rcd), a.option, out)
    print(json.dumps(summ["best_policy"], indent=1))


def write_task_times(progs, summ, rcd, option, out):
    entries = []
    stations = sorted({m["station"] for m in progs})
    for st in stations:
        S = Station(rcd, option, st)
        ms = [m for m in progs if m["station"] == st]
        for gi, g in enumerate(S.groups):
            cyc = [m["group_cycles"][gi] for m in ms]
            br, bc = {}, {}
            for m in ms:
                for k, v in m["breakdown"][gi].items():
                    br[k] = br.get(k, 0.0) + v / len(ms)
                for k, v in m["by_code"][gi].items():
                    bc[k] = bc.get(k, 0.0) + v / len(ms)
            best = summ["best_policy"].get(st, {})
            entries.append({
                "model": S.rb["model"], "group_id": g["group_id"], "station": st,
                "cycle_time_s": round(float(np.median(cyc)), 3),
                "cycle_time_p95_s": round(pct(cyc, 95), 3),
                "breakdown": {k: round(v, 3) for k, v in br.items()},
                "breakdown_by_code": {k: round(v, 3) for k, v in bc.items()},
                "ik_success_rate": round(sum(m["group_stats"][gi]["ok"] for m in ms)
                                         / max(1, sum(m["group_stats"][gi]["attempts"] for m in ms)), 4),
                "collision_free_rate": round(1 - sum(m["group_stats"][gi]["collisions"] for m in ms)
                                             / max(1, sum(m["group_stats"][gi]["ok"] for m in ms)), 4),
                "collision_free_rate_note": "s6a's own waypoint check (thin panel at the mean target depth); the replay "
                                            "against the package scene counts collisions in results_C.csv",
                "draws": len(ms),
                "station_cycle_with_best_policy": {wa: summ["cells"][st][v["policy"]][wa] | {"policy": v["policy"]}
                                                   for wa, v in best.items()},
                "sim": ("their s6a kinematic model (MuJoCo, vendor URDF M0617, DLS IK, trapezoid timing) re-run per draw "
                        "with joint acceleration U(1.0, 2.0) rad/s^2, rundown U(2.0, 3.5) s, conveyor +/-5%, door pose "
                        "+/-10 mm / +/-1 deg; cycle_time_s = median program time over the draws (no waits). Waits and "
                        "SSM slowdown per policy are in station_cycle_with_best_policy and results_C.csv."),
            })
    doc = {"method": "sim/handoff/option_runner.py on out_options/%s (v3 request, section 3 C)" % option, "entries": entries}
    (out / "robot_task_times.json").write_text(json.dumps(doc, indent=1))


# ------------------------------------------------------------------ worker-only stations (option A, B's WS10 and WS30)
def cmd_workeronly(a):
    """Station cycle = drawn standard times of the elements the option's s5 scenario gives the worker (WALK times the
    walk factor) + the unobserved block. No robot, so no motion replay is needed for the cycle."""
    rcd, out = Path(a.rcd), Path(a.out)
    stage = rcd / "out_options" / a.option
    alloc = json.loads((stage / "s5" / "allocation.json").read_text())
    rec = [x for x in alloc["scenarios"] if x.get("recommended")][0]
    ts = json.loads((stage / "s2" / "time_study.json").read_text())["element_stats"]
    rows = []
    for st in a.stations:
        stats = {r["code"]: r for r in ts if r["station"] == st}
        asg = [x for x in rec["assignment"] if x["station"] == st]
        assert all(x["resource"] in ("human", "helper") for x in asg), f"{a.option} {st} has a robot"
        codes = [x["code"] for x in asg if x["code"] != "UNOBS"]
        unobs = sum(x["time_s"] for x in asg if x["code"] == "UNOBS")
        for wf in a.walk:
            for k in range(a.draws):
                dr = draw_elements(stats, st, SEEDS[k], codes, wf)
                cyc = sum(v[1] for v in dr.values()) + unobs
                rows.append(dict(option=a.option, station=st, policy="none", worker_after="", draw=k, seed=SEEDS[k],
                                 walk_factor=wf, station_cycle_s=round(cyc, 4), over_cap=cyc > TAKT_CAP,
                                 worker_s=round(cyc, 4), walk_s=round(dr.get("WALK", (0, 0))[1], 4)))
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"results_{a.option}.csv"
    new = not (a.append and f.exists())
    with open(f, "w" if new else "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS + EXTRA)
        if new:
            w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows -> {f}")


def _load(path):
    rows = []
    for r in csv.DictReader(open(path)):
        for k, v in r.items():
            try:
                r[k] = float(v) if v not in ("", "True", "False") else (v == "True" if v else v)
            except ValueError:
                pass
        rows.append(r)
    return rows


def cmd_compare(a):
    """Combined summary of A, B and C on the same draws, plus the line view: per draw k the slowest station."""
    out = Path(a.out)
    rows = [r for p in a.inputs for r in _load(p)]
    cells = {}
    for r in rows:
        key = (r["option"], r["station"], r["policy"], r["worker_after"], r["walk_factor"])
        cells.setdefault(key, []).append(r)

    def stat(rs):
        c = [r["station_cycle_s"] for r in rs]
        return {"n": len(c), "station_cycle_p50_s": round(pct(c, 50), 1), "station_cycle_p95_s": round(pct(c, 95), 1),
                "share_above_228_s": round(float(np.mean([x > TAKT_CAP for x in c])), 3),
                "share_above_240_s": round(float(np.mean([x > 240 for x in c])), 3)}

    summ = {"cap_s": TAKT_CAP, "takt_s": 240, "seeds": SEEDS, "cells": [], "best_policy": {}, "line": [], "ranking": {}}
    for (o, st, pol, wa, wf), rs in sorted(cells.items(), key=lambda kv: tuple(map(str, kv[0]))):
        summ["cells"].append({"option": o, "station": st, "policy": pol, "worker_after": wa, "walk_factor": wf, **stat(rs)})
    # best policy per (option, robot station, worker case): same rule as summarize()
    for (o, st, pol, wa, wf), rs in cells.items():
        if pol == "none":
            continue
        b = summ["best_policy"].setdefault(o, {}).setdefault(st, {})
        p95 = pct([r["station_cycle_s"] for r in rs], 95)
        key = (sum(r["collisions"] for r in rs) > 0, round(p95, 1), -min(r["min_dist_m"] for r in rs))
        if wa not in b or key < tuple(b[wa]["_key"]):
            b[wa] = {"policy": pol, "station_cycle_p95_s": round(p95, 1), "_key": list(key)}
    for o in summ["best_policy"].values():
        for st in o.values():
            for v in st.values():
                v.pop("_key")

    def station_draws(o, st, wa, wf):
        pol = summ["best_policy"].get(o, {}).get(st, {}).get(wa, {}).get("policy", "none")
        rs = cells.get((o, st, pol, wa if pol != "none" else "", wf), [])
        return {int(r["draw"]): r["station_cycle_s"] for r in rs}

    scen = [("A", "walk 0.5 (IE assumption)", "", 0.5), ("A", "walk 0.7", "", 0.7), ("A", "walk 1.0", "", 1.0),
            ("B", "worker leaves", "leaves", 0.5), ("B", "worker stays", "stays", 0.5),
            ("C", "worker leaves", "leaves", 0.5), ("C", "worker stays", "stays", 0.5)]
    for o, label, wa, wf in scen:
        per = [station_draws(o, st, wa, wf) for st in ("WS10", "WS20", "WS30")]
        if not all(per):
            continue
        ks = sorted(set.intersection(*[set(p) for p in per]))
        worst = [max(p[k] for p in per) for k in ks]
        allc = [p[k] for p in per for k in ks]
        summ["line"].append({"option": o, "case": label, "draws": len(ks),
                             "slowest_station_p50_s": round(pct(worst, 50), 1),
                             "slowest_station_p95_s": round(pct(worst, 95), 1),
                             "share_draws_any_station_above_228_s": round(float(np.mean([w > TAKT_CAP for w in worst])), 3),
                             "share_station_cycles_above_240_s": round(float(np.mean([c > 240 for c in allc])), 3),
                             "p95_by_station_s": {st: round(pct(list(p.values()), 95), 1) for st, p in zip(("WS10", "WS20", "WS30"), per)},
                             "share_above_228_by_station": {st: round(float(np.mean([v > TAKT_CAP for v in p.values()])), 3)
                                                            for st, p in zip(("WS10", "WS20", "WS30"), per)}})
    for case in ("worker leaves", "worker stays"):
        pick = [l for l in summ["line"] if l["case"] in (case, "walk 0.5 (IE assumption)")]
        order = sorted(pick, key=lambda l: (l["share_draws_any_station_above_228_s"], l["slowest_station_p95_s"]))
        summ["ranking"][case] = [l["option"] for l in order]
    summ["ranking"]["ie_prediction"] = ["C", "B", "A"]
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps({"line": summ["line"], "ranking": summ["ranking"], "best_policy": summ["best_policy"]}, indent=1))


# ------------------------------------------------------------------ baseline check (their section 7)
def cmd_baseline(a):
    rcd = Path(a.rcd)
    nominal = {"conveyor_rel": 0.0, "door_dx_m": 0.0, "door_dy_m": 0.0, "door_dz_m": 0.0, "door_yaw_rad": 0.0,
               "joint_accel": 1.4}
    out = {}
    for st in a.stations or ["WS10", "WS20", "WS30"]:
        S = Station(rcd, a.option, st)
        sim = Sim(S)
        prog = {"ref": S.ref0, "ik_failures": sum(not w["ik_ok"] for w in S.ref0["waypoints"]), "tcp_err_mm": 0.0}
        o = {}
        for after, theirs in (("stays", "kpi_tobe.json"), ("leaves", "kpi_tobe_away.json")):
            w = Worker(S, SEEDS[0], after)
            w.map = None   # observed to-be motion at its recorded pace, as in their replay
            w.t_motion = len(S.J) / S.fps
            w.total_s = 1e9 if after == "stays" else w.t_motion   # their replay: hold forever / away after the data
            r = run_one(S, sim, prog, w, "P0", nominal, ssm_on=False)
            k = json.loads((S.pk / theirs).read_text()) if (S.pk / theirs).exists() else {}
            o[after] = {"ours_ssm_violation_share": round(r["ssm_violation_share"], 3),
                        "theirs_ssm_violation_share": k.get("share_active_time_ssm_violation"),
                        "ours_min_dist_m": round(r["min_dist_m"], 3),
                        "theirs_min_dist_m": k.get("min_distance_while_robot_active_m"),
                        "robot_program_s": round(r["robot_program_s"], 2)}
        # P3 with the whole worker load in series, at the IE means (scenario_matrix.yaml comment)
        hum = sum(v * (S.walk_factor if c == "WALK" else 1.0) for c, v in S.tl["standard_times_s"].items()) + S.unobs_s
        o["P3_whole_load_in_series_s"] = round(hum + float(S.ref0["cycle_time_s"]) + S.handover_s, 1)
        o["ie_station_cycle_s"] = S.st_times.get("cycle_s")
        o["ie_human_s"] = round(hum, 1)
        out[st] = o
    print(json.dumps(out, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))


# ------------------------------------------------------------------ video
def _camera(S, sim, m, d, high=False):
    tg = [t for t in S.targets if t["on"] == "carrier"]
    n = -np.asarray(tg[0]["approach_axis"], float)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = np.mean([t["pos_m"] for t in tg], axis=0) + n * 0.7
    cam.distance, cam.elevation = (4.6, -48.0) if high else (4.0, -24.0)   # high: see the worker behind the door
    cam.azimuth = math.degrees(math.atan2(-n[1], -n[0])) + 35.0
    r = mujoco.Renderer(m, 540, 960)
    mujoco.mj_forward(m, d)
    r.update_scene(d, camera=cam)
    if np.dot(np.asarray(r.scene.camera[0].pos) - cam.lookat, n) < 0:
        cam.azimuth += 180.0
    return r, cam


class VideoOut:
    def __init__(self, path, w, h, fps):
        self.p = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                   "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-vcodec", "libx264",
                                   "-pix_fmt", "yuv420p", "-crf", "26", str(path)], stdin=subprocess.PIPE)

    def write(self, img):
        self.p.stdin.write(np.ascontiguousarray(img).tobytes())

    def close(self):
        self.p.stdin.close()
        self.p.wait()


def _text(img, lines, color=(255, 255, 255)):
    import cv2
    img = img.copy()
    for i, (s, c) in enumerate(lines):
        y = 30 + 28 * i
        cv2.putText(img, s, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, s, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, c or color, 2, cv2.LINE_AA)
    return img


def render_run(S, policy, after, draw, every=0.2, hide_robot=False, asis=False, t_end=None, label=None, high=False,
               clean=False):
    """Frames (list of RGB arrays) of one run sampled every `every` s of cycle time, plus its KPIs."""
    sim = Sim(S)
    seed = SEEDS[draw]
    p = draw_params(seed)
    prog = S.program(seed, p)
    w = Worker(S, seed, after, asis=asis)
    if hide_robot:
        for g in range(sim.m.ngeom):
            nm = sim.m.geom(g).name or ""
            if g in sim.robot_g or nm in ("tray", "pedestal", "track_rail"):
                sim.m.geom_rgba[g, 3] = 0.0
        car = sim.m.body("carriage").id
        for i in range(sim.m.nsite):
            b = sim.m.site_bodyid[i]
            while b not in (0, car):
                b = sim.m.body_parentid[b]
            if b == car:
                sim.m.site_rgba[i, 3] = 0.0
    rend, cam = _camera(S, sim, sim.m, sim.d, high)
    frames = []
    nxt = [0.0]

    def grab(t, tau, T, d, started, state):
        if t_end is not None and t > t_end:
            return
        if t + 1e-6 < nxt[0]:
            return
        nxt[0] += every
        rend.update_scene(sim.d, camera=cam)
        img = rend.render()
        sp_ok = d >= S.sp0
        status = "no robot" if hide_robot else f"robot {min(tau, T):5.1f}/{T:.1f} s  {state}"
        if clean:   # customer-facing: one plain label, no codes or KPIs
            frames.append(_text(img, [(label, None)]) if label else img.copy())
            return
        lines = [(label or f"{S.st} {policy} worker {after}", None), (f"t {t:6.1f} s   {status}", None)]
        if not hide_robot:
            lines.append((f"min distance {d:+.2f} m  " + ("clear" if sp_ok else "inside SSM distance"),
                          (90, 220, 110) if sp_ok else (240, 80, 60)))
        frames.append(_text(img, lines))

    if hide_robot:
        prog = dict(prog, ref={**prog["ref"], "waypoints": prog["ref"]["waypoints"][:1]})
        prog["ref"]["waypoints"][0] = {**prog["ref"]["waypoints"][0], "dur_s": 0.0, "t_start_s": 0.0}
    kpi = run_one(S, sim, prog, w, policy, p, frames=grab)
    return frames, kpi


def cmd_video(a):
    S = Station(Path(a.rcd), a.option, a.station)
    frames, kpi = render_run(S, a.policy, a.after, a.draw, every=a.every, t_end=a.t_end)
    v = VideoOut(a.out, 960, 540, 25)
    for f in frames:
        v.write(f)
    v.close()
    print(json.dumps({k: (round(x, 2) if isinstance(x, float) else x) for k, x in kpi.items()}))


def cmd_sidebyside(a):
    """Left: worker alone, as-is (observed motion, no robot). Right: to-be worker plus robot under a policy."""
    S = Station(Path(a.rcd), a.option, a.station)
    ll = "Worker alone" if a.clean else f"{a.station}  worker alone (as-is, observed motion)"
    lr = "Worker + robot arm" if a.clean else f"{a.station}  worker + M0617 ({a.policy}), kitting halves walking"
    L, _ = render_run(S, "P1", "leaves", a.draw, every=a.every, hide_robot=True, asis=True, t_end=a.t_end,
                      label=ll, high=True, clean=a.clean)
    Rf, kr = render_run(S, a.policy, a.after, a.draw, every=a.every, t_end=a.t_end, label=lr, high=True,
                        clean=a.clean)
    n = max(len(L), len(Rf))
    v = VideoOut(a.out, 1920, 540, 25)
    for i in range(n):
        v.write(np.concatenate([L[min(i, len(L) - 1)], Rf[min(i, len(Rf) - 1)]], axis=1))
    v.close()
    print(json.dumps({"right_kpi": {k: (round(x, 2) if isinstance(x, float) else x) for k, x in kr.items()}}))


# ------------------------------------------------------------------ contrast: human only (as-is) vs option C
def cmd_contrast(a):
    """Same seeds: the as-is worker doing every element alone (no robot, no AMR) against option C's best policy."""
    rcd, out = Path(a.rcd), Path(a.out)
    summ = json.loads((out / "summary.json").read_text())
    rows = list(csv.DictReader(open(out / f"results_{a.option}.csv")))
    table = []
    for st in sorted(summ["cells"]):
        S = Station(rcd, a.option, st)
        ws = [Worker(S, SEEDS[k], "leaves", walk_factor=1.0, asis=True) for k in range(N_DRAWS)]
        cyc = [w.total_s for w in ws]
        er = ergo_station(S)
        table.append({"station": st, "setup": "human only (as-is)", "policy": "", "worker_after": "",
                      "station_cycle_p50_s": round(pct(cyc, 50), 1), "station_cycle_p95_s": round(pct(cyc, 95), 1),
                      "share_above_228_s": round(float(np.mean([c > TAKT_CAP for c in cyc])), 3),
                      "walk_mean_s": round(float(np.mean([w.walk_s for w in ws])), 1), **er["asis"],
                      "robot_wait_mean_s": "", "ssm_slowdown_mean_s": "", "min_dist_min_m": ""})
        for wa in ("stays", "leaves"):
            b = summ["best_policy"][st][wa]
            c = summ["cells"][st][b["policy"]][wa]
            rs = [r for r in rows if r["station"] == st and r["policy"] == b["policy"] and r["worker_after"] == wa]
            table.append({"station": st, "setup": "worker + M0617 + AMR kitting (option C)", "policy": b["policy"],
                          "worker_after": wa, "station_cycle_p50_s": c["station_cycle_p50_s"],
                          "station_cycle_p95_s": c["station_cycle_p95_s"], "share_above_228_s": c["share_above_228_s"],
                          "walk_mean_s": round(float(np.mean([float(r["walk_s"]) for r in rs])), 1), **er["tobe"],
                          "robot_wait_mean_s": c["robot_wait_mean_s"], "ssm_slowdown_mean_s": c["ssm_slowdown_mean_s"],
                          "min_dist_min_m": c["min_dist_min_m"]})
    keys = list(table[0])
    with open(out / "contrast.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(table)
    md = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    md += ["| " + " | ".join(str(r[k]) for k in keys) + " |" for r in table]
    (out / "contrast_table.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def ergo_station(S: Station):
    """Ergonomic load the pipeline reports, per door: s5 awkward-posture seconds (sum of ergo_s over the worker's
    elements, as-is = all elements, option C = the elements the worker keeps, after layout levers), plus their
    s2 RULA upper-arm and REBA trunk means recomputed on the motion each worker actually does (as-is clip,
    to-be splice; the unobserved block has no motion data)."""
    from rcd.common.io import load_config
    from rcd.s2_elements import ergonomics as ER
    from rcd.s2_elements import features as F
    from rcd.s5_optimize.data import build
    from rcd.s5_optimize.inputs import load_input
    cfg = load_config(S.rcd / "config" / "demo_regress.yaml")
    cfg = {**cfg, "out_root": str(S.stage.relative_to(S.rcd))}
    rows = build(cfg, *[load_input(cfg, n, {}) for n in ("time_study", "elements", "automatability", "robot_catalog",
                                                         "candidates", "robot_task_times")])["rows"]
    asis = sum(r.get("ergo_s", 0.0) for r in rows.values() if r["station"] == S.st)
    tobe = sum(rows[a["element_key"]].get("ergo_s", 0.0) * (1 - a.get("lever_reduction", 0.0))
               for a in S.rec["assignment"] if a["station"] == S.st and a["resource"] in ("human", "helper")
               and a["element_key"] in rows)
    out = {}
    for tag, J, fps, aw in (("asis", S.Ja, S.fps_a, asis), ("tobe", S.J, S.fps, tobe)):
        f = F.extract(J, S.names, fps)
        sc = ER.rula_reba(f)
        out[tag] = {"awkward_posture_s": round(aw, 1), "rula_upper_arm_mean": sc["rula_upper_arm"]["mean"],
                    "reba_trunk_mean": sc["reba_trunk"]["mean"], "motion_s": round(len(J) / fps, 1)}
    return out


# ------------------------------------------------------------------ self-check
def selfcheck():
    assert SEEDS == [int(s) for s in np.random.SeedSequence(MASTER_SEED).generate_state(N_DRAWS)]
    a = stream(SEEDS[3], "WS10/worker/INSERT").random()
    _ = stream(SEEDS[3], "line/joint_accel").random()                   # drawing another variable first ...
    assert a == stream(SEEDS[3], "WS10/worker/INSERT").random()          # ... does not change this one (CRN)
    x = [lognormal(np.random.default_rng(i), 10.0, 3.0) for i in range(20000)]
    assert abs(np.mean(x) - 10) < 0.1 and abs(np.std(x) - 3) < 0.15, (np.mean(x), np.std(x))
    y = [lognormal(np.random.default_rng(i), 10.0, 9.0) for i in range(20000)]
    assert abs(np.std(y) - 5) < 0.3, np.std(y)                           # CV capped at 0.5
    p = draw_params(SEEDS[0])
    assert 1.0 <= p["joint_accel"] <= 2.0 and abs(p["conveyor_rel"]) <= 0.05 and abs(p["door_dx_m"]) <= 0.01
    print("selfcheck ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "baseline", "video", "sidebyside", "contrast", "workeronly", "compare",
                                    "selfcheck"])
    ap.add_argument("--rcd", default="robot_consult_demo")
    ap.add_argument("--option", default="C")
    ap.add_argument("--stations", nargs="*")
    ap.add_argument("--station", default="WS10")
    ap.add_argument("--policies", nargs="*", default=["P1", "P2", "P3"])
    ap.add_argument("--policy", default="P1")
    ap.add_argument("--after", nargs="*", default=["stays", "leaves"])
    ap.add_argument("--draws", type=int, default=N_DRAWS)
    ap.add_argument("--draw", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--t-end", type=float, default=None)
    ap.add_argument("--every", type=float, default=0.4, help="video: cycle seconds between frames (25 fps output)")
    ap.add_argument("--walk", nargs="*", type=float, default=[0.5], help="workeronly: walk factors")
    ap.add_argument("--append", action="store_true", help="workeronly: append to an existing results_<OPT>.csv")
    ap.add_argument("--inputs", nargs="*", help="compare: results_<OPT>.csv files")
    ap.add_argument("--clean", action="store_true", help="sidebyside: plain labels only, no codes or KPIs")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.cmd in ("video", "sidebyside"):
        a.after = a.after[0]
    {"run": cmd_run, "baseline": cmd_baseline, "video": cmd_video, "sidebyside": cmd_sidebyside,
     "contrast": cmd_contrast, "workeronly": cmd_workeronly, "compare": cmd_compare,
     "selfcheck": lambda _: selfcheck()}[a.cmd](a)


if __name__ == "__main__":
    main()
