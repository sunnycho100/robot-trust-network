"""MuJoCo kinematic cell simulation: scene assembly, damped-least-squares IK on MuJoCo Jacobians, and
trapezoidal-profile timing of the s4 robot program with conveyor tracking.

Scene (MjSpec): robot (MuJoCo Menagerie MJCF via robot_descriptions, or an MJCF built by models/build.py from the
vendor's public ROS description) attached at the s4 base pose on a pedestal box,
a mocap "door panel" box that moves with the conveyor, and the screw presenter / part tray box beside the pedestal.
TCP = attachment_site + tool length along the site z axis (tool not modelled as a body).

IK (5-D task): TCP position + tool-axis direction (roll about the tool axis is free, as for a screwdriver bit or a
2-finger gripper on a symmetric part). dq = J^T (J J^T + lambda^2 I)^-1 e with the rotational rows projected
orthogonal to the tool axis; step clamped to 0.15 rad; joint limits enforced by clipping. Converged when
|e_pos| < 0.5 mm and tool-axis error < 0.5 deg. Waypoints are seeded from the previous solution (continuity);
the first one tries 24 deterministic random seeds and keeps the solution closest to the nominal elbow-up posture.

Timing: free moves (move_j) are synchronised joint-space trapezoids with the datasheet per-joint maximum speeds and
a common joint acceleration (process.JOINT_ACCEL_RAD_S2), and never faster than the collaborative TCP cap on the
straight-line distance. Linear moves (approach/retract) use a Cartesian trapezoid. Conveyor tracking: every target
on the carrier is re-evaluated at the time the robot reaches it, x(t) = x_ref + v * d * (t - t_mid), and IK is
re-solved there (also at the end of each run-down while the robot tracks).
"""
from __future__ import annotations

import importlib
import math
import warnings

import mujoco
import numpy as np

from rcd.s4_catalog import process as P
from rcd.s4_catalog import scene as S

from .models import build



def _mjcf_path(desc: str) -> str:
    if desc == "xarm7_mj_description":  # use the bare arm: our tool replaces the UFACTORY gripper
        p = importlib.import_module(f"robot_descriptions.{desc}").MJCF_PATH
        return p.replace("xarm7.xml", "xarm7_nohand.xml")
    return importlib.import_module(f"robot_descriptions.{desc}").MJCF_PATH


def _yaw_quat(yaw: float) -> list[float]:
    return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


class Cell:
    def __init__(self, desc: str, base_xyz, yaw: float, group: dict, tool_len: float):
        if desc.startswith(build.PREFIX):
            child = build.child_spec(desc[len(build.PREFIX):])
        else:
            child = mujoco.MjSpec.from_file(_mjcf_path(desc))
        spec = mujoco.MjSpec()
        spec.compiler.degree = False
        bx, by, bz = [float(v) for v in base_xyz]
        spec.worldbody.add_geom(name="pedestal", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.2, 0.2, bz / 2 - 0.005],
                                pos=[bx, by, bz / 2 - 0.005])
        geo = group["geometry"]
        n = np.asarray(geo["normal"], float)
        d = np.asarray(geo["conveyor_dir"], float)
        T = np.asarray(group["targets_m"], float)
        ext = max(0.3, float(np.ptp(T @ d)) + 0.3)
        hz = max(0.3, float(np.ptp(T[:, 2])) + 0.3)
        self.panel_ref = T.mean(0) - n * 0.02
        pb = spec.worldbody.add_body(name="panel", mocap=True, pos=self.panel_ref.tolist(),
                                     quat=_yaw_quat(math.atan2(d[1], d[0])))
        pb.add_geom(name="panel_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=[ext / 2, 0.01, hz / 2])
        tray = S.tray_position(group, np.asarray(base_xyz, float))
        spec.worldbody.add_geom(name="tray", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.15, 0.15, 0.03],
                                pos=(tray - np.array([0, 0, 0.035])).tolist())
        spec.option.integrator = child.option.integrator
        frame = spec.worldbody.add_frame(pos=[bx, by, bz], quat=_yaw_quat(yaw))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            frame.attach_body(child.worldbody.first_body(), "r_", "")
            self.m = spec.compile()
        self.d = mujoco.MjData(self.m)
        self.site = self.m.site("r_attachment_site").id
        self.body = self.m.site_bodyid[self.site]
        # arm joints = hinge joints on the kinematic chain of the TCP body
        chain, b = [], self.body
        while b > 0:
            chain.append(b)
            b = self.m.body_parentid[b]
        self.jnts = sorted(j for j in range(self.m.njnt) if self.m.jnt_bodyid[j] in chain
                           and self.m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE)
        self.qadr = np.array([self.m.jnt_qposadr[j] for j in self.jnts])
        self.dadr = np.array([self.m.jnt_dofadr[j] for j in self.jnts])
        lo = np.where(self.m.jnt_limited[self.jnts], self.m.jnt_range[self.jnts, 0], -2 * np.pi)
        hi = np.where(self.m.jnt_limited[self.jnts], self.m.jnt_range[self.jnts, 1], 2 * np.pi)
        self.lo, self.hi = np.maximum(lo, -2 * np.pi), np.minimum(hi, 2 * np.pi)
        self.L = tool_len
        self.panel_mocap = self.m.body("panel").mocapid[0]
        self.robot_geoms = {g for g in range(self.m.ngeom) if self.m.body(self.m.geom_bodyid[g]).name.startswith("r_")}
        self.env_geoms = {self.m.geom("panel_geom").id, self.m.geom("tray").id, self.m.geom("pedestal").id}
        first = sorted(b for b in range(self.m.nbody) if self.m.body(b).name.startswith("r_"))
        self.mount_links = set(first[:2])  # robot base + first moving link may touch the pedestal top plate
        self.n = len(self.jnts)

    # -------------------------------------------------------------- kinematics
    def fk(self, q):
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)
        R = self.d.site_xmat[self.site].reshape(3, 3)
        z = R[:, 2].copy()
        return self.d.site_xpos[self.site] + self.L * z, z

    def ik(self, p_t, z_t, q0, iters=250):
        q = np.clip(np.asarray(q0, float).copy(), self.lo, self.hi)
        z_t = z_t / np.linalg.norm(z_t)
        jp = np.zeros((3, self.m.nv))
        jr = np.zeros((3, self.m.nv))
        lam = 0.03
        err = (np.inf, np.inf)
        for _ in range(iters):
            tcp, z = self.fk(q)
            ep = p_t - tcp
            eo = np.cross(z, z_t)
            err = (float(np.linalg.norm(ep)), float(np.arcsin(min(1.0, np.linalg.norm(eo)))) if np.dot(z, z_t) > 0 else np.pi)
            if err[0] < 5e-4 and err[1] < np.radians(0.5):
                return q, True, err
            mujoco.mj_jac(self.m, self.d, jp, jr, tcp, self.body)
            Pz = np.eye(3) - np.outer(z, z)
            J = np.vstack([jp[:, self.dadr], Pz @ jr[:, self.dadr]])
            if np.dot(z, z_t) <= 0:  # pointing away: rotate about any axis orthogonal to z
                eo = np.cross(z, z_t + 0.5 * np.array([1.0, 0.3, 0.1]))
            e = np.concatenate([ep, eo])
            dq = J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(6), e)
            nrm = np.abs(dq).max()
            if nrm > 0.15:
                dq *= 0.15 / nrm
            q = np.clip(q + dq, self.lo, self.hi)
        return q, False, err

    def collides(self, q, panel_offset) -> bool:
        self.d.qpos[self.qadr] = q
        self.d.mocap_pos[self.panel_mocap] = self.panel_ref + panel_offset
        mujoco.mj_fwdPosition(self.m, self.d)
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            g1, g2 = c.geom1, c.geom2
            pair = {g1, g2}
            if pair & self.robot_geoms and pair & self.env_geoms:
                other_robot = [g for g in pair if g in self.robot_geoms]
                if self.m.geom_bodyid[other_robot[0]] in self.mount_links and self.m.geom("pedestal").id in pair:
                    continue  # robot base sits on the pedestal
                if c.dist < -0.005:
                    return True
        return False


def trap_time(dist: float, v: float, a: float) -> float:
    dist = abs(dist)
    if dist <= 0:
        return 0.0
    if dist >= v * v / a:
        return dist / v + v / a
    return 2.0 * math.sqrt(dist / a)


def movej_time(dq, vmax, tcp_dist) -> float:
    t = max(trap_time(dqi, vi, P.JOINT_ACCEL_RAD_S2) for dqi, vi in zip(np.abs(dq), vmax))
    return max(t, tcp_dist / P.COLLAB_TCP_CAP_M_S)


def simulate(cell: Cell, group: dict, joint_speed_deg_s, base_xyz, t_mid: float, rng_seed: int = 0) -> dict:
    """Run one robot cycle of the s4 program; returns timing, breakdown, IK statistics and a timeline."""
    vmax = np.radians(np.asarray(joint_speed_deg_s, float))
    if len(vmax) == 1:
        vmax = np.repeat(vmax, cell.n)
    if len(vmax) != cell.n:
        raise ValueError(f"joint speed list ({len(vmax)}) != arm joints ({cell.n})")
    poses = S.task_poses(group, base_xyz)
    geo = group["geometry"]
    v = geo["conveyor_speed_mm_s"] / 1000.0
    dvec = np.asarray(geo["conveyor_dir"], float)
    track = group.get("mounting") == "linear_track"

    def offset(t):
        return np.zeros(3) if track else dvec * v * (t - t_mid)

    def target(name, t):
        p, ax, moving = poses[name]
        return (p + offset(t) if moving else p.copy()), ax, moving

    rng = np.random.default_rng(rng_seed)
    nominal = np.clip(np.zeros(cell.n), cell.lo, cell.hi)
    # first solve: home pose with many seeds
    p0, ax0, _ = target("home", 0.0)
    best = None
    seeds = [nominal] + [rng.uniform(np.maximum(cell.lo, -np.pi), np.minimum(cell.hi, np.pi)) for _ in range(24)]
    for s in seeds:
        q, ok, err = cell.ik(p0, ax0, s)
        if ok:
            # prefer collision-free, then small joint excursions
            cost = float(np.sum((q - nominal) ** 2)) + (1e3 if cell.collides(q, offset(0.0)) else 0.0)
            if best is None or cost < best[0]:
                best = (cost, q)
    if best is None:
        q, ok, err = cell.ik(p0, ax0, nominal, iters=600)
        q_cur = q
    else:
        q_cur = best[1]
    stats = {"attempts": 0, "ok": 0, "collisions": 0, "pos_err_mm": [], "speed_violations": 0}

    def solve(name, t, q_seed):
        p, ax, mv = target(name, t)
        off = offset(t) if mv else np.zeros(3)
        q, ok, err = cell.ik(p, ax, q_seed)
        if not ok or cell.collides(q, off):  # retry from random seeds: closest converged, collision-free solution
            cands = []
            for _ in range(16):
                q2, ok2, err2 = cell.ik(p, ax, rng.uniform(np.maximum(cell.lo, -np.pi), np.minimum(cell.hi, np.pi)))
                if ok2:
                    cands.append((cell.collides(q2, off), float(np.abs(q2 - q_seed).max()), q2, err2))
            if cands:
                cands.sort(key=lambda c: (c[0], c[1]))
                if not ok or not cands[0][0]:
                    q, ok, err = cands[0][2], True, cands[0][3]
        return q, ok, err, p, mv

    def record(q, ok, err, mv, t):
        stats["attempts"] += 1
        stats["ok"] += int(ok)
        stats["pos_err_mm"].append(err[0] * 1000)
        if ok and cell.collides(q, offset(t) if mv else np.zeros(3)):
            stats["collisions"] += 1

    t = 0.0
    breakdown: dict[str, float] = {}
    by_code: dict[str, float] = {}
    timeline = []
    tcp_cur, _ = cell.fk(q_cur)
    cur_name = "home"
    for op in P.program(group["process_type"], group["n_targets"], group["codes"]):
        kind = op["op"]
        if kind == "move_j":
            dur = 1.0
            for _ in range(2):  # target moves while we travel: re-solve at the arrival time
                q, ok, err, p, mv = solve(op["target"], t + dur, q_cur)
                tcp_new, _ = cell.fk(q)
                dur = movej_time(q - q_cur, vmax, float(np.linalg.norm(tcp_new - tcp_cur)))
            record(q, ok, err, mv, t + dur)
            key = "move_free"
        elif kind == "move_l":
            p_end0, _, mv = target(op["target"], t)
            p_start0 = poses[cur_name][0] + (offset(t) if poses[cur_name][2] else 0)
            dist = float(np.linalg.norm(p_end0 - p_start0))
            dur = trap_time(dist, P.LIN_SPEED_M_S, P.LIN_ACCEL_M_S2)
            qm, okm, errm, _, _ = solve(op["target"], t + dur, q_cur)  # end pose at arrival (tracking)
            q, ok, err = qm, okm, errm
            if np.any(np.abs(q - q_cur) / max(dur, 1e-3) > vmax + 1e-9):
                stats["speed_violations"] += 1
            record(q, ok, err, mv, t + dur)
            key = "approach_retract"
        elif kind in ("track_start", "track_stop"):
            dur, q = P.TRACK_SYNC_S, q_cur
            key = "tracking_sync"
        else:
            dur = P.process_time(op["kind"])
            q = q_cur
            if op.get("target"):  # tracking during the process: pose must stay reachable to the end
                q, ok, err, p, mv = solve(op["target"], t + dur, q_cur)
                record(q, ok, err, mv, t + dur)
            key = P.BREAKDOWN_KEY[op["kind"]]
        timeline.append({"op": kind if kind != "process" else op["kind"], "target": op.get("target"),
                         "code": op["code"], "t_start_s": round(t, 3), "dur_s": round(dur, 3)})
        breakdown[key] = breakdown.get(key, 0.0) + dur
        by_code[op["code"]] = by_code.get(op["code"], 0.0) + dur
        t += dur
        q_cur = q
        tcp_cur, _ = cell.fk(q_cur)
        if op.get("target"):
            cur_name = op["target"]
    if track:  # carriage returns along the track to meet the next carrier
        W = v * t  # carrier travel during the robot cycle
        dur = trap_time(W, P.TRACK_RETURN_SPEED_M_S, P.LIN_ACCEL_M_S2)
        breakdown["track_return"] = dur
        by_code[timeline[-1]["code"]] = by_code.get(timeline[-1]["code"], 0.0) + dur
        timeline.append({"op": "track_return", "target": None, "code": timeline[-1]["code"],
                         "t_start_s": round(t, 3), "dur_s": round(dur, 3)})
        t += dur
    return {"cycle_time_s": t, "breakdown": breakdown, "by_code": by_code, "timeline": timeline, "stats": stats}
