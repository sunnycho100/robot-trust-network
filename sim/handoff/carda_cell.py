#!/usr/bin/env python3
"""CarDA door-line cobot cell (SIM_HANDOFF.pdf v2, WS30), rebuilt in MuJoCo from the PDF's numbers.

The handoff's own package (out/sim_package: scene.xml, targets.json, robot_reference.json, replay.py) was not
shared, so this rebuilds its baseline replay: a Doosan M0617 on a 0.9 m pedestal on a linear track, a door box
fitted to the screw points, the worker as bone capsules driven by CarDA's XSens BVH, IK waypoints joined by
joint-space interpolation, and the capsule-distance and ISO/TS 15066 SSM KPIs of the PDF's section 7.

  MUJOCO_GL=egl python carda_cell.py WS30.bvh --mode asis --policy P1 --video ws30.mp4 --out kpi.json
  python carda_cell.py --selfcheck
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess

import mujoco
import numpy as np

# Doosan M0617 joint origins (xyz, rpy), all axes z, from doosan-robotics/doosan-robot
# dsr_description/urdf/m0617.urdf (Apache-2.0). Links drawn as capsules, as in the handoff.
M0617 = [((0, 0, 0.1525), (0, 0, 0)),
         ((0, 0.0345, 0), (0, -1.571, -1.571)),
         ((0.845, 0, 0), (0, 0, 1.571)),
         ((0, -0.734, 0), (1.571, 0, 0)),
         ((0, 0, 0), (-1.571, 0, 0)),
         ((0, -0.121, 0), (1.571, 0, 0))]
LINK_R = [0.09, 0.075, 0.06, 0.055, 0.05, 0.045]
QLIM = 2.617
VMAX = np.radians([100, 100, 150, 225, 225, 225])   # PDF section 4 (Doosan datasheet)
AMAX = 1.4                                           # rad/s^2, PDF assumption
TOOL, RUNDOWN, SYNC, APPROACH, V_LIN, V_TCP = 0.30, 2.51, 0.3, 0.06, 0.25, 1.0
PEDESTAL, START_AFTER, TRACK_RETURN = 0.9, 11.0, 0.75 / 0.5   # WS30 values
SCREW_PICK = 0.5                                     # s at the feeder, assumption (not in the PDF)
V_H, T_R, T_S, C_INTR = 1.6, 0.1, 0.326, 0.12        # SSM terms, PDF section 7; Z_d = Z_r = 0 (not given)
P2_CLEAR = 1.0
TAKT_CAP = 228.0
# Bone radii by parent segment name (first match), handoff: 0.05 to 0.13 m
RADII = [("ForeArm", .045), ("Arm", .055), ("Collar", .06), ("Shoulder", .055), ("Elbow", .045), ("Wrist", .045),
         ("Hand", .045), ("UpLeg", .08), ("Hips", .09), ("Hip", .08), ("Knee", .06), ("Leg", .06), ("Ankle", .05),
         ("Foot", .05), ("Toe", .04), ("Head", .10), ("Neck", .06), ("Chest", .12), ("Spine", .12)]
HANDS = ("RightWrist", "LeftWrist", "RightHand", "LeftHand")


def rpy_quat(r, p, y):
    cr, sr, cp, sp, cy, sy = (f(a / 2) for a in (r, p, y) for f in (math.cos, math.sin))
    return [cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy]


def fmt(v):
    return " ".join(f"{x:.5g}" for x in v)


# ---------------------------------------------------------------- BVH (CarDA XSens MVN, 60 fps)

def _axis_rot(axis, a):
    c, s = np.cos(a), np.sin(a)
    R = np.zeros(a.shape + (3, 3))
    i, j = {"X": (1, 2), "Y": (2, 0), "Z": (0, 1)}[axis]
    k = 3 - i - j
    R[..., k, k] = 1
    R[..., i, i] = c; R[..., j, j] = c; R[..., i, j] = -s; R[..., j, i] = s
    return R


def load_bvh(path):
    """Joint names, parents and world positions (frames, joints, 3) in metres, z up; plus fps."""
    tok = open(path).read().split()
    joints, stack, i = [], [], 0
    while tok[i] != "MOTION":
        t = tok[i]
        if t in ("ROOT", "JOINT", "End"):
            name = tok[i + 1] if t != "End" else joints[stack[-1]][0] + "_end"
            joints.append([name, stack[-1] if stack else -1, np.zeros(3), []]); i += 2
        elif t == "{":
            stack.append(len(joints) - 1); i += 1
        elif t == "}":
            stack.pop(); i += 1
        elif t == "OFFSET":
            joints[-1][2] = np.array(tok[i + 1:i + 4], float); i += 4
        elif t == "CHANNELS":
            n = int(tok[i + 1]); joints[-1][3] = tok[i + 2:i + 2 + n]; i += 2 + n
        else:
            i += 1
    dt, nch = float(tok[i + 5]), sum(len(j[3]) for j in joints)
    m = np.array(tok[i + 6:], float)
    m = m[:len(m) // nch * nch].reshape(-1, nch)   # some CarDA files keep the uncropped "Frames:" count
    n = len(m)
    R, P, col = [], [], 0
    for name, par, off, ch in joints:
        t, rot = np.tile(off, (n, 1)), np.tile(np.eye(3), (n, 1, 1))
        for c in ch:
            v = m[:, col]; col += 1
            if c.endswith("position"):
                t[:, "XYZ".index(c[0])] = v
            else:
                rot = rot @ _axis_rot(c[0], np.radians(v))
        if par < 0:
            R.append(rot); P.append(t)
        else:
            R.append(R[par] @ rot); P.append(P[par] + np.einsum("nij,nj->ni", R[par], t))
    pos = np.stack(P, 1)
    if np.argmax(np.abs(joints[1][2])) == 1:                         # spine offset along y: y up -> z up
        pos = pos[..., [0, 2, 1]] * np.array([1, -1, 1])
    h = np.median(pos[:, 0, 2])                                       # hip height: mm (CarDA) or cm
    pos = pos / (1000 if h > 100 else 100 if h > 10 else 1)
    return [j[0] for j in joints], [j[1] for j in joints], pos, 1 / dt


def bones(names, parents, pos):
    """(parent, child, radius) for every bone longer than 3 cm."""
    out = []
    for c, p in enumerate(parents):
        if p >= 0 and np.linalg.norm(pos[0, c] - pos[0, p]) > 0.03:
            out.append((p, c, next((r for k, r in RADII if k in names[p]), 0.06)))
    return out


# ---------------------------------------------------------------- targets from the worker's wrists

def dwell_points(hand, fps, v_max=0.05, min_s=0.6):
    """Centroid, start time and duration of each still period of one wrist."""
    speed = np.r_[0, np.linalg.norm(np.diff(hand, axis=0), axis=1) * fps]
    speed = np.convolve(speed, np.ones(9) / 9, "same")
    still, out, s = speed < v_max, [], None
    for i, st in enumerate(np.r_[still, False]):
        if st and s is None:
            s = i
        elif not st and s is not None:
            if i - s >= min_s * fps:
                out.append((hand[s:i].mean(0), s / fps, (i - s) / fps))
            s = None
    return out


def cluster(points, radius=0.08):
    """Greedy merge of dwell centroids; returns [pos, total_s, first_t] sorted by total dwell."""
    cl = []
    for p, t0, d in sorted(points, key=lambda x: -x[2]):
        for c in cl:
            if np.linalg.norm(c[0] - p) < radius:
                c[0] = (c[0] * c[1] + p * d) / (c[1] + d); c[1] += d; c[2] = min(c[2], t0); break
        else:
            cl.append([p.copy(), d, t0])
    return cl


# ---------------------------------------------------------------- scene

def robot_xml():
    xml, close = "", ""
    for i, (xyz, rpy) in enumerate(M0617):
        nxt = np.array(M0617[i + 1][0] if i < 5 else (0, 0, 0.04), float)
        geom = (f'<geom name="r_link{i+1}" type="capsule" fromto="0 0 0 {fmt(nxt)}" size="{LINK_R[i]}" material="robot"/>'
                if np.linalg.norm(nxt) > 0.02 else f'<geom name="r_link{i+1}" type="sphere" size="{LINK_R[i]}" material="robot"/>')
        xml += (f'<body name="link{i+1}" pos="{fmt(xyz)}" quat="{fmt(rpy_quat(*rpy))}">'
                f'<joint name="j{i+1}" axis="0 0 1" range="{-QLIM} {QLIM}"/>{geom}')
        close += "</body>"
    xml += (f'<geom name="r_tool" type="capsule" fromto="0 0 0.05 0 0 {TOOL - 0.01}" size="0.03" rgba="0.95 0.75 0.1 1"/>'
            f'<site name="tcp" pos="0 0 {TOOL}" size="0.01"/>')
    return xml + close


def build_xml(cell, human_bones):
    """cell: dict from layout(). human_bones: [(name, half_length, radius)]."""
    n, t = cell["n"], cell["t"]
    door_yaw = math.atan2(t[1], t[0])
    dq = fmt([math.cos(door_yaw / 2), 0, 0, math.sin(door_yaw / 2)])
    bq = fmt([math.cos(cell["base_yaw"] / 2), 0, 0, math.sin(cell["base_yaw"] / 2)])
    sites = "".join(f'<site name="target{k}" pos="{fmt(p)}" size="0.018" rgba="1 0 0 1"/>'
                    for k, p in enumerate(cell["targets"]))
    hum = "".join(f'<body name="h{k}" mocap="true"><geom name="h{k}" type="capsule" size="{r:.3f} {hl:.4f}" '
                  f'material="human"/></body>' for k, (hl, r) in enumerate(human_bones))
    tray = cell["tray"] - cell["base"]
    return f"""
<mujoco model="CarDA WS30 cell (rebuilt from SIM_HANDOFF v2)">
  <compiler angle="radian"/>
  <option timestep="0.002"/>
  <default><geom contype="0" conaffinity="0"/></default>
  <visual><global offwidth="1280" offheight="720"/><quality shadowsize="4096"/>
    <headlight diffuse="0.5 0.5 0.5" ambient="0.3 0.3 0.3"/></visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.25 0.28 0.32" rgb2="0.05 0.06 0.08" width="256" height="1536"/>
    <texture name="floor" type="2d" builtin="checker" rgb1="0.42 0.43 0.45" rgb2="0.38 0.39 0.41" width="300" height="300"/>
    <material name="floor" texture="floor" texrepeat="8 8"/>
    <material name="robot" rgba="0.82 0.84 0.86 1" specular="0.4"/>
    <material name="human" rgba="0.92 0.47 0.12 1"/>
  </asset>
  <worldbody>
    <light pos="{fmt(cell['center'] + [0, 0, 5])}" dir="0 0 -1" directional="true" castshadow="true" diffuse="0.6 0.6 0.6"/>
    <geom name="floor" type="plane" pos="{fmt(cell['center'] * [1, 1, 0])}" size="8 8 0.05" material="floor"/>
    <geom name="belt" type="box" pos="{fmt(cell['belt'])}" quat="{dq}" size="4 {cell['belt_w']/2:.3f} 0.004" rgba="0.12 0.12 0.13 1"/>
    <geom name="door" type="box" pos="{fmt(cell['door'])}" quat="{dq}" size="{fmt(cell['door_size'])}" rgba="0.07 0.07 0.09 1"/>
    <geom name="cart" type="box" pos="{fmt(cell['cart'])}" quat="{dq}" size="{fmt(cell['cart_size'])}" rgba="0.55 0.56 0.6 1"/>
    <geom name="rail" type="box" pos="{fmt(cell['base'] * [1, 1, 0] + [0, 0, 0.03])}" quat="{dq}" size="0.8 0.12 0.03" rgba="0.3 0.3 0.34 1"/>
    {sites}
    <body name="carriage" pos="{fmt(cell['base'] * [1, 1, 0])}">
      <joint name="track" type="slide" axis="{fmt(t)}"/>
      <geom name="pedestal" type="box" pos="0 0 {PEDESTAL/2}" size="0.17 0.17 {PEDESTAL/2 - 0.03}" rgba="0.75 0.76 0.78 1"/>
      <geom name="tray" type="box" pos="{fmt(tray * [1, 1, 0] + [0, 0, cell['tray'][2] - 0.02])}" size="0.12 0.12 0.02" rgba="0.15 0.6 0.2 1"/>
      <geom name="tray_leg" type="cylinder" fromto="{fmt(tray * [1, 1, 0])} {fmt(tray * [1, 1, 0] + [0, 0, cell['tray'][2] - 0.04])}" size="0.03" rgba="0.5 0.5 0.5 1"/>
      <body name="m0617" pos="0 0 {PEDESTAL}" quat="{bq}">
        <geom name="r_base" type="cylinder" fromto="0 0 0 0 0 0.1525" size="0.08" material="robot"/>
        {robot_xml()}
      </body>
    </body>
    {hum}
  </worldbody>
</mujoco>"""


def layout(targets, worker_xy, side, back=0.75, lateral=0.95):
    """Door, cart, base and tray around the screw targets. n points from the worker into the door."""
    c = targets.mean(0)
    n = c[:2] - worker_xy
    n = np.r_[n / np.linalg.norm(n), 0]
    t = np.array([-n[1], n[0], 0])
    face = max(p @ n for p in targets) + 0.01                  # box face on the deepest target (PDF section 5)
    along = [(p - c) @ t for p in targets]
    width = max(1.1, max(along) - min(along) + 0.5)
    zs = [p[2] for p in targets]
    z0, z1 = min(0.55, min(zs) - 0.15), max(1.45, max(zs) + 0.15)
    door = c + n * (face - c @ n + 0.04)
    door[2] = (z0 + z1) / 2
    cart = door.copy(); cart[2] = z0 / 2
    base = c - n * back + t * lateral * side
    base[2] = 0
    to_c = c[:2] - base[:2]
    return dict(n=n, t=t, center=c, targets=targets, base=base, base_yaw=math.atan2(to_c[1], to_c[0]),
                door=door, door_size=np.array([width / 2, 0.04, (z1 - z0) / 2]),
                cart=cart, cart_size=np.array([width / 2 - 0.05, 0.25, z0 / 2]),
                belt=np.r_[(c - n * 0.3)[:2], 0.004], belt_w=2.2,
                tray=base + n * 0.5 + t * side * 0.1 + np.array([0, 0, PEDESTAL - 0.05]),   # J1 limit is +-150 deg
                depth_offset=[face - p @ n for p in targets])


# ---------------------------------------------------------------- robot: IK and program

class Arm:
    def __init__(self, model):
        self.m, self.d = model, mujoco.MjData(model)
        jid = [model.joint(f"j{i}").id for i in range(1, 7)]
        self.qadr, self.dadr = model.jnt_qposadr[jid], model.jnt_dofadr[jid]
        self.site = model.site("tcp").id
        self.jp, self.jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))

    def fk(self, q):
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        return self.d.site_xpos[self.site].copy(), self.d.site_xmat[self.site].reshape(3, 3)[:, 2].copy()

    def ik(self, p, axis, seeds, rng=None):
        """Tool tip at p with the tool z along axis. Returns the solution nearest the first seed."""
        sols, rng = [], np.random.default_rng(0)   # fixed seeds: the same layout always gives the same program
        for q in list(seeds) + [rng.uniform(-2.5, 2.5, 6) for _ in range(30)]:
            q = np.array(q, float)
            for _ in range(200):
                x, z = self.fk(q)
                e = np.r_[p - x, np.cross(z, axis)]
                if np.linalg.norm(e[:3]) < 2e-4 and np.linalg.norm(e[3:]) < 2e-3:
                    sols.append(q.copy()); break
                mujoco.mj_comPos(self.m, self.d)
                mujoco.mj_jacSite(self.m, self.d, self.jp, self.jr, self.site)
                J = np.vstack([self.jp[:, self.dadr], self.jr[:, self.dadr]])
                dq = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(6), e)
                q = np.clip(q + dq * min(1.0, 0.5 / np.abs(dq).max()), -QLIM, QLIM)
            if len(sols) >= 3:
                break
        if not sols:
            return None
        return min(sols, key=lambda s: np.abs(s - seeds[0]).max())


def move_time(qa, qb, lin=0.0):
    """Synchronised joint move: slowest joint's trapezoid time (VMAX, AMAX); linear moves at least lin / V_LIN."""
    d = np.abs(qb - qa)
    t = np.where(d > VMAX ** 2 / AMAX, d / VMAX + VMAX / AMAX, 2 * np.sqrt(d / AMAX))
    return max(float(t.max()), lin / V_LIN, 0.05)


class Program:
    """Robot program for one station: per screw, feeder pick, approach, track sync, rundown, retract."""

    def __init__(self, arm, cell, rng=None, standoff=0.0):
        """standoff > 0 adds a waypoint that far in front of each screw, so joint moves stay off the door
        (the handoff's baseline has none and warns that its joint-space moves are unchecked)."""
        n = cell["n"]
        down = np.array([0, 0, -1.0])
        above, tray = cell["tray"] + [0, 0, 0.10], cell["tray"]
        q = arm.ik(above, down, [np.zeros(6)])
        if q is None:
            raise RuntimeError("no IK for the feeder pose")
        self.home, self.steps, self.ik_fail = q, [], 0

        def go(op, code, k, p, axis, lin=0.0, hold=0.0):
            nonlocal q
            qn = arm.ik(p, axis, [q], rng)
            if qn is None:
                self.ik_fail += 1; qn = q
            path = np.linalg.norm(np.diff([arm.fk(q + (qn - q) * s)[0] for s in np.linspace(0, 1, 11)], axis=0), axis=1).sum()
            # TCP cap 1.0 m/s (PDF section 4); the min-jerk peak is 1.875x the mean speed
            self.steps.append((op, code, k, q, qn, max(move_time(q, qn, lin), 1.875 * path / V_TCP)))
            if hold:
                self.steps.append((op + "_hold", code, k, qn, qn, hold))
            q = qn

        for k, p in enumerate(cell["targets"]):
            app = p - n * APPROACH
            go("move_j", "GET_TOOL", -1, above, down)
            go("screw_pick", "GET_TOOL", -1, tray, down, 0.10, SCREW_PICK)
            go("move_l", "GET_TOOL", -1, above, down, 0.10)
            if standoff:
                go("move_j", "REACH_FASTENER", k, p - n * standoff, n)
                go("move_l", "REACH_FASTENER", k, app, n, standoff - APPROACH)
            else:
                go("move_j", "REACH_FASTENER", k, app, n)
            self.steps.append(("track_sync", "FASTEN", k, q, q, SYNC))
            go("move_l", "FASTEN", k, p, n, APPROACH, RUNDOWN)
            go("move_l", "FASTEN", k, app, n, APPROACH)
            if standoff:
                go("move_l", "FASTEN", k, p - n * standoff, n, standoff - APPROACH)
        go("move_j", "RETURN", -1, above, down)
        self.steps.append(("track_return", "RETURN", -1, q, q, TRACK_RETURN))
        self.t0 = np.cumsum([0] + [s[5] for s in self.steps])
        self.T = float(self.t0[-1])

    def at(self, tau):
        i = min(int(np.searchsorted(self.t0, tau, "right")) - 1, len(self.steps) - 1)
        op, code, k, qa, qb, T = self.steps[max(i, 0)]
        s = np.clip((tau - self.t0[max(i, 0)]) / T, 0, 1)
        s = s * s * s * (10 - 15 * s + 6 * s * s)   # ponytail: min-jerk shape on the trapezoid's duration
        return qa + (qb - qa) * s, op, code, k

    def breakdown(self):
        out = {}
        for s in self.steps:
            out[s[1]] = out.get(s[1], 0) + s[5]
        return {k: round(v, 2) for k, v in out.items()}


# ---------------------------------------------------------------- replay

class VideoOut:
    def __init__(self, path, w, h, fps):
        self.p = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                   "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-vcodec", "libx264",
                                   "-pix_fmt", "yuv420p", "-crf", "26", path], stdin=subprocess.PIPE)

    def write(self, img):
        self.p.stdin.write(np.ascontiguousarray(img).tobytes())

    def close(self):
        self.p.stdin.close(); self.p.wait()


def human_cut(mode, cell, pos, fps):
    """asis: the whole clip, then the worker leaves. hold: stops at the end of INSERT and stays at the door
    (upper bound, PDF section 3). away: leaves at the end of INSERT (lower bound)."""
    return (len(pos) - 1) / fps if mode == "asis" else cell["insert_end_s"], mode != "hold"


def human_pose(pos, fps, t, t_cut, leave, away_dir):
    p = pos[min(int(min(t, t_cut) * fps), len(pos) - 1)]
    if leave and t > t_cut:
        p = p + away_dir * min(3.0, 1.2 * (t - t_cut))   # ponytail: glides off, no walking animation
    return p


def replay(model, prog, pos, fps, bone_list, cell, mode, policy, video=None, hud=None, t_max=TAKT_CAP):
    data = mujoco.MjData(model)
    arm = Arm(model)
    qadr = arm.qadr
    rgeoms = [model.geom(f"r_{g}").id for g in ["base", "tool"] + [f"link{i}" for i in range(1, 7)]]
    hgeoms = [model.geom(f"h{k}").id for k in range(len(bone_list))]
    fixed = [model.geom(g).id for g in ("door", "cart")]
    t_cut, leave = human_cut(mode, cell, pos, fps)
    t_human_done = t_cut + 2.5 if leave else math.inf   # P3: after the worker has left; never in hold
    away = -cell["n"]
    dt, t, tau, wait = 1 / 30, 0.0, 0.0, 0.0
    rows, prev_tcp, fromto, t_done = [], None, np.zeros(6), None
    rend = vid = None
    if video:
        rend = mujoco.Renderer(model, 720, 1280)
        vid = VideoOut(video, 1280, 720, 30)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = cell["center"] * [1, 1, 0] + [0, 0, 0.9]
        cam.distance, cam.elevation = 4.2, -26
        side = cell["t"] * cell.get("side", 1)
        v = -cell["n"] - side * 0.4
        cam.azimuth = math.degrees(math.atan2(v[1], v[0])) + 180
    while t < t_max:
        hp = human_pose(pos, fps, t, t_cut, leave, away)
        for k, (p, c, _) in enumerate(bone_list):
            a, b = hp[p], hp[c]
            data.mocap_pos[k] = (a + b) / 2
            z = (b - a) / max(np.linalg.norm(b - a), 1e-6)
            q = np.zeros(4)
            mujoco.mju_quatZ2Vec(q, z)
            data.mocap_quat[k] = q
        q, op, code, k = prog.at(tau)
        data.qpos[qadr] = q
        mujoco.mj_kinematics(model, data)
        tcp = data.site_xpos[arm.site].copy()
        v_r = 0.0 if prev_tcp is None else np.linalg.norm(tcp - prev_tcp) / dt
        d = min(mujoco.mj_geomDistance(model, data, g, h, 3.0, fromto) for g in rgeoms for h in hgeoms)
        sp = V_H * (T_R + T_S) + v_r * T_R + C_INTR
        active = cell["targets"][k] if k >= 0 else (cell["tray"] if code == "GET_TOOL" else None)
        worker_d = 9.9 if active is None else float(np.linalg.norm(hp - active, axis=1).min())
        crash = sum(mujoco.mj_geomDistance(model, data, g, f, 0.1, fromto) < -0.005
                    for g in rgeoms if not (g == rgeoms[1] and code == "FASTEN") for f in fixed)
        started = t >= START_AFTER and tau < prog.T
        allowed = started and {"P1": True, "P1s": d >= sp, "P2": worker_d >= P2_CLEAR,
                               "P3": t >= t_human_done}[policy]
        if started and not allowed:
            wait += dt
        rows.append(dict(t=round(t, 3), tau=round(tau, 3), op=op, code=code, target=k, dist=round(d, 4),
                         s_p=round(sp, 3), v_tcp=round(v_r, 3), worker_to_target=round(worker_d, 3),
                         moving=int(allowed), crash=int(crash)))
        if vid:
            rend.update_scene(data, cam)
            img = rend.render()
            if hud:
                img = hud(img, t, tau, prog.T, code, d, sp, allowed, policy, mode)
            vid.write(img)
        prev_tcp = tcp
        if allowed:
            tau += dt
        if tau >= prog.T and t_done is None:
            t_done = t + dt
        if tau >= prog.T and t >= min(t_human_done, t_max) and t >= START_AFTER:
            break
        t += dt
    if vid:
        vid.close()
    mv = [r for r in rows if r["moving"]]
    run = [r for r in rows if r["t"] >= START_AFTER]
    return rows, dict(
        mode=mode, policy=policy, robot_program_s=round(prog.T, 2), finished=bool(tau >= prog.T),
        robot_end_s=round(t_done, 2) if t_done else None, robot_wait_s=round(wait, 2),
        min_robot_worker_distance_m=round(min(r["dist"] for r in run), 3) if run else None,
        within_0p5m_frac=round(np.mean([r["dist"] < 0.5 for r in mv]), 3) if mv else None,
        ssm_violation_frac=round(np.mean([r["dist"] < r["s_p"] for r in mv]), 3) if mv else None,
        collision_frames=int(sum(r["crash"] > 0 for r in rows)), ik_failures=prog.ik_fail,
        breakdown=prog.breakdown())


def hud(img, t, tau, T, code, d, sp, ok, policy, mode):
    import cv2
    img = img.copy()
    col = (60, 200, 90) if d >= sp else (230, 60, 50)
    lines = [f"CarDA WS30 rebuild  policy {policy}  worker {mode}",
             f"t {t:5.1f} s   robot {min(tau, T):4.1f}/{T:.1f} s   {code if tau < T and t >= START_AFTER else 'idle'}",
             f"min distance {d:+.2f} m   S_p {sp:.2f} m   {'clear' if d >= sp else 'SSM violation'}"
             + ("" if ok or t < START_AFTER or tau >= T else "   robot waiting")]
    for i, s in enumerate(lines):
        cv2.putText(img, s, (24, 40 + 34 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, s, (24, 40 + 34 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.85, col if i == 2 else (255, 255, 255), 2, cv2.LINE_AA)
    return img


# ---------------------------------------------------------------- setup

def make_cell(bvh, n_targets=4, side=None, seed=0, standoff=0.0):
    names, parents, pos, fps = load_bvh(bvh)
    hand = next(i for i, nm in enumerate(names) if nm in ("RightWrist", "RightHand"))   # screwdriver hand
    pts = dwell_points(pos[:, hand], fps)
    # screws: the longest right-wrist dwell clusters 0.6 to 1.6 m high, after the robot's start (insert done)
    cl = [c for c in cluster(pts) if 0.6 < c[0][2] < 1.6 and c[2] >= START_AFTER]
    cl = sorted(cl, key=lambda c: -c[1])[:n_targets]
    targets = np.array([c[0] for c in sorted(cl, key=lambda c: c[2])])
    hips = pos[:, 0, :2]
    worker_xy = hips[int(START_AFTER * fps):].mean(0)
    # end of INSERT: last frame at the door (hips within 0.5 m) before the worker first walks over 0.8 m away
    dc = np.linalg.norm(hips - targets.mean(0)[:2], axis=1)
    far = np.nonzero((dc > 0.8) & (np.arange(len(dc)) > 2 * fps))[0]
    near = np.nonzero(dc[:far[0]] < 0.5)[0] if len(far) else []
    insert_end = near[-1] / fps if len(near) else START_AFTER
    rng = np.random.default_rng(seed)
    bl = bones(names, parents, pos)
    best = None
    for sd in ([side] if side else [1, -1]):
        for back, lat in ((0.75, 0.95), (0.6, 1.1), (0.9, 0.8)):
            cell = layout(targets, worker_xy, sd, back, lat)
            cell["side"] = sd
            model = mujoco.MjModel.from_xml_string(build_xml(cell, [(0.1, 0.05)] * len(bl)))
            try:
                prog = Program(Arm(model), cell, rng)
            except RuntimeError:
                continue
            # fewest IK failures, then the base far from where the worker stands, then a short program
            score = (prog.ik_fail, -np.linalg.norm(cell["base"][:2] - worker_xy), prog.T)
            if best is None or score < best[0]:
                best = (score, cell)
    cell = best[1]
    cell["insert_end_s"] = insert_end
    hb = [(np.linalg.norm(pos[0, c] - pos[0, p]) / 2, r) for p, c, r in bl]
    model = mujoco.MjModel.from_xml_string(build_xml(cell, hb))
    prog = Program(Arm(model), cell, rng, standoff)
    info = dict(bvh=bvh, frames=len(pos), fps=fps, mocap_s=round(len(pos) / fps, 1), joints=len(names), bones=len(bl),
                dwell_clusters=[dict(pos=np.round(c[0], 3).tolist(), dwell_s=round(c[1], 2), first_t=round(c[2], 2))
                                for c in cluster(pts)[:10]],
                insert_end_s=round(insert_end, 2), targets=np.round(targets, 3).tolist(), depth_offset_m=np.round(cell["depth_offset"], 3).tolist(),
                base=np.round(cell["base"], 3).tolist(), base_yaw=round(cell["base_yaw"], 3),
                reach_m=[round(float(np.linalg.norm(p - (cell["base"] + [0, 0, PEDESTAL + 0.1525]))), 3)
                         for p in targets],
                program_s=round(prog.T, 2), breakdown=prog.breakdown(), ik_failures=prog.ik_fail)
    return model, prog, pos, fps, bl, cell, info


def selfcheck():
    # BVH: root 1 m up (y up, cm), spine 10 cm up, wrist 10 cm along +x; root yawed 90 deg about y
    import os, tempfile
    txt = ("HIERARCHY\nROOT Hips\n{\nOFFSET 0 0 0\nCHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation\n"
           "JOINT Chest\n{\nOFFSET 0 10 0\nCHANNELS 3 Zrotation Xrotation Yrotation\n"
           "JOINT RightWrist\n{\nOFFSET 10 0 0\nCHANNELS 3 Zrotation Xrotation Yrotation\nEnd Site\n{\nOFFSET 0 5 0\n}\n}\n}\n}\n"
           "MOTION\nFrames: 1\nFrame Time: 0.016667\n0 100 0 0 0 90 0 0 0 0 0 0\n")
    f = tempfile.NamedTemporaryFile("w", suffix=".bvh", delete=False); f.write(txt); f.close()
    names, parents, pos, fps = load_bvh(f.name); os.unlink(f.name)
    assert names == ["Hips", "Chest", "RightWrist", "RightWrist_end"] and parents == [-1, 0, 1, 2] and round(fps) == 60
    # +x yawed 90 about y goes to -z (bvh), which is +y once z is up
    assert np.allclose(pos[0, 2], [0, 0.1, 1.1], atol=1e-6), pos[0, 2]
    # IK reaches a point in front of the robot with the tool horizontal, FK error under 1 mm
    cell = layout(np.array([[1.0, 0.2, 1.1], [1.0, -0.2, 1.1], [1.0, 0.0, 0.9], [1.0, 0.1, 1.3]]),
                  np.array([0.4, 0.0]), 1)
    model = mujoco.MjModel.from_xml_string(build_xml(cell, [(0.1, 0.05)]))
    arm, rng = Arm(model), np.random.default_rng(0)
    prog = Program(arm, cell, rng)
    assert prog.ik_fail == 0, prog.ik_fail
    for k, p in enumerate(cell["targets"]):
        q = [s for s in prog.steps if s[0] == "move_l" and s[2] == k][0][4]
        x, z = arm.fk(q)
        assert np.linalg.norm(x - p) < 1e-3 and z @ cell["n"] > 0.999, (x, p, z)
    assert abs(move_time(np.zeros(6), np.r_[1.4, 0, 0, 0, 0, 0]) - 2.0) < 1e-6   # 2*sqrt(1.4/1.4)
    print("selfcheck ok, program", round(prog.T, 1), "s", prog.breakdown())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bvh", nargs="?")
    ap.add_argument("--mode", default="asis", choices=["asis", "hold", "away"])
    ap.add_argument("--policy", default="P1", choices=["P1", "P1s", "P2", "P3"])
    ap.add_argument("--video")
    ap.add_argument("--out")
    ap.add_argument("--trace")
    ap.add_argument("--xml", help="also save the scene MJCF")
    ap.add_argument("--standoff", type=float, default=0.0, help="m in front of each screw for joint moves (0 = baseline)")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        return selfcheck()
    model, prog, pos, fps, bl, cell, info = make_cell(a.bvh, standoff=a.standoff)
    if a.xml:
        open(a.xml, "w").write(build_xml(cell, [(np.linalg.norm(pos[0, c] - pos[0, p]) / 2, r) for p, c, r in bl]))
    rows, kpi = replay(model, prog, pos, fps, bl, cell, a.mode, a.policy, a.video, hud if a.video else None)
    out = dict(cell=info, kpi=kpi)
    print(json.dumps(out, indent=1))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
    if a.trace:
        import csv
        with open(a.trace, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
