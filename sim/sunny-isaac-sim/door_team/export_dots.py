"""Skinned dots for the website: the WS30 door job as moving point clouds, today (the real Xsens worker) and with the
robot team (door v3). Surface points are sampled once per rigid part in its own frame; each frame stores one rigid
transform per part. Same motion as door_split_v3: the v3 plan through rig.py (same IK as the render) and the
registered Xsens joints of CarDA cycle 003.

  usd-core python export_dots.py PLAN_DIR ASSET_DIR STATION_USD WORKER_NPZ OUT_DIR [--fps 15]

Writes OUT_DIR/data/{today,team}.bin and OUT_DIR/data/manifest.json. Binary layout per side (little endian, int16):
points (3 per point, mm, part-local) then for every frame: per part qx qy qz qw (x32767) tx ty tz (mm), then the
camera eye xyz and target xyz (mm). The manifest has the part list (name, colour, point count), fps, frames, the
camera field of view and the timeline events.
"""
import argparse
import json
import math
import os
import sys

import numpy as np
from pxr import Usd, UsdGeom

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rig  # noqa: E402
from kin import lerp, minjerk, rot, rot_between, tf  # noqa: E402

ap = argparse.ArgumentParser()
for a in ("plan", "assets", "station", "worker", "out"):
    ap.add_argument(a)
ap.add_argument("--fps", type=float, default=15.0)
ap.add_argument("--density", type=float, default=1.0, help="scale every part's point count")
ap.add_argument("--check", action="store_true", help="clearance check on the team motion only, write nothing")
args = ap.parse_args()
rng = np.random.default_rng(3)
P = np.load(os.path.join(args.plan, "plan.npz"))
J = json.load(open(os.path.join(args.plan, "plan.json")))
L = J["layout"]
FPS_PLAN = J["fps"]
STEP = int(round(FPS_PLAN / args.fps))
FR = list(range(0, len(P["t"]), STEP))
T_END = float(P["t"][-1])


# ------------------------------------------------------------------ surface sampling
def tri_sample(V, F, n):
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    if area.sum() <= 0 or n <= 0:
        return np.zeros((0, 3))
    i = rng.choice(len(F), n, p=area / area.sum())
    u, v = rng.random(n), rng.random(n)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    return a[i] + (b[i] - a[i]) * u[:, None] + (c[i] - a[i]) * v[:, None]


def box_mesh(c=(0, 0, 0), s=(1, 1, 1)):
    c, h = np.asarray(c, float), np.asarray(s, float) / 2
    V = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float) * h + c
    F = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
                  [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    return V, F


def cyl_mesh(r, h, axis="Z", n=16, caps=True):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.c_[r * np.cos(a), r * np.sin(a)]
    V = np.r_[np.c_[ring, np.full(n, -h / 2)], np.c_[ring, np.full(n, h / 2)], [[0, 0, -h / 2], [0, 0, h / 2]]]
    F = [[i, (i + 1) % n, n + (i + 1) % n] for i in range(n)] + [[i, n + (i + 1) % n, n + i] for i in range(n)]
    if caps:
        F += [[2 * n, (i + 1) % n, i] for i in range(n)] + [[2 * n + 1, n + i, n + (i + 1) % n] for i in range(n)]
    V = V[:, {"X": [2, 0, 1], "Y": [1, 2, 0], "Z": [0, 1, 2]}[axis]]
    return V, np.array(F)


def capsule_pts(r, ln, n):
    """Capsule from z = 0 to z = ln, radius r."""
    k = int(n * (2 * np.pi * r * ln) / (2 * np.pi * r * ln + 4 * np.pi * r * r))
    a, z = rng.random(k) * 2 * np.pi, rng.random(k) * ln
    side = np.c_[r * np.cos(a), r * np.sin(a), z]
    d = rng.normal(size=(n - k, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    caps = r * d + np.where(d[:, 2:3] > 0, [[0, 0, ln]], [[0, 0, 0]])
    return np.r_[side, caps]


def sphere_pts(r, c, n):
    d = rng.normal(size=(n, 3))
    return r * d / np.linalg.norm(d, axis=1, keepdims=True) + np.asarray(c, float)


def prim_mesh(p):
    """Triangles of one gprim in its own frame, or None."""
    if p.IsA(UsdGeom.Mesh):
        m = UsdGeom.Mesh(p)
        V = np.array(m.GetPointsAttr().Get() or [], float)
        cnt, idx = m.GetFaceVertexCountsAttr().Get(), m.GetFaceVertexIndicesAttr().Get()
        if len(V) == 0 or cnt is None:
            return None
        F, o = [], 0
        for c in cnt:
            F += [[idx[o], idx[o + j], idx[o + j + 1]] for j in range(1, c - 1)]
            o += c
        return V, np.array(F)
    if p.IsA(UsdGeom.Cube):
        return box_mesh((0, 0, 0), [UsdGeom.Cube(p).GetSizeAttr().Get()] * 3)
    if p.IsA(UsdGeom.Cylinder):
        c = UsdGeom.Cylinder(p)
        return cyl_mesh(c.GetRadiusAttr().Get(), c.GetHeightAttr().Get(), c.GetAxisAttr().Get())
    if p.IsA(UsdGeom.Sphere):
        r = UsdGeom.Sphere(p).GetRadiusAttr().Get()
        return sphere_pts(r, (0, 0, 0), 64), None
    return None


def sample_prims(prims, T_frame, n_per_m2, nmin=30, nmax=4000, skip=()):
    """Surface points of every visible gprim under prims, expressed in the frame T_frame (world 4x4)."""
    xc = UsdGeom.XformCache()
    Ti = np.linalg.inv(T_frame)
    Vs, Fs, extra = [], [], []
    for root in prims:
        for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if any(s in p.GetName() for s in skip) or not p.IsA(UsdGeom.Gprim):
                continue
            if UsdGeom.Imageable(p).ComputePurpose() == "guide" or UsdGeom.Imageable(p).ComputeVisibility() == "invisible":
                continue
            m = prim_mesh(p)
            if m is None:
                continue
            M = Ti @ rig.gf2np(xc.GetLocalToWorldTransform(p))
            V = (M[:3, :3] @ m[0].T).T + M[:3, 3]
            if m[1] is None:
                extra.append(V)
                continue
            Fs.append(m[1] + sum(len(v) for v in Vs))
            Vs.append(V)
    if not Vs and not extra:
        return np.zeros((0, 3))
    pts = []
    if Vs:
        V, F = np.concatenate(Vs), np.concatenate(Fs)
        area = 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1).sum()
        pts.append(tri_sample(V, F, int(np.clip(area * n_per_m2 * args.density, nmin, nmax))))
    return np.concatenate(pts + extra)


# ------------------------------------------------------------------ the two sides
class Side:
    def __init__(self, name):
        self.name, self.parts = name, []        # (name, colour, points (n,3) local, transforms (F,4,4))

    def add(self, name, colour, pts, Ts):
        self.parts.append((name, colour, np.asarray(pts, float), np.asarray(Ts, float)))


def quat(R):
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = math.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2
    y = math.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2
    z = math.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2
    x, y, z = math.copysign(x, R[2, 1] - R[1, 2]), math.copysign(y, R[0, 2] - R[2, 0]), math.copysign(z, R[1, 0] - R[0, 1])
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def write_side(side, cams, path, frames=None):
    if frames is not None:
        side.parts = [(n, c, p, Ts[:frames]) for n, c, p, Ts in side.parts]
        cams = cams[:frames]
    pts = np.concatenate([p[2] for p in side.parts])
    assert np.abs(pts).max() < 32.0, "part-local points must fit int16 mm"
    out = [np.round(pts * 1000).astype("<i2").ravel()]
    nF = len(cams)
    for f in range(nF):
        rows = []
        for _, _, _, Ts in side.parts:
            T = Ts[f]
            q = quat(T[:3, :3])
            if f and np.dot(q, side_prevq.get(id(Ts), q)) < 0:
                q = -q                                   # keep quaternions continuous for interpolation
            side_prevq[id(Ts)] = q
            rows.append(np.r_[np.round(q * 32767), np.round(T[:3, 3] * 1000)])
        e, g = cams[f]
        rows.append(np.r_[np.round(np.asarray(e) * 1000), np.round(np.asarray(g) * 1000)])
        out.append(np.concatenate(rows).astype("<i2"))
    blob = np.concatenate(out).tobytes()
    open(path, "wb").write(blob)
    return dict(points=int(len(pts)), parts=[dict(name=n, c=c, n=int(len(p))) for n, c, p, _ in side.parts], frames=nF,
                bytes=len(blob))


side_prevq = {}

# ------------------------------------------------------------------ station: trolley and door, sampled in the trolley frame
sst = Usd.Stage.Open(args.station)
CAR_X0, CAR_Y = L["CARRIER"]
DDX, DDY, DZ0 = L["DOOR_ON_CARRIER"]
T_DOOR_ON_CAR = tf(rot(2, math.pi), (DDX, DDY, DZ0))
car_prim = sst.GetPrimAtPath("/World/WS30/Carrier")
T_car0 = tf(None, (CAR_X0, CAR_Y, 0.0))
door_pts = sample_prims([sst.GetPrimAtPath("/World/WS30/Carrier/door")], T_car0, 2200, nmax=3200)
PLAIN = ("door", "klt0", "klt1", "shelf0", "shelf1", "shelf0_strut", "shelf1_strut", "trim_panel", "trim_holder")
trolley_pts = sample_prims([c for c in car_prim.GetChildren() if c.GetName() not in PLAIN], T_car0, 700, nmax=2600)
print(f"[dots] door {len(door_pts)} pts, trolley {len(trolley_pts)} pts")
times = [float(P["t"][k]) for k in FR]
T_car = [tf(None, (CAR_X0 + L["BELT"] * t, CAR_Y, 0.0)) for t in times]

# ------------------------------------------------------------------ TODAY: the real worker (Xsens) with the door on its trolley
W = np.load(args.worker)
Jw, wfps = W["joints"].astype(float), float(W["fps"])
names, par = [str(n) for n in W["names"]], [int(p) for p in W["parents"]]
RAD = [("Hip", 0.075), ("Knee", 0.06), ("Ankle", 0.045), ("Chest", 0.135), ("Neck", 0.055), ("Collar", 0.06),
       ("Shoulder", 0.055), ("Elbow", 0.045)]
today = Side("today")
today.add("door", "w", door_pts, T_car)
today.add("trolley", "w", trolley_pts, T_car)
wk = [min(int(round(t * wfps)), len(Jw) - 1) for t in times]
for c, p in enumerate(par):
    if p < 0 or np.linalg.norm(Jw[0, c] - Jw[0, p]) < 0.03:
        continue
    pn, cn = names[p], names[c]
    r = next((v for k, v in RAD if k in pn), 0.11 if pn == "Hips" else 0.06)
    if pn == "Hips" and cn == "Chest":
        r = 0.13
    ln = float(np.median(np.linalg.norm(Jw[:, c] - Jw[:, p], axis=1)))
    pts = capsule_pts(r, ln, int(np.clip(2 * np.pi * r * ln * 2600, 40, 420)))
    if cn == "Head":                                   # head sphere rides the neck bone
        pts = np.r_[pts, sphere_pts(0.1, (0, 0, ln + 0.09), 260)]
    if cn.endswith("Wrist"):
        pts = np.r_[pts, sphere_pts(0.045, (0, 0, ln + 0.07), 50)]
    Ts = []
    for k in wk:
        a, b = Jw[k, p], Jw[k, c]
        Ts.append(tf(rot_between(np.array([0, 0, 1.0]), b - a), a))
    today.add(f"worker_{cn}", "w", pts, Ts)


def orbit(t):
    """ws30_station.orbit: the human-alone clip camera, riding with the trolley (focal 14 mm)."""
    x = CAR_X0 + L["BELT"] * t
    return (x - 1.2, 7.0, 3.3), (x + 0.7, 3.8, 0.8)


cams_today = [orbit(t) for t in times]

# ------------------------------------------------------------------ WITH A ROBOT TEAM: v3 plan through the same rigs as the render
ast = Usd.Stage.CreateInMemory()
UsdGeom.SetStageUpAxis(ast, UsdGeom.Tokens.z)
for path, url in [("/World/UR", "UniversalRobots/ur20/ur20.usd"), ("/World/G1", "Unitree/G1/g1.usd")]:
    ast.DefinePrim(path, "Xform").GetReferences().AddReference(os.path.join(args.assets, url))
PED, RAILS = L["PED"], L["RAILS"]
T_MOUNT = tf(None, (0, 0, PED))
team = Side("team")
team.add("trolley", "w", trolley_pts, T_car)
team.add("door", "w", door_pts, [P["door"][k] @ np.linalg.inv(T_DOOR_ON_CAR) for k in FR])
gl = np.array(L["GLASS_PTS"], float)
gl_tris = np.array([[0, i, i + 1] for i in range(1, len(gl) - 1)])
team.add("glass", "w", tri_sample(gl, gl_tris, 900), [P["glass"][k] for k in FR])


def lifter_pts():
    """Handheld suction-cup glass lifter (door_team.py draws the same shapes): frame at the handle centre,
    x along the bar, +y towards the glass."""
    return np.r_[tri_sample(*cyl_mesh(0.016, 0.26, "X"), 70),
                 tri_sample(*box_mesh((0, 0.065, 0), (0.46, 0.03, 0.03)), 70),
                 tri_sample(*box_mesh((-0.12, 0.03, 0), (0.02, 0.06, 0.02)), 10), tri_sample(*box_mesh((0.12, 0.03, 0), (0.02, 0.06, 0.02)), 10),
                 tri_sample(*cyl_mesh(0.06, 0.012, "Y"), 50) + [-0.20, 0.09, 0], tri_sample(*cyl_mesh(0.06, 0.012, "Y"), 50) + [0.20, 0.09, 0]]


if "lifter" in P.files:
    team.add("lifter", "w", lifter_pts(), [P["lifter"][k] for k in FR])
TIPV = np.array([0.20, 0.0, 0.075])
drv = np.r_[tri_sample(*box_mesh((0, 0, 0.015), (0.045, 0.04, 0.12)), 60), tri_sample(*cyl_mesh(0.032, 0.17, "X"), 120) + [0.03, 0, 0.075],
            tri_sample(*cyl_mesh(0.006, 0.09, "X"), 30) + [0.17, 0, 0.075]]
team.add("driver", "w", drv, [P["driver"][k] for k in FR])
team.add("screw", "w", np.r_[tri_sample(*cyl_mesh(0.016, 0.008, "X"), 30), tri_sample(*cyl_mesh(0.006, 0.03, "X"), 14) + [0.016, 0, 0]],
         [P["screw"][k] for k in FR])

xc = UsdGeom.XformCache()
for name in J["arms"]:
    arm = rig.Arm(ast, "/World/UR", T_MOUNT, None)
    G = rot_between(np.array([0, 0, 1.0]), arm.axis_l)
    arm.finger_l = G[:3, 0]
    root_W = rig.gf2np(xc.GetLocalToWorldTransform(ast.GetPrimAtPath("/World/UR")))
    link_rel = {ln: arm.rel(ast.GetPrimAtPath(ln)) for ln in arm.links + [arm.base]}
    pts_link = {ln: sample_prims([ast.GetPrimAtPath(ln)], root_W @ link_rel[ln], 1500, nmax=900) for ln in arm.links + [arm.base]}
    # gripper (closed) on the tool link
    grip = np.r_[tri_sample(*box_mesh((0, 0, 0.08), (0.13, 0.09, 0.16)), 120)] @ G.T
    fingers = np.r_[tri_sample(*box_mesh((0.0625, 0, 0.25), (0.025, 0.07, 0.14)), 50), tri_sample(*box_mesh((-0.0625, 0, 0.25), (0.025, 0.07, 0.14)), 50)] @ G.T
    pts_link[arm.tool] = np.r_[pts_link[arm.tool], grip, fingers]
    # carriage, pedestal and the UR base in the carriage frame
    carr = np.r_[tri_sample(*box_mesh((0, 0, 0.15), (0.6, 0.6, 0.06)), 260), tri_sample(*box_mesh((0, 0, 0.18 + (PED - 0.18) / 2), (0.34, 0.34, PED - 0.18)), 220)]
    base = (T_MOUNT[:3, :3] @ (link_rel[arm.base][:3, :3] @ pts_link[arm.base].T + link_rel[arm.base][:3, 3:4])).T + T_MOUNT[:3, 3]
    Ts = {ln: [] for ln in arm.links}
    Tc_all = []
    for k in range(len(P["t"])):
        Tc = tf(None, (P[f"{name}.cx"][k], RAILS[name], 0.0))
        q, ok = arm.solve(Tc, P[f"{name}.tip"][k], P[f"{name}.axis"][k], finger_w=np.array([0, 1.0, 0]))
        if k % STEP == 0:
            T = arm.fk(q)
            for ln in arm.links:
                Ts[ln].append(Tc @ T_MOUNT @ T[ln])
            Tc_all.append(Tc)
    team.add(f"{name}_carriage", "a", np.r_[carr, base], Tc_all)
    for ln in arm.links:
        team.add(f"{name}_{ln.rsplit('/', 1)[-1]}", "a", pts_link[ln], Ts[ln])
    print(f"[dots] {name}: {sum(len(v) for v in pts_link.values()) + len(carr)} pts")

g1 = rig.G1(ast, "/World/G1", J["joints"][:13])
root_W = rig.gf2np(xc.GetLocalToWorldTransform(ast.GetPrimAtPath("/World/G1")))
g1_links = [g1.base] + g1.links
g1_pts = {}
for ln in g1_links:
    rel = g1.rel(ast.GetPrimAtPath(ln)) if ln != g1.base else g1.T_base
    g1_pts[ln] = sample_prims([ast.GetPrimAtPath(ln)], root_W @ rel, 2600, nmin=20, nmax=500)
print(f"[dots] G1: {sum(len(v) for v in g1_pts.values())} pts per robot over {len(g1_links)} links")
for name in [h for h in J["humanoids"] if h in ("HUMANOID_A", "HUMANOID_B")]:   # the job's G1s, not the aisle walker
    Ts = {ln: [] for ln in g1_links}
    for k in range(len(P["t"])):
        T_root, q, _ = g1.solve(P[f"{name}.root"][k], P[f"{name}.q"][k][:13], P[f"{name}.hand"][k], P[f"{name}.w"][k])
        if k % STEP == 0:
            T = g1.fk(q)
            for ln in g1_links:
                Ts[ln].append(T_root @ (T[ln] if ln in T else g1.T_base))
    for ln in g1_links:
        if len(g1_pts[ln]) >= 40:                      # ponytail: tiny links (waist, ankle pitch) add a pose stream, not a shape
            team.add(f"{name}_{ln.rsplit('/', 1)[-1]}", "a", g1_pts[ln], Ts[ln])


# camera: the v3 shots, with the extreme screw close-up opened to a medium shot (dots need some distance)
DX = P["door"][:, 0, 3]
X2 = L["X2"]


def label_t(n, text):
    return next(t for t, l in J["labels"][n] if l.startswith(text))


def ev_t(h):
    return next(e["t"] for e in J["events"] if e["headline"] == h)


SHOTS = [(0.0, lambda k: (13.6, 0.3, 4.8), lambda k: (15.4, 3.4, 0.0)),
         (3.0, lambda k: (10.6, 5.9, 3.0), lambda k: (13.1, 3.1, 1.0)),
         (6.3, lambda k: (10.6, 0.6, 2.2), lambda k: (16.5, 3.0, 0.9)),
         (9.0, lambda k: (DX[k] - 2.6, 6.7, 2.7), lambda k: (DX[k] + 0.4, 3.1, 1.0)),
         (ev_t("DOOR DELIVERED TO WS2") + 0.3, lambda k: (22.6, 6.0, 2.6), lambda k: (20.0, 3.6, 0.9)),
         # both G1s at work on the two faces of the door: high, from past the downstream end
         (label_t("HUMANOID_B", "approach WS2") + 3.0, lambda k: (22.3, 3.1, 3.0), lambda k: (19.9, 3.1, 0.95)),
         (ev_t("STATION 2 COMPLETE") - 0.5, lambda k: (21.8, 5.6, 2.7), lambda k: (19.8, 3.1, 0.9))]


def camera(k):
    t = float(P["t"][k])
    i = max(j for j, sh in enumerate(SHOTS) if sh[0] <= t)
    e, g = np.array(SHOTS[i][1](k), float), np.array(SHOTS[i][2](k), float)
    if i > 0:
        s = minjerk((t - SHOTS[i][0]) / (3.0 if i == 1 else 1.6))
        if s < 1:
            e, g = lerp(np.array(SHOTS[i - 1][1](k), float), e, s), lerp(np.array(SHOTS[i - 1][2](k), float), g, s)
    return e, g


cams_team = [camera(k) for k in FR]

# ------------------------------------------------------------------ clearance check (team side)
def clearance(side, keep=90, near=0.05):
    """Minimum distance between part groups over the whole timeline, on up to `keep` surface dots per part.
    Intended contacts are left out: gripper links on the door, G1 wrists and hands on the door and glass."""
    sub = []
    for name, c, pts, Ts in side.parts:
        idx = rng.choice(len(pts), min(keep, len(pts)), replace=False) if len(pts) else []
        sub.append((name, pts[idx], Ts))
    def grp(pred):
        return [i for i, (n, _, _) in enumerate(sub) if pred(n)]
    g1 = {h: grp(lambda n, h=h: n.startswith(h)) for h in ("HUMANOID_A", "HUMANOID_B")}
    ur = {a: grp(lambda n, a=a: n.startswith(a)) for a in ("ARM_L", "ARM_R")}
    door, glass, lifter = grp(lambda n: n == "door"), grp(lambda n: n == "glass"), grp(lambda n: n == "lifter")
    hand = lambda n: "wrist" in n or "hand" in n  # noqa: E731
    tool = lambda n: "wrist_3" in n  # noqa: E731
    pairs = []
    for h, gi in g1.items():
        for a, ui in ur.items():
            pairs.append((f"{h} x {a}", gi, ui))
        pairs.append((f"{h} x door", [i for i in gi if not hand(sub[i][0])], door))
        pairs.append((f"{h} x glass", [i for i in gi if not hand(sub[i][0])], glass))
        pairs.append((f"{h} x lifter", [i for i in gi if not (h == "HUMANOID_B" and hand(sub[i][0]))], lifter))
    for a, ui in ur.items():
        pairs.append((f"{a} x door", [i for i in ui if not tool(sub[i][0])], door))
        pairs.append((f"{a} x glass", ui, glass))
        pairs.append((f"{a} x lifter", ui, lifter))
    pairs.append(("HUMANOID_A x HUMANOID_B", g1["HUMANOID_A"], g1["HUMANOID_B"]))
    world = lambda i, f: (sub[i][2][f][:3, :3] @ sub[i][1].T).T + sub[i][2][f][:3, 3]  # noqa: E731
    worst = {name: (9.0, 0, "", "") for name, _, _ in pairs}
    bad = []
    for f in range(len(times)):
        W = {}
        for name, A, B in pairs:
            for i in A:
                pa = W.get(i)
                if pa is None:
                    pa = W[i] = world(i, f)
                for j in B:
                    pb = W.get(j)
                    if pb is None:
                        pb = W[j] = world(j, f)
                    if np.linalg.norm(pa.mean(0) - pb.mean(0)) > 2.5:   # far apart: skip the dense test
                        continue
                    d = float(np.sqrt(((pa[:, None] - pb[None]) ** 2).sum(-1).min()))
                    if d < worst[name][0]:
                        worst[name] = (d, times[f], sub[i][0], sub[j][0])
                    if d < near:
                        bad.append((times[f], name, sub[i][0], sub[j][0], d))
    print("[clearance] worst per pair (m, at t, parts):")
    for name, (d, t, a, b) in sorted(worst.items(), key=lambda kv: kv[1][0]):
        print(f"   {d:6.3f}  t={t:5.2f}  {name:26s} {a} / {b}")
    if bad:
        ts = sorted({round(b[0], 1) for b in bad})
        print(f"[clearance] {len(bad)} part pairs under {near * 100:.0f} cm, at t = {ts[:12]}{' ...' if len(ts) > 12 else ''}")
    return min(v[0] for v in worst.values())


if args.check:
    print(f"[clearance] minimum over all checked pairs: {clearance(team):.3f} m")
    sys.exit(0)

# ------------------------------------------------------------------ write
os.makedirs(os.path.join(args.out, "data"), exist_ok=True)
meta = dict(fps=args.fps, frames=len(FR), duration=round(T_END, 3), belt=L["BELT"],
            floor=dict(x=[9.0, 23.0], y=[-0.6, 6.8], belt_x=[0.0, 19.1], belt_y=[1.95, 3.83]),
            rails=[dict(y=y, x=L["RAIL_X"]) for y in RAILS.values()], ws2=[L["X2"], L["DOOR_Y"], L["FIX_Z"]],
            sides=dict(today=dict(label="Today", fov_deg=2 * math.degrees(math.atan(20.955 / 2 / 14.0)),
                                  note="CarDA WS30 cycle 003, real Xsens worker (CC BY 4.0)"),
                       team=dict(label="With a robot team", fov_deg=2 * math.degrees(math.atan(20.955 / 2 / 18.0)),
                                 note="door v3: 2 UR20 on rails + 2 G1, hardcoded; signed events")),
            events=[dict(t=e["t"], name=e["headline"], signed_by=e["signed_by"]) for e in J["events"] if e["headline"]])
n_mocap = sum(1 for t in times if t <= (len(Jw) - 1) / wfps + 1e-6)
meta["sides"]["today"].update(write_side(today, cams_today, os.path.join(args.out, "data", "today.bin"), frames=n_mocap))
meta["sides"]["team"].update(write_side(team, cams_team, os.path.join(args.out, "data", "team.bin")))
json.dump(meta, open(os.path.join(args.out, "data", "manifest.json"), "w"), separators=(",", ":"))
# the site page loads one script instead of fetching: same format as RTN tools/pack_dots.py, written next to the data
import base64  # noqa: E402
pack = {"manifest": meta}
for s in ("today", "team"):
    pack[s] = base64.b64encode(open(os.path.join(args.out, "data", s + ".bin"), "rb").read()).decode()
with open(os.path.join(args.out, "dots_data.js"), "w") as f:
    f.write("/* Generated by export_dots.py (robot repo), same format as RTN tools/pack_dots.py: manifest plus base64 int16 "
            "point and pose data. */\n")
    f.write("window.DOTS=" + json.dumps(pack, separators=(",", ":")) + ";\n")
print(f"[dots] packed {os.path.getsize(os.path.join(args.out, 'dots_data.js')) // 1024} KB -> dots_data.js")
for s in ("today", "team"):
    m = meta["sides"][s]
    print(f"[dots] {s}: {m['points']} pts, {len(m['parts'])} parts, {m['frames']} frames, {m['bytes'] / 1e6:.2f} MB")
