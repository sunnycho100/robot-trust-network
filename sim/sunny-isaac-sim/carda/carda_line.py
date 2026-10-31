"""CarDA-style car-door line in Isaac Sim 6.2: three stations (WS10, WS20, WS30), a cobot on a linear track at each,
a worker fitting the glass at the WS10 door while its cobot drives screws on the same door, and an AMR bringing kits.

Kinematic replay, like the MuJoCo rebuild in sim/handoff: nothing is simulated by PhysX. Every pose is computed per
frame (joint-space moves between IK waypoints, carriage tracking the belt, the worker's arms by two-bone IK) and written
to USD, then the viewport is captured. Layout numbers follow sim/handoff (0.9 m pedestal, 2.51 s rundown, 0.06 m
approach, 24 mm/s belt, ISO/TS 15066 SSM terms); stations, doors and screw points are authored, not CarDA geometry.

  python carda_line.py OUT_DIR [--secs 32] [--stills 3,9,20] [--worker character|mannequin]

Writes OUT_DIR/frame_00000.png ... and OUT_DIR/meta.json (camera and labels per frame) for overlay.py.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kin import Kin, lerp, minjerk, rot, rot_between, tf, two_bone  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--secs", type=float, default=32.0)
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--stills", default="", help="comma-separated times: render only these frames")
ap.add_argument("--worker", default="character", choices=["character", "mannequin"])
ap.add_argument("--dome", type=float, default=600.0)
ap.add_argument("--sun", type=float, default=1500.0)
ap.add_argument("--env", default="/Isaac/Environments/Simple_Warehouse/warehouse.usd")
ap.add_argument("--origin", type=float, nargs=2, default=[0.0, 0.0], help="line centre on the floor (x y)")
args = ap.parse_args()
W, H = 1280, 720

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "width": W, "height": H})

import omni.usd  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402
from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, UsdSkel, Vt  # noqa: E402

ROOT = get_assets_root_path()
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
FPS, N = args.fps, int(round(args.secs * args.fps))
DT = 1.0 / FPS

# ------------------------------------------------------------------ numbers (sim/handoff and the CarDA study)
BELT = 0.024                 # m/s, conveyor speed (handoff analysis)
PED = 0.9                    # m, pedestal height (handoff)
RUNDOWN = 2.51               # s, screw rundown (handoff)
APPROACH = 0.06              # m, approach distance (handoff)
PICK = 0.5                   # s at the screw feeder (sim/handoff assumption)
V_H, T_R, T_S, C_INTR = 1.6, 0.1, 0.326, 0.12   # ISO/TS 15066 SSM terms (handoff section 7)
SLOW = 0.5                   # program speed while inside the protective distance (P1 with SSM slowdown)
PITCH = 5.0                  # m between stations (ours)
QD_MAX = np.radians([120, 120, 180, 180, 180, 180]) * 0.6    # UR10e joint speeds, run at 60%
STATIONS = ["WS10", "WS20", "WS30"]
OX, OY = args.origin
SX = {s: OX + (i - 1) * PITCH for i, s in enumerate(STATIONS)}
Y_TRACK = OY - 0.85
DOOR_X0 = -0.40              # door centre relative to the station at t = 0; moves +x with the belt


def door_x(t):
    return DOOR_X0 + BELT * t


# ------------------------------------------------------------------ small math
def gf2np(m):
    return np.array(m, dtype=float).T          # Gf is row-vector; numpy here is column-vector


def np2gf(T):
    return Gf.Matrix4d(*[float(v) for v in T.T.reshape(-1)])


def quat_R(q):
    w, (x, y, z) = q.GetReal(), q.GetImaginary()
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


# ------------------------------------------------------------------ authored geometry
LOOKS = {}


def material(name, rgb, rough=0.5, metal=0.0):
    if name in LOOKS:
        return LOOKS[name]
    m = UsdShade.Material.Define(stage, f"/World/Looks/{name}")
    sh = UsdShade.Shader.Define(stage, f"/World/Looks/{name}/S")
    sh.CreateIdAttr("UsdPreviewSurface")
    col = sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
    col.Set(Gf.Vec3f(*rgb))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metal)
    m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    LOOKS[name] = (m, col)
    return LOOKS[name]


def bind(prim, name):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(LOOKS[name][0])


def xf(path):
    return UsdGeom.Xform.Define(stage, path)


def setT(prim, T):
    ops = UsdGeom.Xformable(prim).GetOrderedXformOps()
    op = ops[0] if len(ops) == 1 and ops[0].GetOpType() == UsdGeom.XformOp.TypeTransform else UsdGeom.Xformable(prim).MakeMatrixXform()
    op.Set(np2gf(T))


def box(path, c, size, mat, ry=0.0):
    g = UsdGeom.Cube.Define(stage, path)
    g.CreateSizeAttr(1.0)
    x = UsdGeom.Xformable(g)
    x.AddTranslateOp().Set(Gf.Vec3d(*c))
    if ry:
        x.AddRotateYOp().Set(ry)
    x.AddScaleOp().Set(Gf.Vec3f(*size))
    bind(g.GetPrim(), mat)
    return g


def cyl(path, c, r, h, mat, axis="Z"):
    g = UsdGeom.Cylinder.Define(stage, path)
    g.CreateRadiusAttr(r)
    g.CreateHeightAttr(h)
    g.CreateAxisAttr(axis)
    UsdGeom.Xformable(g).AddTranslateOp().Set(Gf.Vec3d(*c))
    bind(g.GetPrim(), mat)
    return g


def ring_mesh(path, r_in, r_out, z, mat, n=72):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = [Gf.Vec3f(float(r * math.cos(t)), float(r * math.sin(t)), z) for r in (r_in, r_out) for t in a]
    faces = [[i, (i + 1) % n, n + (i + 1) % n, n + i] for i in range(n)] if r_in > 0 else None
    if r_in == 0:
        pts = [Gf.Vec3f(0, 0, z)] + pts[n:]
        faces = [[0, 1 + i, 1 + (i + 1) % n] for i in range(n)]
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(pts)
    m.CreateFaceVertexCountsAttr([len(f) for f in faces])
    m.CreateFaceVertexIndicesAttr([i for f in faces for i in f])
    m.CreateDoubleSidedAttr(True)
    bind(m.GetPrim(), mat)
    return m


def quad_mesh(path, pts, mat):
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr([Gf.Vec3f(*p) for p in pts])
    m.CreateFaceVertexCountsAttr([len(pts)])
    m.CreateFaceVertexIndicesAttr(list(range(len(pts))))
    m.CreateDoubleSidedAttr(True)
    bind(m.GetPrim(), mat)
    return m


def reference(path, url):
    p = stage.DefinePrim(path, "Xform")
    p.GetReferences().AddReference(url)
    return p


for name, rgb, rough, metal in [
        ("steel", (0.55, 0.57, 0.6), 0.35, 0.8), ("dark", (0.12, 0.12, 0.13), 0.6, 0.1), ("belt", (0.08, 0.08, 0.09), 0.9, 0.0),
        ("frame", (0.62, 0.64, 0.66), 0.5, 0.3), ("ped", (0.78, 0.79, 0.8), 0.6, 0.0), ("door", (0.08, 0.2, 0.42), 0.25, 0.6),
        ("doortrim", (0.05, 0.05, 0.06), 0.5, 0.2), ("glass", (0.55, 0.7, 0.78), 0.05, 0.0), ("yellow", (0.95, 0.72, 0.05), 0.5, 0.0),
        ("orange", (0.95, 0.45, 0.08), 0.6, 0.0), ("blue", (0.1, 0.35, 0.85), 0.6, 0.0), ("green", (0.12, 0.6, 0.2), 0.6, 0.0),
        ("tray", (0.15, 0.55, 0.22), 0.55, 0.0), ("part", (0.75, 0.75, 0.72), 0.4, 0.5), ("tool", (0.2, 0.2, 0.22), 0.4, 0.4),
        ("bit", (0.85, 0.85, 0.85), 0.2, 1.0), ("toolband", (0.95, 0.45, 0.08), 0.5, 0.0), ("speaker", (0.03, 0.03, 0.03), 0.8, 0.0),
        ("mannequin", (0.95, 0.5, 0.15), 0.6, 0.0)]:
    material(name, rgb, rough, metal)
SCREW_RED, SCREW_GREEN = (0.85, 0.08, 0.06), (0.1, 0.85, 0.2)
ZONE = {"clear": ((0.1, 0.75, 0.25), (0.05, 0.22, 0.09)), "violate": ((0.95, 0.1, 0.08), (0.35, 0.05, 0.04))}

# ------------------------------------------------------------------ environment and light
xf("/World")
reference("/World/Env", ROOT + args.env)
if args.dome > 0:
    UsdLux.DomeLight.Define(stage, "/World/Dome").CreateIntensityAttr(args.dome)
if args.sun > 0:
    sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
    sun.CreateIntensityAttr(args.sun)
    sun.CreateAngleAttr(2.0)
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(40, 0, 30))

# conveyor along the whole line
L0, L1 = SX["WS10"] - 3.2, SX["WS30"] + 3.2
box("/World/Line/belt", ((L0 + L1) / 2, OY, 0.15), (L1 - L0, 0.5, 0.30), "belt")
for sgn in (-1, 1):
    box(f"/World/Line/edge{sgn + 1}", ((L0 + L1) / 2, OY + sgn * 0.27, 0.28), (L1 - L0, 0.04, 0.06), "yellow")
# AMR lane (blue tape) and the station floor marks
for k, y in enumerate((OY - 2.0, OY - 3.0)):
    box(f"/World/Line/lane{k}", ((L0 + L1) / 2, y, 0.003), (L1 - L0 + 2, 0.06, 0.006), "blue")


# ------------------------------------------------------------------ UR10e: generic FK/IK from the asset's own joints
class Chain(Kin):
    """Kin built from a robot asset's UsdPhysics joints; poses are written to the link prims."""

    def __init__(self, root_path):
        self.root = stage.GetPrimAtPath(root_path)
        xc = UsdGeom.XformCache()
        Wi = np.linalg.inv(gf2np(xc.GetLocalToWorldTransform(self.root)))
        rel = lambda p: Wi @ gf2np(xc.GetLocalToWorldTransform(p))  # noqa: E731
        edges = []
        for p in Usd.PrimRange(self.root):
            if not p.IsA(UsdPhysics.Joint):
                continue
            j = UsdPhysics.Joint(p)
            b0, b1 = j.GetBody0Rel().GetTargets(), j.GetBody1Rel().GetTargets()
            if not b0 or not b1:
                continue
            kind = "rev" if p.IsA(UsdPhysics.RevoluteJoint) else "pri" if p.IsA(UsdPhysics.PrismaticJoint) else "fix"
            ax = p.GetAttribute("physics:axis").Get() if kind != "fix" else "X"
            lo, hi = p.GetAttribute("physics:lowerLimit").Get(), p.GetAttribute("physics:upperLimit").Get()
            k = math.pi / 180 if kind == "rev" else 1.0
            edges.append(dict(name=p.GetName(), b0=str(b0[0]), b1=str(b1[0]), kind=kind, axis="XYZ".index(ax),
                              J0=tf(quat_R(j.GetLocalRot0Attr().Get()), j.GetLocalPos0Attr().Get()),
                              J1i=np.linalg.inv(tf(quat_R(j.GetLocalRot1Attr().Get()), j.GetLocalPos1Attr().Get())),
                              lo=(lo if lo is not None else -360) * k, hi=(hi if hi is not None else 360) * k))
        kids = {e["b1"] for e in edges}
        base = next(e["b0"] for e in edges if e["b0"] not in kids)
        super().__init__(edges, base, rel(stage.GetPrimAtPath(base)))
        self.parent, self.static, self.ops = {}, {}, {}
        for l in self.links:
            par = stage.GetPrimAtPath(l).GetParent()
            self.parent[l] = str(par.GetPath())
            if self.parent[l] not in self.links and self.parent[l] != self.base:
                self.static[self.parent[l]] = rel(par)
        T = self.fk(np.zeros(len(self.dofs)))
        err = max(np.abs(T[l] - rel(stage.GetPrimAtPath(l))).max() for l in self.links)
        print(f"[chain] {root_path}: {len(self.dofs)} dofs {[e['name'] for e in self.dofs]}, tool {self.tool}, "
              f"FK(0) vs authored pose max diff {err:.2e}")

    def write(self, q):
        T = self.fk(q)
        for l in self.links:
            if l not in self.ops:
                self.ops[l] = UsdGeom.Xformable(stage.GetPrimAtPath(l)).MakeMatrixXform()
            par = self.parent[l]
            Tp = T[par] if par in T else self.static.get(par, np.eye(4))
            self.ops[l].Set(np2gf(np.linalg.inv(Tp) @ T[l]))
        return T


# ------------------------------------------------------------------ stations
UR = ROOT + "/Isaac/Robots/UniversalRobots/ur10e/ur10e.usd"
U_LOCAL = tf(rot(2, math.pi / 2), (0, 0, PED))   # robot root on the pedestal, facing the door (+y)
DOOR_FACE = -0.05                                 # door front face (y, door frame)
ST = {}
for s in STATIONS:
    P = f"/World/{s}"
    xf(P)
    d = xf(f"{P}/door")                           # door, its carrier and the screws ride the belt
    box(f"{P}/door/carrier", (0, 0, 0.34), (1.4, 0.46, 0.08), "frame")
    for k, x in enumerate((-0.45, 0.45)):
        box(f"{P}/door/post{k}", (x, 0.0, 0.58), (0.06, 0.06, 0.42), "frame")
    box(f"{P}/door/panel", (0, 0, 1.075), (1.2, 0.10, 0.55), "door")
    box(f"{P}/door/beltline", (0.02, 0, 1.36), (1.16, 0.11, 0.03), "doortrim")
    box(f"{P}/door/rearpillar", (0.55, 0, 1.6), (0.06, 0.09, 0.5), "doortrim")
    box(f"{P}/door/roofrail", (0.2, 0, 1.83), (0.76, 0.09, 0.05), "doortrim")
    box(f"{P}/door/apillar", (-0.31, 0, 1.6), (0.05, 0.09, 0.56), "doortrim", ry=32.0)
    glass_pts = [(-0.44, 0, 1.38), (0.52, 0, 1.38), (0.52, 0, 1.81), (-0.18, 0, 1.81)]
    g = xf(f"{P}/door/glass")
    quad_mesh(f"{P}/door/glass/pane", glass_pts, "glass")
    # robot cell: carriage on the track, pedestal, UR10e, screw feeder
    off = {"WS10": -1.0, "WS20": -0.35, "WS30": -0.75}[s]
    box(f"{P}/track", (SX[s] - 0.9, Y_TRACK, 0.04), (3.2, 0.5, 0.08), "dark")
    for k, dy in enumerate((-0.15, 0.15)):
        box(f"{P}/rail{k}", (SX[s] - 0.9, Y_TRACK + dy, 0.1), (3.2, 0.05, 0.05), "steel")
    car = xf(f"{P}/carriage")
    box(f"{P}/carriage/plate", (0, 0, 0.15), (0.6, 0.5, 0.06), "dark")
    box(f"{P}/carriage/pedestal", (0, 0, 0.15 + (PED - 0.15) / 2), (0.32, 0.32, PED - 0.15), "ped")
    box(f"{P}/carriage/feederpost", (-0.5, -0.3, 0.45), (0.06, 0.06, 0.6), "steel")
    box(f"{P}/carriage/feeder", (-0.5, -0.3, 0.8), (0.22, 0.18, 0.12), "yellow")
    ur = reference(f"{P}/carriage/ur", UR)
    UsdGeom.Xformable(ur).MakeMatrixXform().Set(np2gf(U_LOCAL))
    zone = xf(f"{P}/zone")
    material(f"zonefill_{s}", ZONE["clear"][1], 0.8)
    material(f"zonering_{s}", ZONE["clear"][0], 0.6)
    ring_mesh(f"{P}/zone/fill", 0, 1.30, 0.004, f"zonefill_{s}")
    ring_mesh(f"{P}/zone/ring", 1.22, 1.30, 0.006, f"zonering_{s}")
    # line-side kit table (AMR drop)
    box(f"{P}/kit/top", (SX[s] + 1.2, OY - 1.45, 0.74), (0.7, 0.5, 0.04), "frame")
    for k, (dx, dy) in enumerate([(-0.3, -0.2), (0.3, -0.2), (-0.3, 0.2), (0.3, 0.2)]):
        box(f"{P}/kit/leg{k}", (SX[s] + 1.2 + dx, OY - 1.45 + dy, 0.36), (0.04, 0.04, 0.72), "steel")
    ST[s] = dict(P=P, door=d.GetPrim(), glass=g.GetPrim(), car=car.GetPrim(), ur=ur, off=off, zone=zone.GetPrim())

# station specifics: screw points in the door frame (x along the door, z height), on the front face
SCREWS = {
    "WS10": [(-0.50, 0.98), (-0.50, 1.24), (-0.34, 1.10), (-0.20, 0.95)],                           # 4 screws, hinge side
    "WS20": [(mx + dx, mz + dz) for mx, mz in [(-0.32, 1.02), (0.02, 1.2), (0.3, 0.98)]
             for dx, dz in [(-0.04, -0.03), (0.04, -0.03), (0.0, 0.04)]],                           # 3 mounts, 9 screws
    "WS30": [(-0.12 + dx, 1.0 + dz) for dx, dz in [(-0.11, -0.11), (0.11, -0.11), (0.11, 0.11), (-0.11, 0.11)]],  # 4 around the speaker
}
for i, (mx, mz) in enumerate([(-0.32, 1.02), (0.02, 1.2), (0.3, 0.98)]):
    box(f"/World/WS20/door/mount{i}", (mx, DOOR_FACE - 0.008, mz), (0.13, 0.016, 0.11), "part")
cyl("/World/WS30/door/speaker", (-0.12, DOOR_FACE - 0.02, 1.0), 0.085, 0.04, "speaker", axis="Y")
cyl("/World/WS30/door/speakerring", (-0.12, DOOR_FACE - 0.01, 1.0), 0.1, 0.02, "doortrim", axis="Y")
SCREW_MAT = {}
for s in STATIONS:
    for i, (x, z) in enumerate(SCREWS[s]):
        y = DOOR_FACE - (0.02 if s == "WS20" else 0.006) - (0.03 if s == "WS30" else 0.0)
        material(f"screw_{s}_{i}", SCREW_RED, 0.4, 0.3)
        cyl(f"/World/{s}/door/screw{i}", (x, y, z), 0.014, 0.012, f"screw_{s}_{i}", axis="Y")
        SCREW_MAT[(s, i)] = LOOKS[f"screw_{s}_{i}"][1]
        SCREWS[s][i] = (x, y - 0.006, z)          # tip contact point
# WS20 tool rack at the far end of its track
RACK_X = SX["WS20"] - 2.1
box("/World/WS20/rack/top", (RACK_X, Y_TRACK - 0.55, 0.8), (0.5, 0.3, 0.04), "frame")
box("/World/WS20/rack/post", (RACK_X, Y_TRACK - 0.55, 0.4), (0.06, 0.06, 0.8), "steel")
cyl("/World/WS20/rack/gripper", (RACK_X + 0.12, Y_TRACK - 0.55, 0.88), 0.045, 0.12, "toolband")
# WS10 worker zone (orange), as in the study's renders
box("/World/WS10/workerzone", (SX["WS10"] + 0.1, OY - 0.55, 0.005), (1.2, 0.6, 0.006), "orange")

for _ in range(30):
    app.update()

# tools on the flanges
chains = {}
for s in STATIONS:
    c = Chain(str(ST[s]["ur"].GetPath()))
    # ponytail: keep the arm in one family (upper arm above the shoulder, elbow up) so moves never swing through the
    # pedestal; a real cell would use a collision-checked planner (cuRobo, MoveIt), as the study itself notes
    c.lo[1], c.hi[1] = math.radians(-175), math.radians(-5)
    c.lo[2], c.hi[2] = math.radians(10), math.radians(165)
    chains[s] = c
    R = rot_between(np.array([0, 0, 1.0]), c.axis_l)
    T0 = tf(R, c.axis_l * 0.0)
    tp = f"{c.tool}/screwdriver"
    xf(tp)
    UsdGeom.Xformable(stage.GetPrimAtPath(tp)).MakeMatrixXform().Set(np2gf(T0))
    cyl(f"{tp}/body", (0, 0, 0.09), 0.035, 0.16, "tool")
    cyl(f"{tp}/band", (0, 0, 0.03), 0.037, 0.03, "toolband")
    cyl(f"{tp}/bit", (0, 0, 0.21), 0.006, 0.08, "bit")


# ------------------------------------------------------------------ robot programs (joint-space moves between IK waypoints)
def root_from_door(s):
    """Robot root pose in the door frame while the carriage tracks the door."""
    return tf(None, (ST[s]["off"], Y_TRACK - OY, 0)) @ U_LOCAL


def to_root(Tr, p):
    return (np.linalg.inv(Tr) @ np.r_[p, 1])[:3]


SEEDS = [np.radians(v) for v in ([-90, -60, 120, -150, -90, 0], [90, -60, 120, -150, -90, 0], [0, -60, 120, -150, -90, 0],
                                  [-90, -90, 140, 40, 30, 0], [0, -90, 140, 40, 30, 0])]


def solve_home(c, p, a):
    best = None
    for pan in np.radians([-180, -90, 0, 90, 180]):
        for el in (1.6, -1.6):
            q0 = np.array([pan, -1.2, el, -1.5, -1.57, 0.0])
            q, ok = c.ik(p, a, q0)
            if not ok:
                continue
            T = c.fk(q)
            elbow_z = T[c.dofs[2]["b1"]][2, 3]
            score = elbow_z - 0.05 * np.abs(q[3:]).sum()
            if best is None or score > best[0]:
                best = (score, q)
    assert best, "no home IK"
    return best[1]


def build_program(s, t0):
    """[(t, q)] keyframes, screw finish times, and phase labels."""
    c = chains[s]
    Tr = root_from_door(s)
    down, into = np.array([0, 0, -1.0]), np.array([0, 1.0, 0])
    Rr = Tr[:3, :3].T                                                   # door and carriage axes -> root axes
    Ui = np.linalg.inv(U_LOCAL)
    feeder = (Ui @ np.r_[-0.5, -0.3, 0.86, 1])[:3]                     # feeder top, carriage frame -> root frame
    up8 = Rr @ np.array([0, 0, 0.08])
    home_p = to_root(Tr, np.array([ST[s]["off"] + 0.1, Y_TRACK - OY + 0.45, 1.45]))
    q = solve_home(c, home_p, Rr @ down)
    keys, labels, done = [(t0, q)], [], {}
    t = t0

    def move(p, a, min_t=0.35):
        nonlocal q, t
        qn, ok = c.ik(p, a, q, seeds=SEEDS)
        if not ok:
            print(f"[ik] {s} FAILED at {np.round(p, 3)}")
        dur = max(min_t, float(np.max(np.abs(qn - q) / QD_MAX)))
        t += dur
        q = qn
        keys.append((t, q))

    def hold(d):
        nonlocal t
        t += d
        keys.append((t, q))

    if s == "WS20":   # tool change at the rack first (carriage profile below)
        labels.append((t, "tool change"))
        hold(2.6)
        rack = (Ui @ np.r_[0.12, -0.55, 0.95, 1])[:3]                    # carriage parked at RACK_X
        move(rack + up8, Rr @ down)
        move(rack, Rr @ down)
        hold(1.2)
        move(rack + up8, Rr @ down)
        move(home_p, Rr @ down)
        hold(2.6)
        ST[s]["toolchange"] = (t0, t)
    for i, pt in enumerate(SCREWS[s]):
        labels.append((t, f"screw {i + 1}/{len(SCREWS[s])}"))
        move(feeder + up8, Rr @ down)
        move(feeder, Rr @ down)
        hold(PICK)
        move(feeder + up8, Rr @ down)
        pr = to_root(Tr, np.array(pt))
        a = Rr @ into
        move(pr - a * APPROACH, a)
        move(pr, a, 0.24)
        hold(RUNDOWN)
        done[i] = t
        move(pr - a * APPROACH, a, 0.24)
    move(home_p, Rr @ down)
    labels.append((t, "done"))
    return keys, labels, done


def q_at(keys, tau):
    if tau <= keys[0][0]:
        return keys[0][1]
    for (ta, qa), (tb, qb) in zip(keys[:-1], keys[1:]):
        if tau <= tb:
            return lerp(qa, qb, minjerk((tau - ta) / max(tb - ta, 1e-6)))
    return keys[-1][1]


PROG = {}
for s, t0 in [("WS10", 1.5), ("WS20", 0.3), ("WS30", 1.0)]:
    PROG[s] = build_program(s, t0)
    k = PROG[s][0]
    print(f"[prog] {s}: {len(k)} keys, program {k[0][0]:.1f} to {k[-1][0]:.1f} s ({k[-1][0] - k[0][0]:.1f} s)")


# ------------------------------------------------------------------ worker at WS10 (door frame), two-bone arm IK
WORKER_ROOT = np.array([0.12, -0.47])          # door frame: just in front of the door
HAND_Y = DOOR_FACE + 0.02
G_START = np.array([0.0, -0.28, 0.06])         # glass offset while still in the worker's hands


def glass_offset(t):
    return G_START * (1 - minjerk((t - 1.2) / 4.5))


PRESS = [((-0.28, 1.62), (0.42, 1.74)), ((-0.38, 1.44), (0.45, 1.5)), ((-0.18, 1.76), (0.2, 1.78)),
         ((-0.30, 1.50), (0.32, 1.42)), ((-0.24, 1.70), (0.46, 1.62)), ((-0.40, 1.42), (0.1, 1.45))]


def hands(t):
    """Left and right hand targets in the door frame."""
    if t < 6.2:
        g = glass_offset(t)
        return np.array([-0.32, HAND_Y, 1.42]) + g, np.array([0.40, HAND_Y, 1.42]) + g
    u = (t - 6.2) / 2.2
    i, f = int(u), minjerk(min((u % 1) / 0.45, 1.0))
    (la, ra), (lb, rb) = PRESS[i % len(PRESS)], PRESS[(i + 1) % len(PRESS)]
    if i == 0:
        la, ra = (-0.32, 1.42), (0.40, 1.42)
    L = lerp(la, lb, f)
    R = lerp(ra, rb, f)
    return np.array([L[0], HAND_Y, L[1]]), np.array([R[0], HAND_Y, R[1]])


class Mannequin:
    """Capsule worker, as in the study's renders."""

    def __init__(self, path):
        self.path = path
        xf(path)
        self.parts = {}
        for name, r in [("torso", 0.15), ("head", 0.11), ("lua", 0.05), ("lfa", 0.045), ("rua", 0.05), ("rfa", 0.045),
                        ("lleg", 0.08), ("rleg", 0.08), ("neck", 0.05)]:
            g = UsdGeom.Capsule.Define(stage, f"{path}/{name}")
            g.CreateRadiusAttr(r)
            g.CreateHeightAttr(1.0)
            bind(g.GetPrim(), "mannequin")
            self.parts[name] = g
        self.a, self.b, self.sh = 0.30, 0.30, 1.42

    def cap(self, name, p, q):
        g = self.parts[name]
        g.GetHeightAttr().Set(float(max(np.linalg.norm(q - p), 1e-3)))
        setT(g.GetPrim(), tf(rot_between(np.array([0, 0, 1.0]), q - p), (p + q) / 2))

    def pose(self, Tw, lt, rt):
        """Tw: worker root in world (facing +y). lt, rt: hand targets in world. Returns body points for SSM."""
        P = lambda v: (Tw @ np.r_[v, 1])[:3]  # noqa: E731
        pel, chest, head = P([0, 0, 0.95]), P([0, 0.03, 1.3]), P([0, 0.05, 1.62])
        out = Tw[:3, :3] @ np.array([1.0, 0, 0])
        pts = [pel, chest, head]
        for side, t, ua, fa in [(-1, lt, "lua", "lfa"), (1, rt, "rua", "rfa")]:
            S = P([side * 0.2, 0.02, self.sh])
            E, Wr = two_bone(S, self.a, self.b, t, np.array([0, 0, -1.0]) + side * 0.4 * out)
            self.cap(ua, S, E)
            self.cap(fa, E, Wr)
            pts += [S, E, Wr]
        self.cap("torso", pel, P([0, 0.03, 1.38]))
        self.cap("neck", P([0, 0.03, 1.45]), P([0, 0.04, 1.52]))
        self.cap("head", P([0, 0.05, 1.58]), P([0, 0.05, 1.68]))
        self.cap("lleg", P([-0.11, 0, 0.08]), P([-0.1, 0, 0.88]))
        self.cap("rleg", P([0.11, 0, 0.08]), P([0.1, 0, 0.88]))
        return np.array(pts)


class Character:
    """Stock Isaac People character; the arms are posed by our own SkelAnimation (two-bone IK), the rest stays at rest."""

    CHAR = "/Isaac/People/Characters/male_adult_construction_01_new/male_adult_construction_01_new.usd"

    def __init__(self, path):
        self.root = xf(path).GetPrim()
        self.fix = xf(f"{path}/fix").GetPrim()
        reference(f"{path}/fix/char", ROOT + self.CHAR)
        for _ in range(40):
            app.update()
        sk = next(p for p in Usd.PrimRange(self.fix) if p.IsA(UsdSkel.Skeleton))
        self.skel = UsdSkel.Skeleton(sk)
        joints = [str(j) for j in self.skel.GetJointsAttr().Get()]
        self.joints = joints
        rest = [gf2np(m) for m in self.skel.GetRestTransformsAttr().Get()]
        idx = {j: i for i, j in enumerate(joints)}
        par = [idx.get(j.rsplit("/", 1)[0], -1) if "/" in j else -1 for j in joints]
        Wr = []
        for i, L in enumerate(rest):
            Wr.append(L if par[i] < 0 else Wr[par[i]] @ L)
        self.rest, self.Wrest, self.par = rest, Wr, par

        def find(side, names):
            for nm in names:
                for i, j in enumerate(joints):
                    if j.rsplit("/", 1)[-1].lower().endswith(f"{side}_{nm}"):
                        return i
            raise KeyError(f"{side} {names} not in {[j.rsplit('/', 1)[-1] for j in joints if 'arm' in j.lower() or 'hand' in j.lower()][:20]}")
        self.arm = {sd: [find(sd, ["upperarm"]), find(sd, ["forearm"]), find(sd, ["hand"])] for sd in ("l", "r")}
        xc = UsdGeom.XformCache()
        self.W_skel0 = np.linalg.inv(gf2np(xc.GetLocalToWorldTransform(self.root))) @ gf2np(xc.GetLocalToWorldTransform(sk))
        pos = lambda i: (self.W_skel0 @ self.Wrest[i])[:3, 3]  # noqa: E731
        right = pos(self.arm["r"][0]) - pos(self.arm["l"][0])
        right[2] = 0
        fwd = np.cross([0, 0, 1.0], right / np.linalg.norm(right))
        yaw = math.atan2(fwd[1], fwd[0])
        foot = min(pos(i)[2] for i, j in enumerate(joints) if j.endswith("_Foot"))
        Tfix = tf(rot(2, math.pi / 2 - yaw), (0, 0, 0.085 - foot))   # face +y, ankles at 8.5 cm
        UsdGeom.Xformable(self.fix).MakeMatrixXform().Set(np2gf(Tfix))
        self.W_skel = Tfix @ self.W_skel0
        print(f"[worker] {len(joints)} joints, arm {[joints[i].rsplit('/', 1)[-1] for i in self.arm['l'] + self.arm['r']]}, "
              f"facing fix {math.degrees(math.pi / 2 - yaw):.0f} deg, foot z {foot:.3f}")
        anim = UsdSkel.Animation.Define(stage, f"{path}/armAnim")
        self.ids = self.arm["l"][:2] + self.arm["r"][:2]
        anim.CreateJointsAttr([joints[i] for i in self.ids])
        anim.CreateTranslationsAttr(Vt.Vec3fArray([Gf.Vec3f(*self.rest[i][:3, 3]) for i in self.ids]))
        anim.CreateScalesAttr(Vt.Vec3hArray([Gf.Vec3h(1, 1, 1)] * len(self.ids)))
        self.rot_attr = anim.CreateRotationsAttr()
        UsdSkel.BindingAPI.Apply(sk).CreateAnimationSourceRel().SetTargets([anim.GetPath()])

    def pose(self, Tw, lt, rt):
        setT(self.root, Tw)
        Ws = Tw @ self.W_skel                     # skel space -> world
        Wsi = np.linalg.inv(Ws)
        toS = lambda p: (Wsi @ np.r_[p, 1])[:3]  # noqa: E731
        toW = lambda p: (Ws @ np.r_[p, 1])[:3]  # noqa: E731
        Rws = Ws[:3, :3] / np.linalg.norm(Ws[:3, 0])
        out_w = Tw[:3, :3] @ np.array([1.0, 0, 0])
        quats, pts = [], [toW(self.Wrest[i][:3, 3]) for i, j in enumerate(self.joints)
                          if j.rsplit("/", 1)[-1] in ("Pelvis", "Head", "Spine02", "Waist", "Neck")]
        for sd, t, sg in (("l", lt, -1), ("r", rt, 1)):
            u, f, h = self.arm[sd]
            S, E0, H0 = (self.Wrest[i][:3, 3] for i in (u, f, h))
            a, b = np.linalg.norm(E0 - S), np.linalg.norm(H0 - E0)
            pole = Rws.T @ (np.array([0, 0, -1.0]) + sg * 0.4 * out_w)
            E, Wr = two_bone(S, a, b, toS(t), pole)
            RU0, RF0 = self.Wrest[u][:3, :3], self.Wrest[f][:3, :3]
            RU = rot_between(E0 - S, E - S) @ RU0
            dcur = RU @ RU0.T @ (H0 - E0)
            RF = rot_between(dcur, Wr - E) @ RU @ RU0.T @ RF0
            Rp = self.Wrest[self.par[u]][:3, :3]
            for Rl in (Rp.T @ RU, RU.T @ RF):
                qd = np2gf(tf(Rl)).ExtractRotationQuat()
                quats.append(Gf.Quatf(qd.GetReal(), Gf.Vec3f(*qd.GetImaginary())))
            pts += [toW(S), toW(E), toW(Wr)]
        self.rot_attr.Set(Vt.QuatfArray(quats))
        return np.array(pts)


worker_kind = args.worker
try:
    WORKER = Character("/World/WS10/worker") if worker_kind == "character" else Mannequin("/World/WS10/worker")
except Exception as e:  # noqa: BLE001 - fall back to the capsule worker
    print(f"[worker] character failed ({e!r}), using the mannequin")
    stage.RemovePrim("/World/WS10/worker")
    WORKER, worker_kind = Mannequin("/World/WS10/worker"), "mannequin"

# ------------------------------------------------------------------ AMR with the kits
AMR_URL = ROOT + "/Isaac/Robots/Idealworks/iwhub/iw_hub_static.usd"
amr = reference("/World/AMR", AMR_URL)
for _ in range(30):
    app.update()
bb = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeWorldBound(amr).ComputeAlignedRange()
AMR_YAW = 0.0 if bb.GetSize()[0] >= bb.GetSize()[1] else math.pi / 2   # drive along its long axis
AMR_C = np.array(bb.GetMidpoint())                                     # the asset's origin is off centre
AMR_LIFT = -float(bb.GetMin()[2])                                      # and below its wheels
CART = AMR_LIFT + float(bb.GetMax()[2]) + 0.55                        # kit cart deck the AMR carries
print(f"[amr] bbox {bb}, yaw {math.degrees(AMR_YAW):.0f}, cart deck {CART:.2f}")
xf("/World/AMR/cart")
setT(stage.GetPrimAtPath("/World/AMR/cart"), tf(None, (AMR_C[0], AMR_C[1], 0)))
box("/World/AMR/cart/deck", (0, 0, CART - AMR_LIFT - 0.02), (1.35, 0.62, 0.04), "frame")
box("/World/AMR/cart/base", (0, 0, float(bb.GetMax()[2]) + 0.03), (1.35, 0.62, 0.04), "frame")
for k, (dx, dy) in enumerate([(-0.64, -0.28), (0.64, -0.28), (-0.64, 0.28), (0.64, 0.28)]):
    box(f"/World/AMR/cart/post{k}", (dx, dy, (float(bb.GetMax()[2]) + CART - AMR_LIFT) / 2), (0.04, 0.04, CART - AMR_LIFT - float(bb.GetMax()[2])), "steel")
AMR_Y = OY - 2.5
TRAYS = []
for i in range(3):
    p = f"/World/kit{i}"
    xf(p)
    box(f"{p}/tray", (0, 0, 0.03), (0.42, 0.3, 0.06), "tray")
    for k, (dx, dy) in enumerate([(-0.1, -0.06), (0.08, 0.05), (0.1, -0.07)]):
        box(f"{p}/part{k}", (dx, dy, 0.08), (0.08, 0.06, 0.05), "part")
    TRAYS.append(stage.GetPrimAtPath(p))
STOPS = [SX[s] + 1.2 for s in STATIONS]
AMR_PLAN, t_, x_ = [], 0.0, SX["WS10"] - 4.5
for xs in STOPS + [SX["WS30"] + 6.0]:
    dur = max(2.0, abs(xs - x_) / 0.8)
    AMR_PLAN.append(("move", t_, t_ + dur, x_, xs))
    t_ += dur
    x_ = xs
    if len(AMR_PLAN) <= 2 * len(STOPS) - 1:
        AMR_PLAN.append(("stop", t_, t_ + 3.0, xs, xs))
        t_ += 3.0


def amr_state(t):
    """AMR x and, per tray, its world position."""
    x, stop_i, stops_done = AMR_PLAN[0][3], -1, []
    k = 0
    for kind, ta, tb, xa, xb in AMR_PLAN:
        if kind == "stop":
            if t >= ta:
                stops_done.append((k, ta))
            k += 1
        if ta <= t <= tb or (t > tb and kind == "move"):
            x = xa + (xb - xa) * (minjerk((t - ta) / (tb - ta)) if kind == "move" else 1.0)
    trays = []
    for i in range(3):
        on_amr = np.array([x + AMR_C[0] - 0.45 + 0.45 * i, AMR_Y + AMR_C[1], CART])
        hit = [ta for k_, ta in stops_done if k_ == i]
        if not hit:
            trays.append(on_amr)
            continue
        s = minjerk((t - hit[0] - 0.6) / 1.4)
        table = np.array([STOPS[i], OY - 1.45, 0.76])
        if s <= 0:
            trays.append(on_amr)
            continue
        if s >= 1:
            trays.append(table)
            continue
        # tray leaves the AMR: once it is gone, it rides from the AMR position at hand-off time
        start = np.array([STOPS[i] + AMR_C[0] - 0.45 + 0.45 * i, AMR_Y + AMR_C[1], CART])
        p = lerp(start, table, s)
        p[2] += 0.12 * math.sin(math.pi * s)
        trays.append(p)
    return x, trays


# ------------------------------------------------------------------ camera
cam = UsdGeom.Camera.Define(stage, "/World/Cam")
FOCAL, HAP = 18.0, 20.955
cam.CreateFocalLengthAttr(FOCAL)
cam.CreateHorizontalApertureAttr(HAP)
cam.CreateVerticalApertureAttr(HAP * H / W)
cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 500))
cam_op = UsdGeom.Xformable(cam).MakeMatrixXform()
vp = get_active_viewport()
vp.camera_path = "/World/Cam"


def camera(t):
    """Overview of the line, then a close look at the shared WS10 door."""
    dx = door_x(t)
    w10 = np.array([SX["WS10"] + dx - 0.35, OY - 0.3, 1.05])
    A0 = (np.array([OX - 9.5, OY - 10.5, 5.6]), np.array([OX - 2.5, OY - 1.0, 0.5]))
    A1 = (np.array([OX - 4.5, OY - 11.0, 5.2]), np.array([OX + 0.5, OY - 1.0, 0.5]))

    def orbit(deg):
        a = math.radians(deg)
        return w10 + np.array([3.5 * math.sin(a), -3.5 * math.cos(a), 1.0])
    C = (orbit(-40 + 25 * min(max((t - 15) / 17, 0), 1)), w10)
    if t < 11:
        s = t / 11
        return lerp(A0[0], A1[0], s), lerp(A0[1], A1[1], s)
    s = minjerk((t - 11) / 4.0)
    C15 = (orbit(-40), w10)
    if t < 15:
        return lerp(A1[0], C15[0], s), lerp(A1[1], C15[1], s)
    return C


def set_camera(eye, target):
    m = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1)).GetInverse()
    cam_op.Set(m)


# ------------------------------------------------------------------ precompute every frame
def carriage_x(s, t):
    track = SX[s] + door_x(t) + ST[s]["off"]
    if s != "WS20":
        return track
    t0, t1 = ST[s]["toolchange"]
    w = minjerk((t - t0) / 2.5) * (1 - minjerk((t - (t1 - 2.5)) / 2.5))
    return (1 - w) * track + w * RACK_X


def S_p(v_r):
    return V_H * (T_R + T_S) + v_r * T_R + C_INTR


frames, tau, prev_tip, kpi = [], PROG["WS10"][0][0][0] * 0, None, dict(red=0, active=0, min_sep=9.0, slow_s=0.0)
for k in range(N):
    t = k * DT
    f = dict(t=t, door=door_x(t))
    for s in STATIONS:
        keys = PROG[s][0]
        if s == "WS10":
            f["tau"] = tau
            q = q_at(keys, tau)
        else:
            q = q_at(keys, t)
        f[f"q_{s}"] = q
        f[f"cx_{s}"] = carriage_x(s, t)
    # WS10 separation: robot points vs worker points (world)
    c = chains["WS10"]
    Troot = tf(None, (f["cx_WS10"], Y_TRACK, 0)) @ U_LOCAL
    pts = c.points(c.fk(f["q_WS10"]))
    rp = (Troot @ np.c_[pts, np.ones(len(pts))].T).T[:, :3]
    D = tf(None, (SX["WS10"] + f["door"], OY, 0))
    Tw = D @ tf(None, (WORKER_ROOT[0], WORKER_ROOT[1], 0))
    lt, rt = [(D @ np.r_[h, 1])[:3] for h in hands(t)]
    f["lt"], f["rt"], f["Tw"] = lt, rt, Tw
    wp = np.array([Tw[:3, 3] + [0, 0, z] for z in (0.95, 1.3, 1.62)] + [lt, rt])
    sep = float(np.min(np.linalg.norm(rp[:, None] - wp[None], axis=2))) - 0.08 - 0.1
    tip = rp[-1]
    v_r = 0.0 if prev_tip is None else float(np.linalg.norm(tip - prev_tip) / DT)
    prev_tip = tip
    sp = S_p(min(v_r, 1.0))
    active = keys[0][0] <= tau <= PROG["WS10"][0][-1][0]
    red = bool(sep < sp)
    f.update(sep=sep, sp=sp, red=red, active=active)
    if active:
        kpi["active"] += 1
        kpi["red"] += red
        kpi["min_sep"] = min(kpi["min_sep"], sep)
        kpi["slow_s"] += DT * (1 - SLOW) if red else 0.0
    tau += DT * (SLOW if red and active else 1.0)
    f["amr"] = amr_state(t)
    f["cam"] = camera(t)
    frames.append(f)
kpi["red_share"] = kpi["red"] / max(kpi["active"], 1)
print(f"[ssm] WS10 while the robot program runs: inside S_p {kpi['red_share']:.0%} of the time, "
      f"min separation {kpi['min_sep']:.2f} m, slowdown {kpi['slow_s']:.1f} s")


# ------------------------------------------------------------------ apply one frame
def label_for(s, tt):
    lab = "idle"
    for ta, txt in PROG[s][1]:
        if tt >= ta:
            lab = txt
    return lab


def apply(f):
    t = f["t"]
    with Sdf.ChangeBlock():
        for s in STATIONS:
            setT(ST[s]["door"], tf(None, (SX[s] + f["door"], OY, 0)))
            setT(ST[s]["car"], tf(None, (f[f"cx_{s}"], Y_TRACK, 0)))
            setT(ST[s]["zone"], tf(None, (f[f"cx_{s}"], Y_TRACK, 0)))
            tt = f["tau"] if s == "WS10" else t
            for i, td in PROG[s][2].items():
                SCREW_MAT[(s, i)].Set(Gf.Vec3f(*(SCREW_GREEN if tt >= td else SCREW_RED)))
        setT(ST["WS10"]["glass"], tf(None, glass_offset(t)))
        x, trays = f["amr"]
        setT(amr, tf(rot(2, AMR_YAW), (x, AMR_Y, AMR_LIFT)))
        for p, pos in zip(TRAYS, trays):
            setT(p, tf(None, pos))
        st = "violate" if f["red"] else "clear"
        LOOKS["zonering_WS10"][1].Set(Gf.Vec3f(*ZONE[st][0]))
        LOOKS["zonefill_WS10"][1].Set(Gf.Vec3f(*ZONE[st][1]))
        set_camera(*f["cam"])
    for s in STATIONS:
        chains[s].write(f[f"q_{s}"])
    f["pts"] = WORKER.pose(f["Tw"], f["lt"], f["rt"])


# ------------------------------------------------------------------ render
os.makedirs(args.out, exist_ok=True)
for i in range(400):   # let the references and textures load
    app.update()
    msg, loaded, total = omni.usd.get_context().get_stage_loading_status()
    if i > 60 and total == 0:
        break
print("[render] stage loaded after", i, "updates")
for _ in range(30):
    apply(frames[0])
    app.update()

meta = dict(W=W, H=H, focal=FOCAL, hap=HAP, fps=FPS, worker=worker_kind, kpi=kpi,
            stations={s: [SX[s], OY] for s in STATIONS}, y_track=Y_TRACK, amr_y=AMR_Y, frames=[])
which = [int(round(float(x) * FPS)) for x in args.stills.split(",")] if args.stills else range(N)
for k in which:
    f = frames[min(k, N - 1)]
    apply(f)
    for _ in range(8 if args.stills else 1):
        app.update()
    path = os.path.join(args.out, f"frame_{k:05d}.png")
    capture_viewport_to_file(vp, path)
    for _ in range(4 if args.stills else 1):
        app.update()
    eye, target = f["cam"]
    meta["frames"].append(dict(k=k, t=round(f["t"], 3), eye=list(map(float, eye)), target=list(map(float, target)),
                               door=f["door"], amr_x=float(f["amr"][0]), cx10=float(f["cx_WS10"]),
                               worker=[float(v) for v in f["Tw"][:2, 3]], sep=round(f["sep"], 3), sp=round(f["sp"], 3),
                               red=f["red"], active=f["active"],
                               labels={s: label_for(s, f["tau"] if s == "WS10" else f["t"]) for s in STATIONS}))
    if k % 90 == 0:
        print(f"[render] frame {k}/{N}", flush=True)
for _ in range(20):
    app.update()
json.dump(meta, open(os.path.join(args.out, "meta.json"), "w"))
print("[render] done", len(meta["frames"]), "frames")
app.close()
