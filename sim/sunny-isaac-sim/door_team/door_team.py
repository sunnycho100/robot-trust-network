"""Render a door-team plan (orchestrate.py) in Isaac Sim 6.2, headless.

Two stock UR20 on carriages riding parallel rails carry the door from WS1 to WS2 (assisted grasp: the door follows
both grippers as if fixed to them), two stock G1 replay the MuJoCo walking-policy roots and legs with arms by IK, and
the door, glass, screw and screwdriver are authored geometry. Nothing is simulated by PhysX here: every pose comes
from the plan and is written to USD per frame, then the viewport is captured.

  python door_team.py PLAN_DIR OUT_DIR [--stills 2,10,20] [--secs 40]

Writes OUT_DIR/frame_00000.png ... and OUT_DIR/meta.json (camera and label anchors per frame) for overlay.py.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("plan")
ap.add_argument("out")
ap.add_argument("--stills", default="")
ap.add_argument("--secs", type=float, default=0.0, help="render only up to SECS seconds (0: all)")
ap.add_argument("--start", type=float, default=0.0, help="render from START seconds (to split a render across processes)")
ap.add_argument("--hq", action="store_true", help="1080p path tracing with the denoiser, textured look (look_hq.py), JPEG frames")
ap.add_argument("--spp", type=int, default=16, help="--hq: path-traced samples per pixel per subframe")
ap.add_argument("--subframes", type=int, default=2, help="--hq: subframes accumulated per time sample")
ap.add_argument("--dof", type=float, default=0.0, help="--hq: f-stop, focused on the shot target (0: off)")
ap.add_argument("--mblur", type=int, default=1, help="--hq: time samples over a 180 degree shutter (1: off)")
ap.add_argument("--dome", type=float, default=0.0, help="extra light; the station has its own")
ap.add_argument("--sun", type=float, default=0.0)
ap.add_argument("--station", default="/robot/ws30_station/out/ws30_station.usd")
args = ap.parse_args()
W, H = (1920, 1080) if args.hq else (1280, 720)

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "width": W, "height": H})

import omni.usd  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402
from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade  # noqa: E402

import rig  # noqa: E402
from kin import lerp, minjerk, rot, rot_between, tf  # noqa: E402

ROOT = get_assets_root_path()
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
P = np.load(os.path.join(args.plan, "plan.npz"))
J = json.load(open(os.path.join(args.plan, "plan.json")))
L = J["layout"]
FPS, N = J["fps"], len(P["t"])
X1, X2, FIX_Z, PED = L["X1"], L["X2"], L["FIX_Z"], L["PED"]

# ------------------------------------------------------------------ authored geometry (as carda_line.py)
LOOKS = {}


def material(name, rgb, rough=0.5, metal=0.0, opacity=1.0):
    m = UsdShade.Material.Define(stage, f"/World/Looks/{name}")
    sh = UsdShade.Shader.Define(stage, f"/World/Looks/{name}/S")
    sh.CreateIdAttr("UsdPreviewSurface")
    col = sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
    col.Set(Gf.Vec3f(*rgb))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metal)
    if opacity < 1:
        sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
    m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    LOOKS[name] = (m, col)


def bind(prim, name):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(LOOKS[name][0])


def xf(path):
    return UsdGeom.Xform.Define(stage, path).GetPrim()


def setT(prim, T):
    ops = UsdGeom.Xformable(prim).GetOrderedXformOps()
    op = ops[0] if len(ops) == 1 and ops[0].GetOpType() == UsdGeom.XformOp.TypeTransform else UsdGeom.Xformable(prim).MakeMatrixXform()
    op.Set(rig.np2gf(T))


def box(path, c, size, mat, rz=0.0, ry=0.0):
    g = UsdGeom.Cube.Define(stage, path)
    g.CreateSizeAttr(1.0)
    x = UsdGeom.Xformable(g)
    x.AddTranslateOp().Set(Gf.Vec3d(*c))
    if rz:
        x.AddRotateZOp().Set(rz)
    if ry:
        x.AddRotateYOp().Set(ry)
    x.AddScaleOp().Set(Gf.Vec3f(*size))
    bind(g.GetPrim(), mat)
    return g.GetPrim()


def cyl(path, c, r, h, mat, axis="Z"):
    g = UsdGeom.Cylinder.Define(stage, path)
    g.CreateRadiusAttr(r)
    g.CreateHeightAttr(h)
    g.CreateAxisAttr(axis)
    UsdGeom.Xformable(g).AddTranslateOp().Set(Gf.Vec3d(*c))
    bind(g.GetPrim(), mat)
    return g.GetPrim()


def quad(path, pts, mat):
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr([Gf.Vec3f(*p) for p in pts])
    m.CreateFaceVertexCountsAttr([len(pts)])
    m.CreateFaceVertexIndicesAttr(list(range(len(pts))))
    m.CreateDoubleSidedAttr(True)
    bind(m.GetPrim(), mat)
    return m.GetPrim()


def reference(path, url):
    p = stage.DefinePrim(path, "Xform")
    p.GetReferences().AddReference(url)
    return p


for name, rgb, rough, metal in [
        ("steel", (0.55, 0.57, 0.6), 0.35, 0.8), ("dark", (0.12, 0.12, 0.13), 0.6, 0.1), ("frame", (0.62, 0.64, 0.66), 0.5, 0.3),
        ("ped", (0.78, 0.79, 0.8), 0.6, 0.0), ("door", (0.08, 0.2, 0.42), 0.25, 0.6), ("doortrim", (0.05, 0.05, 0.06), 0.5, 0.2),
        ("yellow", (0.95, 0.72, 0.05), 0.5, 0.0), ("orange", (0.95, 0.45, 0.08), 0.6, 0.0), ("blue", (0.1, 0.35, 0.85), 0.6, 0.0),
        ("tool", (0.2, 0.2, 0.22), 0.4, 0.4), ("bit", (0.85, 0.85, 0.85), 0.2, 1.0), ("wood", (0.55, 0.4, 0.25), 0.7, 0.0),
        ("finger", (0.25, 0.25, 0.28), 0.4, 0.5), ("bin", (0.12, 0.45, 0.2), 0.6, 0.0)]:
    material(name, rgb, rough, metal)
material("glass", (0.55, 0.75, 0.85), 0.05, 0.0, opacity=0.35)
material("screwhead", (0.85, 0.08, 0.06), 0.4, 0.3)
SCREW_RED, SCREW_GREEN = (0.85, 0.08, 0.06), (0.1, 0.85, 0.2)

xf("/World")
# the CarDA WS30 station rebuilt from the real video (ws30_station.py): hall, belt, platform, trolley with its door,
# line-side rack, desks, gantry, cameras and lights. Our cell is added around it without covering it.
ENV = reference("/World/Station", args.station)
ST = "/World/Station/WS30"
if args.dome > 0:
    UsdLux.DomeLight.Define(stage, "/World/Dome").CreateIntensityAttr(args.dome)
if args.sun > 0:
    sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
    sun.CreateIntensityAttr(args.sun)
    sun.CreateAngleAttr(2.0)
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(40, 0, 30))
DOOR_Y, RAILS, RAIL_X = L["DOOR_Y"], L["RAILS"], L["RAIL_X"]
CAR_X0, CAR_Y = L["CARRIER"]
DDX, DDY, DZ0 = L["DOOR_ON_CARRIER"]
T_DOOR_ON_CAR = tf(rot(2, math.pi), (DDX, DDY, DZ0))  # our door frame inside the trolley frame (the station's door)

# WS2 stand past the belt end, rails on the walkways (layout anchors)
RAIL_Z = 0.0
box("/World/WS2/base", (X2, DOOR_Y, (FIX_Z - 0.04) / 2), (1.0, 0.30, FIX_Z - 0.04), "frame")
box("/World/WS2/cradle", (X2, DOOR_Y, FIX_Z - 0.02), (1.1, 0.18, 0.04), "dark")
for k, dx in enumerate((-0.45, 0.45)):
    box(f"/World/WS2/locator{k}", (X2 + dx, DOOR_Y, FIX_Z + 0.02), (0.05, 0.18, 0.04), "yellow")
for k, (dx, dy, sx, sy) in enumerate([(0, -0.75, 1.9, 0.06), (0, 0.75, 1.9, 0.06), (-0.95, 0, 0.06, 1.5), (0.95, 0, 0.06, 1.5)]):
    box(f"/World/WS2/tape{k}", (X2 + dx, DOOR_Y + dy, 0.003), (sx, sy, 0.006), "yellow")
for name, y in RAILS.items():
    if name not in J["arms"]:
        continue
    cx, ln = (RAIL_X[0] + RAIL_X[1]) / 2, RAIL_X[1] - RAIL_X[0]
    box(f"/World/Rail_{name}/bed", (cx, y, RAIL_Z + 0.04), (ln, 0.5, 0.08), "dark")
    for k, dy in enumerate((-0.15, 0.15)):
        box(f"/World/Rail_{name}/r{k}", (cx, y + dy, RAIL_Z + 0.1), (ln, 0.05, 0.05), "steel")

# the station's own door rides its trolley until the arms take it; a ring marks our screw spot on its inner face
DOOR_MARKS = xf("/World/DoorMarks")
sx_, sy_, sz_ = L["SCREW_IN_DOOR"]
sgn_ = 1.0 if sy_ > 0 else -1.0                   # marks sit on the outside of whichever face the screw is on
cyl("/World/DoorMarks/hole", (sx_, sy_ + sgn_ * 0.001, sz_), 0.03, 0.004, "yellow", axis="Y")
cyl("/World/DoorMarks/holein", (sx_, sy_ + sgn_ * 0.003, sz_), 0.012, 0.004, "dark", axis="Y")
GLASS = xf("/World/Glass")
quad("/World/Glass/pane", [tuple(p) for p in L["GLASS_PTS"]], "glass")
# handheld suction-cup glass lifter: frame at the handle centre, x along the bar, +y towards the glass
LIFTER = xf("/World/Lifter")
cyl("/World/Lifter/handle", (0, 0, 0), 0.016, 0.26, "tool", axis="X")
box("/World/Lifter/bar", (0, 0.065, 0), (0.46, 0.03, 0.03), "steel")
for k, x in enumerate((-0.12, 0.12)):
    box(f"/World/Lifter/post{k}", (x, 0.03, 0), (0.02, 0.06, 0.02), "steel")
for k, x in enumerate((-0.20, 0.20)):
    cyl(f"/World/Lifter/cup{k}", (x, 0.09, 0), 0.06, 0.012, "orange", axis="Y")
SCREW = xf("/World/Screw")                       # head at the origin, shank along +x (oversized for the camera)
cyl("/World/Screw/head", (-0.003, 0, 0), 0.016, 0.008, "screwhead", axis="X")
cyl("/World/Screw/shank", (0.016, 0, 0), 0.006, 0.03, "steel", axis="X")
# pistol-grip electric driver: origin in the grip (the hand), bit along +x, tip at TIPV
TIPV = np.array([0.20, 0.0, 0.075])
DRIVER = xf("/World/Driver")
box("/World/Driver/grip", (0.0, 0, 0.015), (0.045, 0.04, 0.12), "tool", ry=-12)
box("/World/Driver/battery", (-0.005, 0, -0.06), (0.09, 0.07, 0.05), "tool")
cyl("/World/Driver/body", (0.03, 0, 0.075), 0.032, 0.17, "orange", axis="X")
cyl("/World/Driver/chuck", (0.13, 0, 0.075), 0.016, 0.04, "tool", axis="X")
BIT = xf("/World/Driver/bit")
cyl("/World/Driver/bit/shaft", (0.175, 0, 0.075), 0.004, 0.05, "bit", axis="X")
box("/World/Driver/bit/flat", (0.196, 0, 0.075), (0.008, 0.012, 0.003), "bit")

# a tool board in the station's line-side rack (driver and screw bin), and our glass rack past the belt end
RX, RY = L["RACK"]
box("/World/RackBoard/board", (RX, RY - 0.05, 0.895), (0.56, 0.36, 0.02), "wood")
SB = L.get("SCREW_BIN", [RX + 0.15, 5.12, 0.94])
box("/World/RackBoard/bin", (SB[0], SB[1], 0.915), (0.12, 0.12, 0.03), "bin")
GX, GY = L["GR"]
box("/World/GlassRack/base", (GX, GY, 0.38), (1.25, 0.34, 0.76), "frame")
box("/World/GlassRack/slot", (GX, GY, 0.785), (1.2, 0.08, 0.03), "dark")
for k, dx in enumerate((-0.58, 0.58)):
    box(f"/World/GlassRack/post{k}", (GX + dx, GY, 1.05), (0.05, 0.08, 0.6), "steel")
for k, dy in enumerate((0.12, 0.24)):   # spare panes behind ours
    quad(f"/World/GlassRack/spare{k}", [(GX + p[0], GY + dy, 0.80 + p[2]) for p in L["GLASS_PTS"]], "glass")

# rail arms: carriage, pedestal, UR20, authored two-finger gripper
UR = ROOT + "/Isaac/Robots/UniversalRobots/ur20/ur20.usd"
T_MOUNT = tf(None, (0, 0, PED))
ARMS = {}
import types  # noqa: E402

import plant  # noqa: E402

def world_of(prim):
    return rig.gf2np(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0))


plant.build(types.SimpleNamespace(box=box, cyl=cyl, xf=xf, setT=setT, reference=reference, material=material, app=app,
                                  rig=rig, ROOT=ROOT, L=L, stage=stage, world_of=world_of))
for name in J["arms"]:
    side = -1 if name == "ARM_L" else 1
    c = f"/World/{name}"
    car = xf(c)
    box(f"{c}/plate", (0, 0, 0.15), (0.6, 0.6, 0.06), "dark")
    box(f"{c}/pedestal", (0, 0, 0.18 + (PED - 0.18) / 2), (0.34, 0.34, PED - 0.18), "ped")
    box(f"{c}/stripe", (0, 0, PED - 0.06), (0.35, 0.35, 0.04), "yellow" if name == "ARM_L" else "blue")
    ur = reference(f"{c}/ur", UR)
    setT(ur, T_MOUNT)
    ARMS[name] = dict(car=car, ur=ur, side=side)
for _ in range(40):
    app.update()
CAR = stage.GetPrimAtPath(f"{ST}/Carrier")
SDOOR = stage.GetPrimAtPath(f"{ST}/Carrier/door")
SDOOR.SetTypeName("Xform")                         # authored as a plain grouping prim; make it movable
MARK_OPS = [UsdGeom.Xformable(stage.GetPrimAtPath(f"{ST}/Conveyor/{n}")).GetOrderedXformOps()[0] for n in ("slats", "ticks", "lanes")]
print("[ws30] station loaded:", CAR.IsValid(), SDOOR.IsValid(), len(MARK_OPS))
for n in ("klt0", "klt1", "shelf0", "shelf1", "shelf0_strut", "shelf1_strut", "trim_panel", "trim_holder"):
    pr = stage.GetPrimAtPath(f"{ST}/Carrier/{n}")     # a plain trolley: no bins, speakers or trim kit near the action
    if pr.IsValid():
        pr.SetActive(False)                            # some are plain grouping prims, so deactivate rather than hide
for name, a in ARMS.items():
    arm = rig.Arm(stage, str(a["ur"].GetPath()), T_MOUNT, None)
    G = rot_between(np.array([0, 0, 1.0]), arm.axis_l)
    arm.finger_l = G[:3, 0]
    g = f"{arm.tool}/gripper"
    setT(xf(g), tf(G))
    box(f"{g}/body", (0, 0, 0.08), (0.13, 0.09, 0.16), "dark")
    a["fingers"] = [box(f"{g}/finger{k}", (0, 0, 0.25), (0.025, 0.07, 0.14), "finger") for k in range(2)]
    a["arm"] = arm

# humanoids: stock G1 (29 dof, three-finger hands)
G1_URL = ROOT + "/Isaac/Robots/Unitree/G1/g1.usd"
HUM = {}
for name in J["humanoids"]:
    HUM[name] = dict(prim=reference(f"/World/{name}", G1_URL))
for _ in range(60):
    app.update()
for name, h in HUM.items():
    h["rig"] = rig.G1(stage, str(h["prim"].GetPath()), J["joints"][:13])

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
DX = P["door"][:, 0, 3]                       # door x per frame (it rides the belt, then the arms)


def label_t(name, text, default=None):
    return next((t for t, l in J["labels"].get(name, []) if l.startswith(text)), default)


def ev_t(head, default=None):
    return next((e["t"] for e in J["events"] if e["headline"] == head), default)


OVER = (lambda k: (2.6, -3.8, 6.4), lambda k: (2.6, 0.4, 0.0))      # under the hall's roof lights
if J["solo"]:
    w = "WORKER"
    SHOTS = [(0.0, *OVER),
             (3.0, lambda k: (DX[k] + 1.3, -5.2, 2.9), lambda k: (DX[k] + 1.0, 0.2, 0.8)),
             (label_t(w, "walking to the glass rack", 0) + 6.0, lambda k: (4.2, -9.6, 4.3), lambda k: (4.4, 0.4, 0.6)),
             (label_t(w, "placing glass") - 0.8, lambda k: (DX[k] + 1.5, -1.7, 1.8), lambda k: (DX[k] + 0.05, 0.15, 1.15)),
             (label_t(w, "walking to the parts shelf") + 1.0, lambda k: (4.2, -9.6, 4.3), lambda k: (4.4, 0.4, 0.6)),
             (label_t(w, "driving the screw") - 0.6, lambda k: (DX[k] + 0.09, 1.55, 1.08), lambda k: (DX[k] + 0.10, 0.45, 0.82)),
             (ev_t("STATION COMPLETE") - 0.5, lambda k: (DX[k] + 2.3, -3.1, 2.6), lambda k: (DX[k] + 0.3, 0.2, 0.9))]
else:
    SS = -1.0 if L["SCREW_IN_DOOR"][0] < 0 else 1.0       # screw on the downstream (-1) or upstream (+1) half
    SX = X2 - L["SCREW_IN_DOOR"][0]                        # screw x in the plant (door frame x points to plant -x)
    # WS30 plant frame: overhead, zoom onto the trolley, along the OUT rail, follow the door, G1s arriving, close-ups
    SY = DOOR_Y - L["SCREW_IN_DOOR"][1]                    # screw face y in the plant (door frame y points to plant -y)
    SHOTS = [(0.0, lambda k: (13.6, 0.3, 4.8), lambda k: (15.4, 3.4, 0.0)),
             (3.0, lambda k: (10.6, 5.9, 3.0), lambda k: (13.1, 3.1, 1.0)),
             (6.3, lambda k: (10.6, 0.6, 2.2), lambda k: (16.5, 3.0, 0.9)),
             (9.0, lambda k: (DX[k] - 2.6, 6.7, 2.7), lambda k: (DX[k] + 0.4, 3.1, 1.0)),
             (ev_t("DOOR DELIVERED TO WS2") + 0.3, lambda k: (22.6, 6.0, 2.6), lambda k: (20.0, 3.6, 0.9)),
             # both G1s at work on the two faces of the door: high, from past the downstream end
             (label_t("HUMANOID_B", "approach WS2") + 2.0, lambda k: (22.3, 3.1, 3.0), lambda k: (19.9, 3.1, 0.95)),
             (label_t("HUMANOID_A", "driving the screw") - 0.4,
              lambda k: (SX + 0.45, SY - 0.8 if SY < DOOR_Y else SY + 0.8, 1.2), lambda k: (SX, SY, 0.8)),
             (label_t("HUMANOID_A", "screw seated") + 0.4, lambda k: (21.0, 5.4, 2.0), lambda k: (19.85, 3.2, 1.2)),
             (ev_t("STATION 2 COMPLETE") - 0.5, lambda k: (21.8, 5.6, 2.7), lambda k: (19.8, 3.1, 0.9))]
    SHOTS.sort(key=lambda sh: sh[0])
print("[camera] shots at", [round(sh[0], 1) for sh in SHOTS])


def camera(k):
    """Current shot, blended in from the previous one (3 s for the opening zoom, 1.2 s otherwise)."""
    t = float(P["t"][k])
    i = max(j for j, sh in enumerate(SHOTS) if sh[0] <= t)
    e, g = np.array(SHOTS[i][1](k), float), np.array(SHOTS[i][2](k), float)
    if i > 0:
        s = minjerk((t - SHOTS[i][0]) / (3.0 if i == 1 else 1.2))
        if s < 1:
            e = lerp(np.array(SHOTS[i - 1][1](k), float), e, s)
            g = lerp(np.array(SHOTS[i - 1][2](k), float), g, s)
    return e, g


# ------------------------------------------------------------------ precompute every frame
frames = []
worst = {}
for k in range(N):
    f = dict(t=float(P["t"][k]))
    for name, a in ARMS.items():
        Tc = tf(None, (P[f"{name}.cx"][k], RAILS[name], RAIL_Z))
        q, ok = a["arm"].solve(Tc, P[f"{name}.tip"][k], P[f"{name}.axis"][k], finger_w=np.array([0, 1.0, 0]))
        f[name] = (Tc, q, float(P[f"{name}.grip"][k]))
    for name, h in HUM.items():
        T_root, q, err = h["rig"].solve(P[f"{name}.root"][k], P[f"{name}.q"][k][:13], P[f"{name}.hand"][k], P[f"{name}.w"][k])
        f[name] = (T_root, q)
        worst[name] = max(worst.get(name, 0.0), err)
    f["cam"] = camera(k)
    frames.append(f)
print("[precompute] hand IK worst error (m):", {n: round(v, 3) for n, v in worst.items()})


def _quat(R):
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return np.array([w, x, y, z])


def mblend(A, B, u):
    """Rigid transform between A and B (normalised quaternion blend): motion-blur time samples between frames."""
    if u <= 0:
        return A
    qa, qb = _quat(A[:3, :3]), _quat(B[:3, :3])
    qb = qb if np.dot(qa, qb) >= 0 else -qb
    w, x, y, z = (qa * (1 - u) + qb * u) / np.linalg.norm(qa * (1 - u) + qb * u)
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    return tf(R, lerp(A[:3, 3], B[:3, 3], u))


def apply(k, u=0.0):
    """Pose the scene at frame k, or u of the way to frame k + 1 (motion blur samples)."""
    k2 = min(k + 1, N - 1)
    u = u if k2 > k else 0.0
    f, g = frames[k], frames[k2]
    PK = lambda name: mblend(P[name][k], P[name][k2], u)  # noqa: E731
    t = f["t"] + (g["t"] - f["t"]) * u
    plant.animate(t)
    s_belt = L["BELT"] * t
    Tcar = tf(None, (CAR_X0 + s_belt, CAR_Y, 0.0))
    with Sdf.ChangeBlock():
        UsdGeom.Xformable(CAR).GetOrderedXformOps()[0].Set(Gf.Vec3d(*Tcar[:3, 3]))
        for op, pitch in zip(MARK_OPS, (0.1, 0.3, None)):      # belt marks ride the belt (ws30_station.belt_marks)
            op.Set(Gf.Vec3d(s_belt % pitch if pitch else s_belt, 0, 0))
        Tdoor = PK("door")
        setT(SDOOR, np.linalg.inv(Tcar) @ Tdoor @ np.linalg.inv(T_DOOR_ON_CAR))
        setT(DOOR_MARKS, Tdoor)
        setT(GLASS, PK("glass"))
        if "lifter" in P.files:
            setT(LIFTER, PK("lifter"))
        setT(DRIVER, PK("driver"))
        setT(SCREW, PK("screw"))
        setT(BIT, tf(rot(0, float(P["spin"][k] + (P["spin"][k2] - P["spin"][k]) * u))))
        LOOKS["screwhead"][1].Set(Gf.Vec3f(*(SCREW_GREEN if P["screw_state"][k] >= 2 else SCREW_RED)))
        for name, a in ARMS.items():
            setT(a["car"], mblend(f[name][0], g[name][0], u))
            gap = lerp(0.16, 0.10, f[name][2] + (g[name][2] - f[name][2]) * u)
            for sgn, fp in zip((-1, 1), a["fingers"]):
                UsdGeom.Xformable(fp).GetOrderedXformOps()[0].Set(Gf.Vec3d(sgn * (gap / 2 + 0.0125), 0, 0.25))
        for name, h in HUM.items():
            setT(h["prim"], mblend(f[name][0], g[name][0], u))
        eye, tgt = lerp(f["cam"][0], g["cam"][0], u), lerp(f["cam"][1], g["cam"][1], u)
        cam_op.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*tgt), Gf.Vec3d(0, 0, 1)).GetInverse())
        if args.hq and args.dof > 0:                                 # focus on what the shot looks at
            cam.GetFocusDistanceAttr().Set(float(np.linalg.norm(np.asarray(tgt) - np.asarray(eye))))
    for name, a in ARMS.items():
        a["arm"].write(lerp(f[name][1], g[name][1], u))
    for name, h in HUM.items():
        h["rig"].write(lerp(f[name][1], g[name][1], u))


# ------------------------------------------------------------------ render
os.makedirs(args.out, exist_ok=True)
for i in range(400):
    app.update()
    msg, loaded, total = omni.usd.get_context().get_stage_loading_status()
    if i > 60 and total == 0:
        break
print("[render] stage loaded after", i, "updates")
if args.hq:
    import carb  # noqa: E402
    import omni.replicator.core as rep  # noqa: E402
    from PIL import Image  # noqa: E402

    import look_hq  # noqa: E402
    look_hq.apply(stage, list(ARMS), [((13.2, CAR_Y, 1.1), (2.0, 1.4), 2500, (1.0, 0.96, 0.9)),
                                      ((X2, DOOR_Y, 1.0), (2.4, 1.5), 3200, (1.0, 0.97, 0.92))])
    st_ = carb.settings.get_settings()
    st_.set("/rtx/rendermode", "PathTracing")
    st_.set("/rtx/pathtracing/spp", args.spp)
    st_.set("/rtx/pathtracing/totalSpp", args.spp * args.subframes)
    st_.set("/rtx/pathtracing/optixDenoiser/enabled", True)
    st_.set("/rtx/post/motionblur/enabled", False)
    st_.set("/omni/replicator/captureOnPlay", False)
    st_.set("/rtx-transient/resourcemanager/enableTextureStreaming", False)   # full-resolution mips from the start
    if args.dof > 0:
        cam.CreateFStopAttr(args.dof)
        cam.CreateFocusDistanceAttr(5.0)
    rp = rep.create.render_product("/World/Cam", (W, H))
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach(rp)
    for _ in range(60):                                # materials and textures finish loading
        app.update()
for _ in range(30):
    apply(0)
    app.update()
meta = dict(W=W, H=H, focal=FOCAL, hap=HAP, fps=FPS, frames=[])
t_start, first = time.time(), True
last = N if args.secs <= 0 else min(N, int(args.secs * FPS))
which = [min(int(round(float(x) * FPS)), N - 1) for x in args.stills.split(",")] if args.stills else range(int(args.start * FPS), last)
for k in which:
    if args.hq:                                        # path traced: time samples over half a frame, averaged
        acc = 0.0
        for j in range(args.mblur):
            apply(k, (j / (args.mblur - 1) - 0.5) * 0.5 + 0.5 if args.mblur > 1 else 0.0)
            rep.orchestrator.step(rt_subframes=args.subframes * (4 if (first or args.stills) else 1), delta_time=0.0, pause_timeline=True)
            first = False
            acc = acc + np.asarray(rgb.get_data())[..., :3].astype(np.float32)
        Image.fromarray((acc / args.mblur + 0.5).astype(np.uint8)).save(os.path.join(args.out, f"frame_{k:05d}.jpg"), quality=95)
    else:
        apply(k)
        for _ in range(8 if args.stills else 1):
            app.update()
        capture_viewport_to_file(vp, os.path.join(args.out, f"frame_{k:05d}.png"))
        for _ in range(4 if args.stills else 1):
            app.update()
    f = frames[k]
    anchors = {n: (f[n][0][:3, 3] + [0, 0, PED + 1.1]).tolist() for n in ARMS}
    anchors.update({n: (f[n][0][:3, 3] + [0, 0, 0.75]).tolist() for n in HUM})
    meta["frames"].append(dict(k=k, t=round(f["t"], 3), eye=list(map(float, f["cam"][0])), target=list(map(float, f["cam"][1])),
                               anchors=anchors))
    if k % 90 == 0 or args.stills:
        n_ = len(meta["frames"])
        print(f"[render] frame {k}/{last}, {(time.time() - t_start) / n_:.2f} s per frame", flush=True)
for _ in range(20):
    app.update()
json.dump(meta, open(os.path.join(args.out, "meta.json"), "w"))
print("[render] done", len(meta["frames"]), "frames")
app.close()
