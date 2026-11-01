"""CarDA WS30 station (Stellantis Melfi lab line) as an Isaac Sim 6.2 scene, built from layout.json, plus the
"human alone" clip: the real Xsens worker (register_bvh.py) as capsules beside the door riding the belt.

Everything is authored geometry in the CarDA plant frame (metres, z up, +x = belt travel), so the station USD
has no external references. Kinematic replay: nothing is simulated by PhysX.

  python ws30_station.py LAYOUT OUT_DIR --cams cameras_ws30.json [--worker worker_plant.npz]
         [--clip 6:30] [--stills in@36,out@9,orbit@20]

Writes OUT_DIR/ws30_station.usd (station only, default prim /World, cameras CAMERA_IN30 and CAMERA_OUT30 with the
CarDA intrinsics), then optionally OUT_DIR/clip/frame_*.png (orbit view) and OUT_DIR/still_<view>_<t>.png.
"""
import argparse
import json
import math
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("layout")
ap.add_argument("out")
ap.add_argument("--cams", required=True)
ap.add_argument("--worker", default="")
ap.add_argument("--clip", default="", help="cycle seconds A:B rendered from the orbit view at --fps")
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--stills", default="", help="view@t,...; view is in, out (CarDA cameras) or orbit")
args = ap.parse_args()
W, H = 1280, 720

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "width": W, "height": H})

import omni.usd  # noqa: E402
from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport  # noqa: E402
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade  # noqa: E402

L = json.load(open(args.layout))
CAMS = json.load(open(args.cams))["cams"]
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
stage.SetDefaultPrim(world)


# ------------------------------------------------------------------ helpers
def mat(name, rgb, rough=0.5, metal=0.0, opacity=1.0):
    m = UsdShade.Material.Define(stage, f"/World/Looks/{name}")
    sh = UsdShade.Shader.Define(stage, f"/World/Looks/{name}/S")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metal)
    if opacity < 1:
        sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
    m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return m


MATS = {k: mat(k, *v) for k, v in {
    "hall": ((0.42, 0.43, 0.44), 0.8), "plate": ((0.08, 0.3, 0.6), 0.55, 0.2), "belt": ((0.07, 0.07, 0.075), 0.9),
    "slat": ((0.035, 0.035, 0.04), 0.9), "yellow": ((0.95, 0.7, 0.04), 0.55), "black": ((0.02, 0.02, 0.02), 0.6),
    "white": ((0.9, 0.9, 0.88), 0.6), "alu": ((0.72, 0.74, 0.77), 0.35, 0.8), "grey": ((0.55, 0.56, 0.58), 0.5, 0.2),
    "paint": ((0.015, 0.015, 0.018), 0.18, 0.4), "rubber": ((0.03, 0.03, 0.03), 0.9), "klt": ((0.05, 0.22, 0.62), 0.5),
    "orange": ((0.95, 0.4, 0.05), 0.5), "desk": ((0.86, 0.84, 0.78), 0.6), "screen": ((0.02, 0.03, 0.05), 0.2),
    "cabinet": ((0.8, 0.81, 0.8), 0.4, 0.3), "blue_beam": ((0.1, 0.25, 0.75), 0.4, 0.3), "cone": ((0.1, 0.1, 0.1), 0.7),
    "shirt": ((0.03, 0.03, 0.035), 0.8), "jeans": ((0.13, 0.17, 0.28), 0.85), "skin": ((0.78, 0.56, 0.44), 0.6),
    "hair": ((0.06, 0.04, 0.03), 0.8), "shoe": ((0.25, 0.25, 0.27), 0.7), "strap": ((0.95, 0.42, 0.08), 0.6)}.items()}


def bind(prim, name):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(MATS[name])


def xform(path, t=(0, 0, 0), rz=0.0):
    x = UsdGeom.Xform.Define(stage, path)
    x.AddTranslateOp().Set(Gf.Vec3d(*map(float, t)))
    if rz:
        x.AddRotateZOp().Set(float(rz))
    return x


def box(path, c, size, m, rot=None):
    g = UsdGeom.Cube.Define(stage, path)
    g.CreateSizeAttr(1.0)
    x = UsdGeom.Xformable(g)
    x.AddTranslateOp().Set(Gf.Vec3d(*map(float, c)))
    if rot:
        x.AddRotateXYZOp().Set(Gf.Vec3f(*map(float, rot)))
    x.AddScaleOp().Set(Gf.Vec3f(*map(float, size)))
    bind(g.GetPrim(), m)
    return g


def cyl(path, c, r, h, m, axis="Z"):
    g = UsdGeom.Cylinder.Define(stage, path)
    g.CreateRadiusAttr(float(r))
    g.CreateHeightAttr(float(h))
    g.CreateAxisAttr(axis)
    UsdGeom.Xformable(g).AddTranslateOp().Set(Gf.Vec3d(*map(float, c)))
    bind(g.GetPrim(), m)
    return g


def rects(path, rs, z, m):
    """One mesh of axis-aligned floor rectangles (x0, x1, y0, y1) at height z."""
    pts, idx = [], []
    for x0, x1, y0, y1 in rs:
        n = len(pts)
        pts += [(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)]
        idx += [n, n + 1, n + 2, n + 3]
    g = UsdGeom.Mesh.Define(stage, path)
    g.CreatePointsAttr([Gf.Vec3f(*map(float, p)) for p in pts])
    g.CreateFaceVertexCountsAttr([4] * len(rs))
    g.CreateFaceVertexIndicesAttr(idx)
    g.CreateNormalsAttr([Gf.Vec3f(0, 0, 1)] * len(pts))
    g.SetNormalsInterpolation("vertex")
    bind(g.GetPrim(), m)
    return g


def tube(path, a, b, w, m):
    """Square tube of width w from a to b (any direction)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    g = UsdGeom.Cube.Define(stage, path)
    g.CreateSizeAttr(1.0)
    x = UsdGeom.Xformable(g)
    x.AddTranslateOp().Set(Gf.Vec3d(*map(float, (a + b) / 2)))
    pitch = -math.degrees(math.atan2(d[2], math.hypot(d[0], d[1])))
    x.AddRotateXYZOp().Set(Gf.Vec3f(0.0, pitch, math.degrees(math.atan2(d[1], d[0]))))   # Rz(yaw) Ry(pitch)
    x.AddScaleOp().Set(Gf.Vec3f(float(np.linalg.norm(d)), w, w))
    bind(g.GetPrim(), m)
    return g


def prism(path, outline_xz, y0, y1, m):
    """Polygon in the x-z plane extruded along y (door skins)."""
    n = len(outline_xz)
    pts = [(x, y0, z) for x, z in outline_xz] + [(x, y1, z) for x, z in outline_xz]
    counts, idx = [n, n], list(range(n))[::-1] + [n + i for i in range(n)]
    for i in range(n):
        j = (i + 1) % n
        counts.append(4)
        idx += [i, j, n + j, n + i]
    g = UsdGeom.Mesh.Define(stage, path)
    g.CreatePointsAttr([Gf.Vec3f(*map(float, p)) for p in pts])
    g.CreateFaceVertexCountsAttr(counts)
    g.CreateFaceVertexIndicesAttr(idx)
    g.CreateSubdivisionSchemeAttr("none")
    g.CreateDoubleSidedAttr(True)
    bind(g.GetPrim(), m)
    return g


# ------------------------------------------------------------------ floor, platform, conveyor
S = "/World/WS30"
xform(S)
Z0 = 0.002   # everything on the floor sits 2 mm up so the viewport grid at z = 0 stays hidden
box(f"{S}/Hall/floor", (10, 2.9, Z0 - 0.026), (44, 26, 0.05), "hall")
P = L["platform"]
px0, px1, py0, py1 = *P["x"], *P["y"]
box(f"{S}/Platform/plates", ((px0 + px1) / 2, (py0 + py1) / 2, Z0 - 0.004), (px1 - px0, py1 - py0, 0.008), "plate")
seams = [(x, x + 0.01, py0, py1) for x in np.arange(px0 + 1.0, px1, 1.0)] + \
        [(px0, px1, y, y + 0.01) for y in np.arange(py0 + 1.0, py1, 1.0)]
rects(f"{S}/Platform/seams", seams, Z0 + 0.0005, "black")
haz = [(x, x + 0.1, y0, y0 + 0.06) for x in np.arange(px0, px1, 0.2) for y0 in (py0, py1 - 0.06)]
rects(f"{S}/Platform/hazard_y", haz, Z0 + 0.0008, "black")
rects(f"{S}/Platform/hazard_y_bg", [(px0, px1, py0, py0 + 0.06), (px0, px1, py1 - 0.06, py1)], Z0 + 0.0006, "yellow")

C = L["conveyor"]
bx0, bx1 = C["belt_x"]
by0, by1 = C["belt_y"]
box(f"{S}/Conveyor/belt", ((bx0 + bx1) / 2, (by0 + by1) / 2, Z0 - 0.002), (bx1 - bx0, by1 - by0, 0.006), "belt")
for i, (y0, y1) in enumerate(C["edge_strips_y"]):
    box(f"{S}/Conveyor/edge{i}", ((bx0 + bx1) / 2, (y0 + y1) / 2, Z0 - 0.001), (bx1 - bx0, y1 - y0, 0.008), "yellow")
    inner = (y0, y0 + 0.55 * (y1 - y0)) if i == 1 else (y1 - 0.55 * (y1 - y0), y1)
    rects(f"{S}/Conveyor/teeth{i}", [(x, x + 0.05, *inner) for x in np.arange(bx0, bx1, 0.1)], Z0 + 0.0035, "black")
# marks on the belt itself ride with it: slats and ticks shift modulo their pitch, lane lines shift and hide past the end
MARKS = []
for name, pitch, rs, z, m in (
        ("slats", 0.1, [(x, x + 0.012, by0, by1) for x in np.arange(bx0, bx1 - 0.1, 0.1)], Z0 + 0.0012, "slat"),
        ("ticks", 0.3, [(x, x + 0.03, y, y + 0.12) for x in np.arange(bx0, bx1 - 0.3, 0.3) for y in (by0 + 0.03, by1 - 0.15)],
         Z0 + 0.0015, "white")):
    MARKS.append((xform(f"{S}/Conveyor/{name}").GetOrderedXformOps()[0], pitch))
    rects(f"{S}/Conveyor/{name}/mesh", rs, z, m)
LANE_OP = xform(f"{S}/Conveyor/lanes").GetOrderedXformOps()[0]
LANES = [rects(f"{S}/Conveyor/lanes/lane{i}", [(x, x + 0.05, by0, by1)], Z0 + 0.0016, "white") for i, x in enumerate(C["lane_lines_x"])]


def belt_marks(t):
    s = C["speed_mps"] * t
    for op, pitch in MARKS:
        op.Set(Gf.Vec3d(s % pitch, 0, 0))
    LANE_OP.Set(Gf.Vec3d(s, 0, 0))
    for g, x in zip(LANES, C["lane_lines_x"]):
        g.GetVisibilityAttr().Set("invisible" if x + s > bx1 - 0.05 else "inherited")

# ------------------------------------------------------------------ door carrier (moves with the belt)
K = L["carrier"]
CAR = f"{S}/Carrier"
car = xform(CAR, (K["pose_t0"]["x"], K["pose_t0"]["y"], 0.0))
CAR_OP = car.GetOrderedXformOps()[0]
for i, (cx, cy) in enumerate(K["castors_local"]):
    cyl(f"{CAR}/castor{i}/wheel", (cx, cy, 0.06), 0.06, 0.04, "rubber", axis="Y")
    box(f"{CAR}/castor{i}/fork", (cx, cy, 0.105), (0.08, 0.06, 0.05), "grey")
F = K["frame_size_local"]
fx0, fx1, fy0, fy1, ftop = *F["x"], *F["y"], F["z_top"]
dy_door = L["door"]["local"]["y_center"]
w = 0.045
for sy, y, top in (("o", fy0, ftop), ("i", fy1, 0.5)):      # IN side low: the worker reaches the door over it
    tube(f"{CAR}/frame/base_{sy}", (fx0, y, 0.15), (fx1, y, 0.15), w, "alu")
    tube(f"{CAR}/frame/top_{sy}", (fx0, y, top), (fx1, y, top), w, "alu")
    for sx, x in (("u", fx0 + 0.03), ("d", fx1 - 0.03)):
        tube(f"{CAR}/frame/post_{sy}{sx}", (x, y, 0.13), (x, y, top), w, "alu")
tube(f"{CAR}/frame/mid_o", (fx0, fy0, 0.6), (fx1, fy0, 0.6), w, "alu")
for sx, x in (("u", fx0 + 0.03), ("d", fx1 - 0.03)):
    tube(f"{CAR}/frame/cross_{sx}15", (x, fy0, 0.15), (x, fy1, 0.15), w, "alu")
    tube(f"{CAR}/frame/cross_{sx}top", (x, fy0, ftop), (x, dy_door, ftop), w, "alu")
box(f"{CAR}/frame/centre_plate", (0.05, fy0 + 0.02, 0.55), (0.28, 0.02, 0.75), "grey")
for k, b in enumerate(K["klt_bins_local"]):
    c, s = np.array(b["center"]), np.array(b["size"])
    sh = c - [0, 0, s[2] / 2 + 0.012]
    box(f"{CAR}/shelf{k}", sh, (s[0] + 0.06, s[1] + 0.06, 0.02), "alu")
    tube(f"{CAR}/shelf{k}_strut", (sh[0], sh[1], sh[2] - 0.01), (np.sign(sh[0]) * (abs(fx1) - 0.03), sh[1], 0.3), 0.035, "alu")
    bz = c[2] - s[2] / 2
    box(f"{CAR}/klt{k}/bottom", (c[0], c[1], bz + 0.005), (s[0], s[1], 0.01), "klt")
    for j, (dx, dy, sx_, sy_) in enumerate([(s[0] / 2, 0, 0.012, s[1]), (-s[0] / 2, 0, 0.012, s[1]),
                                            (0, s[1] / 2, s[0], 0.012), (0, -s[1] / 2, s[0], 0.012)]):
        box(f"{CAR}/klt{k}/wall{j}", (c[0] + dx, c[1] + dy, c[2]), (sx_, sy_, s[2]), "klt")
    for j in range(3):                                   # speakers waiting in the bin
        cyl(f"{CAR}/klt{k}/part{j}", (c[0] - 0.12 + 0.12 * j, c[1], bz + 0.06), 0.06, 0.08, "cone")
T = K["trim_panel_local"]
box(f"{CAR}/trim_panel", T["center"], T["size"], "paint", rot=(T["tilt_deg"], 0, 0))
tube(f"{CAR}/trim_holder", (fx0, T["center"][1], 0.3), (T["center"][0], T["center"][1], 0.3), 0.035, "alu")

D = L["door"]["local"]
dx0, dx1, dy, dt = D["x"][0], D["x"][1], D["y_center"], D["thickness"]
zb, zl, zt = D["z_bottom"], D["z_beltline"], D["z_top"]
lower = [(dx0, zb + 0.08), (dx0 + 0.06, zb), (dx1 - 0.14, zb), (dx1, zb + 0.16), (dx1, zl), (dx0, zl)]
prism(f"{CAR}/door/skin", lower, dy - dt / 2, dy + dt / 2, "paint")
fr = [(dx0 + 0.02, zl), (dx0 + 0.02, zt - 0.12), (dx0 + 0.16, zt), (dx1 - 0.42, zt), (dx1 - 0.02, zl + 0.2), (dx1 - 0.02, zl)]
for i in range(len(fr) - 1):
    tube(f"{CAR}/door/frame{i}", (fr[i][0], dy, fr[i][1]), (fr[i + 1][0], dy, fr[i + 1][1]), 0.035, "paint")
prism(f"{CAR}/door/sail", [(dx1 - 0.02, zl), (dx1 - 0.02, zl + 0.18), (dx1 - 0.14, zl)], dy - 0.02, dy + 0.02, "paint")
sp = L["door"]["speaker_local"]
cyl(f"{CAR}/door/speaker", (sp["center"][0], dy + dt / 2 + 0.012, sp["center"][2]), sp["radius"], 0.025, "cone", axis="Y")
cyl(f"{CAR}/door/speaker_ring", (sp["center"][0], dy + dt / 2 + 0.004, sp["center"][2]), sp["radius"] + 0.02, 0.008, "grey", axis="Y")
box(f"{CAR}/door/handle", (dx0 + 0.22, dy + dt / 2 + 0.01, zl - 0.12), (0.14, 0.02, 0.05), "grey")
for j, x in enumerate((dx0 + 0.15, (dx0 + dx1) / 2, dx1 - 0.2)):           # fixture arms and clamps under the door
    tube(f"{CAR}/fixture/arm{j}", (x, dy, ftop), (x, dy, zb), 0.04, "alu")
    box(f"{CAR}/fixture/clamp{j}", (x, dy, zb + 0.02), (0.06, dt + 0.04, 0.05), "orange")

# ------------------------------------------------------------------ line side
LS = L["line_side"]
R = LS["screwdriver_rack"]
(rx, ry), (sx, sy, sz) = R["center"], R["size"]
for i, (ox, oy) in enumerate([(-1, -1), (1, -1), (-1, 1), (1, 1)]):
    tube(f"{S}/Rack/post{i}", (rx + ox * sx / 2, ry + oy * sy / 2, 0), (rx + ox * sx / 2, ry + oy * sy / 2, sz), 0.03, "white")
for z in (0.3, 0.9, 1.4, sz):
    for oy in (-1, 1):
        tube(f"{S}/Rack/rail{int(z * 100)}{oy + 1}", (rx - sx / 2, ry + oy * sy / 2, z), (rx + sx / 2, ry + oy * sy / 2, z), 0.03, "white")
box(f"{S}/Rack/shelf", (rx, ry, 0.9), (sx, sy, 0.02), "grey")
box(f"{S}/Rack/tool_body", (rx - 0.1, ry - sy / 2 + 0.05, 1.12), (0.2, 0.06, 0.07), "orange")
box(f"{S}/Rack/tool_grip", (rx - 0.05, ry - sy / 2 + 0.05, 1.02), (0.05, 0.05, 0.16), "black")


def desk(path, c, s, chair_side):
    (x, y), (lx, ly, h) = c, s
    box(f"{path}/top", (x, y, h - 0.015), (lx, ly, 0.03), "desk")
    for i, (ox, oy) in enumerate([(-1, -1), (1, -1), (-1, 1), (1, 1)]):
        tube(f"{path}/leg{i}", (x + ox * (lx / 2 - 0.05), y + oy * (ly / 2 - 0.05), 0), (x + ox * (lx / 2 - 0.05), y + oy * (ly / 2 - 0.05), h - 0.03), 0.04, "grey")
    box(f"{path}/monitor", (x, y - chair_side * 0.2, h + 0.3), (0.6, 0.03, 0.36), "screen")
    tube(f"{path}/monitor_arm", (x, y - chair_side * 0.2, h), (x, y - chair_side * 0.2, h + 0.15), 0.04, "black")
    cyl(f"{path}/chair_seat", (x, y + chair_side * (ly / 2 + 0.3), 0.48), 0.24, 0.08, "black")
    box(f"{path}/chair_back", (x, y + chair_side * (ly / 2 + 0.52), 0.8), (0.42, 0.05, 0.5), "black")
    cyl(f"{path}/chair_post", (x, y + chair_side * (ly / 2 + 0.3), 0.24), 0.03, 0.45, "grey")


desk(f"{S}/Desk_in", LS["desk_in_side"]["center"], LS["desk_in_side"]["size"], 1)
desk(f"{S}/Desk_out", LS["desks_out_side"]["center"], LS["desks_out_side"]["size"], -1)
mc = LS["monitor_cart_out_side"]
box(f"{S}/MonitorCart/frame", (*mc["center"], mc["size"][2] / 2), mc["size"], "alu")
box(f"{S}/MonitorCart/screen", (mc["center"][0], mc["center"][1] + 0.31, 1.3), (0.55, 0.02, 0.35), "screen")
cab = LS["electrical_cabinet"]
box(f"{S}/Cabinet", (*cab["center"], cab["size"][2] / 2), cab["size"], "cabinet")
cb = LS["checkerboard_stand"]
(cx_, cy_), (bw, bh), (bz0, bz1) = cb["center"], cb["board_size"], cb["board_z"]
box(f"{S}/Checker/board", (cx_, cy_, (bz0 + bz1) / 2), (bw, 0.01, bz1 - bz0), "white")
sq = []
n_c, n_r = 6, 8
for i in range(n_c):
    for j in range(n_r):
        if (i + j) % 2 == 0:
            sq.append((cx_ - bw / 2 + 0.03 + i * (bw - 0.06) / n_c, cx_ - bw / 2 + 0.03 + (i + 1) * (bw - 0.06) / n_c,
                       bz0 + 0.03 + j * (bh - 0.06) / n_r, bz0 + 0.03 + (j + 1) * (bh - 0.06) / n_r))
pts, idx = [], []
for x0, x1, z0, z1 in sq:
    n = len(pts)
    pts += [(x0, cy_ - 0.006, z0), (x1, cy_ - 0.006, z0), (x1, cy_ - 0.006, z1), (x0, cy_ - 0.006, z1)]
    idx += [n, n + 3, n + 2, n + 1]
g = UsdGeom.Mesh.Define(stage, f"{S}/Checker/squares")
g.CreatePointsAttr([Gf.Vec3f(*map(float, p)) for p in pts])
g.CreateFaceVertexCountsAttr([4] * len(sq))
g.CreateFaceVertexIndicesAttr(idx)
g.CreateDoubleSidedAttr(True)
bind(g.GetPrim(), "black")
for i, ox in enumerate((-0.25, 0.25)):
    tube(f"{S}/Checker/leg{i}", (cx_ + ox, cy_, 0), (cx_ + ox, cy_, bz0), 0.03, "white")
for i, (gx, gy) in enumerate(LS["gantry_posts"]):
    hgt = LS["gantry_post_height"]
    for k, z in enumerate(np.arange(0, hgt, 0.25)):
        box(f"{S}/Gantry/post{i}_{k}", (gx, gy, z + 0.125), (0.08, 0.08, 0.25), "yellow" if k % 2 == 0 else "black")
    tube(f"{S}/Gantry/beam{i}", (gx - 0.1, gy, hgt + 0.08), (gx + 3.0, gy, hgt + 0.08), 0.14, "blue_beam")


# ------------------------------------------------------------------ CarDA cameras (ZED 2 on rods) as USD cameras
def cam_prim(path, K_, R_, C_):
    cam = UsdGeom.Camera.Define(stage, path)
    hap = 20.955
    cam.CreateHorizontalApertureAttr(hap)
    cam.CreateVerticalApertureAttr(hap * H / W)
    cam.CreateFocalLengthAttr(float(K_[0][0]) * hap / W)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 300))
    M = np.eye(4)
    M[:3, :3] = np.array(R_).T @ np.diag([1, -1, -1])        # OpenCV camera -> USD camera (looks down -z, y up)
    M[:3, 3] = C_
    UsdGeom.Xformable(cam).AddTransformOp().Set(Gf.Matrix4d(*M.T.reshape(-1).tolist()))
    return cam


for side, cm in CAMS.items():
    Cc = np.array(cm["C"])
    cam_prim(f"{S}/Cameras/CAMERA_{side.upper()}30", cm["K"], cm["R"], Cc)
    cyl(f"{S}/Cameras/rod_{side}", (Cc[0], Cc[1], (Cc[2] - 0.05) / 2), 0.02, Cc[2] - 0.05, "grey")
    box(f"{S}/Cameras/zed_{side}", Cc, (0.175, 0.04, 0.033), "black")

# ------------------------------------------------------------------ light
dome = UsdLux.DomeLight.Define(stage, f"{S}/Lights/dome")
dome.CreateIntensityAttr(900.0)
dome.CreateColorAttr(Gf.Vec3f(0.95, 0.97, 1.0))
sun = UsdLux.DistantLight.Define(stage, f"{S}/Lights/key")
sun.CreateIntensityAttr(1200.0)
sun.CreateAngleAttr(20.0)
UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(25, 10, 30))
for i, x in enumerate((12.0, 16.0)):
    rl = UsdLux.RectLight.Define(stage, f"{S}/Lights/ceiling{i}")
    rl.CreateWidthAttr(3.0)
    rl.CreateHeightAttr(1.0)
    rl.CreateIntensityAttr(3000.0)
    UsdGeom.Xformable(rl).AddTranslateOp().Set(Gf.Vec3d(x, 3.0, 5.0))

os.makedirs(args.out, exist_ok=True)
stage.GetRootLayer().Export(os.path.join(args.out, "ws30_station.usd"))
print("[ws30] wrote", os.path.join(args.out, "ws30_station.usd"))


def carrier_x(t):
    return K["pose_t0"]["x"] + C["speed_mps"] * t


# ------------------------------------------------------------------ worker: Xsens joints as capsules
class Worker:
    RAD = [("Hip", 0.075), ("Knee", 0.06), ("Ankle", 0.045), ("Chest", 0.135), ("Neck", 0.055), ("Collar", 0.06),
           ("Shoulder", 0.055), ("Elbow", 0.045)]

    def __init__(self, npz):
        d = np.load(npz)
        self.J, self.fps = d["joints"], float(d["fps"])
        names, par = [str(n) for n in d["names"]], [int(p) for p in d["parents"]]
        self.idx = {n: i for i, n in enumerate(names)}
        xform("/World/Worker")
        self.bones = []
        for c, p in enumerate(par):
            if p < 0 or np.linalg.norm(self.J[0, c] - self.J[0, p]) < 0.03:
                continue
            pn = names[p]
            r = next((v for k, v in self.RAD if k in pn), 0.11 if pn == "Hips" else 0.06)
            m = ("jeans" if any(k in pn for k in ("Hip", "Knee")) else "shoe" if "Ankle" in pn
                 else "skin" if any(k in pn for k in ("Elbow", "Neck")) else "shirt")
            if pn == "Hips" and names[c] == "Chest":
                r = 0.13
            g = UsdGeom.Capsule.Define(stage, f"/World/Worker/b{c}")
            g.CreateRadiusAttr(r)
            g.CreateHeightAttr(0.1)
            g.CreateAxisAttr("Z")
            bind(g.GetPrim(), m)
            self.bones.append((p, c, g, UsdGeom.Xformable(g).AddTransformOp()))
        self.extra = {}
        for name, r, m in (("head", 0.1, "skin"), ("hair", 0.102, "hair"), ("handR", 0.045, "skin"), ("handL", 0.045, "skin")):
            s = UsdGeom.Sphere.Define(stage, f"/World/Worker/{name}")
            s.CreateRadiusAttr(r)
            bind(s.GetPrim(), m)
            self.extra[name] = UsdGeom.Xformable(s).AddTranslateOp()

    def pose(self, t):
        P = self.J[min(int(round(t * self.fps)), len(self.J) - 1)]
        for p, c, g, op in self.bones:
            a, b = P[p].astype(float), P[c].astype(float)
            d = b - a
            ln = max(float(np.linalg.norm(d)), 1e-4)
            z = d / ln
            ax = np.cross([0, 0, 1.0], z)
            s = np.linalg.norm(ax)
            Rm = np.eye(3) if s < 1e-8 else _rodrigues(ax / s, math.atan2(s, z[2]))
            M = np.eye(4)
            M[:3, :3], M[:3, 3] = Rm, (a + b) / 2
            g.GetHeightAttr().Set(ln)
            op.Set(Gf.Matrix4d(*M.T.reshape(-1).tolist()))
        i = self.idx
        up = P[i["Head"]] - P[i["Neck"]]
        up = up / max(np.linalg.norm(up), 1e-6)
        head = P[i["Head"]] + 0.09 * up
        self.extra["head"].Set(Gf.Vec3d(*map(float, head)))
        self.extra["hair"].Set(Gf.Vec3d(*map(float, head + 0.015 * up)))
        for side in ("R", "L"):
            nm = "Right" if side == "R" else "Left"
            e, wr = P[i[nm + "Elbow"]], P[i[nm + "Wrist"]]
            v = (wr - e) / max(np.linalg.norm(wr - e), 1e-6)
            self.extra["hand" + side].Set(Gf.Vec3d(*map(float, wr + 0.07 * v)))


def _rodrigues(k, a):
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(a) * Kx + (1 - math.cos(a)) * Kx @ Kx


worker = Worker(args.worker) if args.worker else None
import carb  # noqa: E402
for key in ("/app/viewport/grid/enabled", "/persistent/app/viewport/Viewport/Viewport0/guide/grid/visible"):
    carb.settings.get_settings().set(key, False)
if not (args.clip or args.stills):
    app.close()
    raise SystemExit

# ------------------------------------------------------------------ render
rc = UsdGeom.Camera.Define(stage, "/World/RenderCam")
rc.CreateFocalLengthAttr(14.0)
rc.CreateHorizontalApertureAttr(20.955)
rc.CreateVerticalApertureAttr(20.955 * H / W)
rc.CreateClippingRangeAttr(Gf.Vec2f(0.05, 300))
rc_op = UsdGeom.Xformable(rc).AddTransformOp()
vp = get_active_viewport()


def orbit(t):
    """High 3/4 view from the IN side, upstream, riding along with the carrier; the line-side rack stays in frame."""
    x = carrier_x(t)
    eye = np.array([x - 1.2, 7.0, 3.3])
    tgt = np.array([x + 0.7, 3.8, 0.8])
    m = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*tgt), Gf.Vec3d(0, 0, 1)).GetInverse()
    rc_op.Set(m)
    return "/World/RenderCam"


def apply(t, view):
    CAR_OP.Set(Gf.Vec3d(carrier_x(t), K["pose_t0"]["y"], 0.0))
    belt_marks(t)
    if worker:
        worker.pose(t)
    vp.camera_path = orbit(t) if view == "orbit" else f"{S}/Cameras/CAMERA_{view.upper()}30"


for i in range(300):
    app.update()
    if i > 40 and omni.usd.get_context().get_stage_loading_status()[2] == 0:
        break
for _ in range(20):
    apply(0.0, "orbit")
    app.update()
meta = dict(fps=args.fps, frames=[])
for item in [s for s in args.stills.split(",") if s]:
    view, t = item.split("@")
    apply(float(t), view)
    for _ in range(30):
        app.update()
    p = os.path.join(args.out, f"still_{view}_{float(t):05.1f}.png")
    capture_viewport_to_file(vp, p)
    for _ in range(10):
        app.update()
    print("[ws30] still", p, flush=True)
if args.clip:
    a, b = map(float, args.clip.split(":"))
    os.makedirs(os.path.join(args.out, "clip"), exist_ok=True)
    n = int(round((b - a) * args.fps))
    for k in range(n):
        t = a + k / args.fps
        apply(t, "orbit")
        app.update()
        capture_viewport_to_file(vp, os.path.join(args.out, "clip", f"frame_{k:05d}.png"))
        app.update()
        meta["frames"].append(dict(k=k, t=round(t, 3), carrier_x=round(carrier_x(t), 4)))
        if k % 60 == 0:
            print(f"[ws30] frame {k}/{n}", flush=True)
    for _ in range(20):
        app.update()
    json.dump(meta, open(os.path.join(args.out, "clip_meta.json"), "w"))
print("[ws30] done", flush=True)
app.close()
