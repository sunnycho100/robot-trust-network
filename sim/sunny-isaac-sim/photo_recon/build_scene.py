"""Build the single-photo reconstruction in Isaac Sim 6.2 and render it from the photo camera, then a slow orbit.
Usage (Colab, via carda/colab_run.sh): build_scene.py scene.json perception.json OUTDIR [--orbit 9] [--fps 30] [--still]
scene.json items: kind usd (asset placed by its bbox bottom centre), box (centre + size), rail (tiled GuardRail_A01 from a to b).
"""
import argparse
import json
import math
import os
import re

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("scene")
ap.add_argument("perc")
ap.add_argument("out")
ap.add_argument("--orbit", type=float, default=9.0)
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--still", action="store_true", help="photo view only")
args = ap.parse_args()
W, H = 1200, 800

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "width": W, "height": H})

import omni.usd  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402
from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade  # noqa: E402

ROOT = get_assets_root_path()                    # .../Assets/Isaac/6.2
NV = ROOT + "/NVIDIA"                            # SimReady DigitalTwin assets
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.Xform.Define(stage, "/World")
scene = json.load(open(args.scene))
perc = json.load(open(args.perc))
LOOKS = {}


def material(rgb, rough=0.6, emit=None):
    key = (tuple(rgb), rough, tuple(emit) if emit else None)
    if key not in LOOKS:
        p = f"/World/Looks/M{len(LOOKS)}"
        m = UsdShade.Material.Define(stage, p)
        sh = UsdShade.Shader.Define(stage, p + "/S")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
        if emit:
            sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emit))
        m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        LOOKS[key] = m
    return LOOKS[key]


def place(prim, pos, yaw=0.0, scale=1.0):
    x = UsdGeom.Xformable(prim)
    x.AddTranslateOp().Set(Gf.Vec3d(*pos))
    x.AddRotateZOp().Set(float(yaw))
    if scale != 1.0:
        x.AddScaleOp().Set(Gf.Vec3f(scale, scale, scale))


def box(path, center, size, yaw, rgb, rough=0.6, emit=None, tilt=(0.0, 0.0)):
    g = UsdGeom.Cube.Define(stage, path)
    g.CreateSizeAttr(1.0)
    x = UsdGeom.Xformable(g)
    x.AddTranslateOp().Set(Gf.Vec3d(*center))
    x.AddRotateXYZOp().Set(Gf.Vec3f(float(tilt[0]), float(tilt[1]), float(yaw)))
    x.AddScaleOp().Set(Gf.Vec3f(*size))
    UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(material(rgb, rough, emit))


bbc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])


def usd(path, url, pos, yaw=0.0, scale=1.0):
    """Reference an asset so its bbox bottom centre sits at pos; returns the asset size in metres."""
    url = url.replace("{ROOT}", ROOT).replace("{NV}", NV)
    par = UsdGeom.Xform.Define(stage, path).GetPrim()
    ch = stage.DefinePrim(path + "/a", "Xform")   # our offset; the asset root keeps its own xform ops
    ref = stage.DefinePrim(path + "/a/r", "Xform")
    ref.GetReferences().AddReference(url)
    r = bbc.ComputeRelativeBound(ref, par).ComputeAlignedRange()
    lo, hi = np.array(r.GetMin()), np.array(r.GetMax())
    assert not r.IsEmpty(), f"asset did not load: {url}"
    s = scale * (0.01 if (hi - lo).max() > 60 else 1.0)   # ponytail: cm-authored assets show up as > 60 units
    UsdGeom.Xformable(ch).AddScaleOp().Set(Gf.Vec3f(s, s, s))
    UsdGeom.Xformable(ch).AddTranslateOp().Set(Gf.Vec3d(-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]))
    place(par, pos, yaw)
    size = ((hi - lo) * s).round(2).tolist()
    print(f"[asset] {path} {url.rsplit('/', 1)[-1]} size {size}", flush=True)
    return size


RAIL = "{NV}/Assets/DigitalTwin/Assets/Warehouse/Safety/Barriers/GuardRail_A/GuardRail_A01_01.usd"
for i, it in enumerate(scene["items"]):
    p = "/World/S/" + re.sub(r"\W", "_", it["name"])   # USD prim names: letters, digits, underscore
    if it["kind"] == "usd":
        usd(p, it["url"], it["pos"], it.get("yaw", 0.0), it.get("scale", 1.0))
    elif it["kind"] == "box":
        box(p, it["center"], it["size"], it.get("yaw", 0.0), it["rgb"], it.get("rough", 0.6), it.get("emit"), it.get("tilt", (0.0, 0.0)))
    elif it["kind"] == "rail":   # tile the stock guardrail section along a to b
        a, b = np.array(it["a"], float), np.array(it["b"], float)
        L, yaw = np.linalg.norm(b - a), math.degrees(math.atan2(*(b - a)[::-1]))
        sz = usd(p + "_0", RAIL, [*a, 0], yaw)
        if sz[1] > sz[0]:   # section is authored along y
            yaw += 90
            UsdGeom.Xformable(stage.GetPrimAtPath(p + "_0")).GetOrderedXformOps()[1].Set(float(yaw))
        seg = max(sz[:2])
        n = max(1, int(round(L / seg)))
        for k in range(n):
            c = a + (b - a) * (k + 0.5) / n
            if k == 0:
                UsdGeom.Xformable(stage.GetPrimAtPath(p + "_0")).GetOrderedXformOps()[0].Set(Gf.Vec3d(*c, 0))
            else:
                usd(p + f"_{k}", RAIL, [*c, 0], yaw)

# ---------------------------------------------------------------- light
for lt in scene.get("lights", []):
    if lt["kind"] == "dome":
        UsdLux.DomeLight.Define(stage, "/World/Dome").CreateIntensityAttr(lt["intensity"])
    elif lt["kind"] == "sphere":
        for j, q in enumerate(lt["at"]):
            s = UsdLux.SphereLight.Define(stage, f"/World/L{j}")
            s.CreateIntensityAttr(lt["intensity"])
            s.CreateRadiusAttr(lt.get("radius", 0.3))
            UsdGeom.Xformable(s).AddTranslateOp().Set(Gf.Vec3d(*q))
    elif lt["kind"] == "rect":   # light coming through an opening; aims along -Z, so rotate
        for j, q in enumerate(lt["at"]):
            r = UsdLux.RectLight.Define(stage, f"/World/R{j}")
            r.CreateIntensityAttr(lt["intensity"])
            r.CreateWidthAttr(lt["w"])
            r.CreateHeightAttr(lt["h"])
            x = UsdGeom.Xformable(r)
            x.AddTranslateOp().Set(Gf.Vec3d(*q["pos"]))
            x.AddRotateXYZOp().Set(Gf.Vec3f(*q["rot"]))

# ---------------------------------------------------------------- camera (EXIF intrinsics, pose from the floor fit)
cam = UsdGeom.Camera.Define(stage, "/World/Cam")
cam.CreateFocalLengthAttr(17.0)
cam.CreateHorizontalApertureAttr(35.9)
cam.CreateVerticalApertureAttr(35.9 * H / W)
cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 500))
cam_op = UsdGeom.Xformable(cam).MakeMatrixXform()
T0 = np.array(perc["usd_cam_to_world"])
T0[2, 3] = scene.get("cam_height", T0[2, 3])   # metric rescale K from make_scene.py
cam_op.Set(Gf.Matrix4d(*T0.T.reshape(-1).tolist()))   # Gf is row-vector: transpose
vp = get_active_viewport()
vp.camera_path = "/World/Cam"

os.makedirs(args.out, exist_ok=True)
for i in range(600):   # let references and textures stream in
    app.update()
    msg, loaded, total = omni.usd.get_context().get_stage_loading_status()
    if i > 80 and total == 0:
        break
for _ in range(60):
    app.update()
print("[render] stage loaded after", i, "updates", flush=True)
capture_viewport_to_file(vp, os.path.join(args.out, "photo_view.png"))
for _ in range(10):
    app.update()

if not args.still:
    eye0, fwd = T0[:3, 3], -T0[:3, 2]
    tgt = np.array(scene["orbit_target"], float)
    rad = eye0[:2] - tgt[:2]
    N = int(args.orbit * args.fps)
    for k in range(N):
        s = 0.5 - 0.5 * math.cos(math.pi * k / (N - 1))   # ease in and out
        a = math.radians(scene.get("orbit_deg", 70) * s)
        xy = tgt[:2] + np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]) @ rad * (1 + 0.6 * s)
        eye = np.array([*xy, eye0[2] + scene.get("orbit_rise", 6.0) * s])
        look = (eye0 + fwd * np.linalg.norm(tgt - eye0)) * (1 - s) + tgt * s   # start on the photo's own line of sight
        m = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*look), Gf.Vec3d(0, 0, 1)).GetInverse()
        cam_op.Set(m)
        for _ in range(2):
            app.update()
        capture_viewport_to_file(vp, os.path.join(args.out, f"orbit_{k:04d}.png"))
        if k % 60 == 0:
            print(f"[render] orbit {k}/{N}", flush=True)
    for _ in range(20):
        app.update()
print("[render] done", flush=True)
app.close()
