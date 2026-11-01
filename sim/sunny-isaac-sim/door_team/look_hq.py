"""High-quality look for the final door team render (door_team.py --hq): textured materials, real glass, an HDRI
factory dome and two soft area lights. Looks only: the motion and the scene layout are unchanged. Textures are
ambientCG CC0 PBR sets fetched by fetch_tex.py (container path /robot/door_team/tex_cc0, or set DOOR_TEX).
"""
import os

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade

ROOT = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.2"
HDRI = ROOT + "/NVIDIA/Assets/Skies/Indoor/autoshop_01_4k.hdr"      # Poly Haven, CC0
TEX = os.environ.get("DOOR_TEX", "/robot/door_team/tex_cc0")   # fetch_tex.py puts the sets here


def srgb(c):
    return Gf.Vec3f(*[v ** 2.2 for v in c])


def _mdl(stage, name, asset, sub):
    mat = UsdShade.Material.Define(stage, f"/World/Looks/HQ_{name}")
    sh = UsdShade.Shader.Define(stage, f"/World/Looks/HQ_{name}/Shader")
    sh.SetSourceAsset(asset, "mdl")
    sh.SetSourceAssetSubIdentifier(sub, "mdl")
    out = sh.CreateOutput("out", Sdf.ValueTypeNames.Token)
    for o in (mat.CreateSurfaceOutput("mdl"), mat.CreateVolumeOutput("mdl"), mat.CreateDisplacementOutput("mdl")):
        o.ConnectToSource(out)
    return mat, sh


def textured(stage, name, tex_id, tint=(1, 1, 1), scale=1.0, metal_map=True, rough=None, color=None, world=True,
             metallic=None):
    """OmniPBR with an ambientCG colour, normal, roughness (and metalness) set; color= keeps a flat paint colour."""
    mat, sh = _mdl(stage, name, "OmniPBR.mdl", "OmniPBR")
    f = f"{TEX}/{tex_id}/{tex_id}_2K-JPG"
    i = sh.CreateInput
    if color is None:
        i("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(f + "_Color.jpg"))
        i("diffuse_tint", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*tint))
    else:
        i("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(srgb(color))
    i("normalmap_texture", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(f + "_NormalGL.jpg"))
    i("reflectionroughness_texture", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(f + "_Roughness.jpg"))
    i("reflection_roughness_texture_influence", Sdf.ValueTypeNames.Float).Set(1.0 if rough is None else 0.5)
    if rough is not None:
        i("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(rough)
    if metal_map:
        i("metallic_texture", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(f + "_Metalness.jpg"))
        i("metallic_texture_influence", Sdf.ValueTypeNames.Float).Set(1.0)
    if metallic is not None:
        i("metallic_constant", Sdf.ValueTypeNames.Float).Set(metallic)
    i("texture_scale", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(scale, scale))
    if world:                                                     # boxes and cylinders without metric UVs
        i("project_uvw", Sdf.ValueTypeNames.Bool).Set(True)
        i("world_or_object", Sdf.ValueTypeNames.Bool).Set(True)
    return mat


def glass(stage):
    """Real glass: OmniGlass with a window-glass IOR, thin walled (the pane is a single sheet)."""
    mat, sh = _mdl(stage, "Glass", "OmniGlass.mdl", "OmniGlass")
    sh.CreateInput("glass_ior", Sdf.ValueTypeNames.Float).Set(1.52)
    sh.CreateInput("glass_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.90, 0.97, 0.96))
    sh.CreateInput("frosting_roughness", Sdf.ValueTypeNames.Float).Set(0.0)
    sh.CreateInput("thin_walled", Sdf.ValueTypeNames.Bool).Set(True)
    return mat


def bind(stage, path, mat, descend=True):
    p = stage.GetPrimAtPath(path)
    if not p.IsValid():
        return 0
    n = 0
    for q in (Usd.PrimRange(p) if descend else [p]):
        if q.IsA(UsdGeom.Gprim):
            UsdShade.MaterialBindingAPI.Apply(q).Bind(mat, UsdShade.Tokens.strongerThanDescendants)
            n += 1
    return n


def apply(stage, arms, focus_points):
    """arms: the rail arm names; focus_points: world points the two soft area lights aim at."""
    steel = textured(stage, "BrushedSteel", "Metal009", tint=(1.6, 1.64, 1.7), scale=2.0)
    dark_steel = textured(stage, "DarkSteel", "Metal038", tint=(0.55, 0.56, 0.6), scale=2.0)
    paint_door = textured(stage, "DoorPaint", "Metal027", scale=1.5, metal_map=False, rough=0.32, color=(0.05, 0.055, 0.065),
                          metallic=0.35)
    paint_yellow = textured(stage, "YellowPaint", "Metal027", scale=1.5, metal_map=False, rough=0.45, color=(0.95, 0.72, 0.05))
    concrete = textured(stage, "EpoxyConcrete", "Concrete048", tint=(0.82, 0.84, 0.85), scale=0.35, metal_map=False)
    rubber = textured(stage, "Rubber", "Rubber004", tint=(0.9, 0.45, 0.12), scale=3.0, metal_map=False)
    black_rubber = textured(stage, "BlackRubber", "Rubber004", tint=(0.25, 0.25, 0.27), scale=3.0, metal_map=False)
    plastic = textured(stage, "OrangePlastic", "Plastic013A", tint=(1.0, 0.55, 0.15), scale=3.0, metal_map=False, rough=0.35)
    window = glass(stage)
    n = 0
    for a in arms:                                                # rails, carriages, pedestals and grippers
        n += bind(stage, f"/World/Rail_{a}", steel)
        for part in ("plate", "pedestal"):
            n += bind(stage, f"/World/{a}/{part}", steel)
    for p in Usd.PrimRange(stage.GetPrimAtPath("/World")):
        if p.GetName() == "gripper":
            n += bind(stage, str(p.GetPath()), dark_steel)
    n += bind(stage, "/World/WS2", steel)
    n += bind(stage, "/World/WS2/locator0", paint_yellow) + bind(stage, "/World/WS2/locator1", paint_yellow)
    n += bind(stage, "/World/Station/WS30/Carrier/door", paint_door)
    n += bind(stage, "/World/Glass", window)
    n += bind(stage, "/World/Lifter/cup0", rubber) + bind(stage, "/World/Lifter/cup1", rubber)
    n += bind(stage, "/World/Lifter/handle", black_rubber)
    n += bind(stage, "/World/Lifter/bar", steel) + bind(stage, "/World/Lifter/post0", steel) + bind(stage, "/World/Lifter/post1", steel)
    n += bind(stage, "/World/Driver/body", plastic) + bind(stage, "/World/Driver/grip", black_rubber)
    n += bind(stage, "/World/Driver/battery", black_rubber)
    n += bind(stage, "/World/Station/WS30/Hall/floor", concrete)
    print(f"[hq] bound {n} gprims", flush=True)
    # factory HDRI on the station's dome, plus two soft area lights over the trolley and WS2
    dome = stage.GetPrimAtPath("/World/Station/WS30/Lights/dome")
    if dome.IsValid():
        d = UsdLux.DomeLight(dome)
        d.CreateTextureFileAttr().Set(Sdf.AssetPath(HDRI))
        d.GetColorAttr().Set(Gf.Vec3f(1, 1, 1))
    for k, (p, size, power, col) in enumerate(focus_points):
        light = UsdLux.RectLight.Define(stage, f"/World/HQ_Soft{k}")
        light.CreateWidthAttr(size[0])
        light.CreateHeightAttr(size[1])
        light.CreateIntensityAttr(power)
        light.CreateColorAttr(Gf.Vec3f(*col))
        eye = Gf.Vec3d(p[0] + 1.2, p[1] + 2.2, p[2] + 2.6)
        UsdGeom.Xformable(light).AddTransformOp().Set(Gf.Matrix4d().SetLookAt(eye, Gf.Vec3d(*p), Gf.Vec3d(0, 0, 1)).GetInverse())
