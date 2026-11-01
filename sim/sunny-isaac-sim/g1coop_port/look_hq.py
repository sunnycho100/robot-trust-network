"""High-quality look layer for the sheet-bundle and T-frame close-ups and the overview (motion unchanged: looks only).
Usage: python look_hq.py USD_DIR   (writes USD_DIR/look_hq.usda, which sublayers USD_DIR/factory.usd)
Textures: ambientCG CC0 sets in ../tex_cc0 (fetched by fetch_tex.py); steel straps and wood are NVIDIA Base MDL.
"""
import sys
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade

import props

usd_dir = Path(sys.argv[1])
TEX = '../tex_cc0'
HDRI = props.ROOT + '/NVIDIA/Assets/Skies/Indoor/autoshop_01_4k.hdr'  # Poly Haven, CC0
CRATE = np.array([-8.94, -2.09, .80])  # bundle resting point (world), from the v3 recording
FRAME = np.array([3.99, -.97, .85])    # T-frame resting point (world)
G1S = ('sofa_carrier_', 'crate_left_', 'crate_right_', 'frame_left_', 'frame_right_', 'frame_rear_')

layer = Sdf.Layer.CreateNew(str(usd_dir / 'look_hq.usda'))
layer.subLayerPaths.append('./factory.usd')
stage = Usd.Stage.Open(layer)
stage.SetEditTarget(layer)
L = props.Looks(stage)


def textured(name, tex_id, tint=(1, 1, 1), scale=1., metal_map=True, rough_bias=None, world=False, color=None):
    """OmniPBR with an ambientCG colour, normal, roughness (and metalness) set; color= keeps a flat paint colour."""
    mat = UsdShade.Material.Define(stage, f'/World/Looks/HQ_{name}')
    sh = UsdShade.Shader.Define(stage, f'/World/Looks/HQ_{name}/Shader')
    sh.SetSourceAsset('OmniPBR.mdl', 'mdl')
    sh.SetSourceAssetSubIdentifier('OmniPBR', 'mdl')
    f = f'{TEX}/{tex_id}/{tex_id}_2K-JPG'
    i = sh.CreateInput
    if color is None:
        i('diffuse_texture', Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(f + '_Color.jpg'))
        i('diffuse_tint', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*tint))
    else:
        i('diffuse_color_constant', Sdf.ValueTypeNames.Color3f).Set(props.srgb(color))
    i('normalmap_texture', Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(f + '_NormalGL.jpg'))
    i('reflectionroughness_texture', Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(f + '_Roughness.jpg'))
    i('reflection_roughness_texture_influence', Sdf.ValueTypeNames.Float).Set(1.0 if rough_bias is None else .6)
    if rough_bias is not None:
        i('reflection_roughness_constant', Sdf.ValueTypeNames.Float).Set(rough_bias)
    if metal_map:
        i('metallic_texture', Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(f + '_Metalness.jpg'))
        i('metallic_texture_influence', Sdf.ValueTypeNames.Float).Set(1.0)
    i('texture_scale', Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(scale, scale))
    if world:  # Cube and plane prims without metric UVs: project in world space
        i('project_uvw', Sdf.ValueTypeNames.Bool).Set(True)
        i('world_or_object', Sdf.ValueTypeNames.Bool).Set(True)
    out = sh.CreateOutput('out', Sdf.ValueTypeNames.Token)
    for o in (mat.CreateSurfaceOutput('mdl'), mat.CreateVolumeOutput('mdl'), mat.CreateDisplacementOutput('mdl')):
        o.ConnectToSource(out)
    return mat


steel = textured('SheetSteel', 'Metal009', tint=(1.7, 1.74, 1.8), scale=1.2)  # brushed steel, lifted to bright mill finish
paint = textured('StandPaint', 'Metal027', scale=1.5, metal_map=False, rough_bias=.45, color=(.14, .33, .66))
concrete = textured('Concrete', 'Concrete048', tint=(.85, .87, .88), scale=.35, metal_map=False, world=True)
plastic = textured('RobotPlastic', 'Plastic013A', tint=(1, 1, 1), scale=3., metal_map=False, rough_bias=.35)
rubber = textured('RobotRubber', 'Rubber004', tint=(.35, .35, .37), scale=3., metal_map=False)
yellow = textured('FramePaint', 'Metal027', scale=1.5, metal_map=False, rough_bias=.4, color=(.93, .69, .06))
green = textured('BenchPaint', 'Metal027', scale=1.5, metal_map=False, rough_bias=.45, color=(.16, .30, .24))
plate = textured('BenchTop', 'Metal009', tint=(.9, .92, .95), scale=1.)
blued, wood = L.base('Metals', 'Steel_Blued'), L.base('Wood', 'Timber')


def rebind(prim, mat):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, UsdShade.Tokens.strongerThanDescendants)


def base_color(prim):
    """Diffuse colour of the UsdPreviewSurface the base layer bound to a gprim."""
    m = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
    sh = m and UsdShade.Shader(stage.GetPrimAtPath(m.GetPath().AppendChild('Shader')))
    v = sh and sh.GetInput('diffuseColor') and sh.GetInput('diffuseColor').Get()
    return np.array(v) if v is not None else None


bodies = {p.GetName().split('_', 1)[1]: p for p in stage.GetPrimAtPath('/World/Bodies').GetChildren()}
bundle = str(bodies['crate_payload'].GetPath()) + '/look'
stand = '/World/Bodies/b0_world/crate_delivery_stand_look'
tframe = str(bodies['frame_payload'].GetPath()) + '/look'
bench = '/World/Bodies/b0_world/frame_delivery_stand_look'

# 1. Materials on the bundle, stand, floor and the two G1s.
for p in stage.GetPrimAtPath(bundle).GetChildren():
    n = p.GetName()
    rebind(p, steel if n.startswith('sheet') else blued if n.startswith('strap') else wood)
for p in stage.GetPrimAtPath(stand).GetChildren():
    n = p.GetName()
    rebind(p, paint if n.startswith(('upright', 'beam')) else steel if n.startswith('sheet') else wood)
for p in stage.GetPrimAtPath('/World/Bodies/b0_world').GetChildren():
    if any(k in p.GetName() for k in ('finish_epoxy', 'finish_zone', 'finish_main_aisle', 'Depot_main_FLOOR')):
        rebind(p, concrete)
for p in stage.GetPrimAtPath(tframe).GetChildren():
    if p.GetName().startswith('tube'):
        rebind(p, yellow)
for p in stage.GetPrimAtPath(bench).GetChildren():
    n = p.GetName()
    rebind(p, plate if n in ('top', 'shelf') or n.startswith('foot') else green)
for key, b in bodies.items():
    if key.startswith(G1S):
        for g in b.GetChildren():
            c = base_color(g)
            if c is not None:
                rebind(g, plastic if c.mean() > .3 else rubber)

# 2. Models: chamfered runners and pallet boards (replacing the boxes), strap seals, welds, background props.
for p in stage.GetPrimAtPath(bundle).GetChildren():
    if p.GetName().startswith('runner'):
        e = UsdGeom.Mesh(p).GetExtentAttr().Get()
        p.SetActive(False)
        props.chamfer_box(stage, f'{bundle}/{p.GetName()}_chamfer', e[0], e[1], wood, .006)
for p in stage.GetPrimAtPath(stand).GetChildren():
    if p.GetName().startswith('pallet_'):
        e = UsdGeom.Mesh(p).GetExtentAttr().Get()
        p.SetActive(False)
        props.chamfer_box(stage, f'{stand}/{p.GetName()}_chamfer', e[0], e[1], wood, .008)
for i, x in enumerate((-.22, .22)):  # crimped seal on top of each strap join, plus the side seal already there
    props.box(stage, f'{bundle}/strap{i}_buckle', (x - .024, -.035, .117), (x + .024, .035, .123), blued)
    for j, dy in enumerate((-.018, .018)):
        props.box(stage, f'{bundle}/strap{i}_crimp{j}', (x - .026, dy - .004, .1225), (x + .026, dy + .004, .1245), blued)
# Stand welds: fillet beads where beams meet the uprights, gussets and anchored foot plates.
up = [p for p in stage.GetPrimAtPath(stand).GetChildren() if p.GetName().startswith('upright')]
for k, p in enumerate(up):
    lo, hi = UsdGeom.Mesh(p).GetExtentAttr().Get()
    cx, cy, top = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, hi[2]
    props.box(stage, f'{stand}/weld{k}', (cx - .036, cy - .036, top - .091), (cx + .036, cy + .036, top - .079), paint)
    gy = sorted((cy - .07 * np.sign(cy), cy))
    props.box(stage, f'{stand}/gusset{k}', (cx - .004, gy[0], top - .2), (cx + .004, gy[1], top - .085), paint)
    props.box(stage, f'{stand}/foot{k}', (cx - .06, cy - .06, 0), (cx + .06, cy + .06, .012), paint)
    for j, (bx, by) in enumerate(((-.04, -.04), (.04, .04))):
        props.cyl(stage, f'{stand}/anchor{k}_{j}', (cx + bx, cy + by, .016), .009, .008, blued)
# Background props behind the robots (out of focus): a second sheet pallet, a drum and a pallet stack.
bg = '/World/HQ_Background'
UsdGeom.Xform.Define(stage, bg)
for i, (x, y) in enumerate(((-11.2, -0.95), (-6.6, -1.0))):
    for j in range(3):
        props.chamfer_box(stage, f'{bg}/pallet{i}_{j}', (x - .6, y - .4, j * .14), (x + .6, y + .4, j * .14 + .12), wood, .01)
props.chamfer_box(stage, f'{bg}/sheets_on_pallet', (-11.75, -.9 - .35, .42), (-10.65, -.9 + .35, .58), steel, .004)
drum = props.cyl(stage, f'{bg}/drum', (-7.1, -1.35, .44), .29, .88, paint)
for j, z in enumerate((.25, .6)):
    props.cyl(stage, f'{bg}/drum_rib{j}', (-7.1, -1.35, z), .297, .02, paint)

# 3. Lighting: CC0 industrial HDRI on the dome, a warm key and a cool fill softbox aimed at the bundle.
dome = UsdLux.DomeLight(stage.GetPrimAtPath('/World/Lights/Dome'))
dome.CreateTextureFileAttr().Set(Sdf.AssetPath(HDRI))
dome.GetColorAttr().Set(Gf.Vec3f(1, 1, 1))
dome.GetIntensityAttr().Set(900)
for name, centre, offset, size, power, color in (
        ('HQ_Key', CRATE, (2.2, -2.6, 2.6), (1.8, 1.2), 2600, (1, .95, .88)),
        ('HQ_Fill', CRATE, (-2.6, -1.8, 1.8), (2.2, 1.4), 900, (.88, .93, 1)),
        ('HQ_FrameKey', FRAME, (2.4, 2.2, 2.6), (1.8, 1.2), 2600, (1, .95, .88)),
        ('HQ_FrameFill', FRAME, (-2.4, 1.6, 1.9), (2.2, 1.4), 900, (.88, .93, 1))):
    light = UsdLux.RectLight.Define(stage, f'/World/Lights/{name}')
    light.CreateWidthAttr(size[0])
    light.CreateHeightAttr(size[1])
    light.CreateIntensityAttr(power)
    light.CreateColorAttr(Gf.Vec3f(*color))
    pos = Gf.Vec3d(*(centre + offset))
    UsdGeom.Xformable(light).AddTransformOp().Set(Gf.Matrix4d().SetLookAt(pos, Gf.Vec3d(*centre), Gf.Vec3d(0, 0, 1)).GetInverse())
layer.Save()
print('look layer written:', layer.identifier)
