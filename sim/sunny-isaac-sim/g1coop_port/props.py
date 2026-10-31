"""Factory-object looks for the three G1 payloads and their delivery stands (visual only).
Each replaces the plain MuJoCo boxes inside the same outer volume, under the same body prim, so the recorded
motion and the hand contacts are unchanged. Materials are MDL: NVIDIA Base materials from the Isaac asset
root, plus OmniPBR for paint, plastic and the galvanized sheets. The tote is Isaac's stock small KLT, scaled to the seat volume.
"""
import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdShade

ROOT = 'https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.2'
KLT = ROOT + '/Isaac/Props/KLT_Bin/small_KLT_visual.usd'
KLT_SIZE = (0.1978, 0.2966, 0.1464)  # measured in Isaac Sim 6.2: centered, long axis +y, Z up
REPLACED_BODIES = ('sofa_payload', 'crate_payload', 'frame_payload')
REPLACED_GEOMS = ('sofa_delivery_stand', 'crate_delivery_stand', 'frame_delivery_stand')


def srgb(c):
    return Gf.Vec3f(*[v ** 2.2 for v in c])


class Looks:
    def __init__(self, stage):
        self.stage, self.cache = stage, {}

    def _mat(self, name):
        return UsdShade.Material.Define(self.stage, f'/World/Looks/{name}')

    def base(self, group, name):
        """NVIDIA Base library material, e.g. base('Metals', 'Steel_Carbon')."""
        if name not in self.cache:
            mat = self._mat(name)
            sh = UsdShade.Shader.Define(self.stage, f'/World/Looks/{name}/Shader')
            sh.SetSourceAsset(f'{ROOT}/NVIDIA/Materials/Base/{group}/{name}.mdl', 'mdl')
            sh.SetSourceAssetSubIdentifier(name, 'mdl')
            out = sh.CreateOutput('out', Sdf.ValueTypeNames.Token)
            for o in (mat.CreateSurfaceOutput('mdl'), mat.CreateVolumeOutput('mdl'), mat.CreateDisplacementOutput('mdl')):
                o.ConnectToSource(out)
            self.cache[name] = mat
        return self.cache[name]

    def pbr(self, name, color, rough, metal=0.0):
        """OmniPBR: painted metal and plastic."""
        if name not in self.cache:
            mat = self._mat(name)
            sh = UsdShade.Shader.Define(self.stage, f'/World/Looks/{name}/Shader')
            sh.SetSourceAsset('OmniPBR.mdl', 'mdl')
            sh.SetSourceAssetSubIdentifier('OmniPBR', 'mdl')
            sh.CreateInput('diffuse_color_constant', Sdf.ValueTypeNames.Color3f).Set(srgb(color))
            sh.CreateInput('reflection_roughness_constant', Sdf.ValueTypeNames.Float).Set(rough)
            sh.CreateInput('metallic_constant', Sdf.ValueTypeNames.Float).Set(metal)
            out = sh.CreateOutput('out', Sdf.ValueTypeNames.Token)
            for o in (mat.CreateSurfaceOutput('mdl'), mat.CreateVolumeOutput('mdl'), mat.CreateDisplacementOutput('mdl')):
                o.ConnectToSource(out)
            self.cache[name] = mat
        return self.cache[name]


def bind(prim, mat):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, UsdShade.Tokens.strongerThanDescendants)


# Unit cube faces: normal axis, sign, and the two in-plane axes used for UVs.
FACES = [(0, 1, 1, 2), (0, -1, 1, 2), (1, 1, 0, 2), (1, -1, 0, 2), (2, 1, 0, 1), (2, -1, 0, 1)]


def box(stage, path, lo, hi, mat):
    """Axis-aligned box mesh from corner lo to corner hi, UVs in metres so MDL textures keep real scale."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    pts, uvs, nrm, idx = [], [], [], []
    for ax, sgn, a, b in FACES:
        n = np.zeros(3)
        n[ax] = sgn
        quad = [(0, 0), (1, 0), (1, 1), (0, 1)] if sgn > 0 else [(0, 0), (0, 1), (1, 1), (1, 0)]
        for ua, ub in quad:
            p = np.empty(3)
            p[ax] = hi[ax] if sgn > 0 else lo[ax]
            p[a] = hi[a] if ua else lo[a]
            p[b] = hi[b] if ub else lo[b]
            idx.append(len(pts))
            pts.append(Gf.Vec3f(*p))
            uvs.append(Gf.Vec2f(float(p[a]), float(p[b])))
            nrm.append(Gf.Vec3f(*n))
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(pts)
    m.CreateFaceVertexCountsAttr([4] * 6)
    m.CreateFaceVertexIndicesAttr(idx)
    m.CreateNormalsAttr(nrm)
    m.SetNormalsInterpolation('faceVarying')
    m.CreateSubdivisionSchemeAttr('none')
    m.CreateDoubleSidedAttr(True)
    m.CreateExtentAttr([Gf.Vec3f(*lo), Gf.Vec3f(*hi)])
    UsdGeom.PrimvarsAPI(m).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying).Set(uvs)
    bind(m.GetPrim(), mat)
    return m


def cyl(stage, path, center, r, h, mat, axis='Z', sides=None):
    c = UsdGeom.Cylinder.Define(stage, path)
    c.CreateRadiusAttr(r)
    c.CreateHeightAttr(h)
    c.CreateAxisAttr(axis)
    e = [r, r, r]
    e['XYZ'.index(axis)] = h / 2
    c.CreateExtentAttr([Gf.Vec3f(*[-v for v in e]), Gf.Vec3f(*e)])
    c.AddTranslateOp().Set(Gf.Vec3d(*center))
    bind(c.GetPrim(), mat)
    return c


def tote(stage, base, L):
    """1 G1 load: blue-grey KLT tote in the seat volume, a slotted rack of steel brackets in the back volume."""
    k = stage.DefinePrim(base + '/KLT', 'Xform')
    k.GetReferences().AddReference(KLT)
    x = UsdGeom.Xformable(k)
    x.AddScaleOp().Set(Gf.Vec3f(.28 / KLT_SIZE[0], .43 / KLT_SIZE[1], .18 / KLT_SIZE[2]))
    plastic = L.pbr('KLT_Plastic', (.36, .45, .52), .55)
    for mesh in ('FOF_Mesh_Magenta_Box', 'FOF_Mesh_Label_1', 'FOF_Mesh_Label_6_Small'):
        bind(stage.OverridePrim(f'{base}/KLT/Visuals/{mesh}'), plastic if 'Box' in mesh else L.pbr('Label', (.92, .92, .88), .7))
    paint, steel, part = L.pbr('Rack_Paint', (.16, .2, .24), .45, .2), L.base('Metals', 'Steel_Carbon'), L.base('Metals', 'Steel_Cast')
    for s in (-1, 1):  # rack end plates: the grip faces at y = +-0.215 above the rim
        box(stage, f'{base}/rack_end{"np"[s > 0]}', (-.14, s * .215 - (s > 0) * .012, .08), (-.06, s * .215 + (s < 0) * .012, .35), paint)
    box(stage, base + '/rack_rail_top', (-.11, -.203, .33), (-.09, .203, .35), steel)
    box(stage, base + '/rack_rail_low', (-.11, -.203, -.06), (-.08, .203, -.04), steel)
    for i, y in enumerate(np.linspace(-.16, .16, 6)):  # upright brackets in the slots
        box(stage, f'{base}/bracket{i}', (-.125, y - .006, -.08), (-.065, y + .006, .30), part)
        box(stage, f'{base}/bracket{i}_flange', (-.125, y - .02, .28), (-.065, y + .02, .292), part)
    bush = L.base('Metals', 'Steel_Stainless')
    for i, (px, py) in enumerate([(.02, -.12), (.08, -.06), (.03, .02), (.09, .09), (.02, .14)]):
        cyl(stage, f'{base}/bushing{i}', (px, py, -.06), .028, .05, bush)


def sheet_bundle(stage, base, L):
    """2 G1 load: strapped bundle of steel sheets on three wooden runners (1.10 x 0.64 x 0.24 m)."""
    wood, sheet, strap = L.base('Wood', 'Timber'), L.pbr('Galvanized_Sheet', (.70, .72, .74), .36, .65), L.base('Metals', 'Steel_Blued')
    for i, x in enumerate((-.30, 0., .30)):  # runners where the hands and the stand support the bundle
        box(stage, f'{base}/runner{i}', (x - .045, -.32, -.12), (x + .045, .32, -.065), wood)
    rng = np.random.default_rng(7)
    z, t, gap = -.065, .0104, .0018
    for i in range(15):
        dx, dy = rng.uniform(-.004, .004, 2)
        box(stage, f'{base}/sheet{i:02d}', (-.546 + dx, -.316 + dy, z), (.546 + dx, .316 + dy, z + t), sheet)
        z += t + gap
    for i, x in enumerate((-.42, .42)):  # two steel straps over the top, down the sides, under the stack
        box(stage, f'{base}/strap{i}_top', (x - .016, -.32, .117), (x + .016, .32, .12), strap)
        for s in (-1, 1):
            box(stage, f'{base}/strap{i}_side{"np"[s > 0]}', (x - .016, s * .3175 - .0015, -.068), (x + .016, s * .3175 + .0015, .12), strap)
        box(stage, f'{base}/strap{i}_bottom', (x - .016, -.32, -.068), (x + .016, .32, -.065), strap)
        box(stage, f'{base}/strap{i}_seal', (x - .022, .3155, .02), (x + .022, .32, .055), strap)


def t_frame(stage, base, L):
    """3 G1 load: welded jig frame of 80 mm square tube, safety yellow, with locator posts on base plates."""
    yellow, plate = L.pbr('Safety_Yellow', (.93, .69, .06), .38), L.base('Metals', 'Steel_Carbon')
    pin, bolt = L.base('Metals', 'Steel_Stainless'), L.base('Metals', 'Steel_Blued')
    z0, z1 = -.08, 0.
    tubes = [((-.56, .25), (.56, .33)), ((-.56, -.23), (.56, -.15)),        # crossbar front and back rails
             ((-.56, -.15), (-.48, .25)), ((.48, -.15), (.56, .25)),        # crossbar ends
             ((-.40, -.15), (-.32, .25)), ((.32, -.15), (.40, .25)),        # cross members under the side grips
             ((-.04, -.61), (.04, .25)),                                     # spine, rests on the stand
             ((-.26, -.61), (-.18, -.23)), ((.18, -.61), (.26, -.23)),      # stem sides under the rear grips
             ((-.26, -.69), (.26, -.61))]                                    # stem end
    for i, ((x0, y0), (x1, y1)) in enumerate(tubes):
        box(stage, f'{base}/tube{i}', (x0, y0, z0), (x1, y1, z1), yellow)
    for i, (x, y) in enumerate([(-.52, .29), (.52, .29), (-.36, .05), (.36, .05), (-.22, -.65), (.22, -.65), (0, -.30)]):
        box(stage, f'{base}/pad{i}', (x - .055, y - .04 if abs(y) > .2 else y - .055, z1), (x + .055, y + .04 if abs(y) > .2 else y + .055, z1 + .012), plate)
        box(stage, f'{base}/post{i}', (x - .025, y - .025, z1 + .012), (x + .025, y + .025, .068), plate)
        cyl(stage, f'{base}/pin{i}', (x, y, .074), .012, .012, pin)
        for j, (bx, by) in enumerate([(-.04, -.025), (.04, -.025), (-.04, .025), (.04, .025)]):
            cyl(stage, f'{base}/bolt{i}_{j}', (x + bx, y + by, z1 + .015), .008, .006, bolt)


def workstand(stage, base, half, height, L):
    """Delivery stand: welded dark-steel workstand in the old stand's box (half x, half y, full height)."""
    sx, sy = half
    dark, top = L.pbr('Stand_Paint', (.13, .14, .15), .42, .35), L.base('Metals', 'Steel_Carbon')
    box(stage, base + '/top', (-sx, -sy, height - .016), (sx, sy, height), top)
    leg, foot = .035, .06
    for i, (ax, ay) in enumerate([(-1, -1), (1, -1), (-1, 1), (1, 1)]):
        cx, cy = ax * (sx - foot / 2), ay * (sy - foot / 2)
        box(stage, f'{base}/foot{i}', (cx - foot / 2, cy - foot / 2, 0), (cx + foot / 2, cy + foot / 2, .008), top)
        box(stage, f'{base}/leg{i}', (cx - leg / 2, cy - leg / 2, .008), (cx + leg / 2, cy + leg / 2, height - .016), dark)
    for k, z in enumerate((.12, height * .55)):
        for s in (-1, 1):
            box(stage, f'{base}/rail_x{k}{"np"[s > 0]}', (-sx + .01, s * (sy - foot / 2) - .012, z), (sx - .01, s * (sy - foot / 2) + .012, z + .03), dark)
            box(stage, f'{base}/rail_y{k}{"np"[s > 0]}', (s * (sx - foot / 2) - .012, -sy + .01, z), (s * (sx - foot / 2) + .012, sy - .01, z + .03), dark)


def author_payload(stage, body_path, body_name, L):
    base = body_path + '/look'
    UsdGeom.Xform.Define(stage, base)
    {'sofa_payload': tote, 'crate_payload': sheet_bundle, 'frame_payload': t_frame}[body_name](stage, base, L)


def author_stand(stage, parent_path, name, pos, size, L):
    """pos and size are the MuJoCo box geom's center and half sizes (world frame, no rotation)."""
    base = f'{parent_path}/{name}_look'
    UsdGeom.Xform.Define(stage, base).AddTranslateOp().Set(Gf.Vec3d(float(pos[0]), float(pos[1]), 0.))
    workstand(stage, base, (float(size[0]), float(size[1])), float(2 * size[2]), L)
