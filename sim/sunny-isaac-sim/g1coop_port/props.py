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


def chamfer_box(stage, path, lo, hi, mat, c):
    """Box with every edge bevelled by c (m): 6 faces, 12 edge strips, 8 corner triangles. Planar UVs in metres."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    c = min(c, *(0.45 * (hi - lo)))
    corners = [(sx, sy, sz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    P = {}
    for k in corners:
        base = np.where(np.array(k) > 0, hi, lo)
        for a in range(3):  # vertex of corner k on the face normal to axis a: inset along the other two axes
            v = base.copy()
            for o in range(3):
                if o != a:
                    v[o] -= k[o] * c
            P[k, a] = v
    faces = []
    for a in range(3):
        for s in (-1, 1):
            o1, o2 = [o for o in range(3) if o != a]
            ring = [(s if i == a else None) for i in range(3)]
            quad = []
            for u, v in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                k = list(ring); k[o1], k[o2] = u, v
                quad.append(P[tuple(k), a])
            n = np.zeros(3); n[a] = s
            faces.append((quad, n))
    for e in range(3):
        u, v = [o for o in range(3) if o != e]
        for su in (-1, 1):
            for sv in (-1, 1):
                ks = []
                for se in (-1, 1):
                    k = [0, 0, 0]; k[e], k[u], k[v] = se, su, sv
                    ks.append(tuple(k))
                quad = [P[ks[0], u], P[ks[1], u], P[ks[1], v], P[ks[0], v]]
                n = np.zeros(3); n[u], n[v] = su, sv
                faces.append((quad, n / np.linalg.norm(n)))
    for k in corners:
        faces.append(([P[k, 0], P[k, 1], P[k, 2]], np.array(k) / np.sqrt(3)))
    pts, nrm, uvs, counts = [], [], [], []
    for poly, n in faces:
        a = int(np.argmax(np.abs(n)))
        ax = [o for o in range(3) if o != a]
        for p in poly:
            pts.append(Gf.Vec3f(*p)); nrm.append(Gf.Vec3f(*n)); uvs.append(Gf.Vec2f(float(p[ax[0]]), float(p[ax[1]])))
        counts.append(len(poly))
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(pts)
    m.CreateFaceVertexCountsAttr(counts)
    m.CreateFaceVertexIndicesAttr(list(range(len(pts))))
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


def tote(stage, base, L, back=True):
    """1 G1 load: a blue-grey KLT tote filling the seat volume exactly, so the palms rest on its walls.
    back=True (her sofa_side carry, hands on the back block): a strapped bundle of steel flat bars fills that block
    flush with its faces. back=False (tote_side, no back): the tote is closed with a lid."""
    k = stage.DefinePrim(base + '/KLT', 'Xform')
    k.GetReferences().AddReference(KLT)
    x = UsdGeom.Xformable(k)
    x.AddScaleOp().Set(Gf.Vec3f(.28 / KLT_SIZE[0], .43 / KLT_SIZE[1], .18 / KLT_SIZE[2]))
    plastic = L.pbr('KLT_Plastic', (.36, .45, .52), .55)
    for mesh in ('FOF_Mesh_Magenta_Box', 'FOF_Mesh_Label_1', 'FOF_Mesh_Label_6_Small'):
        bind(stage.OverridePrim(f'{base}/KLT/Visuals/{mesh}'), plastic if 'Box' in mesh else L.pbr('Label', (.92, .92, .88), .7))
    if not back:
        lid = L.pbr('KLT_Lid', (.30, .38, .45), .5)  # closed lid inside the rim, flush with the top of the volume
        box(stage, base + '/lid', (-.135, -.21, .078), (.135, .21, .09), lid)
        box(stage, base + '/lid_grip', (-.04, -.12, .09 - .004), (.04, .12, .09), lid)
        return
    bar, band = L.base('Metals', 'Steel_Cast'), L.base('Metals', 'Steel_Blued')
    ys = np.linspace(-.215, .215, 14)
    for i, (y0, y1) in enumerate(zip(ys[:-1], ys[1:])):
        box(stage, f'{base}/bar{i:02d}', (-.14, y0 + .0008, -.08), (-.06, y1 - .0008, .35 - .004 * (i % 3)), bar)
    for i, z in enumerate((.12, .26)):  # two bands around the bundle, inside the outer faces
        box(stage, f'{base}/band{i}', (-.1405, -.2145, z), (-.0595, .2145, z + .025), band)


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
    for i, x in enumerate((-.22, .22)):  # two steel straps, clear of the palms at x = +-0.45
        box(stage, f'{base}/strap{i}_top', (x - .016, -.32, .117), (x + .016, .32, .12), strap)
        for s in (-1, 1):
            box(stage, f'{base}/strap{i}_side{"np"[s > 0]}', (x - .016, s * .3175 - .0015, -.068), (x + .016, s * .3175 + .0015, .12), strap)
        box(stage, f'{base}/strap{i}_bottom', (x - .016, -.32, -.068), (x + .016, .32, -.065), strap)
        box(stage, f'{base}/strap{i}_seal', (x - .022, .3155, .02), (x + .022, .32, .055), strap)


def t_frame(stage, base, L):
    """3 G1 load: welded jig frame of 80 x 144 mm tube, safety yellow, with locator plates, pins and bolts."""
    yellow, plate = L.pbr('Safety_Yellow', (.93, .69, .06), .38), L.base('Metals', 'Steel_Carbon')
    pin, bolt = L.base('Metals', 'Steel_Stainless'), L.base('Metals', 'Steel_Blued')
    z0, z1 = -.08, .064  # rectangular tube, tall enough that the side grips meet steel over the palm height
    tubes = [((-.56, .25), (.56, .33)), ((-.56, -.23), (.56, -.15)),        # crossbar front and back rails
             ((-.56, -.15), (-.48, .25)), ((.48, -.15), (.56, .25)),        # crossbar ends
             ((-.40, -.15), (-.32, .25)), ((.32, -.15), (.40, .25)),        # cross members under the side grips
             ((-.04, -.61), (.04, .25)),                                     # spine, rests on the stand
             ((-.26, -.61), (-.18, -.23)), ((.18, -.61), (.26, -.23)),      # stem sides under the rear grips
             ((-.26, -.69), (.26, -.61))]                                    # stem end
    for i, ((x0, y0), (x1, y1)) in enumerate(tubes):
        box(stage, f'{base}/tube{i}', (x0, y0, z0), (x1, y1, z1), yellow)
    for i, (x, y) in enumerate([(-.52, .29), (.52, .29), (-.36, .05), (.36, .05), (-.22, -.65), (.22, -.65), (0, -.30)]):
        box(stage, f'{base}/pad{i}', (x - .055, y - .04 if abs(y) > .2 else y - .055, z1), (x + .055, y + .04 if abs(y) > .2 else y + .055, z1 + .01), plate)
        cyl(stage, f'{base}/pin{i}', (x, y, z1 + .013), .012, .006, pin)
        for j, (bx, by) in enumerate([(-.04, -.025), (.04, -.025), (-.04, .025), (.04, .025)]):
            cyl(stage, f'{base}/bolt{i}_{j}', (x + bx, y + by, z1 + .013), .008, .006, bolt)


def low_table(stage, base, half, height, L):
    """Tote surface: a low wide steel-frame table with a plywood top and a lower shelf."""
    sx, sy = half
    frame, wood = L.pbr('Stand_Paint', (.13, .14, .15), .42, .35), L.base('Wood', 'Plywood')
    box(stage, base + '/top', (-sx, -sy, height - .025), (sx, sy, height), wood)
    box(stage, base + '/shelf', (-sx + .02, -sy + .02, .10), (sx - .02, sy - .02, .118), wood)
    for i, (ax, ay) in enumerate([(-1, -1), (1, -1), (-1, 1), (1, 1)]):
        cx, cy = ax * (sx - .025), ay * (sy - .025)
        box(stage, f'{base}/leg{i}', (cx - .02, cy - .02, 0), (cx + .02, cy + .02, height - .025), frame)


def sheet_rack(stage, base, half, height, L):
    """Steel-sheet surface: a pallet of sheets on a low rack; the stack top is the placement height."""
    sx, sy = half
    frame, wood = L.pbr('Rack_Blue', (.12, .27, .45), .4, .3), L.base('Wood', 'Timber')
    sheet = L.pbr('Galvanized_Sheet', (.70, .72, .74), .36, .65)
    deck = height - .25  # rack beams, then a 0.12 m pallet, then 0.13 m of sheets
    for i, (ax, ay) in enumerate([(-1, -1), (1, -1), (-1, 1), (1, 1)]):
        cx, cy = ax * (sx - .03), ay * (sy - .03)
        box(stage, f'{base}/upright{i}', (cx - .03, cy - .03, 0), (cx + .03, cy + .03, deck), frame)
    for s_, y in enumerate((-sy + .03, sy - .03)):
        box(stage, f'{base}/beam{s_}', (-sx, y - .035, deck - .08), (sx, y + .035, deck), frame)
    for i, x in enumerate(np.linspace(-sx + .06, sx - .06, 3)):  # pallet: blocks and deck boards
        box(stage, f'{base}/pallet_block{i}', (x - .05, -sy + .02, deck), (x + .05, sy - .02, deck + .08), wood)
    for i, y in enumerate(np.linspace(-sy + .06, sy - .06, 6)):
        box(stage, f'{base}/pallet_board{i}', (-sx + .01, y - .045, deck + .08), (sx - .01, y + .045, deck + .12), wood)
    rng = np.random.default_rng(3)
    z = deck + .12
    for i in range(11):
        dx, dy = rng.uniform(-.004, .004, 2)
        top = height if i == 10 else z + .0104
        box(stage, f'{base}/sheet{i:02d}', (-sx + .012 + dx, -sy + .012 + dy, z), (sx - .012 + dx, sy - .012 + dy, top), sheet)
        z = top + .0014


def workbench(stage, base, half, height, L):
    """T-frame surface: a wide welded workbench with a steel top plate, apron, lower shelf and a vise."""
    sx, sy = half
    frame, top = L.pbr('Bench_Green', (.16, .3, .24), .4, .3), L.base('Metals', 'Steel_Carbon')
    box(stage, base + '/top', (-sx, -sy, height - .02), (sx, sy, height), top)
    box(stage, base + '/apron', (-sx + .02, -sy + .02, height - .10), (sx - .02, sy - .02, height - .02), frame)
    box(stage, base + '/shelf', (-sx + .04, -sy + .04, .16), (sx - .04, sy - .04, .175), top)
    for i, (ax, ay) in enumerate([(-1, -1), (1, -1), (-1, 1), (1, 1)]):
        cx, cy = ax * (sx - .04), ay * (sy - .04)
        box(stage, f'{base}/leg{i}', (cx - .03, cy - .03, 0), (cx + .03, cy + .03, height - .10), frame)
        box(stage, f'{base}/foot{i}', (cx - .045, cy - .045, 0), (cx + .045, cy + .045, .01), top)


def author_payload(stage, body_path, body_name, L, back=True):
    base = body_path + '/look'
    UsdGeom.Xform.Define(stage, base)
    if body_name == 'sofa_payload':
        tote(stage, base, L, back)
    else:
        {'crate_payload': sheet_bundle, 'frame_payload': t_frame}[body_name](stage, base, L)


def author_stand(stage, parent_path, name, pos, size, L):
    """pos and size are the MuJoCo box geom's center and half sizes (world frame, no rotation)."""
    base = f'{parent_path}/{name}_look'
    UsdGeom.Xform.Define(stage, base).AddTranslateOp().Set(Gf.Vec3d(float(pos[0]), float(pos[1]), 0.))
    look = {'sofa_delivery_stand': low_table, 'crate_delivery_stand': sheet_rack, 'frame_delivery_stand': workbench}[name]
    look(stage, base, (float(size[0]), float(size[1])), float(2 * size[2]), L)
