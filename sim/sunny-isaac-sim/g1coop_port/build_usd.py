"""Build an animated USD of the factory from the compiled MuJoCo model and the recorded poses.
Usage: python build_usd.py REC_DIR OUT_DIR   (REC_DIR from record_mujoco.py; needs mujoco, usd-core, pillow)
Every visible geom (group < 3, alpha > 0) becomes a USD gprim under its body's Xform; body Xforms carry
one time sample per 30 fps frame, so the stage replays MuJoCo's motion (kinematic, no Isaac physics).
Meshes are written once per unique shape and referenced; materials are UsdPreviewSurface.
"""
import hashlib
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image
from pxr import Gf, Sdf, Tf, Usd, UsdGeom, UsdLux, UsdShade, Vt

import props
import shots

rec, out = Path(sys.argv[1]), Path(sys.argv[2])
(out / 'textures').mkdir(parents=True, exist_ok=True)
m = mujoco.MjModel.from_binary_path(str(rec / 'plant.mjb'))
P = np.load(rec / 'poses.npz')
xpos, xquat, rgba_t = P['xpos'], P['xquat'], P['rgba']
N = len(xpos)
G = mujoco.mjtGeom
RGB_ROLE = int(mujoco.mjtTextureRole.mjTEXROLE_RGB)

stage = Usd.Stage.CreateNew(str(out / 'factory.usd'))
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
stage.SetTimeCodesPerSecond(30)
stage.SetFramesPerSecond(30)
stage.SetStartTimeCode(0)
stage.SetEndTimeCode(N - 1)
world = UsdGeom.Xform.Define(stage, '/World')
stage.SetDefaultPrim(world.GetPrim())
stage.CreateClassPrim('/MeshLib')


def lin(c):  # MuJoCo colors are shown as-is (sRGB-ish); USD shading inputs are linear
    return Gf.Vec3f(*[float(v) ** 2.2 for v in c[:3]])


def geom_rgba(i, rgba):
    mat = m.geom_matid[i]
    if mat >= 0 and np.allclose(rgba, (.5, .5, .5, 1)):
        return m.mat_rgba[mat]
    return rgba


tex_paths = {}
def texture(t):
    if t not in tex_paths:
        w, h, c, a = m.tex_width[t], m.tex_height[t], m.tex_nchannel[t], m.tex_adr[t]
        img = m.tex_data[a:a + w * h * c].reshape(h, w, c)
        name = Tf.MakeValidIdentifier(m.tex(t).name) + '.png'
        Image.fromarray(img[..., :3] if c >= 3 else img[..., 0]).save(out / 'textures' / name)
        tex_paths[t] = './textures/' + name
    return tex_paths[t]


mats = {}
def material(rgba, rough, tex=-1, key=None):
    key = key or (tuple(np.round(rgba, 3)), round(rough, 2), tex)
    if key in mats:
        return mats[key]
    path = f'/World/Looks/M{len(mats)}'
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + '/Shader')
    sh.CreateIdAttr('UsdPreviewSurface')
    color = sh.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f)
    color.Set(lin(rgba))
    sh.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(rough)
    sh.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(0.0)
    if rgba[3] < 1:
        sh.CreateInput('opacity', Sdf.ValueTypeNames.Float).Set(float(rgba[3]))
    if tex >= 0:
        st = UsdShade.Shader.Define(stage, path + '/st')
        st.CreateIdAttr('UsdPrimvarReader_float2')
        st.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('st')
        tx = UsdShade.Shader.Define(stage, path + '/tex')
        tx.CreateIdAttr('UsdUVTexture')
        tx.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(texture(tex))
        tx.CreateInput('sourceColorSpace', Sdf.ValueTypeNames.Token).Set('sRGB')
        for wrap in ('wrapS', 'wrapT'):
            tx.CreateInput(wrap, Sdf.ValueTypeNames.Token).Set('repeat')
        tx.CreateInput('st', Sdf.ValueTypeNames.Float2).ConnectToSource(st.ConnectableAPI(), 'result')
        tx.CreateOutput('rgb', Sdf.ValueTypeNames.Float3)
        color.ConnectToSource(tx.ConnectableAPI(), 'rgb')
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), 'surface')
    mats[key] = (mat, color)
    return mats[key]


meshes = {}
def mesh_proto(d, flip_v):
    va, nv, fa, nf = m.mesh_vertadr[d], m.mesh_vertnum[d], m.mesh_faceadr[d], m.mesh_facenum[d]
    verts, faces = m.mesh_vert[va:va + nv], m.mesh_face[fa:fa + nf]
    key = hashlib.sha1(verts.tobytes() + faces.tobytes()).hexdigest()
    if key in meshes:
        return meshes[key]
    path = f'/MeshLib/mesh{len(meshes)}'
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(verts.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(nf, 3, np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.ravel().astype(np.int32)))
    mesh.CreateExtentAttr([Gf.Vec3f(*verts.min(0).tolist()), Gf.Vec3f(*verts.max(0).tolist())])
    mesh.CreateSubdivisionSchemeAttr('none')
    na = m.mesh_normaladr[d]
    normals = m.mesh_normal[na:na + m.mesh_normalnum[d]]
    mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals[m.mesh_facenormal[fa:fa + nf].ravel()].astype(np.float32)))
    mesh.SetNormalsInterpolation('faceVarying')
    ta = m.mesh_texcoordadr[d]
    if ta >= 0:
        uv = m.mesh_texcoord[ta:ta + m.mesh_texcoordnum[d]][m.mesh_facetexcoord[fa:fa + nf].ravel()].copy()
        if flip_v:
            uv[:, 1] = 1 - uv[:, 1]
        UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray,
                                                UsdGeom.Tokens.faceVarying).Set(Vt.Vec2fArray.FromNumpy(uv.astype(np.float32)))
    meshes[key] = path
    return path


def plane(path, sx, sy, rep):
    sx, sy = sx or 40., sy or 40.  # MuJoCo size 0 = infinite plane
    mesh = UsdGeom.Mesh.Define(stage, path)
    pts = [(-sx, -sy, 0), (sx, -sy, 0), (sx, sy, 0), (-sx, sy, 0)]
    mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in pts])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateExtentAttr([Gf.Vec3f(-sx, -sy, 0), Gf.Vec3f(sx, sy, 0)])
    mesh.CreateSubdivisionSchemeAttr('none')
    # Checked against MuJoCo's own render: texture spans the plane once, image top at +y.
    u, v = float(rep[0]), float(rep[1])
    st = [(0, 0), (u, 0), (u, v), (0, v)]
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray,
                                            UsdGeom.Tokens.vertex).Set([Gf.Vec2f(*s) for s in st])
    return mesh


def place(prim, pos, quat, scale=None):
    x = UsdGeom.Xformable(prim)
    x.AddTranslateOp().Set(Gf.Vec3d(*pos.tolist()))
    x.AddOrientOp().Set(Gf.Quatf(float(quat[0]), *quat[1:].tolist()))
    if scale is not None:
        x.AddScaleOp().Set(Gf.Vec3f(*scale))


changing = set(np.where(np.any(rgba_t != rgba_t[0], axis=(0, 2)))[0].tolist())
moving = np.abs(xpos - xpos[0]).max(axis=(0, 2)) + np.abs(xquat - xquat[0]).max(axis=(0, 2)) > 1e-5
body_prims = {}
counts = dict(geoms=0, skipped=0)
replaced = []
for i in range(m.ngeom):
    rgba = geom_rgba(i, m.geom_rgba[i])
    if m.geom_group[i] >= 3 or (rgba[3] <= 0 and i not in changing):
        counts['skipped'] += 1
        continue
    b = int(m.geom_bodyid[i])
    if b not in body_prims:
        bp = UsdGeom.Xform.Define(stage, f'/World/Bodies/b{b}_' + Tf.MakeValidIdentifier(m.body(b).name or 'world'))
        body_prims[b] = bp
    if m.body(b).name in props.REPLACED_BODIES or m.geom(i).name in props.REPLACED_GEOMS:
        replaced.append((b, i))  # factory-object look authored below, same body and volume
        continue
    path = str(body_prims[b].GetPath()) + f'/g{i}_' + Tf.MakeValidIdentifier(m.geom(i).name or 'geom')
    t, s = int(m.geom_type[i]), m.geom_size[i]  # int: mjtGeom == np.int32 is False inside `in`
    mat = m.geom_matid[i]
    tex = m.mat_texid[mat, RGB_ROLE] if mat >= 0 else -1
    rough = float(np.clip(1 - m.mat_shininess[mat], .25, .9)) if mat >= 0 else .6
    scale = None
    if t == G.mjGEOM_MESH:
        prim = stage.DefinePrim(path, 'Mesh')
        prim.GetReferences().AddInternalReference(mesh_proto(m.geom_dataid[i], flip_v=True))
    elif t == G.mjGEOM_PLANE:
        rep = m.mat_texrepeat[mat] if mat >= 0 else (1, 1)
        prim = plane(path, s[0], s[1], rep).GetPrim()
    elif t == G.mjGEOM_BOX:
        prim = UsdGeom.Cube.Define(stage, path).GetPrim()
        prim.GetAttribute('size').Set(2.0)
        scale = s.tolist()
    elif t in (G.mjGEOM_SPHERE, G.mjGEOM_ELLIPSOID):
        prim = UsdGeom.Sphere.Define(stage, path).GetPrim()
        prim.GetAttribute('radius').Set(1.0)
        scale = [s[0]] * 3 if t == G.mjGEOM_SPHERE else s.tolist()
    elif t in (G.mjGEOM_CYLINDER, G.mjGEOM_CAPSULE):
        g = (UsdGeom.Cylinder if t == G.mjGEOM_CYLINDER else UsdGeom.Capsule).Define(stage, path)
        g.CreateRadiusAttr(float(s[0]))
        g.CreateHeightAttr(float(2 * s[1]))
        g.CreateAxisAttr('Z')
        r, h = float(s[0]), float(s[1]) + (float(s[0]) if t == G.mjGEOM_CAPSULE else 0)
        g.CreateExtentAttr([Gf.Vec3f(-r, -r, -h), Gf.Vec3f(r, r, h)])
        prim = g.GetPrim()
    else:
        print('skip geom type', t, m.geom(i).name)
        counts['skipped'] += 1
        continue
    place(prim, m.geom_pos[i], m.geom_quat[i], scale)
    if i in changing:  # the CNC tower light: own material, color keyed per frame
        look, color = material(rgba_t[0, i] / 255., rough, key=('anim', i))
        for f in range(N):
            color.Set(lin(rgba_t[f, i] / 255.), f)
    else:
        look, _ = material(rgba, rough, tex)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(look)
    counts['geoms'] += 1

looks = props.Looks(stage)
for b in sorted({b for b, _ in replaced if b}):
    live_back = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'sofa_sofa_back') >= 0 and m.geom_contype[m.geom('sofa_sofa_back').id] > 0
    props.author_payload(stage, str(body_prims[b].GetPath()), m.body(b).name, looks, back=live_back)
for b, i in replaced:
    if not b:
        props.author_stand(stage, str(body_prims[b].GetPath()), m.geom(i).name, m.geom_pos[i], m.geom_size[i], looks)

with Sdf.ChangeBlock():
    for b, bp in body_prims.items():
        tr, ori = bp.AddTranslateOp(), bp.AddOrientOp()
        frames = range(N) if moving[b] else [None]
        for f in frames:
            k = f or 0
            q = xquat[k, b]
            tr.Set(Gf.Vec3d(*xpos[k, b].tolist()), Usd.TimeCode(f) if f is not None else Usd.TimeCode.Default())
            ori.Set(Gf.Quatf(float(q[0]), *q[1:].tolist()), Usd.TimeCode(f) if f is not None else Usd.TimeCode.Default())

# Lights: soft sky dome plus a warm key from the open front, and strip lights over the cells.
dome = UsdLux.DomeLight.Define(stage, '/World/Lights/Dome')
dome.CreateIntensityAttr(600)
dome.CreateColorAttr(Gf.Vec3f(.78, .86, 1.0))
key = UsdLux.DistantLight.Define(stage, '/World/Lights/Key')
key.CreateIntensityAttr(2500)
key.CreateAngleAttr(2.0)
key.CreateColorAttr(Gf.Vec3f(1.0, .95, .88))
UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(48, 0, -35))
for n, x in enumerate((-10, -5, 0, 5, 10)):
    for y in (-3.5, 4.0):
        rl = UsdLux.RectLight.Define(stage, f'/World/Lights/Strip{n}_{"ns"[y > 0]}')
        rl.CreateWidthAttr(3.0)
        rl.CreateHeightAttr(.4)
        rl.CreateIntensityAttr(9000)
        rl.CreateColorAttr(Gf.Vec3f(1.0, .97, .92))
        UsdGeom.Xformable(rl).AddTranslateOp().Set(Gf.Vec3d(x, y, 6.0))

shots.author_camera(stage, xpos, list(P['body_names']), N)
stage.GetRootLayer().Save()
print(counts, 'replaced', len(replaced), 'bodies', len(body_prims), 'materials', len(mats), 'meshes', len(meshes), 'textures', len(tex_paths), 'frames', N)
