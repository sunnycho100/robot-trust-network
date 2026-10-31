"""CooHOI's pair carry in the levelled ETH3D delivery_area: pick the task origin and heading on the free floor and
write da_world.usda, the scan under one transform (floor at z = 0, task origin at (0, 0), longest free stretch on x).

    python da_coohoi_scene.py MESH.ply IMAGES.txt OUT_DIR [--splat SPLAT.usdz] [--floor_plane Z]
    (open3d venv: open3d, scipy, usd-core; needs level_scene.py and score_robot.py next to it)

MESH is the PatchMatch Poisson mesh in the ETH3D frame. It becomes OUT_DIR/da_collider.usdc, a triangle-mesh collider,
visible with its vertex colours, or invisible when --splat is given (a NuRec USDZ exported with --no-transform, so in
the same frame). --floor_plane Z adds an invisible 12 x 4 m collision slab with its top at z = Z under the task area.
The pair layout needs about 11 m along x (humanoids start at x = -5 and +5). Prints the play_scene.py camera arguments:
the training camera that sees the most of that stretch, as the eye.
"""
import argparse
import os
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_robot as sr  # noqa: E402

HALF = 5.6  # half length of the stretch the pair layout needs: start x +-5 (--home) plus a body radius


def free_map(pts, lo, shape, fill=0.0):
    """Clearance (m) to the nearest obstacle on a 5 cm grid, the floor height per cell (nan where none), the floor
    and the occupied masks. fill: also count floor holes up to 2 * fill m wide (for use with --floor_plane)."""
    fz = sr.floor_map(pts, lo, shape, 0.0, 0.15)
    floor = ndimage.binary_closing(~np.isnan(fz), sr.SQ, iterations=max(1, round(fill / sr.CELL)))
    occ = sr.occupied_map(pts, lo, shape, 0.0, (0.2, 1.8))
    return ndimage.distance_transform_edt(floor & ~occ) * sr.CELL, fz, floor, occ


def save_map(path, floor, occ, clear, pts_xy, lo):
    """Top view as a PPM: no floor dark grey, floor light grey, obstacles black, 0.5 m clearance white, given points red."""
    img = np.full(floor.shape + (3,), 60, np.uint8)
    img[floor] = 170; img[clear > 0.5] = 255; img[occ] = 0
    for xy, col in pts_xy:
        ij = np.floor((np.atleast_2d(xy) - lo) / sr.CELL).astype(int)
        ok = (ij >= 0).all(1) & (ij[:, 0] < floor.shape[0]) & (ij[:, 1] < floor.shape[1])
        for i, j in ij[ok]:
            img[max(i - 1, 0):i + 2, max(j - 1, 0):j + 2] = col
    img = img.transpose(1, 0, 2)[::-1]  # x right, y up
    with open(path, "wb") as f:
        f.write(b"P6 %d %d 255\n" % (img.shape[1], img.shape[0]) + img.tobytes())


def best_stretch(clear, lo, step_deg=3):
    """(centre xy, yaw rad, min clearance) of the 2 * HALF m segment with the largest minimum clearance along it."""
    cells = np.argwhere(clear > 0.4)[::4]
    if not len(cells):
        sys.exit("no free floor")
    cen = (cells + 0.5) * sr.CELL + lo
    s = np.arange(-HALF, HALF + 1e-6, 0.1)
    best = (-1.0, None, None)
    for yaw in np.radians(np.arange(0, 180, step_deg)):
        p = cen[:, None, :] + s[None, :, None] * np.array([np.cos(yaw), np.sin(yaw)])
        ij = np.floor((p - lo) / sr.CELL).astype(int)
        ok = (ij >= 0).all(-1) & (ij[..., 0] < clear.shape[0]) & (ij[..., 1] < clear.shape[1])
        c = np.where(ok, clear[ij[..., 0].clip(0, clear.shape[0] - 1), ij[..., 1].clip(0, clear.shape[1] - 1)], 0)
        m = c.min(1)
        k = np.argmax(m)
        if m[k] > best[0]:
            best = (m[k], cen[k], yaw)
    return best[1], best[2], best[0]


def rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def write_usd(mesh, m, out_dir, splat, plane_z):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt
    st = Usd.Stage.CreateNew(os.path.join(out_dir, "da_collider.usdc"))
    UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(st, 1.0)
    g = UsdGeom.Mesh.Define(st, "/Collider")
    v, f = np.asarray(mesh.vertices, np.float32), np.asarray(mesh.triangles, np.int32)
    g.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(v))
    g.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(f), 3, np.int32)))
    g.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(f.reshape(-1)))
    g.CreateDoubleSidedAttr(True)
    if mesh.has_vertex_colors():
        pv = g.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex)
        pv.Set(Vt.Vec3fArray.FromNumpy(np.asarray(mesh.vertex_colors, np.float32)))
    UsdPhysics.CollisionAPI.Apply(g.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(g.GetPrim()).CreateApproximationAttr("none")  # triangle mesh, static only
    st.SetDefaultPrim(g.GetPrim()); st.GetRootLayer().Save()

    st = Usd.Stage.CreateNew(os.path.join(out_dir, "da_world.usda"))
    UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(st, 1.0)
    root = UsdGeom.Xform.Define(st, "/Scene")
    recon = UsdGeom.Xform.Define(st, "/Scene/Recon")
    recon.AddTransformOp().Set(Gf.Matrix4d(*m.flatten().tolist()))
    col = UsdGeom.Mesh.Define(st, "/Scene/Recon/Collider")
    col.GetPrim().GetReferences().AddReference("./da_collider.usdc")
    if splat:
        col.MakeInvisible()
        sp = st.DefinePrim("/Scene/Recon/Splat")
        sp.GetReferences().AddReference(os.path.abspath(splat))
    if plane_z is not None:  # helper, not scan: flat ground under the task area for a policy trained on a plane
        c = UsdGeom.Cube.Define(st, "/Scene/FloorPlane")
        c.CreateSizeAttr(1.0)
        x = c.AddTranslateOp(); x.Set(Gf.Vec3d(0, 0, plane_z - 0.05))
        c.AddScaleOp().Set(Gf.Vec3f(12.0, 4.0, 0.1))
        c.MakeInvisible()
        UsdPhysics.CollisionAPI.Apply(c.GetPrim())
    st.SetDefaultPrim(root.GetPrim()); st.GetRootLayer().Save()


def main():
    import open3d as o3d
    import level_scene as ls
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh"); ap.add_argument("images"); ap.add_argument("out")
    ap.add_argument("--splat"); ap.add_argument("--floor_plane", type=float)
    ap.add_argument("--home", type=float, default=5.0, help="humanoid start |x| (play_scene.py --home)")
    ap.add_argument("--fill", type=float, default=0.0, help="treat floor holes up to 2 * FILL m wide as floor")
    a = ap.parse_args()
    global HALF
    HALF = a.home + 0.6
    os.makedirs(a.out, exist_ok=True)
    mesh = o3d.io.read_triangle_mesh(a.mesh)
    mesh.compute_vertex_normals()
    imgs = sr.read_images(a.images)
    cams = sr.camera_centres(a.images)
    fwd = np.array([r[2] for _, r, _ in imgs])
    up = -np.mean([r[1] for _, r, _ in imgs], 0); up /= np.linalg.norm(up)
    rot, t = ls.level(np.asarray(mesh.vertices), np.asarray(mesh.vertex_normals), cams, up)
    pts = sr.load_points(a.mesh, 0.02) @ rot.T + t
    cl = cams @ rot.T + t
    lo = cl[:, :2].min(0) - 3
    shape = tuple(np.ceil((cl[:, :2].max(0) + 3 - lo) / sr.CELL).astype(int))
    clear, fz, floor, occ = free_map(pts, lo, shape, a.fill)
    print(f"floor height p5 p50 p95 {np.nanpercentile(fz, [5, 50, 95]).round(3)} m, max clearance {clear.max():.2f} m",
          file=sys.stderr)
    c, yaw, mc = best_stretch(clear, lo)
    # floor under the task area (12 x 2 m along the stretch)
    ij = np.argwhere(~np.isnan(fz))
    xy = (ij + 0.5) * sr.CELL + lo
    loc = (xy - c) @ rz(-yaw)[:2, :2].T
    band = (np.abs(loc[:, 0]) < HALF + 0.4) & (np.abs(loc[:, 1]) < 1.0)
    z = fz[ij[band, 0], ij[band, 1]]
    z0 = float(np.median(z))
    print(f"stretch centre {c.round(2)} yaw {np.degrees(yaw):.0f} deg, min clearance along it {mc:.2f} m; floor there "
          f"p5 p50 p95 {np.percentile(z, [5, 50, 95]).round(3)} m over {band.sum()} of "
          f"{int((2 * HALF + 0.8) * 2 / sr.CELL ** 2)} cells", file=sys.stderr)
    # p'' = A p + b: level, then the stretch centre (at its floor height) to the origin and the stretch onto x
    A = rz(-yaw) @ rot
    b = rz(-yaw) @ (t - np.r_[c, z0])
    m = np.eye(4); m[:3, :3] = A.T; m[3, :3] = b  # USD row-vector layout: p'' = [p, 1] @ M
    write_usd(mesh, m, a.out, a.splat, a.floor_plane)
    seg = c + np.arange(-HALF, HALF, 0.1)[:, None] * np.array([np.cos(yaw), np.sin(yaw)])
    # camera: the training camera that sees the most of the stretch (at 0.8 m), 4 to 14 m away
    ce, cf = cams @ A.T + b, fwd @ A.T
    sx = np.arange(-HALF, HALF + 1e-6, 0.4)
    s = np.c_[sx, np.zeros_like(sx), np.full_like(sx, 0.8)]
    rel = s[None] - ce[:, None]
    d = np.linalg.norm(rel, axis=-1)
    seen = ((rel @ cf[:, :, None])[..., 0] / d > np.cos(np.radians(30))).mean(1)
    dist = np.linalg.norm(ce[:, :2], axis=1)
    k = np.argmax(np.where((dist > 4) & (dist < 14), seen, -1) - 0.01 * np.abs(dist - 8))
    old = rz(-yaw) @ (np.array([-0.89, 4.88, 1.87]) - np.r_[c, z0])  # dah1b's good shot, in the levelled frame
    print(f"camera {k} ({imgs[k][0]}) sees {seen[k]:.0%} of the stretch from {dist[k]:.1f} m; dah1b eye here "
          f"{old.round(2)}", file=sys.stderr)
    eye_l = (ce[k] - b) @ A @ rot.T + t  # back to the levelled frame for the map
    save_map(os.path.join(a.out, "map.ppm"), floor, occ, clear, [(seg, (220, 0, 0)), (cl[:, :2], (0, 0, 220)),
                                                                 (eye_l[:2], (0, 200, 0))], lo)
    print(f"--eye {ce[k][0]:.2f} {ce[k][1]:.2f} {ce[k][2]:.2f}")


def demo():
    """A 3 m wide, 16 m corridor at 40 degrees with walls: the stretch lies along it."""
    rng = np.random.default_rng(0)
    u, n = np.array([np.cos(0.7), np.sin(0.7)]), np.array([-np.sin(0.7), np.cos(0.7)])
    a, w = rng.uniform(-8, 8, 80000), rng.uniform(-1.5, 1.5, 80000)
    floor = np.c_[a[:, None] * u + w[:, None] * n, np.zeros(80000)]
    a2, h = rng.uniform(-8, 8, 40000), rng.uniform(0, 2, 40000)
    wall = np.c_[a2[:, None] * u + np.where(rng.random(40000) < 0.5, -1.6, 1.6)[:, None] * n, h]
    pts = np.r_[floor, wall]
    lo = pts[:, :2].min(0) - 1
    shape = tuple(np.ceil((pts[:, :2].max(0) + 1 - lo) / sr.CELL).astype(int))
    clear, _, _, _ = free_map(pts, lo, shape)
    c, yaw, mc = best_stretch(clear, lo)
    assert abs(np.degrees(yaw) - 40.1) < 4, np.degrees(yaw)
    assert np.linalg.norm(c) < 2.5 and mc > 1.0, (c, mc)
    print("ok")


if __name__ == "__main__":
    demo() if len(sys.argv) == 1 else main()
