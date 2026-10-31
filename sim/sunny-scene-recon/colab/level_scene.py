"""Level a real reconstruction for Isaac Sim and pick a robot start and goal on its free floor.

    python level_scene.py MESH.ply IMAGES.txt SCENE.usdz OUT.usda     (3dgrut venv: pxr, open3d, scipy)

MESH and the cameras must share the USDZ frame (true for grut_collider.py output with --no-transform exports).
Fits the floor plane among up-facing faces below the cameras, writes OUT.usda referencing ./SCENE.usdz (keep the
two files together) under a transform that puts the floor at z = 0 with +Z up, so the stock spawn heights apply.
Prints the multi_robot_scene.py arguments (an H1 from near the first camera to the farthest free cell within
10 m that is connected to it with 0.8 m clearance) on stdout; everything else goes to stderr.
"""
import os
import sys
import numpy as np
from scipy import ndimage
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_robot as sr


def level(pts, normals, cams, up):
    """Rigid (R, t) with p' = R p + t: floor plane to z = 0, +Z towards the cameras. Also the floor inlier mask."""
    import open3d as o3d
    h = pts @ up
    sel = (np.abs(normals @ up) > 0.95) & (h < (cams @ up).min() - 0.5)
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts[sel]))
    (a, b, c, d), inl = pc.segment_plane(0.03, 3, 2000)
    n = np.array([a, b, c])
    if n @ (cams.mean(0) - pts[sel][inl].mean(0)) < 0:
        n = -n
    rot = Rotation.align_vectors([[0, 0, 1]], [n])[0].as_matrix()
    t = np.array([0, 0, -np.median(pts[sel][inl] @ rot[2])])
    print(f"floor normal {np.round(n, 3)}, tilt {np.degrees(np.arccos(abs(n @ up))):.1f} deg from the camera up, "
          f"{len(inl)} of {sel.sum()} up-facing points", file=sys.stderr)
    return rot, t


def pick(pts, cams, clearance=0.8, reach=10.0):
    lo, hi = cams[:, :2].min(0) - 3, cams[:, :2].max(0) + 3
    shape = tuple(np.ceil((hi - lo) / sr.CELL).astype(int))
    fz = sr.floor_map(pts, lo, shape, 0.0, 0.15)  # real ground is not one plane: 15 cm slack, steps start at 20 cm
    print(f"floor height p5 p50 p95: {np.nanpercentile(fz, [5, 50, 95]).round(3)} m", file=sys.stderr)
    floor = ndimage.binary_closing(~np.isnan(fz), sr.SQ)
    free = floor & ~sr.occupied_map(pts, lo, shape, 0.0, (0.2, 1.8))
    clear = ndimage.distance_transform_edt(free) * sr.CELL
    print(f"floor cells {floor.sum()}, free {free.sum()}, max clearance {clear.max():.2f} m", file=sys.stderr)
    if clear.max() < 0.45:
        sys.exit("no free floor wide enough for an H1 (body radius 0.3 m)")
    if clear.max() <= clearance:  # tight scene: take the widest band there is
        clearance = 0.8 * clear.max()
        print(f"clearance lowered to {clearance:.2f} m", file=sys.stderr)
    comp, _ = ndimage.label(clear > clearance)
    cells = np.argwhere(comp > 0)
    xy = (cells + 0.5) * sr.CELL + lo
    s = np.argmin(np.hypot(*(xy - cams[0, :2]).T))
    mine = comp[tuple(cells[s])] == comp[cells[:, 0], cells[:, 1]]
    d = np.hypot(*(xy - xy[s]).T)
    g = np.argmax(np.where(mine & (d <= reach), d, -1))
    print(f"start {xy[s].round(2)} goal {xy[g].round(2)} ({d[g]:.1f} m)", file=sys.stderr)
    return xy[s], xy[g]


def main():
    import open3d as o3d
    from pxr import Gf, Usd, UsdGeom
    mesh_path, images, usdz, out = sys.argv[1:5]
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    cams = sr.camera_centres(images)
    up = -np.mean([r[1] for _, r, _ in sr.read_images(images)], 0)  # camera -y in world, COLMAP convention
    up /= np.linalg.norm(up)
    rot, t = level(np.asarray(mesh.vertices), np.asarray(mesh.vertex_normals), cams, up)
    pts = sr.load_points(mesh_path, 0.02) @ rot.T + t
    cams = cams @ rot.T + t
    print(f"camera heights above the floor: {np.percentile(cams[:, 2], [5, 50, 95]).round(2)} m", file=sys.stderr)
    s, g = pick(pts, cams)
    m = np.eye(4); m[:3, :3] = rot.T; m[3, :3] = t  # USD row-vector layout: p' = [p, 1] @ M
    st = Usd.Stage.CreateNew(out)
    UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(st, 1.0)
    x = UsdGeom.Xform.Define(st, "/Recon")
    x.GetPrim().GetReferences().AddReference("./" + os.path.basename(usdz))
    x.AddTransformOp().Set(Gf.Matrix4d(*m.flatten().tolist()))
    st.SetDefaultPrim(x.GetPrim()); st.GetRootLayer().Save()
    yaw = np.degrees(np.arctan2(*(g - s)[::-1]))
    back = s - 3 * (g - s) / np.linalg.norm(g - s)  # watch from a real camera spot (known free) behind the start
    eye = cams[np.argmin(np.hypot(*(cams[:, :2] - back).T))] + [0, 0, 0.4]
    print(f"--robot h1 {s[0]:.2f} {s[1]:.2f} {yaw:.0f} 0 0 0 --goal {g[0]:.2f} {g[1]:.2f} "
          f"--eye {eye[0]:.2f} {eye[1]:.2f} {eye[2]:.2f}")


def demo():
    """A tilted floor with a box on it, cameras 1.5 m up: levelled floor at 0, box above it, goal away from the box."""
    rng = np.random.default_rng(0)
    floor = np.c_[rng.uniform(-6, 6, (60000, 2)), np.zeros(60000)]
    box = np.c_[rng.uniform(1, 2, (4000, 2)), rng.uniform(0, 1, 4000)]
    tilt = Rotation.from_euler("xy", [8, -5], degrees=True).as_matrix()
    off = np.array([3.0, -2.0, 7.0])
    world = lambda p: p @ tilt.T + off
    pts, normals = world(np.r_[floor, box]), np.r_[np.tile([0, 0, 1.0], (60000, 1)), np.tile([1.0, 0, 0], (4000, 1))]
    normals = normals @ tilt.T
    cams = world(np.c_[rng.uniform(-3, 3, (20, 2)), np.full(20, 1.5)])
    rot, t = level(pts, normals, cams, tilt[:, 2])
    lv = pts @ rot.T + t
    assert np.abs(lv[:60000, 2]).max() < 0.01 and lv[60000:, 2].max() > 0.9, "floor not at z = 0"
    s, g = pick(lv, cams @ rot.T + t)
    assert np.hypot(*(g - s)) > 5, "goal too close"
    print("ok")


if __name__ == "__main__":
    demo() if len(sys.argv) == 1 else main()
