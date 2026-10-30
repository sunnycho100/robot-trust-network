"""Do the collision proxies match the visual mesh?

    python check_collisions.py SCENE_DIR

1. Wall boxes: distance from each box's wall-facing surface to the visual mesh (pass: median < 5 cm).
2. Heightfield ground (outdoor scenes): height error against the mesh surface found by casting down from
   0.5 m above the heightfield, on scanned ground (pass: 95th percentile < 3 cm; a few edge cells always
   miss by more). Curbs and slopes live here, so this is checked on its own.
3. Objects: at 0.5 m and 1.0 m above the local ground, compare the mesh's surface with the outline of the
   solid proxies (IoU on a 5 cm grid, both dilated by one cell). Surface against surface, so a perfect fit
   scores near 1 whether the object is a thin wall or a solid crate.
"""
import json, sys
from pathlib import Path
import mujoco
import numpy as np
import open3d as o3d
from scipy import ndimage
from scipy.spatial import Delaunay, cKDTree

CELL = 0.05


def world_mesh(model, data, name):
    g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    m = model.geom_dataid[g]
    v = model.mesh_vert[model.mesh_vertadr[m]:model.mesh_vertadr[m] + model.mesh_vertnum[m]]
    f = model.mesh_face[model.mesh_faceadr[m]:model.mesh_faceadr[m] + model.mesh_facenum[m]]
    return v @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g], f  # MuJoCo stores meshes re-centered


def ground_model(scene, info):
    """Ground height at any xy: the heightfield outdoors, z = 0 indoors."""
    hf = info.get("heightfield")
    if not hf:
        return lambda xy: np.zeros(len(xy)), None
    raw = np.fromfile(scene / "ground.bin", dtype=np.float32)
    h = raw[2:].reshape(hf["shape"])  # int32 nrow, ncol header, then heights
    lo, cell = np.array(hf["origin"]), hf["cell"]
    def at(xy):
        ij = np.clip(np.round((xy - lo) / cell).astype(int), 0, np.array(h.shape[::-1]) - 1)
        return h[ij[:, 1], ij[:, 0]]
    return at, (h, lo, cell)


def main():
    scene = Path(sys.argv[1])
    model = mujoco.MjModel.from_xml_path(str(scene / "scene.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    info = json.loads((scene / "scene_info.json").read_text())
    v, f = world_mesh(model, data, "scan")
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(f))
    surface = np.asarray(mesh.sample_points_uniformly(int(min(mesh.get_surface_area() / 0.02 ** 2, 3e6))).points)
    tree = cKDTree(surface)
    ground_at, grid = ground_model(scene, info)
    report = {"walls": [], "iou": {}}

    if info["walls"]:
        print("wall boxes, distance from the wall-facing side to the mesh:")
    for i, w in enumerate(info["walls"]):
        n = np.array([np.cos(w["yaw"]), np.sin(w["yaw"]), 0.0])
        u = np.array([-n[1], n[0], 0.0])
        pos, (hx, hy, hz) = np.array(w["pos"]), w["half"]
        s, z = np.meshgrid(np.linspace(-hy, hy, 25), np.linspace(-hz, hz, 15))
        best = None
        for side in (1, -1):  # the side that faces the scanned wall surface is the closer one
            face = pos + side * hx * n + s.ravel()[:, None] * u + z.ravel()[:, None] * np.array([0, 0, 1.0])
            d = tree.query(face)[0]
            best = d if best is None or np.median(d) < np.median(best) else best
        med = float(np.median(best))
        report["walls"].append({"median_cm": round(med * 100, 1), "p90_cm": round(float(np.percentile(best, 90)) * 100, 1)})
        print(f"  wall_{i:02d}: median {med * 100:5.1f} cm, 90th percentile {np.percentile(best, 90) * 100:5.1f} cm"
              f"  {'ok' if med < 0.05 else 'TOO FAR'}")

    if grid is not None:  # heightfield vs the lowest mesh surface, where the ground was actually scanned
        h, lo, cell = grid
        seen = np.load(scene / "ground_heights.npy")
        rr, cc = np.nonzero(~np.isnan(seen))
        xy = lo + np.stack([cc, rr], axis=1) * cell
        start = h[rr, cc] + 0.5  # from just above the ground, down: floaters below the ground can't be hit first
        rays = np.hstack([xy, start[:, None], np.tile([0, 0, -1.0], (len(xy), 1))]).astype(np.float32)
        rc = o3d.t.geometry.RaycastingScene()
        rc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        t = rc.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
        hit = np.isfinite(t)
        err = np.abs(h[rr, cc][hit] - (start[hit] - t[hit]))
        g = {"cells": int(hit.sum()), "rms_cm": round(float(np.sqrt((err ** 2).mean())) * 100, 2),
             "p95_cm": round(float(np.percentile(err, 95)) * 100, 2), "max_cm": round(float(err.max()) * 100, 1)}
        report["ground"] = g
        print(f"heightfield vs mesh ground over {g['cells']} cells: RMS {g['rms_cm']} cm, 95th percentile {g['p95_cm']} cm, "
              f"max {g['max_cm']} cm  {'ok' if g['p95_cm'] < 3 else 'TOO FAR'}")

    hulls = [world_mesh(model, data, f"hull_{i:02d}")[0] for i in range(info["clutter_hulls"])]
    hulls = [(Delaunay(p), p.min(0), p.max(0)) for p in hulls if len(p) >= 4]
    lo2, hi2 = surface[:, :2].min(0), surface[:, :2].max(0)
    gx, gy = np.meshgrid(np.arange(lo2[0], hi2[0], CELL), np.arange(lo2[1], hi2[1], CELL))
    xy = np.stack([gx.ravel(), gy.ravel()], axis=1)
    base = ground_at(xy)
    surface_h = surface[:, 2] - ground_at(surface[:, :2])
    for height in (0.5, 1.0):
        pts = np.hstack([xy, (base + height)[:, None]])
        solid = np.zeros(len(pts), dtype=bool)
        for w in info["walls"] + [dict(b, yaw=0.0) for b in info.get("object_boxes", [])]:  # inside a yawed box
            c, s = np.cos(w["yaw"]), np.sin(w["yaw"])
            local = pts - np.array(w["pos"])
            lx, ly = local[:, 0] * c + local[:, 1] * s, -local[:, 0] * s + local[:, 1] * c
            solid |= (np.abs(lx) <= w["half"][0]) & (np.abs(ly) <= w["half"][1]) & (np.abs(local[:, 2]) <= w["half"][2])
        for tri, a, b in hulls:  # bounding-box prefilter, then the exact inside test
            near = np.nonzero((pts >= a).all(axis=1) & (pts <= b).all(axis=1))[0]
            if len(near):
                solid[near[tri.find_simplex(pts[near]) >= 0]] = True
        band = surface[np.abs(surface_h - height) < CELL]
        mesh_occ = np.zeros(gx.shape, dtype=bool)
        if len(band):
            ij = ((band[:, :2] - lo2) / CELL).astype(int)
            ok = (ij[:, 0] < gx.shape[1]) & (ij[:, 1] < gx.shape[0])
            mesh_occ[ij[ok, 1], ij[ok, 0]] = True
        mesh_occ = ndimage.binary_dilation(mesh_occ)
        solid = solid.reshape(gx.shape)
        outline = ndimage.binary_dilation(solid & ~ndimage.binary_erosion(solid))
        iou = (outline & mesh_occ).sum() / max((outline | mesh_occ).sum(), 1)
        report["iou"][str(height)] = round(float(iou), 3)
        print(f"surface IoU {height} m above the ground: {iou:.2f} (mesh {mesh_occ.sum()} cells, proxy outline {outline.sum()} cells)")
    (scene / "collision_check.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
