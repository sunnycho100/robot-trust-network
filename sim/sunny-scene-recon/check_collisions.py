"""Do the collision proxies match the visual mesh?

    python check_collisions.py SCENE_DIR [LASER_SCAN]

1. Wall boxes: distance from each box's wall-facing surface to the visual mesh (pass: median < 5 cm).
2. Heightfield ground (outdoor scenes): height error against the mesh surface found by casting down from
   0.5 m above the heightfield, on scanned ground (pass: 95th percentile < 3 cm; a few edge cells always
   miss by more). Curbs and slopes live here, so this is checked on its own.
   With a laser scan (an ETH3D .mlp or a .ply in the scan's frame), the heightfield is also scored against the
   laser's own ground (same cloth filter), since the mesh check alone shares the mesh with the heightfield.
3. Objects: at 0.5 m and 1.0 m above the local ground, the mesh's surface against the outline of the solid
   proxies, both ways: the share of mesh within 5 and 10 cm of a proxy face and of proxy face within 5 and 10 cm
   of the mesh. Mesh over 1 m from walkable ground is left out on both sides, as the proxies leave it out.
   IoU on a 5 cm grid (both dilated by one cell) is kept for comparison; it tops out near 0.5 on thin walls.
"""
import json, sys, warnings
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


def laser_points(path):
    """A laser scan as points in its own frame: a .ply, or an ETH3D .mlp (MeshLab project of aligned scans)."""
    if path.suffix != ".mlp":
        return np.asarray(o3d.io.read_point_cloud(str(path)).points)
    import xml.etree.ElementTree as ET
    out = []
    for m in ET.parse(path).getroot().iter("MLMesh"):
        T = np.array(m.find("MLMatrix44").text.split(), dtype=float).reshape(4, 4)
        p = np.asarray(o3d.io.read_point_cloud(str(path.parent / m.get("filename"))).points)
        out.append(p @ T[:3, :3].T + T[:3, 3])
    return np.vstack(out)


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
        bad = np.zeros(h.shape, dtype=bool)
        bad[rr[hit], cc[hit]] = err > 0.20
        patches = np.bincount(ndimage.label(bad)[0].ravel())[1:]
        g.update({"missed_rays": int((~hit).sum()), "over_5cm_m2": round(float((err > 0.05).sum() * cell ** 2), 1),
                  "over_20cm_m2": round(float(bad.sum() * cell ** 2), 1),
                  "largest_bad_patch_m2": round(float(patches.max() * cell ** 2 if len(patches) else 0), 2)})
        report["ground"] = g
        print(f"heightfield vs mesh ground over {g['cells']} cells: RMS {g['rms_cm']} cm, 95th percentile {g['p95_cm']} cm, "
              f"max {g['max_cm']} cm  {'ok' if g['p95_cm'] < 3 else 'TOO FAR'}")
        print(f"  off by over 5 cm: {g['over_5cm_m2']} m2, over 20 cm: {g['over_20cm_m2']} m2 (largest patch "
              f"{g['largest_bad_patch_m2']} m2), rays that hit nothing: {g['missed_rays']}")
        if len(sys.argv) > 2:
            laser = laser_points(Path(sys.argv[2]))
            T = np.array(info["transform_from_scan"])
            laser = laser @ T[:3, :3].T + T[:3, 3]
            laser = laser[(laser[:, :2] >= lo - 1).all(1) & (laser[:, :2] <= lo + np.array(h.shape[::-1]) * cell + 1).all(1)]
            laser = np.asarray(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(laser)).voxel_down_sample(0.02).points)  # 6 per 5 cm cell
            # the same height band as the mesh (crop_height), around a rough floor from the laser alone (5th percentile
            # per 1 m cell, 3x3 median): under a roof the cloth climbs to the ceiling where the scanner missed the
            # floor, and a glossy floor mirrors the ceiling below the ground. Where the scanner saw no floor for
            # meters, a cell's low point is the ceiling itself; over crop_height above the lowest floor within 3 m,
            # it takes that floor instead, so the ceiling falls out of the band and the cell goes unscored.
            lo1 = laser[:, :2].min(0)
            ij1 = np.floor((laser[:, :2] - lo1)).astype(int)
            low = np.full(ij1.max(0) + 1, np.nan)
            order = np.lexsort((laser[:, 2], ij1[:, 1], ij1[:, 0]))
            cells1, start, n = np.unique(ij1[order], axis=0, return_index=True, return_counts=True)
            low[cells1[:, 0], cells1[:, 1]] = laser[order[start + (n * 0.05).astype(int)], 2]
            with warnings.catch_warnings():  # cells with no laser at all stay empty
                warnings.simplefilter("ignore", RuntimeWarning)
                # before the median, since one pillar is enough. ponytail: one cell of mirror points (none in ETH3D's
                # cleaned scans) would drag its 7 x 7 m down; then drop points facing away from the scanner instead
                floor = ndimage.generic_filter(low, np.nanmin, size=7, mode="nearest")
                low = ndimage.generic_filter(low, np.nanmedian, size=3, mode="nearest")
            low = np.where(low - floor > info["crop_height"], floor, low)
            dz = laser[:, 2] - low[ij1[:, 0], ij1[:, 1]]
            laser = laser[(dz > -1.5) & (dz < info["crop_height"])]  # the mirror sits at -4.6 m here; steps down survive
            from build_scene import ground_heightfield  # the laser's own ground, by the same cloth filter
            truth = ground_heightfield(laser)["ground"]
            ij = np.round((truth[:, :2] - lo) / cell).astype(int)
            ok = (ij >= 0).all(1) & (ij[:, 0] < h.shape[1]) & (ij[:, 1] < h.shape[0])
            flat = ij[ok, 1] * h.shape[1] + ij[ok, 0]
            count = np.bincount(flat, minlength=h.size)
            mean = np.bincount(flat, truth[ok, 2], minlength=h.size) / np.maximum(count, 1)
            def score(height, cells):
                e = np.abs(height.ravel()[cells] - mean[cells])
                return {"cells": int(cells.sum()), "median_cm": round(float(np.median(e)) * 100, 2),
                        "p95_cm": round(float(np.percentile(e, 95)) * 100, 2),
                        "over_5cm_m2": round(float((e > 0.05).sum() * cell ** 2), 1),
                        "over_20cm_m2": round(float((e > 0.20).sum() * cell ** 2), 1)}
            scanned = ~np.isnan(seen).ravel()
            # every scanned cell against the laser's rough floor, laser ground or not: a heightfield on the ceiling
            # where the laser has no ground goes unscored above, and the mesh check shares the mesh
            rr, cc = np.divmod(np.arange(h.size), h.shape[1])
            k = np.floor(lo + np.stack([cc, rr], axis=1) * cell - lo1).astype(int)
            inside = (k >= 0).all(1) & (k < low.shape).all(1)
            off = np.zeros(h.size, dtype=bool)
            off[inside] = np.abs(h.ravel()[inside] - low[k[inside, 0], k[inside, 1]]) > 0.5
            report["ground_vs_laser"] = {
                "scanned ground": score(h, scanned & (count >= 3)),  # ground both our scan and the laser saw
                "filled ground": score(h, ~scanned & (count >= 3)),  # holes we filled, under boxes: the robot walks there too
                "flat floor at z = 0, scanned ground": score(np.zeros_like(h), scanned & (count >= 3)),  # what the heightfield adds
                "scanned ground without laser ground_m2": round(float((scanned & (count < 3)).sum() * cell ** 2), 1),
                "scanned ground over 0.5 m off the laser floor_m2": {
                    "with laser ground": round(float((scanned & (count >= 3) & off).sum() * cell ** 2), 1),
                    "without": round(float((scanned & (count < 3) & off).sum() * cell ** 2), 1)}}
            for k, v in report["ground_vs_laser"].items():
                print(f"heightfield vs laser scan ground, {k}: {v}")

    hulls = [world_mesh(model, data, f"hull_{i:02d}")[0] for i in range(info["clutter_hulls"])]
    hulls = [(Delaunay(p), p.min(0), p.max(0)) for p in hulls if len(p) >= 4]
    lo2, hi2 = surface[:, :2].min(0), surface[:, :2].max(0)
    gx, gy = np.meshgrid(np.arange(lo2[0], hi2[0], CELL) + CELL / 2, np.arange(lo2[1], hi2[1], CELL) + CELL / 2)  # cell centres
    reach = lambda p: np.ones(len(p), dtype=bool)
    if grid is not None:  # proxies leave out what is over 1 m from walkable ground on purpose, and so does the score
        far = ndimage.distance_transform_edt(np.isnan(np.load(scene / "ground_heights.npy"))) * grid[2]
        def reach(p):
            ij = np.clip(np.round((p[:, :2] - grid[1]) / grid[2]).astype(int), 0, np.array(far.shape[::-1]) - 1)
            return far[ij[:, 1], ij[:, 0]] <= 1.0
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
        band = band[reach(band)]
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
        edge = solid & ~ndimage.binary_erosion(solid)
        faces = np.stack([gx[edge], gy[edge]], axis=1)
        faces = faces[reach(faces)]
        two_way = {}
        if len(faces) and len(band):
            d_mesh, d_face = cKDTree(faces).query(band[:, :2])[0], cKDTree(band[:, :2]).query(faces)[0]
            for tol in (0.05, 0.10):
                two_way[f"mesh_within_{int(tol * 100)}cm"] = round(float((d_mesh < tol).mean()), 3)
                two_way[f"proxy_within_{int(tol * 100)}cm"] = round(float((d_face < tol).mean()), 3)
        report.setdefault("surface", {})[str(height)] = two_way
        print(f"{height} m above the ground: mesh covered by proxies {two_way.get('mesh_within_5cm')} / "
              f"{two_way.get('mesh_within_10cm')} (5 / 10 cm), proxy faces on the mesh {two_way.get('proxy_within_5cm')} / "
              f"{two_way.get('proxy_within_10cm')}, IoU {iou:.2f}")
    (scene / "collision_check.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
