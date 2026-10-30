"""Do the collision proxies match the visual mesh? (critic round 2 checks)

    python check_collisions.py SCENE_DIR

1. Wall boxes: distance from each box's wall-facing surface to the visual mesh (pass: median < 5 cm).
2. Free space: at 0.1 m and 1.0 m height, compare where the mesh has a surface with where a proxy is
   solid (IoU of occupied cells, 5 cm grid, mesh surface dilated by one cell since walls are thin).
"""
import json, sys
from pathlib import Path
import mujoco
import numpy as np
from scipy import ndimage
from scipy.spatial import Delaunay, cKDTree


def world_mesh(model, data, name):
    g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    m = model.geom_dataid[g]
    v = model.mesh_vert[model.mesh_vertadr[m]:model.mesh_vertadr[m] + model.mesh_vertnum[m]]
    return v @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]  # MuJoCo stores meshes re-centered


def main():
    scene = Path(sys.argv[1])
    model = mujoco.MjModel.from_xml_path(str(scene / "scene.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    info = json.loads((scene / "scene_info.json").read_text())
    surface = world_mesh(model, data, "scan")
    tree = cKDTree(surface)

    print("wall boxes, distance from the wall-facing side to the mesh:")
    report = {"walls": [], "iou": {}}
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

    hulls = [world_mesh(model, data, f"hull_{i:02d}") for i in range(info["clutter_hulls"])]
    hulls = [Delaunay(h) for h in hulls if len(h) >= 4]
    lo, hi = surface[:, :2].min(0), surface[:, :2].max(0)
    gx, gy = np.meshgrid(np.arange(lo[0], hi[0], 0.05), np.arange(lo[1], hi[1], 0.05))
    for height in (0.1, 1.0):
        pts = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, height)], axis=1)
        solid = np.zeros(len(pts), dtype=bool)
        for w in info["walls"]:  # inside a yawed box
            c, s = np.cos(w["yaw"]), np.sin(w["yaw"])
            local = pts - np.array(w["pos"])
            lx, ly = local[:, 0] * c + local[:, 1] * s, -local[:, 0] * s + local[:, 1] * c
            solid |= (np.abs(lx) <= w["half"][0]) & (np.abs(ly) <= w["half"][1]) & (np.abs(local[:, 2]) <= w["half"][2])
        for h in hulls:
            solid |= h.find_simplex(pts) >= 0
        near = surface[np.abs(surface[:, 2] - height) < 0.05]
        mesh_occ = np.zeros(gx.shape, dtype=bool)
        if len(near):
            ij = ((near[:, :2] - lo) / 0.05).astype(int)
            ok = (ij[:, 0] < gx.shape[1]) & (ij[:, 1] < gx.shape[0])
            mesh_occ[ij[ok, 1], ij[ok, 0]] = True
        mesh_occ = ndimage.binary_dilation(mesh_occ)
        solid = solid.reshape(gx.shape)
        iou = (solid & mesh_occ).sum() / max((solid | mesh_occ).sum(), 1)
        report["iou"][str(height)] = round(float(iou), 3)
        print(f"occupied-space IoU at {height} m: {iou:.2f} (mesh {mesh_occ.sum()} cells, proxies {solid.sum()} cells)")
    (scene / "collision_check.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
