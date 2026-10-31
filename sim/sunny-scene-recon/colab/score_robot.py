"""Score a reconstruction for a walking robot against ground truth (critic_reconstruction.md section 3).

    python score_robot.py RECON.ply GT OUT.json [--poses sparse/0/images.txt] [--band 0.1 1.8] [--radius 0.3]
    python score_robot.py oracle WALK_DIR OUT.ply      GT depth of the training frames as fused points with normals
    python score_robot.py lid MESH.ply WALK_DIR IMAGES_TXT  share of GT floor cells whose mesh is floor, lid or absent

GT is a capture_walkthrough.py folder (its training-frame depth back-projected: only what the cameras saw, as
critic_synthetic_scoring.md finding 3 asks) or a mesh. Both in the same metric, Z up frame. RECON can be points or a
mesh (meshes are sampled). On a 5 cm grid over the camera centres' XY box plus 3 m:
  floor: per GT floor cell, recon floor height error (median, p95), bumps and dips over 2 cm, holes (no recon within
    5 cm). Cells with GT surface 2 to 30 cm above the floor (pallet decks, base plates) are left out of the error.
  free space: occupied = any surface in the height band above the floor; free = floor and not occupied, eroded by
    the robot radius. IoU, false-free (recon free where GT is not: collision danger), false-occupied (blocked paths)
  obstacle recall: GT occupied cells with recon occupied within one cell
Shares are of different denominators and overlap (one floor hole also blocks every free cell within the radius), so
never add them up. ponytail: obstacle recall is per cell, not per GT prim, so thin posts are not a separate row.
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy import ndimage

CELL = 0.05
SQ = np.ones((3, 3), bool)


def grid_index(xy, lo, shape):
    ij = np.floor((xy - lo) / CELL).astype(int)
    ok = (ij[:, 0] >= 0) & (ij[:, 0] < shape[0]) & (ij[:, 1] >= 0) & (ij[:, 1] < shape[1])
    return ij, ok


def floor_map(pts, lo, shape, z_floor, tol):
    """Per cell median height of points within tol of the global floor height (nan where none)."""
    near = np.abs(pts[:, 2] - z_floor) < tol
    ij, ok = grid_index(pts[near, :2], lo, shape)
    ij, z = ij[ok], pts[near][ok, 2]
    flat = ij[:, 0] * shape[1] + ij[:, 1]
    order = np.argsort(flat)
    flat, z = flat[order], z[order]
    out = np.full(shape[0] * shape[1], np.nan)
    if len(flat):
        starts = np.r_[0, np.flatnonzero(np.diff(flat)) + 1]
        ends = np.r_[starts[1:], len(flat)]
        out[flat[starts]] = [np.median(z[s:e]) for s, e in zip(starts, ends)]
    return out.reshape(shape)


def occupied_map(pts, lo, shape, z_floor, band):
    m = np.zeros(shape, bool)
    sel = (pts[:, 2] > z_floor + band[0]) & (pts[:, 2] < z_floor + band[1])
    ij, ok = grid_index(pts[sel, :2], lo, shape)
    m[ij[ok, 0], ij[ok, 1]] = True
    return m


def find_floor(gt, z_max=None):
    """Most common height in the GT, below the lowest camera so a ceiling of similar area cannot win."""
    z = gt[:, 2] if z_max is None else gt[gt[:, 2] < z_max, 2]
    hist, edges = np.histogram(z, bins=np.arange(z.min(), z.max() + 0.02, 0.01))
    return edges[np.argmax(hist)] + 0.005


def score(recon, gt, lo, hi, band=(0.1, 1.8), radius=0.3, z_max=None, maps=False):
    """recon, gt: (N, 3) points in metres, Z up. lo, hi: XY corners of the region to score.
    maps: also return the grids (collider_audit.py)."""
    shape = tuple(np.ceil((np.asarray(hi) - lo) / CELL).astype(int))
    z_floor = find_floor(gt, z_max)
    gt_floor = floor_map(gt, lo, shape, z_floor, 0.05)
    rc_floor = floor_map(recon, lo, shape, z_floor, 0.3)
    # closing fills single empty cells left by sampling, which erosion would otherwise grow into radius-sized holes
    is_floor = ndimage.binary_closing(~np.isnan(gt_floor), SQ)
    rc_has = ndimage.binary_closing(~np.isnan(floor_map(recon, lo, shape, z_floor, 0.05)), SQ)
    low = ndimage.binary_dilation(occupied_map(gt, lo, shape, z_floor, (0.02, 0.3)), SQ)
    eval_cells = is_floor & ~low & ~np.isnan(gt_floor)
    err = (rc_floor - gt_floor)[eval_cells]
    err = err[~np.isnan(err)]
    gt_occ, rc_occ = occupied_map(gt, lo, shape, z_floor, band), occupied_map(recon, lo, shape, z_floor, band)
    k = max(1, int(round(radius / CELL)))
    disk = np.hypot(*np.mgrid[-k:k + 1, -k:k + 1]) <= k
    gt_free = ndimage.binary_erosion(is_floor & ~gt_occ, disk)
    rc_free = ndimage.binary_erosion(rc_has & ~rc_occ, disk)
    union = (gt_free | rc_free).sum()
    rc_occ_near = ndimage.binary_dilation(rc_occ, SQ)
    n_eval, n_floor, n_free = int(eval_cells.sum()), int(is_floor.sum()), int(gt_free.sum())
    if maps:
        return dict(z_floor=z_floor, gt_free=gt_free, rc_free=rc_free, gt_occ=gt_occ, rc_occ=rc_occ)
    return {
        "z_floor": float(z_floor), "cells_floor": n_floor, "cells_floor_scored": n_eval,
        "floor_err_median_m": float(np.median(np.abs(err))) if len(err) else None,
        "floor_err_p95_m": float(np.percentile(np.abs(err), 95)) if len(err) else None,
        "floor_bumps_over_2cm": float((err > 0.02).sum() / max(n_eval, 1)),
        "floor_dips_over_2cm": float((err < -0.02).sum() / max(n_eval, 1)),
        "floor_holes": float((is_floor & ~rc_has).sum() / max(n_floor, 1)),
        "free_iou": float((gt_free & rc_free).sum() / max(union, 1)),
        "false_free": float((rc_free & ~gt_free).sum() / max(n_free, 1)),
        "false_occupied": float((gt_free & ~rc_free).sum() / max(n_free, 1)),
        "obstacle_recall": float((gt_occ & rc_occ_near).sum() / max(gt_occ.sum(), 1)),
        "cells_gt_occupied": int(gt_occ.sum()),
    }


def read_images(images_txt):
    """(name, R world-to-camera, t) per image. Image lines alternate with POINTS2D lines, which may be empty."""
    from scipy.spatial.transform import Rotation
    text = open(images_txt).read().rstrip("\n")
    lines = [l for l in text.splitlines() if not l.startswith("#")][::2]
    out = []
    for line in lines:
        e = line.split()
        qw, qx, qy, qz, tx, ty, tz = map(float, e[1:8])
        out.append((e[9], Rotation.from_quat([qx, qy, qz, qw]).as_matrix(), np.array([tx, ty, tz])))
    return out


def camera_centres(images_txt):
    return np.array([-r.T @ t for _, r, t in read_images(images_txt)])


def depth_points(walk, step=2, normals=False):
    """World points from capture_walkthrough.py's training-frame depth PNGs (test.txt frames left out), 1 cm voxels.
    With normals: an open3d cloud, normals oriented towards each frame's camera (for Poisson)."""
    from PIL import Image
    fx, fy, cx, cy = map(float, open(f"{walk}/sparse/0/cameras.txt").read().split()[4:8])
    test = set(open(f"{walk}/test.txt").read().split())
    clouds = []
    for name, r, t in read_images(f"{walk}/sparse/0/images.txt"):
        if name in test:
            continue
        z = np.asarray(Image.open(f"{walk}/depth/{name}"), np.float32)[::step, ::step] / 1000.0
        v, u = np.nonzero(z > 0)
        zz = z[v, u]
        cam = np.stack([(u * step + 0.5 - cx) / fx * zz, (v * step + 0.5 - cy) / fy * zz, zz], 1)
        world = (cam - t) @ r  # R^T (p - t), row vectors
        if normals:
            import open3d as o3d
            pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(world))
            pc.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(20))
            pc.orient_normals_towards_camera_location(-r.T @ t)
            clouds.append(pc.voxel_down_sample(0.01))
        else:
            clouds.append(world)
    if normals:
        out = clouds[0]
        for c in clouds[1:]:
            out += c
        return out.voxel_down_sample(0.01)
    pts = np.concatenate(clouds)
    _, keep = np.unique(np.floor(pts / 0.01).astype(np.int64), axis=0, return_index=True)
    return pts[keep]


def load_points(path, spacing=0.02, lo=None, hi=None):
    if os.path.isdir(path):
        return depth_points(path)
    import open3d as o3d
    mesh = o3d.io.read_triangle_mesh(path)
    if len(mesh.triangles):
        if lo is not None:  # crop before sampling so far geometry (a skybox) cannot use up the sample cap
            box = o3d.geometry.AxisAlignedBoundingBox([*lo, -1e3], [*hi, 1e3])
            mesh = mesh.crop(box)
        n = int(min(mesh.get_surface_area() / spacing ** 2, 30_000_000))
        return np.asarray(mesh.sample_points_uniformly(n).points)
    return np.asarray(o3d.io.read_point_cloud(path).points)


def demo():
    rng = np.random.default_rng(0)
    g = np.mgrid[0:10:0.02, 0:10:0.02].reshape(2, -1).T
    floor = np.c_[g, np.zeros(len(g))]
    box = np.c_[rng.uniform(4, 5, (20000, 2)), rng.uniform(0, 1, 20000)]  # 1 m crate in the middle
    gt = np.r_[floor, box]
    s = score(gt + rng.normal(0, 0.003, gt.shape), gt, (0, 0), (10, 10))
    assert s["floor_err_median_m"] < 0.005 and s["floor_holes"] < 0.01 and s["free_iou"] > 0.97, s
    assert s["obstacle_recall"] > 0.99 and s["false_free"] < 0.01, s
    # uniform random sampling at 2 cm leaves empty cells; a perfect recon must still score near perfect (critic 1)
    rand = np.c_[rng.uniform(0, 10, (250_000, 2)), np.zeros(250_000)]
    s = score(np.r_[rand, box], np.r_[rand[::-1], box], (0, 0), (10, 10))
    assert s["free_iou"] > 0.97 and s["false_free"] < 0.02 and s["floor_holes"] < 0.01, s
    # a ceiling with more area than the floor must not be taken as the floor (critic 2)
    ceil = np.c_[rng.uniform(-1, 11, (400_000, 2)), np.full(400_000, 9.0)]
    s = score(gt, np.r_[gt, ceil], (0, 0), (10, 10), z_max=1.6)
    assert abs(s["z_floor"]) < 0.01, s
    hole = ~((floor[:, 0] > 1) & (floor[:, 0] < 2) & (floor[:, 1] > 1) & (floor[:, 1] < 2))
    bad = np.r_[floor[hole], box[box[:, 0] < 4.5]]  # 1 m2 floor hole, half the crate missing
    s = score(bad, gt, (0, 0), (10, 10))
    assert 0.008 < s["floor_holes"] < 0.012, s  # 1 of 100 m2 (the crate's floor cells are GT floor too)
    assert 0.4 < s["obstacle_recall"] < 0.6 and s["false_free"] > 0.0, s
    lifted = floor + [0, 0, 0.03]
    s = score(np.r_[lifted, box], gt, (0, 0), (10, 10))
    assert s["floor_bumps_over_2cm"] > 0.9 and abs(s["floor_err_median_m"] - 0.03) < 0.005, s
    # a 14 cm pallet deck is an obstacle, not a floor bump
    deck = np.c_[rng.uniform(7, 8, (20000, 2)), np.full(20000, 0.14)]
    s = score(np.r_[gt, deck], np.r_[gt, deck], (0, 0), (10, 10))
    assert s["floor_bumps_over_2cm"] < 0.005, s
    print("score_robot self-check passed")


def lid(mesh_path, walk, poses):
    """Missing floor or lid (critic_tsdf_collider.md action 1): cast down from 1.85 m over every GT floor cell into
    the mesh and sort the first hit into floor (within 5 cm), lid (0.1 to 1.8 m up), other, or nothing."""
    import open3d as o3d
    c = camera_centres(poses)
    lo, hi = c[:, :2].min(0) - 3, c[:, :2].max(0) + 3
    gt = depth_points(walk)
    gt = gt[np.all((gt[:, :2] >= lo) & (gt[:, :2] <= hi), 1)]
    z_floor = find_floor(gt, c[:, 2].min())
    shape = tuple(np.ceil((hi - lo) / CELL).astype(int))
    cells = np.argwhere(~np.isnan(floor_map(gt, lo, shape, z_floor, 0.05)))
    xy = lo + (cells + 0.5) * CELL
    rays = np.c_[xy, np.full(len(xy), z_floor + 1.85), np.zeros((len(xy), 2)), -np.ones(len(xy))]
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(o3d.io.read_triangle_mesh(mesh_path)))
    t = scene.cast_rays(o3d.core.Tensor(rays.astype(np.float32)))["t_hit"].numpy()
    up = 1.85 - t  # hit height above the floor (-inf where nothing was hit)
    n = len(up)
    return {"cells": n, "floor": float((np.abs(up) < 0.05).sum() / n), "lid": float(((up > 0.1) & (up < 1.8)).sum() / n),
            "nothing": float(np.isinf(t).sum() / n), "z_floor": float(z_floor)}


def main():
    if len(sys.argv) == 1:
        return demo()
    if sys.argv[1] == "oracle":
        import open3d as o3d
        pc = depth_points(sys.argv[2], normals=True)
        o3d.io.write_point_cloud(sys.argv[3], pc)
        return print(f"oracle {len(pc.points)} points -> {sys.argv[3]}")
    if sys.argv[1] == "lid":
        return print(json.dumps(lid(*sys.argv[2:5]), indent=1))
    ap = argparse.ArgumentParser()
    ap.add_argument("recon"), ap.add_argument("gt"), ap.add_argument("out")
    ap.add_argument("--poses"), ap.add_argument("--band", type=float, nargs=2, default=(0.1, 1.8))
    ap.add_argument("--radius", type=float, default=0.3)
    a = ap.parse_args()
    lo = hi = z_max = None
    if a.poses:
        c = camera_centres(a.poses)
        lo, hi, z_max = c[:, :2].min(0) - 3, c[:, :2].max(0) + 3, c[:, 2].min()
    recon, gt = load_points(a.recon, lo=lo, hi=hi), load_points(a.gt, lo=lo, hi=hi)
    if lo is None:
        lo, hi = gt[:, :2].min(0), gt[:, :2].max(0)
    s = score(recon, gt, lo, hi, tuple(a.band), a.radius, z_max)
    s.update(recon=a.recon, gt=a.gt, region=[lo.tolist(), hi.tolist()], band=list(a.band), radius=a.radius)
    json.dump(s, open(a.out, "w"), indent=1)
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
