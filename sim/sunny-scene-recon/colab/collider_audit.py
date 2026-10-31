"""Audit a splat collider against ground truth (critic_floor_clip.md actions 1, 2, 3 and 6).

    python collider_audit.py RAW.ply FILTERED.ply WALK_DIR IMAGES_TXT POINTS3D_TXT OUT.json     (no args: self-check)

RAW is the floor-clipped TSDF before the cluster filter, FILTERED the same after it (fclip_k2000.ply). Reports:
  blobs: false free cells as 8-connected blobs (m2) and their distance to the nearest GT obstacle cell
  recall: per height bin (0.02-0.10, 0.1-0.3, 0.3-1.0, 1.0-1.8 m) and per GT obstacle over 0.1 m2
  dropped: share of the triangles the filter dropped that lie within 5 cm of GT surface (real geometry lost)
  rule: the critic's GT-free cluster rule on RAW, written next to OUT as *_rule.ply and scored like the rest.
The rule keeps a piece that reaches within 5 cm of the SfM floor or exceeds 1 m2; it drops the rest only if fewer
than 3 training cameras see its centre or it comes within 1 m of a camera. ponytail: "sees" is a frustum test with
no occlusion, so it overcounts views and keeps more (the safe side).
"""
import json
import sys

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from score_robot import CELL, camera_centres, find_floor, load_points, occupied_map, read_images, score

BINS = [(0.02, 0.10), (0.1, 0.3), (0.3, 1.0), (1.0, 1.8)]
EIGHT = np.ones((3, 3), bool)


def audit(recon, gt, lo, hi, z_max=None):
    m = score(recon, gt, lo, hi, z_max=z_max, maps=True)
    shape, zf = m["gt_free"].shape, m["z_floor"]
    ff = m["rc_free"] & ~m["gt_free"]
    lab, n = ndimage.label(ff, EIGHT)
    areas = np.bincount(lab.ravel())[1:] * CELL ** 2
    dist = ndimage.distance_transform_edt(~m["gt_occ"]) * CELL  # metres to the nearest GT obstacle cell
    out = {"false_free_cells": int(ff.sum()), "blobs": int(n),
           "blob_max_m2": float(areas.max()) if n else 0.0, "blobs_over_0.1m2": int((areas > 0.1).sum()),
           "ff_within_10cm_of_gt_obstacle": float((dist[ff] <= 0.10).mean()) if ff.any() else 1.0}
    for b in BINS:  # recon counts if occupied in the same bin (5 cm slack) within one cell
        g = occupied_map(gt, lo, shape, zf, b)
        r = ndimage.binary_dilation(occupied_map(recon, lo, shape, zf, (max(b[0] - 0.05, 0.02), b[1] + 0.05)), EIGHT)
        out[f"recall_{b[0]}_{b[1]}"] = float((g & r).sum() / g.sum()) if g.any() else None
    near = ndimage.binary_dilation(m["rc_occ"], EIGHT)
    lab, n = ndimage.label(m["gt_occ"], EIGHT)
    cells = np.bincount(lab.ravel(), minlength=n + 1)[1:]
    hit = np.bincount(lab[near], minlength=n + 1)[1:]
    big = cells * CELL ** 2 > 0.1
    rec = hit[big] / cells[big]
    cen = ndimage.center_of_mass(np.ones(shape), lab, np.flatnonzero(big) + 1)
    worst = np.argsort(rec)[:5]
    out.update({"obstacles_over_0.1m2": int(big.sum()), "obstacles_under_50pct": int((rec < 0.5).sum()),
               "worst": [{"m2": float(cells[big][i] * CELL ** 2), "recall": float(rec[i]),
                       "xy": (np.asarray(lo) + (np.asarray(cen[i]) + 0.5) * CELL).round(2).tolist()} for i in worst]})
    return out


def clusters(mesh):
    idx, count, area = mesh.cluster_connected_triangles()
    return np.asarray(idx), np.asarray(count), np.asarray(area)


def rule_drop(mesh, z_floor, poses, wh, intr):
    """Per triangle: True where the GT-free rule drops its connected piece."""
    idx, count, area = clusters(mesh)
    v, t = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    tz = v[t, 2].min(1)
    zmin = np.full(len(count), np.inf)
    np.minimum.at(zmin, idx, tz)
    cen = np.zeros((len(count), 3))
    np.add.at(cen, idx, v[t].mean(1))
    cen /= count[:, None]
    imgs = read_images(poses)
    cams = np.array([-r.T @ tt for _, r, tt in imgs])
    dcam = np.full(len(count), np.inf)
    np.minimum.at(dcam, idx, cKDTree(cams).query(v[t].mean(1))[0])
    fx, fy, cx, cy = intr
    views = np.zeros(len(count), int)
    for _, r, tt in imgs:
        p = cen @ r.T + tt
        with np.errstate(divide="ignore", invalid="ignore"):
            u, w = fx * p[:, 0] / p[:, 2] + cx, fy * p[:, 1] / p[:, 2] + cy
        views += (p[:, 2] > 0.1) & (u >= 0) & (u < wh[0]) & (w >= 0) & (w < wh[1])
    keep = (zmin <= z_floor + 0.05) | (area > 1.0)
    drop = ~keep & ((views < 3) | (dcam < 1.0))
    return drop[idx], {"pieces": int(len(count)), "pieces_dropped": int(drop.sum()),
                       "triangles_dropped": int(drop[idx].sum()), "triangles": int(len(idx))}


def dropped_near_gt(raw, kept_mask, gt_tree):
    """Share of the dropped triangles whose centre is within 5 cm of a GT point."""
    v, t = np.asarray(raw.vertices), np.asarray(raw.triangles)
    c = v[t[~kept_mask]].mean(1)
    return float((gt_tree.query(c)[0] < 0.05).mean()) if len(c) else 0.0


def main(raw_p, filt_p, walk, poses, pts3d, out_p):
    import open3d as o3d
    c = camera_centres(poses)
    lo, hi, z_max = c[:, :2].min(0) - 3, c[:, :2].max(0) + 3, c[:, 2].min()
    gt = load_points(walk, lo=lo, hi=hi)
    gt = gt[np.all((gt[:, :2] >= lo) & (gt[:, :2] <= hi), 1)]
    tree = cKDTree(gt)
    raw = o3d.io.read_triangle_mesh(raw_p)
    filt = o3d.io.read_triangle_mesh(filt_p)
    sfm = np.array([l.split()[1:4] for l in open(pts3d) if l.strip() and not l.startswith("#")], float)
    zf = find_floor(sfm, z_max)
    cam = open(f"{walk}/sparse/0/cameras.txt").read().split()
    wh, intr = (float(cam[2]), float(cam[3])), list(map(float, cam[4:8]))
    drop, info = rule_drop(raw, zf, poses, wh, intr)
    rule = o3d.geometry.TriangleMesh(raw)
    rule.remove_triangles_by_mask(drop)
    rule.remove_unreferenced_vertices()
    rule_p = out_p.replace(".json", "_rule.ply")
    o3d.io.write_triangle_mesh(rule_p, rule)
    # k2000's kept triangles, matched by count (its filter is count < 2000 on the same mesh)
    idx, count, _ = clusters(raw)
    k_kept = count[idx] >= 2000
    assert k_kept.sum() == len(filt.triangles), (k_kept.sum(), len(filt.triangles))
    res = {"sfm_floor_z": float(zf), "rule_info": info,
           "dropped_within_5cm_of_gt": {"k2000": dropped_near_gt(raw, k_kept, tree), "rule": dropped_near_gt(raw, ~drop, tree)}}
    for name, path in (("raw", raw_p), ("k2000", filt_p), ("rule", rule_p)):
        rc = load_points(path, lo=lo, hi=hi)
        res[name] = {**{k: v for k, v in score(rc, gt, lo, hi, z_max=z_max).items()
                        if k in ("free_iou", "false_free", "false_occupied", "floor_holes", "obstacle_recall")},
                     **audit(rc, gt, lo, hi, z_max)}
    json.dump(res, open(out_p, "w"), indent=1)
    print(json.dumps(res, indent=1))


def demo():
    import open3d as o3d
    rng = np.random.default_rng(0)
    g = np.mgrid[0:10:0.02, 0:10:0.02].reshape(2, -1).T
    floor = np.c_[g, np.zeros(len(g))]
    crate = lambda x, y, s, h: np.c_[rng.uniform(x, x + s, (20000, 2)) * [1, 1] + [0, y - x], rng.uniform(0, h, 20000)]
    a, b = crate(2, 2, 1, 1), crate(6, 6, 1, 0.5)
    gt = np.r_[floor, a, b]
    # crate b missing: one obstacle at 0 recall, one blob of the crate plus its 0.3 m robot-radius ring (1.6 m)^2
    s = audit(np.r_[floor, a], gt, (0, 0), (10, 10))
    assert s["obstacles_over_0.1m2"] == 2 and s["obstacles_under_50pct"] == 1, s
    assert s["worst"][0]["recall"] < 0.05 and abs(s["worst"][0]["xy"][0] - 6.5) < 0.1, s
    assert s["blobs"] == 1 and 2.2 < s["blob_max_m2"] < 2.7 and s["ff_within_10cm_of_gt_obstacle"] < 0.7, s
    assert s["recall_1.0_1.8"] is None and s["recall_0.3_1.0"] > 0.45 and s["recall_0.1_0.3"] < 0.55, s
    s = audit(gt, gt, (0, 0), (10, 10))
    assert s["blobs"] == 0 and s["obstacles_under_50pct"] == 0 and s["recall_0.1_0.3"] > 0.99, s
    # rule: a floor slab and a crate on it stay; a speck 0.5 m from the camera goes; a shelf box far away stays
    slab = o3d.geometry.TriangleMesh.create_box(10, 10, 0.01).translate([0, 0, -0.01])
    box_on = o3d.geometry.TriangleMesh.create_box(1, 1, 1).translate([2, 2, 0])
    speck = o3d.geometry.TriangleMesh.create_sphere(0.05).translate([5, 5, 1.0])
    shelf = o3d.geometry.TriangleMesh.create_box(0.3, 0.3, 0.3).translate([6.35, 6.35, 0.3])  # seen by 4, over 1 m away
    mesh = slab + box_on + speck + shelf
    r = np.diag([1.0, -1.0, -1.0])  # cameras looking straight down
    with open("/tmp/_audit_images.txt", "w") as f:
        for i, (x, y) in enumerate([(5, 5), (6, 6), (7, 7), (7, 6), (6, 7)]):
            t = -r @ np.array([x, y, 1.5])
            f.write(f"{i} 0 1 0 0 {t[0]} {t[1]} {t[2]} 1 f{i}.png\n\n")
    drop, info = rule_drop(mesh, 0.0, "/tmp/_audit_images.txt", (640, 480), (150, 150, 320, 240))
    n = np.cumsum([len(x.triangles) for x in (slab, box_on, speck, shelf)])
    assert not drop[:n[1]].any() and drop[n[1]:n[2]].all() and not drop[n[2]:].any(), info
    print("collider_audit self-check passed")


if __name__ == "__main__":
    main(*sys.argv[1:7]) if len(sys.argv) > 1 else demo()
