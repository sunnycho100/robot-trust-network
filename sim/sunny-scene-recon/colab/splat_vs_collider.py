"""Where do the 3DGRUT splat (what you see) and the collider mesh (what you touch) disagree?

    python splat_vs_collider.py CKPT MESH.ply OUT.json [POINTS.ply [SCAN_DIR]]   (3dgrut venv, from the 3dgrut repo root)

For every training view, cast the same rays at the collider (Open3D raycasting) and compare with the splat's
surface distance (pred_dist / opacity, as in grut_points.py; both Euclidean along unit rays). Reports, over
pixels where the splat is opaque, the share where the collider is over 10 cm nearer (an invisible wall in front
of what you see), over 10 cm farther or missing (something you see but walk through), and the median gap.
The mesh must be in the checkpoint's world frame (the dataset frame; true for our --no-transform exports).
With POINTS.ply (the stereo points the mesh came from), also reports which side is backed by measured points:
the share of collider-nearer pixels whose collider hit is within 5 cm of a point (real geometry the splat missed)
and the same for the splat hit (an invented mesher surface in front of a real one shows low collider support).
That check is circular (the mesh was fitted to those points), so SCAN_DIR (ETH3D dslr_scan_eval: scan*.ply plus
scan_alignment.mlp) adds the independent one: per ray, the laser depth is the nearest scan point within about 1.5
pixels of the ray direction, and the report gives the share of collider and splat hits within 5 cm of it.
"""
import json
import sys
import numpy as np
import open3d as o3d
import torch
from threedgrut.render import Renderer
import threedgrut.datasets as datasets


def compare(t_splat, t_mesh, tol=0.10):
    """t_splat: splat hit distances (opaque pixels); t_mesh: collider hit distances (inf = no hit)."""
    gap = t_mesh - t_splat
    hit = np.isfinite(t_mesh)
    return {"pixels": int(len(gap)),
            "collider_nearer_10cm": float(np.mean(hit & (gap < -tol))),
            "collider_farther_10cm": float(np.mean(hit & (gap > tol))),
            "collider_missing": float(np.mean(~hit)),
            "median_abs_gap_m": float(np.median(np.abs(gap[hit]))) if hit.any() else None}


def support(hits, points, tol=0.05, n=200000, seed=0):
    """Share of hit points within tol of a measured point (random subsample of n hits)."""
    from scipy.spatial import cKDTree
    if len(hits) == 0:
        return None
    hits = hits[np.random.default_rng(seed).choice(len(hits), min(n, len(hits)), replace=False)]
    return float(np.mean(cKDTree(points).query(hits, distance_upper_bound=tol)[0] <= tol))


def load_scans(scan_dir, n=4000000, seed=0):
    """ETH3D laser scans in the image frame (MeshLab matrices from scan_alignment.mlp), subsampled to n points."""
    import re
    xml = open(f"{scan_dir}/scan_alignment.mlp").read()
    out = []
    for name, m in re.findall(r'filename="([^"]+)">\s*<MLMatrix44>([^<]+)</MLMatrix44>', xml):
        M = np.array(m.split(), float).reshape(4, 4)
        P = np.asarray(o3d.io.read_point_cloud(f"{scan_dir}/{name}").points)
        out.append(P @ M[:3, :3].T + M[:3, 3])
    P = np.concatenate(out)
    return P[np.random.default_rng(seed).choice(len(P), min(n, len(P)), replace=False)]


def laser_depth(L, centre, dirs, eps=0.0015, k=16):
    """Euclidean distance along each unit world ray from centre to the front-most laser point within eps radians
    (inf when none). ponytail: angular nearest neighbours, not a true z-buffer; a surface the scanner never saw
    lets the one behind it through, which can only make the laser look deeper."""
    from scipy.spatial import cKDTree
    v = L - centre
    r = np.linalg.norm(v, axis=1)
    u = v / r[:, None]
    m = dirs.mean(0) / np.linalg.norm(dirs.mean(0))
    keep = u @ m > (dirs @ m).min() - 0.01  # inside the view cone
    u, r = u[keep], r[keep]
    if len(r) == 0:
        return np.full(len(dirs), np.inf)
    dist, idx = cKDTree(u).query(dirs, k=k, distance_upper_bound=eps)
    return np.where(np.isfinite(dist), r[np.minimum(idx, len(r) - 1)], np.inf).min(1)


def laser_report(t_s, t_m, t_l, tol=0.05):
    has = np.isfinite(t_l)
    near = np.isfinite(t_m) & (t_m - t_s < -0.10)
    agree = np.isfinite(t_m) & (np.abs(t_m - t_s) <= 0.10)
    ok = lambda t, sel: float(np.mean(np.abs(t[sel] - t_l[sel]) <= tol)) if sel.any() else None
    return {"laser_coverage": float(has.mean()),
            "collider_within_5cm_laser": ok(t_m, has & np.isfinite(t_m)), "splat_within_5cm_laser": ok(t_s, has),
            "nearer_collider_within_5cm_laser": ok(t_m, has & near), "nearer_splat_within_5cm_laser": ok(t_s, has & near),
            "agree_collider_within_5cm_laser": ok(t_m, has & agree),
            "median_collider_minus_laser_m": float(np.median((t_m - t_l)[has & np.isfinite(t_m)])) if (has & np.isfinite(t_m)).any() else None,
            "median_splat_minus_laser_m": float(np.median((t_s - t_l)[has])) if has.any() else None}


def main():
    ckpt, mesh_path, out = sys.argv[1:4]
    pts_path = sys.argv[4] if len(sys.argv) > 4 else None
    L = load_scans(sys.argv[5]) if len(sys.argv) > 5 else None
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(o3d.io.read_triangle_mesh(mesh_path))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)
    renderer = Renderer.from_checkpoint(ckpt, "/tmp/svc_render", computes_extra_metrics=False)
    train, _ = datasets.make(name=renderer.conf.dataset.type, config=renderer.conf, ray_jitter=None)
    loader = torch.utils.data.DataLoader(train, batch_size=1, shuffle=False, num_workers=0)
    ts, tm, tl, ps, pm, centres = [], [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            b = train.get_gpu_batch_with_intrinsics(batch)
            o = renderer.model(b)
            alpha = o["pred_opacity"][0, ::4, ::4, 0]
            t = (o["pred_dist"][0, ::4, ::4, 0] / alpha.clamp(min=1e-6)).cpu().numpy()
            c2w = b.T_to_world[0].cpu().numpy()
            d = torch.nn.functional.normalize(b.rays_dir[0, ::4, ::4], dim=-1).cpu().numpy()
            ori = b.rays_ori[0, ::4, ::4].cpu().numpy()
            ok = (alpha.cpu().numpy() > 0.5) & (t < 30)
            rays = np.concatenate([ori[ok] @ c2w[:3, :3].T + c2w[:3, 3], d[ok] @ c2w[:3, :3].T], 1)
            th = scene.cast_rays(o3d.core.Tensor(rays.astype(np.float32)))["t_hit"].numpy()
            tm.append(th)
            ts.append(t[ok])
            ps.append(rays[:, :3] + rays[:, 3:] * t[ok][:, None])
            pm.append(rays[:, :3] + rays[:, 3:] * np.where(np.isfinite(th), th, 0)[:, None])
            centres.append(c2w[:3, 3])
            if L is not None:
                tl.append(laser_depth(L, c2w[:3, 3], rays[:, 3:]))
    res = compare(np.concatenate(ts), np.concatenate(tm))
    res["views"] = len(ts)
    res["camera_centres_first3"] = np.round(centres[:3], 3).tolist()  # check against the stereo poses
    if pts_path:
        P = np.asarray(o3d.io.read_point_cloud(pts_path).points)
        t_s, t_m = np.concatenate(ts), np.concatenate(tm)
        near = np.isfinite(t_m) & (t_m - t_s < -0.10)
        ps, pm = np.concatenate(ps), np.concatenate(pm)
        res["nearer_collider_hit_supported_5cm"] = support(pm[near], P)
        res["nearer_splat_hit_supported_5cm"] = support(ps[near], P)
        agree = np.isfinite(t_m) & (np.abs(t_m - t_s) <= 0.10)
        res["agree_collider_hit_supported_5cm"] = support(pm[agree], P)
    if L is not None:
        res.update(laser_report(np.concatenate(ts), np.concatenate(tm), np.concatenate(tl)))
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    if sys.argv[1:] == ["test"]:
        r = compare(np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.05, 1.5, 3.5, np.inf]))
        assert r["collider_nearer_10cm"] == 0.25 and r["collider_farther_10cm"] == 0.25
        assert r["collider_missing"] == 0.25 and abs(r["median_abs_gap_m"] - 0.5) < 1e-9
        P = np.random.default_rng(1).random((1000, 3))
        assert support(P[:50] + 0.001, P) == 1.0 and support(P[:50] + 1.0, P) == 0.0
        g = np.stack(np.meshgrid(np.linspace(-2, 2, 400), np.linspace(-2, 2, 400)), -1).reshape(-1, 2)
        L = np.concatenate([np.c_[g, np.full(len(g), 10.0)], np.c_[g[:, 0] * 0.25, g[:, 1] * 0.25, np.full(len(g), 5.0)]])
        dirs = np.array([[0, 0, 1.0], [0.05, 0, 1], [0.15, 0, 1]])  # front plane only covers |x| < 0.5 at z 5
        dirs /= np.linalg.norm(dirs, axis=1)[:, None]
        t = laser_depth(L, np.zeros(3), dirs, eps=0.01)
        assert abs(t[0] - 5) < 0.05 and abs(t[1] - 5 / dirs[1, 2]) < 0.05 and abs(t[2] - 10 / dirs[2, 2]) < 0.1, t
        assert laser_depth(L, np.zeros(3), -dirs[:1], eps=0.01)[0] == np.inf
        q = laser_report(np.array([5.0, 5, 5]), np.array([4.0, 5.02, np.inf]), np.array([4.01, 5.0, np.inf]))
        assert q["nearer_collider_within_5cm_laser"] == 1.0 and q["nearer_splat_within_5cm_laser"] == 0.0, q
        assert q["laser_coverage"] == 2 / 3 and q["agree_collider_within_5cm_laser"] == 1.0, q
        print("ok")
    else:
        main()
