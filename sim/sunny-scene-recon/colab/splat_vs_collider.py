"""Where do the 3DGRUT splat (what you see) and the collider mesh (what you touch) disagree?

    python splat_vs_collider.py CKPT MESH.ply OUT.json [POINTS.ply]   (3dgrut venv, from the 3dgrut repo root)

For every training view, cast the same rays at the collider (Open3D raycasting) and compare with the splat's
surface distance (pred_dist / opacity, as in grut_points.py; both Euclidean along unit rays). Reports, over
pixels where the splat is opaque, the share where the collider is over 10 cm nearer (an invisible wall in front
of what you see), over 10 cm farther or missing (something you see but walk through), and the median gap.
The mesh must be in the checkpoint's world frame (the dataset frame; true for our --no-transform exports).
With POINTS.ply (the stereo points the mesh came from), also reports which side is backed by measured points:
the share of collider-nearer pixels whose collider hit is within 5 cm of a point (real geometry the splat missed)
and the same for the splat hit (an invented mesher surface in front of a real one shows low collider support).
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


def main():
    ckpt, mesh_path, out = sys.argv[1:4]
    pts_path = sys.argv[4] if len(sys.argv) > 4 else None
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(o3d.io.read_triangle_mesh(mesh_path))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)
    renderer = Renderer.from_checkpoint(ckpt, "/tmp/svc_render", computes_extra_metrics=False)
    train, _ = datasets.make(name=renderer.conf.dataset.type, config=renderer.conf, ray_jitter=None)
    loader = torch.utils.data.DataLoader(train, batch_size=1, shuffle=False, num_workers=0)
    ts, tm, ps, pm, centres = [], [], [], [], []
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
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    if sys.argv[1:] == ["test"]:
        r = compare(np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.05, 1.5, 3.5, np.inf]))
        assert r["collider_nearer_10cm"] == 0.25 and r["collider_farther_10cm"] == 0.25
        assert r["collider_missing"] == 0.25 and abs(r["median_abs_gap_m"] - 0.5) < 1e-9
        P = np.random.default_rng(1).random((1000, 3))
        assert support(P[:50] + 0.001, P) == 1.0 and support(P[:50] + 1.0, P) == 0.0
        print("ok")
    else:
        main()
