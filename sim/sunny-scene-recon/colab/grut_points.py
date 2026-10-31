"""3DGRUT checkpoint -> WORK/fused.ply (dense points in the dataset's world frame) for eth3d_prep.py meshing.

    python grut_points.py CKPT WORK     (run inside the 3dgrut venv, from the 3dgrut repo root)

The 3DGRUT counterpart of compare_eval.splat_points: render each training view, take the ray distance where
opacity > 0.5, back-project, and keep a point only if 2 other views' renders agree with its depth within 1%
(COLMAP fusion's max_depth_error). normalize_world_space is off in the colmap config, so the world frame is
the COLMAP frame (ETH3D's laser-scan frame for its own calibration).
"""
import sys
from pathlib import Path
import numpy as np
import open3d as o3d
import torch
from threedgrut.render import Renderer
import threedgrut.datasets as datasets


def main():
    ckpt, work = sys.argv[1], Path(sys.argv[2])
    renderer = Renderer.from_checkpoint(ckpt, str(work / "render_tmp"), computes_extra_metrics=False)
    conf, model = renderer.conf, renderer.model
    train, _ = datasets.make(name=conf.dataset.type, config=conf, ray_jitter=None)
    loader = torch.utils.data.DataLoader(train, batch_size=1, shuffle=False, num_workers=0)
    views = []
    with torch.no_grad():
        for batch in loader:
            b = train.get_gpu_batch_with_intrinsics(batch)
            out = model(b)
            alpha = out["pred_opacity"][0, ..., 0]
            # ponytail: pred_dist is the opacity-weighted ray parameter, so divide by opacity for the surface
            t = out["pred_dist"][0, ..., 0] / alpha.clamp(min=1e-6)
            # the 3DGUT kernel's hit distance is Euclidean, so step along the unit direction
            dirs = torch.nn.functional.normalize(b.rays_dir[0], dim=-1)
            cam = b.rays_ori[0] + t[..., None] * dirs  # camera frame, z forward
            z = torch.where(alpha > 0.5, cam[..., 2], torch.zeros_like(t))
            views.append((z.cpu().numpy(), cam.cpu().numpy(), b.T_to_world[0].cpu().numpy(),
                          out["pred_rgb"][0].clip(0, 1).cpu().numpy() if "pred_rgb" in out else None))
    # pinhole intrinsics recovered from the rays themselves: u = fx * x/z + cx (pixel centres at +0.5)
    h, w = views[0][0].shape
    d = b.rays_dir[0].cpu().numpy()
    fx = (w - 1) / (d[0, -1, 0] / d[0, -1, 2] - d[0, 0, 0] / d[0, 0, 2])
    fy = (h - 1) / (d[-1, 0, 1] / d[-1, 0, 2] - d[0, 0, 1] / d[0, 0, 2])
    cx, cy = 0.5 - fx * d[0, 0, 0] / d[0, 0, 2], 0.5 - fy * d[0, 0, 1] / d[0, 0, 2]
    print(f"{len(views)} views {w}x{h}, fx {fx:.1f} fy {fy:.1f} cx {cx:.1f} cy {cy:.1f}", flush=True)

    merged = o3d.geometry.PointCloud()
    for a, (z, cam, c2w, rgb) in enumerate(views):
        r, c = np.mgrid[0:h:2, 0:w:2]
        ok = (z[r, c] > 0) & (z[r, c] < 30)
        world = cam[r, c][ok] @ c2w[:3, :3].T + c2w[:3, 3]
        agree = np.zeros(len(world), dtype=int)
        for bi, (zb, _, c2w_b, _) in enumerate(views):
            if bi == a:
                continue
            p = (world - c2w_b[:3, 3]) @ c2w_b[:3, :3]
            pz = p[:, 2]
            u = np.floor(p[:, 0] / np.maximum(pz, 1e-6) * fx + cx).astype(int)
            v = np.floor(p[:, 1] / np.maximum(pz, 1e-6) * fy + cy).astype(int)
            inside = (pz > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
            db = np.zeros(len(pz))
            db[inside] = zb[v[inside], u[inside]]
            agree += inside & (db > 0) & (np.abs(db - pz) < 0.01 * pz)
        keep = agree >= 2
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(world[keep]))
        if rgb is not None:
            pc.colors = o3d.utility.Vector3dVector(rgb[r, c][ok][keep])
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(20))
        pc.orient_normals_towards_camera_location(c2w[:3, 3])
        print(f"view {a}: kept {keep.sum()} of {len(keep)} points", flush=True)
        merged += pc
    merged = merged.voxel_down_sample(0.005)
    work.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(work / "fused.ply"), merged)
    xyz = np.asarray(merged.points, dtype=np.float32)  # the ETH3D evaluator reads float x y z only
    o3d.t.io.write_point_cloud(str(work / "fused_xyz.ply"), o3d.t.geometry.PointCloud(xyz))
    print(f"FUSED {len(xyz)} points", flush=True)


if __name__ == "__main__":
    main()
