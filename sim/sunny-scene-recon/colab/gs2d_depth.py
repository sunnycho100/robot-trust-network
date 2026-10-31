"""gsplat 2DGS checkpoint -> median depth per training view -> TSDF mesh -> WORK/mesh_points.ply for the ETH3D evaluator.

    python3 gs2d_depth.py render CKPT DATA_DIR WORK          (system python with gsplat; writes WORK/views/*.npz)
    /content/o3d19/bin/python gs2d_depth.py fuse WORK [VOXEL_M]  (Open3D 0.19; 0.20's legacy TSDF returns empty meshes)

Uses the trainer's own Runner (same parser, factor 3, no world normalization, so the frame is ETH3D's laser-scan
frame). Self-check before fusing: project each view's SfM points and compare their camera z with the rendered
median depth at that pixel, split into centre and edge pixels. z-depth gives a ratio near 1 everywhere; a ray
distance would read about 1.3 at the edges of this lens.
"""
import sys
from pathlib import Path
import numpy as np


def render(ckpt, data_dir, work):
    sys.path.insert(0, "/content/gsplat/examples")
    import torch
    from simple_trainer_2dgs import Config, Runner
    cfg = Config(data_dir=data_dir, data_factor=3, normalize_world_space=False, disable_viewer=True,
                 result_dir=str(work / "export_tmp"))
    runner = Runner(cfg)
    state = torch.load(ckpt, map_location="cuda")
    for k in runner.splats.keys():
        runner.splats[k].data = state["splats"][k]
    p = runner.parser
    (work / "views").mkdir(parents=True, exist_ok=True)
    centre, edge = [], []
    with torch.no_grad():
        for i in range(len(runner.trainset)):
            d = runner.trainset[i]
            c2w, K = d["camtoworld"][None].cuda(), d["K"][None].cuda()
            h, w = d["image"].shape[:2]
            out = runner.stage.render(runner.scene.id, camtoworlds=c2w, Ks=K, width=w, height=h, sh_degree=cfg.sh_degree,
                                      near_plane=cfg.near_plane, far_plane=cfg.far_plane, render_mode="RGB+ED")
            alpha, med = out[1][0, ..., 0].cpu().numpy(), out[5][0, ..., 0].cpu().numpy()
            depth = np.where(alpha > 0.5, med, 0).astype(np.float32)
            Kn, w2c = K[0].cpu().numpy(), np.linalg.inv(d["camtoworld"].numpy())
            # self-check against this view's SfM points
            name = p.image_names[runner.trainset.indices[i]]
            pts = p.points[p.point_indices.get(name, np.zeros(0, int))]
            pc = pts @ w2c[:3, :3].T + w2c[:3, 3]
            uv = pc[:, :2] / pc[:, 2:] * Kn[[0, 1], [0, 1]] + Kn[[0, 1], [2, 2]]
            ui, vi = np.floor(uv).astype(int).T
            ok = (pc[:, 2] > 0) & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
            r = depth[vi[ok], ui[ok]] / pc[ok, 2]
            off = np.hypot((uv[ok, 0] - Kn[0, 2]) / Kn[0, 0], (uv[ok, 1] - Kn[1, 2]) / Kn[1, 1])
            good = r > 0
            centre += list(r[good & (off < 0.3)]); edge += list(r[good & (off > 0.7)])
            np.savez(work / "views" / f"{i:04d}.npz", depth=depth, K=Kn, w2c=w2c)
            print(f"view {i} {name} alpha>0.5 {np.mean(alpha > 0.5):.2f} sfm pts {ok.sum()}", flush=True)
    rc, re = np.median(centre), np.median(edge)
    print(f"SELFCHECK median depth over SfM z: centre {rc:.3f} (n {len(centre)}), edge {re:.3f} (n {len(edge)})")
    print("SELFCHECK", "PASS" if abs(rc - 1) < 0.05 and abs(re - 1) < 0.05 else "FAIL (depth convention or geometry off)")


def fuse(work, voxel):
    import open3d as o3d
    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel, sdf_trunc=4 * voxel, color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor)
    for f in sorted((work / "views").glob("*.npz")):
        v = np.load(f)
        depth, Kn = v["depth"], v["K"]
        h, w = depth.shape
        # gsplat pixel centres sit at +0.5, Open3D's at integer indices
        intr = o3d.camera.PinholeCameraIntrinsic(w, h, Kn[0, 0], Kn[1, 1], Kn[0, 2] - 0.5, Kn[1, 2] - 0.5)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.zeros((h, w, 3), np.uint8)), o3d.geometry.Image(depth), depth_scale=1.0,
            depth_trunc=15.0, convert_rgb_to_intensity=False)
        vol.integrate(rgbd, intr, v["w2c"])
    mesh = vol.extract_triangle_mesh()
    o3d.io.write_triangle_mesh(str(work / "mesh.ply"), mesh)
    pcd = o3d.geometry.PointCloud(mesh.vertices)
    o3d.io.write_point_cloud(str(work / "mesh_points.ply"), pcd)
    print("mesh vertices", len(mesh.vertices), "triangles", len(mesh.triangles))


if __name__ == "__main__":
    if sys.argv[1] == "render":
        render(sys.argv[2], sys.argv[3], Path(sys.argv[4]))
    else:
        fuse(Path(sys.argv[2]), float(sys.argv[3]) if len(sys.argv) > 3 else 0.01)
