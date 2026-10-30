"""Splats (vid2scene trainer) vs our dense mesh on the same poses and the same training views.

    python compare_eval.py geometry V D   # pose error, Sim3 into the laser-scan frame, splat depth to points
    python compare_eval.py report V D W   # held-out view quality for every mesh and the splats, scores, table

V is the 1920 px workspace (images/, sparse/0 from GLOMAP-style global SfM, gs/ splats, mvs/ our dense run),
D the ETH3D scene, W our 3200 px run on ETH3D's own poses (reference row).
"""
import json, sys
from pathlib import Path
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

SIZE = 1920


def read_model(folder):
    """COLMAP text model -> {name: camera-to-world}, {name: (K, w, h)}."""
    cams = {}
    for l in Path(folder, "cameras.txt").read_text().splitlines():
        if l and not l.startswith("#"):
            c = l.split()
            fx, fy, cx, cy = map(float, c[4:8])
            cams[c[0]] = (np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]), int(c[2]), int(c[3]))
    poses, intr = {}, {}
    lines = [l for l in Path(folder, "images.txt").read_text().splitlines() if l and not l.startswith("#")]
    for line in lines[::2]:
        f = line.split()
        qw, qx, qy, qz, tx, ty, tz = map(float, f[1:8])
        R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R.T, -R.T @ np.array([tx, ty, tz])
        name = Path(f[9]).name
        poses[name], intr[name] = T, cams[f[8]]
    return poses, intr


def scaled(K, w, h):
    """Intrinsics of a w x h camera resized to SIZE px on the long side."""
    s = SIZE / max(w, h)
    K = K.copy()
    K[:2, :2] *= s
    K[:2, 2] = (K[:2, 2] + 0.5) * s - 0.5
    return K, round(w * s), round(h * s)


def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    U, S, Vt = np.linalg.svd((dst - mu_d).T @ (src - mu_s) / len(src))
    E = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ E @ Vt
    s = (S * np.diag(E)).sum() / ((src - mu_s) ** 2).sum(1).mean()
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = s * R, mu_d - s * R @ mu_s
    return M


def apply(M, c2w):
    """Similarity M on a camera-to-world pose (keeps the rotation orthonormal)."""
    s = np.cbrt(np.linalg.det(M[:3, :3]))
    out = np.eye(4)
    out[:3, :3], out[:3, 3] = M[:3, :3] @ c2w[:3, :3] / s, M[:3, :3] @ c2w[:3, 3] + M[:3, 3]
    return out


def test_names(V):
    return sorted(read_model(V / "sparse_txt")[0])[::8]  # gsplat's split: every 8th image by name


def geometry(V, D):
    gt, _ = read_model(D / "dslr_calibration_undistorted")
    est, _ = read_model(V / "sparse_txt")
    names = sorted(set(gt) & set(est))
    M = umeyama(np.array([est[n][:3, 3] for n in names]), np.array([gt[n][:3, 3] for n in names]))
    aligned = {n: apply(M, est[n]) for n in names}
    rot = [np.degrees(Rotation.from_matrix(gt[n][:3, :3].T @ aligned[n][:3, :3]).magnitude()) for n in names]
    pos = [np.linalg.norm(gt[n][:3, 3] - aligned[n][:3, 3]) * 100 for n in names]
    # No laser scan in real use: scale from one known distance (two camera positions), compare with the full fit.
    s = np.cbrt(np.linalg.det(M[:3, :3]))
    ratios = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            d_gt = np.linalg.norm(gt[names[i]][:3, 3] - gt[names[j]][:3, 3])
            if d_gt > 1.0:
                ratios.append(abs(d_gt / np.linalg.norm(est[names[i]][:3, 3] - est[names[j]][:3, 3]) / s - 1) * 100)
    poses = {"registered": f"{len(est)}/{len(gt)}", "rotation_deg_median": round(float(np.median(rot)), 3),
             "center_cm_median": round(float(np.median(pos)), 2), "center_cm_p90": round(float(np.percentile(pos, 90)), 2),
             "one_distance_scale_error_pct_median": round(float(np.median(ratios)), 2),
             "one_distance_scale_error_pct_p90": round(float(np.percentile(ratios, 90)), 2)}
    print("poses:", poses, flush=True)
    np.savetxt(V / "sim3.txt", M)

    pc = o3d.io.read_point_cloud(str(V / "mvs" / "fused_local.ply"))  # our dense points, into the laser-scan frame
    pc.points = o3d.utility.Vector3dVector(np.asarray(pc.points) @ M[:3, :3].T + M[:3, 3])
    pc.normals = o3d.utility.Vector3dVector(np.asarray(pc.normals) @ (M[:3, :3] / s).T)
    write_float32(pc, V / "mvs" / "fused.ply")

    splat_points(V, D, M, test_names(V), poses)


def write_float32(pc, path):
    """fused.ply for meshing, plus fused_xyz.ply for the ETH3D evaluator, which scores double coordinates and
    float colors as an empty cloud."""
    o3d.io.write_point_cloud(str(path), pc)
    xyz = np.asarray(pc.points, dtype=np.float32)
    o3d.t.io.write_point_cloud(str(path.with_name(path.stem + "_xyz.ply")), o3d.t.geometry.PointCloud(xyz))


def load_splats(V):
    import torch
    sys.path.insert(0, "/content/v2s_gsplat/examples")
    from datasets.colmap import Parser
    parser = Parser(data_dir=str(V), factor=1, normalize=True, test_every=8)
    ckpt = sorted((V / "gs" / "ckpts").glob("ckpt_*_rank0.pt"), key=lambda p: int(p.stem.split("_")[1]))[-1]
    splats = {k: v.cuda() for k, v in torch.load(ckpt, map_location="cuda")["splats"].items()}
    return parser, splats


def render_splats(splats, c2w, K, w, h):
    import torch
    from gsplat.rendering import rasterization
    colors = torch.cat([splats["sh0"], splats["shN"]], 1)
    degree = int(round(np.sqrt(colors.shape[1]))) - 1
    with torch.no_grad():
        out, alpha, _ = rasterization(
            splats["means"], splats["quats"], torch.exp(splats["scales"]), torch.sigmoid(splats["opacities"]), colors,
            torch.tensor(np.linalg.inv(c2w), dtype=torch.float32, device="cuda")[None],
            torch.tensor(K, dtype=torch.float32, device="cuda")[None], w, h, sh_degree=degree, render_mode="RGB+ED")
    out, alpha = out[0].cpu().numpy(), alpha[0, ..., 0].cpu().numpy()
    return np.clip(out[..., :3], 0, 1), out[..., 3], alpha


def splat_points(V, D, M, tests, poses):
    """Expected depth from the splats at every training view (alpha < 0.5 masked), back-projected into the
    laser-scan frame. This is the splat counterpart of COLMAP's fused.ply and is meshed the same way, so the
    only difference between the two rows is where the depth comes from. Like PatchMatch fusion, a point is kept
    only if at least 2 other views' renders agree with its depth within 1% (COLMAP fusion's max_depth_error),
    since expected depth smears at edges."""
    parser, splats = load_splats(V)
    N = M @ np.linalg.inv(parser.transform)  # gsplat's normalized frame -> laser-scan frame
    s = np.cbrt(np.linalg.det(N[:3, :3]))
    views = []
    for i, name in enumerate(parser.image_names):
        if name not in tests:
            K = parser.Ks_dict[parser.camera_ids[i]]
            w, h = parser.imsize_dict[parser.camera_ids[i]]
            rgb, depth, alpha = render_splats(splats, parser.camtoworlds[i], K, w, h)
            views.append((K, w, h, rgb, np.where(alpha > 0.5, depth, 0) * s, N @ parser.camtoworlds[i]))  # depth in metres
    merged = o3d.geometry.PointCloud()
    for a, (K, w, h, rgb, depth, c2w) in enumerate(views):
        r, c = np.mgrid[0:h:2, 0:w:2]
        d = depth[r, c]
        ok = (d > 0) & (d < 30)
        x = np.stack([(c[ok] + 0.5 - K[0, 2]) * d[ok] / K[0, 0], (r[ok] + 0.5 - K[1, 2]) * d[ok] / K[1, 1], d[ok]], axis=1)
        world = x @ (c2w[:3, :3] / s).T + c2w[:3, 3]
        agree = np.zeros(len(world), dtype=int)
        for b, (Kb, wb, hb, _, depth_b, c2w_b) in enumerate(views):
            if a == b:
                continue
            Rb = c2w_b[:3, :3] / s
            p = (world - c2w_b[:3, 3]) @ Rb  # into camera b (Rb is orthonormal, so its inverse is its transpose)
            z = p[:, 2]
            u = np.floor(p[:, 0] / np.maximum(z, 1e-6) * Kb[0, 0] + Kb[0, 2]).astype(int)
            v = np.floor(p[:, 1] / np.maximum(z, 1e-6) * Kb[1, 1] + Kb[1, 2]).astype(int)
            inside = (z > 0) & (u >= 0) & (u < wb) & (v >= 0) & (v < hb)
            db = np.zeros(len(z))
            db[inside] = depth_b[v[inside], u[inside]]
            agree += inside & (db > 0) & (np.abs(db - z) < 0.01 * z)
        keep = agree >= 2
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(world[keep]))
        pc.colors = o3d.utility.Vector3dVector(rgb[r, c][ok][keep])
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(20))
        pc.orient_normals_towards_camera_location(c2w[:3, 3])
        print(f"view {a}: kept {keep.sum()} of {len(keep)} points", flush=True)
        merged += pc
    merged = merged.voxel_down_sample(0.005)
    (V / "splat").mkdir(exist_ok=True)
    write_float32(merged, V / "splat" / "fused.ply")
    print(f"splat depth: {len(merged.points)} points", flush=True)
    (V / "poses.json").write_text(json.dumps(poses, indent=2))


def raycast(mesh_path, c2w, K, w, h):
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    rays = scene.create_rays_pinhole(o3d.core.Tensor(K), o3d.core.Tensor(np.linalg.inv(c2w)), w, h)
    hit = scene.cast_rays(rays)
    tri = hit["primitive_ids"].numpy().astype(np.int64)
    uv = hit["primitive_uvs"].numpy()
    ok = tri != scene.INVALID_ID
    t = np.asarray(mesh.triangles)[np.where(ok, tri, 0)]
    c = np.asarray(mesh.vertex_colors)
    u, v = uv[..., :1], uv[..., 1:]
    rgb = (1 - u - v) * c[t[..., 0]] + u * c[t[..., 1]] + v * c[t[..., 2]]
    return np.where(ok[..., None], rgb, 0), ok


def scores(pred, gt, mask, metrics):
    import torch
    p, g = (torch.tensor(x, dtype=torch.float32).permute(2, 0, 1)[None].cuda() for x in (pred, gt))
    mse_hit = ((pred - gt) ** 2)[mask].mean() if mask.any() else np.nan
    return {"psnr": float(-10 * np.log10(((pred - gt) ** 2).mean())), "psnr_hit": float(-10 * np.log10(mse_hit)),
            "ssim": float(metrics["ssim"](p, g)), "lpips": float(metrics["lpips"](p, g)), "coverage": float(mask.mean())}


def eth3d(path):
    lines = Path(path).read_text().splitlines()
    pick = lambda key: [float(x) for x in next(l for l in lines if l.startswith(key)).split(":")[1].split()]
    return {"accuracy": pick("Accuracies"), "completeness": pick("Completenesses"), "f1": pick("F1-scores")}


def report(V, D, W):
    import imageio.v2 as imageio
    import torch
    from torchmetrics.image import StructuralSimilarityIndexMeasure
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    metrics = {"ssim": StructuralSimilarityIndexMeasure(data_range=1.0).cuda(),
               "lpips": LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).cuda()}
    gt, gt_intr = read_model(D / "dslr_calibration_undistorted")
    est, est_intr = read_model(V / "sparse_txt")
    M = np.loadtxt(V / "sim3.txt")
    parser, splats = load_splats(V)
    tests = test_names(V)
    rows = {"splats (vid2scene trainer)": [], "splat depth, Poisson mesh": [], "ours 1920 (same poses, 38 views)": [],
            "ours 3200 (ETH3D poses, all 44 views)": []}
    for name in tests:
        photo = imageio.imread(V / "images" / name)[..., :3] / 255.0
        i = parser.image_names.index(name)
        K, (w, h) = parser.Ks_dict[parser.camera_ids[i]], parser.imsize_dict[parser.camera_ids[i]]
        rgb, _, alpha = render_splats(splats, parser.camtoworlds[i], K, w, h)
        rows["splats (vid2scene trainer)"].append(scores(rgb, photo, alpha > 0.5, metrics))
        Ke, we, he = est_intr[name]
        pose = apply(M, est[name])  # same pose the splats see, in the laser-scan frame
        for key, mesh in (("splat depth, Poisson mesh", V / "splat" / "frames" / "mesh.ply"),
                          ("ours 1920 (same poses, 38 views)", V / "mvs" / "frames" / "mesh.ply")):
            rgb, hit = raycast(mesh, pose, Ke, we, he)
            rows[key].append(scores(rgb, photo, hit, metrics))
        Kg, wg, hg = scaled(*gt_intr[name])
        rgb, hit = raycast(W / "frames" / "mesh.ply", gt[name], Kg, wg, hg)
        rows["ours 3200 (ETH3D poses, all 44 views)"].append(scores(rgb, photo, hit, metrics))
    out = {"poses": json.loads((V / "poses.json").read_text()), "methods": {}}
    dirs = {"splat depth, Poisson mesh": V / "splat", "ours 1920 (same poses, 38 views)": V / "mvs"}
    for key, r in rows.items():
        entry = {"views": {k: round(float(np.mean([x[k] for x in r])), 3) for k in r[0]}}
        if key in dirs:
            d = dirs[key]
            entry["eth3d_points"] = eth3d(d / "eval_fused_xyz.txt")
            entry["eth3d_mesh"] = eth3d(d / "eval_mesh_points.txt")
            for field, name in (("collisions", "collision_check.json"), ("walk", "walk_results.json")):
                f = d / "frames" / "mujoco" / name
                entry[field] = json.loads(f.read_text()) if f.exists() else "no scene (build_scene found no floor)"
        out["methods"][key] = entry
        print(key, json.dumps(entry), flush=True)
    timing = V / "timing.txt"
    out["seconds"] = dict(l.split()[1:] for l in timing.read_text().splitlines()) if timing.exists() else {}
    (V / "compare.json").write_text(json.dumps(out, indent=2))
    print("seconds:", out["seconds"])


if __name__ == "__main__":
    cmd, args = sys.argv[1], [Path(a) for a in sys.argv[2:]]
    geometry(*args) if cmd == "geometry" else report(*args)
