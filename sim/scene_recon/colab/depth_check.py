# Per-pixel depth vs ETH3D's ground-truth depth maps (rendered from the laser scan), on the held-out and training
# views: vid2scene's rendered splat depth, its depth fused and meshed like ours, and our PatchMatch mesh, all on the
# same poses. Run by depth_check.sh after compare_splat.sh; writes /content/depth_check.json and depth_check.jpg.
import sys, glob, json
sys.path[:0] = ["/content/robosim/sim/scene_recon/colab"]
from pathlib import Path
import numpy as np, open3d as o3d
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
from compare_eval import read_model, apply, test_names, load_splats, render_splats

V, D, S = Path("/content/v2s/delivery_area"), Path("/content/eth3d/delivery_area"), 4
gt, gt_intr = read_model(D / "dslr_calibration_undistorted")
_, dist_intr = read_model(D / "dslr_calibration_jpg")  # the depth maps are for the original (distorted) photos
dist_params = {}
for l in (D / "dslr_calibration_jpg/cameras.txt").read_text().splitlines():
    if l and not l.startswith("#"):
        assert l.split()[1] == "THIN_PRISM_FISHEYE", l
        dist_params[l.split()[0]] = np.array(l.split()[8:16], dtype=float)
cam_of = {}
lines = [l for l in (D / "dslr_calibration_jpg/images.txt").read_text().splitlines() if l and not l.startswith("#")]
for line in lines[::2]:
    cam_of[Path(line.split()[9]).name] = line.split()[8]

def gt_depth(name, Ku, wu, hu):
    """ETH3D depth (distorted photo) sampled on our undistorted pixel grid, stride S: COLMAP's thin-prism fisheye
    forward model maps each undistorted ray to its distorted pixel. Same pose, so z carries over."""
    Kd, wd, hd = dist_intr[name]
    k1, k2, p1, p2, k3, k4, sx1, sy1 = dist_params[cam_of[name]]
    us, vs = np.meshgrid(np.arange(0, wu, S), np.arange(0, hu, S))
    x, y = (us - Ku[0, 2]) / Ku[0, 0], (vs - Ku[1, 2]) / Ku[1, 1]
    r = np.hypot(x, y)
    f = np.where(r > 1e-9, np.arctan(r) / np.maximum(r, 1e-9), 1.0)
    u, v = x * f, y * f
    t2 = u * u + v * v
    rad = k1 * t2 + k2 * t2 ** 2 + k3 * t2 ** 3 + k4 * t2 ** 4
    ud = u + u * rad + 2 * p1 * u * v + p2 * (t2 + 2 * u * u) + sx1 * t2
    vd = v + v * rad + 2 * p2 * u * v + p1 * (t2 + 2 * v * v) + sy1 * t2
    px, py = np.round(Kd[0, 0] * ud + Kd[0, 2]).astype(int), np.round(Kd[1, 1] * vd + Kd[1, 2]).astype(int)
    f = [p for p in glob.glob(str(D / "ground_truth_depth" / "**" / name), recursive=True)]
    full = np.fromfile(f[0], dtype=np.float32).reshape(hd, wd)
    ok = (px >= 0) & (px < wd) & (py >= 0) & (py < hd)
    d = np.full(us.shape, np.nan)
    d[ok] = full[py[ok], px[ok]]
    d[~np.isfinite(d) | (d <= 0)] = np.nan
    return d, np.sqrt(1 + x * x + y * y)
M = np.loadtxt(V / "sim3.txt")
parser, splats = load_splats(V)
P = parser.transform @ np.linalg.inv(M)  # laser frame -> the splats' normalized frame
units = np.cbrt(np.linalg.det(P[:3, :3]))  # splat units per meter
print("splat units per meter", round(units, 4))
casters = {}
for key, p in (("splat", V / "splat/frames/mesh.ply"), ("ours", V / "mvs/frames/mesh.ply")):
    casters[key] = o3d.t.geometry.RaycastingScene()
    casters[key].add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(o3d.io.read_triangle_mesh(str(p))))

def mesh_depth(rc, c2w, K, w, h):
    rays = rc.create_rays_pinhole(o3d.core.Tensor(K), o3d.core.Tensor(np.linalg.inv(c2w)), w, h)
    t = rc.cast_rays(rays)["t_hit"].numpy()
    r = rays.numpy()
    ok = np.isfinite(t)
    p = r[..., :3] + np.where(ok, t, 0)[..., None] * r[..., 3:]
    return np.where(ok, (p - c2w[:3, 3]) @ c2w[:3, 2], np.nan)

def splat_depth(c2w, K, w, h):
    _, ed, alpha = render_splats(splats, apply(P, c2w), K, w, h)
    return np.where(alpha > 0.5, ed / units, np.nan)

KEYS = ["splats, rendered depth", "splats, depth fused and meshed like ours", "ours (PatchMatch mesh)"]
names = sorted(read_model(V / "sparse_txt")[0])
sets = {"held out": names[::8], "training": names[4::8]}  # gsplat holds out every 8th; PatchMatch got the same 38
RAY, rows, figs = None, {}, []
for split, views in sets.items():
    for name in views:
        K, w, h = gt_intr[name]
        dgt, ray = gt_depth(name, K, w, h)
        Ks = K.copy(); Ks[:2] /= S
        ws, hs = dgt.shape[1], dgt.shape[0]
        preds = dict(zip(KEYS, (splat_depth(gt[name], Ks, ws, hs), mesh_depth(casters["splat"], gt[name], Ks, ws, hs),
                                mesh_depth(casters["ours"], gt[name], Ks, ws, hs))))
        if RAY is None:  # z depth or distance along the ray? the one that agrees with our mesh at the edges
            o = preds[KEYS[2]]
            good = np.isfinite(o) & np.isfinite(dgt) & (np.abs(o / dgt - 1) < 0.1)
            edge = lambda d, m: np.median((o / d)[good & m])
            cz = abs(np.log(edge(dgt, ray > 1.1) / edge(dgt, ray < 1.02)))
            cr = abs(np.log(edge(dgt / ray, ray > 1.1) / edge(dgt / ray, ray < 1.02)))
            RAY = cr < cz
            print("ETH3D depth is", "distance along the ray" if RAY else "z depth", round(cz, 4), round(cr, 4))
        if RAY:
            dgt = dgt / ray
        valid = np.isfinite(dgt)
        common = valid & np.all([np.isfinite(d) for d in preds.values()], axis=0)
        for key, dm in preds.items():
            has = valid & np.isfinite(dm)
            e, rel = np.abs(dm - dgt)[has], (dm / dgt)[has]
            rows.setdefault((split, key), []).append({
                "has_depth": has.sum() / valid.sum(), "within_5cm": (e < 0.05).sum() / valid.sum(),
                "within_10cm": (e < 0.10).sum() / valid.sum(), "within_1pct": (np.abs(rel - 1) < 0.01).sum() / valid.sum(),
                "median_cm": np.median(e) * 100, "median_cm_common": np.median(np.abs(dm - dgt)[common]) * 100,
                "ratio_p25": np.percentile(rel, 25), "ratio_p50": np.median(rel), "ratio_p75": np.percentile(rel, 75)})
        print(split, name, {k: round(float(rows[(split, k)][-1]["within_5cm"]), 3) for k in KEYS}, flush=True)
        if split == "held out":
            figs.append((name, dgt, preds))

out = {f"{s}: {k}": {m: round(float(np.mean([r[m] for r in v])), 3) for m in v[0]} for (s, k), v in rows.items()}
for k, v in out.items():
    print("MEAN", k, json.dumps(v))
json.dump(out, open("/content/depth_check.json", "w"), indent=2)

def pool(a, k=4):  # the laser depth is sparse at this size: median over k x k blocks for display
    hh, ww = a.shape[0] // k * k, a.shape[1] // k * k
    b = a[:hh, :ww].reshape(hh // k, k, ww // k, k).transpose(0, 2, 1, 3).reshape(hh // k, ww // k, -1)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanmedian(b, axis=2)
pick = [x for x in figs if x[0].split(".")[0] in ("DSC_0707", "DSC_0683")]
fig, ax = plt.subplots(len(pick), 5, figsize=(25, 3.7 * len(pick)))
cm_err = plt.get_cmap("RdBu_r").copy(); cm_err.set_bad("#33404d")
cm_d = plt.get_cmap("turbo").copy(); cm_d.set_bad("#33404d")
for r, (name, dgt, preds) in enumerate(pick):
    ax[r, 0].imshow(Image.open(V / "images" / name).convert("RGB")); ax[r, 0].set_title(f"{name} (held out)")
    im = ax[r, 1].imshow(pool(dgt), cmap=cm_d); ax[r, 1].set_title("laser ground-truth depth (m)")
    plt.colorbar(im, ax=ax[r, 1], fraction=0.03)
    for c, (key, dm) in enumerate(preds.items()):
        im = ax[r, 2 + c].imshow(pool(dm - dgt), cmap=cm_err, vmin=-0.5, vmax=0.5)
        m = rows[("held out", key)][[x[0] for x in figs].index(name)]
        ax[r, 2 + c].set_title(f"{key}\n{m['within_5cm'] * 100:.0f}% within 5 cm, median {m['median_cm']:.1f} cm")
        plt.colorbar(im, ax=ax[r, 2 + c], fraction=0.03)
for a in ax.ravel(): a.axis("off")
fig.text(0.5, 0.005, "depth minus true depth (m). Red: surface placed behind the real one, blue: in front of it, gray: no depth", ha="center")
plt.tight_layout(); plt.savefig("/content/depth_check.jpg", dpi=80)
print("DEPTH_DONE")
