"""3DGRUT checkpoint -> a capture-style folder of rendered depth, for tsdf_mesh.py.

    python grut_depth.py CKPT OUT     (3dgrut venv, from the 3dgrut repo root)

Writes OUT/images/view_NNNN.png (render), OUT/depth/view_NNNN.png (z-depth in mm, 0 where opacity < 0.5),
OUT/sparse/0/{cameras,images}.txt (PINHOLE intrinsics recovered from the rays, as in grut_points.py; poses from
T_to_world) and an empty OUT/test.txt. Same depth as grut_points.py, without the multi-view filter: the TSDF
averages views itself.
"""
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation
from threedgrut.render import Renderer
import threedgrut.datasets as datasets


def main():
    ckpt, out = sys.argv[1], Path(sys.argv[2])
    for d in ("images", "depth", "sparse/0"):
        (out / d).mkdir(parents=True, exist_ok=True)
    renderer = Renderer.from_checkpoint(ckpt, str(out / "render_tmp"), computes_extra_metrics=False)
    train, _ = datasets.make(name=renderer.conf.dataset.type, config=renderer.conf, ray_jitter=None)
    loader = torch.utils.data.DataLoader(train, batch_size=1, shuffle=False, num_workers=0)
    lines = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            b = train.get_gpu_batch_with_intrinsics(batch)
            o = renderer.model(b)
            alpha = o["pred_opacity"][0, ..., 0]
            t = o["pred_dist"][0, ..., 0] / alpha.clamp(min=1e-6)
            dirs = torch.nn.functional.normalize(b.rays_dir[0], dim=-1)
            z = (b.rays_ori[0] + t[..., None] * dirs)[..., 2]
            z = torch.where((alpha > 0.5) & (z > 0) & (z < 60), z, torch.zeros_like(z)).cpu().numpy()
            name = f"view_{i:04d}.png"
            Image.fromarray((z * 1000).round().astype(np.uint16)).save(out / "depth" / name)
            rgb = o["pred_rgb"][0].clip(0, 1).cpu().numpy() if "pred_rgb" in o else np.full((*z.shape, 3), 0.5)
            Image.fromarray((rgb * 255).astype(np.uint8)).save(out / "images" / name)  # TSDF colour only
            c2w = b.T_to_world[0].cpu().numpy()
            r = c2w[:3, :3].T  # world -> camera
            tt = -r @ c2w[:3, 3]
            qx, qy, qz, qw = Rotation.from_matrix(r).as_quat()
            lines += [f"{i + 1} {qw} {qx} {qy} {qz} {tt[0]} {tt[1]} {tt[2]} 1 {name}", ""]
    h, w = z.shape
    d = b.rays_dir[0].cpu().numpy()
    fx = (w - 1) / (d[0, -1, 0] / d[0, -1, 2] - d[0, 0, 0] / d[0, 0, 2])
    fy = (h - 1) / (d[-1, 0, 1] / d[-1, 0, 2] - d[0, 0, 1] / d[0, 0, 2])
    cx, cy = 0.5 - fx * d[0, 0, 0] / d[0, 0, 2], 0.5 - fy * d[0, 0, 1] / d[0, 0, 2]
    (out / "sparse/0/cameras.txt").write_text(f"1 PINHOLE {w} {h} {fx} {fy} {cx} {cy}\n")
    (out / "sparse/0/images.txt").write_text("\n".join(lines) + "\n")
    (out / "test.txt").write_text("")
    print(f"{i + 1} views {w}x{h} -> {out}")


if __name__ == "__main__":
    main()
