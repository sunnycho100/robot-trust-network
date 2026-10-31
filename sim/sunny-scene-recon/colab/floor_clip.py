"""Clip rendered splat depth at the floor plane, so floor reflections stop reading as geometry below the floor.

    python floor_clip.py SPLAT_DEPTH POINTS3D.txt OUT     (no args: self-check)

SPLAT_DEPTH is grut_depth.py's folder. The floor height comes from SfM points only (no ground truth): the most
common height below the lowest camera. Any pixel whose point lands under floor + 2 cm gets the depth where its ray
meets the floor. Writes a grut_depth-style folder (links images and sparse) for tsdf_mesh.py.
ponytail: horizontal floor (z up, as in the capture frame); real scans need level_scene.py first, or a RANSAC plane.
"""
import os
import sys
from pathlib import Path
import numpy as np
from score_robot import find_floor, read_images


def clip(z, r, t, intr, z_floor, tol=0.02):
    """z-depth map -> the same with every point below z_floor + tol moved onto the floor along its ray."""
    fx, fy, cx, cy = intr
    v, u = np.mgrid[:z.shape[0], :z.shape[1]]
    d = np.stack([(u + 0.5 - cx) / fx, (v + 0.5 - cy) / fy, np.ones(z.shape)], -1)  # camera ray at unit z-depth
    o = -r.T @ t
    dw = d @ r  # R^T d per pixel, row vectors
    wz = o[2] + z * dw[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        hit = (z_floor - o[2]) / dw[..., 2]  # z-depth where the ray meets the floor
    low = (z > 0) & (wz < z_floor + tol) & (hit > 0)
    out = z.copy()
    out[low] = hit[low]
    return out, low.mean()


def main():
    from PIL import Image
    sd, pts, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    xyz = np.array([l.split()[1:4] for l in open(pts) if l.strip() and not l.startswith("#")], float)
    imgs = read_images(sd / "sparse/0/images.txt")
    cams = np.array([-r.T @ t for _, r, t in imgs])
    z_floor = find_floor(xyz, cams[:, 2].min())
    print(f"floor z {z_floor:.3f} from {len(xyz)} SfM points (lowest camera {cams[:, 2].min():.2f})")
    intr = list(map(float, open(sd / "sparse/0/cameras.txt").read().split()[4:8]))
    (out / "depth").mkdir(parents=True, exist_ok=True)
    for d in ("images", "sparse", "test.txt"):
        if not (out / d).exists():
            os.symlink((sd / d).resolve(), out / d)
    share = []
    for name, r, t in imgs:
        z = np.asarray(Image.open(sd / "depth" / name), np.float32) / 1000
        c, s = clip(z, r, t, intr, z_floor)
        share.append(s)
        Image.fromarray((c * 1000).round().clip(0, 65535).astype(np.uint16)).save(out / "depth" / name)
    print(f"{len(imgs)} views, clipped share of pixels median {np.median(share):.3f}, max {max(share):.3f}")


def demo():
    r = np.diag([1.0, -1.0, -1.0]); t = -r @ np.array([0, 0, 1.5])  # camera 1.5 m up looking straight down
    z = np.full((20, 30), 3.0)  # a reflection: the floor rendered 1.5 m too deep
    z[:5] = 1.0  # a box top 0.5 m up stays
    z[5:8] = 0  # empty pixels stay empty
    c, s = clip(z, r, t, (40, 40, 15, 10), 0.0)
    assert np.allclose(c[8:], 1.5) and np.allclose(c[:5], 1.0) and (c[5:8] == 0).all(), c
    assert abs(s - 12 / 20) < 1e-9, s
    print("floor_clip self-check passed")


if __name__ == "__main__":
    main() if len(sys.argv) > 1 else demo()
