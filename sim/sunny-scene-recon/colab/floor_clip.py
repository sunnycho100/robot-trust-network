"""Clip rendered splat depth at the floor plane, so floor reflections stop reading as geometry below the floor.

    python floor_clip.py SPLAT_DEPTH POINTS3D.txt OUT [--guard]     (no args: self-check)

SPLAT_DEPTH is grut_depth.py's folder. The floor plane comes from SfM points only (no ground truth): the most
common height below the lowest camera, then a least-squares plane through the SfM points within 2 cm of it. Any
pixel whose point lands under plane + 2 cm gets the depth where its ray meets the plane. --guard clips only where
that hit lies in a 0.5 m cell with SfM floor support and makes the pixel a hole elsewhere (critic_floor_clip.md:
holes are safe, floor invented over a pit is not). Off by default: on the glossy warehouse floor SfM supports only
7% of cells, so the guard left 82% of the floor as holes (free IoU 0.02 against 0.96).
Writes a grut_depth-style folder (links images and sparse) for tsdf_mesh.py.
ponytail: one plane; a site with several floor levels needs a plane per level.
"""
import os
import sys
from pathlib import Path
import numpy as np
from score_robot import find_floor, read_images


CELL = 0.5


def floor_plane(xyz, z_max):
    """(unit normal, offset, support) from SfM points: histogram height, then a plane fit through points near it.
    support: (lo, bool grid) of 0.5 m XY cells holding an SfM point within 3 cm of the plane."""
    z0 = find_floor(xyz, z_max)
    near = xyz[np.abs(xyz[:, 2] - z0) < 0.05]
    for tol in (0.05, 0.02):
        c = near.mean(0)
        n = np.linalg.svd(near - c, full_matrices=False)[2][2]
        n *= np.sign(n[2])
        near = near[np.abs((near - c) @ n) < tol]
    c = near.mean(0)
    n = np.linalg.svd(near - c, full_matrices=False)[2][2]
    n *= np.sign(n[2])
    sup = xyz[np.abs(xyz @ n - n @ c) < 0.03, :2]
    lo = sup.min(0)
    ij = np.floor((sup - lo) / CELL).astype(int)
    grid = np.zeros(ij.max(0) + 1, bool)
    grid[ij[:, 0], ij[:, 1]] = True
    return n, float(n @ c), (lo, grid)


def supported(xy, support):
    lo, grid = support
    ij = np.floor((xy - lo) / CELL).astype(int)
    ok = np.all((ij >= 0) & (ij < grid.shape), -1)
    out = np.zeros(xy.shape[:-1], bool)
    out[ok] = grid[ij[ok][:, 0], ij[ok][:, 1]]
    return out


def clip(z, r, t, intr, plane, support=None, tol=0.02):
    """z-depth map -> the same with every point below the plane + tol moved onto it along its ray, where the hit
    has floor support (a hole elsewhere). plane: (unit normal, offset) with normal . x = offset on the floor."""
    fx, fy, cx, cy = intr
    n, off = plane
    v, u = np.mgrid[:z.shape[0], :z.shape[1]]
    d = np.stack([(u + 0.5 - cx) / fx, (v + 0.5 - cy) / fy, np.ones(z.shape)], -1)  # camera ray at unit z-depth
    o = -r.T @ t
    dw = d @ r  # R^T d per pixel, row vectors
    dn = dw @ n
    with np.errstate(divide="ignore", invalid="ignore"):
        hit = (off - n @ o) / dn  # z-depth where the ray meets the plane
    low = (z > 0) & (n @ o + z * dn - off < tol) & (hit > 0)
    out = z.copy()
    ok = low if support is None else low & supported(o[:2] + hit[..., None] * dw[..., :2], support)
    out[ok] = hit[ok]
    out[low & ~ok] = 0
    return out, low.mean(), (low & ~ok).mean()


def main():
    from PIL import Image
    sd, pts, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    xyz = np.array([l.split()[1:4] for l in open(pts) if l.strip() and not l.startswith("#")], float)
    imgs = read_images(sd / "sparse/0/images.txt")
    cams = np.array([-r.T @ t for _, r, t in imgs])
    n, off, support = floor_plane(xyz, cams[:, 2].min())
    if "--guard" not in sys.argv:
        support = None
    tilt = np.degrees(np.arccos(n[2]))
    print(f"floor plane n {n.round(4)} offset {off:.3f} (tilt {tilt:.2f} deg) from {len(xyz)} SfM points, "
          f"support {support[1].mean() if support else 1:.2f} of {support[1].size if support else 0} cells")
    intr = list(map(float, open(sd / "sparse/0/cameras.txt").read().split()[4:8]))
    (out / "depth").mkdir(parents=True, exist_ok=True)
    for d in ("images", "sparse", "test.txt"):
        if not (out / d).exists():
            os.symlink((sd / d).resolve(), out / d)
    share, holed = [], []
    for name, r, t in imgs:
        z = np.asarray(Image.open(sd / "depth" / name), np.float32) / 1000
        c, s, h = clip(z, r, t, intr, (n, off), support)
        share.append(s), holed.append(h)
        Image.fromarray((c * 1000).round().clip(0, 65535).astype(np.uint16)).save(out / "depth" / name)
    print(f"{len(imgs)} views, low share of pixels median {np.median(share):.3f}, max {max(share):.3f}; "
          f"holed (no support) median {np.median(holed):.3f}, max {max(holed):.3f}")


def demo():
    r = np.diag([1.0, -1.0, -1.0]); t = -r @ np.array([0, 0, 1.5])  # camera 1.5 m up looking straight down
    z = np.full((20, 30), 3.0)  # a reflection: the floor rendered 1.5 m too deep
    z[:5] = 1.0  # a box top 0.5 m up stays
    z[5:8] = 0  # empty pixels stay empty
    flat = (np.array([0, 0, 1.0]), 0.0)
    c, s, _ = clip(z, r, t, (40, 40, 15, 10), flat)
    assert np.allclose(c[8:], 1.5) and np.allclose(c[:5], 1.0) and (c[5:8] == 0).all(), c
    assert abs(s - 12 / 20) < 1e-9, s
    # a 1 m pit: SfM sees the floor around it and the pit bottom 1 m down. Unguarded, the pit gets a phantom floor;
    # guarded, its pixels become holes while the reflection on the supported floor is still clipped
    rng = np.random.default_rng(0)
    xy = rng.uniform(-3, 3, (20000, 2))
    pit = np.all(np.abs(xy) < 0.5, 1)
    sfm = np.c_[xy, np.where(pit, -1.0, 0.0) + rng.normal(0, 0.003, len(xy))]
    n, off, sup = floor_plane(np.r_[sfm, [[0, 0, 1.4]]], 1.5)
    assert n[2] > 0.9999 and abs(off) < 0.002, (n, off)
    fx = 40
    v, u = np.mgrid[:60, :60]
    x, y = (u + 0.5 - 30) / fx * 1.5, -(v + 0.5 - 30) / fx * 1.5  # floor xy per pixel (camera flips y)
    inpit = (np.abs(x) < 0.5) & (np.abs(y) < 0.5)
    z = np.where(inpit, 2.5, 3.0)  # pit bottom at 2.5 m, the rest a reflection 1.5 m too deep
    c, _, _ = clip(z, r, t, (fx, fx, 30, 30), (n, off))
    assert np.allclose(c[inpit], 1.5, atol=1e-3), "unguarded clip should floor the pit (the failure)"
    c, _, h = clip(z, r, t, (fx, fx, 30, 30), (n, off), sup)
    inner = (np.abs(x) < 0.4) & (np.abs(y) < 0.4)  # cells straddling the pit edge keep floor support
    assert (c[inner] == 0).all() and np.allclose(c[~inpit], 1.5, atol=1e-3), (c[inner].max(), c[~inpit].min(), c[~inpit].max())
    # a 1 degree tilt is recovered
    a = np.radians(1.0)
    tilted = np.c_[xy, xy[:, 0] * np.tan(a)]
    n, off, _ = floor_plane(tilted, 1.5)
    assert abs(np.degrees(np.arccos(n[2])) - 1.0) < 0.05, n
    print("floor_clip self-check passed")


if __name__ == "__main__":
    main() if len(sys.argv) > 1 else demo()
