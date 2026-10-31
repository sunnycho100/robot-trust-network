"""Rendered splat depth over GT depth, per pixel, floor and non-floor split (critic_tsdf_collider.md action 2).

    python depth_ratio.py SPLAT_DEPTH CAPTURE OUT.json     (o3d19 venv; no args: self-check)

SPLAT_DEPTH is grut_depth.py's (or gs2d_depth.py's) output folder, CAPTURE is capture_walkthrough.py's. Views are
paired by camera pose (nearest centre plus rotation check), never by index, since the trainer drops held-out frames.
Floor pixels: the GT surface point lies within 5 cm of the floor height. Bar from the critic: a floor median within
1 plus or minus 0.03 retracts the README claim that splat depth sits behind the surfaces.
"""
import json
import sys
import numpy as np
from score_robot import camera_centres, find_floor, read_images


def pair(a, b):
    """For each pose in a, the index of the pose in b with the nearest centre; returns (index, centre gap, rot gap)."""
    ca = np.array([-r.T @ t for r, t in a])
    cb = np.array([-r.T @ t for r, t in b])
    j = np.argmin(np.linalg.norm(ca[:, None] - cb[None], axis=2), 1)
    gap = np.linalg.norm(ca - cb[j], axis=1)
    rot = np.array([np.abs(ra - b[k][0]).max() for (ra, _), k in zip(a, j)])
    return j, gap, rot


def ratios(sz, gz, world_z, z_floor):
    """Per-pixel stats where both depths are valid. Returns (floor ratios, other ratios, floor abs errors)."""
    ok = (sz > 0) & (gz > 0)
    fl = ok & (np.abs(world_z - z_floor) < 0.05)
    nf = ok & ~fl
    return sz[fl] / gz[fl], sz[nf] / gz[nf], np.abs(sz[fl] - gz[fl])


def world_z(gz, r, t, intr):
    fx, fy, cx, cy = intr
    v, u = np.mgrid[:gz.shape[0], :gz.shape[1]]
    cam = np.stack([(u + 0.5 - cx) / fx * gz, (v + 0.5 - cy) / fy * gz, gz], -1)
    return ((cam - t) @ r)[..., 2]


def summary(fr, nr, fe, per_view):
    q = lambda x: [float(np.percentile(x, p)) for p in (10, 50, 90)] if len(x) else None
    return {"floor_ratio_p10_p50_p90": q(fr), "other_ratio_p10_p50_p90": q(nr),
            "floor_within_5cm": float((fe < 0.05).mean()) if len(fe) else None,
            "floor_pixels": int(len(fr)), "other_pixels": int(len(nr)),
            "per_view_floor_median_p10_p50_p90": q([v for v in per_view if v is not None])}


def main():
    from PIL import Image
    sd, cap, out = sys.argv[1:4]
    s_imgs, g_imgs = read_images(f"{sd}/sparse/0/images.txt"), read_images(f"{cap}/sparse/0/images.txt")
    j, gap, rot = pair([(r, t) for _, r, t in s_imgs], [(r, t) for _, r, t in g_imgs])
    print(f"pairing: max centre gap {gap.max():.4f} m, max rotation gap {rot.max():.4f}")
    assert gap.max() < 0.01 and rot.max() < 0.01, "splat views do not match capture poses"
    intr = list(map(float, open(f"{cap}/sparse/0/cameras.txt").read().split()[4:8]))
    cams = camera_centres(f"{cap}/sparse/0/images.txt")
    loads = []
    for (sn, _, _), k in zip(s_imgs, j):
        gn, r, t = g_imgs[k]
        sz = np.asarray(Image.open(f"{sd}/depth/{sn}"), np.float32) / 1000
        gz = np.asarray(Image.open(f"{cap}/depth/{gn}"), np.float32) / 1000
        assert sz.shape == gz.shape, (sn, sz.shape, gn, gz.shape)
        loads.append((sz, gz, world_z(gz, r, t, intr)))
    sample = np.concatenate([np.c_[np.zeros(((w := l[2][l[1] > 0][::50]).size, 2)), w] for l in loads])
    z_floor = find_floor(sample, cams[:, 2].min())
    fr, nr, fe, per_view = [], [], [], []
    for sz, gz, wz in loads:
        a, b, c = ratios(sz, gz, wz, z_floor)
        fr.append(a), nr.append(b), fe.append(c)
        per_view.append(float(np.median(a)) if len(a) > 100 else None)
    s = summary(np.concatenate(fr), np.concatenate(nr), np.concatenate(fe), per_view)
    s.update(z_floor=float(z_floor), views=len(loads), splat=sd, capture=cap)
    json.dump(s, open(out, "w"), indent=1)
    print(json.dumps(s, indent=1))


def demo():
    from scipy.spatial.transform import Rotation
    rng = np.random.default_rng(0)
    poses = [(Rotation.from_euler("xyz", rng.uniform(-1, 1, 3)).as_matrix(), rng.uniform(-5, 5, 3)) for _ in range(6)]
    j, gap, rot = pair(poses[::-1][:4], poses)  # shuffled subset still finds its own poses
    assert list(j) == [5, 4, 3, 2] and gap.max() < 1e-9 and rot.max() < 1e-9, (j, gap)
    # camera 1.5 m up looking straight down at a floor (z = 0): every pixel is floor at depth 1.5
    r = np.diag([1.0, -1.0, -1.0]); t = -r @ np.array([0, 0, 1.5])
    gz = np.full((40, 60), 1.5)
    wz = world_z(gz, r, t, (50, 50, 30, 20))
    assert np.abs(wz).max() < 1e-9, wz
    gz[:10] = 1.0  # a 0.5 m box top in the first rows: not floor
    wz = world_z(gz, r, t, (50, 50, 30, 20))
    sz = gz * 1.1
    sz[30:] = 0  # splat empty there
    fr, nr, fe = ratios(sz, gz, wz, 0.0)
    assert len(fr) == 20 * 60 and len(nr) == 10 * 60 and np.allclose(fr, 1.1) and np.allclose(nr, 1.1), (len(fr), len(nr))
    assert summary(fr, nr, fe, [1.1])["floor_within_5cm"] == 0.0  # 15 cm off
    print("depth_ratio self-check passed")


if __name__ == "__main__":
    main() if len(sys.argv) > 1 else demo()
