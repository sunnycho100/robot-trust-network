"""Write COLMAP's patch-match.cfg from poses alone, for known-pose models with no 2D tracks (COLMAP's "__auto__"
source selection needs shared points, so it finds none and PatchMatch skips every image).

    python pm_sources.py IMAGES_TXT OUT_CFG [N]     (no args: self-check)

Sources per image: the N nearest cameras (default 20) that look within 45 degrees of the same way and sit at least
5 cm away. ponytail: no overlap test; pass --PatchMatchStereo.depth_min and depth_max, which also need points.
"""
import sys

import numpy as np

from score_robot import read_images


def sources(names, centres, axes, n=20, max_deg=45, min_base=0.05):
    d = np.linalg.norm(centres[:, None] - centres[None], axis=-1)
    ok = (axes @ axes.T > np.cos(np.radians(max_deg))) & (d > min_base)
    d[~ok] = np.inf
    out = {}
    for i, name in enumerate(names):
        j = [k for k in np.argsort(d[i])[:n] if np.isfinite(d[i, k])]
        out[name] = [names[k] for k in j]
    return out


def main(images_txt, out, n=20):
    imgs = read_images(images_txt)
    names = [m for m, _, _ in imgs]
    c = np.array([-r.T @ t for _, r, t in imgs])
    ax = np.array([r[2] for _, r, _ in imgs])  # camera +z in world: third row of R (world-to-camera)
    src = sources(names, c, ax, n)
    with open(out, "w") as f:
        for name in names:
            if src[name]:  # an image with no source would make PatchMatch stop
                f.write(f"{name}\n{', '.join(src[name])}\n")
    k = [len(v) for v in src.values()]
    print(f"{len(names)} images, sources min {min(k)} median {int(np.median(k))}, {sum(x == 0 for x in k)} with none")


def demo():
    names = ["a", "b", "c", "d"]
    c = np.array([[0, 0, 0], [0.5, 0, 0], [0.01, 0, 0], [1.0, 0, 0]], float)
    ax = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, -1]], float)  # d looks the other way
    s = sources(names, c, ax, n=5)
    assert s["a"] == ["b"] and s["b"] == ["c", "a"] and s["d"] == [], s
    print("pm_sources self-check passed")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], *map(int, sys.argv[3:4])) if len(sys.argv) > 1 else demo()
