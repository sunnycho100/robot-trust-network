"""Wall-contact test for a reconstruction collider (critic_hybrid_walk action 7): H1s walk straight into walls.

    python wall_sites.py pick MESH.ply IMAGES.txt WORLD.usda SITES.json [N]   (3dgrut venv: pxr, open3d, scipy)
    python wall_sites.py check SITES.json TRAJ.csv

pick: in the levelled frame of WORLD.usda (level_scene.py output), finds N sites, at least 3 m apart, where an H1
standing on collider floor faces a wall 1 m away (points 1.0 to 1.8 m up, flat across 0.8 m) with floor all the way.
Each robot's goal is 1.5 m behind the wall face, so the fleet's heading pursuit pushes it into the wall. Prints the
multi_robot_scene.py arguments. check: per robot, how far the base got along its heading compared with the wall
face (pass-through if over 0.3 m past it), and falls (base z under 60% of its standing height).
"""
import json
import os
import sys
import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_robot as sr


def pick(pts, lo, shape, n=6, dist=1.0, apart=3.0):
    c = sr.CELL
    floor = ndimage.binary_closing(~np.isnan(sr.floor_map(pts, lo, shape, 0.0, 0.15)), sr.SQ)
    occ = sr.occupied_map(pts, lo, shape, 0.0, (0.2, 1.8))
    tall = sr.occupied_map(pts, lo, shape, 0.0, (1.0, 1.8))
    d, (ii, jj) = ndimage.distance_transform_edt(~occ, return_indices=True)
    floor_r = ndimage.binary_erosion(floor, iterations=int(dist / c))  # the whole 1 m disk round the start has floor
    cand = np.argwhere(floor_r & ~occ & (np.abs(d * c - dist) < 0.05))
    sites = []
    for a in cand[np.random.default_rng(0).permutation(len(cand))]:
        hit = np.array([ii[tuple(a)], jj[tuple(a)]])
        if not tall[tuple(hit)]:
            continue
        s, h = (a + 0.5) * c + lo, (hit - a) / np.linalg.norm(hit - a)
        face = (hit + 0.5) * c + lo
        lat = np.array([-h[1], h[0]])
        ray = [s + h * k for k in np.arange(0, np.linalg.norm(face - s) - 0.1, c)]
        wall = [face + h * 0.05 + lat * k for k in np.arange(-0.4, 0.41, c)]  # flat face the shoulders also meet
        ij = lambda q: tuple(np.clip(((np.asarray(q) - lo) / c).astype(int), 0, np.array(shape) - 1).T)
        if not floor[ij(ray)].all() or occ[ij(ray)].any() or occ[ij(wall)].mean() < 0.7:
            continue
        if all(np.hypot(*(s - q["start"])) > apart and np.hypot(*(face - q["face"])) > apart for q in sites):
            sites.append({"start": s, "heading": h, "face": face, "wall_dist": float((face - s) @ h), "goal": face + h * 1.5})
        if len(sites) == n:
            break
    return sites


def check(sites, traj):
    d = np.genfromtxt(traj, delimiter=",", names=True)
    out = []
    for r, q in enumerate(sites):
        x = d[d["robot"] == r]
        adv = (np.c_[x["x"], x["y"]] - q["start"]) @ np.asarray(q["heading"])
        med = np.median(x["z"][(x["t"] >= 0.5) & (x["t"] <= 2)])
        fall = x["t"][x["z"] < 0.6 * med]
        past = adv.max() - q["wall_dist"]
        out.append({"robot": r, "wall_dist": round(q["wall_dist"], 2), "max_advance": round(float(adv.max()), 2),
                    "past_face_m": round(float(past), 2), "pass_through": bool(past > 0.3),
                    "fall_s": round(float(fall[0]), 1) if len(fall) else None, "z_median": round(float(med), 2)})
        print(out[-1])
    print(f"pass-through {sum(o['pass_through'] for o in out)} of {len(out)}, falls {sum(o['fall_s'] is not None for o in out)}")
    return out


def demo():
    """Floor with a wall along x = 4 and a 20 cm curb along y = 3: sites face the wall, never the curb; check flags a pass."""
    rng = np.random.default_rng(1)
    floor = np.c_[rng.uniform(-6, 6, (300000, 2)), np.zeros(300000)]
    wall = np.c_[np.full(40000, 4.0) + rng.uniform(0, 0.2, 40000), rng.uniform(-6, 6, 40000), rng.uniform(0, 2.5, 40000)]
    curb = np.c_[rng.uniform(-6, 3, 10000), np.full(10000, 3.0), rng.uniform(0, 0.2, 10000)]
    pts = np.r_[floor[floor[:, 0] < 4], wall, curb]
    lo = np.array([-6.0, -6.0]); shape = (int(12 / sr.CELL),) * 2
    sites = pick(pts, lo, shape, n=3)
    assert len(sites) == 3, sites
    for q in sites:
        assert np.allclose(q["heading"], [1, 0], atol=0.1) and abs(q["face"][0] - 4.0) < 0.1 and abs(q["wall_dist"] - 1.0) < 0.1, q
    import tempfile
    t = np.arange(0, 10, 0.02)
    rows = [(tt, 0, *(sites[0]["start"] + sites[0]["heading"] * min(tt * 0.6, 0.85)), 0, 0.9) for tt in t]
    rows += [(tt, 1, *(sites[1]["start"] + sites[1]["heading"] * tt * 0.6), 0, 0.9 if tt < 5 else 0.2) for tt in t]
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        f.write("t,robot,x,y,yaw,z\n" + "".join(",".join(map(str, r)) + "\n" for r in rows))
    o = check(sites[:2], f.name)
    assert not o[0]["pass_through"] and o[0]["fall_s"] is None and o[1]["pass_through"] and o[1]["fall_s"] == 5.0, o
    print("ok")


def main():
    if sys.argv[1] == "check":
        return check(json.load(open(sys.argv[2])), sys.argv[3])
    from pxr import Usd, UsdGeom
    mesh, images, world, out = sys.argv[2:6]
    st = Usd.Stage.Open(world)  # keep the stage alive, or its prims go invalid
    m = np.array(UsdGeom.Xformable(st.GetPrimAtPath("/Recon")).GetLocalTransformation())
    cams = sr.camera_centres(images) @ m[:3, :3] + m[3, :3]
    lo = cams[:, :2].min(0) - 6
    shape = tuple(np.ceil((cams[:, :2].max(0) + 6 - lo) / sr.CELL).astype(int))
    pts = sr.load_points(mesh, 0.02) @ m[:3, :3] + m[3, :3]  # USD row-vector layout, as level_scene.py writes it
    sites = pick(pts, lo, shape, n=int(sys.argv[6]) if len(sys.argv) > 6 else 6)
    json.dump([{k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in q.items()} for q in sites], open(out, "w"), indent=1)
    print(f"{len(sites)} wall sites", file=sys.stderr)
    print(" ".join(f"--robot h1 {q['start'][0]:.2f} {q['start'][1]:.2f} {np.degrees(np.arctan2(q['heading'][1], q['heading'][0])):.0f} 0 0 0"
                   for q in sites) + " " + " ".join(f"--stop {i} {q['goal'][0]:.2f} {q['goal'][1]:.2f} 0" for i, q in enumerate(sites)))


if __name__ == "__main__":
    demo() if len(sys.argv) == 1 else main()
