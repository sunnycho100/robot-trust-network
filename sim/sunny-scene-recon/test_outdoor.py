"""Self-check for build_scene.py --outdoor on a synthetic site: 5% slope, a 15 cm curb, a 1.3 m tall crate,
and a canopy floating 1.55 m above the ground (its underside must stay open).

    python test_outdoor.py   (needs CSF and coacd; runs build_scene.py and check_collisions.py in a temp dir)
"""
import itertools, json, subprocess, sys, tempfile
from pathlib import Path
import mujoco
import numpy as np
import open3d as o3d

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))


def check_greedy_boxes():
    """Merged boxes cover a solid exactly once, and a plain block stays one box."""
    from scipy import ndimage
    from build_scene import greedy_boxes
    rng = np.random.default_rng(0)
    for _ in range(20):
        solid = ndimage.binary_closing(rng.random((12, 10, 8)) > 0.55)
        cover = np.zeros(solid.shape, dtype=int)
        for i0, j0, k0, i1, j1, k1 in greedy_boxes(solid):
            cover[i0:i1, j0:j1, k0:k1] += 1
        assert (cover == solid).all()
    block = np.zeros((20, 20, 10), dtype=bool)
    block[2:18, 3:9, 0:6] = True
    assert len(greedy_boxes(block)) == 1


def site():
    n = 201  # 10 x 10 m ground grid at 5 cm
    x, y = np.meshgrid(np.linspace(-5, 5, n), np.linspace(-5, 5, n))
    z = 0.05 * y + np.where(x > 2, 0.15, 0.0)  # slope along y, curb along x = 2
    v = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
    i = np.arange(n * n).reshape(n, n)
    a, b, c, d = i[:-1, :-1].ravel(), i[:-1, 1:].ravel(), i[1:, 1:].ravel(), i[1:, :-1].ravel()
    ground = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(np.r_[np.c_[a, b, c], np.c_[a, c, d]]))
    crate = o3d.geometry.TriangleMesh.create_box(1, 1, 1.3).subdivide_midpoint(4)  # both IoU slices cut through it
    crate.translate([-2.5, -2.5, 0.05 * -2.5 - 0.02])
    canopy = o3d.geometry.TriangleMesh.create_box(2, 2, 0.1).subdivide_midpoint(3)
    canopy.translate([-1, 2, 1.7])  # over (0, 3), where the ground is at 0.15
    mesh = ground + crate + canopy
    mesh.paint_uniform_color([0.6, 0.6, 0.6])
    return mesh


def main():
    check_greedy_boxes()
    with tempfile.TemporaryDirectory() as tmp:
        frames = Path(tmp)
        o3d.io.write_triangle_mesh(str(frames / "mesh.ply"), site())
        # a perfect laser scan, except under a roof 4.5 m up on pillars over half the site, where the scanner never saw
        # the ground: only the roof and the pillars are there, and the roof is no ground
        laser = np.asarray(site().sample_points_uniformly(400000).points)
        roof = o3d.geometry.TriangleMesh.create_box(6, 10, 0.1).translate([-1, -5, 4.5])
        for x, y in itertools.product((1.5, 4), (-3.5, -0.5, 2.5)):
            roof += o3d.geometry.TriangleMesh.create_box(0.3, 0.3, 4.5).translate([x, y, 0])
        laser = np.vstack([laser[laser[:, 0] < -1], roof.sample_points_uniformly(150000).points])
        o3d.io.write_point_cloud(str(frames / "laser.ply"), o3d.geometry.PointCloud(o3d.utility.Vector3dVector(laser)))
        pose = np.eye(4)
        pose[:3, :3] = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]  # camera looking along +y, its -y axis is world up
        pose[2, 3] = 1.5
        np.savetxt(frames / "poses.txt", pose.reshape(1, 16))
        for cmd in (["build_scene.py", str(frames), "--outdoor", "--crop-height", "2.2"], ["check_collisions.py", str(frames / "mujoco"), str(frames / "laser.ply")]):
            subprocess.run([sys.executable, str(HERE / cmd[0]), *cmd[1:]], check=True)
        info = json.loads((frames / "mujoco" / "scene_info.json").read_text())
        check = json.loads((frames / "mujoco" / "collision_check.json").read_text())
        assert len(info["object_boxes"]) >= 1, "no object boxes"
        assert check["ground"]["rms_cm"] < 3, check["ground"]
        assert check["iou"]["0.5"] > 0.3 and check["iou"]["1.0"] > 0.3, check["iou"]
        assert all(v["mesh_within_10cm"] > 0.8 and v["proxy_within_10cm"] > 0.8 for v in check["surface"].values()), check["surface"]
        g = check["ground_vs_laser"]
        assert g["scanned ground"]["p95_cm"] < 3, g
        assert g["scanned ground"]["cells"] * 0.05 ** 2 > 30 and g["scanned ground without laser ground_m2"] > 50, g  # floor in view scored, roof not
        assert max(g["scanned ground over 0.5 m off the laser floor_m2"].values()) < 1, g
        assert check["ground"]["cells"] * 0.05 ** 2 > 90, check["ground"]  # the raised terrace past the curb is ground too
        model = mujoco.MjModel.from_xml_path(str(frames / "mujoco" / "scene.xml"))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        T = np.array(info["transform_from_scan"])
        start = T[:3, :3] @ [0, 3, 1.6] + T[:3, 3]  # just under the canopy, in the leveled scene frame
        hit = np.zeros(1, dtype=np.int32)
        mujoco.mj_ray(model, data, start, np.array([0, 0, -1.0]), np.array([0, 0, 0, 1, 0, 0], dtype=np.uint8), 1, -1, hit)
        assert mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, hit[0]) == "ground", "space under the canopy is filled"
        print("outdoor self-check ok:", len(info["object_boxes"]), "object boxes,", check["ground"], check["iou"])


if __name__ == "__main__":
    main()
