"""Hilti SLAM Challenge 2022 construction site: a realistic handheld walk-through scored against the site's laser scan.

    python hilti_run.py /content/hilti exp04_construction_upper_level WORK_DIR

Stages (each is skipped when its output exists):
1. LiDAR odometry on /hesai/pandar with KISS-ICP, deskewed from each point's own timestamp (one 10 Hz sweep
   moves 12 cm at walking speed), and a 2 cm map from every second deskewed scan. Poses are at mid-sweep.
2. The terrestrial laser scan (e57) merged at 2 cm; our map aligned to it from the surveyed control points
   (ground_truth/*.txt, positions of the rig's measurement tip), then refined by ICP.
3. Frames from the front stereo pair (cam0, cam1) at 5 Hz, fisheye undistorted, named by time so both cameras
   interleave for COLMAP's sequential matcher, posed from the LiDAR trajectory and the factory calibration
   (quaternions are x, y, z, w: the IMU's identity is written [0, 0, 0, 1]). A small camera to LiDAR time
   offset is checked by triangulating with each and keeping the lowest reprojection error.
4. COLMAP: features, point_triangulator on the fixed poses, PatchMatch, fusion (run_hilti.sh does this).
5. score: accuracy, completeness and F1 at 2 to 50 cm against the laser scan, within 5 m of the walked path.
"""
import json, sqlite3, sys
from pathlib import Path
import cv2
import numpy as np
import open3d as o3d
import yaml
from rosbags.highlevel import AnyReader
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lidar_poses import icp, interpolate  # noqa: E402

CAMS = ("cam0", "cam1")
STRIDE = 8  # 40 Hz cameras -> 5 Hz


def pose(q_xyzw, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = Rotation.from_quat(q_xyzw).as_matrix(), t
    return T


def read_hesai(bag, near=1.0, far=30.0):
    """Scans with each point's time, for deskewing."""
    kinds = {6: "<u4", 7: "<f4", 8: "<f8"}  # PointField datatypes
    with AnyReader([bag]) as reader:
        conn = [c for c in reader.connections if c.topic == "/hesai/pandar"]
        for c, _, raw in reader.messages(connections=conn):
            msg = reader.deserialize(raw, c.msgtype)
            f = {x.name: x for x in msg.fields}
            dtype = np.dtype({"names": ["x", "y", "z", "t"], "formats": ["<f4", "<f4", "<f4", kinds[f["timestamp"].datatype]],
                              "offsets": [f["x"].offset, f["y"].offset, f["z"].offset, f["timestamp"].offset],
                              "itemsize": msg.point_step})
            a = np.frombuffer(msg.data, dtype=dtype)
            xyz = np.stack([a["x"], a["y"], a["z"]], axis=1).astype(np.float64)
            r = np.linalg.norm(xyz, axis=1)
            keep = np.isfinite(r) & (r > near) & (r < far)  # near points are the operator and the rig
            yield xyz[keep], a["t"][keep].astype(np.float64)


def odometry(root, seq, W):
    if (W / "lidar_poses.npz").exists():
        z = np.load(W / "lidar_poses.npz")
        return z["times"], z["poses"]
    from kiss_icp.config import KISSConfig
    from kiss_icp.kiss_icp import KissICP
    config = KISSConfig()
    config.data.deskew, config.data.max_range, config.data.min_range = True, 30.0, 1.0
    config.mapping.voxel_size = 0.3  # KISS-ICP's default, max_range / 100
    odom = KissICP(config=config)
    times, poses, lidar_map = [], [], o3d.geometry.PointCloud()
    for k, (xyz, t) in enumerate(read_hesai(root / "rosbags" / f"{seq}.bag")):
        frame, _ = odom.register_frame(xyz, (t - t.min()) / max(t.max() - t.min(), 1e-9))
        T = np.asarray(odom.last_pose).copy()
        times.append(int((t.min() + t.max()) / 2 * 1e9))  # deskewed to the middle of the sweep
        poses.append(T)
        if k % 2 == 0:
            lidar_map += o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(frame))).transform(T)
        if k % 40 == 0:
            lidar_map = lidar_map.voxel_down_sample(0.02)
    times, poses = np.array(times), np.array(poses)
    np.savez(W / "lidar_poses.npz", times=times, poses=poses)
    o3d.io.write_point_cloud(str(W / "lidar_map.ply"), lidar_map.voxel_down_sample(0.02))
    print(f"KISS-ICP: {len(poses)} scans, path {np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1).sum():.1f} m", flush=True)
    return times, poses


def laser_scan(root, W):
    if (W / "tls.ply").exists():
        return o3d.io.read_point_cloud(str(W / "tls.ply"))
    import pye57
    e57 = pye57.E57(str(next((root / "laser_scans").glob("construction_site*.e57"))))
    tls = o3d.geometry.PointCloud()
    for i in range(e57.scan_count):
        d = e57.read_scan(i, ignore_missing_fields=True, transform=True)
        tls += o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.stack([d["cartesianX"], d["cartesianY"], d["cartesianZ"]], 1)))
        tls = tls.voxel_down_sample(0.02)
    tls.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    o3d.io.write_point_cloud(str(W / "tls.ply"), tls)
    return tls


def align(root, seq, W, times, poses, T_imu_lidar, tls):
    """Rigid map -> laser scan: control points for the start, ICP for the rest."""
    if (W / "T_tls_map.txt").exists():
        return np.loadtxt(W / "T_tls_map.txt")
    calib = yaml.safe_load((root / "calibration" / "calibration_files" / "lidar_calibration.yaml").read_text())
    tip = np.array(calib["sensors"]["measurement_tip"]["extrinsics"]["translation"])
    gt = np.loadtxt(root / "ground_truth" / f"{seq}.txt")
    ours = np.array([(interpolate(times, poses, t * 1e9) @ np.linalg.inv(T_imu_lidar) @ np.r_[tip, 1])[:3] for t in gt[:, 0]])
    theirs = gt[:, 1:4]
    mu_o, mu_t = ours.mean(0), theirs.mean(0)
    U, _, Vt = np.linalg.svd((theirs - mu_t).T @ (ours - mu_o))
    R = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    T0 = np.eye(4)
    T0[:3, :3], T0[:3, 3] = R, mu_t - R @ mu_o
    print("control points before ICP (cm):", (np.linalg.norm(ours @ R.T + T0[:3, 3] - theirs, axis=1) * 100).round(1))
    lidar_map = o3d.io.read_point_cloud(str(W / "lidar_map.ply")).voxel_down_sample(0.05)
    lidar_map.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.15, max_nn=30))
    # only the part of the site we walked (the scan covers other floors too), and a tight start, so ICP can't slide
    path = (np.c_[poses[:, :3, 3], np.ones(len(poses))] @ T0.T)[:, :3]
    g = np.asarray(tls.points)
    near = tls.select_by_index(np.nonzero(cKDTree(path).query(g, distance_upper_bound=8.0)[0] < 8.0)[0])
    T, result = icp(lidar_map, near, T0, [0.3, 0.1, 0.05])
    print(f"ICP to the laser scan: fitness {result.fitness:.3f}, inlier RMSE {result.inlier_rmse * 100:.1f} cm")
    before = np.linalg.norm(ours @ R.T + T0[:3, 3] - theirs, axis=1)
    after = np.linalg.norm(np.array([(T @ np.r_[p, 1])[:3] for p in ours]) - theirs, axis=1)
    print("control points after ICP (cm):", (after * 100).round(1))
    if np.median(after) > np.median(before):
        print("ICP moved the control points away, keeping the control-point fit")
        T = T0
    np.savetxt(W / "T_tls_map.txt", T)
    return T


def frames(root, seq, W, times):
    """Undistorted frames; the COLMAP model with the database's ids is written after feature extraction."""
    out = W / "images"
    if (W / "frame_poses.json").exists():
        return json.loads((W / "frame_poses.json").read_text())
    chain = yaml.safe_load((root / "calibration" / "calibration_files" / "calib_3_cam0-1-camchain-imucam.yaml").read_text())
    maps, info = {}, {"cameras": {}, "images": []}
    for cam in CAMS:
        c = chain[cam]
        fu, fv, pu, pv = c["intrinsics"]
        K, D = np.array([[fu, 0, pu], [0, fv, pv], [0, 0, 1]]), np.array(c["distortion_coeffs"])
        size = tuple(c["resolution"])
        Kn = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, size, np.eye(3), balance=0.0)
        maps[cam] = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), Kn, size, cv2.CV_16SC2)
        info["cameras"][cam] = {"K": Kn.tolist(), "size": size, "T_cam_imu": c["T_cam_imu"]}
    out.mkdir(parents=True, exist_ok=True)
    with AnyReader([root / "rosbags" / f"{seq}.bag"]) as reader:
        for cam in CAMS:
            conn = [c for c in reader.connections if c.topic == f"/alphasense/{cam}/image_raw"]
            for k, (c, _, raw) in enumerate(reader.messages(connections=conn)):
                if k % STRIDE:
                    continue
                msg = reader.deserialize(raw, c.msgtype)
                t = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
                if not times[0] <= t <= times[-1]:
                    continue
                img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width)
                name = f"{t}_{cam}.png"
                cv2.imwrite(str(out / name), cv2.remap(img, *maps[cam], cv2.INTER_LINEAR))
                info["images"].append({"name": name, "cam": cam, "t": int(t)})
    (W / "frame_poses.json").write_text(json.dumps(info))
    return info


def T_imu_lidar(root):
    e = yaml.safe_load((root / "calibration" / "calibration_files" / "lidar_calibration.yaml").read_text())["sensors"]["PandarXT-32"]["extrinsics"]
    return pose(e["quaternion"], e["translation"])


def write_model(W, info, times, poses, T_imu_lidar, offset_ns, name):
    """COLMAP text model with the database's own image and camera ids, cameras fixed at the calibrated pinhole."""
    db = sqlite3.connect(W / "db.db")
    ids = {n: (i, c) for i, n, c in db.execute("SELECT image_id, name, camera_id FROM images")}
    model = W / f"sparse_{name}"
    model.mkdir(exist_ok=True)
    cams = {}
    for img in info["images"]:
        cams[ids[img["name"]][1]] = img["cam"]
    with open(model / "cameras.txt", "w") as f:
        for cid, cam in cams.items():
            K, (w, h) = np.array(info["cameras"][cam]["K"]), info["cameras"][cam]["size"]
            f.write(f"{cid} PINHOLE {w} {h} {K[0, 0]} {K[1, 1]} {K[0, 2]} {K[1, 2]}\n")
            db.execute("UPDATE cameras SET model=1, width=?, height=?, params=?, prior_focal_length=1 WHERE camera_id=?",
                       (w, h, np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=np.float64).tobytes(), cid))
    with open(model / "images.txt", "w") as f:
        for img in info["images"]:
            T_cam_imu = np.array(info["cameras"][img["cam"]]["T_cam_imu"])
            c2w = interpolate(times, poses, img["t"] + offset_ns) @ np.linalg.inv(T_imu_lidar) @ np.linalg.inv(T_cam_imu)
            w2c = np.linalg.inv(c2w)
            qx, qy, qz, qw = Rotation.from_matrix(w2c[:3, :3]).as_quat()
            i, c = ids[img["name"]]
            f.write(f"{i} {qw} {qx} {qy} {qz} {w2c[0, 3]} {w2c[1, 3]} {w2c[2, 3]} {c} {img['name']}\n\n")
    (model / "points3D.txt").write_text("")
    db.commit()
    return model


def score(W, cloud_path, T_tls_map, tls, label):
    """Accuracy and completeness within 5 m of the walked path, both clouds at 1 cm."""
    poses = np.load(W / "lidar_poses.npz")["poses"]
    path = (np.c_[poses[::10, :3, 3], np.ones(len(poses[::10]))] @ T_tls_map.T)[:, :3]
    near = lambda p: cKDTree(path).query(p, distance_upper_bound=5.0)[0] < 5.0
    rec = o3d.io.read_point_cloud(str(cloud_path)).transform(T_tls_map).voxel_down_sample(0.01)
    r, g = np.asarray(rec.points), np.asarray(tls.points)
    r, g = r[near(r)], g[near(g)]
    d_rg, d_gr = cKDTree(g).query(r)[0], cKDTree(r).query(g)[0]
    out = {}
    print(f"{label}: median distance to the laser scan {np.median(d_rg) * 100:.1f} cm, laser scan to it {np.median(d_gr) * 100:.1f} cm")
    for tau in (0.02, 0.05, 0.10, 0.20, 0.50):  # alignment is only as good as the control points (~10 cm)
        acc, comp = (d_rg < tau).mean(), (d_gr < tau).mean()
        out[f"{int(tau * 100)}cm"] = {"accuracy": round(float(acc), 3), "completeness": round(float(comp), 3),
                                      "f1": round(float(2 * acc * comp / max(acc + comp, 1e-9)), 3)}
    print(label, json.dumps(out), flush=True)
    return out


def main():
    root, seq, W = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    stage = sys.argv[4] if len(sys.argv) > 4 else "prepare"
    W.mkdir(parents=True, exist_ok=True)
    times, poses = odometry(root, seq, W)
    tls = laser_scan(root, W)
    T_il = T_imu_lidar(root)
    if stage == "align":  # redo only the alignment
        (W / "T_tls_map.txt").unlink(missing_ok=True)
        align(root, seq, W, times, poses, T_il, tls)
    elif stage == "prepare":
        align(root, seq, W, times, poses, T_il, tls)
        frames(root, seq, W, times)
    elif stage == "model":  # after feature extraction: a few time offsets, triangulation picks
        info = json.loads((W / "frame_poses.json").read_text())
        for offset_ms in (-20, 0, 20):
            print("wrote", write_model(W, info, times, poses, T_il, offset_ms * 10**6, f"{offset_ms:+d}ms"))
    elif stage == "score":
        T_tls_map = np.loadtxt(W / "T_tls_map.txt")
        results = {"lidar_map": score(W, W / "lidar_map.ply", T_tls_map, tls, "LiDAR map")}
        if (W / "fused.ply").exists():
            results["camera_mvs"] = score(W, W / "fused.ply", T_tls_map, tls, "camera MVS")
        (W / "scores.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
