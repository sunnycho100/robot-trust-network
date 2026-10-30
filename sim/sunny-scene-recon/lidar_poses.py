"""Camera poses from a LiDAR in the same bag: LiDAR odometry plus a camera-to-LiDAR extrinsic.

    python lidar_poses.py BAG FRAMES_DIR [--lidar /ouster/points]

Use this when the camera alone drifts (bare walls give RGB-D odometry nothing to hold on to).
1. LiDAR odometry: each scan is ICP-registered (point-to-plane) against a map of the scans so far.
2. Extrinsic: the camera-to-LiDAR transform is estimated by aligning a camera depth cloud to the
   LiDAR map, starting from a few yaw guesses, so it works when the bag's TF tree is missing or
   placeholder. Pass --extrinsic file.json (a 4x4 "T_lidar_camera") to use a known calibration.
3. Each camera frame gets the LiDAR pose interpolated at its timestamp, times the extrinsic.
Writes FRAMES_DIR/poses.txt (camera-to-world, for reconstruct.py) and extrinsic.json.
"""
import argparse, json
from pathlib import Path
import numpy as np
import open3d as o3d
from rosbags.highlevel import AnyReader
from scipy.spatial.transform import Rotation, Slerp

REG = o3d.pipelines.registration


def read_scans(bag, topic, near=0.6, far=30.0):
    times, scans = [], []
    with AnyReader([bag]) as reader:
        connection = [c for c in reader.connections if c.topic == topic]
        if not connection:
            raise SystemExit(f"no {topic} in the bag")
        for c, _, raw in reader.messages(connections=connection):
            msg = reader.deserialize(raw, c.msgtype)
            offsets = {f.name: f.offset for f in msg.fields}
            dtype = np.dtype({"names": ["x", "y", "z"], "formats": ["<f4"] * 3,
                              "offsets": [offsets["x"], offsets["y"], offsets["z"]], "itemsize": msg.point_step})
            xyz = np.frombuffer(msg.data, dtype=dtype)
            xyz = np.stack([xyz["x"], xyz["y"], xyz["z"]], axis=1).astype(np.float64)
            r = np.linalg.norm(xyz, axis=1)
            times.append(msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec)
            scans.append(xyz[(r > near) & (r < far)])  # near points are the operator and the rig
    return np.array(times), scans


def cloud(points, voxel):
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)).voxel_down_sample(voxel)
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=3 * voxel, max_nn=30))
    return pc


def icp(source, target, init, distances):
    T, result = init, None
    for d in distances:  # coarse to fine
        result = REG.registration_icp(source, target, d, T, REG.TransformationEstimationPointToPlane())
        T = result.transformation
    return T, result


def lidar_odometry(scans, voxel):
    """Scan-to-map ICP with a constant-velocity guess. ponytail: no deskew of the spinning scan,
    fine for a hand-carried rig moving slowly; deskew with the per-point time if it smears."""
    world = cloud(scans[0], voxel)
    poses, step = [np.eye(4)], np.eye(4)
    for scan in scans[1:]:
        T, result = icp(cloud(scan, voxel), world, poses[-1] @ step, [6 * voxel, 2 * voxel])
        step = np.linalg.inv(poses[-1]) @ T
        poses.append(T)
        world += cloud(scan, voxel).transform(T)
        world = world.voxel_down_sample(voxel)
        world.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=3 * voxel, max_nn=30))
    return np.array(poses), world


def camera_cloud(frames, index, intr):
    depth = o3d.io.read_image(str(frames / "depth" / f"{index:06d}.png"))
    camera = o3d.camera.PinholeCameraIntrinsic(intr["width"], intr["height"], intr["fx"], intr["fy"], intr["cx"], intr["cy"])
    return o3d.geometry.PointCloud.create_from_depth_image(depth, camera, depth_scale=1.0 / intr["depth_scale"], depth_trunc=4.0)


def estimate_extrinsic(frames, intr, cam_index, lidar_pose, world, voxel):
    """T_lidar_camera by aligning one camera depth cloud to the LiDAR map, from several guesses."""
    optical = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=float)  # camera z forward = sensor x forward
    source = camera_cloud(frames, cam_index, intr).voxel_down_sample(voxel / 2)
    source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=3 * voxel, max_nn=30))
    best = None
    for yaw in (0, 90, 180, 270):
        for height in (-0.15, 0.0, 0.15):
            guess = np.eye(4)
            guess[:3, :3] = Rotation.from_euler("z", yaw, degrees=True).as_matrix() @ optical
            guess[2, 3] = height
            T, result = icp(source, world, lidar_pose @ guess, [0.3, 0.1, 0.04])
            score = (result.fitness, -result.inlier_rmse)
            if best is None or score > best[0]:
                best = (score, np.linalg.inv(lidar_pose) @ T, result)
    return best[1], best[2]


def interpolate(times, poses, t):
    i = int(np.clip(np.searchsorted(times, t) - 1, 0, len(times) - 2))
    a = float(np.clip((t - times[i]) / (times[i + 1] - times[i]), 0.0, 1.0))
    rot = Slerp([0, 1], Rotation.from_matrix(poses[i:i + 2, :3, :3]))(a).as_matrix()
    out = np.eye(4)
    out[:3, :3] = rot
    out[:3, 3] = (1 - a) * poses[i, :3, 3] + a * poses[i + 1, :3, 3]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("bag", type=Path)
    p.add_argument("frames", type=Path)
    p.add_argument("--lidar", default="/ouster/points")
    p.add_argument("--voxel", type=float, default=0.05)
    p.add_argument("--extrinsic", type=Path, help="JSON with a 4x4 T_lidar_camera, skips the estimate")
    a = p.parse_args()
    intr = json.loads((a.frames / "intrinsics.json").read_text())
    stamps = np.loadtxt(a.frames / "timestamps.txt", dtype=np.int64)

    times, scans = read_scans(a.bag, a.lidar)
    lidar, world = lidar_odometry(scans, a.voxel)
    travelled = np.linalg.norm(np.diff(lidar[:, :3, 3], axis=0), axis=1).sum()
    print(f"lidar odometry: {len(scans)} scans, {travelled:.2f} m travelled, map {len(world.points)} points")

    if a.extrinsic:
        T_lc, how = np.array(json.loads(a.extrinsic.read_text())["T_lidar_camera"]), f"from {a.extrinsic}"
    else:
        cam_index = int(np.argmin(np.abs(stamps - times[len(times) // 2])))  # a frame mid-sequence
        T_lc, result = estimate_extrinsic(a.frames, intr, cam_index, interpolate(times, lidar, stamps[cam_index]), world, a.voxel)
        how = f"estimated by ICP on frame {cam_index}: fitness {result.fitness:.2f}, rmse {result.inlier_rmse * 100:.1f} cm"
    print(f"camera-to-lidar extrinsic {how}; translation {np.round(T_lc[:3, 3], 3).tolist()} m")

    poses = np.array([interpolate(times, lidar, t) @ T_lc for t in stamps])
    np.savetxt(a.frames / "poses.txt", poses.reshape(len(poses), 16))
    (a.frames / "extrinsic.json").write_text(json.dumps({"T_lidar_camera": T_lc.tolist(), "how": how}, indent=2))
    o3d.io.write_point_cloud(str(a.frames / "lidar_map.ply"), world)
    print(f"{len(poses)} camera poses -> {a.frames / 'poses.txt'}")


if __name__ == "__main__":
    main()
