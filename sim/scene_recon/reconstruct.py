"""RGB-D frames -> colored triangle mesh, by TSDF fusion.

    python reconstruct.py FRAMES_DIR [--voxel 0.02]

FRAMES_DIR holds intrinsics.json, color/*.jpg|png, depth/*.png (uint16) and, if the camera poses
(extrinsics) are known, poses.txt with one camera-to-world 4x4 per line (16 numbers, row-major).
With poses.txt the frames are fused directly. Without it, the poses are estimated by chaining
frame-to-frame RGB-D odometry, written to poses.txt, and then fused the same way.
Writes FRAMES_DIR/mesh.ply.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import open3d as o3d


def load_frames(root):
    intr = json.loads((root / "intrinsics.json").read_text())
    colors = sorted((root / "color").glob("*.[jp][pn]g"))
    depths = sorted((root / "depth").glob("*.png"))
    if not colors or len(colors) != len(depths):
        raise SystemExit(f"need matching color and depth frames, got {len(colors)} and {len(depths)}")
    poses = None
    if (root / "poses.txt").exists():
        poses = np.loadtxt(root / "poses.txt").reshape(-1, 4, 4)
        if len(poses) != len(colors):
            raise SystemExit(f"poses.txt has {len(poses)} poses for {len(colors)} frames")
    return intr, colors, depths, poses


def fuse_known_poses(intr, colors, depths, poses, voxel, depth_max):
    """The classic path: every frame's extrinsics are known, so just integrate."""
    camera = o3d.camera.PinholeCameraIntrinsic(intr["width"], intr["height"], intr["fx"], intr["fy"], intr["cx"], intr["cy"])
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel, sdf_trunc=4 * voxel, color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    for color_path, depth_path, pose in zip(colors, depths, poses):
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.io.read_image(str(color_path)), o3d.io.read_image(str(depth_path)),
            depth_scale=1.0 / intr["depth_scale"], depth_trunc=depth_max, convert_rgb_to_intensity=False)
        volume.integrate(rgbd, camera, np.linalg.inv(pose))  # integrate wants world-to-camera
    return volume.extract_triangle_mesh()


def estimate_poses(intr, colors, depths, depth_max):
    """No extrinsics: chain frame-to-frame RGB-D odometry (photometric + geometric) into camera poses.

    ponytail: frame-to-frame drifts on long sequences; add loop closure (pose graph) if a room
    does not close up.
    """
    camera = o3d.camera.PinholeCameraIntrinsic(intr["width"], intr["height"], intr["fx"], intr["fy"], intr["cx"], intr["cy"])
    option = o3d.pipelines.odometry.OdometryOption(depth_max=depth_max)
    jacobian = o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm()

    def rgbd(i):
        return o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.io.read_image(str(colors[i])), o3d.io.read_image(str(depths[i])),
            depth_scale=1.0 / intr["depth_scale"], depth_trunc=depth_max, convert_rgb_to_intensity=True)

    poses, lost, previous = [np.eye(4)], 0, rgbd(0)
    for i in range(1, len(colors)):
        current = rgbd(i)
        ok, step, _ = o3d.pipelines.odometry.compute_rgbd_odometry(current, previous, camera, np.eye(4), jacobian, option)
        if not ok:  # keep the last pose rather than jumping; the frame still fuses, slightly off
            lost += 1
            step = np.eye(4)
        poses.append(poses[-1] @ step)
        previous = current
    if lost:
        print(f"odometry failed on {lost} of {len(colors) - 1} frame pairs")
    return np.array(poses)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("frames", type=Path)
    p.add_argument("--voxel", type=float, default=0.02, help="TSDF voxel size in meters")
    p.add_argument("--depth-max", type=float, default=5.0, help="ignore depth beyond this many meters")
    a = p.parse_args()
    intr, colors, depths, poses = load_frames(a.frames)
    start = time.time()
    how = "given poses"
    if poses is None:
        poses = estimate_poses(intr, colors, depths, a.depth_max)
        np.savetxt(a.frames / "poses.txt", poses.reshape(len(poses), 16))
        how = "poses estimated by RGB-D odometry, saved to poses.txt"
    mesh = fuse_known_poses(intr, colors, depths, poses, a.voxel, a.depth_max)
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(a.frames / "mesh.ply"), mesh)
    print(f"{len(colors)} frames ({how}) -> {len(mesh.triangles)} triangles in {time.time() - start:.0f} s: "
          f"{a.frames / 'mesh.ply'}")


if __name__ == "__main__":
    main()
