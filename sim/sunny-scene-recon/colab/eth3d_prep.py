"""ETH3D scene (posed DSLR images + COLMAP dense points) -> a frames folder for build_scene.py.

    python eth3d_prep.py ETH3D_SCENE_DIR WORK_DIR [CALIBRATION]   (default dslr_calibration_undistorted)

Expects WORK_DIR/fused.ply from COLMAP stereo_fusion. Writes WORK_DIR/frames/mesh.ply (Poisson
mesh with colors, low-support faces trimmed), WORK_DIR/frames/poses.txt (camera-to-world from
ETH3D's images.txt, for the up direction) and WORK_DIR/mesh_points.ply (points sampled from the
mesh at 1 cm spacing, for scoring against the laser scan in ETH3D's own frame; sparser sampling caps the
2 cm score on a large scene).
"""
import sys
from pathlib import Path
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation


def camera_to_world(images_txt):
    poses = []
    # keep empty POINTS2D lines (a known-pose model has them) so [::2] lands on image lines
    lines = [l for l in Path(images_txt).read_text().rstrip("\n").splitlines() if not l.startswith("#")]
    for line in lines[::2]:  # each image line is followed by its 2D points line
        qw, qx, qy, qz, tx, ty, tz = map(float, line.split()[1:8])
        R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()  # world -> camera
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R.T, -R.T @ np.array([tx, ty, tz])
        poses.append(T)
    return np.array(poses)


def main():
    scene, work = Path(sys.argv[1]), Path(sys.argv[2])
    calibration = sys.argv[3] if len(sys.argv) > 3 else "dslr_calibration_undistorted"
    frames = work / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    poses = camera_to_world(scene / calibration / "images.txt")
    np.savetxt(frames / "poses.txt", poses.reshape(len(poses), 16))

    points = o3d.io.read_point_cloud(str(work / "fused.ply"))
    mesh, density = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(points, depth=10)
    density = np.asarray(density)
    mesh.remove_vertices_by_mask(density < np.quantile(density, 0.12))  # Poisson invents surface where no points were
    mesh = mesh.crop(points.get_axis_aligned_bounding_box())
    clusters, sizes, _ = mesh.cluster_connected_triangles()  # floating fragments
    clusters, sizes = np.asarray(clusters), np.asarray(sizes)
    mesh.remove_triangles_by_mask(sizes[clusters] < 0.005 * len(clusters))
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(frames / "mesh.ply"), mesh)
    n = int(min(mesh.get_surface_area() / 0.01 ** 2, 40e6))
    sample = np.asarray(mesh.sample_points_uniformly(n).points, dtype=np.float32)
    o3d.t.io.write_point_cloud(str(work / "mesh_points.ply"), o3d.t.geometry.PointCloud(sample))  # float32: the ETH3D evaluator reads float x y z
    print(f"{len(points.points)} fused points -> mesh with {len(mesh.triangles)} triangles; {len(poses)} camera poses")


if __name__ == "__main__":
    main()
