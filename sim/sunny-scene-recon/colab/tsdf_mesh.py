"""Depth maps + known poses -> TSDF mesh, the alternative collider mesher to eth3d_prep.py's Poisson.

    python tsdf_mesh.py CAPTURE OUT.ply [VOXEL_M]

CAPTURE is a capture_walkthrough.py folder: sparse/0/{cameras,images}.txt (PINHOLE), images/, depth/ (uint16 mm,
z-depth), test.txt (held-out frames, skipped). Poisson closes surfaces through unseen space; TSDF only carves
and fills along observed rays, so free floor stays free.
"""
import sys
import numpy as np
import open3d as o3d
from score_robot import read_images


def integrate(frames, intr, voxel, trunc_m=12.0):
    """frames: iterable of (color o3d.Image, depth o3d.Image in mm, 4x4 world->camera)."""
    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel, sdf_trunc=4 * voxel, color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    for color, depth, ext in frames:
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_scale=1000.0, depth_trunc=trunc_m, convert_rgb_to_intensity=False)
        vol.integrate(rgbd, intr, ext)
    return vol.extract_triangle_mesh()


def main():
    cap, out = sys.argv[1], sys.argv[2]
    voxel = float(sys.argv[3]) if len(sys.argv) > 3 else 0.02
    w, h, fx, fy, cx, cy = map(float, open(f"{cap}/sparse/0/cameras.txt").read().split()[2:8])
    intr = o3d.camera.PinholeCameraIntrinsic(int(w), int(h), fx, fy, cx, cy)
    test = set(open(f"{cap}/test.txt").read().split())

    def frames():
        for name, r, t in read_images(f"{cap}/sparse/0/images.txt"):
            if name in test:
                continue
            ext = np.eye(4)
            ext[:3, :3], ext[:3, 3] = r, t  # camera = R world + t
            yield (o3d.io.read_image(f"{cap}/images/{name}"), o3d.io.read_image(f"{cap}/depth/{name}"), ext)

    mesh = integrate(frames(), intr, voxel)
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(out, mesh)
    print(f"TSDF {voxel} m: {len(mesh.triangles)} triangles -> {out}")


if __name__ == "__main__":
    if sys.argv[1:] == ["test"]:
        # a camera 1 m above a flat floor looking straight down: the mesh must be a flat patch at z = 0
        w = h = 64
        intr = o3d.camera.PinholeCameraIntrinsic(w, h, 50.0, 50.0, 32.0, 32.0)
        depth = o3d.geometry.Image(np.full((h, w), 1000, np.uint16))
        color = o3d.geometry.Image(np.zeros((h, w, 3), np.uint8))
        r = np.diag([1.0, -1.0, -1.0])  # camera z looks down world -z
        frames = []
        for dx in (-0.1, 0.0, 0.1):
            c = np.array([dx, 0.0, 1.0])
            ext = np.eye(4)
            ext[:3, :3], ext[:3, 3] = r, -r @ c
            frames.append((color, depth, ext))
        v = np.asarray(integrate(frames, intr, 0.02).vertices)
        assert len(v) > 100 and np.abs(v[:, 2]).max() < 0.02, (len(v), np.abs(v[:, 2]).max())
        print("ok", len(v), "vertices, max |z|", round(float(np.abs(v[:, 2]).max()), 4))
    else:
        main()
