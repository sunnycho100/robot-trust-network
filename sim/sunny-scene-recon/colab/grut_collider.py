"""Put a COLMAP-frame mesh into the frame of a 3DGRUT NuRec USDZ, so add_mesh_to_usdz can pack it as the collider.

    python grut_collider.py SPLAT.usdz MESH.ply OUT.ply     (3dgrut venv: has pxr and open3d)

The NuRec exporter keeps Gaussian positions in the COLMAP frame and puts its pose-derived normalizing transform
on the Volume prim. add_mesh_to_usdz adds /World/mesh as the Volume's sibling with no transform, so the mesh
vertices get the Volume's local-to-world matrix baked in here. Also reports the floor normal in that frame,
since Isaac Sim needs Z up.
"""
import sys
import numpy as np
import open3d as o3d
from pxr import Usd, UsdGeom, UsdVol


def bake(vertices, m):
    """USD matrices act on row vectors: p' = [p, 1] @ M."""
    assert np.allclose(m[:, 3], [0, 0, 0, 1], atol=1e-6), f"projective matrix, not a rigid transform:\n{m}"
    return np.c_[vertices, np.ones(len(vertices))] @ m[:, :3]


def main():
    usdz, mesh_in, mesh_out = sys.argv[1:4]
    stage = Usd.Stage.Open(usdz)
    vol = next(p for p in stage.Traverse() if p.IsA(UsdVol.Volume))
    m = np.array(UsdGeom.Xformable(vol).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
    print(f"volume {vol.GetPath()}, stage upAxis {UsdGeom.GetStageUpAxis(stage)}, local-to-world:\n{np.round(m, 4)}")
    mesh = o3d.io.read_triangle_mesh(mesh_in)
    mesh.vertices = o3d.utility.Vector3dVector(bake(np.asarray(mesh.vertices), m))
    o3d.io.write_triangle_mesh(mesh_out, mesh)
    pc = o3d.geometry.PointCloud(mesh.vertices)
    (a, b, c, d), inl = pc.segment_plane(0.03, 3, 2000)
    print(f"largest plane normal ({a:.3f} {b:.3f} {c:.3f}), {len(inl)} of {len(pc.points)} vertices; wrote {mesh_out}")


if __name__ == "__main__":
    if len(sys.argv) == 1:  # self-check: a pure translation in USD's row-vector layout moves points by the last row
        m = np.eye(4); m[3, :3] = [1, 2, 3]
        assert np.allclose(bake(np.zeros((1, 3)), m), [[1, 2, 3]])
        print("ok")
    else:
        main()
