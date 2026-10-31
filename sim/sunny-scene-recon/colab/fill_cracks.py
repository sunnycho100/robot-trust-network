"""TSDF collider with its cracks filled from the Poisson mesh of the same depth.

    python fill_cracks.py TSDF.ply POISSON.ply OUT.ply [BAND_M]      (any venv with open3d, scipy)
    python fill_cracks.py --selfcheck

The TSDF mesh is the more accurate surface (closer to the laser scan) but leaves cracks that drop robots; on
delivery_area an H1 sank through a 16 cm one. Poisson triangles are added only where they lie more than 3 cm
from the TSDF surface and less than BAND_M (default 0.5 m) from it, so cracks and small holes close while the
TSDF stays the surface everywhere it exists and Poisson's blobs far from any seen geometry are not added.
"""
import sys
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def fill(tsdf, poisson, band=0.5, near=0.03):
    ref = np.asarray(tsdf.sample_points_uniformly(int(max(tsdf.get_surface_area() * 1e4, 1e4))).points)  # 1 cm
    v, f = np.asarray(poisson.vertices), np.asarray(poisson.triangles)
    d = cKDTree(ref).query(v[f].mean(1))[0]
    f = f[(d > near) & (d < band)]
    vt, ft = np.asarray(tsdf.vertices), np.asarray(tsdf.triangles)
    out = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.r_[vt, v]),
                                    o3d.utility.Vector3iVector(np.r_[ft, f + len(vt)]))
    return out.remove_unreferenced_vertices(), len(f)


def selfcheck():
    def plane(x0, x1):
        m = o3d.geometry.TriangleMesh.create_box(x1 - x0, 4, 0.001).translate((x0, 0, 0))
        return m.subdivide_midpoint(4)
    tsdf = plane(0, 2) + plane(2.16, 4)                      # 16 cm crack at x 2.00 to 2.16
    poisson = plane(0, 4) + plane(10, 12).translate((0, 0, 1))  # full floor, plus a blob 6 m away
    out, n = fill(tsdf, poisson)
    pts = np.asarray(out.sample_points_uniformly(200000).points)
    assert ((pts[:, 0] > 2.02) & (pts[:, 0] < 2.14)).any(), "crack not filled"
    assert pts[:, 0].max() < 4.6, "far Poisson blob was added"
    print(f"selfcheck ok: {n} Poisson triangles added")


if __name__ == "__main__":
    if sys.argv[1] == "--selfcheck":
        selfcheck()
        sys.exit()
    t, p = o3d.io.read_triangle_mesh(sys.argv[1]), o3d.io.read_triangle_mesh(sys.argv[2])
    out, n = fill(t, p, float(sys.argv[4]) if len(sys.argv) > 4 else 0.5)
    o3d.io.write_triangle_mesh(sys.argv[3], out)
    print(f"added {n} Poisson triangles to {len(t.triangles)} TSDF triangles -> {sys.argv[3]}")
