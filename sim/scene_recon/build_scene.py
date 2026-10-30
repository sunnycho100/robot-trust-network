"""Reconstructed mesh -> MuJoCo scene that a humanoid can walk in.

    python build_scene.py FRAMES_DIR [--crop-height 2.0]

Reads FRAMES_DIR/mesh.ply and poses.txt (camera-to-world, used for the up direction) and writes
FRAMES_DIR/mujoco/: scene.xml, scene.obj with a baked color texture, collision hulls, and
scene_info.json (bounds, a free spawn point, collision counts).

- The floor is found as the lowest large plane facing the cameras' up direction; it becomes z = 0,
  and the dominant wall is turned to face the x axis.
- Everything above --crop-height is cut away (the ceiling), so the room can be seen from above.
- Visuals and physics are separate. The textured mesh is visual only (MuJoCo would collide with its
  convex hull, which fills the room). Collisions are a plane for the floor, a thin box per wall
  plane, and CoACD convex pieces for the remaining clutter (pipes, machines, debris). A heightfield
  would turn tables and pipes into solid columns down to the floor, so it is not used indoors.
"""
import argparse, json
from pathlib import Path
import cv2
import numpy as np
import open3d as o3d
from scipy import ndimage


def rotation_between(a, b):
    """Smallest rotation turning unit vector a onto unit vector b."""
    v, c = np.cross(a, b), float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def find_planes(points, count=8):
    planes, rest = [], o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    for _ in range(count):
        if len(rest.points) < 1000:
            break
        (a, b, c, d), inliers = rest.segment_plane(distance_threshold=0.02, ransac_n=3, num_iterations=2000)
        planes.append((np.array([a, b, c]), d, np.asarray(rest.points)[inliers]))
        rest = rest.select_by_index(inliers, invert=True)
    return planes


def level(mesh, poses, extra_points=None):
    """Rigid transform putting the floor at z = 0, up = +z, the main wall along x.

    extra_points (a LiDAR map, say) help when the camera rarely looked down at the floor.
    """
    up = -poses[:, :3, 1].mean(axis=0)  # cameras use y down, so their mean -y is roughly up
    up /= np.linalg.norm(up)
    points = np.asarray(mesh.sample_points_uniformly(200_000).points)
    if extra_points is not None:
        points = np.vstack([points, extra_points])
    planes = find_planes(points)  # largest first: mostly walls
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    flat = np.abs(np.asarray(pc.normals) @ up) > 0.9  # surfaces facing up or down: floor, ceiling, tables
    floors = []
    for n, d, pts in find_planes(points[flat], count=4):
        n = n / np.linalg.norm(n)
        if abs(n @ up) > np.cos(np.radians(15)) and len(pts) > 0.1 * flat.sum():
            floors.append((n if n @ up > 0 else -n, pts))
    if not floors:
        raise SystemExit("no floor-like plane found; check poses.txt or the up direction")
    normal, floor_pts = min(floors, key=lambda f: (f[1] @ up).mean())  # lowest of the horizontal planes
    R = rotation_between(normal / np.linalg.norm(normal), np.array([0.0, 0.0, 1.0]))
    walls = [R @ (n / np.linalg.norm(n)) for n, d, pts in planes if abs((n / np.linalg.norm(n)) @ normal) < np.sin(np.radians(20))]
    if walls:  # turn the biggest wall (planes come out largest first) to face the x axis
        yaw = -np.arctan2(walls[0][1], walls[0][0])
        R = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]) @ R
    T = np.eye(4)
    T[:3, :3] = R
    floor = floor_pts @ R.T
    T[:3, 3] = [-floor[:, 0].mean(), -floor[:, 1].mean(), -np.median(floor[:, 2])]
    return T, len(walls)


def crop(mesh, z_min, z_max):
    z = np.asarray(mesh.vertices)[:, 2]
    keep = (z > z_min) & (z < z_max)
    triangles = np.asarray(mesh.triangles)
    mesh.remove_triangles_by_mask(~keep[triangles].all(axis=1))
    mesh.remove_unreferenced_vertices()
    return mesh


def keep_large_parts(mesh, fraction=0.01):
    clusters, sizes, _ = mesh.cluster_connected_triangles()
    clusters, sizes = np.asarray(clusters), np.asarray(sizes)
    mesh.remove_triangles_by_mask(sizes[clusters] < fraction * len(clusters))
    mesh.remove_unreferenced_vertices()
    return mesh


def write_textured_obj(mesh, out, texture_size):
    """Bake vertex colors into a UV texture; MuJoCo cannot draw vertex colors."""
    # UV unwrapping needs a manifold mesh; Poisson trimming and decimation leave a few bad edges.
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    t.compute_uvatlas(size=texture_size)
    textures = t.bake_vertex_attr_textures(texture_size, {"colors"})
    image = np.clip(textures["colors"].numpy() * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(str(out / "scene_color.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    v = t.vertex.positions.numpy()
    f = t.triangle.indices.numpy()
    uv = t.triangle.texture_uvs.numpy().reshape(-1, 2)
    with open(out / "scene.obj", "w") as obj:
        obj.write("".join(f"v {x:.4f} {y:.4f} {z:.4f}\n" for x, y, z in v))
        obj.write("".join(f"vt {u:.5f} {w:.5f}\n" for u, w in uv))  # Open3D already uses OBJ's bottom-up v
        obj.write("".join(f"f {a + 1}/{3 * i + 1} {b + 1}/{3 * i + 2} {c + 1}/{3 * i + 3}\n" for i, (a, b, c) in enumerate(f)))


def copy_visual_obj(src, T, out):
    """Use an already-textured OBJ (e.g. OpenMVS TextureMesh) as the visual: move it into the leveled frame.

    ponytail: assumes one texture atlas (TextureMesh's default for a scene this size); split per
    material with obj2mjcf if a scene comes out with several.
    """
    src = Path(src)
    lines, textures = [], []
    for line in src.read_text().splitlines():
        if line.startswith("v "):
            x = T[:3, :3] @ np.array(line.split()[1:4], dtype=float) + T[:3, 3]
            lines.append(f"v {x[0]:.4f} {x[1]:.4f} {x[2]:.4f}")
        elif line.startswith("mtllib"):
            mtl = (src.parent / line.split(maxsplit=1)[1]).read_text()
            textures = [l.split(maxsplit=1)[1] for l in mtl.splitlines() if l.strip().startswith("map_Kd")]
        elif not line.startswith(("usemtl", "vn ")):  # one material, normals recomputed by MuJoCo
            lines.append(line)
    if not textures:
        raise SystemExit(f"{src} has no texture (map_Kd)")
    if len(textures) > 1:
        print(f"warning: {len(textures)} texture atlases, using the first")
    cv2.imwrite(str(out / "scene_color.png"), cv2.imread(str(src.parent / textures[0])))
    (out / "scene.obj").write_text("\n".join(lines) + "\n")


def collision_proxies(mesh, out, z_max, max_hulls=96, floor_tolerance=0.06):
    """Thin boxes for wall planes and CoACD convex pieces for everything else above the floor."""
    import coacd
    pts = np.asarray(mesh.sample_points_uniformly(400_000).points)
    floor_pts = pts[pts[:, 2] < floor_tolerance]
    rest = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts[(pts[:, 2] >= floor_tolerance) & (pts[:, 2] < z_max)]))
    total, boxes, clutter = len(rest.points), [], []
    for _ in range(14):  # peel planes off, largest first; vertical big ones become wall boxes
        if len(rest.points) < 500:
            break
        (a, b, c, d), idx = rest.segment_plane(distance_threshold=0.03, ransac_n=3, num_iterations=2000)
        inl = np.asarray(rest.points)[idx]
        n = np.array([a, b, c]) / np.linalg.norm([a, b, c])
        if abs(n[2]) < np.sin(np.radians(15)) and len(inl) > 0.02 * total:
            u = np.cross([0.0, 0.0, 1.0], n)
            u /= np.linalg.norm(u)
            s_lo, s_hi = np.percentile(inl @ u, [1, 99])
            top = float(np.percentile(inl[:, 2], 99))
            on_plane = inl.mean(axis=0) - (inl.mean(axis=0) @ u) * u  # plane point at s = 0
            center = on_plane + (s_lo + s_hi) / 2 * u
            center[2] = top / 2
            boxes.append({"pos": center, "half": [0.05, (s_hi - s_lo) / 2, top / 2], "yaw": float(np.arctan2(n[1], n[0]))})
        else:
            clutter.append(inl)
        rest = rest.select_by_index(idx, invert=True)
    clutter.append(np.asarray(rest.points))
    clutter = np.vstack(clutter)

    # Group the clutter into objects, cut each one's triangles out of the mesh, decompose it.
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(clutter))
    labels = np.asarray(pc.cluster_dbscan(eps=0.08, min_points=30))
    tree = o3d.geometry.KDTreeFlann(pc)
    tri = np.asarray(mesh.triangles)
    centroids = np.asarray(mesh.vertices)[tri].mean(axis=1)
    owner = np.full(len(tri), -1)
    for i, x in enumerate(centroids):
        k, j, dist = tree.search_knn_vector_3d(x, 1)
        if dist[0] < 0.04 ** 2:
            owner[i] = labels[j[0]]
    hulls = []
    for label in np.argsort(-np.bincount(labels[labels >= 0]))[:max_hulls]:
        part = tri[owner == label]
        if len(part) < 50 or len(hulls) >= max_hulls:
            continue
        used, faces = np.unique(part, return_inverse=True)
        verts = np.asarray(mesh.vertices)[used]
        for v, f in coacd.run_coacd(coacd.Mesh(verts, faces.reshape(-1, 3)), threshold=0.08, max_convex_hull=6):
            if len(hulls) < max_hulls:
                hulls.append(np.asarray(v))
    return boxes, hulls, floor_pts, clutter


def spawn_point(floor_pts, obstacle_pts, cell=0.05):
    """Most open scanned floor spot: farthest from any obstacle, among cells where floor was seen."""
    lo = floor_pts[:, :2].min(axis=0)
    shape = np.ceil((floor_pts[:, :2].max(axis=0) - lo) / cell).astype(int) + 1
    def grid(p):
        g = np.zeros(shape[::-1], dtype=bool)
        ij = ((p[:, :2] - lo) / cell).astype(int)
        ok = (ij >= 0).all(axis=1) & (ij[:, 0] < shape[0]) & (ij[:, 1] < shape[1])
        g[ij[ok, 1], ij[ok, 0]] = True
        return g
    free = ndimage.binary_closing(grid(floor_pts), iterations=2) & ~grid(obstacle_pts)
    dist = ndimage.distance_transform_edt(free)
    r, c = np.unravel_index(dist.argmax(), dist.shape)
    xy = lambda rr, cc: np.stack([lo[0] + (cc + 0.5) * cell, lo[1] + (rr + 0.5) * cell], axis=-1)
    rng = np.random.default_rng(0)
    def sample(lo_m, hi_m):
        rr, cc = np.nonzero((dist * cell >= lo_m) & (dist * cell < hi_m))
        pick = rng.permutation(len(rr))[:300]
        return xy(rr[pick], cc[pick]).round(3).tolist()
    # open: reachable without brushing anything; near: 15-30 cm from a wall or clutter, to test the proxies
    return xy(r, c).tolist(), float(dist.max() * cell), {"open": sample(0.35, 99), "near": sample(0.15, 0.30)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("frames", type=Path)
    p.add_argument("--crop-height", type=float, default=2.0, help="cut away everything above this height (m)")
    p.add_argument("--max-triangles", type=int, default=200_000)
    p.add_argument("--texture-size", type=int, default=2048)
    p.add_argument("--visual-obj", type=Path, help="already-textured OBJ to use as the visual instead of baking vertex colors")
    a = p.parse_args()
    out = a.frames / "mujoco"
    out.mkdir(exist_ok=True)

    mesh = keep_large_parts(o3d.io.read_triangle_mesh(str(a.frames / "mesh.ply")))
    poses = np.loadtxt(a.frames / "poses.txt").reshape(-1, 4, 4)
    lidar_map = a.frames / "lidar_map.ply"
    extra = np.asarray(o3d.io.read_point_cloud(str(lidar_map)).points) if lidar_map.exists() else None
    T, walls = level(mesh, poses, extra)
    mesh.transform(T)
    mesh = crop(mesh, -0.2, a.crop_height)
    if len(mesh.triangles) > a.max_triangles:
        mesh = mesh.simplify_quadric_decimation(a.max_triangles)
    if a.visual_obj:
        copy_visual_obj(a.visual_obj, T, out)
    else:
        write_textured_obj(mesh, out, a.texture_size)

    boxes, hulls, floor_pts, clutter = collision_proxies(mesh, out, a.crop_height)
    n_hulls = len(hulls)
    spawn, clearance, waypoints = spawn_point(floor_pts, clutter)
    v = np.asarray(mesh.vertices)
    info = {"spawn_xy": spawn, "clearance_m": clearance, "size_m": (v.max(0) - v.min(0)).round(2).tolist(),
            "crop_height": a.crop_height, "triangles": len(mesh.triangles), "wall_boxes": len(boxes),
            "clutter_hulls": n_hulls, "waypoints": waypoints, "walls": [{k: np.asarray(v).tolist() for k, v in b.items()} for b in boxes],
            "transform_from_scan": T.tolist()}
    (out / "scene_info.json").write_text(json.dumps(info, indent=2))
    # Vertices only: MuJoCo builds each piece's convex hull itself.
    hull_assets = "".join(f'    <mesh name="hull_{i:02d}" vertex="{" ".join(f"{x:.4f}" for x in v.ravel())}"/>\n' for i, v in enumerate(hulls))
    hull_geoms = "".join(f'    <geom name="hull_{i:02d}" type="mesh" mesh="hull_{i:02d}" class="collision"/>\n' for i in range(n_hulls))
    box_geoms = "".join(
        f'    <geom name="wall_{i:02d}" type="box" class="collision" pos="{b["pos"][0]:.3f} {b["pos"][1]:.3f} {b["pos"][2]:.3f}" '
        f'size="{b["half"][0]:.3f} {b["half"][1]:.3f} {b["half"][2]:.3f}" euler="0 0 {b["yaw"]:.4f}"/>\n' for i, b in enumerate(boxes))
    (out / "scene.xml").write_text(f"""<mujoco model="reconstructed scene">
  <compiler angle="radian"/>
  <visual><headlight diffuse="0.7 0.7 0.7" ambient="0.35 0.35 0.35"/><global azimuth="135" elevation="-35"/></visual>
  <default><default class="collision"><geom group="3" rgba="0.2 0.6 1 0.4"/></default></default>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.6 0.7 0.8" rgb2="0.2 0.25 0.3" width="256" height="256"/>
    <texture name="scan" type="2d" file="scene_color.png"/>
    <material name="scan" texture="scan" specular="0.1" emission="0.3"/>
    <mesh name="scan" file="scene.obj" inertia="shell"/>
{hull_assets}  </asset>
  <worldbody>
    <light pos="0 0 6" dir="0 0 -1" directional="true" diffuse="0.5 0.5 0.5"/>
    <geom name="floor" type="plane" size="0 0 0.05" rgba="0.35 0.35 0.33 1"/>
    <geom name="scan" type="mesh" mesh="scan" material="scan" contype="0" conaffinity="0"/>
{box_geoms}{hull_geoms}    <site name="spawn" pos="{spawn[0]:.3f} {spawn[1]:.3f} 0.01" size="0.08" rgba="0.2 1 0.3 0.6"/>
  </worldbody>
</mujoco>
""")
    print(f"scene {info['size_m'][0]:.1f} x {info['size_m'][1]:.1f} m, {len(mesh.triangles)} triangles; collisions: floor plane, "
          f"{len(boxes)} wall boxes, {n_hulls} clutter hulls; spawn ({spawn[0]:+.2f}, {spawn[1]:+.2f}) with {clearance:.2f} m clearance "
          f"-> {out / 'scene.xml'}")


if __name__ == "__main__":
    main()
