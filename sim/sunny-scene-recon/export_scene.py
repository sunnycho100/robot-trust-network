"""Export a build_scene.py output for Isaac (Gym or Sim) and lay out a CooHOI two-humanoid box carry in it.

    python export_scene.py SCENE_DIR [--coohoi-humanoid amp_humanoid.xml]

Writes SCENE_DIR/export/:
  scene.urdf         one fixed link: the textured scan as the visual, floor slab, wall boxes and hulls as collisions
  ground.obj         outdoor scenes: the heightfield as a 10 cm terrain mesh, to load as a static triangle mesh
                     (Isaac Gym add_triangle_mesh), not through the URDF, which would turn it into a convex hull
  scene.obj/.mtl/png visual mesh with its texture, hull_XX.obj convex collision pieces
  carry_task.json    CooHOI-sized box (2.0 x 0.5 x 0.5 m), its pose, the target 1.5 to 4 m away along a clear path,
                     and the two humanoid start poses at the box ends, all on scanned free floor
  preview.xml        MuJoCo scene with the box, the target marker and (with --coohoi-humanoid) two CooHOI humanoids
"""
import argparse, json, math, shutil
from pathlib import Path
import mujoco
import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull

BOX = (2.0, 0.5, 0.5)  # CooHOI two-agent box: length, width, height (share_humanoid_amp_carryobject.py)
CELL = 0.1


def world_mesh(model, data, g):
    m = model.geom_dataid[g]
    v = model.mesh_vert[model.mesh_vertadr[m]:model.mesh_vertadr[m] + model.mesh_vertnum[m]]
    return v @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]  # MuJoCo stores meshes re-centered


def ground_model(scene, info):
    """Ground height and scanned-ground mask at any xy: the heightfield outdoors, z = 0 indoors."""
    hf = info.get("heightfield")
    if not hf:
        return lambda xy: np.zeros(np.shape(xy)[:-1]), None
    h = np.fromfile(scene / "ground.bin", dtype=np.float32)[2:].reshape(hf["shape"])
    seen = ~np.isnan(np.load(scene / "ground_heights.npy"))
    lo, cell = np.array(hf["origin"]), hf["cell"]
    def cells(xy):
        ij = np.clip(np.round((np.asarray(xy) - lo) / cell).astype(int), 0, np.array(h.shape[::-1]) - 1)
        return ij[..., 1], ij[..., 0]
    at = lambda xy: h[cells(xy)]
    at.seen = lambda xy: seen[cells(xy)]
    at.heights = h
    return at, hf


def clearance_map(model, data, top, ground, outdoor):
    """Cast rays down: scanned ground counts as free, anything else (proxies, raised scan, unscanned plane) does not."""
    scan = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "scan")
    terrain = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
    v = world_mesh(model, data, scan)
    lo, hi = v[:, :2].min(0), v[:, :2].max(0)
    xs, ys = np.arange(lo[0], hi[0], CELL), np.arange(lo[1], hi[1], CELL)
    free = np.zeros((len(ys), len(xs)), dtype=bool)
    geomid = np.zeros(1, dtype=np.int32)
    groups = np.ones(6, dtype=np.uint8)
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            d = mujoco.mj_ray(model, data, np.array([x, y, top]), np.array([0, 0, -1.0]), groups, 1, -1, geomid)
            xy = np.array([x, y])
            low = d >= 0 and top - d - ground(xy) < 0.1
            free[i, j] = low and (ground.seen(xy) and geomid[0] in (scan, terrain) if outdoor else geomid[0] == scan)
    free = ndimage.binary_closing(free, iterations=2)  # small holes in the scan are not obstacles
    return ndimage.distance_transform_edt(free) * CELL, lo, xs, ys


def layout(clear, lo):
    """Box where the floor is most open, target along the clearest direction, humanoids at both box ends."""
    need = math.hypot(BOX[0] / 2 + 0.6, BOX[1] / 2) + 0.3  # half the box plus a humanoid at each end, plus margin
    at = lambda p: clear[int((p[1] - lo[1]) / CELL), int((p[0] - lo[0]) / CELL)] if (
        0 <= int((p[1] - lo[1]) / CELL) < clear.shape[0] and 0 <= int((p[0] - lo[0]) / CELL) < clear.shape[1]) else 0.0
    order = np.argsort(clear.ravel())[::-1][:200]
    for k in order:
        i, j = np.unravel_index(k, clear.shape)
        box = lo + np.array([j, i]) * CELL
        if clear[i, j] < need:
            break
        best = None
        for yaw in np.linspace(0, 2 * math.pi, 36, endpoint=False):
            d = np.array([math.cos(yaw), math.sin(yaw)])
            reach = 0.0
            for s in np.arange(0.1, 4.0 + 1e-6, 0.1):  # walk out until the path gets too tight
                if at(box + s * d) < need:
                    break
                reach = s
            if reach >= 1.5 and (best is None or reach > best[1]):
                best = (yaw, reach)
        if best:
            yaw, reach = best
            d = np.array([math.cos(yaw), math.sin(yaw)])
            n = np.array([-d[1], d[0]])  # carry sideways: the long box axis is across the path, a humanoid at each end
            ends = [box + s * (BOX[0] / 2 + 0.6) * n for s in (1, -1)]
            return {"box": {"size": BOX, "pos": [*box.round(3).tolist(), BOX[2] / 2], "yaw": round(float(yaw + math.pi / 2), 4)},
                    "target": [*(box + reach * d).round(3).tolist(), BOX[2] / 2], "carry_distance_m": round(reach, 2),
                    "humanoids": [{"pos": [*e.round(3).tolist(), 0.0], "yaw": round(float(math.atan2(*(box - e)[::-1])), 4)} for e in ends],
                    "clearance_m": round(float(clear[i, j]), 2), "path_clearance_needed_m": round(need, 2)}
    raise SystemExit(f"no open floor patch with {need:.2f} m clearance and a 1.5 m clear path")


def obj_from_hull(v, path):
    faces = ConvexHull(v).simplices
    c = v.mean(0)
    lines = [f"v {x:.4f} {y:.4f} {z:.4f}" for x, y, z in v]
    for f in faces:  # outward winding
        a, b, d = v[f]
        if np.cross(b - a, d - a) @ (a - c) < 0:
            f = f[[0, 2, 1]]
        lines.append("f " + " ".join(str(i + 1) for i in f))
    path.write_text("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("scene", type=Path)
    p.add_argument("--coohoi-humanoid", type=Path, help="CooHOI's coohoi/data/assets/mjcf/amp_humanoid.xml")
    a = p.parse_args()
    info = json.loads((a.scene / "scene_info.json").read_text())
    model = mujoco.MjModel.from_xml_path(str(a.scene / "scene.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    out = a.scene / "export"
    out.mkdir(exist_ok=True)

    # URDF: visual mesh with an .mtl for the texture, collisions as the same proxies MuJoCo uses
    shutil.copy(a.scene / "scene_color.png", out / "scene_color.png")
    if (a.scene / "ground.bin").exists():  # preview.xml's heightfield
        shutil.copy(a.scene / "ground.bin", out / "ground.bin")
    (out / "scene.mtl").write_text("newmtl scan\nKd 1 1 1\nmap_Kd scene_color.png\n")
    (out / "scene.obj").write_text("mtllib scene.mtl\nusemtl scan\n" + (a.scene / "scene.obj").read_text())
    ground, hf = ground_model(a.scene, info)
    size = np.array(info["size_m"][:2]) + 4.0
    col = [] if hf else [f'<collision name="floor"><origin xyz="0 0 -0.05"/><geometry><box size="{size[0]:.2f} {size[1]:.2f} 0.1"/></geometry></collision>']
    if hf:  # terrain mesh from every second heightfield node (10 cm)
        h = ground.heights[::2, ::2]
        r, c = np.mgrid[0:h.shape[0], 0:h.shape[1]]
        xy = np.array(hf["origin"]) + np.stack([c, r], axis=-1) * hf["cell"] * 2
        idx = np.arange(h.size).reshape(h.shape) + 1
        a0, a1, a2, a3 = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel(), idx[1:, :-1].ravel()
        (out / "ground.obj").write_text("".join(f"v {x:.3f} {y:.3f} {z:.3f}\n" for (x, y), z in zip(xy.reshape(-1, 2), h.ravel()))
                                        + "".join(f"f {i} {j} {k}\nf {i} {k} {l}\n" for i, j, k, l in zip(a0, a1, a2, a3)))
    for i, w in enumerate(info["walls"]):
        h = 2 * np.array(w["half"])
        col.append(f'<collision name="wall_{i:02d}"><origin xyz="{w["pos"][0]:.3f} {w["pos"][1]:.3f} {w["pos"][2]:.3f}" rpy="0 0 {w["yaw"]:.4f}"/>'
                   f'<geometry><box size="{h[0]:.3f} {h[1]:.3f} {h[2]:.3f}"/></geometry></collision>')
    for i, b in enumerate(info.get("object_boxes", [])):
        col.append(f'<collision name="obj_{i:04d}"><origin xyz="{b["pos"][0]:.3f} {b["pos"][1]:.3f} {b["pos"][2]:.3f}"/>'
                   f'<geometry><box size="{2 * b["half"][0]:.3f} {2 * b["half"][1]:.3f} {2 * b["half"][2]:.3f}"/></geometry></collision>')
    for i in range(info["clutter_hulls"]):
        g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"hull_{i:02d}")
        obj_from_hull(world_mesh(model, data, g), out / f"hull_{i:02d}.obj")
        col.append(f'<collision name="hull_{i:02d}"><geometry><mesh filename="hull_{i:02d}.obj"/></geometry></collision>')
    (out / "scene.urdf").write_text(
        '<?xml version="1.0"?>\n<robot name="reconstructed_scene">\n  <link name="scene">\n'
        '    <visual><geometry><mesh filename="scene.obj"/></geometry></visual>\n    '
        + "\n    ".join(col) + "\n  </link>\n</robot>\n")

    top = (hf["z_min"] + hf["z_range"] if hf else 0.0) + info["crop_height"] + 0.3
    clear, lo, _, _ = clearance_map(model, data, top, ground, bool(hf))
    task = layout(clear, lo)
    task["box"]["pos"][2] += float(ground(task["box"]["pos"][:2]))  # sit everything on the local ground
    task["target"][2] += float(ground(task["target"][:2]))
    for h in task["humanoids"]:
        h["pos"][2] = float(ground(h["pos"][:2]))
    if hf:
        task["terrain"] = "ground.obj: load as a static triangle mesh; scene.urdf then has no floor"
    (out / "carry_task.json").write_text(json.dumps(task, indent=2))
    print(json.dumps(task))

    # MuJoCo preview: box, target marker, and optionally the two CooHOI humanoids standing at the box ends
    spec = mujoco.MjSpec.from_file(str(a.scene / "scene.xml"))
    bx = task["box"]
    body = spec.worldbody.add_body(name="box", pos=bx["pos"], quat=[math.cos(bx["yaw"] / 2), 0, 0, math.sin(bx["yaw"] / 2)])
    body.add_freejoint()
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[s / 2 for s in BOX], rgba=[0.8, 0.5, 0.2, 1], mass=10.0)
    spec.worldbody.add_geom(name="target", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.3, 0.005, 0], pos=[*task["target"][:2], task["target"][2] - BOX[2] / 2 + 0.005],
                            rgba=[0.1, 0.9, 0.2, 0.6], contype=0, conaffinity=0)
    if a.coohoi_humanoid:
        for k, h in enumerate(task["humanoids"]):
            child = mujoco.MjSpec.from_file(str(a.coohoi_humanoid))
            floor = child.geom("floor")  # the scene has its own floor
            floor.contype, floor.conaffinity, floor.rgba, floor.material = 0, 0, [0, 0, 0, 0], ""
            frame = spec.worldbody.add_frame(pos=h["pos"], quat=[math.cos(h["yaw"] / 2), 0, 0, math.sin(h["yaw"] / 2)])
            spec.attach(child, prefix=f"h{k}_", frame=frame)
    spec.compile()
    (out / "preview.xml").write_text(spec.to_xml())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
