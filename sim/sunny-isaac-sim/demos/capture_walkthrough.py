# Renders a camera walkthrough of an Isaac Sim scene and saves what a reconstruction needs plus ground truth to score it.
# Usage: /isaac-sim/python.sh capture_walkthrough.py OUT_DIR [--scene URL_OR_FILE] [--frames 240] [--height 1.6]
#   OUT_DIR/images/frame_00000.png      RGB, 1280x720 by default
#   OUT_DIR/sparse/0/{cameras,images,points3D}.txt   COLMAP text model: exact intrinsics and poses (PINHOLE)
#   OUT_DIR/depth/frame_00000.png       16-bit depth along the optical axis, millimetres (0 = no hit)
#   OUT_DIR/gt_mesh.ply                 every visible mesh of the scene in world coordinates (metres, Z up)
#   OUT_DIR/test.txt                    every 8th frame, held out for PSNR, SSIM and LPIPS (the usual llffhold)
#   OUT_DIR/camera_params_0.json        Replicator's own camera params for frame 0, to cross-check the poses
# points3D.txt is seeded from the GT depth (fair for appearance; for a geometry score, triangulate with COLMAP
# instead: `colmap point_triangulator` with these poses). Default scene: the stock full warehouse.
"""Isaac Sim 6.2. The camera pose is written straight onto a USD camera as a matrix, so the saved COLMAP pose is
the pose that rendered the frame (no look-at convention to guess). Path: per --height, two loops around the scene
centre, one looking inward and one looking outward, pitched down by --pitch.
Known limits (critic_reconstruction.md): the loop can pass through racks (no free-space check), units and up axis
are assumed metres and Z up, the GT mesh skips PointInstancers and keeps proxy geometry, and the renders are clean
(no noise, blur or exposure change), so this is an easier case than a real capture."""
import argparse
import os
import sys

import numpy as np

FLIP = np.diag([1.0, -1.0, -1.0])  # USD camera axes (x right, y up, looks down -z) to COLMAP (x right, y down, +z)


def look_at_c2w(eye, target, up=(0.0, 0.0, 1.0)):
    """USD camera-to-world 4x4 (column vectors) for a camera at eye looking at target, Z up world."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    back = eye - target
    back /= np.linalg.norm(back)
    right = np.cross(up, back)
    right /= np.linalg.norm(right)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = right, np.cross(back, right), back, eye
    return m


def colmap_pose(c2w_usd):
    """World-to-camera (qvec wxyz, tvec) in COLMAP's convention from a USD camera-to-world matrix."""
    r_c2w = c2w_usd[:3, :3] @ FLIP
    r = r_c2w.T
    t = -r @ c2w_usd[:3, 3]
    return rotmat_to_qvec(r), t


def rotmat_to_qvec(r):
    """Same method as COLMAP's scripts/python/read_write_model.py rotmat2qvec."""
    rxx, ryx, rzx, rxy, ryy, rzy, rxz, ryz, rzz = r.flat
    k = np.array([[rxx - ryy - rzz, 0, 0, 0], [ryx + rxy, ryy - rxx - rzz, 0, 0],
                  [rzx + rxz, rzy + ryz, rzz - rxx - ryy, 0], [ryz - rzy, rzx - rxz, rxy - ryx, rxx + ryy + rzz]]) / 3.0
    vals, vecs = np.linalg.eigh(k)
    q = vecs[[3, 0, 1, 2], np.argmax(vals)]
    return -q if q[0] < 0 else q


def ring_path(lo, hi, n, height, pitch_deg=15.0):
    """n (eye, target) pairs: half on an inward-looking loop, half on an outward-looking one, inside the box lo..hi,
    pitched down so the floor near the robot is in view."""
    c = (np.asarray(lo, float) + hi) / 2
    half = (np.asarray(hi, float) - lo) / 2
    z = lo[2] + height
    poses = []
    for i in range(n):
        a = 2 * np.pi * (i % (n // 2)) / (n // 2)
        inward = i < n // 2
        rx, ry = half[0] * (0.55 if inward else 0.3), half[1] * (0.55 if inward else 0.3)
        eye = np.array([c[0] + rx * np.cos(a), c[1] + ry * np.sin(a), z])
        d = np.array([c[0], c[1], z]) - eye if inward else eye - np.array([c[0], c[1], z])
        d[2] = 0.0
        d /= np.linalg.norm(d)
        d[2] = -np.tan(np.radians(pitch_deg))
        poses.append((eye, eye + d))
    return poses


def backproject(depth_m, fx, fy, cx, cy, c2w_usd, k, rng):
    """k random world points (and their pixel indices) from a depth image along the optical axis."""
    v, u = np.nonzero(depth_m > 0)
    pick = rng.choice(len(u), size=min(k, len(u)), replace=False)
    u, v = u[pick], v[pick]
    z = depth_m[v, u]
    cam = np.stack([(u + 0.5 - cx) / fx * z, (v + 0.5 - cy) / fy * z, z])  # COLMAP camera frame
    r_c2w = c2w_usd[:3, :3] @ FLIP
    return (r_c2w @ cam).T + c2w_usd[:3, 3], v, u


def write_colmap(out, w, h, fx, fy, cx, cy, names, poses, pts, cols):
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "cameras.txt"), "w") as f:
        f.write(f"1 PINHOLE {w} {h} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n")
    with open(os.path.join(out, "images.txt"), "w") as f:
        for i, (name, (q, t)) in enumerate(zip(names, poses), 1):
            f.write(f"{i} {' '.join(f'{x:.9f}' for x in q)} {' '.join(f'{x:.9f}' for x in t)} 1 {name}\n\n")
    with open(os.path.join(out, "points3D.txt"), "w") as f:
        for i, (p, c) in enumerate(zip(pts, cols), 1):
            f.write(f"{i} {p[0]:.5f} {p[1]:.5f} {p[2]:.5f} {c[0]} {c[1]} {c[2]} 0\n")


def write_ply(path, verts, faces):
    with open(path, "wb") as f:
        f.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(verts)}\nproperty float x\n"
                 f"property float y\nproperty float z\nelement face {len(faces)}\nproperty list uchar int vertex_indices\n"
                 "end_header\n").encode())
        f.write(verts.astype("<f4").tobytes())
        rec = np.empty(len(faces), dtype=[("n", "u1"), ("i", "<i4", 3)])
        rec["n"], rec["i"] = 3, faces
        f.write(rec.tobytes())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out")
    ap.add_argument("--scene", default="/Isaac/Environments/Simple_Warehouse/full_warehouse.usd",
                    help="asset-root path (starts with /Isaac) or a local .usd or .usdz")
    ap.add_argument("--frames", type=int, default=240)
    ap.add_argument("--height", type=float, nargs="+", default=[1.6, 0.4],
                    help="camera heights above the scene's lowest point, m (default: a person and a wheeled robot)")
    ap.add_argument("--pitch", type=float, default=15.0, help="downward camera pitch, degrees")
    ap.add_argument("--res", type=int, nargs=2, default=[1280, 720])
    ap.add_argument("--hfov", type=float, default=80.0, help="horizontal field of view, degrees")
    ap.add_argument("--subframes", type=int, default=8, help="RTX subframes per frame (less noise, slower)")
    args, sys.argv[1:] = ap.parse_known_args()

    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True, "width": args.res[0], "height": args.res[1]})
    import omni.replicator.core as rep
    import omni.usd
    from isaacsim.storage.native import get_assets_root_path
    from PIL import Image
    from pxr import Gf, Usd, UsdGeom

    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    url = get_assets_root_path() + args.scene if args.scene.startswith("/Isaac") else os.path.abspath(args.scene)
    scene = stage.DefinePrim("/World/Scene", "Xform")
    scene.GetReferences().AddReference(url)
    for _ in range(60):  # let references and textures load
        app.update()

    # ground truth mesh and the bounds the path lives in
    verts, faces, nv = [], [], 0
    for p in Usd.PrimRange(scene, Usd.TraverseInstanceProxies()):
        if not p.IsA(UsdGeom.Mesh) or UsdGeom.Imageable(p).ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue
        m = UsdGeom.Mesh(p)
        pts, counts, idx = m.GetPointsAttr().Get(), m.GetFaceVertexCountsAttr().Get(), m.GetFaceVertexIndicesAttr().Get()
        if not pts or not counts:
            continue
        xf = np.array(UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T  # to column vectors
        pts = np.asarray(pts, float)
        verts.append(pts @ xf[:3, :3].T + xf[:3, 3])
        idx, start = np.asarray(idx), 0
        for c in counts:  # fan triangulation
            faces += [(nv + idx[start], nv + idx[start + j], nv + idx[start + j + 1]) for j in range(1, c - 1)]
            start += c
        nv += len(pts)
    verts, faces = np.concatenate(verts), np.asarray(faces, np.int64)
    os.makedirs(args.out, exist_ok=True)
    write_ply(os.path.join(args.out, "gt_mesh.ply"), verts, faces)
    lo, hi = np.percentile(verts, 2, axis=0), np.percentile(verts, 98, axis=0)  # ignore a far-out skybox or two
    print(f"[walk] gt mesh {len(verts)} verts {len(faces)} tris, bounds {lo.round(2)} .. {hi.round(2)}", flush=True)

    # camera with exact intrinsics: square pixels, principal point at the centre
    w, h = args.res
    cam_path = "/World/WalkCam"
    cam = UsdGeom.Camera.Define(stage, cam_path)
    ha = 20.955
    focal = ha / 2 / np.tan(np.radians(args.hfov) / 2)
    cam.CreateFocalLengthAttr(focal)
    cam.CreateHorizontalApertureAttr(ha)
    cam.CreateVerticalApertureAttr(ha * h / w)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 1000.0))
    op = UsdGeom.Xformable(cam).AddTransformOp()
    fx = fy = w * focal / ha
    cx, cy = w / 2, h / 2

    rp = rep.create.render_product(cam_path, (w, h))
    rgb, dist, params = (rep.annotators.get(n) for n in ("rgb", "distance_to_image_plane", "camera_params"))
    for a in (rgb, dist, params):
        a.attach(rp)
    for d in ("images", "depth"):
        os.makedirs(os.path.join(args.out, d), exist_ok=True)

    rng = np.random.default_rng(0)
    names, poses, pts, cols = [], [], [], []
    k = args.frames // len(args.height)
    path = [p for hgt in args.height for p in ring_path(lo, hi, k, hgt, args.pitch)]
    for i, (eye, target) in enumerate(path):
        c2w = look_at_c2w(eye, target)
        op.Set(Gf.Matrix4d(c2w.T.tolist()))  # USD stores row-vector matrices
        rep.orchestrator.step(rt_subframes=args.subframes, delta_time=0.0)
        img = np.asarray(rgb.get_data())[..., :3]
        depth = np.nan_to_num(np.asarray(dist.get_data(), np.float32), nan=0.0, posinf=0.0)
        name = f"frame_{i:05d}.png"
        Image.fromarray(img).save(os.path.join(args.out, "images", name))
        Image.fromarray(np.clip(depth * 1000, 0, 65535).astype(np.uint16)).save(os.path.join(args.out, "depth", name))
        if i == 0:
            import json
            with open(os.path.join(args.out, "camera_params_0.json"), "w") as f:
                json.dump({k: np.asarray(v).tolist() for k, v in params.get_data().items()}, f, default=str)
        if i % 8:  # held-out frames (test.txt) must not seed the points
            p, v, u = backproject(depth, fx, fy, cx, cy, c2w, 400, rng)
            pts.append(p)
            cols.append(img[v, u])
        names.append(name)
        poses.append(colmap_pose(c2w))
        if i % 20 == 0:
            print(f"[walk] frame {i}/{len(path)}, mean depth {depth[depth > 0].mean():.2f} m", flush=True)

    write_colmap(os.path.join(args.out, "sparse", "0"), w, h, fx, fy, cx, cy, names, poses,
                 np.concatenate(pts), np.concatenate(cols))
    with open(os.path.join(args.out, "test.txt"), "w") as f:
        f.write("\n".join(names[::8]) + "\n")
    print(f"[walk] done: {len(names)} frames in {args.out}", flush=True)
    app.close()


if __name__ == "__main__":
    main()
