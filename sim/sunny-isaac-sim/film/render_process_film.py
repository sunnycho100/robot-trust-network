"""Render the site-to-simulation process film from our scan floor4_static3 (1920x1080, 30 fps).

Run with the sim venv (mujoco, torch, pygame):
  sim/mujoco_houses/.venv/bin/python sim/sunny-isaac-sim/film/render_process_film.py              # docs/evidence/process/process_film.mp4
  sim/mujoco_houses/.venv/bin/python sim/sunny-isaac-sim/film/render_process_film.py --still 6.5 30   # PNG stills at those seconds
  sim/mujoco_houses/.venv/bin/python sim/sunny-isaac-sim/film/render_process_film.py --check      # dots line up with MuJoCo's render

Every stage comes from sim/sunny-scene-recon/data/floor4_static3:
  01 Video frames   color/*.jpg, the RealSense RGB frames (extract_bag.py)
  02 Camera poses   poses.txt (lidar_poses.py: LiDAR ICP plus the camera extrinsic); the dots are depth/*.png back-projected at those poses
  03 Point cloud    every other depth frame fused at those poses on a 2.5 cm grid (the depth reconstruct.py's TSDF fuses),
                    cropped like site/tools/export_scan.py and shaded like site/engine.js (the site itself draws lidar_map.ply)
  04 Surface mesh   mujoco/scene.obj, the TSDF mesh (reconstruct.py) leveled and textured by build_scene.py
  05 Collider       the floor plane, 12 wall boxes and CoACD hulls in mujoco/scene.xml (build_scene.py), raw, over the raw mesh;
                    then a cleaned view of them (clean_walls: drift copies dropped, parallel boxes merged, square corners,
                    the scan's openings kept), labelled on screen as such
  06 Simulation     Unitree G1 on the unitree_rl_gym walking policy, simulated against the raw scene.xml collider (all 12 boxes and
                    the hulls) and drawn in the cleaned view; walk_clearance checks the path stays inside it, clear of every wall
Not pipeline outputs: the clean look (materials, lights, fog, near walls cut away), the camera moves, the sparse sampling, dot thinning and reveal timing, the robot fade-in and the spawn ring.
"""
import argparse, functools, json, math, os, subprocess, sys
from pathlib import Path
import numpy as np
import mujoco

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import pygame

ROOT = Path(__file__).resolve().parents[3]
SCAN = ROOT / "sim/sunny-scene-recon/data/floor4_static3"
OUT = ROOT / "docs/evidence/process/process_film.mp4"
sys.path.insert(0, str(ROOT / "sim/mujoco_houses/src"))
from unitree_demo.g1_policy import G1Controller, gait_phase, load_g1_config  # noqa: E402

W, H, FPS, FOVY = 1920, 1080, 30, 45.0
BG, INK, ACC = (np.array(c, np.float32) for c in ((10, 12, 12), (228, 234, 231), (111, 224, 207)))  # site --bg --ink --accent
LINE = np.array((34, 42, 41), np.float32)  # --line
FONT = str(Path(__file__).parent / "fonts/IBMPlexMono-Medium.ttf")  # the site's mono (see README.md)

# Timeline, seconds. The capture plays at 2x, then the last frame turns into its own depth dots.
LAST = 352                     # last capture frame, also a keyframe (multiple of KEY)
KEY, SPARSE_N = 8, 400         # every 8th pose, 400 random depth pixels: the sparse dots
T_PLAY = LAST / 60             # capture done
T_MATCH = (5.9, 6.9)           # image dissolves into dots
T_PULL = (6.6, 9.8)            # camera pulls back from the capture pose to the orbit
T_PATH = (9.6, 13.6)           # camera path drawn pose by pose
T_CLOUD = (14.0, 16.8)         # LiDAR map grows along the path
T_MESH = (20.6, 22.2)          # dots resolve into the mesh
T_COLL = (24.3, 25.1)          # collider shapes in
T_CLEAN = (26.2, 27.6)         # raw mesh and cyan shapes resolve into the clean collider geometry
SIM0, FADE = 28.4, (28.9, 30.4)  # sim clock starts; robot alpha ramp
SIM_STAND, SIM_WALK, SIM_HOLD = 2.0, 4.2, 3.2
END = SIM0 + SIM_STAND + SIM_WALK + SIM_HOLD
SPAWN, YAW, SPEED = (0.43, -1.15), 125.0, 0.4  # open floor diagonal, checked clear of walls and clutter
CAPTIONS = [(0.0, "Video frames"), (6.6, "Camera poses"), (14.0, "Point cloud"), (20.6, "Surface mesh"), (24.3, "Collider"), (28.0, "Simulation")]


def smooth(t, a, b):  # scalars or arrays
    x = np.clip((t - a) / (b - a), 0.0, 1.0)
    return x * x * (3 - 2 * x)


# ---------- data ----------
info = json.loads((SCAN / "mujoco/scene_info.json").read_text())
T = np.array(info["transform_from_scan"])
intr = json.loads((SCAN / "intrinsics.json").read_text())
FX, FY, CX, CY = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
POSES = np.einsum("ij,njk->nik", T, np.loadtxt(SCAN / "poses.txt").reshape(-1, 4, 4))  # camera-to-world, OpenCV axes, leveled
assert len(POSES) == LAST + 1


def color(i):
    return pygame.surfarray.array3d(pygame.image.load(str(SCAN / f"color/{i:06d}.jpg"))).swapaxes(0, 1)


def depth(i):  # 16-bit PNG through ffmpeg (pygame would drop it to 8 bits)
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(SCAN / f"depth/{i:06d}.png"), "-f", "rawvideo",
                          "-pix_fmt", "gray16le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, "<u2").reshape(360, 640) * intr["depth_scale"]


def backproject(i, stride):
    """Depth frame i at its pose: world points, world normals (facing that camera), pixel coords."""
    d = depth(i)[::stride, ::stride]
    v, u = np.mgrid[0:360:stride, 0:640:stride]
    pc = np.stack([(u - CX) / FX * d, (v - CY) / FY * d, d], -1)
    n = np.zeros_like(pc)
    n[1:-1, 1:-1] = np.cross(pc[1:-1, 2:] - pc[1:-1, :-2], pc[2:, 1:-1] - pc[:-2, 1:-1])
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
    facing = -(n * pc).sum(-1) / np.linalg.norm(pc, axis=-1).clip(1e-6)
    valid = d > 0.1
    ok = np.zeros_like(valid)
    ok[1:-1, 1:-1] = valid[1:-1, 1:-1] & valid[1:-1, 2:] & valid[1:-1, :-2] & valid[2:, 1:-1] & valid[:-2, 1:-1]
    ok &= np.abs(facing) > 0.25  # drops depth edges and grazing surfaces
    R, t = POSES[i][:3, :3], POSES[i][:3, 3]
    return pc[ok] @ R.T + t, (n * np.sign(facing)[..., None])[ok] @ R.T, u[ok], v[ok]


def in_room(p):  # the same crop as export_scan.py: wall footprint plus 0.3 m, floor to crop height
    wx = [w["pos"][0] for w in info["walls"]]; wy = [w["pos"][1] for w in info["walls"]]
    return ((p[:, 0] > min(wx) - 0.3) & (p[:, 0] < max(wx) + 0.3) & (p[:, 1] > min(wy) - 0.3) & (p[:, 1] < max(wy) + 0.3)
            & (p[:, 2] > -0.05) & (p[:, 2] < info["crop_height"]))


@functools.lru_cache
def fused_cloud(every=2, stride=3, voxel=0.025):
    """Every other depth frame at its pose, merged on a voxel grid: the depth reconstruct.py fuses into the TSDF.
    Returns points, normals and the first frame that saw each voxel."""
    P, N, F = [], [], []
    for i in range(0, LAST + 1, every):
        p, n, _, _ = backproject(i, stride)
        P.append(p); N.append(n); F.append(np.full(len(p), i))
    P, N, F = np.concatenate(P), np.concatenate(N), np.concatenate(F)
    k = in_room(P)
    P, N, F = P[k], N[k], F[k]
    q = np.floor(P / voxel).astype(np.int64) + 2000
    _, inv, cnt = np.unique((q[:, 0] * 4000 + q[:, 1]) * 4000 + q[:, 2], return_inverse=True, return_counts=True)
    p = np.stack([np.bincount(inv, P[:, j]) for j in range(3)], 1) / cnt[:, None]
    n = np.stack([np.bincount(inv, N[:, j]) for j in range(3)], 1)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    first = np.full(len(cnt), LAST); np.minimum.at(first, inv, F)
    keep = cnt >= 3  # seen at least three times: drops lone noise
    return p[keep], n[keep], first[keep]


# ---------- cameras: (center, world-to-camera rows [right, down, forward], fx, fy, cx, cy), pixel centers at +0.5 ----------
PX, PY, PS = 320, 180, 2  # the capture shows 2x in a centered 1280x720 panel


def capture_cam(i):
    return POSES[i][:3, 3], POSES[i][:3, :3].T, FX * PS, FY * PS, PX + PS * (CX + 0.5), PY + PS * (CY + 0.5)


def free_cam(lookat, dist, az, el):  # MuJoCo's free camera, so dots and MuJoCo renders line up
    a, e = math.radians(az), math.radians(el)
    fwd = np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
    up = np.array([-math.sin(e) * math.cos(a), -math.sin(e) * math.sin(a), math.cos(e)])
    f = H / 2 / math.tan(math.radians(FOVY) / 2)
    return np.asarray(lookat) - dist * fwd, np.stack([np.cross(fwd, up), -up, fwd]), f, f, W / 2, H / 2


def orbit(t, robot_xy=None):
    """lookat, distance, azimuth, elevation of the orbit at film time t."""
    look = np.array([-0.15, -0.4, 0.5])
    if robot_xy is not None:  # drift toward the robot once it is there
        look[:2] += 0.45 * smooth(t, SIM0, SIM0 + 3) * (np.asarray(robot_xy) - look[:2])
    az = 33 - 95 * smooth(t, T_PULL[1] - 1.5, END)
    el = -36 - 12 * smooth(t, 12.0, 24.0)
    dist = 4.4 + 1.0 * smooth(t, 12.0, 18.0) - 1.5 * smooth(t, SIM0 - 1, END - 1)
    return look, dist, az, el


def quat(R):
    q = np.zeros(4); mujoco.mju_mat2Quat(q, R.ravel()); return q


def blend_cam(c0, c1, s):
    q0, q1 = quat(c0[1]), quat(c1[1])
    d = q0 @ q1
    if d < 0:
        q1, d = -q1, -d
    th = math.acos(min(d, 1.0))
    q = q0 if th < 1e-6 else (math.sin((1 - s) * th) * q0 + math.sin(s * th) * q1) / math.sin(th)
    R = np.zeros(9); mujoco.mju_quat2Mat(R, q / np.linalg.norm(q))
    return (c0[0] + s * (c1[0] - c0[0]), R.reshape(3, 3), *(a + s * (b - a) for a, b in zip(c0[2:], c1[2:])))


def facing(cam, p, n):  # 1 for surfaces facing the camera, 0 for back faces, soft in between
    d = cam[0] - p
    return smooth((d * n).sum(1) / np.linalg.norm(d, axis=1).clip(1e-6), -0.05, 0.2)


def project(cam, p):
    C, R, fx, fy, cx, cy = cam
    q = (p - C) @ R.T
    z = q[:, 2]
    ok = z > 0.05
    zs = np.where(ok, z, 1.0)
    return fx * q[:, 0] / zs + cx, fy * q[:, 1] / zs + cy, z, ok


# ---------- drawing: dots and lines splatted with bilinear weights, sized by box sums, alpha = 1 - exp(-sum) ----------
def dots(img, u, v, size, alpha, col):
    n = len(u)
    if n == 0:
        return
    col = np.broadcast_to(np.asarray(col, np.float32), (n, 3))
    size = np.broadcast_to(np.clip(np.rint(size), 1, 4).astype(int), (n,))
    alpha = np.broadcast_to(np.asarray(alpha, np.float32), (n,))
    acc = np.zeros((4, H, W), np.float32)
    x, y = u - 0.5, v - 0.5
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = (x - x0).astype(np.float32), (y - y0).astype(np.float32)
    for s in np.unique(size):
        k = (size == s) & (x0 >= 0) & (y0 >= 0) & (x0 < W - 1) & (y0 < H - 1) & (alpha > 0)
        if not k.any():
            continue
        i00 = y0[k] * W + x0[k]
        idx = np.concatenate([i00, i00 + 1, i00 + W, i00 + W + 1])
        a, gx, gy = alpha[k], fx[k], fy[k]
        w = np.concatenate([(1 - gx) * (1 - gy), gx * (1 - gy), (1 - gx) * gy, gx * gy]) * np.tile(a, 4)
        c = np.tile(col[k], (4, 1))
        layer = np.stack([np.bincount(idx, w, H * W)] + [np.bincount(idx, w * c[:, j], H * W) for j in range(3)])
        layer = layer.reshape(4, H, W).astype(np.float32)
        if s > 1:  # box sum: every pixel the s x s dot covers gets the dot's alpha
            for ax in (1, 2):
                cs = np.cumsum(np.pad(layer, [(0, 0)] + [((s + 1) // 2, s // 2) if d == ax else (0, 0) for d in (1, 2)]), axis=ax)
                layer = np.take(cs, np.arange(s, cs.shape[ax]), axis=ax) - np.take(cs, np.arange(0, cs.shape[ax] - s), axis=ax)
        acc += layer
    A = acc[0]
    cover = 1 - np.exp(-A)
    rgb = acc[1:] / np.maximum(A, 1e-6)
    img *= 1 - cover[..., None]
    img += np.moveaxis(rgb, 0, -1) * cover[..., None]


def lines(img, cam, a3, b3, alpha, col, width=1):
    ua, va, _, oka = project(cam, a3)
    ub, vb, _, okb = project(cam, b3)
    ok = oka & okb & (np.abs(ua) < 4 * W) & (np.abs(ub) < 4 * W) & (np.abs(va) < 4 * H) & (np.abs(vb) < 4 * H)
    ua, va, ub, vb = ua[ok], va[ok], ub[ok], vb[ok]
    alpha = np.broadcast_to(np.asarray(alpha, np.float32), ok.shape)[ok]
    n = np.maximum(np.ceil(np.hypot(ub - ua, vb - va) / 0.5).astype(int), 1)
    seg = np.repeat(np.arange(len(n)), n)
    s = (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n) + 0.5) / np.repeat(n, n)
    u, v = ua[seg] + s * (ub - ua)[seg], va[seg] + s * (vb - va)[seg]
    dots(img, u, v, width, alpha[seg] * 0.5, col)


def frustum_lines(i, depth_m):
    C, Rcw = POSES[i][:3, 3], POSES[i][:3, :3]
    corners = [C + Rcw @ np.array([(u - CX) / FX * depth_m, (v - CY) / FY * depth_m, depth_m]) for u, v in ((0, 0), (640, 0), (640, 360), (0, 360))]
    a = [C] * 4 + corners
    b = corners + corners[1:] + corners[:1]
    return np.array(a), np.array(b)


# ---------- text ----------
pygame.font.init()
_font = pygame.font.Font(FONT, 30)


def text_mask(s):
    return pygame.surfarray.array_alpha(_font.render(s, True, (255, 255, 255))).T.astype(np.float32) / 255


CAP_MASKS = [(text_mask(f"{k + 1:02d}"), text_mask(label)) for k, (_, label) in enumerate(CAPTIONS)]
SUB_MASK = pygame.surfarray.array_alpha(pygame.font.Font(FONT, 22).render("Cleaned view of the collider", True, (255,) * 3)).T / 255.0


def paint(img, mask, x, y, col, a):
    h, w = mask.shape
    m = mask[..., None] * a
    img[y:y + h, x:x + w] = img[y:y + h, x:x + w] * (1 - m) + col * m


def caption(img, t):
    k = max(i for i, (t0, _) in enumerate(CAPTIONS) if t >= t0)
    a = smooth(t, CAPTIONS[k][0], CAPTIONS[k][0] + 0.5) * (1 - smooth(t, END - 0.6, END)) * smooth(t, 0.2, 0.7)
    x, y = 72, H - 72 - 36
    shade = smooth(np.arange(280), 0, 220)[:, None] * (1 - smooth(np.arange(760), 420, 760))[None, :]  # keeps the caption legible over light walls
    img[H - 280:, :760] += 0.8 * smooth(t, T_MESH[0], T_MESH[1]) * shade[..., None] * (BG - img[H - 280:, :760])  # mesh stages on
    for j in range(len(CAPTIONS)):  # progress rail, like the site's stage rail
        w = 44 if j == k else 22
        c = ACC if j == k else (INK * 0.45 + BG * 0.55 if j < k else LINE * 1.6)
        img[y - 22:y - 20, x + 54 * j:x + 54 * j + w] = c
    num, lab = CAP_MASKS[k]
    paint(img, num, x, y, ACC, a)
    paint(img, lab, x + num.shape[1] + 22, y, INK, a)
    sub = smooth(t, *T_CLEAN) * (1 - smooth(t, END - 0.6, END))  # the walls from here on are a cleaned visual, say so
    if sub > 0:
        paint(img, SUB_MASK, x + num.shape[1] + 22, y + 46, INK * 0.6 + BG * 0.4, sub)


# ---------- cleaned view of the collider walls ----------
def clean_walls(cloud, t=0.1, merge=0.3, snap=0.4, min_gap=0.3):
    """A cleaned visual of the collider's 12 wall boxes, which build_scene.py fit after the scan drifted into doubled walls
    (parallel slabs, a rotated copy crossing the room). Walls more than 8 degrees off the room's axes that cross or touch an
    on-axis wall are those copies and are dropped; parallel walls within `merge` m become one at their median plane; ends
    within `snap` m of another wall's line meet it square; and a wall stays open wherever the fused depth shows no wall
    surface for `min_gap` m or more (the doorway, the window). All walls get the tallest scanned wall height.
    Returns boxes as (center xyz, half sizes, yaw, inward normal xy)."""
    ws = info["walls"]
    yaw, half, pos = (np.array([w[k] for w in ws]) for k in ("yaw", "half", "pos"))
    fold = lambda a: (a + np.pi / 4) % (np.pi / 2) - np.pi / 4  # mod 90 degrees
    on = np.abs(fold(yaw - yaw[np.argmax(half[:, 1])])) < np.radians(8)
    th = np.angle((half[on, 1] * np.exp(4j * yaw[on])).sum()) / 4  # room axis: length-weighted, from the on-axis walls
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])  # room frame -> world
    along = np.stack([-np.sin(yaw), np.cos(yaw)], 1)
    ends = np.stack([pos[:, :2] - along * half[:, 1:2], pos[:, :2] + along * half[:, 1:2]], 1)
    def seg_dist(p, a, b):
        ab = b - a
        s = np.clip((p - a) @ ab / (ab @ ab), 0, 1)
        return np.linalg.norm(p - (a + s[:, None] * ab), axis=1)
    for i in np.flatnonzero(~on):  # off-axis: a drift copy if it crosses or touches an on-axis wall
        pts = ends[i, 0] + np.linspace(0, 1, 50)[:, None] * (ends[i, 1] - ends[i, 0])
        assert min(seg_dist(pts, *ends[j]).min() for j in np.flatnonzero(on)) < merge, f"wall_{i:02d} is off-axis and free"
    c = pos[:, :2] @ R  # room frame
    vert = np.abs(np.sin(yaw - th)) < 0.5  # normal along room x: the wall runs along room y
    walls = []
    for i in np.flatnonzero(on):  # merge parallel neighbours
        kind, plane, mid = ("v", c[i, 0], c[i, 1]) if vert[i] else ("h", c[i, 1], c[i, 0])
        lo, hi = mid - half[i, 1], mid + half[i, 1]
        for w in walls:
            if w["kind"] == kind and abs(np.median(w["planes"]) - plane) < merge and lo < w["hi"] + merge and hi > w["lo"] - merge:
                w["planes"].append(plane); w["lo"], w["hi"] = min(w["lo"], lo), max(w["hi"], hi)
                break
        else:
            walls.append({"kind": kind, "planes": [plane], "lo": lo, "hi": hi})
    for w in walls:
        w["plane"] = float(np.median(w["planes"]))
    height = 2 * half[on, 2].max()
    joints = []  # square corners: (wall, which end, the other wall's plane, L corner or T)
    for w in walls:
        for e in ("lo", "hi"):
            for o in walls:
                if o["kind"] != w["kind"] and abs(o["plane"] - w[e]) < snap and o["lo"] - snap < w["plane"] < o["hi"] + snap:
                    joints.append((w, e, o["plane"], min(abs(o["lo"] - w["plane"]), abs(o["hi"] - w["plane"])) < snap))
    for w, e, plane, corner in joints:  # horizontal walls run through L corners, everything else stops at the other face
        out = 1 if e == "hi" else -1
        w[e] = plane + out * t / 2 * (1 if (w["kind"] == "h" and corner) else -1)
    pr = cloud[:, :2] @ R
    centre = POSES[:, :2, 3].mean(0) @ R
    boxes = []
    for w in walls:
        a, b = (0, 1) if w["kind"] == "h" else (1, 0)  # (along, across) room axes
        k = (np.abs(pr[:, b] - w["plane"]) < 0.12) & (cloud[:, 2] > 0.05) & (cloud[:, 2] < height)
        edges = np.arange(w["lo"], w["hi"] + 0.05, 0.05)
        seen = np.histogram(pr[k, a], edges)[0] >= 4
        cuts, i = [], 1
        while i < len(seen) - 1:  # interior runs with no wall surface: openings
            j = i
            while j < len(seen) - 1 and not seen[j]:
                j += 1
            if j - i >= min_gap / 0.05:
                cuts.append((edges[i], edges[j]))
            i = j + 1
        spans = [(w["lo"], w["hi"])]
        for g0, g1 in cuts:
            spans = [q for s0, s1 in spans for q in ((s0, min(s1, g0)), (max(s0, g1), s1)) if q[1] - q[0] > 0.05]
        n = np.zeros(2); n[b] = np.sign(centre[b] - w["plane"])
        for s0, s1 in spans:
            m = np.zeros(2); m[a], m[b] = (s0 + s1) / 2, w["plane"]
            hs = [(s1 - s0) / 2, t / 2, height / 2] if w["kind"] == "h" else [t / 2, (s1 - s0) / 2, height / 2]
            boxes.append(((*(R @ m), height / 2), hs, th, R @ n))
        w["cuts"] = cuts
    print(f"clean walls: room axis {math.degrees(th):.1f} deg, {int((~on).sum())} drift copies dropped, "
          f"{len(walls)} walls from {int(on.sum())}, openings " + ", ".join(f"{w['kind']}@{w['plane']:.2f}: {g0:.2f}..{g1:.2f}"
                                                                          for w in walls for g0, g1 in w["cuts"]))
    return boxes


def walk_clearance(boxes, xy):
    """Smallest gap between the robot's path (pelvis) and a clean wall face, and whether the path stays on the room
    side of every wall, that is inside the room."""
    gap, inside = np.inf, True
    for c, hs, th, n in boxes:
        c, u = np.asarray(c[:2]), np.array([-n[1], n[0]])  # along the wall
        s = np.clip((xy - c) @ u, -max(hs[:2]), max(hs[:2]))
        gap = min(gap, np.linalg.norm(xy - (c + s[:, None] * u), axis=1).min() - min(hs[:2]))
        inside &= bool(((xy - c) @ n > 0).all())
    return gap, inside


def concrete(n=512, tiles=2):
    """Dark concrete floor tiles for one square meter (texuniform, texrepeat 1): soft blotches, grain, a tone per tile, thin joints."""
    rng = np.random.default_rng(3)
    def blotch(k):  # tileable noise: a k x k random grid, upsampled bilinearly with wraparound
        g, x = rng.standard_normal((k, k)), np.arange(n) * k / n
        i, f = x.astype(int), (x - x.astype(int))[:, None]
        rows = g[i] * (1 - f) + g[(i + 1) % k] * f
        return rows[:, i] * (1 - f.T) + rows[:, (i + 1) % k] * f.T
    tile = np.arange(n) * tiles // n
    v = 0.155 + 0.004 * rng.standard_normal((tiles, tiles))[tile][:, tile] + 0.01 * blotch(6) + 0.006 * blotch(24)
    v += 0.005 * rng.standard_normal((n, n))
    joint = (np.arange(n) % (n // tiles) < 2) | (np.arange(n) % (n // tiles) >= n // tiles - 1)
    v[joint] -= 0.045; v[:, joint] -= 0.045
    return (np.clip(np.stack([v, v * 1.05, v * 1.04], -1), 0, 1) * 255).astype(np.uint8)


# ---------- MuJoCo scene, robot and its walk ----------
def build_model(clean=()):
    """The scan's scene.xml for rendering, the G1 attached at the spawn, and `clean` (clean_walls boxes) as hidden visual-only geoms."""
    xml = (SCAN / "mujoco/scene.xml").read_text()
    d = str(SCAN / "mujoco")
    xml = xml.replace('<compiler angle="radian"/>', f'<compiler angle="radian" meshdir="{d}" texturedir="{d}"/>')
    bg = " ".join(f"{c / 255:.4f}" for c in BG)
    xml = xml.replace('rgb1="0.6 0.7 0.8" rgb2="0.2 0.25 0.3"', f'rgb1="{bg}" rgb2="{bg}"')
    xml = xml.replace("<asset>", '<asset>\n    <texture name="grid" type="2d" builtin="flat" mark="edge" rgb1="0.052 0.062 0.062" '
                                 'markrgb="0.16 0.2 0.195" width="400" height="400"/>\n    '
                                 '<material name="grid" texture="grid" texrepeat="2 2" texuniform="true"/>\n    '
                                 # the clean look: matte light gray walls, a darker concrete tile floor (texture added below)
                                 '<material name="floor_clean" texture="concrete" texrepeat="1 1" texuniform="true" specular="0" shininess="0"/>\n    '
                                 '<material name="wall_clean" rgba="0.8 0.81 0.8 1" specular="0" shininess="0"/>')
    xml = xml.replace("</worldbody>", '  <light name="key" pos="0 0 6" dir="0.45 0.3 -0.85" directional="true" diffuse="0 0 0" castshadow="true"/>\n  </worldbody>')
    xml = xml.replace('<geom name="floor" type="plane" size="0 0 0.05" rgba="0.35 0.35 0.33 1"/>',
                      '<geom name="floor" type="plane" size="0 0 0.05" material="grid" group="1"/>')
    assert 'material="grid" group="1"' in xml and f'rgb1="{bg}"' in xml
    spec = mujoco.MjSpec.from_string(xml)
    spec.visual.global_.offwidth, spec.visual.global_.offheight = W, H
    spec.visual.rgba.haze = [*(BG / 255), 1]
    tex = spec.add_texture(name="concrete", type=mujoco.mjtTexture.mjTEXTURE_2D, width=512, height=512, nchannel=3)
    tex.data = concrete().tobytes()
    for i, (c, hs, th, _) in enumerate(clean):
        spec.worldbody.add_geom(name=f"clean_{i:02d}", type=mujoco.mjtGeom.mjGEOM_BOX, size=hs, pos=c, material="wall_clean",
                                quat=[math.cos(th / 2), 0, 0, math.sin(th / 2)], contype=0, conaffinity=0, group=5)
    cfg = load_g1_config()
    frame = spec.worldbody.add_frame(name="g1_spawn", pos=(SPAWN[0], SPAWN[1], 0))
    spec.attach(mujoco.MjSpec.from_file(str(cfg.robot_xml)), prefix="g1_", frame=frame)
    model = spec.compile()
    model.opt.timestep = cfg.simulation_dt
    for g in range(model.ngeom):  # robot visuals to group 4, robot collision shapes to hidden group 5
        if model.body(model.geom_bodyid[g]).name.startswith("g1_"):
            model.geom_group[g] = 4 if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0 else 5
        elif (model.geom(g).name or "").startswith("hull_"):  # clutter hulls apart from the wall boxes: cyan with them, never clean
            model.geom_group[g] = 2
    return model, cfg


def simulate(model, cfg):
    """Stand, walk the open diagonal, stop. Returns qpos sampled at FPS."""
    data = mujoco.MjData(model)
    g1 = G1Controller(model, cfg, "g1_")
    g1.reset(data)
    data.qpos[g1.base_qpos + 3:g1.base_qpos + 7] = [math.cos(math.radians(YAW) / 2), 0, 0, math.sin(math.radians(YAW) / 2)]
    mujoco.mj_forward(model, data)
    traj, cmd = [data.qpos.copy()], np.zeros(3, np.float32)
    for k in range(int((SIM_STAND + SIM_WALK + SIM_HOLD + 0.1) / cfg.simulation_dt)):  # same loop as walk_in_scene.py
        t = data.time
        g1.apply(data)
        mujoco.mj_step(model, data)
        if len(traj) <= data.time * FPS:
            traj.append(data.qpos.copy())
        if k % cfg.control_decimation:
            continue
        cmd[0] = SPEED if SIM_STAND <= t < SIM_STAND + SIM_WALK else 0.0
        g1.update_policy(data, cmd, gait_phase(data))
    pelvis = np.array(traj)[:, g1.base_qpos:g1.base_qpos + 3]
    assert pelvis[:, 2].min() > 0.6, "G1 fell"
    print(f"G1 walked {np.linalg.norm(pelvis[-1, :2] - pelvis[0, :2]):.2f} m, pelvis {pelvis[0, :2].round(2)} -> {pelvis[-1, :2].round(2)}")
    return np.array(traj), g1.base_qpos


class Scene:
    def __init__(self):
        clean = clean_walls(fused_cloud()[0])
        self.model, cfg = build_model(clean)
        self.data = mujoco.MjData(self.model)
        self.traj, self.base = simulate(self.model, cfg)
        gap, inside = walk_clearance(clean, self.traj[:, self.base:self.base + 2])
        print(f"walk vs clean walls: {gap:.2f} m to the nearest face, inside the room: {inside}")
        assert gap > 0.25 and inside, "the cleaned walls cut into the robot's walk"
        self.clean_c = np.array([c[:2] for c, _, _, _ in clean])
        self.clean_n = np.array([n for _, _, _, n in clean])
        self.clean_len = np.array([max(hs[:2]) for _, hs, _, _ in clean])
        m = self.model  # fog is baked into the GL context, so set it before the renderer exists: floor fades into the page
        m.vis.rgba.fog = [*(BG / 255), 1]
        m.vis.map.fogstart, m.vis.map.fogend = 5.5 / m.stat.extent, 11.0 / m.stat.extent
        self.r = mujoco.Renderer(self.model, H, W)
        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        self.opt.sitegroup[:] = 0
        m = self.model
        mat = lambda name: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, name)
        name = lambda g: m.geom(g).name or ""
        self.walls = np.array([name(g).startswith("wall_") for g in range(m.ngeom)])  # the raw collider boxes
        self.clean = np.flatnonzero([name(g).startswith("clean_") for g in range(m.ngeom)])
        m.geom_rgba[(m.geom_group == 2) | self.walls] = [*(ACC / 255), 0.42]
        self.floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.grid, self.floor_clean, self.wall_clean = mat("grid"), mat("floor_clean"), mat("wall_clean")
        self.key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_LIGHT, "key")
        self.lights = m.light_diffuse.copy(), m.vis.headlight.diffuse.copy(), m.vis.headlight.ambient.copy()
        self.scan_mat = mat("scan")
        self.scan_rgba = m.mat_rgba[self.scan_mat].copy()

    def robot_xy(self, t):
        q = self.traj[min(max(int(round((t - SIM0) * FPS)), 0), len(self.traj) - 1)]
        return q[self.base:self.base + 2]

    def cutaway(self, cam_xy, targets, margin=0.5):
        """Opacity per clean wall: 0 where the wall stands between the camera and a target (the room centre, the robot),
        counting `margin` m past its ends; fades as the sightline nears an end. The dollhouse view the raw mesh had."""
        keep = np.ones(len(self.clean))
        u = np.stack([-self.clean_n[:, 1], self.clean_n[:, 0]], 1)
        for tgt in targets:
            d = tgt - cam_xy
            s = ((self.clean_c - cam_xy) * self.clean_n).sum(1) / (self.clean_n @ d)  # where the sightline meets each wall line
            along = np.abs(((cam_xy + s[:, None] * d - self.clean_c) * u).sum(1))
            keep = np.minimum(keep, np.where((s > 0) & (s < 1), 1 - smooth(self.clean_len + margin - along, 0, margin), 1))
        return keep

    def render(self, t, groups, cam, dim=0.0, clean=False):
        """clean: the cleaned walls as matte light gray solids, the concrete floor, a soft key light, fog into the page color."""
        m = self.model
        q = self.traj[min(max(int(round((t - SIM0) * FPS)), 0), len(self.traj) - 1)]
        self.data.qpos[:] = q
        mujoco.mj_forward(m, self.data)
        m.mat_rgba[self.scan_mat] = self.scan_rgba * [1 - dim, 1 - dim, 1 - dim, 1]
        m.geom_group[self.walls] = 5 if clean else 3  # raw collider boxes, or their cleaned view
        m.geom_group[self.clean] = 3 if clean else 5
        if clean:  # walls between the camera and the room or robot fade out
            keep = self.cutaway(free_cam(*cam)[0][:2], [POSES[:, :2, 3].mean(0), self.robot_xy(t)])
            m.geom_rgba[self.clean] = m.mat_rgba[self.wall_clean]
            m.geom_rgba[self.clean, 3] = keep
            m.geom_group[self.clean[keep < 0.02]] = 5
        m.geom_matid[self.floor] = self.floor_clean if clean else self.grid
        m.light_diffuse[:] = self.lights[0]
        m.vis.headlight.diffuse[:], m.vis.headlight.ambient[:] = self.lights[1], self.lights[2]
        if clean:
            m.light_diffuse[:] = 0.22
            m.light_diffuse[self.key] = 0.38
            m.vis.headlight.diffuse[:], m.vis.headlight.ambient[:] = 0.3, 0.42
        self.r.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = clean
        self.opt.geomgroup[:] = 0
        for g in groups:
            self.opt.geomgroup[g] = 1
        (self.cam.lookat[:], self.cam.distance, self.cam.azimuth, self.cam.elevation) = cam
        self.r.update_scene(self.data, self.cam, self.opt)
        return self.r.render().astype(np.float32)


# ---------- the film ----------
class Film:
    def __init__(self):
        rng = np.random.default_rng(1)
        p, n, u, v = backproject(LAST, 2)  # the last frame, nearly every pixel: the match cut
        self.match_p, self.match_n = p, n
        self.match_c = color(LAST)[v, u].astype(np.float32)
        self.match_sparse = np.zeros(len(p), bool); self.match_sparse[rng.choice(len(p), SPARSE_N, replace=False)] = True
        sp, sn, sk = [], [], []
        for i in range(0, LAST, KEY):  # LAST's sparse dots are the match subset above
            p, n, _, _ = backproject(i, 1)
            k = rng.choice(len(p), min(SPARSE_N, len(p)), replace=False)
            sp.append(p[k]); sn.append(n[k]); sk.append(np.full(len(k), i))
        self.sparse_p, self.sparse_n, self.sparse_k = np.concatenate(sp), np.concatenate(sn), np.concatenate(sk)
        self.cloud_p, self.cloud_n, first = fused_cloud()
        self.cloud_a = 0.3 + 0.35 * np.abs(self.cloud_n[:, 2])  # tabletops and ledges brighter, like engine.js
        self.cloud_t = T_CLOUD[0] + (T_CLOUD[1] - T_CLOUD[0] - 0.4) * first / LAST + rng.uniform(0, 0.4, len(first))
        fr = [frustum_lines(i, 0.22) for i in range(0, LAST + 1, 16)]
        self.trail = np.concatenate([f[0] for f in fr]), np.concatenate([f[1] for f in fr]), np.repeat(np.arange(0, LAST + 1, 16), 8)
        print(f"{len(self.match_p)} match dots, {len(self.sparse_p) + SPARSE_N} sparse dots from {LAST // KEY + 1} keyframes, "
              f"{len(self.cloud_p)} fused depth points")
        self.scene = Scene()
        self._frames = {}

    def cam(self, t):
        rxy = self.scene.robot_xy(t) if t >= SIM0 else None
        o = orbit(t, rxy)
        if t >= T_PULL[1]:
            return free_cam(*o), o
        return blend_cam(capture_cam(LAST), free_cam(*o), smooth(t, *T_PULL)), o

    def point_size(self, z, f, k=0.008):
        return np.clip(k * f / np.maximum(z, 0.1), 1, 3)

    def frame(self, t):
        cam, o = self.cam(t)
        img = np.empty((H, W, 3), np.float32); img[:] = BG
        mesh_in = smooth(t, T_MESH[0], T_MESH[0] + 1.2)
        if mesh_in > 0:  # MuJoCo stages
            floor = smooth(t, *T_COLL)  # the floor plane is a collider too, and stays for the simulation
            clean = smooth(t, *T_CLEAN)
            if clean < 1:  # raw mesh, then the cyan collider shapes over it
                base = self.scene.render(t, [0, 1] if floor >= 1 else [0], o)
                if 0 < floor < 1:
                    base += floor * (self.scene.render(t, [0, 1], o) - base)
                if floor > 0:
                    base += floor * (self.scene.render(t, [0, 1, 2, 3], o, dim=0.45) - base)
            if clean > 0:  # only the floor plane and the 12 wall boxes, finished cleanly; hulls stay physics only
                c = self.scene.render(t, [1, 3], o, clean=True)
                base = c if clean >= 1 else base + clean * (c - base)
            if t >= FADE[0]:
                a = smooth(t, *FADE)
                withr = self.scene.render(t, [1, 3, 4], o, clean=True)
                mask = np.clip((withr - base).sum(-1, keepdims=True) / 40, 0, 1)  # the robot, not its shadow
                tint = (1 - smooth(t, FADE[0] + 0.6, FADE[1] + 0.8)) * mask * 0.7
                withr += tint * (ACC - withr)
                base += a * (withr - base)
            img += mesh_in * (base - img)
            ring = smooth(t, FADE[0] - 0.4, FADE[0]) * (1 - smooth(t, FADE[1], FADE[1] + 1.0))
            if ring > 0:  # spawn marker under the robot
                th = np.linspace(0, 2 * np.pi, 97)
                c = np.stack([SPAWN[0] + 0.38 * np.cos(th), SPAWN[1] + 0.38 * np.sin(th), np.full(97, 0.01)], 1)
                lines(img, cam, c[:-1], c[1:], 0.9 * ring, ACC, 2)
        # dots, from the match cut to the mesh; a dot fades when its surface faces away, as MuJoCo culls back faces
        dots_out = 1 - smooth(t, T_MESH[0] + 0.6, T_MESH[1])
        if t >= T_MATCH[0] and dots_out > 0:
            fade = (1 - 0.7 * smooth(t, 15.0, 17.0)) * dots_out
            u, v, z, ok = project(cam, self.match_p)
            thin = np.where(self.match_sparse, 1.0, 1 - smooth(t, 7.2, 8.8))
            a = smooth(t, T_MATCH[0], T_MATCH[0] + 0.5) * thin * fade * facing(cam, self.match_p, self.match_n)
            gray = smooth(t, 7.0, 9.0)
            col = self.match_c * (1 - gray) + INK * gray
            dots(img, u[ok], v[ok], self.point_size(z[ok], cam[2], 0.003 + 0.005 * gray), 0.85 * a[ok], col[ok])
            ci = LAST * smooth(t, *T_PATH)
            u, v, z, ok = project(cam, self.sparse_p)
            born = np.clip((ci - self.sparse_k) / 12 + 0.2, 0, 1) * (ci > 0)
            a = born * fade * facing(cam, self.sparse_p, self.sparse_n)
            dots(img, u[ok], v[ok], self.point_size(z[ok], cam[2]), 0.85 * a[ok], INK)
            grow = smooth(t - self.cloud_t, 0, 0.35) * dots_out
            if grow.any():
                u, v, z, ok = project(cam, self.cloud_p)
                a = self.cloud_a * grow * facing(cam, self.cloud_p, self.cloud_n)
                dots(img, u[ok], v[ok], self.point_size(z[ok], cam[2]), a[ok], INK)
            path_a = smooth(t, T_PULL[0], T_PULL[0] + 0.6) * (1 - smooth(t, 15.5, 17.0))
            if path_a > 0:
                a3, b3 = frustum_lines(LAST, 0.3)
                lines(img, cam, a3, b3, 0.95 * path_a, ACC, 2)
                n = int(ci)
                if n >= 1:
                    c = POSES[:n + 1, :3, 3]
                    lines(img, cam, c[:-1], c[1:], 0.9 * path_a, ACC, 2)
                    ta, tb, ti = self.trail
                    k = ti <= n
                    lines(img, cam, ta[k], tb[k], 0.45 * path_a, ACC)
                    if n < LAST:
                        a3, b3 = frustum_lines(n, 0.3)
                        lines(img, cam, a3, b3, 0.95 * path_a, ACC, 2)
        # the capture, in its panel, on top until it dissolves
        img_a = 1 - smooth(t, T_MATCH[0] + 0.3, T_MATCH[1])
        if img_a > 0:
            i = min(int(t * 60), LAST)
            if i not in self._frames:
                s = pygame.transform.smoothscale(pygame.surfarray.make_surface(color(i).swapaxes(0, 1)), (640 * PS, 360 * PS))
                self._frames = {i: pygame.surfarray.array3d(s).swapaxes(0, 1).astype(np.float32)}
            sub = img[PY:PY + 360 * PS, PX:PX + 640 * PS]
            sub += img_a * (self._frames[i] - sub)
            edge = img_a * 0.9
            for sl in ((slice(PY - 1, PY), slice(PX - 1, PX + 640 * PS + 1)), (slice(PY + 360 * PS, PY + 360 * PS + 1), slice(PX - 1, PX + 640 * PS + 1)),
                       (slice(PY, PY + 360 * PS), slice(PX - 1, PX)), (slice(PY, PY + 360 * PS), slice(PX + 640 * PS, PX + 640 * PS + 1))):
                img[sl] += edge * (LINE * 1.6 - img[sl])
        caption(img, t)
        img = BG + (img - BG) * (smooth(t, 0.0, 0.5) * (1 - smooth(t, END - 0.5, END)))
        return np.clip(img, 0, 255).astype(np.uint8)


def check():
    """free_cam vs the GL camera MuJoCo builds for the same free camera: position, axes and focal length must agree,
    or the dots would not line up with the MuJoCo renders they crossfade into."""
    sc = Scene()
    for o in ([-0.15, -0.4, 0.5], 6.0, 33.0, -30.0), ([0.0, -0.5, 0.5], 4.5, -62.0, -48.0):
        sc.render(SIM0, [0], o)
        gl = mujoco.mjv_averageCamera(sc.r.scene.camera[0], sc.r.scene.camera[1])  # mono render = mean of the two eyes
        C, R, f, _, _, _ = free_cam(*o)
        f_gl = H * gl.frustum_near / (gl.frustum_top - gl.frustum_bottom)
        err = [np.abs(C - gl.pos).max(), np.abs(R[2] - gl.forward).max(), np.abs(-R[1] - gl.up).max(), abs(f - f_gl) / f, abs(gl.frustum_center)]
        print(f"view {o}: position {err[0]:.1e} m, forward {err[1]:.1e}, up {err[2]:.1e}, focal {err[3]:.1e}, center {err[4]:.1e}")
        assert max(err) < 1e-4 and abs(gl.frustum_bottom + gl.frustum_top) < 1e-9, "free_cam does not match MuJoCo's camera"
    print("check passed")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--still", type=float, nargs="*", help="write PNG stills at these times instead of the film")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    if a.check:
        return check()
    film = Film()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.still:
        for t in a.still:
            p = a.out.with_name(f"still_{t:05.2f}.png")
            pygame.image.save(pygame.surfarray.make_surface(film.frame(t).swapaxes(0, 1)), str(p))
            print(p)
        return
    ff = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
                           "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                           str(a.out)], stdin=subprocess.PIPE)
    n = int(round(END * FPS))
    for i in range(n):
        ff.stdin.write(film.frame(i / FPS).tobytes())
        if i % 60 == 0:
            print(f"{i}/{n}", flush=True)
    ff.stdin.close(); ff.wait()
    print(a.out)


if __name__ == "__main__":
    main()
