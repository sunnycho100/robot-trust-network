"""3D worker skeleton from ordinary calibrated video (no wearable sensors).

Input: per-frame candidate poses from `pose2d` (MediaPipe Pose Landmarker: 33 image landmarks with visibility,
plus metric body-relative "world" landmarks) for every calibrated camera that filmed the clip, and the s0 K, R, t.

1. Worker selection (per camera). Candidates are linked into tracklets by image-space continuity; duplicates of
   one person are merged. Each tracklet is scored by persistence x image size x time spent in the work area
   (floor position inside the given work region, e.g. the conveyor band, when known). The worker track is the
   best tracklet extended by every tracklet that does not overlap it in time and continues it in floor
   position; bystanders (other people in the background) therefore drop out.
2. Monocular 3D (one camera). Posture = the model's metric body-relative 3D, rotated from camera to world axes
   (R^T, with a perspective correction that turns the optical axis onto the ray through the body). Global
   position = least-squares translation (and, per track, a metric scale) that reprojects the body onto its 2D
   landmarks with the visible feet on the floor plane z = 0; frames without visible feet use the reprojection
   plus the track's median pelvis height above the floor as a prior.
3. Multi-view (>= 2 cameras of the same clip). Each joint is triangulated (DLT, visibility-weighted) from the
   cameras' worker detections; the left/right labelling of the second view is chosen by reprojection error
   (front/back views of a person are easily mirrored by single-view models); joints whose reprojection error
   exceeds `REPROJ_MAX_PX` fall back to the best camera's monocular joint, translated onto the triangulated
   joints of that frame.
4. Temporal processing. Hampel outlier rejection, gap interpolation (gaps up to `GAP_FILL_S` keep `valid`),
   Savitzky-Golay smoothing, then a fixed-length skeleton: bone lengths = median over valid frames and every
   frame is rebuilt by forward kinematics from the pelvis along the measured bone directions, so lengths are
   exactly rigid.

Output joints are metres, world z-up, with names compatible with the downstream kinematic features.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from rcd.s1_scene.pose2d import LM, LR_SWAP

NAMES = ["Hips", "Chest", "Neck", "Head", "RightShoulder", "RightElbow", "RightWrist", "LeftShoulder",
         "LeftElbow", "LeftWrist", "RightHip", "RightKnee", "RightAnkle", "RightToe", "LeftHip", "LeftKnee",
         "LeftAnkle", "LeftToe"]
# forward-kinematics tree (child -> parent), in topological order
PARENT = {"Chest": "Hips", "Neck": "Chest", "Head": "Neck", "RightShoulder": "Neck", "RightElbow": "RightShoulder",
          "RightWrist": "RightElbow", "LeftShoulder": "Neck", "LeftElbow": "LeftShoulder", "LeftWrist": "LeftElbow",
          "RightHip": "Hips", "RightKnee": "RightHip", "RightAnkle": "RightKnee", "RightToe": "RightAnkle",
          "LeftHip": "Hips", "LeftKnee": "LeftHip", "LeftAnkle": "LeftKnee", "LeftToe": "LeftAnkle"}

VIS_MIN = 0.5          # landmark visibility to count as observed
REPROJ_MAX_PX = 30.0   # triangulated joint rejected above this reprojection error (1280x720 images)
GAP_FILL_S = 1.0       # gaps up to this long are interpolated and stay valid
NECK_ABOVE_SHOULDERS = 0.06   # m along the trunk: cervico-thoracic joint above the shoulder-joint line
FOOT_LM = [LM["l_heel"], LM["r_heel"], LM["l_foot"], LM["r_foot"]]
SOLE_Z = 0.02          # heel / toe-tip landmarks are ~2 cm above the walking surface


# ---------------------------------------------------------------- geometry helpers
def project(cam: dict, X: np.ndarray) -> np.ndarray:
    x = (cam["K"] @ (cam["R"] @ np.asarray(X, float).reshape(-1, 3).T + cam["t"].reshape(3, 1))).T
    return x[:, :2] / x[:, 2:3]


def _ray_rotation(cam: dict, uv) -> np.ndarray:
    """Rotation turning the camera optical axis onto the viewing ray through pixel uv (camera coordinates)."""
    r = np.linalg.solve(cam["K"], np.array([uv[0], uv[1], 1.0]))
    r /= np.linalg.norm(r)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, r)
    s, c = np.linalg.norm(v), float(z @ r)
    if s < 1e-9:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / s ** 2)


def triangulate(cams: list[dict], uv: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted DLT for one point. uv (V,2), w (V,)."""
    A = []
    for cam, (u, v), wi in zip(cams, uv, w):
        P = cam["P"]
        A.append(wi * (u * P[2] - P[0]))
        A.append(wi * (v * P[2] - P[1]))
    _, _, Vt = np.linalg.svd(np.asarray(A))
    X = Vt[-1]
    return X[:3] / X[3]


# ---------------------------------------------------------------- named joints from the 33 landmarks
def named(X: np.ndarray) -> np.ndarray:
    """(..., 33, 3) landmarks -> (..., len(NAMES), 3) joints."""
    g = lambda k: X[..., LM[k], :]
    lh, rh, ls, rs = g("l_hip"), g("r_hip"), g("l_shoulder"), g("r_shoulder")
    hips = 0.5 * (lh + rh)
    sh = 0.5 * (ls + rs)
    tr = sh - hips
    tr = tr / (np.linalg.norm(tr, axis=-1, keepdims=True) + 1e-9)
    neck = sh + NECK_ABOVE_SHOULDERS * tr
    head = 0.5 * (g("l_ear") + g("r_ear"))
    chest = hips + 0.45 * (neck - hips)
    J = {"Hips": hips, "Chest": chest, "Neck": neck, "Head": head,
         "RightShoulder": rs, "RightElbow": g("r_elbow"), "RightWrist": g("r_wrist"),
         "LeftShoulder": ls, "LeftElbow": g("l_elbow"), "LeftWrist": g("l_wrist"),
         "RightHip": rh, "RightKnee": g("r_knee"), "RightAnkle": g("r_ankle"), "RightToe": g("r_foot"),
         "LeftHip": lh, "LeftKnee": g("l_knee"), "LeftAnkle": g("l_ankle"), "LeftToe": g("l_foot")}
    return np.stack([J[n] for n in NAMES], -2)


# ---------------------------------------------------------------- candidates and worker tracks
def _bbox(img: np.ndarray) -> np.ndarray:
    """(N,P,33,5) -> (N,P,4) u0,v0,u1,v1 over landmarks with presence."""
    u, v = img[..., 0], img[..., 1]
    return np.stack([np.nanmin(u, -1), np.nanmin(v, -1), np.nanmax(u, -1), np.nanmax(v, -1)], -1)


def floor_point(cam: dict, uv) -> np.ndarray:
    K, R, t = cam["K"], cam["R"], cam["t"]
    C = -R.T @ t
    d = R.T @ np.linalg.solve(K, np.array([uv[0], uv[1], 1.0]))
    if d[2] >= -1e-6:
        return np.array([np.nan, np.nan])
    return (C + (-C[2] / d[2]) * d)[:2]


def tracklets(img: np.ndarray, max_jump: float = 0.6, max_miss: int = 8) -> np.ndarray:
    """Link candidates over frames: (N,P) tracklet id (-1 = none). Linking by hip-centre distance relative to the
    person's image height; duplicate detections of one person (same place, same frame) are dropped."""
    N, P = img.shape[:2]
    ok = np.isfinite(img[..., 0, 0])
    hip = 0.5 * (img[..., LM["l_hip"], :2] + img[..., LM["r_hip"], :2])
    bb = _bbox(img)
    hgt = np.maximum(bb[..., 3] - bb[..., 1], 20.0)
    for i in range(N):  # duplicates: same person detected twice
        idx = [p for p in range(P) if ok[i, p]]
        for a in idx:
            for b in idx:
                if a < b and ok[i, a] and ok[i, b] and np.linalg.norm(hip[i, a] - hip[i, b]) < 0.15 * max(hgt[i, a], hgt[i, b]):
                    drop = b if np.nanmean(img[i, a, :, 3]) >= np.nanmean(img[i, b, :, 3]) else a
                    ok[i, drop] = False
    tid = np.full((N, P), -1, int)
    active: dict[int, tuple[int, np.ndarray, float]] = {}   # id -> (last frame, hip, height)
    nxt = 0
    for i in range(N):
        cands = [p for p in range(P) if ok[i, p]]
        pairs = []
        for k, (lf, h, hg) in active.items():
            if i - lf > max_miss:
                continue
            for p in cands:
                d = np.linalg.norm(hip[i, p] - h) / max(hg, hgt[i, p])
                if d < max_jump * (1 + 0.25 * (i - lf - 1)):
                    pairs.append((d, k, p))
        used_k, used_p = set(), set()
        for d, k, p in sorted(pairs):
            if k in used_k or p in used_p:
                continue
            tid[i, p] = k
            used_k.add(k)
            used_p.add(p)
        for p in cands:
            if p not in used_p:
                tid[i, p] = nxt
                nxt += 1
        for p in cands:
            active[tid[i, p]] = (i, hip[i, p], hgt[i, p])
    return tid


def candidate_floor_xy(cam: dict, img: np.ndarray) -> np.ndarray:
    """(N,P,2) floor position of each candidate: visible feet ray-floor intersection, else NaN."""
    N, P = img.shape[:2]
    out = np.full((N, P, 2), np.nan)
    for i in range(N):
        for p in range(P):
            if not np.isfinite(img[i, p, 0, 0]):
                continue
            f = img[i, p, FOOT_LM]
            vis = f[:, 3] > VIS_MIN
            if vis.sum() >= 1:
                uv = f[vis, :2]
                # lowest visible foot point in the image = contact with the floor
                out[i, p] = floor_point(cam, uv[np.argmax(uv[:, 1])])
    return out


def select_worker(img: np.ndarray, cam: dict, work_region=None) -> np.ndarray:
    """(N,) index of the worker candidate per frame (-1 none) for one camera clip.

    work_region: callable xy -> bool (e.g. inside the conveyor band +- margin) or None."""
    N, P = img.shape[:2]
    tid = tracklets(img)
    bb = _bbox(img)
    hgt = bb[..., 3] - bb[..., 1]
    vis = np.nanmean(img[..., 3], -1)
    fxy = candidate_floor_xy(cam, img)
    ids = sorted(set(tid[tid >= 0].tolist()))
    info = {}
    for k in ids:
        m = tid == k
        fr = np.where(m.any(1))[0]
        h = np.nanmedian(hgt[m])
        v = np.nanmedian(vis[m])
        xy = fxy[m]
        inside = 1.0
        if work_region is not None:
            okxy = np.isfinite(xy[:, 0])
            inside = float(np.mean([work_region(p) for p in xy[okxy]])) if okxy.any() else 0.3
        info[k] = {"f0": fr[0], "f1": fr[-1], "n": len(fr), "score": len(fr) * h * v * (0.2 + inside),
                   "xy0": np.nanmedian(xy[:max(1, len(xy) // 10 + 1)], 0) if len(xy) else np.array([np.nan, np.nan]),
                   "xy1": np.nanmedian(xy[-max(1, len(xy) // 10 + 1):], 0) if len(xy) else np.array([np.nan, np.nan]),
                   "h": h}
    sel = np.full(N, -1, int)
    if not ids:
        return sel
    # greedy: best-scoring tracklets first, each accepted if it does not overlap the frames already taken
    order = sorted(ids, key=lambda k: -info[k]["score"])
    best = info[order[0]]["score"]
    taken = np.zeros(N, bool)
    for k in order:
        I = info[k]
        if I["score"] < 0.01 * best or I["n"] < 3:
            continue
        m = tid == k
        fr = np.where(m.any(1))[0]
        if taken[fr].mean() > 0.2:
            continue
        free = fr[~taken[fr]]
        for i in free:
            sel[i] = int(np.where(m[i])[0][0])
        taken[free] = True
    return sel


# ---------------------------------------------------------------- monocular 3D
ORIENT = "level"


def body_axes(cam: dict, uv, ray_correct: bool = True, orient: str | None = None) -> np.ndarray:
    """3x3 matrix whose columns are the world directions of the model's x (right), y (down), z (away) axes.

    "camera": the model frame is the camera frame (R^T), optionally turned onto the viewing ray through uv.
    "level": the model frame is a levelled (gravity-aligned) view along the horizontal direction of that ray --
    single-image body models are trained on mostly level cameras and return an upright-referenced body, so
    the camera pitch must not be applied a second time."""
    orient = orient or ORIENT
    R = cam["R"]
    if orient == "camera":
        Rr = _ray_rotation(cam, uv) if ray_correct else np.eye(3)
        return R.T @ Rr
    d = R.T @ np.linalg.solve(cam["K"], np.array([uv[0], uv[1], 1.0])) if ray_correct else R[2].copy()
    f = np.array([d[0], d[1], 0.0])
    f /= np.linalg.norm(f) + 1e-12
    up = np.array([0.0, 0.0, 1.0])
    return np.stack([np.cross(f, up), -up, f], 1)


LEG_CHAINS = {"l": ["l_hip", "l_knee", "l_ankle", "l_heel", "l_foot"], "r": ["r_hip", "r_knee", "r_ankle", "r_heel", "r_foot"]}


def _occluded_legs_standing(Y: np.ndarray, lm: np.ndarray, cam: dict) -> np.ndarray:
    """Occluded or out-of-image legs (knee and ankle not visible) are hallucinated by single-view models, often
    crouched. Replace them by the standing prior -- the posture of a line worker whose legs are hidden behind
    the workpiece -- straight down from the hip with the model's own segment lengths (heel/toe kept relative to
    the ankle)."""
    W, H = cam.get("image_size", (1280, 720))
    seen = lambda k: lm[LM[k], 3] >= VIS_MIN and 0 <= lm[LM[k], 0] < W and 0 <= lm[LM[k], 1] < H
    Y = Y.copy()
    for side, ch in LEG_CHAINS.items():
        if seen(ch[1]) or seen(ch[2]):
            continue
        hip, knee, ank = (Y[LM[k]] for k in ch[:3])
        l1, l2 = np.linalg.norm(knee - hip), np.linalg.norm(ank - knee)
        new_knee = hip + np.array([0, 0, -l1])
        new_ank = new_knee + np.array([0, 0, -l2])
        for k in ch[3:]:
            Y[LM[k]] = new_ank + (Y[LM[k]] - ank)
        Y[LM[ch[1]]], Y[LM[ch[2]]] = new_knee, new_ank
    return Y


def mono_frame(cam: dict, lm: np.ndarray, wl: np.ndarray, scale: float | None = None, hip_z: float | None = None,
               ray_correct: bool = True) -> tuple[np.ndarray, float, bool]:
    """World-frame landmarks (33,3) for one detection. Returns (X, scale, feet_used)."""
    hipuv = 0.5 * (lm[LM["l_hip"], :2] + lm[LM["r_hip"], :2])
    Y = wl.astype(float) @ body_axes(cam, hipuv, ray_correct).T     # model axes -> world axes
    Y = _occluded_legs_standing(Y, lm, cam)
    Kinv = np.linalg.inv(cam["K"])
    R, t = cam["R"], cam["t"]
    w = np.clip(lm[:, 3], 0, 1)
    use = w > 0.3
    rows, rhs = [], []
    for j in np.where(use)[0]:
        m = Kinv @ np.array([lm[j, 0], lm[j, 1], 1.0])
        for k in (0, 1):
            a_ = R[k] - m[k] * R[2]
            rows.append(w[j] * np.r_[a_, a_ @ Y[j]])
            rhs.append(-w[j] * (t[k] - m[k] * t[2]))
    fv = lm[FOOT_LM, 3]
    feet = bool((fv > VIS_MIN).sum() >= 2)
    fz = Y[FOOT_LM, 2]
    jf = FOOT_LM[int(np.argmin(np.where(fv > VIS_MIN, fz, np.inf)))] if feet else FOOT_LM[int(np.argmin(fz))]
    wf = 6.0 if feet else 1.5
    if hip_z is not None and not feet:
        hz = 0.5 * (Y[LM["l_hip"], 2] + Y[LM["r_hip"], 2])
        rows.append(wf * np.r_[0, 0, 1, hz])
        rhs.append(wf * hip_z)
    else:
        rows.append(wf * np.r_[0, 0, 1, Y[jf, 2]])
        rhs.append(wf * SOLE_Z)
    A, b = np.asarray(rows), np.asarray(rhs)
    if scale is None:
        sol = np.linalg.lstsq(A, b, rcond=None)[0]
        T, s = sol[:3], float(sol[3])
    else:
        s = scale
        T = np.linalg.lstsq(A[:, :3], b - A[:, 3] * s, rcond=None)[0]
    return s * Y + T, s, feet


STATURE_M = 1.70        # adult working-population mean stature (prior when the metric scale is not observable)


def anthropometric_scale(wl: np.ndarray) -> float:
    """Scale of the model's body (world landmarks, (N,33,3)) to an adult of STATURE_M: hip-to-ankle leg length
    0.491 H and hip-centre to shoulder-centre 0.288 H (Winter's segment proportions), median over frames.
    Where both are measurable the feet-on-floor scale agrees with this prior to a few percent."""
    if len(wl) == 0:
        return 1.0
    g = lambda k: wl[:, LM[k]]
    leg = 0.5 * (np.linalg.norm(g("l_hip") - g("l_knee"), axis=1) + np.linalg.norm(g("l_knee") - g("l_ankle"), axis=1)
                 + np.linalg.norm(g("r_hip") - g("r_knee"), axis=1) + np.linalg.norm(g("r_knee") - g("r_ankle"), axis=1))
    tr = np.linalg.norm(0.5 * (g("l_shoulder") + g("r_shoulder")) - 0.5 * (g("l_hip") + g("r_hip")), axis=1)
    m = np.nanmedian(leg + tr)
    return float(np.clip((0.491 + 0.288) * STATURE_M / m, 0.85, 1.25)) if np.isfinite(m) and m > 0 else 1.0


def mono_track(cam: dict, img: np.ndarray, wld: np.ndarray, sel: np.ndarray, ray_correct: bool = True) -> dict:
    """World landmarks for the selected worker in every frame of one camera clip (monocular path)."""
    N = len(sel)
    X = np.full((N, 33, 3), np.nan)
    vis = np.zeros((N, 33))
    ok = sel >= 0
    lm = np.full((N, 33, 5), np.nan)
    wl = np.full((N, 33, 3), np.nan)
    lm[ok] = img[np.where(ok)[0], sel[ok]]
    wl[ok] = wld[np.where(ok)[0], sel[ok]]
    ok &= np.isfinite(wl[:, 0, 0])
    # metric scale of the body model for this person: from frames with both feet visible on the floor
    ss = []
    for i in np.where(ok)[0]:
        if (lm[i, FOOT_LM, 3] > VIS_MIN).sum() >= 3 and (lm[i, :, 3] > VIS_MIN).mean() > 0.7:
            _, s_, _ = mono_frame(cam, lm[i], wl[i], None, None, ray_correct)
            if 0.6 < s_ < 1.6:
                ss.append(s_)
    if len(ss) >= 10:
        scale, scale_src = float(np.clip(np.median(ss), 0.85, 1.25)), "feet on the floor"
    else:
        scale, scale_src = anthropometric_scale(wl[ok]), "stature prior"
    feet = np.zeros(N, bool)
    for i in np.where(ok)[0]:
        X[i], _, feet[i] = mono_frame(cam, lm[i], wl[i], scale, None, ray_correct)
    hz = 0.5 * (X[:, LM["l_hip"], 2] + X[:, LM["r_hip"], 2])
    hip_z = float(np.nanmedian(hz[feet])) if feet.sum() >= 10 else 0.95
    for i in np.where(ok & ~feet)[0]:
        X[i], _, _ = mono_frame(cam, lm[i], wl[i], scale, hip_z, ray_correct)
    vis[ok] = lm[ok, :, 3]
    return {"X": X, "lm": lm, "vis": vis, "ok": ok, "scale": scale, "scale_n": len(ss), "scale_source": scale_src, "hip_z": hip_z,
            "feet_frac": float(feet[ok].mean()) if ok.any() else 0.0}


TORSO = [LM["l_shoulder"], LM["r_shoulder"], LM["l_hip"], LM["r_hip"]]
TORSO_VIS = 0.5
INSIDE_MIN = 0.75   # share of the 33 landmarks inside the image: a person cut by the image border (more than a quarter of the body outside) is not lifted from that view alone


def torso_ok(view: dict, cam: dict, i: int) -> bool:
    """Monocular quality gate: shoulders and hips confidently visible and inside the image (a truncated or
    hidden torso makes the single-view metric lift unreliable)."""
    if not view["ok"][i]:
        return False
    W, H = cam.get("image_size", (1280, 720))
    allm = view["lm"][i]
    inside = (allm[:, 0] > 2) & (allm[:, 0] < W - 2) & (allm[:, 1] > 2) & (allm[:, 1] < H - 2)
    lm = allm[TORSO]
    return bool((lm[:, 3] >= TORSO_VIS).all() and inside[TORSO].all() and inside.mean() >= INSIDE_MIN)


def fuse(views: list[dict], cams: list[dict], min_joints: int = 8, gate: bool = False) -> dict:
    """Per frame: triangulate the worker's landmarks from all cameras that see him, monocular fallback.
    gate=True: the monocular fallback only uses views passing `torso_ok` (else the frame is left empty)."""
    N = len(views[0]["ok"])
    X = np.full((N, 33, 3), np.nan)
    mode = np.zeros(N, int)            # 0 none, 1 mono, 2 multi
    err = np.full(N, np.nan)
    for c in cams:
        c["P"] = c["K"] @ np.hstack([c["R"], c["t"].reshape(3, 1)])
    last = None
    for i in range(N):
        if i > 0 and np.isfinite(X[i - 1, 0, 0]):
            last = (i - 1, 0.5 * (X[i - 1, LM["l_hip"]] + X[i - 1, LM["r_hip"]]))
        vs = [k for k, v in enumerate(views) if v["ok"][i]]
        if not vs:
            continue
        good = [k for k in vs if torso_ok(views[k], cams[k], i)] if gate else vs
        pool = good or vs

        def score(k):
            sc = float(np.nanmean(views[k]["vis"][i]))
            if last is not None and i - last[0] <= 15:   # prefer the view consistent with the recent track
                hip = 0.5 * (views[k]["X"][i, LM["l_hip"]] + views[k]["X"][i, LM["r_hip"]])
                sc -= float(np.linalg.norm(hip[:2] - last[1][:2]))
            return sc
        best = max(pool, key=score)
        if good:
            X[i] = views[best]["X"][i]
            mode[i] = 1
        if len(vs) < 2:
            continue
        a, b = vs[0], vs[1]
        la, lb = views[a]["lm"][i], views[b]["lm"][i]
        res = None
        for perm in (np.arange(33), LR_SWAP):
            lbp = lb[perm]
            T = np.full((33, 3), np.nan)
            e = np.full(33, np.inf)
            for j in range(33):
                wa, wb = la[j, 3], lbp[j, 3]
                if wa < 0.3 or wb < 0.3:
                    continue
                T[j] = triangulate([cams[a], cams[b]], np.stack([la[j, :2], lbp[j, :2]]), np.array([wa, wb]))
                e[j] = max(np.linalg.norm(project(cams[a], T[j])[0] - la[j, :2]),
                           np.linalg.norm(project(cams[b], T[j])[0] - lbp[j, :2]))
            acc = e < REPROJ_MAX_PX
            score = (acc.sum(), -np.median(e[np.isfinite(e)]) if np.isfinite(e).any() else -1e9)
            if res is None or score > res[0]:
                res = (score, T, acc, e)
        _, T, acc, e = res
        if acc.sum() >= min_joints:
            base = views[best]["X"][i]
            off = np.median(T[acc] - base[acc], 0)
            Xi = base + off
            Xi[acc] = T[acc]
            X[i] = Xi
            mode[i] = 2
            err[i] = float(np.median(e[acc]))
    return {"X": X, "mode": mode, "reproj_px": err}


def _hampel(x: np.ndarray, w: int = 7, k: float = 3.0, floor: float = 0.12) -> np.ndarray:
    from scipy.ndimage import median_filter
    y = x.copy()
    ok = np.isfinite(x)
    if ok.sum() < w:
        return y
    xi = _interp(x)
    med = median_filter(xi, size=w, mode="nearest")
    mad = median_filter(np.abs(xi - med), size=w, mode="nearest") * 1.4826
    bad = ok & (np.abs(x - med) > np.maximum(k * mad, floor))
    y[bad] = np.nan
    return y


def _interp(x: np.ndarray) -> np.ndarray:
    ok = np.isfinite(x)
    if not ok.any():
        return np.zeros_like(x)
    i = np.arange(len(x))
    return np.interp(i, i[ok], x[ok])


SG_S = 1.0   # Savitzky-Golay window (s): video joint noise is mostly frame-to-frame jitter of the depth estimate


def temporal(J: np.ndarray, fps: float, sg_s: float | None = None, length_frames: np.ndarray | None = None
             ) -> tuple[np.ndarray, np.ndarray, dict]:
    """(N,J,3) joints with NaN gaps -> smoothed, gap-filled, fixed-bone-length skeleton + valid mask.
    length_frames: frames whose bone lengths are trusted most (e.g. multi-view triangulated); used for the
    fixed lengths when they are at least 2 s and a quarter of the observed frames, else all valid frames."""
    N = len(J)
    raw_ok = np.isfinite(J[:, 0, 0])
    Jc = J.copy()
    for j in range(J.shape[1]):
        for c in range(3):
            Jc[:, j, c] = _hampel(Jc[:, j, c])
    ok = np.isfinite(Jc).all(axis=(1, 2))
    # valid: observed, or inside a short gap between observations
    valid = ok.copy()
    idx = np.where(ok)[0]
    gmax = int(round(GAP_FILL_S * fps))
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= gmax + 1:
            valid[a + 1:b] = True
    Ji = np.empty_like(Jc)
    for j in range(J.shape[1]):
        for c in range(3):
            Ji[:, j, c] = _interp(Jc[:, j, c])
    win = max(5, int(round((sg_s or SG_S) * fps)) | 1)
    if N > win:
        Ji = savgol_filter(Ji, win, 2, axis=0, mode="interp")
    # fixed bone lengths: median over valid frames, then forward kinematics along measured directions
    I = {n: k for k, n in enumerate(NAMES)}
    out = Ji.copy()
    lengths = {}
    use = valid
    if length_frames is not None and (length_frames & ok).sum() >= max(2 * fps, 0.25 * ok.sum()):
        use = length_frames & ok
    for ch, pa in PARENT.items():
        L = np.linalg.norm(Ji[:, I[ch]] - Ji[:, I[pa]], axis=1)
        lengths[ch] = float(np.median(L[use])) if use.any() else float(np.median(L))
    for ch, pa in PARENT.items():
        d = Ji[:, I[ch]] - Ji[:, I[pa]]
        d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
        out[:, I[ch]] = out[:, I[pa]] + lengths[ch] * d
    return out, valid, {"bone_lengths_m": {k: round(v, 4) for k, v in lengths.items()},
                        "observed_frac": float(raw_ok.mean()), "valid_frac": float(valid.mean())}


# ---------------------------------------------------------------- top-down second pass helpers
def fill_landmarks(X: np.ndarray, max_gap: int) -> np.ndarray:
    """Linear interpolation of (N,33,3) landmark tracks over gaps up to max_gap frames (for ROI prediction)."""
    Y = X.copy()
    ok = np.isfinite(X[:, 0, 0])
    idx = np.where(ok)[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= max_gap + 1:
            w = (np.arange(a + 1, b) - a) / (b - a)
            Y[a + 1:b] = (1 - w)[:, None, None] * X[a] + w[:, None, None] * X[b]
    return Y


def rois(cam: dict, X: np.ndarray, image_size, own_box: np.ndarray | None = None) -> np.ndarray:
    """(N,4) image box of the worker in this camera: the camera's own detection box, else the projection of the
    3D estimate (from all cameras); NaN where neither exists or the person is outside the view."""
    W, H = image_size
    N = len(X)
    out = np.full((N, 4), np.nan)
    for i in range(N):
        b = None
        if own_box is not None and np.isfinite(own_box[i]).all():
            b = own_box[i]
        elif np.isfinite(X[i, 0, 0]):
            Xc = (cam["R"] @ X[i].T + cam["t"].reshape(3, 1)).T
            if (Xc[:, 2] > 0.3).all():
                uv = project(cam, X[i])
                b = np.r_[uv.min(0), uv.max(0)]
        if b is None:
            continue
        w, h = b[2] - b[0], b[3] - b[1]
        # keep boxes that overlap the image by at least a third of their area and are not tiny
        ov = max(0, min(b[2], W) - max(b[0], 0)) * max(0, min(b[3], H) - max(b[1], 0))
        if max(w, h) >= 40 and ov >= 0.33 * w * h:
            out[i] = b
    return out


def merge_pass(img: np.ndarray, wld: np.ndarray, sel: np.ndarray, crop: dict) -> tuple[np.ndarray, np.ndarray]:
    """Single-person tracks (N,1,33,5)/(N,1,33,3): per frame the crop result or the first-pass worker detection,
    whichever has the higher mean landmark visibility (crop preferred on ties)."""
    N = len(sel)
    oi = np.full((N, 1, 33, 5), np.nan, np.float32)
    ow = np.full((N, 1, 33, 3), np.nan, np.float32)
    f1 = sel >= 0
    oi[f1, 0] = img[np.where(f1)[0], sel[f1]]
    ow[f1, 0] = wld[np.where(f1)[0], sel[f1]]
    idx = crop["index"]
    for k, i in enumerate(idx):
        ci = crop["img"][k]
        if not np.isfinite(ci[0, 0]):
            continue
        # a crop answer much smaller than its box is a part / another object, not the person the box was put on
        if "boxes" in crop:
            bx = crop["boxes"][k]
            if (np.nanmax(ci[:, 1]) - np.nanmin(ci[:, 1])) < 0.4 * (bx[3] - bx[1]):
                continue
        if not f1[i] or np.nanmean(ci[:, 3]) >= np.nanmean(oi[i, 0, :, 3]) - 0.02:
            oi[i, 0] = ci
            ow[i, 0] = crop["world"][k]
    return oi, ow
