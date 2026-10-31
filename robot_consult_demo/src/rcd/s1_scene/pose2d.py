"""Per-frame 2D/3D body keypoints from an ordinary video with an off-the-shelf model (MediaPipe Pose Landmarker).

Every `stride`-th frame of a clip is passed to the Pose Landmarker (VIDEO running mode, i.e. with its own
landmark tracking/filtering) with up to `MAX_POSES` people per frame, so bystanders are kept as separate
candidates and the worker is chosen later (rcd.s1_scene.video_pose). For every candidate we keep
  * image landmarks: 33 x (u_px, v_px, rel_depth, visibility, presence)
  * "world" landmarks: 33 x (x, y, z) metres, origin between the hips, axes aligned with the camera
    (x right, y down, z away from the camera) -- the model's metric body-relative 3D.

Results are cached per clip (keyed by model file, stride and clip size) because inference dominates the
stage runtime; several clips are processed in parallel worker processes.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

MAX_POSES = 3
N_LM = 33
MODEL_URLS = {
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
}

# MediaPipe BlazePose 33-landmark topology
LM = {name: i for i, name in enumerate([
    "nose", "l_eye_in", "l_eye", "l_eye_out", "r_eye_in", "r_eye", "r_eye_out", "l_ear", "r_ear", "mouth_l",
    "mouth_r", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist", "l_pinky", "r_pinky",
    "l_index", "r_index", "l_thumb", "r_thumb", "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle",
    "l_heel", "r_heel", "l_foot", "r_foot"])}
# landmark index permutation that swaps the body's left and right side
LR_SWAP = np.arange(N_LM)
for _a, _b in [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10)] + [(k, k + 1) for k in range(11, 33, 2)]:
    LR_SWAP[_a], LR_SWAP[_b] = _b, _a


def model_path(root: Path, model: str = "heavy") -> Path:
    """Local copy of the public model weights (downloaded once into <root>/models/)."""
    p = Path(root) / "models" / f"pose_landmarker_{model}.task"
    if not p.exists():
        import urllib.request
        p.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URLS[model], p)
    return p


def _cache_name(video: Path, model: Path, stride: int) -> str:
    st = Path(video).stat()
    key = f"{Path(video).name}|{st.st_size}|{Path(model).name}|{Path(model).stat().st_size}|{stride}|{MAX_POSES}"
    return f"{Path(video).stem}__{hashlib.md5(key.encode()).hexdigest()[:10]}.npz"


def extract(video: str | Path, model: str | Path, stride: int = 2) -> dict:
    """Run the landmarker on every `stride`-th frame. Missing people/landmarks are NaN."""
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision

    opts = vision.PoseLandmarkerOptions(
        base_options=mpt.BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.VIDEO, num_poses=MAX_POSES,
        min_pose_detection_confidence=0.3, min_pose_presence_confidence=0.3, min_tracking_confidence=0.3)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = (total + stride - 1) // stride
    img = np.full((n, MAX_POSES, N_LM, 5), np.nan, np.float32)
    wld = np.full((n, MAX_POSES, N_LM, 3), np.nan, np.float32)
    frames = np.arange(n) * stride
    k, i = 0, 0
    with vision.PoseLandmarker.create_from_options(opts) as lm:
        while i < n:
            if not cap.grab():
                break
            if k % stride == 0:
                ok, im = cap.retrieve()
                if ok:
                    rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                    r = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(round(k * 1000.0 / fps)))
                    for p, (pl, pw) in enumerate(zip(r.pose_landmarks, r.pose_world_landmarks)):
                        img[i, p] = [[q.x * W, q.y * H, q.z * W, q.visibility, q.presence] for q in pl]
                        wld[i, p] = [[q.x, q.y, q.z] for q in pw]
                i += 1
            k += 1
    cap.release()
    return {"img": img[:i], "world": wld[:i], "frames": frames[:i], "fps": np.float32(fps),
            "stride": np.int32(stride), "n_frames_video": np.int32(total), "image_size": np.array([W, H], np.int32)}


def _job(args):
    video, model, stride, cache = args
    out = Path(cache)
    if not out.exists():
        res = extract(video, model, stride)
        tmp = out.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **res)
        os.replace(tmp, out)
    return str(out)


def extract_many(videos: list[Path], cache_dir: Path, model: Path, stride: int = 2, workers: int = 4) -> dict[str, dict]:
    """{str(video): result dict}; clips are processed in parallel (longest first) and cached on disk."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(str(v), str(model), stride, str(cache_dir / _cache_name(Path(v), Path(model), stride))) for v in videos]
    todo = sorted([j for j in jobs if not Path(j[3]).exists()], key=lambda j: -Path(j[0]).stat().st_size)
    if todo:
        if workers > 1 and len(todo) > 1:
            import multiprocessing as mp_
            with mp_.get_context("spawn").Pool(min(workers, len(todo))) as pool:
                for _ in pool.imap_unordered(_job, todo):
                    pass
        else:
            for j in todo:
                _job(j)
    out = {}
    for v, _, _, c in jobs:
        d = np.load(c)
        out[v] = {k: d[k] for k in d.files}
    return out


# ---------------------------------------------------------------- second pass: top-down crops around the worker
CROP_SIZE = 384


def _crop(im: np.ndarray, box) -> tuple[np.ndarray, float, float, float]:
    """Square crop (zero-padded outside the image) around box (u0,v0,u1,v1), resized to CROP_SIZE."""
    import cv2
    u0, v0, u1, v1 = box
    cu, cv_ = 0.5 * (u0 + u1), 0.5 * (v0 + v1)
    side = max(u1 - u0, v1 - v0) * 1.35 + 40
    x0, y0 = int(round(cu - side / 2)), int(round(cv_ - side / 2))
    s = int(round(side))
    H, W = im.shape[:2]
    out = np.zeros((s, s, 3), im.dtype)
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + s), min(H, y0 + s)
    if xb > xa and yb > ya:
        out[ya - y0:yb - y0, xa - x0:xb - x0] = im[ya:yb, xa:xb]
    return cv2.resize(out, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA if s > CROP_SIZE else cv2.INTER_LINEAR), x0, y0, s / CROP_SIZE


def extract_crops(video: str | Path, model: str | Path, frames: np.ndarray, boxes: np.ndarray) -> dict:
    """Landmarks of the single person inside each given ROI (IMAGE mode, one pose per crop).

    frames (M,) video frame indices (increasing), boxes (M,4) image boxes; returns img (M,33,5), world (M,33,3)."""
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision

    opts = vision.PoseLandmarkerOptions(base_options=mpt.BaseOptions(model_asset_path=str(model)),
                                        running_mode=vision.RunningMode.IMAGE, num_poses=1,
                                        min_pose_detection_confidence=0.3, min_pose_presence_confidence=0.3)
    M = len(frames)
    img = np.full((M, N_LM, 5), np.nan, np.float32)
    wld = np.full((M, N_LM, 3), np.nan, np.float32)
    cap = cv2.VideoCapture(str(video))
    k, im, ok = -1, None, False
    with vision.PoseLandmarker.create_from_options(opts) as lm:
        for i in range(M):
            f = int(frames[i])
            fresh = False
            while k < f:
                if not cap.grab():
                    break
                k += 1
                fresh = True
            if k != f:
                break
            if fresh:
                ok, im = cap.retrieve()
            if not ok:
                continue
            crop, x0, y0, sc = _crop(im, boxes[i])
            r = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
            if r.pose_landmarks:
                pl, pw = r.pose_landmarks[0], r.pose_world_landmarks[0]
                img[i] = [[x0 + q.x * CROP_SIZE * sc, y0 + q.y * CROP_SIZE * sc, q.z * CROP_SIZE * sc, q.visibility, q.presence] for q in pl]
                wld[i] = [[q.x, q.y, q.z] for q in pw]
    cap.release()
    return {"img": img, "world": wld, "frames": np.asarray(frames), "boxes": np.asarray(boxes, np.float32)}


def _crop_job(args):
    video, model, frames, boxes, cache = args
    out = Path(cache)
    if not out.exists():
        res = extract_crops(video, model, frames, boxes)
        tmp = out.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **res)
        os.replace(tmp, out)
    return str(out)


def extract_crops_many(jobs: list[tuple], cache_dir: Path, model: Path, workers: int = 4) -> list[dict]:
    """jobs: [(video, frames, boxes)] -> results (cached by content hash of the ROIs)."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    full = []
    for v, fr, bx in jobs:
        h = hashlib.md5(np.ascontiguousarray(fr).tobytes() + np.round(np.ascontiguousarray(bx, np.float32), 0).tobytes()
                        + Path(model).name.encode()).hexdigest()[:12]
        full.append((str(v), str(model), np.asarray(fr), np.asarray(bx), str(cache_dir / f"{Path(v).stem}__crop_{h}.npz")))
    todo = sorted([j for j in full if not Path(j[4]).exists()], key=lambda j: -len(j[2]))
    if todo:
        if workers > 1 and len(todo) > 1:
            import multiprocessing as mp_
            with mp_.get_context("spawn").Pool(min(workers, len(todo))) as pool:
                for _ in pool.imap_unordered(_crop_job, todo):
                    pass
        else:
            for j in todo:
                _crop_job(j)
    res = []
    for j in full:
        d = np.load(j[4])
        res.append({k: d[k] for k in d.files})
    return res
