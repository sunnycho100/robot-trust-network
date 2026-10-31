"""Input video frames for the report (grabbed with OpenCV from data/carda/<ws>/mp4, local only)."""
from __future__ import annotations

import base64
from pathlib import Path

from rcd.common.io import ROOT, data_path


def _jpeg_uri(img, width=560, q=74):
    import cv2
    h, w = img.shape[:2]
    if w > width:
        img = cv2.resize(img, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode() if ok else None


def station_frames(cfg, frac=0.4):
    """[{station, side, file, t_s, uri}] one in-camera and one out-camera frame per station (first cycle)."""
    import cv2
    out = []
    for st in cfg["data"]["stations"]:
        d = data_path(cfg, st.lower(), "mp4")
        if not d.exists():
            continue
        files = sorted(d.glob("*.mp4"))
        for side in ("in", "out"):
            f = next((x for x in files if f"_{side}_" in x.name), None)
            if f is None:
                continue
            cap = cv2.VideoCapture(str(f))
            n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            k = int(n * frac)
            cap.set(cv2.CAP_PROP_POS_FRAMES, k)
            ok, img = cap.read()
            cap.release()
            if ok:
                out.append({"station": st, "side": side, "file": f.name, "t_s": k / fps, "uri": _jpeg_uri(img)})
    if not out:   # fallback: pre-extracted samples next to the project
        for p in sorted((ROOT.parent / "carda_samples").glob("*full_frame.jpg")):
            import cv2 as _cv
            img = _cv.imread(str(p))
            if img is not None:
                out.append({"station": p.name.split("_")[0].upper(), "side": "in" if "_in_" in p.name else "out",
                            "file": p.name, "t_s": None, "uri": _jpeg_uri(img)})
    return out


def contact_sheet(cfg, station="WS30", n=6):
    """A strip of n frames across one clip (in-camera), for the 'from video' story."""
    import cv2
    import numpy as np
    d = data_path(cfg, station.lower(), "mp4")
    files = sorted(d.glob("*_in_*.mp4")) if d.exists() else []
    if not files:
        return None
    cap = cv2.VideoCapture(str(files[0]))
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    tiles = []
    for k in np.linspace(0.05 * N, 0.95 * N, n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(k))
        ok, img = cap.read()
        if not ok:
            continue
        img = cv2.resize(img, (320, int(img.shape[0] * 320 / img.shape[1])))
        cv2.rectangle(img, (0, 0), (92, 26), (20, 30, 45), -1)
        cv2.putText(img, f"{k / fps:5.1f}s", (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(img)
    cap.release()
    if not tiles:
        return None
    rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles) - len(tiles) % 3 or len(tiles), 3)]
    rows = [r for r in rows if r.shape[1] == rows[0].shape[1]]
    return {"file": files[0].name, "uri": _jpeg_uri(np.vstack(rows), width=960, q=72)}
