"""Side-by-side stills (real CarDA frame | Isaac Sim from the same calibrated camera) and the captioned clip, with the
CarDA credit on every image.

  python finish.py OUT_DIR IN_MP4 OUT_MP4

Reads OUT_DIR/still_<in|out>_<t>.png, OUT_DIR/clip/frame_*.png and clip_meta.json (ws30_station.py); writes
OUT_DIR/side_by_side_<view>_<t>.jpg and OUT_DIR/ws30_human_alone.mp4.
"""
import glob
import json
import os
import re
import subprocess
import sys

import cv2
import numpy as np

out, mp4s = sys.argv[1], {"in": sys.argv[2], "out": sys.argv[3]}
CREDIT = "CarDA dataset, CC BY 4.0"
# the job as read from the video and mocap (annotation sheet order, times ours)
STEPS = [(7.5, "1.1-1.2 takes a speaker from the cart and seats it in the door"),
         (12.0, "1.3 takes the screwdriver from the line-side rack"),
         (53.0, "1.4 drives 4 screws into the speaker"), (99, "1.5 returns the screwdriver")]


def text(img, s, org, scale=0.8):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)


for p in sorted(glob.glob(os.path.join(out, "still_*.png"))):
    view, t = re.match(r"still_(\w+)_([\d.]+)\.png", os.path.basename(p)).groups()
    if view not in mp4s:
        continue
    cap = cv2.VideoCapture(mp4s[view])
    cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000)
    _, real = cap.read()
    sim = cv2.resize(cv2.imread(p), (real.shape[1], real.shape[0]))
    text(real, f"Real: CarDA WS30 cycle 003, {view.upper()} camera, t = {float(t):.0f} s", (20, 40))
    text(sim, "Isaac Sim 6.2 rebuild, same calibrated camera", (20, 40))
    both = np.hstack([real, sim])
    text(both, CREDIT, (20, both.shape[0] - 20), 0.7)
    q = os.path.join(out, f"side_by_side_{view}_{float(t):04.1f}.jpg")
    cv2.imwrite(q, both, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("wrote", q)

frames = sorted(glob.glob(os.path.join(out, "clip", "frame_*.png")))
if frames:
    meta = json.load(open(os.path.join(out, "clip_meta.json")))
    h, w = cv2.imread(frames[0]).shape[:2]
    q = os.path.join(out, "ws30_human_alone.mp4")
    ff = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                           "-r", str(meta["fps"]), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", q],
                          stdin=subprocess.PIPE)
    for f, m in zip(frames, meta["frames"]):
        img = cv2.imread(f)
        text(img, "Human alone: CarDA WS30, real Xsens worker, door riding the belt at 23.9 mm/s", (20, 40), 0.75)
        text(img, f"t = {m['t']:4.1f} s   " + next(s for t1, s in STEPS if m["t"] < t1), (20, 74), 0.7)
        text(img, CREDIT, (20, h - 20), 0.7)
        ff.stdin.write(img.tobytes())
    ff.stdin.close()
    ff.wait()
    print("wrote", q, len(frames), "frames")
