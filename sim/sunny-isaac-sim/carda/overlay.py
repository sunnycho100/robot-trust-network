"""Draw station labels and the WS10 SSM readout on the frames carda_line.py rendered, from its meta.json.
Usage: python3 overlay.py FRAMES_DIR OUT_DIR   (then ffmpeg, see colab_job.sh)
"""
import json
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

src, dst = sys.argv[1:3]
os.makedirs(dst, exist_ok=True)
meta = json.load(open(os.path.join(src, "meta.json")))
W, H = meta["W"], meta["H"]
TAN = (meta["hap"] / 2) / meta["focal"]


def font(size, bold=False):
    import matplotlib
    d = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data/fonts/ttf")
    return ImageFont.truetype(os.path.join(d, "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"), size)


F_T, F_S, F_L, F_LB, F_F = font(26, True), font(17), font(15), font(19, True), font(13)
TAG = {"WS10": "glass fit + 4 screws, shared door", "WS20": "tool changer, 9 screws", "WS30": "4 screws after the speaker"}


def project(eye, target, p):
    eye, target, p = map(np.asarray, (eye, target, p))
    f = target - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, [0, 0, 1.0])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    d = p - eye
    z = float(np.dot(d, f))
    if z < 0.3:
        return None
    x = float(np.dot(d, r)) / z / TAN
    y = float(np.dot(d, u)) / z / (TAN * H / W)
    if abs(x) > 1.05 or abs(y) > 1.05:
        return None
    return (x + 1) / 2 * W, (1 - y) / 2 * H


def tag(dr, xy, title, sub, color=(255, 255, 255)):
    x, y = xy
    w = max(dr.textlength(title, font=F_LB), dr.textlength(sub, font=F_L)) + 16
    x = min(max(x - w / 2, 4), W - w - 4)
    dr.rounded_rectangle([x, y - 50, x + w, y - 4], 6, fill=(15, 18, 24, 190))
    dr.text((x + 8, y - 48), title, font=F_LB, fill=color)
    dr.text((x + 8, y - 25), sub, font=F_L, fill=(205, 210, 220))


def draw(fr):
    im = Image.open(os.path.join(src, f"frame_{fr['k']:05d}.png")).convert("RGBA")
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(ov)
    t, eye, tgt = fr["t"], fr["eye"], fr["target"]
    close = t >= 13.0
    for s, (sx, sy) in meta["stations"].items():
        if close:
            break
        xy = project(eye, tgt, [sx + fr["door"], sy, 2.25])
        if xy:
            tag(dr, xy, f"{s}   {fr['labels'][s]}", TAG[s])
    if close:   # tags on the cobot and the worker instead
        xy = project(eye, tgt, [fr["cx10"], meta["y_track"], 1.75])
        if xy:
            tag(dr, xy, f"Cobot   {fr['labels']['WS10']}", "UR10e on a linear track, screwdriver")
        xy = project(eye, tgt, [fr["worker"][0] + 0.6, fr["worker"][1], 1.35])
        if xy:
            tag(dr, xy, "Worker", "fitting the door glass", (255, 190, 120))
    if not close:
        xy = project(eye, tgt, [fr["amr_x"], meta["amr_y"], 1.0])
        if xy:
            tag(dr, xy, "AMR", "kit delivery to the line side", (130, 190, 255))
    # title
    title = "Car-door line, takt 240 s" if not close else "WS10: worker and cobot on the same door"
    sub = ("Three stations, one cobot per station on a linear track, AMR kitting" if not close else
           "Worker fits the glass while the cobot drives screws; ISO/TS 15066 speed and separation monitoring")
    w = max(dr.textlength(title, font=F_T), dr.textlength(sub, font=F_S)) + 28
    dr.rounded_rectangle([14, 14, 14 + w, 86], 8, fill=(15, 18, 24, 200))
    dr.text((28, 22), title, font=F_T, fill=(255, 255, 255))
    dr.text((28, 56), sub, font=F_S, fill=(205, 210, 220))
    # SSM readout for WS10
    if True:  # SSM readout in both views
        red = fr["red"] and fr["active"]
        col = (240, 70, 60) if red else (80, 210, 110)
        state = ("inside protective distance: robot slows down" if red else "clear") if fr["active"] else "robot not moving"
        line1 = f"WS10 separation {max(fr['sep'], 0):.2f} m   protective distance {fr['sp']:.2f} m"
        w = max(dr.textlength(line1, font=F_S), dr.textlength(state, font=F_LB)) + 28
        y0 = H - 118
        dr.rounded_rectangle([14, y0, 14 + w, y0 + 62], 8, fill=(15, 18, 24, 205))
        dr.rectangle([14, y0, 20, y0 + 62], fill=col)
        dr.text((30, y0 + 6), line1, font=F_S, fill=(230, 232, 236))
        dr.text((30, y0 + 32), state, font=F_LB, fill=col)
    foot = ("UR10e as a stand-in for the Doosan M0617. Kinematic replay in Isaac Sim 6.2. "
            "Process numbers from the CarDA study (Stellantis Melfi lab line, CC BY 4.0).")
    dr.rectangle([0, H - 30, W, H], fill=(10, 12, 16, 170))
    dr.text((14, H - 24), foot, font=F_F, fill=(190, 195, 205))
    Image.alpha_composite(im, ov).convert("RGB").save(os.path.join(dst, f"frame_{fr['k']:05d}.png"))


if __name__ == "__main__":
    from multiprocessing import Pool

    with Pool() as pool:
        pool.map(draw, meta["frames"], chunksize=8)
    print("overlay done", len(meta["frames"]))
