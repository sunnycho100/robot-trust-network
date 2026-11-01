"""Draw the event timeline (with each event's signers), the agent tags and the honest footer on door_team.py frames.
Usage: python3 overlay.py FRAMES_DIR PLAN_DIR OUT_DIR [--title TEXT]   (then ffmpeg, see colab_job.sh)
"""
import argparse
import json
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("plan")
ap.add_argument("dst")
ap.add_argument("--title", default="")
a = ap.parse_args()
os.makedirs(a.dst, exist_ok=True)
meta = json.load(open(os.path.join(a.src, "meta.json")))
plan = json.load(open(os.path.join(a.plan, "plan.json")))
FW, FH = meta["W"], meta["H"]       # frame size; text is laid out at 1280 x 720 and scaled onto the frame
W, H = 1280, 720
TAN = (meta["hap"] / 2) / meta["focal"]
HEAD = [e for e in plan["events"] if e["headline"]]
STEPS = (["GLASS PLACED", "SCREW INSTALLED", "STATION COMPLETE"] if plan["solo"] else
         ["TEAM FORMED (ARM L, ARM R)", "DOOR PICKED", "DOOR DELIVERED TO WS2", "GLASS PLACED", "SCREW INSTALLED",
          "STATION 2 COMPLETE"])
NAMES = {"ARM_L": ("ARM L", "UR20 on rail"), "ARM_R": ("ARM R", "UR20 on rail"),
         "HUMANOID_A": ("G1 A", "kinematic replay"), "HUMANOID_B": ("G1 B", "kinematic replay"),
         "WORKER": ("WORKER", "G1 standing in for a person")}
TITLE = a.title or ("Human alone: one worker on the moving door" if plan["solo"] else "CarDA WS30 automated: 2 rail arms + 2 G1")


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        import matplotlib
        return ImageFont.truetype(os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data/fonts/ttf", name), size)
    except (ImportError, OSError):
        for d in ("/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/dejavu", ""):
            try:
                return ImageFont.truetype(os.path.join(d, name) if d else name, size)
            except OSError:
                pass
        return ImageFont.load_default(size)


F_T, F_S, F_L, F_LB, F_F, F_E, F_EB, F_X = font(24, True), font(16), font(14), font(17, True), font(12), font(14), font(14, True), font(11)
F_BIG = font(40, True)


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
    if abs(x) > 1.0 or abs(y) > 1.0:
        return None
    return (x + 1) / 2 * W, (1 - y) / 2 * H


def label_at(name, t):
    lab = "idle"
    for tt, txt in plan["labels"][name]:
        if t >= tt - 1e-6:
            lab = txt
    return lab


def tag(dr, xy, title, sub, color):
    x, y = xy
    w = max(dr.textlength(title, font=F_LB), dr.textlength(sub, font=F_L)) + 16
    x = min(max(x - w / 2, 4), W - 346 - w)
    dr.rounded_rectangle([x, y - 48, x + w, y - 4], 6, fill=(15, 18, 24, 185))
    dr.text((x + 8, y - 46), title, font=F_LB, fill=color)
    dr.text((x + 8, y - 24), sub, font=F_L, fill=(215, 220, 228))


def draw(fr):
    src = os.path.join(a.src, f"frame_{fr['k']:05d}.png")
    jpg = not os.path.exists(src)
    im = Image.open(src[:-4] + ".jpg" if jpg else src).convert("RGBA")
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    dr = ImageDraw.Draw(ov)
    t = fr["t"]
    for name, p in fr["anchors"].items():
        if name not in NAMES:                          # background robots stay unlabelled
            continue
        xy = project(fr["eye"], fr["target"], p)
        if xy:
            nm, kind = NAMES.get(name, (name, ""))
            col = (255, 205, 90) if name.startswith("ARM") else (140, 200, 255)
            tag(dr, xy, f"{nm}  ({kind})", label_at(name, t), col)
    # title
    sub = ("CarDA WS30 style floor belt at 24 mm/s; a G1 stands in for the worker" if plan["solo"] else
           "Hardcoded agents acting on signed events; multi-agent RL policies are the next step")
    w = max(dr.textlength(TITLE, font=F_T), dr.textlength(sub, font=F_S)) + 28
    dr.rounded_rectangle([14, 14, 14 + w, 82], 8, fill=(15, 18, 24, 200))
    dr.text((28, 21), TITLE, font=F_T, fill=(255, 255, 255))
    dr.text((28, 54), sub, font=F_S, fill=(205, 210, 220))
    # timeline
    x0, y0, pw = W - 338, 14, 324
    rows = []
    for s in STEPS:
        e = next((e for e in HEAD if e["headline"] == s and e["t"] <= t + 1e-6), None)
        rows.append((s, e))
    ph = 44 + 44 * len(rows)
    dr.rounded_rectangle([x0, y0, x0 + pw, y0 + ph], 8, fill=(15, 18, 24, 210))
    dr.text((x0 + 14, y0 + 10), "EVENT TIMELINE", font=F_EB, fill=(255, 255, 255))
    dr.text((x0 + 200, y0 + 12), f"t = {t:5.1f} s", font=F_L, fill=(190, 196, 206))
    for i, (s, e) in enumerate(rows):
        y = y0 + 38 + 44 * i
        done = e is not None
        dr.rectangle([x0 + 14, y + 4, x0 + 26, y + 16], fill=(80, 210, 110) if done else (70, 74, 82))
        dr.text((x0 + 36, y), s, font=F_EB if done else F_E, fill=(255, 255, 255) if done else (130, 135, 145))
        if done:
            dr.text((x0 + 36, y + 20), f"{e['t']:.1f} s", font=F_X, fill=(190, 196, 206))
            bx = x0 + 84
            dr.rounded_rectangle([bx, y + 19, bx + 52, y + 34], 4, fill=(30, 120, 60))
            dr.text((bx + 6, y + 20), "SIGNED", font=F_X, fill=(255, 255, 255))
            who = ", ".join(n.replace("HUMANOID_", "G1 ").replace("_", " ") for n in e["signed_by"])
            dr.text((bx + 60, y + 20), f"{who}  {e['sig'][0][:8]}", font=F_X, fill=(170, 220, 180))
    if rows[-1][1] is not None:          # the station banner
        msg = STEPS[-1]
        bw = dr.textlength(msg, font=F_BIG) + 48
        bx, by = (W - 340 - bw) / 2, H - 120
        dr.rounded_rectangle([bx, by, bx + bw, by + 64], 10, fill=(25, 120, 55, 230))
        dr.text((bx + 24, by + 8), msg, font=F_BIG, fill=(255, 255, 255))
    # footer
    foot = ("Kinematic replay in Isaac Sim 6.2. G1: legs and root from Unitree's walking policy in MuJoCo, arms by IK. "
            "Door, glass and tools carried kinematically." if plan["solo"] else
            "WS30 station rebuilt from the CarDA video. Kinematic replay (Isaac Sim 6.2): UR20 arms by IK on an assisted grasp; "
            "G1 legs from Unitree's walking policy in MuJoCo.")
    foot2 = "Events signed per robot with Ed25519 (robot trust network protocol); log in signed_events.json."
    dr.rectangle([0, H - 40, W, H], fill=(10, 12, 16, 175))
    dr.text((14, H - 37), foot, font=F_F, fill=(195, 200, 210))
    dr.text((14, H - 20), foot2, font=F_F, fill=(195, 200, 210))
    if ov.size != im.size:
        ov = ov.resize(im.size, Image.LANCZOS)
    out = Image.alpha_composite(im, ov).convert("RGB")
    if jpg:
        out.save(os.path.join(a.dst, f"frame_{fr['k']:05d}.jpg"), quality=95)
    else:
        out.save(os.path.join(a.dst, f"frame_{fr['k']:05d}.png"))


if __name__ == "__main__":
    from multiprocessing import Pool

    with Pool() as pool:
        pool.map(draw, meta["frames"], chunksize=8)
    print("overlay done", len(meta["frames"]))
