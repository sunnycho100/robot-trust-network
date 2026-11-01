"""Process film v4 on the real CarDA WS30 car-door station: record it, label it, rebuild it, automate it (1280x720, 30 fps).

Usage (miniforge python3: numpy and Pillow; ffmpeg on PATH; fonts in fonts/, see README.md):
  python3 sim/sunny-isaac-sim/film/ws30_process_film.py [--station PATH] [--robots PATH]     # docs/evidence/process/process_film_v4.mp4
  python3 sim/sunny-isaac-sim/film/ws30_process_film.py --still 2 10 14 22                    # PNG stills next to the output

01 "Record the place": the CarDA WS30 in and out camera clips (docs/evidence/carda/ws30_real, CarDA, CC BY 4.0).
   These are CarDA's fixed cameras, standing in for a phone recording.
02 "We rebuild it in 3D, as a scene a simulator can run": the out frame freezes and labels pop onto the key objects. The boxes
   were placed by Claude reading that frame (a VLM-style pass by hand, not a detector run). A wipe then reveals the rebuilt
   station: --station, an Isaac Sim render (image or mp4) from docs/evidence/ws30_station/, or a placeholder card until it lands.
03 "Your robots do the job here": --robots, the automated door scene (docs/evidence/door_team/ v2) or the station's
   human-alone clip until then, or a placeholder card.
"""
import argparse, subprocess
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REAL = ROOT / "docs/evidence/carda/ws30_real"
IN_CAM, OUT_CAM = (REAL / f"xsens_003_WS30_2023_10_04_{s}_camera_cropped.mp4" for s in ("in", "out"))
OUT = ROOT / "docs/evidence/process/process_film_v4.mp4"
W, H, FPS = 1280, 720, 30
BG, INK, ACC, LINE = (10, 12, 12), (228, 234, 231), (111, 224, 207), (34, 42, 41)  # site --bg --ink --accent --line
MONO = str(HERE / "fonts/IBMPlexMono-Medium.ttf")

# Timeline, seconds of film. Source times are seconds into the CarDA clips.
IN_SRC = 39.0                  # in camera: the worker at the door on its trolley
XF1 = (4.4, 5.0)               # in -> out crossfade
FREEZE, OUT_SRC = 8.4, 12.0    # the out camera plays into its 12 s frame and holds there
LABEL0, STAGGER, POP = 8.8, 0.35, 0.3
WIPE = (12.2, 13.2)            # labelled frame -> rebuilt station
ROBOTS0, XF = 19.2, 0.6        # robots section starts (after a crossfade from the station)
ROBOTS_MAX = 10.0
CAPTIONS = [(0.0, "Record the place"), (FREEZE, "We rebuild it in 3D, as a scene a simulator can run"),
            (ROBOTS0, "Your robots do the job here")]
# Key objects in the out camera's 12 s frame (pixels): label, box, label chip anchor. Read off the frame by eye.
LABELS = [("floor belt conveyor", (0, 368, 1279, 604), (560, 342)),
          ("door trolley", (0, 236, 390, 452), (6, 458)),
          ("door", (92, 92, 292, 248), (92, 66)),
          ("KLT bin", (273, 265, 383, 322), (388, 280)),
          ("KLT bin", (0, 250, 78, 285), None),
          ("workbench", (308, 148, 492, 320), (308, 122))]


def smooth(t, a, b):
    x = min(max((t - a) / (b - a), 0.0), 1.0)
    return x * x * (3 - 2 * x)


class Clip:
    """Frames of a video from `start` s, decoded on demand in order (ffmpeg, scaled to the film size)."""

    def __init__(self, path, start=0.0):
        self.path, self.start, self.p, self.i, self.last = path, start, None, -1, None
        self.duration = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                                             capture_output=True, text=True, check=True).stdout) - start

    def frame(self, k):  # k = frame index from start; clamps to the last frame
        if self.p is None or k < self.i:
            self.p = subprocess.Popen(["ffmpeg", "-loglevel", "fatal", "-ss", str(self.start + max(k, 0) / FPS), "-i", str(self.path),
                                       "-vf", f"fps={FPS},scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:-1:-1:color=0a0c0c",
                                       "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
            self.i = max(k, 0) - 1
        while self.i < k:
            raw = self.p.stdout.read(W * H * 3)
            if len(raw) < W * H * 3:
                break
            self.last, self.i = np.frombuffer(raw, np.uint8).reshape(H, W, 3), self.i + 1
        return self.last.astype(np.float32)


class Still:
    """An image held on screen with a slow push-in."""

    def __init__(self, path, seconds):
        self.img, self.duration = Image.open(path).convert("RGB"), seconds

    def frame(self, k):
        z = 1 + 0.06 * min(k / (self.duration * FPS), 1)
        w, h = self.img.size
        s = max(W / w, H / h) * z
        im = self.img.resize((round(w * s), round(h * s)), Image.LANCZOS)
        x, y = (im.width - W) // 2, (im.height - H) // 2
        return np.asarray(im.crop((x, y, x + W, y + H)), np.float32)


def placeholder(title, note):
    """A plain card saying which render lands here, in the site palette."""
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    for x in range(0, W, 64):
        d.line((x, 0, x, H), fill=LINE)
    for y in range(0, H, 64):
        d.line((0, y, W, y), fill=LINE)
    d.rectangle((W // 2 - 380, H // 2 - 70, W // 2 + 380, H // 2 + 70), fill=BG, outline=ACC, width=2)
    d.text((W // 2, H // 2 - 20), title, font=ImageFont.truetype(MONO, 28), fill=INK, anchor="mm")
    d.text((W // 2, H // 2 + 26), note, font=ImageFont.truetype(MONO, 17), fill=tuple(int(0.6 * c + 0.4 * b) for c, b in zip(INK, BG)), anchor="mm")
    return np.asarray(im, np.float32)


class Card:
    def __init__(self, img, seconds):
        self.img, self.duration = img, seconds

    def frame(self, k):
        return self.img


def source(path, card, seconds):
    if path is None:
        return Card(placeholder(*card), seconds)
    return Still(path, seconds) if Path(path).suffix.lower() in (".png", ".jpg", ".jpeg") else Clip(path)


class Film:
    def __init__(self, station, robots):
        self.inc, self.outc = Clip(IN_CAM, IN_SRC), Clip(OUT_CAM, OUT_SRC - (FREEZE - XF1[0]))
        self.station = source(station, ("Rebuilt WS30 station in Isaac Sim", "placeholder: render from docs/evidence/ws30_station/"), ROBOTS0 - WIPE[0])
        self.robots = source(robots, ("Robots doing the door job", "placeholder: door_team v2, or the human-alone clip"), ROBOTS_MAX)
        self.end = ROBOTS0 + min(self.robots.duration, ROBOTS_MAX)
        self.mono = {s: ImageFont.truetype(MONO, s) for s in (16, 15, 24)}

    def labels(self, img, t):
        """VLM-style boxes and chips popping onto the frozen frame, one after another."""
        ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        shown = [(j, a, *lab) for j, lab in enumerate(LABELS) if (a := smooth(t, LABEL0 + STAGGER * j, LABEL0 + STAGGER * j + POP)) > 0]
        # fills, then outlines, then chips: ImageDraw overwrites RGBA pixels instead of blending, so later fills would hide them
        for step in ("fill", "outline"):
            for j, a, name, (x0, y0, x1, y1), chip in shown:
                g = 6 * (1 - a)  # the box settles in from slightly larger
                d.rectangle((x0 - g, y0 - g, x1 + g, y1 + g), **({"fill": (*ACC, int(28 * a))} if step == "fill" else
                                                                 {"outline": (*ACC, int(255 * a)), "width": 2}))
        for j, a, name, box, chip in shown:
            if chip:
                f = self.mono[16]
                w = d.textlength(name, font=f)
                d.rectangle((chip[0], chip[1], chip[0] + w + 16, chip[1] + 24), fill=(*ACC, int(255 * a)))
                d.text((chip[0] + 8, chip[1] + 3), name, font=f, fill=(*BG, int(255 * a)))
        return np.asarray(Image.alpha_composite(Image.fromarray(img.astype(np.uint8)).convert("RGBA"), ov).convert("RGB"), np.float32)

    def caption(self, img, t):
        k = max(i for i, (t0, _) in enumerate(CAPTIONS) if t >= t0)
        a = smooth(t, CAPTIONS[k][0], CAPTIONS[k][0] + 0.5) * smooth(t, 0.2, 0.7) * (1 - smooth(t, self.end - 0.6, self.end))
        shade = np.linspace(0, 1, 190)[:, None, None] ** 1.5 * 0.8  # bottom fade so the type reads over bright footage
        img[H - 190:] += shade * (np.array(BG, np.float32) - img[H - 190:])
        im = Image.fromarray(img.astype(np.uint8))
        d = ImageDraw.Draw(im, "RGBA")
        x, y = 48, H - 74
        for j in range(len(CAPTIONS)):  # stage rail, like the site's
            c = ACC if j == k else ((*INK, 115) if j < k else (*LINE, 255))
            d.rectangle((x + 40 * j, y - 18, x + 40 * j + (30 if j == k else 16), y - 17), fill=c)
        num = f"{k + 1:02d}"
        d.text((x, y), num, font=self.mono[24], fill=(*ACC, int(255 * a)))
        d.text((x + d.textlength(num, font=self.mono[24]) + 16, y), CAPTIONS[k][1], font=self.mono[24], fill=(*INK, int(255 * a)))
        if t < WIPE[1]:  # CC BY attribution while the CarDA footage is on screen
            d.text((W - 48, H - 50), "Footage: CarDA dataset, WS30 (CC BY 4.0)", font=self.mono[15],
                   fill=(*INK, int(150 * (1 - smooth(t, *WIPE)))), anchor="ra")
        return np.asarray(im, np.float32)

    def frame(self, t):
        f = round(t * FPS)
        if t < XF1[1]:
            img = self.inc.frame(f)
        if t >= XF1[0]:
            out = self.outc.frame(round((min(t, FREEZE) - XF1[0]) * FPS))
            if t >= FREEZE:
                out = self.labels(out * (1 - 0.25 * smooth(t, FREEZE, FREEZE + 0.4)) + np.array(BG) * 0.25 * smooth(t, FREEZE, FREEZE + 0.4), t)
            img = out if t >= XF1[1] else img + smooth(t, *XF1) * (out - img)
        if t >= WIPE[0]:  # an accent line sweeps across and leaves the rebuilt station behind it
            st = self.station.frame(round((t - WIPE[0]) * FPS))
            xl = W * smooth(t, *WIPE) * 1.05
            if t < WIPE[1]:
                cols = np.arange(W)[None, :, None]
                img = np.where(cols < xl, st, img)
                glow = np.exp(-((cols - xl) / 10.0) ** 2)
                img = img + glow * (np.array(ACC, np.float32) - img) * 0.9
            else:
                img = st
        if t >= ROBOTS0 - XF:
            rb = self.robots.frame(round((t - ROBOTS0 + XF) * FPS))
            img = img + smooth(t, ROBOTS0 - XF, ROBOTS0) * (rb - img)
        img = self.caption(img, t)
        img = np.array(BG) + (img - np.array(BG)) * smooth(t, 0, 0.5) * (1 - smooth(t, self.end - 0.5, self.end))
        return np.clip(img, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--station", help="Isaac Sim render of the rebuilt station (png, jpg or mp4)")
    ap.add_argument("--robots", help="the automated door clip, or the human-alone clip")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--still", type=float, nargs="*", help="write PNG stills at these seconds instead of the film")
    a = ap.parse_args()
    film = Film(a.station, a.robots)
    print(f"station: {a.station or 'placeholder'}, robots: {a.robots or 'placeholder'}, {film.end:.1f} s")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.still:
        for t in a.still:
            p = a.out.with_name(f"v4_still_{t:05.2f}.png")
            Image.fromarray(Film(a.station, a.robots).frame(t)).save(p)  # fresh readers: stills seek anywhere
            print(p)
        return
    ff = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
                           "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                           str(a.out)], stdin=subprocess.PIPE)
    for i in range(round(film.end * FPS)):
        ff.stdin.write(film.frame(i / FPS).tobytes())
    ff.stdin.close(); ff.wait()
    print(a.out)


if __name__ == "__main__":
    main()
