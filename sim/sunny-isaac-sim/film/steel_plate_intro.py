"""Factory overview -> zoom into the sheet cell -> "Lift this steel plate" -> CooHOI pair clip, in the website's type.
Usage: python steel_plate_intro.py FACTORY.mp4 CARRY.mp4 OUT.mp4 [--start 17]
Needs ffmpeg and Pillow; fonts are the site's (Archivo, IBM Plex Mono) in fonts/ (see README.md).
"""
import argparse
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONTS = Path(__file__).parent / "fonts"
BG, INK, ACCENT = (10, 12, 12), (228, 234, 231), (111, 224, 207)  # site --bg --ink --accent
W, H, FPS = 1280, 720, 30
CX, CY, Z = 250, 470, 3.0  # zoom target in the factory frame (the sheet handling cell) and final zoom
HOLD, ZOOM, TITLE, XFADE = 2.0, 2.5, 3.5, 0.6  # seconds: overview, zoom move, title hold, crossfade
STACK = (150, 440, 215, 505)  # the sheet stack, in factory frame pixels


def font(name, size, weight=None):
    f = ImageFont.truetype(str(FONTS / name), size)
    if weight:
        f.set_variation_by_axes([weight, 100])  # Archivo axes: wght, wdth
    return f


def to_zoomed(x, y):
    x0, y0 = CX - W / Z / 2, CY - H / Z / 2
    return (x - x0) * Z, (y - y0) * Z


def tag(draw, xy, text):
    f = font("IBMPlexMono-Medium.ttf", 18)
    x, y = xy
    w = draw.textlength(text, font=f)
    draw.rectangle((x, y, x + w + 24, y + 34), fill=ACCENT)
    draw.text((x + 12, y + 6), text, font=f, fill=BG)


def title_overlay(path):
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    for i in range(220):  # bottom fade so the type reads over the floor
        d.line((0, H - i, W, H - i), fill=(*BG, int(215 * (1 - i / 220))))
    x0, y0 = to_zoomed(STACK[0], STACK[1])
    x1, y1 = to_zoomed(STACK[2], STACK[3])
    L = 34
    for cx, cy, sx, sy in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        d.line((cx, cy, cx + sx * L, cy), fill=ACCENT, width=4)
        d.line((cx, cy, cx, cy + sy * L), fill=ACCENT, width=4)
    d.text((x0, y0 - 30), "STEEL PLATE", font=font("IBMPlexMono-Medium.ttf", 18), fill=ACCENT)
    tag(d, (56, H - 168), "TASK")
    d.text((54, H - 122), "Lift this steel plate", font=font("Archivo.ttf", 64, 760), fill=INK)
    im.save(path)


def caption_overlay(path):
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    for i in range(150):
        d.line((0, H - i, W, H - i), fill=(*BG, int(200 * (1 - i / 150))))
    tag(d, (56, H - 104), "MULTI-AGENT RL")
    d.text((56 + 250, H - 100), "CooHOI policy (NeurIPS 2024), running in our Isaac Lab sim",
           font=font("IBMPlexMono-Medium.ttf", 21), fill=INK)
    im.save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("factory")
    ap.add_argument("carry")
    ap.add_argument("out")
    ap.add_argument("--start", type=float, default=17.0, help="where to cut into the factory video (s)")
    a = ap.parse_args()
    tmp = Path(tempfile.mkdtemp())
    title_overlay(tmp / "title.png")
    caption_overlay(tmp / "caption.png")
    intro = HOLD + ZOOM + TITLE
    f0, n = HOLD * FPS, ZOOM * FPS
    e = f"(st(0,clip((on-{f0})/{n},0,1))*ld(0)*(3-2*ld(0)))"  # smoothstep progress of the zoom move
    zp = (f"scale={W * 2}:{H * 2},zoompan=z='1+{Z - 1}*{e}'"
          f":x='({W}+({CX * 2}-{W})*{e})-iw/zoom/2':y='({H}+({CY * 2}-{H})*{e})-ih/zoom/2'"
          f":d=1:s={W}x{H}:fps={FPS}")
    t_title = HOLD + ZOOM + 0.2
    graph = (f"[0:v]fps={FPS},{zp},trim=duration={intro},setpts=PTS-STARTPTS,format=yuv420p[z];"
             f"[2:v]format=rgba,fade=in:st={t_title}:d=0.5:alpha=1[t];"
             f"[z][t]overlay=0:0:shortest=1,format=yuv420p,setsar=1[a];"
             f"[1:v]fps={FPS},scale={W}:{H},setsar=1,format=yuv420p[c0];"
             f"[3:v]format=rgba,fade=in:st=1.2:d=0.5:alpha=1[cap];"
             f"[c0][cap]overlay=0:0:shortest=1,format=yuv420p[c];"
             f"[a][c]xfade=transition=fade:duration={XFADE}:offset={intro - XFADE}[v]")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", str(a.start), "-t", str(intro + 1), "-i", a.factory,
                    "-i", a.carry, "-loop", "1", "-t", str(intro + 1), "-i", str(tmp / "title.png"),
                    "-loop", "1", "-i", str(tmp / "caption.png"),
                    "-filter_complex", graph, "-map", "[v]", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                    "-shortest", a.out], check=True)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
