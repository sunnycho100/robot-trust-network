"""Side-by-side, 50% blend and orbit video, with the photo credit on screen.
Usage: python compose.py photo.jpg RENDER_DIR OUT_DIR   (RENDER_DIR has photo_view.png and optional orbit_*.png)
"""
import glob
import os
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont

photo, rdir, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
CREDIT = "Photo: Lance Cheung, USDA, via Wikimedia Commons (public domain)"


def font(n):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", n)
    except OSError:
        return ImageFont.load_default(size=n)


def bar(im, text, n=20):
    d = ImageDraw.Draw(im)
    d.rectangle([0, im.height - n - 14, im.width, im.height], fill=(0, 0, 0))
    d.text((10, im.height - n - 7), text, fill=(235, 235, 235), font=font(n))
    return im


ren = Image.open(os.path.join(rdir, "photo_view.png")).convert("RGB")
ph = Image.open(photo).convert("RGB").resize(ren.size, Image.LANCZOS)
sbs = Image.new("RGB", (ren.width * 2 + 10, ren.height), (0, 0, 0))
sbs.paste(ph, (0, 0))
sbs.paste(ren, (ren.width + 10, 0))
d = ImageDraw.Draw(sbs)
for x, t in [(12, "Photo"), (ren.width + 22, "Isaac Sim 6.2, rebuilt from that one photo")]:
    d.rectangle([x - 6, 8, x + 14 * len(t), 44], fill=(0, 0, 0))
    d.text((x, 12), t, fill=(255, 255, 255), font=font(26))
bar(sbs, CREDIT + ". Render: same camera (17 mm on a 35.9 mm sensor, from EXIF)").save(f"{out}/side_by_side.jpg", quality=92)
bar(Image.blend(ph, ren, 0.5), "50% blend of photo and render. " + CREDIT, 18).save(f"{out}/blend.jpg", quality=92)

frames = sorted(glob.glob(os.path.join(rdir, "orbit_*.png")))
if frames:
    os.makedirs(f"{out}/ov", exist_ok=True)
    inset = ph.resize((ren.width // 4, ren.height // 4), Image.LANCZOS)
    for i, f in enumerate(frames):
        im = Image.open(f).convert("RGB")
        im.paste(inset, (im.width - inset.width - 10, 10))
        ImageDraw.Draw(im).rectangle([im.width - inset.width - 11, 9, im.width - 9, inset.height + 11], outline=(255, 255, 255))
        bar(im, "Isaac Sim 6.2 scene rebuilt from one photo (inset). " + CREDIT, 18).save(f"{out}/ov/{i:04d}.jpg", quality=92)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", "30", "-i", f"{out}/ov/%04d.jpg",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", f"{out}/orbit.mp4"], check=True)
print("COMPOSE_DONE")
