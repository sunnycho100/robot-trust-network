"""Render the animated factory USD with Isaac Sim's RTX renderer, shot by shot, to PNG frames.
Usage (in the Isaac Sim 6.2 container):
  /isaac-sim/python.sh render_video.py /robot/g1coop/usd/factory.usd /robot/g1coop/frames [--mode PathTracing] [--every 30]
  [--shots ThreeG1Close] [--span 0 6]
Shots and captions come from shots.py. Frames are replayed from USD time samples (MuJoCo physics, RTX rendering).
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shots  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument('usd')
p.add_argument('out')
p.add_argument('--mode', default='RealTimePathTracing', help='RaytracedLighting | RealTimePathTracing | PathTracing')
p.add_argument('--spp', type=int, default=32, help='samples per pixel per subframe (PathTracing)')
p.add_argument('--subframes', type=int, default=4)
p.add_argument('--every', type=int, default=1, help='render every Nth frame (previews)')
p.add_argument('--shots', nargs='*', default=shots.FILM, help='shot names, default the film')
p.add_argument('--span', nargs=2, type=float, default=(0, 1e9), help='only sim seconds [A, B) of those shots (parallel chunks)')
a = p.parse_args()
W, H = 1280, 720
os.makedirs(a.out, exist_ok=True)

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({'headless': True, 'width': W, 'height': H})
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

enable_extension('omni.replicator.core')
import carb  # noqa: E402
import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from pxr import Gf, UsdGeom  # noqa: E402

ctx = omni.usd.get_context()
ctx.open_stage(a.usd)
while ctx.get_stage_loading_status()[2] > 0:
    app.update()
stage = ctx.get_stage()
s = carb.settings.get_settings()
s.set('/rtx/rendermode', a.mode)
s.set('/rtx/pathtracing/spp', a.spp)
s.set('/rtx/pathtracing/totalSpp', a.spp * a.subframes)  # time holds still across subframes, so they accumulate
s.set('/rtx/pathtracing/optixDenoiser/enabled', True)
s.set('/rtx/post/motionblur/enabled', False)
s.set('/omni/replicator/captureOnPlay', False)

cam = UsdGeom.Camera.Define(stage, '/World/RenderCam')
cam_op = cam.AddTransformOp()
rp = rep.create.render_product('/World/RenderCam', (W, H))
rgb = rep.AnnotatorRegistry.get_annotator('rgb')
rgb.attach(rp)
tl = omni.timeline.get_timeline_interface()
tl.stop()


def font(size, bold=False):
    for f in ('/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf' % ('-Bold' if bold else ''),):
        if os.path.exists(f):
            return ImageFont.truetype(f, size)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # ponytail: Pillow < 10.1 has no sized default font
        return ImageFont.load_default()


BIG, SMALL = font(22, True), font(15)


def overlay(img, caption, t):
    d = ImageDraw.Draw(img, 'RGBA')
    d.rectangle((0, 0, W, 44), fill=(12, 18, 26, 200))
    d.text((18, 10), caption, font=BIG, fill=(240, 244, 246))
    d.text((W - 150, 14), f'sim t = {t:4.1f} s', font=SMALL, fill=(110, 225, 190))
    d.rectangle((0, H - 26, W, H), fill=(12, 18, 26, 200))
    d.text((12, H - 21), shots.CREDIT + '  |  MuJoCo physics, Isaac Sim RTX render', font=SMALL, fill=(190, 196, 205))
    return img


for _ in range(60):  # materials and textures finish loading
    app.update()
n, t0 = 0, time.time()
for name, start, end, caption, _, focal in shots.SHOTS:
    if name not in a.shots:
        continue
    src = UsdGeom.Xformable(stage.GetPrimAtPath(f'/World/Cameras/{name}'))
    cam.GetFocalLengthAttr().Set(focal)
    cam.GetHorizontalApertureAttr().Set(20.955)
    cam.GetVerticalApertureAttr().Set(20.955 * 9 / 16)
    first = True
    lo, hi = max(start, a.span[0]), min(end, a.span[1])
    for f in range(round(lo * shots.FPS), round(hi * shots.FPS), a.every):
        tl.set_current_time(f / shots.FPS)
        cam_op.Set(src.ComputeLocalToWorldTransform(f))
        # a camera cut needs the denoiser history flushed; extra subframes do that
        rep.orchestrator.step(rt_subframes=a.subframes * (4 if first else 1), delta_time=0.0, pause_timeline=True)
        first = False
        img = Image.fromarray(np.asarray(rgb.get_data())[..., :3])
        overlay(img, caption, f / shots.FPS).save(os.path.join(a.out, f'frame_{n:05d}.png'))
        n += 1
        if n % 60 == 0:
            print(f'[g1coop] {n} frames, {time.time() - t0:.0f}s, shot {name} f{f}', flush=True)
print(f'[g1coop] done: {n} frames in {time.time() - t0:.0f}s', flush=True)
app.close()
