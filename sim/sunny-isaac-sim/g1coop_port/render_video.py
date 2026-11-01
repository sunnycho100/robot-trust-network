"""Render the animated factory USD with Isaac Sim's RTX renderer, shot by shot, to PNG frames.
Usage (in the Isaac Sim 6.2 container):
  /isaac-sim/python.sh render_video.py /robot/g1coop/usd/factory.usd /robot/g1coop/frames [--mode PathTracing] [--every 30]
  [--shots ThreeG1Close] [--span 0 6]
High quality, no text: --mode PathTracing --spp 32 --subframes 4 --size 1920 1080 --hq --hdri URL --no-text
Look pass (look_hq.py layer as the USD): ... --hq --no-overlay --dof 60 --mblur 3 --post   (see RENDER.md)
Framing stills, one per shot: ... --shots CNCArm Dock --mid
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
p.add_argument('--size', nargs=2, type=int, default=(1280, 720))
p.add_argument('--no-overlay', '--no-text', dest='no_text', action='store_true',
               help='no title bar, time stamp or credit strip (credits go in a sidecar file)')
p.add_argument('--mid', action='store_true', help='only the middle frame of each shot (framing stills)')
p.add_argument('--hq', action='store_true', help='full-resolution textures and two soft area lights aimed at the shot')
p.add_argument('--hdri', help='dome light texture, e.g. an indoor HDRI from the Isaac asset root')
p.add_argument('--dof', type=float, default=0, help='f-stop for depth of field, focused at the shot target (0 = off)')
p.add_argument('--mblur', type=int, default=1, help='time samples averaged over a 180 deg shutter (1 = off)')
p.add_argument('--post', action='store_true', help='subtle bloom and vignette')
p.add_argument('--first-index', type=int, default=0, help='number of the first frame file (to resume into a folder)')
a = p.parse_args()
W, H = a.size
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
from PIL import Image, ImageDraw, ImageFilter, ImageFont  # noqa: E402
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
if a.hq:
    s.set('/rtx-transient/resourcemanager/enableTextureStreaming', False)  # full-resolution mips from the start
if a.hdri:
    from pxr import Sdf, UsdLux  # noqa: E402
    dome = UsdLux.DomeLight(stage.GetPrimAtPath('/World/Lights/Dome'))
    dome.CreateTextureFileAttr().Set(Sdf.AssetPath(a.hdri))
    dome.GetColorAttr().Set(Gf.Vec3f(1, 1, 1))

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


BIG, SMALL = font(round(22 * W / 1280), True), font(round(15 * W / 1280))


def post(img):
    """Subtle bloom from the brightest highlights and a soft vignette."""
    x = np.asarray(img, np.float32) / 255
    hot = (np.clip(x - .82, 0, None) / .18 * 255).astype(np.uint8)
    glow = np.asarray(Image.fromarray(hot).filter(ImageFilter.GaussianBlur(W / 90)), np.float32) / 255
    x = x + .12 * glow
    yy, xx = np.mgrid[0:H, 0:W]
    r2 = ((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2
    x *= (1 - .09 * r2)[..., None]
    return Image.fromarray((np.clip(x, 0, 1) * 255 + .5).astype(np.uint8))


def overlay(img, caption, t):
    d = ImageDraw.Draw(img, 'RGBA')
    k = W / 1280
    d.rectangle((0, 0, W, 44 * k), fill=(12, 18, 26, 200))
    d.text((18 * k, 10 * k), caption, font=BIG, fill=(240, 244, 246))
    d.text((W - 150 * k, 14 * k), f'sim t = {t:4.1f} s', font=SMALL, fill=(110, 225, 190))
    d.rectangle((0, H - 26 * k, W, H), fill=(12, 18, 26, 200))
    d.text((12 * k, H - 21 * k), shots.CREDIT + '  |  MuJoCo physics, Isaac Sim RTX render', font=SMALL, fill=(190, 196, 205))
    return img


for _ in range(60):  # materials and textures finish loading
    app.update()
n, t0 = a.first_index, time.time()
for name, start, end, caption, _, focal in shots.SHOTS:
    if name not in a.shots:
        continue
    src = UsdGeom.Xformable(stage.GetPrimAtPath(f'/World/Cameras/{name}'))
    cam.GetFocalLengthAttr().Set(focal)
    cam.GetHorizontalApertureAttr().Set(20.955)
    cam.GetVerticalApertureAttr().Set(20.955 * 9 / 16)
    first = True
    lo, hi = max(start, a.span[0]), min(end, a.span[1])
    if a.dof:  # focus on what the shot camera looks at (its orbit distance)
        cam.CreateFStopAttr(a.dof)
        cam.CreateFocusDistanceAttr(float(shots.orbit(name, lo, (0, 0, 0))[1]))
    if a.hq and not stage.GetPrimAtPath('/World/Lights/HQ_Key'):  # softboxes unless a look layer brought its own
        from pxr import UsdLux  # noqa: E402
        m = src.ComputeLocalToWorldTransform(round(lo * shots.FPS))
        eye, fwd = m.ExtractTranslation(), -Gf.Vec3d(*m.GetRow3(2)).GetNormalized()
        side = Gf.Cross(fwd, Gf.Vec3d(0, 0, 1)).GetNormalized()
        target = eye + 3.0 * fwd
        for i, sgn in enumerate((-1, 1)):
            box = UsdLux.RectLight.Define(stage, f'/World/Lights/Softbox{i}')
            box.CreateWidthAttr(1.6)
            box.CreateHeightAttr(1.0)
            box.CreateIntensityAttr(1500)
            box.CreateColorAttr(Gf.Vec3f(1, .98, .95))
            pos = target - 1.2 * fwd + sgn * 2.0 * side + Gf.Vec3d(0, 0, 2.2)
            xf = UsdGeom.Xformable(box)
            xf.ClearXformOpOrder()
            xf.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(pos, target, Gf.Vec3d(0, 0, 1)).GetInverse())
    frames = range(round(lo * shots.FPS), round(hi * shots.FPS), a.every)
    if a.mid:
        frames = [round((lo + hi) / 2 * shots.FPS)]
    for f in frames:
        acc = 0.
        for k in range(a.mblur):  # samples spread over half a frame, centred on the frame time
            t = f + (k / (a.mblur - 1) - .5) * .5 if a.mblur > 1 else f
            tl.set_current_time(t / shots.FPS)
            cam_op.Set(src.ComputeLocalToWorldTransform(t))
            # a camera cut needs the denoiser history flushed; extra subframes do that
            rep.orchestrator.step(rt_subframes=a.subframes * (4 if first else 1), delta_time=0.0, pause_timeline=True)
            first = False
            acc = acc + np.asarray(rgb.get_data())[..., :3].astype(np.float32)
        img = Image.fromarray((acc / a.mblur + .5).astype(np.uint8))
        if a.post:
            img = post(img)
        (img if a.no_text else overlay(img, caption, f / shots.FPS)).save(os.path.join(a.out, f'frame_{n:05d}.png'))
        n += 1
        if (n - a.first_index) % 60 == 0:
            print(f'[g1coop] {n} frames, {time.time() - t0:.0f}s, shot {name} f{f}', flush=True)
k = n - a.first_index
print(f'[g1coop] done: {k} frames in {time.time() - t0:.0f}s, {(time.time() - t0) / max(k, 1):.2f} s per frame', flush=True)
app.close()
