"""Shot list for the video, shared by build_usd.py (authors one animated camera per shot) and render_video.py.
Cameras use MuJoCo's free-camera terms (lookat, distance, azimuth, elevation in degrees) so they match plant_view's VIEWS.
"""
import numpy as np

FPS = 30
# name, sim start s, sim end s, caption, tracked body (None = fixed lookat), focal length mm
SHOTS = [
    ('Overview', 0, 12, 'FLEXIBLE METALWORKS | 20 robots, one MuJoCo world, rendered in Isaac Sim RTX', None, 14.2),
    ('ThreeG1', 0, 11, '3 G1s | eccentric T-fixture, 3 kg, about 2 m', 'frame_payload', 20.0),
    ('TwoG1', 3, 13, '2 G1s | equipment crate, 5 kg, bottom support', 'crate_payload', 20.0),
    ('OneG1', 0, 6, '1 G1 | sofa module, 2 kg, side grip and placement', 'sofa_payload', 20.0),
]
CREDIT = 'Depot: Rafi Abdullah & Louise Poubel / Open Robotics, CC-BY-4.0; robots: roqsim / MuJoCo Menagerie'


def orbit(name, s, look):
    """(lookat, distance, azimuth, elevation) at sim time s for a shot."""
    if name == 'Overview':
        u = min(s / 12., 1.)
        u = u * u * (3 - 2 * u)
        return (0, .3 + .2 * (1 - u), .6), 25.5 - 4.5 * u, 112 - 30 * u, -40 - 12 * u
    if name == 'ThreeG1':
        return look, 5.2, 228 + 2.5 * s, -24
    if name == 'TwoG1':
        return look, 4.4, 128 - 2 * s, -24
    return look, 3.4, 72 + 3 * s, -20


def eye_target(name, s, look):
    lookat, dist, az, el = orbit(name, s, look)
    az, el = np.radians(az), np.radians(el)
    fwd = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    lookat = np.asarray(lookat, float)
    return lookat - dist * fwd, lookat


def tracked(xpos, names, body):
    """Payload position smoothed over 1.5 s so the camera does not shake with the gait."""
    x = xpos[:, names.index(body)].astype(float)
    k = np.ones(45) / 45
    pad = np.pad(x, ((22, 22), (0, 0)), mode='edge')
    sm = np.stack([np.convolve(pad[:, c], k, mode='valid') for c in range(3)], 1)
    sm[:, 2] = .75
    return sm


def author_camera(stage, xpos, names, n):
    from pxr import Gf, UsdGeom
    for name, _, _, _, body, focal in SHOTS:
        cam = UsdGeom.Camera.Define(stage, f'/World/Cameras/{name}')
        cam.CreateFocalLengthAttr(focal)
        cam.CreateHorizontalApertureAttr(20.955)
        cam.CreateVerticalApertureAttr(20.955 * 9 / 16)
        cam.CreateClippingRangeAttr(Gf.Vec2f(.05, 500))
        op = cam.AddTransformOp()
        look = tracked(xpos, names, body) if body else None
        for f in range(n):
            eye, tgt = eye_target(name, f / FPS, look[f] if body else None)
            view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*tgt), Gf.Vec3d(0, 0, 1))
            op.Set(view.GetInverse(), f)


if __name__ == '__main__':  # self-check: azimuth 90 looks along +y, as in MuJoCo
    eye, tgt = eye_target('OneG1', 6, np.zeros(3))
    d = tgt - eye
    assert d[2] < 0 and abs(np.degrees(np.arctan2(d[1], d[0])) - 90) < 1e-6, d
    assert sum(e - s for _, s, e, *_ in SHOTS) <= 40
    print('ok')
