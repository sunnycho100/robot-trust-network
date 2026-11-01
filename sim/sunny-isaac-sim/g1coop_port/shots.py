"""Shot list for the video, shared by build_usd.py (authors one animated camera per shot) and render_video.py.
Cameras use MuJoCo's free-camera terms (lookat, distance, azimuth, elevation in degrees) so they match plant_view's VIEWS.
"""
import numpy as np

FPS = 30
# name, sim start s, sim end s, caption, tracked body (None = fixed lookat), focal length mm
SHOTS = [
    ('Overview', 0, 10, 'FLEXIBLE METALWORKS | 20 robots, one MuJoCo world, rendered in Isaac Sim RTX', None, 14.2),
    ('ThreeG1', 5, 17, '3 G1s | welded steel T-frame jig, side grip, onto a workbench', 'frame_payload', 20.0),
    ('TwoG1', 7, 18, '2 G1s | strapped steel sheet bundle, side grip, onto a sheet stack', 'crate_payload', 20.0),
    ('OneG1', 0, 5, '1 G1 | KLT tote of steel parts, side grip, onto a low table', 'sofa_payload', 20.0),
    # close-up clips of each carry, rendered as their own videos
    ('ThreeG1Close', 4, 18, '3 G1s | welded steel T-frame jig, side grip, close-up', 'frame_payload', 28.0),
    ('TwoG1Close', 5, 19, '2 G1s | strapped steel sheet bundle, side grip, close-up', 'crate_payload', 28.0),
    ('OneG1Close', 0, 10, '1 G1 | KLT tote of steel parts, side grip, close-up', 'sofa_payload', 28.0),
    # one close-up per factory section (fixed look-at points, slow orbit; camera definitions only)
    ('CNCArm', 0, 8, 'C1 MACHINING | UR5e loads and unloads the CNC', None, 24.0),
    ('CleaningArm', 4, 14, 'D1 CLEANING | UR5e at the deburr bench', None, 24.0),
    ('AssemblyArm', 0, 6, 'E1 ASSEMBLY | Panda stacks two components', None, 24.0),
    ('QCPanda', 10, 20, 'F1 QUALITY | TurtleBot hands a part to the QC Panda', None, 24.0),
    ('PackagingArm', 0, 6, 'G1 PACKING | UR10e places two boxes', None, 24.0),
    ('TwoArmPanel', 3, 9, 'G1 PACKING | two UR10e arms carry one panel', None, 24.0),
    ('CourierLanes', 0, 12, 'MAIN AISLE | couriers on their routes', None, 18.0),
    ('SpotInspection', 2, 14, 'INSPECTION | Spot walks its route', None, 24.0),
    ('SpotQuality', 2, 14, 'QUALITY AREA | second Spot on its route', None, 24.0),
    ('Shelving', 0, 6, 'A1 RECEIVING | stocked rack bays', None, 20.0),
    ('Dock', 0, 6, 'H1 SHIPPING | finished goods at the dock door', None, 20.0),
]
FILM = ['Overview', 'ThreeG1', 'TwoG1', 'OneG1']
SECTIONS = ['ThreeG1Close', 'TwoG1Close', 'OneG1Close', 'CNCArm', 'CleaningArm', 'AssemblyArm', 'QCPanda',
            'PackagingArm', 'TwoArmPanel', 'CourierLanes', 'SpotInspection', 'SpotQuality', 'Shelving', 'Dock']
# fixed shots: look-at point (world m), distance, azimuth at s = 0, azimuth change per s, elevation
FIXED = {
    'CNCArm': ((-5.6, 3.7, 1.0), 3.4, 62, 1.5, -24),
    'CleaningArm': ((-5.0, -3.9, .95), 3.2, 195, -1.0, -30),  # from the east: the extraction duct blocks the aisle side
    'AssemblyArm': ((1.5, -4.6, .95), 2.8, 280, -1.5, -28),
    'QCPanda': ((4.4, 3.7, .9), 2.8, 60, 1.0, -24),
    'PackagingArm': ((9.1, 4.2, 1.0), 3.2, 60, 1.5, -26),
    'TwoArmPanel': ((9.55, 4.2, 1.0), 4.8, 141, -2.0, -28),
    'CourierLanes': ((0., 1.5, .3), 8.5, 95, 1.0, -32),
    'SpotInspection': ((-.3, -.9, .45), 3.6, 116, 1.0, -22),
    'SpotQuality': ((6.5, 5.0, .45), 3.6, 71, 1.0, -22),
    'Shelving': ((-8.5, 5.6, 1.2), 6.0, 80, 1.0, -15),
    'Dock': ((12.3, -3.5, 1.3), 6.0, 342, 1.0, -14),
}
CREDIT = 'Depot: Rafi Abdullah & Louise Poubel / Open Robotics, CC-BY-4.0; robots: roqsim / MuJoCo Menagerie'


def orbit(name, s, look):
    """(lookat, distance, azimuth, elevation) at sim time s for a shot."""
    if name in FIXED:
        lookat, dist, az, daz, el = FIXED[name]
        return lookat, dist, az + daz * s, el
    if name == 'Overview':
        u = min(s / 10., 1.)
        u = u * u * (3 - 2 * u)
        return (0, .3 + .2 * (1 - u), .6), 25.5 - 4.5 * u, 112 - 30 * u, -40 - 12 * u
    if name == 'ThreeG1':
        return look, 5.2, 228 + 2.5 * s, -24
    if name == 'TwoG1':
        return look, 4.4, 128 - 2 * s, -24
    if name == 'OneG1':
        return look, 3.4, 72 + 3 * s, -20
    if name == 'ThreeG1Close':
        return look, 3.4, 200 + 2.5 * s, -30
    if name == 'TwoG1Close':
        return look, 3.1, 150 - 2.5 * s, -24
    return look, 2.6, 55 + 4 * s, -22  # OneG1Close


def eye_target(name, s, look):
    lookat, dist, az, el = orbit(name, s, look)
    az, el = np.radians(az), np.radians(el)
    fwd = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    lookat = np.asarray(lookat, float)
    return lookat - dist * fwd, lookat


def tracked(xpos, names, body, body_z=None):
    """Payload position smoothed over 1.5 s so the camera does not shake with the gait."""
    x = xpos[:, names.index(body)].astype(float)
    k = np.ones(45) / 45
    pad = np.pad(x, ((22, 22), (0, 0)), mode='edge')
    sm = np.stack([np.convolve(pad[:, c], k, mode='valid') for c in range(3)], 1)
    sm[:, 2] = .75 if body_z is None else sm[:, 2] + body_z
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
        look = tracked(xpos, names, body, -.1 if name.endswith('Close') else None) if body else None
        for f in range(n):
            eye, tgt = eye_target(name, f / FPS, look[f] if body else None)
            view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*tgt), Gf.Vec3d(0, 0, 1))
            op.Set(view.GetInverse(), f)


if __name__ == '__main__':  # self-check: azimuth 90 looks along +y, as in MuJoCo
    eye, tgt = eye_target('OneG1', 6, np.zeros(3))
    d = tgt - eye
    assert d[2] < 0 and abs(np.degrees(np.arctan2(d[1], d[0])) - 90) < 1e-6, d
    assert 20 <= sum(e - s for n, s, e, *_ in SHOTS if n in FILM) <= 40
    assert all(10 <= e - s <= 15 for n, s, e, *_ in SHOTS if n.endswith('Close'))
    assert all(n in FIXED for n, *_, body, _ in SHOTS if body is None and n != 'Overview')
    assert set(SECTIONS) <= {n for n, *_ in SHOTS}
    print('ok')
