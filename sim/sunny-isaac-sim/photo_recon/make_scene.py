"""Turn perception.json plus the photo inventory into scene.json for build_scene.py.
Usage: python make_scene.py perception.json scene.json
Layout is in a building frame: the yellow floor rectangle and the guardrails line up with axes at 40 deg to the camera heading.
Measured units come from Depth Pro and floor-ray hits; K rescales them so stock assets of known size (forklift guard,
cones, guardrail top rail) match. Scaling about the camera keeps the image alignment exact.
"""
import json
import math
import sys

import numpy as np

d = json.load(open(sys.argv[1]))
R, H0, F, W, H = np.array(d["R_world_from_cam"]), d["cam_height"], d["f_exif"], d["W"], d["H"]
K = 1.12              # measured heights: forklift 1.88 vs 2.15, cones 0.58 vs 0.71, top rail 0.94-1.03 vs 1.07
TH = math.radians(40)  # building axes vs camera heading (floor rectangle edges, guardrail runs)
NV = "{NV}/Assets/DigitalTwin/Assets/Warehouse"
CONE = NV + "/Safety/Cones/Colored_Traffic/ColoredTrafficCone_A01_71cm_PR_V_NVD_01.usd"
COLP = NV + "/Safety/Barriers/ColumnProtector_A/ColumnProtector_A01_Yellow_01.usd"
FORK = "{ROOT}/Isaac/Props/Forklift/forklift.usd"
PALLET = "{ROOT}/Isaac/Props/Pallet/pallet.usd"
YEL, WHITE, BRICK, DARK = [0.95, 0.72, 0.05], [0.82, 0.83, 0.82], [0.24, 0.12, 0.09], [0.1, 0.1, 0.11]


def ray(u, v):
    r = R @ np.array([(u - W / 2) / F, (v - H / 2) / F, 1.0])
    return r


def bld(p):   # world xy (measured) -> building frame
    c, s = math.cos(TH), math.sin(TH)
    return np.array([p[0] * c + p[1] * s, -p[0] * s + p[1] * c])


def at_z(u, v, z):   # pixel ray hitting the plane at height z, in building frame (measured units)
    r = ray(u, v)
    return bld(r[:2] * (z - H0) / r[2])


def W2(pb):   # building frame (measured) -> world (scaled)
    c, s = math.cos(TH), math.sin(TH)
    return [K * (pb[0] * c - pb[1] * s), K * (pb[0] * s + pb[1] * c)]


items = []


def box(name, c, size, rgb, yaw=0.0, rough=0.6, emit=None, tilt=(0, 0), z=None):
    """c is a building-frame centre (x, y) or (x, y, z) in measured units; size in measured units."""
    zc = c[2] if len(c) == 3 else z
    items.append(dict(name=name, kind="box", center=[*W2(c[:2]), K * zc], size=[K * v for v in size], rgb=rgb,
                      yaw=yaw + math.degrees(TH), rough=rough, emit=emit, tilt=list(tilt)))


def asset(name, url, c, yaw=0.0, z=0.0):
    items.append(dict(name=name, kind="usd", url=url, pos=[*W2(c), K * z], yaw=yaw + math.degrees(TH)))


def rail(name, a, b):
    items.append(dict(name=name, kind="rail", a=W2(a), b=W2(b)))


# ---------------------------------------------------------------- shell
box("floor", (12, 20, -0.04), (60, 80, 0.1), [0.43, 0.43, 0.41], rough=0.35)
box("wall_right", (28.5, 20, 5), (0.4, 80, 10), BRICK, rough=0.9)
box("wall_left", (-7, 20, 5), (0.4, 80, 10), BRICK, rough=0.9)
box("wall_end", (12, 47, 5), (70, 0.4, 10), BRICK, rough=0.9)
box("wall_back", (12, -9, 5), (70, 0.4, 10), BRICK, rough=0.9)
xc = at_z(820, 160, 8.9)[0]   # clerestory band seen at the top of the photo, runs along the aisle
box("roof_a", ((xc - 1.6 - 7) / 2, 20, 9.9), (xc - 1.6 + 7, 80, 0.2), [0.2, 0.2, 0.21])
box("roof_b", ((xc + 1.6 + 29) / 2, 20, 9.9), (29 - xc - 1.6, 80, 0.2), [0.2, 0.2, 0.21])
for sgn in (-1, 1):
    box(f"clerestory_{sgn}", (xc + 1.6 * sgn, 20, 9.2), (0.1, 80, 1.2), [0.9, 0.95, 1.0], emit=[3.0, 3.1, 3.3])
for i, yb in enumerate(np.arange(-6, 46, 6.0)):   # trusses span across the aisle, every 6 m
    box(f"truss_bot_{i}", (11, yb, 7.6), (36, 0.18, 0.22), DARK)
    box(f"truss_top_{i}", (11, yb, 9.4), (36, 0.18, 0.22), DARK)
    for j, xb in enumerate(np.arange(-5.5, 28, 3.0)):
        box(f"truss_web_{i}_{j}", (xb, yb, 8.5), (0.1, 0.1, 2.5), DARK, tilt=(0, 40 * (-1) ** j))

# ---------------------------------------------------------------- doors on the right brick wall and the end wall
for name, yb, w, h in [("door_a", 7.7, 4.6, 4.8), ("door_b", 15.5, 4.3, 4.3)]:
    box(name + "_frame", (28.25, yb, h / 2 + 0.15), (0.1, w + 0.5, h + 0.3), [0.8, 0.08, 0.06])
    box(name + "_curtain", (28.15, yb, h / 2), (0.1, w, h), [0.85, 0.88, 0.9], emit=[1.6, 1.65, 1.7], rough=0.2)
box("door_end", (-1.6, 46.75, 2.0), (3.5, 0.1, 4.0), [0.12, 0.13, 0.14])

# ---------------------------------------------------------------- floor marking and cones
box("rect_border", (6.2, 1.73, 0.012), (5.35, 2.4, 0.004), [0.03, 0.03, 0.03])
box("rect", (6.2, 1.73, 0.014), (5.15, 2.2, 0.004), [0.95, 0.8, 0.05], rough=0.25)
for i, uv in enumerate([(1160, 845), (1450, 725), (1800, 755)]):
    asset(f"cone_{i}", CONE, bld(ray(*uv)[:2] * (-H0 / ray(*uv)[2])))
a = bld(ray(30, 785)[:2] * (-H0 / ray(30, 785)[2]))   # yellow aisle line, bottom left of the photo
box("aisle_line", (a[0], 20, 0.013), (0.12, 60, 0.004), YEL)

# ---------------------------------------------------------------- forklift with the carrot load (wheel contacts -> heading -20 deg)
HEAD = -20.0
asset("forklift", FORK, (3.83, 6.2), yaw=HEAD + 90)   # ponytail: asset length runs along its y; flip 180 if forks face back
pc, hd = np.array([4.95, 6.0]), np.array([math.cos(math.radians(HEAD)), math.sin(math.radians(HEAD))])
nm = np.array([-hd[1], hd[0]])
asset("pallet", PALLET, pc, yaw=HEAD, z=0.12)
rng = np.random.default_rng(1)
for L in range(7):
    for i in range(3):
        for j in range(2):
            c = pc + hd * (i - 1) * 0.38 + nm * (j - 0.5) * 0.48 + rng.normal(0, 0.03, 2)
            box(f"sack_{L}_{i}_{j}", (*c, 0.36 + 0.2 * L), (0.4, 0.5, 0.21), [0.86, 0.42, 0.22], yaw=HEAD + rng.normal(0, 6), rough=0.95)

# ---------------------------------------------------------------- columns
for name, c, rgb in [("col_mid", (9.4, 8.5), WHITE), ("col_front", (2.3, 0.05), YEL)]:
    box(name, (*c, 4.5), (0.45, 0.45, 9.0), rgb)
    asset(name + "_guard", COLP, c)

# ---------------------------------------------------------------- guardrails (stock GuardRail_A sections)
rail("rail_left_far", (1.7, 13.0), (1.7, 24.0))
rail("rail_left_stair", (1.85, 8.5), (1.85, 11.5))
rail("rail_stair_front", (1.9, 7.4), (3.4, 7.4))
rail("rail_office", (6.3, 8.4), (8.9, 8.4))
rail("rail_booth", (11.9, 8.7), (17.4, 8.7))

# ---------------------------------------------------------------- white wall block, two-storey office, stair, booth
box("white_block", (7.5, 29, 4.3), (9.8, 32, 8.6), WHITE)
box("office_low", (8.2, 11.3, 1.45), (8.4, 3.6, 2.9), WHITE)
box("office_up", (7.8, 11.3, 4.05), (9.2, 3.6, 2.3), WHITE)   # top at 5.2 m from the photo ray
for i, xb in enumerate([6.4, 8.8]):
    box(f"office_win_{i}", (xb, 9.48, 4.27), (1.3, 0.06, 1.3), [0.75, 0.8, 0.75], emit=[0.9, 0.95, 0.85], rough=0.1)
box("office_door", (3.18, 10.4, 4.1), (0.06, 0.9, 2.1), [0.7, 0.72, 0.72])
for k in range(15):   # flight rises along the aisle toward the office door
    box(f"step_{k}", (3.4, 8.2 + 0.24 * k, 0.2 * k + 0.1), (1.1, 0.26, 0.05), [0.12, 0.12, 0.12])
for sx in (2.82, 3.98):
    box(f"stringer_{sx}", (sx, 9.9, 1.55), (0.06, 4.4, 0.25), YEL, tilt=(math.degrees(math.atan2(3.0, 3.6)), 0))
    box(f"handrail_{sx}", (sx, 9.9, 2.55), (0.05, 4.4, 0.05), YEL, tilt=(math.degrees(math.atan2(3.0, 3.6)), 0))
box("landing", (3.4, 12.0, 2.95), (1.4, 1.4, 0.1), [0.15, 0.15, 0.15])
for k, (px, py) in enumerate([(2.75, 11.35), (4.05, 11.35), (2.75, 12.65), (4.05, 12.65)]):
    box(f"landing_post_{k}", (px, py, 1.5), (0.1, 0.1, 3.0), YEL)
    box(f"landing_rail_post_{k}", (px, py, 3.5), (0.05, 0.05, 1.05), YEL)
box("landing_rail_a", (3.4, 11.35, 4.0), (1.35, 0.05, 0.05), YEL)
box("landing_rail_b", (4.05, 12.0, 4.0), (0.05, 1.35, 0.05), YEL)
box("booth", (15.0, 11.3, 1.35), (5.0, 3.6, 2.7), WHITE)
box("booth_door", (13.6, 9.48, 1.05), (0.9, 0.06, 2.1), [0.75, 0.76, 0.76])
box("booth_win", (15.4, 9.48, 1.6), (1.2, 0.06, 1.0), [0.75, 0.8, 0.75], emit=[0.6, 0.65, 0.6], rough=0.1)

# ---------------------------------------------------------------- lights: high-bay pendants and daylight through the doors
pend = [W2((xb, yb)) + [K * 7.3] for xb in (2, 10, 18, 26) for yb in (2, 10, 18, 26, 34)]
scene = dict(items=items, K=K, theta_deg=math.degrees(TH), cam_height=K * H0,
             lights=[dict(kind="dome", intensity=150), dict(kind="sphere", intensity=22000, radius=0.35, at=pend)],
             orbit_target=[*W2((6.0, 10.0)), 1.5], orbit_deg=75, orbit_rise=6.0)
json.dump(scene, open(sys.argv[2], "w"), indent=1)
print(len(items), "items, clerestory at x_b", round(xc, 2), "aisle line at x_b", round(float(a[0]), 2))
