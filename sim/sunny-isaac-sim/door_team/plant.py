"""Car-plant dressing around the door cell for door_team.py: the CarDA WS30 floor belt with its hazard edges and blue
walkways, door trolleys riding the belt, and a semi-busy background (a body line with car bodies on skids, a fenced
weld cell with two KR210 looping, a cobot station, parts racks, KLT carts, an AGV shuttling, workbenches and a
calibration board). Authored boxes plus stock Isaac assets; background motion is looped joint angles, kinematic only.

build(ctx) authors everything once; animate(t) poses the moving parts for one frame. ctx carries the helpers and
the stage from door_team.py (box, cyl, xf, setT, reference, material, app, rig, ROOT, L: layout dict).
"""
import math

import numpy as np

from kin import minjerk, rot, tf

KLT = "/Isaac/Props/KLT_Bin/small_KLT_visual.usd"
KR210 = "/Isaac/Robots/Kuka/KR210_L150/kr210_l150.usd"
CRX = "/Isaac/Robots/Fanuc/crx10ia/crx10ia.usd"
AMR = "/Isaac/Robots/Idealworks/iwhub/iw_hub_static.usd"
C = None
MOVING = {}


def trolley(path, with_door, door_mat):
    """CarDA-style door trolley: steel frame on casters, door cradle on top, KLT bins on the lower shelf.
    Frame: x along the belt, y across; the door stands across the belt (as in the cell)."""
    b, z = C.box, C.L["FIX_Z"]
    C.xf(path)
    b(f"{path}/base", (0, 0, 0.14), (0.62, 1.30, 0.05), "steel")
    b(f"{path}/shelf", (0, 0, 0.36), (0.58, 1.20, 0.03), "steel")
    for k, (dx, dy) in enumerate([(-0.28, -0.6), (0.28, -0.6), (-0.28, 0.6), (0.28, 0.6)]):
        b(f"{path}/post{k}", (dx, dy, (z + 0.14) / 2), (0.04, 0.04, z - 0.14), "steel")
        C.cyl(f"{path}/wheel{k}", (dx, dy, 0.06), 0.05, 0.04, "dark", axis="Y")
    b(f"{path}/cradle", (0, 0, z - 0.02), (0.16, 1.24, 0.04), "dark")
    for k, dy in enumerate((-0.45, 0.45)):
        b(f"{path}/locator{k}", (0, dy, z + 0.02), (0.16, 0.05, 0.04), "yellow")
    for k, dy in enumerate((-0.35, 0.0, 0.35)):
        b(f"{path}/klt{k}", (0.0, dy, 0.44), (0.30, 0.22, 0.13), "kltblue")
    if with_door:
        door_geom(f"{path}/door", door_mat)
        C.setT(C.stage.GetPrimAtPath(f"{path}/door"), tf(rot(2, math.pi / 2), (0, 0, z)))


def door_geom(path, mat):
    b = C.box
    C.xf(path)
    b(f"{path}/panel", (0, 0, 0.275), (1.2, 0.10, 0.55), mat)
    b(f"{path}/beltline", (0.02, 0, 0.56), (1.16, 0.11, 0.03), "doortrim")
    b(f"{path}/rearpillar", (0.55, 0, 0.80), (0.06, 0.09, 0.5), "doortrim")
    b(f"{path}/roofrail", (0.2, 0, 1.03), (0.76, 0.09, 0.05), "doortrim")
    b(f"{path}/apillar", (-0.31, 0, 0.80), (0.05, 0.09, 0.56), "doortrim", ry=32.0)


def car_body(path, paint):
    """Body in white on a skid: boxes, about 4.4 m long."""
    b = C.box
    C.xf(path)
    for k, y in enumerate((-0.55, 0.55)):
        b(f"{path}/skid{k}", (0, y, 0.10), (4.6, 0.10, 0.08), "steel")
    for k, x in enumerate((-1.8, -0.6, 0.6, 1.8)):
        b(f"{path}/cross{k}", (x, 0, 0.16), (0.10, 1.2, 0.05), "steel")
    b(f"{path}/floor", (0, 0, 0.42), (4.3, 1.75, 0.10), paint)
    b(f"{path}/sill0", (0, -0.86, 0.62), (3.0, 0.06, 0.32), paint)
    b(f"{path}/sill1", (0, 0.86, 0.62), (3.0, 0.06, 0.32), paint)
    b(f"{path}/front", (1.75, 0, 0.72), (0.9, 1.75, 0.50), paint)
    b(f"{path}/rear", (-1.8, 0, 0.74), (0.8, 1.75, 0.55), paint)
    b(f"{path}/roof", (-0.15, 0, 1.38), (1.9, 1.45, 0.06), paint)
    for k, (x, y) in enumerate([(0.85, -0.75), (0.85, 0.75), (-1.1, -0.75), (-1.1, 0.75), (-0.1, -0.8), (-0.1, 0.8)]):
        b(f"{path}/pillar{k}", (x, y, 1.08), (0.08, 0.08, 0.62), paint)


def rack(path, x, y, rz=0.0, levels=3):
    """Yellow pallet rack with KLT bins."""
    b = C.box
    C.xf(path)
    for k, (dx, dy) in enumerate([(-0.9, -0.4), (0.9, -0.4), (-0.9, 0.4), (0.9, 0.4)]):
        b(f"{path}/up{k}", (dx, dy, 1.0), (0.08, 0.08, 2.0), "yellow")
    for lv in range(levels):
        z = 0.25 + lv * 0.62
        b(f"{path}/beam{lv}", (0, 0, z), (1.9, 0.9, 0.04), "orange")
        for j in range(4):
            b(f"{path}/bin{lv}_{j}", (-0.66 + j * 0.44, 0.0, z + 0.09), (0.36, 0.56, 0.14), "kltblue" if (lv + j) % 3 else "kltgray")
    C.setT(C.stage.GetPrimAtPath(path), tf(rot(2, math.radians(rz)), (x, y, 0)))


def klt_cart(path, x, y, rz=0.0):
    b = C.box
    C.xf(path)
    b(f"{path}/deck", (0, 0, 0.22), (0.9, 0.6, 0.04), "steel")
    b(f"{path}/handle", (-0.45, 0, 0.6), (0.04, 0.55, 0.8), "steel")
    for k, (dx, dy) in enumerate([(-0.4, -0.25), (0.4, -0.25), (-0.4, 0.25), (0.4, 0.25)]):
        C.cyl(f"{path}/w{k}", (dx, dy, 0.07), 0.06, 0.04, "dark", axis="Y")
    for j in range(6):
        b(f"{path}/bin{j}", (-0.2 + (j % 2) * 0.4, 0.0, 0.31 + (j // 2) * 0.15), (0.36, 0.56, 0.14), "kltblue")
    C.setT(C.stage.GetPrimAtPath(path), tf(rot(2, math.radians(rz)), (x, y, 0)))


def fence(path, x0, y0, x1, y1, h=2.0):
    """Wire-mesh safety fence around a rectangle, with a gap for the infeed."""
    b = C.box
    C.xf(path)
    sides = [((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))]
    k = 0
    for (ax, ay), (bx, by) in sides:
        n = max(1, int(round(math.hypot(bx - ax, by - ay) / 1.5)))
        for i in range(n):
            if ax == x0 and bx == x0 and i == n // 2:
                continue                                   # infeed gap on the -x side
            s0, s1 = i / n, (i + 1) / n
            cx, cy = ax + (bx - ax) * (s0 + s1) / 2, ay + (by - ay) * (s0 + s1) / 2
            ln = math.hypot(bx - ax, by - ay) / n
            horiz = abs(by - ay) < 1e-6
            b(f"{path}/mesh{k}", (cx, cy, h / 2 + 0.1), (ln - 0.06, 0.02, h - 0.2) if horiz else (0.02, ln - 0.06, h - 0.2), "mesh")
            px, py = ax + (bx - ax) * s0, ay + (by - ay) * s0
            b(f"{path}/post{k}", (px, py, h / 2), (0.06, 0.06, h), "yellow")
            k += 1


def robot_loop(path, url, x, y, rz, amp, period, phase=0.0, ped=0.0):
    """A stock arm on an optional pedestal, joints swinging around a work pose (kinematic loop)."""
    C.xf(path)
    if ped > 0:
        C.box(f"{path}/ped", (0, 0, ped / 2), (0.5, 0.5, ped), "ped")
    r = C.reference(f"{path}/arm", C.ROOT + url)
    C.setT(r, tf(None, (0, 0, ped)))
    C.setT(C.stage.GetPrimAtPath(path), tf(rot(2, math.radians(rz)), (x, y, 0)))
    MOVING.setdefault("arms", []).append(dict(prim=r, amp=np.array(amp), period=period, phase=phase, work=None))


def build(ctx):
    """Author the plant once; returns nothing. Needs ctx.material for the extra looks."""
    global C
    C = ctx
    L = C.L
    for name, rgb, rough, metal, op in [
            ("beltdark", (0.06, 0.06, 0.065), 0.85, 0.0, 1.0), ("walkway", (0.10, 0.42, 0.78), 0.6, 0.0, 1.0),
            ("hazard_k", (0.04, 0.04, 0.04), 0.6, 0.0, 1.0), ("beltline_w", (0.85, 0.85, 0.85), 0.6, 0.0, 1.0),
            ("kltblue", (0.08, 0.28, 0.75), 0.5, 0.0, 1.0), ("kltgray", (0.45, 0.47, 0.5), 0.5, 0.0, 1.0),
            ("bodywhite", (0.86, 0.87, 0.88), 0.3, 0.4, 1.0), ("bodyred", (0.62, 0.05, 0.05), 0.25, 0.5, 1.0),
            ("mesh", (0.15, 0.16, 0.18), 0.6, 0.3, 0.45), ("bench", (0.55, 0.58, 0.6), 0.5, 0.3, 1.0),
            ("rollers", (0.35, 0.36, 0.38), 0.4, 0.7, 1.0), ("spark", (1.0, 0.65, 0.15), 0.3, 0.0, 1.0)]:
        C.material(name, rgb, rough, metal, opacity=op)
    b = C.box
    if L.get("WS30"):
        return build_ws30()
    bx0, bx1 = L["BELT_X"]
    hw = L["BELT_HW"]
    # the WS30 floor belt: dark, flush with the floor, hazard strips, blue walkways both sides
    b("/World/Line/belt", ((bx0 + bx1) / 2, 0, 0.006), (bx1 - bx0, 2 * hw, 0.012), "beltdark")
    for sgn in (-1, 1):
        b(f"/World/Line/edge{sgn + 1}", ((bx0 + bx1) / 2, sgn * (hw + 0.06), 0.015), (bx1 - bx0, 0.12, 0.03), "yellow")
        for i in range(int((bx1 - bx0) / 0.3)):
            b(f"/World/Line/stripe{sgn + 1}_{i}", (bx0 + 0.15 + i * 0.3, sgn * (hw + 0.06), 0.031), (0.12, 0.125, 0.004), "hazard_k",
              rz=35)
        b(f"/World/Line/walk{sgn + 1}", ((bx0 + bx1) / 2 + 1.2, sgn * (hw + 0.72), 0.02), (bx1 - bx0 + 2.4, 1.2, 0.04), "walkway")
    MOVING["lines"] = []
    for i in range(4):                                  # parent Xform moves, the scaled box stays as authored
        MOVING["lines"].append(C.xf(f"/World/Line/mark{i}"))
        b(f"/World/Line/mark{i}/bar", (0, 0, 0.0125), (0.05, 2 * hw - 0.1, 0.002), "beltline_w")
    # door trolleys: ours (MOVING["trolley"]) plus the next one coming in
    trolley("/World/Trolley", False, "door")
    trolley("/World/TrolleyNext", True, "door")
    MOVING["trolleys"] = [("/World/Trolley", 0.0), ("/World/TrolleyNext", -3.4)]
    # body line behind the cell: roller conveyor, two car bodies on skids creeping along
    y_body = -7.0
    b("/World/BodyLine/bed", (0, y_body, 0.12), (22, 1.5, 0.24), "dark")
    for i in range(44):
        C.cyl(f"/World/BodyLine/roller{i}", (-10.75 + i * 0.5, y_body, 0.26), 0.05, 1.3, "rollers", axis="Y")
    for k, paint in enumerate(("bodywhite", "bodyred")):
        car_body(f"/World/Body{k}", paint)
    MOVING["bodies"] = [("/World/Body0", -6.0, y_body), ("/World/Body1", 1.5, y_body)]
    # fenced weld cell with two KR210 at a body on a skid
    fence("/World/WeldCell/fence", -9.5, 3.6, -3.0, 9.0)
    car_body("/World/WeldCell/body", "bodywhite")
    C.setT(C.stage.GetPrimAtPath("/World/WeldCell/body"), tf(None, (-6.2, 6.3, 0.3)))
    b("/World/WeldCell/stand", (-6.2, 6.3, 0.15), (4.8, 1.6, 0.3), "dark")
    robot_loop("/World/WeldCell/kr0", KR210, -6.0, 4.3, 90, [0.35, 0.18, 0.15, 0.6, 0.4, 0.8], 5.0, 0.0)
    robot_loop("/World/WeldCell/kr1", KR210, -6.6, 8.3, -90, [0.35, 0.18, 0.15, 0.6, 0.4, 0.8], 5.0, 1.7)
    MOVING["sparks"] = [C.cyl(f"/World/WeldCell/spark{k}", (0, 0, 0), 0.02, 0.02, "spark") for k in range(2)]
    # a cobot door station further down, parts racks, KLT carts, workbenches, calibration board
    C.xf("/World/Station3")
    b("/World/Station3/table", (10.5, 0.0, 0.4), (1.4, 1.4, 0.8), "bench")
    door_geom("/World/Station3/door", "door")
    C.setT(C.stage.GetPrimAtPath("/World/Station3/door"), tf(rot(2, math.pi / 2), (10.5, 0, 0.8)))
    robot_loop("/World/Station3/crx", CRX, 10.6, -1.3, 90, [0.5, 0.25, 0.3, 0.5, 0.5, 1.0], 4.0, 0.5, ped=0.7)
    robot_loop("/World/Station3/crx2", CRX, 10.4, 1.3, -90, [0.5, 0.25, 0.3, 0.5, 0.5, 1.0], 4.6, 2.0, ped=0.7)
    for k, (x, y, rz) in enumerate([(-2.0, 4.8, 0), (0.2, 4.8, 0), (2.4, 4.8, 0), (12.5, 4.0, 90), (-11.5, -3.0, 90)]):
        rack(f"/World/Rack{k}", x, y, rz)
    for k, (x, y, rz) in enumerate([(5.6, -3.2, 10), (6.7, -3.4, -5), (-3.6, -3.1, 0), (9.0, 2.9, 80)]):
        klt_cart(f"/World/KltCart{k}", x, y, rz)
    for k, (x, y) in enumerate([(5.0, 4.6), (7.0, 4.6)]):
        b(f"/World/Bench{k}/top", (x, y, 0.88), (1.6, 0.75, 0.05), "bench")
        for j, (dx, dy) in enumerate([(-0.75, -0.32), (0.75, -0.32), (-0.75, 0.32), (0.75, 0.32)]):
            b(f"/World/Bench{k}/leg{j}", (x + dx, y + dy, 0.44), (0.05, 0.05, 0.88), "steel")
        b(f"/World/Bench{k}/box", (x - 0.3, y, 0.98), (0.4, 0.3, 0.15), "kltblue")
    C.xf("/World/Checker")
    b("/World/Checker/stand", (3.2, 3.6, 0.8), (0.05, 0.05, 1.6), "steel")
    for i in range(6):
        for j in range(8):
            b(f"/World/Checker/c{i}_{j}", (3.2, 3.6 - 0.28 + j * 0.08, 1.2 + i * 0.08), (0.01, 0.08, 0.08),
              "hazard_k" if (i + j) % 2 else "beltline_w")
    # AGV shuttling along the aisle behind the shelves
    MOVING["agv"] = C.reference("/World/AGV", C.ROOT + AMR)
    MOVING["agv_lane"] = (-4.0, 10.0, 3.3)
    MOVING["body_span"] = (-11.0, 11.0)
    for _ in range(40):
        C.app.update()
    rig_arms()


def build_ws30():
    """Background around the real WS30 station (plant frame): nothing inside x 0..19.7, y 0..6.6 except what the
    station already has. A body line on the OUT side, the weld cell and racks on the IN side, a cobot station and
    KLT carts downstream, the AGV aisle behind the IN desks."""
    b = C.box
    MOVING["lines"], MOVING["trolleys"] = [], []
    y_body = -3.4
    b("/World/BodyLine/bed", (10.0, y_body, 0.12), (24, 1.5, 0.24), "dark")
    for i in range(48):
        C.cyl(f"/World/BodyLine/roller{i}", (-1.75 + i * 0.5, y_body, 0.26), 0.05, 1.3, "rollers", axis="Y")
    for k, paint in enumerate(("bodywhite", "bodyred")):
        car_body(f"/World/Body{k}", paint)
    MOVING["bodies"] = [("/World/Body0", 3.0, y_body), ("/World/Body1", 12.0, y_body)]
    MOVING["body_span"] = (-2.0, 22.0)
    fence("/World/WeldCell/fence", 2.0, 8.0, 8.5, 13.4)
    car_body("/World/WeldCell/body", "bodywhite")
    C.setT(C.stage.GetPrimAtPath("/World/WeldCell/body"), tf(None, (5.3, 10.7, 0.3)))
    b("/World/WeldCell/stand", (5.3, 10.7, 0.15), (4.8, 1.6, 0.3), "dark")
    robot_loop("/World/WeldCell/kr0", KR210, 5.5, 8.7, 90, [0.35, 0.18, 0.15, 0.6, 0.4, 0.8], 5.0, 0.0)
    robot_loop("/World/WeldCell/kr1", KR210, 4.9, 12.7, -90, [0.35, 0.18, 0.15, 0.6, 0.4, 0.8], 5.0, 1.7)
    MOVING["sparks"] = [C.cyl(f"/World/WeldCell/spark{k}", (0, 0, 0), 0.02, 0.02, "spark") for k in range(2)]
    # the neighbouring station: a CRX on a pedestal moving a part between two trays on its table (7 s cycle)
    C.xf("/World/Station3")
    b("/World/Station3/table", (24.0, 3.05, 0.40), (1.2, 0.7, 0.8), "bench")
    for k, x in enumerate(PP_X):
        b(f"/World/Station3/tray{k}", (x, 3.05, 0.81), (0.30, 0.25, 0.02), "dark")
    MOVING["part"] = C.xf("/World/Station3/part")
    b("/World/Station3/part/box", (0, 0, 0.04), (0.10, 0.08, 0.08), "orange")
    b("/World/Station3/ped", (24.0, 2.25, 0.35), (0.4, 0.4, 0.7), "ped")
    MOVING["pp"] = C.reference("/World/Station3/crx", C.ROOT + CRX)
    C.setT(MOVING["pp"], tf(rot(2, math.pi / 2), (24.0, 2.25, 0.7)))
    for k, (x, y, rz) in enumerate([(10.0, 8.6, 0), (12.2, 8.6, 0), (14.4, 8.6, 0), (19.0, 8.6, 0), (26.5, 6.0, 90)]):
        rack(f"/World/Rack{k}", x, y, rz)
    for k, (x, y, rz) in enumerate([(26.2, 0.6, 10), (26.4, 5.6, -5)]):   # kit carts parked well away from the cell
        klt_cart(f"/World/KltCart{k}", x, y, rz)
    # AMR towing nothing: it carries a kit cart deck on its back across the far aisle
    MOVING["agv"] = C.xf("/World/AGVrig")
    C.reference("/World/AGVrig/amr", C.ROOT + AMR)
    b("/World/AGVrig/deck", (0, 0, 0.42), (1.1, 0.65, 0.04), "steel")
    for j in range(4):
        b(f"/World/AGVrig/bin{j}", (-0.32 + (j % 2) * 0.62, -0.15 + (j // 2) * 0.3, 0.52), (0.36, 0.26, 0.14), "kltblue")
    MOVING["agv_lane"] = (9.0, 23.0, 7.3)
    for _ in range(40):
        C.app.update()
    rig_arms()


PP_X = (23.70, 24.30)          # pick and place trays on the neighbouring table (x), at y 3.05


def ik_world(ch, prim, p, a, seeds):
    """IK for a world point and tool axis on a rigged background arm, or None."""
    Tr = np.array(C.world_of(prim))
    Ri = Tr[:3, :3].T
    pl, al = Ri @ (np.asarray(p, float) - Tr[:3, 3]), Ri @ np.asarray(a, float)
    for s0 in seeds:
        q, ok = ch.ik(pl, al, np.clip(np.asarray(s0, float), ch.lo, ch.hi))
        if ok:
            return q
    return None


def rig_arms():
    MOVING["rigs"] = []
    for a in MOVING.get("arms", []):
        try:
            ch = C.rig.Chain(C.stage, str(a["prim"].GetPath()))
            # weld loops swing around a reach-towards-the-body pose instead of the asset's straight-up zero
            Tw = np.array(C.world_of(a["prim"]))
            fwd = Tw[:3, 0] if abs(Tw[2, 0]) < 0.9 else Tw[:3, 1]
            tgt = Tw[:3, 3] + fwd * 1.6 + np.array([0, 0, 0.9])
            a["work"] = ik_world(ch, a["prim"], tgt, fwd, [np.zeros(len(ch.dofs)), np.radians([0, 30, -30, 0, 60, 0])[:len(ch.dofs)]])
            MOVING["rigs"].append((ch, a, a["work"] if a["work"] is not None else np.zeros(len(ch.dofs))))
        except Exception as e:  # noqa: BLE001 - a background arm that will not rig just stays still
            print(f"[plant] {a['prim'].GetPath()} not rigged: {e!r}")
    MOVING["pp_keys"] = None
    if "pp" in MOVING:
        try:
            ch = C.rig.Chain(C.stage, str(MOVING["pp"].GetPath()))
            down = (0, 0, -1.0)
            seeds = [np.radians(v) for v in ([0, 20, -30, 0, -60, 0], [30, 20, -30, 0, -60, 0], [-30, 20, -30, 0, -60, 0],
                                             [0, 40, -10, 0, -80, 0])]
            pose = lambda p: ik_world(ch, MOVING["pp"], p, down, seeds)  # noqa: E731
            home = pose((24.0, 2.80, 1.30))
            ks = {"home": home}
            for k, x in enumerate(PP_X):
                ks[f"up{k}"], ks[f"at{k}"] = pose((x, 3.05, 1.10)), pose((x, 3.05, 0.92))
            if any(v is None for v in ks.values()):
                raise RuntimeError("no IK for the pick and place poses")
            for k, v in ks.items():                       # same joint branch as home: no 360 degree wrist spins
                w = home + (v - home + np.pi) % (2 * np.pi) - np.pi
                ks[k] = np.where((w >= ch.lo) & (w <= ch.hi), w, v)
            MOVING["pp_keys"], MOVING["pp_ch"] = ks, ch
            print("[plant] neighbouring CRX pick and place ready")
        except Exception as e:  # noqa: BLE001 - per the brief: if it cannot work believably, it is removed
            print(f"[plant] neighbouring CRX removed: {e!r}")
            for pth in ("/World/Station3",):
                C.stage.GetPrimAtPath(pth).SetActive(False)


PP_SEQ = [(0.0, "home"), (1.0, "up{a}"), (1.6, "at{a}"), (1.9, "at{a}"), (2.5, "up{a}"), (4.0, "up{b}"), (4.6, "at{b}"),
          (4.9, "at{b}"), (5.5, "up{b}"), (7.0, "home")]


def pick_place(t):
    """CRX joint vector and the part pose for time t: left to right, then back, one 7 s cycle each way."""
    ks, ch = MOVING["pp_keys"], MOVING["pp_ch"]
    n, u = int(t // 7.0), t % 7.0
    a, b_ = (0, 1) if n % 2 == 0 else (1, 0)
    seq = [(tt, ks[name.format(a=a, b=b_)]) for tt, name in PP_SEQ]
    for (ta, qa), (tb, qb) in zip(seq[:-1], seq[1:]):
        if u <= tb:
            q = qa + (qb - qa) * minjerk((u - ta) / max(tb - ta, 1e-6))
            break
    T = ch.write(q)
    if 1.9 <= u <= 4.9:                              # carried: the part hangs under the tool tip
        tip = np.array(C.world_of(MOVING["pp"])) @ T[ch.tool]
        p = tip[:3, :3] @ ch.tip_l + tip[:3, 3] - np.array([0, 0, 0.09])
    else:
        p = np.array([PP_X[a] if u < 1.9 else PP_X[b_], 3.05, 0.82])
    C.setT(MOVING["part"], tf(None, p))


WORK = np.radians([0, -35, 30, 0, 45, 0])       # a reach-forward pose; loops swing around it


def animate(t):
    """Pose the moving background for time t."""
    L = C.L
    bx0, bx1 = L["BELT_X"]
    for i, p in enumerate(MOVING["lines"]):            # belt marks move with the belt (2 m pitch)
        x = bx0 + ((i * 2.0 + L["BELT"] * t) % (bx1 - bx0))
        C.setT(p, tf(None, (x, 0, 0)))
    b0, b1 = MOVING["body_span"]
    for path, x0 in MOVING["trolleys"]:
        C.setT(C.stage.GetPrimAtPath(path), tf(None, (L["X1"] + x0 + L["BELT"] * t, 0, 0)))
    for path, x0, y in MOVING["bodies"]:
        x = b0 + (x0 - b0 + 0.12 * t) % (b1 - b0)
        C.setT(C.stage.GetPrimAtPath(path), tf(None, (x, y, 0.32)))
    if MOVING.get("pp_keys"):
        pick_place(t)
    xa0, xa1, ya = MOVING["agv_lane"]
    u = (t / 14.0) % 2
    xa = xa0 + (xa1 - xa0) * (u if u < 1 else 2 - u)
    C.setT(MOVING["agv"], tf(rot(2, 0.0 if u < 1 else math.pi), (xa, ya, 0.0)))
    for k, (ch, a, q0) in enumerate(MOVING["rigs"]):
        n = len(ch.dofs)
        w = 2 * math.pi / a["period"]
        q = q0 if a["work"] is not None else (WORK[:n] if n >= 6 else np.zeros(n))
        q = np.clip(np.r_[q, np.zeros(max(0, n - len(q)))][:n] + 0.6 * a["amp"][:n] * np.sin(w * t + a["phase"] + np.arange(n) * 0.7),
                    ch.lo, ch.hi)
        T = ch.write(q)
        if k < 2:                                     # weld sparks at the two KR210 flanges, flickering
            Tw = np.array(C.world_of(a["prim"])) @ T[ch.tool]
            on = math.sin(w * 3 * t + k) > 0.2
            C.setT(MOVING["sparks"][k], tf(None, Tw[:3, 3] + Tw[:3, :3] @ ch.axis_l * 0.25) if on else tf(None, (0, 0, -5)))
