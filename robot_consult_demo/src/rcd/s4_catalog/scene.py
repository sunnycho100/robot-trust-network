"""Upstream inputs (real out/ or local fixture), task-group construction and robot base placement.

Base placement is an inverse-reachability style search: every candidate floor position (5 cm grid) and pedestal
height is scored by the largest shoulder-to-flange distance over all task poses (task points offset by the tool
length along the surface normal, sampled over the conveyor-tracking window, plus the robot-side presenter/tray).
Positions inside the observed worker zone (s1 zones, grown by pedestal radius + clearance) or behind the panel
plane (the carrier/conveyor path) are excluded. The position with the smallest maximum distance wins; that distance
is the group's required reach.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from rcd.common.io import ROOT, load_json

from . import process as P

FIXTURES = Path(__file__).resolve().parent / "fixtures"
ROBOT_ALLOC = ("robot", "human_robot")


def load_upstream(cfg: dict, stage: str, fn: str, prov: dict, fixture_dir: Path = FIXTURES):
    p = ROOT / cfg["out_root"] / stage / fn
    if p.exists():
        prov[f"{stage}/{fn}"] = "real"
        return load_json(p)
    f = fixture_dir / stage / fn
    if f.exists():
        prov[f"{stage}/{fn}"] = "fixture"
        return load_json(f)
    prov[f"{stage}/{fn}"] = "missing"
    return None


# ------------------------------------------------------------------ polygon helpers
def _point_in_poly(pt, poly) -> bool:
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1:
            inside = not inside
    return inside


def _dist_to_poly(pt, poly) -> float:
    """0 inside, else distance to the boundary."""
    if _point_in_poly(pt, poly):
        return 0.0
    p = np.asarray(pt, float)
    best = np.inf
    for i in range(len(poly)):
        a, b = np.asarray(poly[i], float), np.asarray(poly[(i + 1) % len(poly)], float)
        ab = b - a
        t = np.clip(np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-12), 0, 1)
        best = min(best, float(np.linalg.norm(p - (a + t * ab))))
    return best


def _cluster(points: np.ndarray, radius: float = 0.04) -> np.ndarray:
    pts = [np.asarray(p, float) for p in points]
    cl: list[list[np.ndarray]] = []
    for p in pts:
        for c in cl:
            if np.linalg.norm(np.mean(c, 0) - p) < radius:
                c.append(p)
                break
        else:
            cl.append([p])
    return np.array([np.mean(c, 0) for c in cl])


def _flat_points(v) -> list[list[float]]:
    """positions_m may be [x,y,z], [[x,y,z],...] or nested per cycle."""
    a = np.asarray(v, dtype=object)
    out = []

    def rec(x):
        if isinstance(x, (list, tuple, np.ndarray)) and len(x) == 3 and all(isinstance(t, (int, float, np.floating)) for t in x):
            out.append([float(t) for t in x])
        elif isinstance(x, (list, tuple, np.ndarray)):
            for y in x:
                rec(y)
        elif isinstance(x, dict):
            for k in ("p", "pos", "position", "xyz", "positions_m"):
                if k in x:
                    rec(x[k])
                    break
    rec(a.tolist() if hasattr(a, "tolist") else v)
    return out


# ------------------------------------------------------------------ station geometry from s1 scene graph
def _zone(zs, key):
    for z in zs or []:
        if key in z.get("name", "").lower() and len(z.get("polygon_xy", [])) >= 3:
            return [[float(a), float(b)] for a, b in z["polygon_xy"]]
    return None


def station_geometry(sg: dict | None, station: str) -> dict:
    """Contact clusters (median cycle, with dwell), zones, panel normal, conveyor direction/speed (fallbacks if absent).

    s1 contact points live in a conveyor-attached frame that coincides with the world at the cycle's t_ref_s; the
    robot program is centred on that instant (see moving_offsets)."""
    g = {"station": station, "source": "default (no s1 geometry for this station)"}
    st = next((s for s in (sg or {}).get("stations", []) if s.get("station") == station), None)
    speeds, dirs, cyc = [], [], []
    zone = conv_zone = line_zone = None
    supply = []
    if st:
        zone, conv_zone, line_zone = _zone(st.get("zones"), "work"), _zone(st.get("zones"), "conveyor"), _zone(st.get("zones"), "line")
        for c in st.get("cycles", []):
            conv = c.get("conveyor", {}) or {}
            if conv.get("speed_mm_s") is not None:
                speeds.append(float(conv["speed_mm_s"]))
            for k in ("direction_world", "direction_xy", "direction"):
                if isinstance(conv.get(k), (list, tuple)) and len(conv[k]) >= 2:
                    dirs.append(np.asarray(conv[k][:2], float))
                    break
            pts, dw = [], []
            for o in c.get("objects", []):
                nm = o.get("name", "").lower()
                if "contact" in nm or "workpiece" in nm:
                    p = _flat_points(o.get("positions_m", []))
                    d = o.get("dwell_s") or [1.0] * len(p)
                    pts += p
                    dw += list(d)[:len(p)] + [1.0] * max(0, len(p) - len(d))
                elif "supply" in nm or "line_side" in nm:
                    supply += _flat_points(o.get("positions_m", []))
            if pts:
                cyc.append((c.get("cycle_id"), np.asarray(pts, float), np.asarray(dw, float)))
    if cyc:
        cyc.sort(key=lambda x: len(x[1]))
        cid, pts, dw = cyc[len(cyc) // 2]  # representative cycle = median number of contact clusters
        order = np.argsort(-dw)
        pts, dw = pts[order], dw[order]
        g["source"] = f"s1 scene_graph workpiece_contact, cycle {cid} (median cluster count), sorted by dwell"
    else:
        pts = np.array([[-0.075, 0.0, 1.125], [0.075, 0.0, 1.125], [-0.075, 0.0, 0.975], [0.075, 0.0, 0.975]])
        dw = np.ones(len(pts))
    c = pts[: max(1, min(6, len(pts)))].mean(0)
    if zone is None:
        zone = [[c[0] - 0.4, c[1] + 0.30], [c[0] + 0.4, c[1] + 0.30], [c[0] + 0.4, c[1] + 1.0], [c[0] - 0.4, c[1] + 1.0]]
        g["worker_zone_source"] = "default 0.8 x 0.7 m in front of the panel"
    else:
        g["worker_zone_source"] = "s1 zones"
    if dirs:
        d2 = np.mean(dirs, 0)
        g["conveyor_dir_source"] = "s1 conveyor direction"
    else:
        d2 = np.array([1.0, 0.0])
        g["conveyor_dir_source"] = "default +x"
    d = np.array([d2[0], d2[1], 0.0]) / (np.linalg.norm(d2) + 1e-12)
    perp = np.array([-d[1], d[0], 0.0])
    zc = np.mean(np.asarray(zone), 0)
    side = (st or {}).get("operator_side")
    if side in ("in", "out") and abs(perp[1]) > 0.5:  # s1 world: +y points towards the IN cameras
        sgn = 1.0 if side == "in" else -1.0
        sgn *= np.sign(perp[1])
        g["normal_source"] = f"s1 operator_side='{side}' (+y towards IN cameras), perpendicular to the conveyor"
    else:
        sgn = 1.0 if np.dot(zc - c[:2], perp[:2]) >= 0 else -1.0
        g["normal_source"] = "towards the worker-zone centroid, perpendicular to the conveyor"
    n = sgn * perp
    v = float(np.median(speeds)) if speeds else 40.0
    g.update({
        "points_m": pts.round(4).tolist(), "dwell_s": np.asarray(dw).round(2).tolist(),
        "centroid_m": c.round(4).tolist(), "normal": n.round(4).tolist(),
        "conveyor_dir": d.round(4).tolist(), "conveyor_speed_mm_s": round(v, 2),
        "conveyor_speed_source": "s1 (median over cycles)" if speeds else "default 40 mm/s",
        "worker_zone_xy": zone, "conveyor_zone_xy": conv_zone, "line_side_zone_xy": line_zone,
        "line_side_supply_m": np.asarray(supply).mean(0).round(3).tolist() if supply else None,
    })
    return g


# ------------------------------------------------------------------ task groups from s3
def _num(tasks, keys):
    vals = [float(t.get("requirements", {}).get(k)) for t in tasks for k in keys
            if isinstance((t.get("requirements") or {}).get(k), (int, float))]
    return max(vals) if vals else None


def build_groups(au: dict, sg: dict | None, cfg: dict) -> list[dict]:
    tasks_by_key = {t.get("element_key") or f"{t['station']}:{t['code']}": t for t in au["tasks"]}
    raw = []
    if au.get("groups"):
        for g in au["groups"]:
            keys = [k for k in g["element_keys"] if not k.endswith(":UNOBS")]
            raw.append((g["station"], g["category"], [tasks_by_key[k] for k in keys if k in tasks_by_key],
                        g.get("requirements") or {}, g.get("allocations") or {}))
    else:
        by: dict = {}
        for t in au["tasks"]:
            if t.get("allocation") in ROBOT_ALLOC and t.get("code") != "UNOBS":
                by.setdefault((t["station"], t.get("category") or "robot_task"), []).append(t)
        for (st, cat), ts in sorted(by.items()):
            raw.append((st, cat, ts, {}, {}))
    groups = []
    for st, cat, tasks, greq, allocs in raw:
        ptype = P.process_type(cat)
        geo = station_geometry(sg, st)
        codes = sorted({t["code"] for t in tasks})
        pts = np.asarray(geo["points_m"], float)
        if ptype == "screwdriving":
            n_t = _num(tasks, ["fasteners_per_cycle", "n_fasteners", "n_screws"]) or min(len(pts), 4)
        else:
            occ = _num([t for t in tasks if t["code"] in ("GET_PART", "INSPECT")] or tasks,
                       ["n_parts", "n_targets", "occurrences_per_cycle"])
            n_t = max(1, int(round(occ))) if occ else 1
        n_t = int(round(n_t))
        if len(pts) >= n_t:
            targets = pts[:n_t]                      # the n longest hand dwells of the representative cycle
        else:
            targets = np.vstack([pts, pts[np.arange(n_t - len(pts)) % len(pts)]])
        if ptype == "pick_place":
            geo["supply_note"] = ("line-side supply (s1) is replaced by a robot-side part tray at the pedestal "
                                  "(layout change); place targets = longest-dwell contact points")
        tool = P.TOOLS[ptype]
        pay = greq.get("payload_kg")
        if pay is None:
            pm = _num(tasks, ["part_mass_kg", "mass_kg"]) or {"screwdriving": 0.005, "inspection": 0.0, "pick_place": 0.9}[ptype]
            pay = tool["mass_kg"] + P.ADAPTER_KG + pm
            pay_src = f"{tool['name']} + adapter + part {pm} kg (s4 default)"
        else:
            pay_src = f"s3 group requirements (tool '{greq.get('tool')}' + part)"
        prec = greq.get("precision_mm", _num(tasks, ["precision_mm"]))
        tracking = greq.get("conveyor_tracking")
        if tracking is None:
            tracking = bool(cfg["line"].get("paced", True)) and geo["conveyor_speed_mm_s"] > 0
        if greq.get("conveyor_speed_mm_s"):
            geo["conveyor_speed_mm_s"] = round(float(greq["conveyor_speed_mm_s"]), 2)
            geo["conveyor_speed_source"] = "s3 group requirements (s1 median)"
        ops = P.program(ptype, len(targets), codes)
        groups.append({
            "group_id": f"{st}:{cat}", "station": st, "category": cat, "process_type": ptype, "codes": codes,
            "element_keys": [f"{st}:{c}" for c in codes],
            "allocations": allocs or {f"{st}:{t['code']}": t["allocation"] for t in tasks},
            "tool": tool, "payload_kg": round(float(pay), 4), "payload_source": pay_src,
            "precision_mm": float(prec) if prec is not None else 0.5,
            "precision_source": "s3 requirements" if prec is not None else "assumption +/-0.5 mm",
            "conveyor_tracking": bool(tracking), "targets_m": np.asarray(targets).round(4).tolist(),
            "geometry": geo, "program_est_s": round(P.analytic_cycle_estimate(ops), 2), "n_targets": int(len(targets)),
            "s3_requirements": {k: v for k, v in greq.items() if k != "work_points"},
        })
    return groups


# ------------------------------------------------------------------ poses
def task_poses(group: dict, base_xyz=None, yaw=None, window_offsets=(-0.5, 0.0, 0.5)) -> dict:
    """World TCP targets {name: (pos, approach_axis)}; *_moving names move with the conveyor.

    approach_axis = direction the tool points (tool z axis). For panel targets it is -normal (into the panel);
    for the tray it is -z (down). The flange point is pos - L * axis.
    """
    geo = group["geometry"]
    n = np.asarray(geo["normal"], float)
    T = {}
    ptype = group["process_type"]
    for i, p in enumerate(np.asarray(group["targets_m"], float)):
        if ptype == "inspection":
            T[f"view_{i}"] = (p + n * P.VIEW_STANDOFF_M, -n, True)
        else:
            T[f"tgt_{i}"] = (p.copy(), -n, True)
            T[f"app_{i}"] = (p + n * P.APPROACH_OFFSET_M, -n, True)
    c = np.asarray(group["targets_m"], float).mean(0)
    T["home"] = (c + n * (P.APPROACH_OFFSET_M + 0.15), -n, True)
    if base_xyz is not None:
        tray = tray_position(group, np.asarray(base_xyz, float))
        T["tray"] = (tray, np.array([0, 0, -1.0]), False)
        T["tray_above"] = (tray + np.array([0, 0, P.FEEDER_LIFT_M]), np.array([0, 0, -1.0]), False)
    return T


def tray_position(group: dict, base: np.ndarray) -> np.ndarray:
    """Screw presenter / part tray beside the pedestal, offset along the conveyor on the side away from the worker
    zone (keeps J1 travel between tray and panel small and the tray off the belt and out of the operator's way)."""
    geo = group["geometry"]
    zc = np.mean(np.asarray(geo["worker_zone_xy"]), 0)
    d = np.asarray(geo["conveyor_dir"], float)[:2]
    sgn = 1.0 if np.dot(base[:2] - zc, d) >= 0 else -1.0
    u = sgn * d / (np.linalg.norm(d) + 1e-9)
    xy = base[:2] + P.TRAY_RADIUS_M * u
    z = base[2] + P.SHOULDER_OFFSET_M - P.TRAY_BELOW_SHOULDER_M
    return np.array([xy[0], xy[1], z])


def tracking_window_m(group: dict) -> float:
    """Carrier travel during one robot cycle (m)."""
    return group["geometry"]["conveyor_speed_mm_s"] / 1000.0 * group["program_est_s"]


def moving_offsets(group: dict) -> np.ndarray:
    """Conveyor displacement samples over the tracking window centred on the cycle midpoint (m), relative to the
    robot base. On a linear track (7th axis synchronised to the conveyor) the base moves with the carrier -> 0."""
    geo = group["geometry"]
    if group.get("mounting") == "linear_track":
        return np.zeros((1, 3))
    W = tracking_window_m(group)
    d = np.asarray(geo["conveyor_dir"], float)
    return np.array([s * W * d for s in (-0.5, -0.25, 0.0, 0.25, 0.5)])


def flange_points(group: dict, base_xyz) -> np.ndarray:
    L = group["tool"]["tcp_offset_m"]
    pts = []
    offs = moving_offsets(group)
    for name, (p, ax, moving) in task_poses(group, base_xyz).items():
        f = p - L * ax
        if moving:
            pts += [f + o for o in offs]
        else:
            pts.append(f)
    return np.array(pts)


def place_base(group: dict) -> dict:
    geo = group["geometry"]
    c = np.asarray(geo["centroid_m"], float)
    n = np.asarray(geo["normal"], float)
    zone = geo["worker_zone_xy"]
    conv, line = geo.get("conveyor_zone_xy"), geo.get("line_side_zone_xy")
    L = group["tool"]["tcp_offset_m"]
    offs = moving_offsets(group)
    moving = []
    for name, (p, ax, mv) in task_poses(group).items():
        f = p - L * ax
        moving += [f + o for o in offs]
    moving = np.array(moving)
    best = None
    xs = np.arange(c[0] - 2.5, c[0] + 2.5001, 0.05)
    ys = np.arange(c[1] - 2.5, c[1] + 2.5001, 0.05)
    n_feasible = 0
    for x in xs:
        for y in ys:
            if np.dot(np.array([x - c[0], y - c[1]]), n[:2]) < P.DOOR_CLEARANCE_M:
                continue  # behind / too close to the panel plane = carrier path
            if conv is not None and _dist_to_poly((x, y), conv) < P.BASE_RADIUS_M + P.ZONE_CLEARANCE_M:
                continue  # on the conveyor belt (s1 conveyor zone)
            if line is not None and _dist_to_poly((x, y), line) < P.BASE_RADIUS_M + P.ZONE_CLEARANCE_M:
                continue  # blocks the line-side supply (s1 line_side zone)
            if _dist_to_poly((x, y), zone) < P.BASE_RADIUS_M + P.WORKER_CLEARANCE_M:
                continue  # inside the observed worker zone
            n_feasible += 1
            for h in P.PEDESTAL_HEIGHTS_M:
                sh = np.array([x, y, h + P.SHOULDER_OFFSET_M])
                dm = np.linalg.norm(moving - sh, axis=1)
                if dm.min() < P.MIN_REACH_M:
                    continue
                tray = tray_position(group, np.array([x, y, h]))
                dt = [np.linalg.norm(tray + np.array([0, 0, z]) + np.array([0, 0, L]) - sh) for z in (0.0, P.FEEDER_LIFT_M)]
                score = max(dm.max(), max(dt))
                if best is None or score < best[0] - 1e-9:
                    best = (score, x, y, h)
    if best is None:
        raise RuntimeError(f"{group['group_id']}: no feasible base position")
    score, x, y, h = best
    yaw = float(np.arctan2(c[1] - y, c[0] - x))
    return {"base_xyz": [round(float(x), 3), round(float(y), 3), round(float(h), 3)], "yaw_rad": round(yaw, 4),
            "pedestal_height_m": h, "shoulder_xyz": [round(float(x), 3), round(float(y), 3), round(h + P.SHOULDER_OFFSET_M, 3)],
            "required_reach_mm": round(float(score) * 1000, 1), "grid_step_m": 0.05,
            "feasible_floor_cells": n_feasible,
            "mounting": group.get("mounting", "fixed_floor"),
            "tracking_window_m": round(tracking_window_m(group), 3),
            "window_relative_to_base_m": round(float(np.linalg.norm(offs[-1] - offs[0])), 3),
            "method": "inverse-reachability grid search: min over feasible floor cells x pedestal heights of the max "
                      "shoulder-to-flange distance (task poses over the tracking window + tray)"}
