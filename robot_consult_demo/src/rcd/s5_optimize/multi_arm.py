"""Second arm at a station: where its base can go and how far it must reach (s4 inverse-reachability, re-run).

s4 places one base per task group; all groups of a station end up on (nearly) the same floor cell, so a second arm
cannot stand there. For every task group g this re-runs the s4 grid search with one more exclusion - the cell must be
at least ARM_SEPARATION_M from the station's first base - and returns the reach the second arm then needs for g.
A model can serve g as the station's second arm only if its usable reach (reach x s4 reach_margin) covers that.

Two arms on one door also lose time to coordination (zone interlocks / shared-workspace collision avoidance):
each arm's program time is inflated by DUAL_ARM_INTERFERENCE when a station has two arms (assumption; the robot
simulation team is asked to measure it).
"""
from __future__ import annotations

import numpy as np

from rcd.s4_catalog import process as P
from rcd.s4_catalog import scene as S

ARM_SEPARATION_M = 2 * P.BASE_RADIUS_M + 0.40   # pedestal plates + 0.4 m arm-to-arm clearance (assumption)
DUAL_ARM_INTERFERENCE = 0.10                    # +10 % program time per arm when two arms share one door (assumption)


def _with_program_time(group):
    """The s4 task group in the form the s4 placement helpers expect (targets, program time; as s6a rebuilds it)."""
    geo = group["geometry"]
    return {**group, "targets_m": geo["targets_m"], "program_est_s": geo["program_est_s"],
            "mounting": group.get("mounting", "fixed_floor")}


def second_base(group, first_xy, sep=ARM_SEPARATION_M):
    """s4 place_base with the extra exclusion |cell - first base| >= sep. Returns None if no cell is feasible."""
    g = _with_program_time(group)
    geo = g["geometry"]
    c = np.asarray(geo["centroid_m"], float)
    n = np.asarray(geo["normal"], float)
    zone, conv, line = geo["worker_zone_xy"], geo.get("conveyor_zone_xy"), geo.get("line_side_zone_xy")
    L = g["tool"]["tcp_offset_m"]
    offs = S.moving_offsets(g)
    moving = np.array([p - L * ax + o for _, (p, ax, mv) in S.task_poses(g).items() for o in offs])
    f0 = np.asarray(first_xy, float)[:2]
    best, n_ok = None, 0
    for x in np.arange(c[0] - 2.5, c[0] + 2.5001, 0.05):
        for y in np.arange(c[1] - 2.5, c[1] + 2.5001, 0.05):
            if np.hypot(x - f0[0], y - f0[1]) < sep:
                continue
            if np.dot(np.array([x - c[0], y - c[1]]), n[:2]) < P.DOOR_CLEARANCE_M:
                continue
            if conv is not None and S._dist_to_poly((x, y), conv) < P.BASE_RADIUS_M + P.ZONE_CLEARANCE_M:
                continue
            if line is not None and S._dist_to_poly((x, y), line) < P.BASE_RADIUS_M + P.ZONE_CLEARANCE_M:
                continue
            if S._dist_to_poly((x, y), zone) < P.BASE_RADIUS_M + P.WORKER_CLEARANCE_M:
                continue
            n_ok += 1
            for h in P.PEDESTAL_HEIGHTS_M:
                sh = np.array([x, y, h + P.SHOULDER_OFFSET_M])
                dm = np.linalg.norm(moving - sh, axis=1)
                if dm.min() < P.MIN_REACH_M:
                    continue
                tray = S.tray_position(g, np.array([x, y, h]))
                dt = [np.linalg.norm(tray + np.array([0, 0, z + L]) - sh) for z in (0.0, P.FEEDER_LIFT_M)]
                score = max(dm.max(), max(dt))
                if best is None or score < best[0] - 1e-9:
                    best = (score, x, y, h)
    if best is None:
        return None
    s, x, y, h = best
    return {"base_xyz": [round(float(x), 3), round(float(y), 3), round(float(h), 3)],
            "required_reach_mm": round(float(s) * 1000, 1), "feasible_floor_cells": n_ok,
            "distance_to_first_base_m": round(float(np.hypot(x - f0[0], y - f0[1])), 3)}


def second_arm_table(cand, catalog, reach_margin):
    """{group_id: {...second base..., 'models_ok': [models whose usable reach covers it]}} for every task group."""
    groups = cand["task_groups"]
    first = {}
    for g in groups:   # the station's first base = the s4 base of its (first) group
        first.setdefault(g["station"], g["placement"]["base_xyz"])
    out = {}
    for g in groups:
        sb = second_base(g, first[g["station"]])
        if sb is None:
            out[g["group_id"]] = {"feasible": False, "models_ok": []}
            continue
        ok = [r["model"] for r in catalog["robots"] if r["reach_mm"] * reach_margin >= sb["required_reach_mm"] - 1e-9]
        out[g["group_id"]] = {"feasible": True, **sb, "first_base_xyz": first[g["station"]],
                              "s4_required_reach_mm": g["placement"]["required_reach_mm"], "models_ok": ok}
    return out
