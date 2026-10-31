"""s1 scene graph / 4D human -> out/s1/scene_graph.json + out/s1/human/<ST>_<cycle>.npz

Per station (stations present on disk) and per cycle (one per BVH file):
* human: Xsens BVH -> joints (T, J, 3) metres, z-up (rcd.common.bvh), registered into the s0 world frame
  by a yaw + xy translation (see `registration` in each cycle) and saved as npz.
* conveyor: belt-surface speed measured independently in every calibrated camera view of the cycle
  (metric bird's-eye view of the floor plane z=0 from the s0 calibration + phase correlation).
* zones / objects from the mocap: work zone (feet while standing at the workpiece), line side (farthest
  stop), workpiece contact points (wrist manipulation dwells), line-side supply points.

Frame note: Xsens integrates motion with foot-contact zero-velocity updates, so while the worker stands on the
moving belt the mocap frame travels with the belt (the IMUs cannot feel a constant 2-3 cm/s drift). Positions
are therefore expressed in a conveyor-attached frame that coincides with the world frame at mid-clip; the
measured belt speed/direction is given so downstream stages can convert (x_world(t) = x + v (t - t_ref)).
"""
from __future__ import annotations

import numpy as np

from rcd.common.bvh import load_bvh
from rcd.common.io import ROOT, load_json, out_dir, save_json
from rcd.s0_calib.discover import discover, rel
from rcd.s1_scene import mocap as M
from rcd.s1_scene import bev_strips as V


def _cams(cfg) -> dict:
    cal = load_json(ROOT / cfg["out_root"] / "s0" / "calibration.json", contract="calibration")
    return {(c["station"], c["side"]): {k: np.asarray(c[k], float) for k in ("K", "R", "t")} |
            {"center": np.asarray(c.get("center_world_m") or (-np.asarray(c["R"]).T @ np.asarray(c["t"])))}
            for c in cal["cameras"]}


def _r3(p) -> list[float]:
    return [round(float(x), 4) for x in p]


def run(cfg: dict) -> None:
    from rcd.common.intake import mode
    if mode(cfg, "human_source", "mocap") == "video":
        from rcd.s1_scene.generic_run import run_generic
        return run_generic(cfg)
    cams = _cams(cfg)
    found = discover(cfg)
    hdir = out_dir(cfg, "s1") / "human"
    hdir.mkdir(parents=True, exist_ok=True)
    stations = []
    for st, cycles in found.items():
        sides = [s for s in ("in", "out") if (st, s) in cams]
        xc = float(np.mean([cams[(st, s)]["center"][0] for s in sides]))

        # ---- video pass 1: belt edges per camera (median background of the clip)
        edges = []
        for c in cycles:
            for s in sides:
                v = c[f"video_{s}"]
                if v is not None:
                    e = V.belt_edges(V.background(v), cams[(st, s)], xc)
                    if e:
                        edges.append(e)
        if edges:
            y_lo = float(np.median([e["y_low"] for e in edges]))
            y_hi = float(np.median([e["y_high"] for e in edges]))
            edge_method = f"yellow edge strips detected in metric floor view (BEV) of {len(edges)} camera clips"
        else:  # fall back to the middle 40% between the two camera rows
            ys = sorted(cams[(st, s)]["center"][1] for s in sides)
            y_lo, y_hi = ys[0] + 0.3 * (ys[-1] - ys[0]), ys[0] + 0.7 * (ys[-1] - ys[0])
            edge_method = "not detected; middle 40% between camera rows"
        y_c = 0.5 * (y_lo + y_hi)
        band = (y_lo + 0.25, y_hi - 0.25)

        # ---- video pass 2: belt speed + operator-side cue per cycle and camera
        scans = {}
        for c in cycles:
            for s in sides:
                v = c[f"video_{s}"]
                if v is not None:
                    scans[(c["cycle_id"], s)] = V.scan(v, cams[(st, s)], xc, band)
        # operator side: the camera on the operator's side sees the orange Xsens straps larger / unoccluded
        orange = {s: float(np.sum([r["orange_fg_px"] or 0 for (cid, ss), r in scans.items() if ss == s]))
                  for s in sides if any(ss == s for (_, ss) in scans)}
        if len(orange) == 2:
            op_side = max(orange, key=orange.get)
            op_method = (f"camera with more moving orange suit-strap pixels (sum of per-clip medians: "
                         f"{', '.join(f'{k}={v:.0f}' for k, v in orange.items())})")
        elif len(orange) == 1:
            op_side = next(iter(orange))
            op_method = ("only one camera recorded at this station; operator assumed on that camera's side "
                         "(the cameras were placed to film the operator)")
        else:
            op_side, op_method = "in", "no video; default"
        facing = np.array([0.0, -1.0]) if op_side == "in" else np.array([0.0, 1.0])

        st_cycles, all_work, all_line, all_contacts = [], [], [], []
        for c in cycles:
            cid = c["cycle_id"]
            b = load_bvh(c["bvh"])
            J, names, fps = b["joints"], b["names"], float(b["fps"])
            an = M.analyse(J, names, fps)
            Mw = M.register(an, facing, np.array([xc, y_c]))
            Jw = M.apply(Mw, J).astype(np.float32)
            npz = hdir / f"{st}_{cid}.npz"
            np.savez_compressed(npz, joints=Jw, names=np.array(names), fps=np.float32(fps),
                                world_from_mocap=Mw.astype(np.float32))
            T = len(J)
            dur = T / fps
            t_ref = dur / 2

            # conveyor estimates
            est = {}
            for s in sides:
                r = scans.get((cid, s))
                if r and r["belt"]:
                    est[f"video_{s}_belt_bev"] = r["belt"]
            speeds = [e["speed_mm_s"] for e in est.values()]
            if speeds:
                v_mm = float(np.mean(speeds))
                dirs = {e["direction_x_sign"] for e in est.values()}
                agree = (max(speeds) - min(speeds)) / v_mm if len(speeds) > 1 else None
            else:
                v_mm, dirs, agree = None, set(), None
            # mocap diagnostic: drift of the pelvis along the line while standing at the work zone
            Pw = M.apply(Mw, an["Js"][:, names.index("Hips")])
            tt = np.arange(T) / fps
            drift = None
            if an["work"].sum() > 2 * fps:
                drift = float(np.polyfit(tt[an["work"]], Pw[an["work"], 0], 1)[0] * 1000)

            # zones (world xy)
            feet = an["feet_idx"]
            Fw = M.apply(Mw, an["Js"][:, feet].reshape(-1, 3)).reshape(T, len(feet), 3)
            work_pts = Fw[an["work"]][:, :, :2].reshape(-1, 2)
            line_pts = Fw[an["line"]][:, :, :2].reshape(-1, 2)
            all_work.append(work_pts[::6])
            if len(line_pts):
                all_line.append(line_pts[::6])
            zones = [{"name": "work_zone", "polygon_xy": M.hull_polygon(work_pts[::6]),
                      "method": "convex hull (+0.1 m) of ankle/toe xy while the pelvis is still (<0.2 m/s) within "
                                "0.6 m of the most-occupied standing spot"}]
            if len(line_pts):
                zones.append({"name": "line_side", "polygon_xy": M.hull_polygon(line_pts[::6], 0.15),
                              "method": "convex hull (+0.15 m) of feet at the stop farthest from the work zone",
                              "distance_from_work_zone_m": round(float(an["line_dist_m"]), 3)})

            # objects
            cl = M.cluster_points(an["contacts"])
            sup = M.cluster_points(an["supply"], 0.15)
            objects = [{
                "name": "workpiece_contact",
                "positions_m": [_r3(M.apply(Mw, k["p"])) for k in cl],
                "dwell_s": [round(k["dwell_s"], 2) for k in cl],
                "hands": [k["hands"] for k in cl],
                "first_t_s": [round(k["first_t_s"], 2) for k in cl],
                "method": f"wrist positions during hand dwells (<{M.HAND_STILL} m/s for >= {M.MIN_HAND_S} s) "
                          f"while standing in the work zone, clustered at {M.CONTACT_CLUSTER} m (dwell-weighted)",
                "frame": "conveyor-attached, coincident with world at t_ref_s",
            }, {
                "name": "line_side_supply",
                "positions_m": [_r3(M.apply(Mw, k["p"])) for k in sup],
                "dwell_s": [round(k["dwell_s"], 2) for k in sup],
                "method": "wrist positions during slow-hand moments at the line-side stop, clustered at 0.15 m",
            }]
            # registration checks (not used to fit): feet inside belt band, line side beyond operator-side edge
            in_belt = float(np.mean((work_pts[:, 1] > y_lo) & (work_pts[:, 1] < y_hi))) if len(work_pts) else None
            ls_y = float(np.median(line_pts[:, 1])) if len(line_pts) else None
            ls_beyond = None if ls_y is None else bool(ls_y > y_hi if op_side == "in" else ls_y < y_lo)

            vid = {"in": rel(c["video_in"]) if c["video_in"] else None,
                   "out": rel(c["video_out"]) if c["video_out"] else None,
                   "fps": float(next((r["fps"] for (ci, _), r in scans.items() if ci == cid), cfg["data"]["video_fps"]))}
            fr = [r["frames_read"] for (ci, _), r in scans.items() if ci == cid]
            if fr:
                vid["duration_s"] = round(fr[0] / vid["fps"], 3)
            if c["video_side_inferred"]:
                vid["side_inferred"] = c["video_side_inferred"]

            st_cycles.append({
                "cycle_id": cid,
                "duration_s": round(dur, 4),
                "duration_note": "mocap clip length (BVH frames / fps); video clip length in video.duration_s",
                "t_ref_s": round(t_ref, 3),
                "video": vid,
                "human": {
                    "joints_npz": rel(npz),
                    "joint_names": names,
                    "fps": fps,
                    "n_frames": T,
                    "source": f"Xsens MVN BVH {rel(c['bvh'])} via rcd.common.bvh (mm, y-up -> m, z-up), "
                              f"registered to world by yaw+translation",
                },
                "registration": {
                    "world_from_mocap": np.round(Mw, 6).tolist(),
                    "method": (f"yaw: mean facing direction (z x hip/shoulder right-vector, sign checked against "
                               f"walking direction: {an['facing_sign_check'] if an['facing_sign_check'] is None else round(an['facing_sign_check'], 2)}"
                               f" agreement) while working -> faces the conveyor from the operator side "
                               f"({op_side}); translation: dwell-weighted contact centroid -> station centre "
                               f"x={xc:.2f} (mean camera x), conveyor centreline y={y_c:.2f}"),
                    "check_work_feet_on_belt_frac": None if in_belt is None else round(in_belt, 3),
                    "check_line_side_beyond_operator_belt_edge": ls_beyond,
                },
                "conveyor": {
                    "speed_mm_s": None if v_mm is None else round(v_mm, 2),
                    "method": ("mean of independent per-camera estimates: belt surface tracked by phase correlation "
                               "of metric floor-plane (z=0) bird's-eye views ~2 s apart, homography from s0 K,R,t; "
                               "median over frame pairs with correlation response >= 0.2"),
                    "direction_world": None if len(dirs) != 1 else [float(next(iter(dirs))), 0.0, 0.0],
                    "estimates": {k: round(e["speed_mm_s"], 2) for k, e in est.items()},
                    "estimate_details": est,
                    "relative_spread": None if agree is None else round(agree, 3),
                    "mocap_pelvis_drift_mm_s": None if drift is None else round(drift, 2),
                    "mocap_note": ("mocap pelvis drift along the line while working is reported for comparison only: "
                                   "the worker stands on the moving belt and Xsens' foot-contact model makes the mocap "
                                   "frame belt-relative, so the belt speed is not observable in the mocap"),
                },
                "objects": objects,
                "zones": zones,
            })
            all_contacts += objects[0]["positions_m"]
            print(f"    {st}/{cid}: {dur:.1f}s conveyor {est and {k: round(e['speed_mm_s'], 1) for k, e in est.items()}} "
                  f"mocap drift {drift if drift is None else round(drift, 1)} mm/s | contacts {len(cl)} "
                  f"line-side {an['line_dist_m']:.2f} m beyond-edge={ls_beyond} feet-on-belt={in_belt if in_belt is None else round(in_belt, 2)}")

        zones = [
            {"name": "conveyor", "polygon_xy": [[round(xc - 3, 3), round(y_lo, 3)], [round(xc + 3, 3), round(y_lo, 3)],
                                                [round(xc + 3, 3), round(y_hi, 3)], [round(xc - 3, 3), round(y_hi, 3)]],
             "method": f"belt edges: {edge_method}; x extent = station centre +/- 3 m (assumed)"},
            {"name": "work_zone", "polygon_xy": M.hull_polygon(np.vstack(all_work)),
             "method": "union over cycles of the per-cycle work zones (conveyor-attached frame at t_ref)"},
        ]
        if all_line:
            zones.append({"name": "line_side", "polygon_xy": M.hull_polygon(np.vstack(all_line), 0.15),
                          "method": "union over cycles of the per-cycle line-side stops"})
        speeds = [c["conveyor"]["speed_mm_s"] for c in st_cycles if c["conveyor"]["speed_mm_s"]]
        stations.append({
            "station": st,
            "operator_side": op_side,
            "operator_side_method": op_method,
            "conveyor_centreline_y_m": round(y_c, 3),
            "belt_edges_y_m": [round(y_lo, 3), round(y_hi, 3)],
            "station_centre_x_m": round(xc, 3),
            "conveyor_speed_mm_s_median": None if not speeds else round(float(np.median(speeds)), 2),
            "zones": zones,
            "cycles": st_cycles,
        })
    sg = {
        "units": "m",
        "frame": ("s0 world frame (x along the line WS10->WS30, y towards IN cameras, z up, floor z=0). Human data "
                  "is conveyor-attached (see module docstring) and coincides with the world at each cycle's t_ref_s."),
        "stations": stations,
    }
    p = save_json(sg, out_dir(cfg, "s1") / "scene_graph.json", contract="scene_graph")
    print(f"    wrote {rel(p)}")
