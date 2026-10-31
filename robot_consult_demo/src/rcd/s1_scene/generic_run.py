"""s1 scene graph from ordinary calibrated video only (config `mode.human_source: video`, `mode.conveyor: flow|intake`).

Inputs: the clips (data/<station>/mp4), the s0 calibration and the customer intake. No wearable/mocap data is read.

Per station:
1. People: MediaPipe Pose Landmarker over every clip (`pose2d`, pass 1 at POSE_FPS/2, all people); the worker is
   selected per camera (`video_pose.select_worker`), a first 3D estimate predicts the worker's image box in every
   camera (also where that camera's detector missed him), and a top-down second pass (`pose2d.extract_crops`,
   POSE_FPS) re-estimates the landmarks inside those boxes.
2. Conveyor (`conveyor_flow`): belt band and direction from coherent floor-plane motion in each camera (people
   masked), combined across cameras; speed per cycle by phase correlation of the belt interior. Intake
   `conveyor.polygon_xy` / `direction_xy` override the detected geometry when given.
3. 3D worker (`video_pose`): multi-view triangulation when >= 2 calibrated cameras saw the clip (mode.pose_views
   auto|multi), else monocular metric lifting anchored on the floor; smoothing and fixed bone lengths.
4. Frame: the worker stands on the moving belt part of the time. Positions are expressed like a foot-contact
   (zero-velocity-update) body tracker would: while both feet are on the belt the belt's displacement is removed,
   i.e. x_out(t) = x_world(t) - v * d * (time on belt between t_ref and t). Contacts at the workpiece are thus in
   the conveyor-attached frame (coincident with the world at t_ref = mid-clip), the convention of the rest of the
   pipeline; walking on the floor keeps true floor displacements. The per-frame shift is saved (`belt_shift_m`).
5. Zones / objects from the video skeleton with the same kinematic rules as the original stage (`mocap.analyse`).
"""
from __future__ import annotations

import json
import multiprocessing as mp_
import os
import time
from pathlib import Path

import numpy as np

from rcd.common.intake import load_intake, mode, station as intake_station
from rcd.common.io import ROOT, load_json, out_dir, save_json
from rcd.s0_calib.discover import discover_videos, rel
from rcd.s1_scene import conveyor_flow as CF
from rcd.s1_scene import mocap as M
from rcd.s1_scene import pose2d
from rcd.s1_scene import video_pose as VP

POSE_FPS = 10.0        # output skeleton rate (the element analysis downstream runs at 10 Hz)
MODEL = "heavy"
WORKERS = max(1, min(5, (os.cpu_count() or 4) - 4))
ROI_GAP_S = 3.0        # predict the worker box across detector misses up to this long
WORK_MARGIN_M = 1.0    # work region for worker selection = belt band widened by this margin
MAP_CLIPS = 2          # clips per camera used to map the belt geometry (it is fixed per station)


def _cams(cfg) -> tuple[dict, dict]:
    cal = load_json(ROOT / cfg["out_root"] / "s0" / "calibration.json", contract="calibration")
    cams, serials = {}, {}
    for c in cal["cameras"]:
        cams[(c["station"], c["side"])] = {k: np.asarray(c[k], float) for k in ("K", "R", "t")} | {
            "image_size": tuple(c["image_size"]), "id": c["id"]}
        if c.get("serial"):
            serials[c["serial"]] = (c["station"], c["side"])
    return cams, serials


def _r3(p) -> list[float]:
    return [round(float(x), 4) for x in p]


def _map_job(args):
    path, cam, image_size, people = args
    return CF.motion_map(path, cam, image_size, people)


def _speed_job(args):
    path, cam, d, across, along = args
    return CF.belt_speed(path, cam, d, across, along)


def _pool_map(fn, jobs, workers):
    if workers > 1 and len(jobs) > 1:
        with mp_.get_context("spawn").Pool(min(workers, len(jobs))) as pool:
            return pool.map(fn, jobs)
    return [fn(j) for j in jobs]


def _people_floor(cam, d):
    """(frames, (N,P,2)) floor positions of all detected people (lowest visible foot point)."""
    return d["frames"], VP.candidate_floor_xy(cam, d["img"])


def _upsample(X: np.ndarray, factor: int, n_out: int) -> np.ndarray:
    """Pass-1 landmarks (N1,33,3) -> (n_out,33,3) on the finer timeline (linear between valid neighbours)."""
    out = np.full((n_out, *X.shape[1:]), np.nan)
    ok = np.isfinite(X[:, 0, 0])
    for i in range(n_out):
        a, r = divmod(i, factor)
        if r == 0 and a < len(X):
            out[i] = X[a]
        elif a + 1 < len(X) and ok[a] and ok[a + 1]:
            w = r / factor
            out[i] = (1 - w) * X[a] + w * X[a + 1]
    return out


def run_generic(cfg: dict) -> None:
    t_start = time.time()
    cams, serials = _cams(cfg)
    found = discover_videos(cfg, serials)
    intake = load_intake(cfg) if cfg.get("intake") else None
    views_mode = mode(cfg, "pose_views", "auto")
    conv_mode = mode(cfg, "conveyor", "flow")
    s1dir = out_dir(cfg, "s1")
    hdir = s1dir / "human"
    mdir = s1dir / "human_mono"
    hdir.mkdir(parents=True, exist_ok=True)
    mdir.mkdir(parents=True, exist_ok=True)
    cache = ROOT / cfg["out_root"] / "s1" / "cache"
    model = pose2d.model_path(ROOT, MODEL)
    timing = {}

    clips = []   # (station, cycle dict, side, path)
    for st, cycles in found.items():
        for c in cycles:
            for side in ("in", "out"):
                v = c[f"video_{side}"]
                if v is not None and (st, side) in cams:
                    clips.append((st, c, side, v))
    import cv2
    vfps = {}
    for _, _, _, v in clips:
        cap = cv2.VideoCapture(str(v))
        vfps[str(v)] = (cap.get(cv2.CAP_PROP_FPS) or cfg["data"].get("video_fps", 30), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        cap.release()
    fps_v = float(np.median([f for f, _ in vfps.values()]))
    stride2 = max(1, int(round(fps_v / POSE_FPS)))
    stride1 = 2 * stride2
    out_fps = fps_v / stride2

    # ---- pass 1: all people, full frames
    t0 = time.time()
    P1 = pose2d.extract_many([v for *_, v in clips], cache / "pose2d", model, stride1, WORKERS)
    timing["pose_pass1_s"] = round(time.time() - t0, 1)

    # ---- conveyor geometry: motion maps of up to MAP_CLIPS clips per camera (people masked) -> station band;
    #      then the belt speed of every clip inside that band
    t0 = time.time()
    st_conv = {}
    map_jobs, map_keys = [], []
    for (st, side) in sorted({(st, side) for st, _, side, _ in clips}):
        ist = intake_station(intake, st) if intake else None
        if conv_mode == "intake" or ((ist or {}).get("conveyor") or {}).get("polygon_xy"):
            continue
        cl = sorted([x for x in clips if x[0] == st and x[2] == side], key=lambda x: -vfps[str(x[3])][1])[:MAP_CLIPS]
        cam = cams[(st, side)]
        for _, c, _, v in cl:
            map_jobs.append((str(v), {k: cam[k] for k in ("K", "R", "t")}, cam["image_size"], _people_floor(cam, P1[str(v)])))
            map_keys.append((st, c["cycle_id"], side))
    maps_all = dict(zip(map_keys, _pool_map(_map_job, map_jobs, WORKERS)))
    for st in found:
        maps = [m for (s, _, _), m in maps_all.items() if s == st and m]
        ist = intake_station(intake, st) if intake else None
        cvi = (ist or {}).get("conveyor") or {}
        if cvi.get("polygon_xy"):
            d, across, along = CF.from_polygon(cvi["polygon_xy"], cvi.get("direction_xy"))
            how = "intake polygon_xy" + (" + direction_xy" if cvi.get("direction_xy") else " (direction = long axis)")
        elif maps:
            d = np.median(np.array([m["direction_xy"] for m in maps]), 0)
            d /= np.linalg.norm(d)
            if cvi.get("direction_xy"):
                di = np.asarray(cvi["direction_xy"], float)
                d = di / np.linalg.norm(di)
            across, how = CF.combine_edges(maps)
            al = np.array([m["along_m"] for m in maps])
            along = [float(np.min(al[:, 0])), float(np.max(al[:, 1]))]
            how = f"coherent floor-plane motion in {len(maps)} camera clips; band: {how}"
        else:
            d, across, along, how = np.array([1.0, 0.0]), None, None, "no coherent floor motion found"
        st_conv[st] = {"d": np.asarray(d, float), "across": across, "along": along, "how": how,
                       "maps": {f"{k[1]}/{k[2]}": m for k, m in maps_all.items() if k[0] == st}}
    sp_jobs, sp_keys = [], []
    for st, c, side, v in clips:
        g = st_conv[st]
        if g["across"] is None:
            continue
        cam = cams[(st, side)]
        sp_jobs.append((str(v), {k: cam[k] for k in ("K", "R", "t")}, g["d"].tolist(), g["across"], g["along"]))
        sp_keys.append((st, c["cycle_id"], side))
    conv_res = {k: {"speed": r} for k, r in zip(sp_keys, _pool_map(_speed_job, sp_jobs, WORKERS))}
    timing["conveyor_s"] = round(time.time() - t0, 1)
    stations = []

    # ---- worker selection + first 3D estimate -> ROIs for the second pass
    def work_region_for(st):
        g = st_conv[st]
        if g["across"] is None:
            return None
        pp = np.array([-g["d"][1], g["d"][0]])
        lo, hi = g["across"][0] - WORK_MARGIN_M, g["across"][1] + WORK_MARGIN_M
        return lambda xy: lo <= float(np.asarray(xy) @ pp) <= hi

    first = {}
    crop_jobs, crop_keys = [], []
    for st, cycles in found.items():
        wr = work_region_for(st)
        for c in cycles:
            vs, cs, info = [], [], []
            for side in ("in", "out"):
                v = c[f"video_{side}"]
                if v is None or (st, side) not in cams:
                    continue
                d = P1[str(v)]
                cam = dict(cams[(st, side)])
                sel = VP.select_worker(d["img"], cam, wr)
                vs.append(VP.mono_track(cam, d["img"], d["world"], sel))
                cs.append(cam)
                info.append((side, v, d, sel))
            if not vs:
                continue
            fu = VP.fuse(vs, cs) if views_mode != "mono" else {"X": vs[0]["X"]}
            n2 = max((vfps[str(v)][1] + stride2 - 1) // stride2 for _, v, _, _ in info)
            X1 = VP.fill_landmarks(fu["X"], int(ROI_GAP_S * out_fps / 2))
            X2 = _upsample(X1, 2, n2)
            for (side, v, d, sel), cam in zip(info, cs):
                own = np.full((len(sel), 4), np.nan)
                ok = sel >= 0
                own[ok] = VP._bbox(d["img"][np.where(ok)[0], sel[ok]][:, None])[:, 0]
                own2 = np.full((n2, 4), np.nan)
                own2[::2][:len(own)] = own[:len(own2[::2])]
                R = VP.rois(cam, X2, cam["image_size"], own2)
                idx = np.where(np.isfinite(R[:, 0]))[0]
                crop_jobs.append((v, idx * stride2, R[idx]))
                crop_keys.append((st, c["cycle_id"], side, idx))
            first[(st, c["cycle_id"])] = info
    t0 = time.time()
    crops = pose2d.extract_crops_many(crop_jobs, cache / "crops", model, WORKERS)
    timing["pose_pass2_s"] = round(time.time() - t0, 1)
    crop_by = {(k[0], k[1], k[2]): (r, k[3]) for k, r in zip(crop_keys, crops)}

    # ---- final 3D per cycle
    t0 = time.time()
    for st, cycles in found.items():
        g = st_conv[st]
        dvec = g["d"]
        pp = np.array([-dvec[1], dvec[0]])
        st_cycles, all_work, all_line = [], [], []
        line_across = []
        for c in cycles:
            cid = c["cycle_id"]
            if (st, cid) not in first:
                continue
            info = first[(st, cid)]
            n2 = max((vfps[str(v)][1] + stride2 - 1) // stride2 for _, v, _, _ in info)
            per_view, cs, sides = [], [], []
            for side, v, d, sel in info:
                r, idx = crop_by[(st, cid, side)]
                # pass-1 worker detections sit on every second frame of the pass-2 timeline
                img = np.full((n2, pose2d.MAX_POSES, 33, 5), np.nan, np.float32)
                wld = np.full((n2, pose2d.MAX_POSES, 33, 3), np.nan, np.float32)
                m = min(len(d["img"]), (n2 + 1) // 2)
                img[0:2 * m:2] = d["img"][:m]
                wld[0:2 * m:2] = d["world"][:m]
                sel2 = np.full(n2, -1, int)
                sel2[0:2 * m:2] = sel[:m]
                oi, ow = VP.merge_pass(img, wld, sel2, {"img": r["img"], "world": r["world"], "boxes": r["boxes"], "index": idx})
                s2 = np.where(np.isfinite(oi[:, 0, 0, 0]), 0, -1)
                cam = dict(cams[(st, side)])
                per_view.append(VP.mono_track(cam, oi, ow, s2))
                cs.append(cam)
                sides.append(side)
            # conveyor speed of this cycle: mean over its cameras
            est = {f"video_{s}_floor_flow": conv_res[(st, cid, s)]["speed"] for s in sides
                   if conv_res.get((st, cid, s), {}).get("speed")}
            speeds = [e["speed_mm_s"] for e in est.values()]
            v_mm = float(np.mean(speeds)) if speeds else None
            sgn = [e["direction_sign"] for e in est.values()]
            dsign = float(np.sign(np.sum(sgn))) if sgn and np.sum(sgn) != 0 else 1.0

            def finish(views, cams_, tag):
                use_multi = views_mode in ("auto", "multi") and len(views) >= 2
                fu = VP.fuse(views, cams_, gate=True) if use_multi else VP.fuse(views[:1], cams_[:1], gate=True)
                J = VP.named(fu["X"])
                Js, valid, tinfo = VP.temporal(J, out_fps, length_frames=fu["mode"] == 2)
                shift = np.zeros(len(Js))
                if v_mm and g["across"] is not None:
                    I = {n: k for k, n in enumerate(VP.NAMES)}
                    fa = Js[:, [I["LeftAnkle"], I["RightAnkle"]], :2] @ pp
                    on = ((fa > g["across"][0]) & (fa < g["across"][1])).all(1).astype(float)
                    disp = np.cumsum(on) / out_fps * v_mm / 1000.0
                    disp -= disp[len(disp) // 2]
                    shift = disp
                    Js = Js - (shift[:, None, None] * np.r_[dsign * dvec, 0.0][None, None, :])
                return Js.astype(np.float32), valid, fu, tinfo, shift

            Jw, valid, fu, tinfo, shift = finish(per_view, cs, "main")
            views_used = "multi" if (views_mode in ("auto", "multi") and len(per_view) >= 2) else "mono"
            npz = hdir / f"{st}_{cid}.npz"
            np.savez_compressed(npz, joints=Jw, names=np.array(VP.NAMES), fps=np.float32(out_fps), valid=valid,
                                frame_mode=fu["mode"].astype(np.int8), views=np.array(views_used),
                                belt_shift_m=shift.astype(np.float32))
            mono_files = {}
            if len(per_view) >= 2:   # monocular reconstructions of the same cycle, for accuracy reporting
                for k, s in enumerate(sides):
                    Jm, vm, fum, _, shm = finish([per_view[k]], [cs[k]], s)
                    f = mdir / f"{st}_{cid}_{s}.npz"
                    np.savez_compressed(f, joints=Jm, names=np.array(VP.NAMES), fps=np.float32(out_fps), valid=vm,
                                        views=np.array("mono"), belt_shift_m=shm.astype(np.float32))
                    mono_files[s] = rel(f)

            T = len(Jw)
            dur = vfps[str(info[0][1])][1] / fps_v
            an = M.analyse(Jw.astype(float), VP.NAMES, out_fps)
            feet = an["feet_idx"]
            Fw = an["Js"][:, feet]
            work_pts = Fw[an["work"]][:, :, :2].reshape(-1, 2)
            line_pts = Fw[an["line"]][:, :, :2].reshape(-1, 2)
            if len(work_pts):
                all_work.append(work_pts[::3])
            if len(line_pts):
                all_line.append(line_pts[::3])
                line_across.append(float(np.median(line_pts @ pp)))
            zones = []
            if len(work_pts):
                zones.append({"name": "work_zone", "polygon_xy": M.hull_polygon(work_pts[::3]),
                              "method": "convex hull (+0.1 m) of video ankle/toe xy while the pelvis is still (<0.2 m/s) "
                                        "within 0.6 m of the most-occupied standing spot"})
            if len(line_pts):
                zones.append({"name": "line_side", "polygon_xy": M.hull_polygon(line_pts[::3], 0.15),
                              "method": "convex hull (+0.15 m) of feet at the stop farthest from the work zone",
                              "distance_from_work_zone_m": round(float(an["line_dist_m"]), 3)})
            cl = M.cluster_points(an["contacts"])
            sup = M.cluster_points(an["supply"], 0.15)
            objects = [{
                "name": "workpiece_contact",
                "positions_m": [_r3(k["p"]) for k in cl],
                "dwell_s": [round(k["dwell_s"], 2) for k in cl],
                "hands": [k["hands"] for k in cl],
                "first_t_s": [round(k["first_t_s"], 2) for k in cl],
                "method": f"video wrist positions during hand dwells (<{M.HAND_STILL} m/s for >= {M.MIN_HAND_S} s) "
                          f"while standing in the work zone, clustered at {M.CONTACT_CLUSTER} m (dwell-weighted)",
                "frame": "conveyor-attached, coincident with world at t_ref_s",
            }, {
                "name": "line_side_supply",
                "positions_m": [_r3(k["p"]) for k in sup],
                "dwell_s": [round(k["dwell_s"], 2) for k in sup],
                "method": "video wrist positions during slow-hand moments at the line-side stop, clustered at 0.15 m",
            }]
            in_belt = None
            if g["across"] is not None and len(work_pts):
                a_ = work_pts @ pp
                in_belt = float(np.mean((a_ > g["across"][0]) & (a_ < g["across"][1])))
            vid = {"in": rel(c["video_in"]) if c["video_in"] else None,
                   "out": rel(c["video_out"]) if c["video_out"] else None,
                   "fps": fps_v, "duration_s": round(dur, 3)}
            if c["video_side_inferred"]:
                vid["side_inferred"] = c["video_side_inferred"]
            dirw = None
            if sgn and len(set(sgn)) == 1:
                dd = dvec * sgn[0]
                dirw = [round(float(dd[0]), 4), round(float(dd[1]), 4), 0.0]
            st_cycles.append({
                "cycle_id": cid,
                "duration_s": round(dur, 4),
                "duration_note": "video clip length (frames / fps)",
                "t_ref_s": round(T / out_fps / 2, 3),
                "video": vid,
                "human": {
                    "joints_npz": rel(npz),
                    "joint_names": VP.NAMES,
                    "fps": out_fps,
                    "n_frames": T,
                    "source": (f"video: MediaPipe Pose Landmarker ({MODEL}) on {len(sides)} calibrated camera(s) "
                               f"[{', '.join(sides)}], " + ("multi-view DLT triangulation with monocular fallback"
                                                            if views_used == "multi" else "monocular metric lifting anchored on the floor plane")
                               + "; worker track selection, top-down second pass, smoothing, fixed bone lengths"),
                    "views": views_used,
                    "valid_frac": round(float(valid.mean()), 4),
                    "observed_frac": round(tinfo["observed_frac"], 4),
                    "multi_view_frame_frac": round(float((fu["mode"] == 2).mean()), 4),
                    "bone_lengths_m": tinfo["bone_lengths_m"],
                    "body_scale": {sd: {"scale": round(pv["scale"], 3), "source": pv["scale_source"], "feet_frames": pv["scale_n"]}
                                   for sd, pv in zip(sides, per_view)},
                    "mono_reconstructions": mono_files,
                    "frame": "foot-contact frame: belt displacement removed while both feet are on the belt (see module doc)",
                },
                "registration": {
                    "method": "none needed: joints are measured directly in the s0 world frame (camera calibration); "
                              "belt displacement removed while standing on the belt (belt_shift_m in the npz)",
                    "check_work_feet_on_belt_frac": None if in_belt is None else round(in_belt, 3),
                },
                "conveyor": {
                    "speed_mm_s": None if v_mm is None else round(v_mm, 2),
                    "method": ("mean of per-camera estimates: belt band from coherent floor-plane optical flow "
                               "(metric bird's-eye view from s0 K,R,t, people masked), speed by phase correlation of the "
                               "belt interior ~2 s apart, median over frame pairs with correlation response >= 0.2"),
                    "direction_world": dirw,
                    "estimates": {k: round(e["speed_mm_s"], 2) for k, e in est.items()},
                    "estimate_details": est,
                    "relative_spread": None if len(speeds) < 2 else round((max(speeds) - min(speeds)) / v_mm, 3),
                },
                "objects": objects,
                "zones": zones,
            })
            print(f"    {st}/{cid}: {dur:.1f}s {views_used} valid={valid.mean():.2f} multi={np.mean(fu['mode'] == 2):.2f} "
                  f"conveyor {v_mm and round(v_mm, 1)} mm/s contacts {len(cl)}", flush=True)

        if not st_cycles:
            continue
        zones = []
        if g["across"] is not None:
            al = g["along"]
            zones.append({"name": "conveyor", "polygon_xy": CF.polygon(g["d"], g["across"], al),
                          "method": f"belt band: {g['how']}; along-line extent = floor visible to the cameras",
                          "direction_xy": [round(float(x), 4) for x in g["d"]]})
        if all_work:
            zones.append({"name": "work_zone", "polygon_xy": M.hull_polygon(np.vstack(all_work)),
                          "method": "union over cycles of the per-cycle work zones (conveyor-attached frame at t_ref)"})
        if all_line:
            zones.append({"name": "line_side", "polygon_xy": M.hull_polygon(np.vstack(all_line), 0.15),
                          "method": "union over cycles of the per-cycle line-side stops"})
        # operator side = the camera side on which the worker's line-side stops lie (else the side of the work zone)
        sides_av = [s for s in ("in", "out") if (st, s) in cams]
        cam_across = {s: float((-cams[(st, s)]["R"].T @ cams[(st, s)]["t"])[:2] @ pp) for s in sides_av}
        ref = np.median(line_across) if line_across else (float(np.median(np.vstack(all_work) @ pp)) if all_work else None)
        if ref is not None and g["across"] is not None and cam_across:
            centre = 0.5 * (g["across"][0] + g["across"][1])
            op_side = min(cam_across, key=lambda s: abs(np.sign(cam_across[s] - centre) - np.sign(ref - centre)))
            op_method = (f"side of the belt (across-line coordinate {ref:.2f} m vs belt centre {centre:.2f} m) where the "
                         f"worker's line-side stops lie, mapped to the camera on that side")
        else:
            op_side, op_method = (sides_av[0] if sides_av else "in"), "default: first calibrated camera"
        speeds = [c["conveyor"]["speed_mm_s"] for c in st_cycles if c["conveyor"]["speed_mm_s"]]
        stations.append({
            "station": st,
            "operator_side": op_side,
            "operator_side_method": op_method,
            "conveyor_centreline_y_m": None if g["across"] is None else round(0.5 * (g["across"][0] + g["across"][1]), 3),
            "belt_edges_y_m": None if g["across"] is None else [round(g["across"][0], 3), round(g["across"][1], 3)],
            "belt_edges_note": "across-line coordinates (equal to world y when the line runs along x)",
            "station_centre_x_m": round(float(np.mean([(-cams[(st, s)]["R"].T @ cams[(st, s)]["t"])[0] for s in sides_av])), 3),
            "conveyor_speed_mm_s_median": None if not speeds else round(float(np.median(speeds)), 2),
            "conveyor_detection": {"method": g["how"], "per_clip": {
                k: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in m.items()
                    if kk in ("direction_xy", "near_edge_m", "far_edge_m", "far_edge_stop", "flow_speed_mm_s")}
                for k, m in g.get("maps", {}).items() if m}},
            "zones": zones,
            "cycles": st_cycles,
        })
    timing["reconstruction_s"] = round(time.time() - t0, 1)
    timing["total_s"] = round(time.time() - t_start, 1)
    sg = {
        "units": "m",
        "frame": ("s0 world frame (z up, floor z=0). Human data: foot-contact frame (belt displacement removed while "
                  "standing on the belt) coinciding with the world at each cycle's t_ref_s."),
        "mode": {"human_source": "video", "pose_views": views_mode, "conveyor": conv_mode},
        "timing_s": timing,
        "stations": stations,
    }
    p = save_json(sg, s1dir / "scene_graph.json", contract="scene_graph")
    print(f"    wrote {rel(p)} timing {json.dumps(timing)}")
