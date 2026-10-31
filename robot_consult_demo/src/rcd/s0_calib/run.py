"""s0 camera calibration -> out/s0/calibration.json

K, R, t per camera come from CarDA `cameras_calibration.xlsx` (x_cam = R X_world + t). They are checked
against (a) the camera world positions listed in the same sheet (C = -R^T t) and (b) targets detected
in the mp4 videos (checkerboard on OUT views, ArUco on the cart in IN views) via solvePnP reprojection.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from rcd.common.io import ROOT, out_dir, save_json
from rcd.s0_calib import verify
from rcd.s0_calib.discover import discover, discover_videos, rel
from rcd.s0_calib.xlsx_calib import parse

# ZED2 native resolutions (w, h); the sheet's principal point tells which one the intrinsics are for
ZED_RES = {"HD2K": (2208, 1242), "HD1080": (1920, 1080), "HD720": (1280, 720), "VGA": (672, 376)}
N_OUT_FRAMES = 4   # checkerboard frames sampled per video
N_IN_FRAMES = 12   # ArUco frames sampled per video (when no checkerboard)


def _orthonormal(R: np.ndarray) -> tuple[np.ndarray, float]:
    U, _, Vt = np.linalg.svd(R)
    Ro = U @ Vt
    if np.linalg.det(Ro) < 0:
        U[:, -1] *= -1
        Ro = U @ Vt
    return Ro, float(np.abs(Ro - R).max())


def _native_res(K: np.ndarray) -> tuple[str, tuple[int, int]]:
    cx, cy = K[0, 2], K[1, 2]
    return min(ZED_RES.items(), key=lambda kv: abs(kv[1][0] / 2 - cx) + abs(kv[1][1] / 2 - cy))


def _video_size(path) -> tuple[int, int] | None:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    wh = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    return wh


def _pool(results: list[dict], key: str, weight: str) -> float | None:
    rs = [r for r in results if r and r.get(key) is not None]
    if not rs:
        return None
    w = np.array([r[weight] for r in rs], float)
    v = np.array([r[key] for r in rs], float)
    return float(np.sqrt(np.sum(w * v ** 2) / np.sum(w)))


def _generic_sources(cfg: dict):
    """Generic mode (config `mode` / `intake` present): calibration source from intake.cameras and cycles
    enumerated from the video clips only. Returns (xlsx path or None, cycles, intake cameras dict)."""
    from rcd.common.intake import load_intake, mode
    it = load_intake(cfg) if cfg.get("intake") else None
    cams_it = (it or {}).get("cameras", {}) or {}
    video_only = mode(cfg, "human_source", "mocap") == "video"
    method = cams_it.get("calibration", "file")
    f = cams_it.get("file") or cfg["data"].get("calibration_xlsx")
    xlsx = ROOT / f if (method == "file" and f and (ROOT / f).exists()) else None
    serials = {e["sn"]: k for k, e in parse(xlsx).items()} if xlsx else {}
    cycles = discover_videos(cfg, serials) if video_only else discover(cfg)
    return xlsx, cycles, cams_it


def _reference_clips(cfg: dict, xlsx) -> dict:
    """Clips recorded together with the calibration file (its own data folder), when the analysed footage lives
    elsewhere: {(station, side): clip}. Used to check that the cameras have not moved since calibration."""
    from pathlib import Path as _P
    try:
        same = _P(xlsx).parent.resolve() == (ROOT / cfg["data"]["root"]).resolve()
    except OSError:
        same = True
    if same:
        return {}
    c2 = dict(cfg)
    c2["data"] = dict(cfg["data"], root=str(_P(xlsx).parent.relative_to(ROOT)) if _P(xlsx).is_relative_to(ROOT) else str(_P(xlsx).parent))
    serials = {e["sn"]: k for k, e in parse(xlsx).items()}
    out = {}
    for st, cyc in discover_videos(c2, serials).items():
        for c in cyc:
            for side in ("in", "out"):
                if c[f"video_{side}"] is not None:
                    out.setdefault((st, side), c[f"video_{side}"])
    return out


def run(cfg: dict) -> None:
    generic = "mode" in cfg or bool(cfg.get("intake"))
    refclips = {}
    if generic:
        xlsx, cycles, cams_it = _generic_sources(cfg)
        if xlsx is None:
            return _run_without_file(cfg, cycles, cams_it)
        refclips = _reference_clips(cfg, xlsx)
    else:
        xlsx = ROOT / cfg["data"]["calibration_xlsx"]
        cycles = discover(cfg)
    sheet = parse(xlsx)
    cams = []
    for st in cfg["data"]["stations"]:
        for side in ("in", "out"):
            if (st, side) not in sheet:
                print(f"    s0: {st}/{side} missing from {xlsx.name}")
                continue
            e = sheet[(st, side)]
            R, r_fix = _orthonormal(e["R"])
            t = e["T"].astype(float)
            K = e["K"].astype(float).copy()
            res_name, (nw, nh) = _native_res(K)
            vids = [c[f"video_{side}"] for c in cycles.get(st, []) if c[f"video_{side}"] is not None]
            vsize = _video_size(vids[0]) if vids else None
            W, H = vsize if vsize else (nw, nh)
            scale = W / nw
            scale_note = ("no rescaling needed" if abs(scale - 1) < 1e-6
                          else f"K rescaled by {scale:.4f} from {nw}x{nh} to {W}x{H}")
            if abs(scale - 1) > 1e-6:
                K[:2] *= scale
            dist = np.zeros(5)
            C = -R.T @ t
            listed = e.get("listed_center")
            c_err = float(np.linalg.norm(C - listed)) if listed is not None else None
            shift = None
            if generic and refclips.get((st, side)) and vids:
                # footage from another session than the calibration package: has the camera moved since?
                from rcd.s0_calib import generic_calib as G
                shift = G.scene_shift(refclips[(st, side)], vids[0], K)
                if shift and shift.get("stale"):
                    Rd = shift.pop("R_delta")
                    R, t = Rd @ R, Rd @ t
                    C = -R.T @ t
                    shift["action"] = "extrinsic rotation re-estimated from the background homography (centre kept)"
                elif shift:
                    shift["action"] = "calibration still valid (camera has not moved)"

            checks, used = [], []
            board = True  # static camera: if the first video shows no board, none of them will
            for c in cycles.get(st, []):
                v = c[f"video_{side}"]
                if v is None:
                    continue
                # checkerboard first (static target, strong check), else ArUco markers on the cart
                r = verify.checkerboard(verify.sample_frames(v, N_OUT_FRAMES), K, dist, R) if board else None
                board = r is not None
                if r:
                    r["n_points"] = r["frames_detected"] * r["pattern_inner_corners"][0] * r["pattern_inner_corners"][1]
                    r["rmse_px"] = r["rmse_px_joint_static_pose"]
                else:
                    r = verify.aruco(verify.sample_frames(v, N_IN_FRAMES), K, dist, R)
                    if r:
                        r["n_points"] = r["detections"] * 4
                if r:
                    r["video"] = rel(v)
                    checks.append(r)
                    used.append(c["cycle_id"])
            rmse = _pool(checks, "rmse_px", "n_points")

            src = (f"{xlsx.name} (CarDA) camera {e['sn']}: K,R,T as given (x_cam = R X_world + T); "
                   f"intrinsics are for ZED {res_name} {nw}x{nh} (principal point ~ image centre), "
                   f"mp4 = left image {W}x{H}: {scale_note}; dist = 0 (ZED-rectified left image, sheet has no "
                   f"distortion terms); camera centre -R^T T vs listed world position: "
                   f"{'%.3f m' % c_err if c_err is not None else 'n/a'}")
            if checks:
                kinds = sorted({r["target"] for r in checks})
                src += (f"; verified: detected {'+'.join(kinds)} in sampled frames of cycles {','.join(used)}, "
                        f"solvePnP reprojection RMSE {rmse:.2f}px")
                if kinds == ["aruco"]:
                    src += " (4-corner marker poses: weak check)"
            else:
                src += "; no video verification (no video or no target detected)"
            if shift is not None:
                src += (f"; staleness check vs calibration-session footage: background shift "
                        f"{shift['median_displacement_px']:.1f}px over {shift['inlier_matches']} matches -> {shift['action']}")
            ext = {}
            if any(r["target"] == "checkerboard" for r in checks):
                for k in ("board_edge_elevation_deg", "board_normal_elevation_deg"):
                    ext[k] = float(np.median([r[k] for r in checks if k in r]))
            if any(r["target"] == "aruco" for r in checks):
                ang = [r["marker_motion_angle_from_x_deg"] for r in checks if "marker_motion_angle_from_x_deg" in r]
                if ang:
                    ext["marker_motion_angle_from_x_deg_median"] = float(np.median(ang))
            cams.append({
                "id": f"{st}_{side}",
                "station": st,
                "side": side,
                "serial": e["sn"],
                "image_size": [int(W), int(H)],
                "K": K.tolist(),
                "dist": dist.tolist(),
                "dist_model": "opencv k1,k2,p1,p2,k3",
                "R": R.tolist(),
                "t": t.tolist(),
                "convention": "x_cam = R @ X_world + t (OpenCV camera: x right, y down, z forward)",
                "center_world_m": C.tolist(),
                "center_listed_m": None if listed is None else listed.tolist(),
                "center_vs_listed_m": c_err,
                "R_orthonormalization_max_abs_change": r_fix,
                "fov_h_deg": float(2 * math.degrees(math.atan(W / 2 / K[0, 0]))),
                "source": src,
                "reproj_rmse_px": rmse,
                "extrinsic_checks": ext,
                "verification": checks,
                **({"staleness_check": shift} if shift is not None else {}),
            })
    cal = {
        "world_frame": {
            "units": "m",
            "up_axis": "z",
            "origin_desc": ("CarDA plant frame from cameras_calibration.xlsx (layout figure in the sheet): origin "
                            "on the floor at the WS10 end of the line on the OUT-camera side; +x along the line "
                            "(WS10 -> WS30, conveyor travel direction), +y across the line towards the IN cameras, "
                            "+z up; floor at z = 0 (camera centres at z ~ 1.8-1.9 m on 1.85 m rods)."),
            "x_axis": "along the line, WS10 -> WS30",
            "y_axis": "across the line, OUT side -> IN side",
        },
        "cameras": cams,
    }
    p = save_json(cal, out_dir(cfg, "s0") / "calibration.json", contract="calibration")
    for c in cams:
        r = c["reproj_rmse_px"]
        print(f"    {c['id']}: fx={c['K'][0][0]:.1f} C={np.round(c['center_world_m'], 2).tolist()} "
              f"|C-listed|={c['center_vs_listed_m'] if c['center_vs_listed_m'] is None else round(c['center_vs_listed_m'], 3)} "
              f"rmse={'-' if r is None else round(r, 3)} {c['extrinsic_checks']}")
    print(f"    wrote {rel(p)}")


def _run_without_file(cfg: dict, cycles: dict, cams_it: dict) -> None:
    """Calibrate from the footage itself when the customer has no calibration file (intake cameras.calibration =
    checkerboard | reference_length). Checkerboard: `cameras.checkerboard = {pattern: [cols, rows], square_m}`;
    reference_length: `cameras.reference_lengths_m = [{station, side, image_px, world_xy_m}]`."""
    from rcd.s0_calib import generic_calib as G
    method = cams_it.get("calibration", "none")
    cams = []
    for st in cfg["data"]["stations"]:
        for side in ("in", "out"):
            vids = [c[f"video_{side}"] for c in cycles.get(st, []) if c[f"video_{side}"] is not None]
            if not vids:
                continue
            W, H = _video_size(vids[0])
            res = None
            if method == "checkerboard":
                cb = cams_it.get("checkerboard", {})
                frames = [g for v in vids for _, g in verify.sample_frames(v, 12)]
                res = G.calibrate_checkerboard(frames, tuple(cb.get("pattern", (6, 8))), float(cb.get("square_m", 0.05)))
            elif method == "reference_length":
                ref = next((r for r in cams_it.get("reference_lengths_m", [])
                            if r.get("station") == st and r.get("side") == side), None)
                if ref:
                    res = G.calibrate_reference_length(ref["image_px"], ref["world_xy_m"], (W, H))
            if res is None:
                print(f"    s0: {st}/{side}: no calibration possible with method '{method}'")
                continue
            R, t, K = res["R"], res["t"], res["K"]
            C = -R.T @ t
            cams.append({
                "id": f"{st}_{side}", "station": st, "side": side, "serial": None, "image_size": [int(W), int(H)],
                "K": K.tolist(), "dist": np.asarray(res["dist"]).tolist(), "dist_model": "opencv k1,k2,p1,p2,k3",
                "R": R.tolist(), "t": t.tolist(),
                "convention": "x_cam = R @ X_world + t (OpenCV camera: x right, y down, z forward)",
                "center_world_m": C.tolist(), "center_listed_m": None, "center_vs_listed_m": None,
                "R_orthonormalization_max_abs_change": 0.0,
                "fov_h_deg": float(2 * math.degrees(math.atan(W / 2 / K[0, 0]))),
                "source": f"{res['method']}: detected targets in the customer video, reprojection RMSE {res['rmse_px']:.2f}px",
                "reproj_rmse_px": res["rmse_px"], "extrinsic_checks": {}, "verification": [],
            })
    cal = {"world_frame": {"units": "m", "up_axis": "z",
                           "origin_desc": "floor-referenced frame defined by the calibration target / reference points; floor z = 0",
                           "x_axis": "target x", "y_axis": "target y"},
           "cameras": cams}
    p = save_json(cal, out_dir(cfg, "s0") / "calibration.json", contract="calibration")
    print(f"    wrote {rel(p)}")
