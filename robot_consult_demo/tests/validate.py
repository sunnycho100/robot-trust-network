"""Acceptance gate for every pipeline stage. Stage implementations should not change these checks.

Each check re-derives numbers from the stage outputs (and, where possible, from the raw CarDA data)
instead of trusting self-reported metrics. Run:  python tests/validate.py [--stage s2 ...]
Exit code 0 only if every selected stage passes.
"""
from __future__ import annotations

import argparse
import os
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rcd.common.io import load_config, validate  # noqa: E402
sys.path.insert(0, str(ROOT / "tests"))
import _ie_checks as IE  # noqa: E402

CFG = load_config(os.environ.get("RCD_CONFIG"))  # e.g. RCD_CONFIG=config/carda_generic.yaml
OUT = ROOT / CFG["out_root"]
RESULTS: dict[str, list[tuple[bool, str]]] = {}


def chk(stage: str, ok: bool, msg: str) -> bool:
    RESULTS.setdefault(stage, []).append((bool(ok), msg))
    return bool(ok)


def load(stage: str, name: str, contract: str | None = None):
    p = OUT / stage / name
    if not p.exists():
        chk(stage, False, f"missing output {p.relative_to(ROOT)}")
        return None
    obj = json.loads(p.read_text())
    if contract:
        try:
            validate(obj, contract)
            chk(stage, True, f"{name} matches contract '{contract}'")
        except Exception as e:  # jsonschema.ValidationError
            chk(stage, False, f"{name} violates contract '{contract}': {str(e).splitlines()[0]}")
            return None
    return obj


def cycle_id(name: str) -> str | None:
    """'xsens_002_WS10_2023_09_21_...' -> '002_0921' (ids repeat across recording days in WS10)."""
    m = re.search(r"xsens_(\d+)_WS\d+_\d{4}_(\d{2})_(\d{2})", name)
    if m:
        return f"{m.group(1)}_{m.group(2)}{m.group(3)}"
    # subset-B style recordings: 'WS30-SN<serial>-HD720dt__DD_MM_YYYY_HH_MM_SS...' -> 'HHMMSS_MMDD'
    m = re.search(r"WS\d+-SN\d+-\w*?__(\d{2})_(\d{2})_\d{4}_(\d{2})_(\d{2})_(\d{2})", name)
    return f"{m.group(3)}{m.group(4)}{m.group(5)}_{m.group(2)}{m.group(1)}" if m else None


def mocap_cycles() -> dict[str, list[str]]:
    """Ground truth list of cycles = BVH files present in the dataset (or, in video mode, the recorded clips)."""
    res = {}
    video = CFG.get("mode", {}).get("human_source", "mocap") == "video"
    for st in CFG["data"]["stations"]:
        d = ROOT / CFG["data"]["root"] / st.lower()
        files = sorted(d.glob("mp4/*.mp4")) if video else sorted((d / "xsens").glob("*.bvh"))
        res[st] = sorted({cycle_id(f.name) for f in files} - {None})
    return res


def annotated_subgoals() -> dict[tuple[str, str], list[tuple[str, float, float]]]:
    """Independent parse of CarDA annotation workbooks -> {(station, cycle_id): [(subgoal, start, end)]}."""
    import pandas as pd
    res = {}
    for st in CFG["data"]["stations"]:
        xl = sorted((ROOT / CFG["data"]["root"] / st.lower()).glob("annotation_data_*.xlsx"))
        if not xl:
            continue
        for sheet, df in pd.read_excel(xl[0], sheet_name=None, header=None).items():
            cid = cycle_id(sheet) or (cycle_id(str(df.iloc[1, 3])) if df.shape[1] > 3 else None)
            if not cid:
                continue
            blocks, cur = [], None
            for _, row in df.iloc[3:].iterrows():
                name = row.iloc[0]
                if isinstance(name, str) and name.strip():
                    if cur:
                        blocks.append(cur)
                    cur = [name.strip(), row.iloc[3], row.iloc[4]]
                elif cur is not None:
                    if not (isinstance(row.iloc[4], float) and math.isnan(row.iloc[4])) and row.iloc[4] is not None:
                        cur[2] = row.iloc[4]
            if cur:
                blocks.append(cur)
            clean = []
            for n, s, e in blocks:
                try:
                    s, e = float(s), float(e)
                except (TypeError, ValueError):
                    continue
                if not (math.isnan(s) or math.isnan(e)):
                    clean.append((n, s, e))
            res[(st, cid)] = clean
    return res


# ------------------------------------------------------------------ S0
def check_s0():
    st = "s0"
    cal = load(st, "calibration.json", "calibration")
    if not cal:
        return
    cams = cal["cameras"]
    need = {(s, side) for s in CFG["data"]["stations"] for side in ("in", "out")}
    have = {(c["station"], c["side"]) for c in cams}
    chk(st, need <= have, f"cameras for every station/side ({len(have & need)}/{len(need)})")
    for c in cams:
        R = np.array(c["R"])
        chk(st, np.allclose(R.T @ R, np.eye(3), atol=1e-3) and abs(np.linalg.det(R) - 1) < 1e-3,
            f"{c['id']}: R is a proper rotation")
        K = np.array(c["K"])
        chk(st, K[0, 0] > 100 and K[1, 1] > 100 and 0 < K[0, 2] < c["image_size"][0],
            f"{c['id']}: K plausible (fx={K[0,0]:.0f}, cx={K[0,2]:.0f})")
    detected = [c for c in cams if "detect" in c["source"].lower() and c["reproj_rmse_px"] is not None]
    chk(st, len(detected) >= 1, f"{len(detected)} camera(s) verified against detected targets in video")
    for c in detected:
        chk(st, c["reproj_rmse_px"] <= 3.0, f"{c['id']}: reprojection RMSE {c['reproj_rmse_px']:.2f}px <= 3.0")
    chk(st, cal["world_frame"]["units"] == "m" and cal["world_frame"]["up_axis"] == "z", "world frame in metres, z-up")


# ------------------------------------------------------------------ S1
def check_s1():
    st = "s1"
    sg = load(st, "scene_graph.json", "scene_graph")
    if not sg:
        return
    truth = mocap_cycles()
    for stn in sg["stations"]:
        ids = sorted(c["cycle_id"] for c in stn["cycles"])
        chk(st, ids == sorted(truth.get(stn["station"], [])), f"{stn['station']}: cycles {ids} == mocap files")
        speeds = []
        for c in stn["cycles"]:
            npz = ROOT / c["human"]["joints_npz"]
            if not chk(st, npz.exists(), f"{stn['station']}/{c['cycle_id']}: joints file exists"):
                continue
            d = np.load(npz)
            J = d["joints"]  # (T, J, 3) metres, z-up world
            T = J.shape[0] / c["human"]["fps"]
            chk(st, abs(T - c["duration_s"]) / c["duration_s"] < 0.03,
                f"{stn['station']}/{c['cycle_id']}: mocap length {T:.1f}s ~ duration {c['duration_s']:.1f}s")
            names = list(c["human"]["joint_names"])
            pairs = [("RightShoulder", "RightElbow"), ("RightElbow", "RightWrist"), ("LeftKnee", "LeftAnkle"),
                     ("RightHip", "RightKnee"), ("LeftShoulder", "LeftElbow")]
            cvs = []
            for a, b in pairs:
                if a in names and b in names:
                    L = np.linalg.norm(J[:, names.index(a)] - J[:, names.index(b)], axis=1)
                    cvs.append(L.std() / L.mean())
            chk(st, cvs and max(cvs) < 0.02, f"{stn['station']}/{c['cycle_id']}: bone lengths rigid (max CV {max(cvs) if cvs else -1:.4f})")
            if "Head" in names:
                feet = [names.index(n) for n in ("LeftAnkle", "RightAnkle") if n in names]
                # 'Head' is the skull-base joint (top of head is ~0.19 m higher), and workers bend a lot,
                # so test the upright (90th percentile) height rather than the median.
                h = np.percentile(J[:, names.index("Head"), 2] - J[:, feet, 2].min(axis=1), 90)
                chk(st, 1.25 <= h <= 1.95, f"{stn['station']}/{c['cycle_id']}: upright skull-base height above feet {h:.2f} m")
            v = c["conveyor"]["speed_mm_s"]
            chk(st, 0 < v < 500, f"{stn['station']}/{c['cycle_id']}: conveyor {v:.1f} mm/s in (0,500)")
            speeds.append(v)
        if len(speeds) > 1:
            cv = np.std(speeds) / np.mean(speeds)
            chk(st, cv < 0.35, f"{stn['station']}: conveyor speed consistent across cycles (CV {cv:.2f})")


# ------------------------------------------------------------------ S2
def check_s2():
    st = "s2"
    el = load(st, "elements.json", "elements")
    erg = load(st, "ergonomics.json", "ergonomics")
    ts = load(st, "time_study.json", "time_study")
    if el:
        codes = {t["code"] for t in el["taxonomy"]}
        used, src_fine, n_el = set(), 0, 0
        ann = annotated_subgoals()
        errs = []
        for c in el["cycles"]:
            E = sorted(c["elements"], key=lambda e: e["start_s"])
            tag = f"{c['station']}/{c['cycle_id']}"
            ok_order = all(E[k]["end_s"] <= E[k + 1]["start_s"] + 0.05 for k in range(len(E) - 1))
            chk(st, ok_order, f"{tag}: elements do not overlap")
            cov = sum(e["end_s"] - e["start_s"] for e in E) / c["duration_s"]
            chk(st, cov >= 0.90, f"{tag}: elements cover {cov:.0%} of clip (>=90%)")
            chk(st, all(abs((e["end_s"] - e["start_s"]) - e["duration_s"]) < 0.05 for e in E), f"{tag}: durations consistent")
            for e in E:
                used.add(e["code"])
                n_el += 1
                src_fine += e["source"] in ("mocap_rules", "merged", "vlm")
            for name, s0, e0 in ann.get((c["station"], c["cycle_id"]), []):
                mine = [e for e in E if (e.get("subgoal") or "").strip() == name]
                if mine:
                    errs += [abs(min(e["start_s"] for e in mine) - s0), abs(max(e["end_s"] for e in mine) - e0)]
                else:
                    errs += [99.0, 99.0]
        chk(st, used <= codes, "all element codes are in the taxonomy")
        chk(st, len(used) >= 4, f"{len(used)} distinct element types used (>=4)")
        chk(st, n_el / max(1, len(el["cycles"])) >= 5, f"{n_el / max(1, len(el['cycles'])):.1f} elements per cycle (>=5, finer than annotation)")
        chk(st, src_fine / max(1, n_el) >= 0.6, f"{src_fine / max(1, n_el):.0%} of elements derived from mocap/vision (>=60%)")
        sop = CFG.get("mode", {}).get("segmentation") == "sop"
        if errs and not sop:
            chk(st, float(np.median(errs)) <= 3.0, f"subgoal boundary error vs CarDA annotation: median {np.median(errs):.2f}s (<=3s)")
        elif errs:
            # SOP mode (design decision): text-only SOP carries no timing, so its boundary error is reported, not gated.
            # The gate is on SOP + standard-work-sheet times (leave-one-cycle-out), 5 s (~2% of a 240 s takt).
            chk(st, True, f"[info] text-only SOP boundary error vs annotation: median {np.median(errs):.2f}s")
            es = load(st, "elements_sop_std.json")
            eb = []
            for c in (es or {"cycles": []})["cycles"]:
                for name, s0, e0 in ann.get((c["station"], c["cycle_id"]), []):
                    mine = [e for e in c["elements"] if (e.get("subgoal") or "").strip() == name]
                    eb += [abs(min(e["start_s"] for e in mine) - s0), abs(max(e["end_s"] for e in mine) - e0)] if mine else [99.0, 99.0]
            chk(st, eb and float(np.median(eb)) <= 5.0,
                f"SOP + standard times (LOO) boundary error vs annotation: median {np.median(eb) if eb else -1:.2f}s (<=5s)")
        else:
            chk(st, False, "no annotated subgoals matched")
    if erg:
        v = erg["validation"]
        # Headline validation must exclude trivially-correct postures (e.g. "standing" == whole subgoal).
        chk(st, v.get("excludes_trivial") is True, "ergonomic validation excludes trivial postures (standing/walking)")
        pairs = np.array(v["pairs"], dtype=float) if v["pairs"] else np.zeros((0, 2))
        chk(st, len(pairs) >= 8, f"ergonomics validated on {len(pairs)} (pred, annotated) pairs (>=8)")
        if len(pairs) >= 3:
            from scipy.stats import spearmanr
            rho = spearmanr(pairs[:, 0], pairs[:, 1]).statistic
            mae = float(np.mean(np.abs(pairs[:, 0] - pairs[:, 1])))
            rel = mae / max(1e-9, float(np.mean(pairs[:, 1])))
            chk(st, rho >= 0.5 or rel <= 0.35,
                f"posture durations vs EAWS annotation: Spearman {rho:.2f} (>=0.5) or MAE {mae:.1f}s = {rel:.0%} (<=35%)")
    if ts:
        z = 1.959963984540054
        a = ts["params"]["accuracy"]
        allow = ts["params"]["allowance_pfd"]
        bad = 0
        for r in ts["element_stats"]:
            if r["n"] >= 2 and r["mean_s"] > 0:
                nreq = math.ceil((z * r["std_s"] / (a * r["mean_s"])) ** 2)
                bad += abs(nreq - r["n_required"]) > 1
            bad += abs(r["standard_s"] - r["normal_s"] * (1 + allow)) > 1e-3 * max(1, r["standard_s"])
        chk(st, bad == 0, f"time-study arithmetic re-derived (n_required, standard time): {bad} mismatches")
        chk(st, all(s["takt_s"] == CFG["line"]["takt_s"] for s in ts["station_stats"]), "takt taken from config")
        IE.s2_ie(ts, chk)


# ------------------------------------------------------------------ S3
def check_s3():
    st = "s3"
    au = load(st, "automatability.json", "automatability")
    if not au:
        return
    w = {c["key"]: c["weight"] for c in au["criteria"]}
    chk(st, abs(sum(w.values()) - 1) < 1e-6, f"criterion weights sum to 1 ({sum(w.values()):.4f})")
    chk(st, all(c["source"] for c in au["criteria"]), "every criterion cites a source")
    bad = 0
    for t in au["tasks"]:
        tot = sum(w[k] * t["scores"][k] for k in w)
        bad += abs(tot - t["total"]) > 1e-6 or not all(0 <= t["scores"][k] <= 1 for k in w)
    chk(st, bad == 0, f"task totals re-derived from weights x scores: {bad} mismatches")
    chk(st, all(t["allocation"] in ("robot", "human", "human_robot", "layout_change") for t in au["tasks"]), "allocations from allowed set")
    chk(st, "sensitivity" in au and au["sensitivity"].get("rank_stability", 0) >= 0.8,
        f"weight-perturbation rank stability {au.get('sensitivity', {}).get('rank_stability', 'missing')} (>=0.8)")
    chk(st, any(t["allocation"] in ("robot", "human_robot") for t in au["tasks"]), "at least one automation candidate found")


# ------------------------------------------------------------------ S4
def check_s4():
    st = "s4"
    cat = load(st, "robot_catalog.json", "robot_catalog")
    cand = load(st, "candidates.json", "candidates")
    if cat:
        R = cat["robots"]
        chk(st, len(R) >= 10, f"{len(R)} commercial models in catalog (>=10)")
        chk(st, all(any(u.startswith("http") for u in r["source_urls"]) for r in R), "every model cites a spec source URL")
        chk(st, sum(bool(r["sim_model"]) for r in R) >= 3, "at least 3 models have a simulation model attached")
    if cat and cand:
        by = {r["model"]: r for r in cat["robots"]}
        m = CFG["robots"]
        bad = 0
        for g in cand["task_groups"]:
            req = g["requirements"]
            passed = [e for e in g["evaluated"] if e["passed"]]
            chk(st, len(passed) >= 1, f"{g['group_id']}: >=1 model passes ({len(passed)})")
            for e in passed:
                r = by[e["model"]]
                bad += r["payload_kg"] < req["payload_kg"] * (1 + m["safety_margin_payload"]) - 1e-9
                bad += r["reach_mm"] * m["reach_margin"] < req["reach_mm"] - 1e-9
                if req.get("conveyor_tracking"):
                    bad += r["conveyor_tracking"] is not True
            chk(st, all(e["reasons"] for e in g["evaluated"]), f"{g['group_id']}: every pass/fail has reasons")
            chk(st, all(e["reachability"] is not None for e in passed), f"{g['group_id']}: reachability evaluated for passed models")
        chk(st, bad == 0, f"passed models re-checked against payload/reach/tracking requirements: {bad} violations")
        IE.s4_ie(cand, chk)


# ------------------------------------------------------------------ S6A
def check_s6a():
    st = "s6a"
    rt = load(st, "robot_task_times.json", "robot_task_times")
    cand = json.loads((OUT / "s4" / "candidates.json").read_text()) if (OUT / "s4" / "candidates.json").exists() else None
    if not rt:
        return
    chk(st, re.search(r"mujoco|ik", rt["method"], re.I) is not None, "times come from a kinematic simulation (MuJoCo/IK)")
    have = {(e["model"], e["group_id"]) for e in rt["entries"]}
    if cand:
        need = {(e["model"], g["group_id"]) for g in cand["task_groups"] for e in g["evaluated"] if e["passed"] and e.get("simulate", True)}
        chk(st, need <= have, f"every passed (model, task group) simulated ({len(need & have)}/{len(need)})")
    for e in rt["entries"]:
        tag = f"{e['model']}/{e['group_id']}"
        chk(st, 1 <= e["cycle_time_s"] <= CFG["line"]["takt_s"], f"{tag}: cycle {e['cycle_time_s']:.1f}s in [1, takt]")
        s = sum(e["breakdown"].values())
        chk(st, abs(s - e["cycle_time_s"]) <= 0.05 * e["cycle_time_s"], f"{tag}: breakdown sums to cycle time")
        chk(st, e["ik_success_rate"] >= 0.9, f"{tag}: IK success {e['ik_success_rate']:.0%} (>=90%)")


# ------------------------------------------------------------------ S5
def check_s5():
    st = "s5"
    al = load(st, "allocation.json", "allocation")
    if not al:
        return
    names = [s["name"] for s in al["scenarios"]]
    chk(st, "baseline_manual" in names, "baseline_manual scenario present")
    chk(st, sum(bool(s.get("recommended")) for s in al["scenarios"]) == 1, "exactly one recommended scenario")
    takt = CFG["line"]["takt_s"]
    for s in al["scenarios"]:
        # The as-is manual line is evaluated, not optimised: at standard time (rating x PFD allowance) it may
        # exceed takt, which is itself a finding. It must then report the overload; every other scenario
        # must be OPTIMAL and fit takt at standard time.
        base = s["name"] == "baseline_manual"
        if base:
            worst = max(t["cycle_s"] for t in s["station_times"].values())
            chk(st, abs(s.get("overload_pct", -99) - max(0.0, worst / takt - 1)) < 1e-3,
                f"baseline_manual: overload vs takt reported ({max(0.0, worst / takt - 1):.1%})")
        else:
            chk(st, s["status"] == "OPTIMAL", f"{s['name']}: solver status {s['status']}")
        recompute = {}
        for a in s["assignment"]:
            recompute.setdefault(a["station"], {"human": 0.0, "robot": 0.0})
            recompute[a["station"]][a["resource"] if a["resource"] in ("human", "robot") else "robot"] += a["time_s"]
        ok = True
        for stn, t in s["station_times"].items():
            r = recompute.get(stn, {"human": 0, "robot": 0})
            ok &= abs(r["human"] - t["human_s"]) < 1e-3 and abs(r["robot"] - t["robot_s"]) < 1e-3
            ok &= (base or t["cycle_s"] <= takt + 1e-6) and t["cycle_s"] >= max(t["human_s"], t["robot_s"]) - 1e-6
        chk(st, ok, f"{s['name']}: station times re-derived from assignment" + ("" if base else f" and all <= takt {takt}s"))
        ids = [a["element_key"] for a in s["assignment"]]
        chk(st, len(ids) == len(set(ids)), f"{s['name']}: every work element assigned exactly once")
        chk(st, all(s["checks"].values()), f"{s['name']}: solver post-checks {s['checks']}")
    IE.s5_ie(al, takt, chk)
    # A model we tell the client to buy must have been simulated with its own kinematics, not a scaled sibling,
    # and must carry a sourced price range.
    rt_p = OUT / "s6a" / "robot_task_times.json"
    cat_p = OUT / "s4" / "robot_catalog.json"
    rec = [x for x in al["scenarios"] if x.get("recommended")]
    if rec and rt_p.exists() and cat_p.exists():
        rt = json.loads(rt_p.read_text())["entries"]
        cat = {r["model"]: r for r in json.loads(cat_p.read_text())["robots"]}
        for rb in rec[0]["robots"]:
            for g in rb.get("groups", []):
                e = [x for x in rt if x["model"] == rb["model"] and x["group_id"] == g]
                own = e and "surrogate" not in json.dumps(e[0]["sim"]).lower()
                chk(st, bool(own), f"recommended {rb['model']} @ {g}: simulated with its own kinematic model")
            pr = cat.get(rb["model"], {}).get("price_usd_range", [0, 0])
            chk(st, pr[1] > pr[0] > 0, f"recommended {rb['model']}: price is a sourced range {pr}")
            # Hidden costs the s4 placement implies must be priced in.
            cand_p = OUT / "s4" / "candidates.json"
            if cand_p.exists():
                mount = {g["group_id"]: g.get("mounting") for g in json.loads(cand_p.read_text())["task_groups"]}
                keys = " ".join(rb.get("capex_breakdown_usd", {})).lower()
                if any(mount.get(g) == "linear_track" for g in rb.get("groups", [])):
                    chk(st, "track" in keys, f"recommended {rb['model']} @ {rb['station']}: linear-track capex counted")
                if len(set(rb.get("categories", []))) > 1:
                    chk(st, "changer" in keys, f"recommended {rb['model']} @ {rb['station']}: tool-changer capex counted")


# ------------------------------------------------------------------ S6B
def check_s6b():
    st = "s6b"
    des = load(st, "des_results.json", "des_results")
    if not des:
        return
    chk(st, des["replications"] >= 30, f"{des['replications']} replications (>=30)")
    for s in des["scenarios"]:
        rel = s["ci95_half_width"] / max(1e-9, s["throughput_per_hr"])
        chk(st, rel <= 0.05, f"{s['name']}: 95% CI half-width {rel:.1%} of throughput (<=5%)")
        chk(st, all(0 <= u <= 1 for u in s["utilization"].values()), f"{s['name']}: utilizations in [0,1]")
    v = des["validation"]
    err = abs(v["des_baseline_throughput"] - v["analytic_baseline_throughput"]) / v["analytic_baseline_throughput"]
    chk(st, err <= 0.10, f"baseline DES vs analytic throughput differ {err:.1%} (<=10%)")
    IE.s6b_ie(des, chk)


# ------------------------------------------------------------------ S7
def check_s7():
    st = "s7"
    roi = load(st, "roi.json", "roi")
    if not roi:
        return
    wage = [a for a in roi["assumptions"] if "wage" in a["name"].lower()]
    chk(st, wage and "bls.gov" in wage[0]["source"], "hourly wage assumption cites BLS")
    for s in roi["scenarios"]:
        if s["annual_savings_usd"] > 0:
            pb = s["capex_usd"] / s["annual_savings_usd"]
            chk(st, abs(pb - s["payback_years"]) <= 0.01 * max(1, pb), f"{s['name']}: payback re-derived {pb:.2f}y")
    chk(st, len(roi["sensitivity"]) >= 3, f"{len(roi['sensitivity'])} sensitivity parameters (>=3)")
    IE.s7_ie(roi, chk)


# ------------------------------------------------------------------ S5X
def check_s5x():
    """Design space: re-derive the factorial size, Pareto sets, dominance claims, SMAA shares and the links to
    s5/s6b/s7 from the stored metrics instead of trusting the flags."""
    st = "s5x"
    ds = load(st, "design_space.json", "design_space")
    al = json.loads((OUT / "s5" / "allocation.json").read_text()) if (OUT / "s5" / "allocation.json").exists() else None
    if not ds or not al:
        chk(st, bool(al), "s5 allocation.json present")
        return
    lv = ds["factors"]["arm_per_station"]
    combos = int(np.prod([len(v) for v in lv.values()]))
    chk(st, combos == ds["counts"]["arm_combinations"], f"arm combinations re-derived {combos}")
    nf = len(ds["factors"]["layout_levers"]) * len(ds["factors"]["pacing"])
    exp = nf * (len(ds["factors"]["task_split"]) * (combos - 1) + 1)   # no-arm combo: the split has nothing to decide
    chk(st, len(ds["runs"]) == exp == ds["counts"]["factor_runs"], f"full factorial: {len(ds['runs'])} runs (expected {exp})")
    D = {d["id"]: d for d in ds["designs"]}
    chk(st, len(D) >= combos, f"{len(D)} distinct designs (>= {combos} arm combinations: different arms are different designs)")
    takt = ds["takt_s"]
    chk(st, all(d["metrics"]["max_cycle_s"] <= takt + 1e-6 for d in D.values()), "every design meets takt at standard time")

    def dom(a, b, crit):
        ge = all((a[k] >= b[k] - 1e-9) if s == "max" else (a[k] <= b[k] + 1e-9) for k, s in crit)
        return ge and any((a[k] > b[k] + 1e-9) if s == "max" else (a[k] < b[k] - 1e-9) for k, s in crit)
    sc = [(c["key"], c["sense"]) for c in ds["criteria_screen"]]
    M = list(D.values())
    front = {d["id"] for d in M if not any(dom(o["metrics"], d["metrics"], sc) for o in M if o is not d)}
    flag = {d["id"] for d in M if d["pareto_screen"]}
    chk(st, front == flag, f"screen Pareto set re-derived ({len(front)} designs, flags agree)")
    bad = [d["id"] for d in M if d["dominated_by"] and not dom(D[d["dominated_by"]]["metrics"], d["metrics"], sc)]
    chk(st, not bad, f"every 'dominated_by' claim holds on all screen criteria ({len(bad)} violations)")
    sim = [d for d in M if d.get("simulated")]
    fc = [(c["key"], c["sense"]) for c in ds["criteria"]]
    ff = {d["id"] for d in sim if not any(dom(o["metrics"], d["metrics"], fc) for o in sim if o is not d)}
    chk(st, ff == {d["id"] for d in sim if d.get("pareto_final")}, f"final (simulated) Pareto set re-derived ({len(ff)})")
    chk(st, front <= {d["id"] for d in sim} or len(sim) >= 120, "every screen-Pareto design was simulated")
    r1 = sum(D[i]["smaa"]["rank1"] for i in ff)
    chk(st, abs(r1 - 1) < 1e-6, f"SMAA rank-1 acceptabilities sum to {r1:.6f}")
    chk(st, all(p["best"] in ff for p in ds["personas"]), "every stakeholder profile's pick is on the final front")
    for p in ds["personas"]:
        chk(st, abs(sum(p["weights"].values()) - 1) < 1e-9, f"profile '{p['name']}' weights sum to 1")
    ok = True
    for f in ds["featured_comparisons"]:
        a, b = D[f["a"]], D[f["b"]]
        for r in f["rows"]:
            sense = next(c["sense"] for c in ds["criteria"] if c["key"] == r["criterion"])
            x, y = a["metrics"][r["criterion"]], b["metrics"][r["criterion"]]
            w = "tie" if abs(x - y) <= 1e-9 * max(1, abs(x), abs(y)) else ("a" if ((x > y) == (sense == "max")) else "b")
            ok &= w == r["winner"]
        ok &= f["p_a_preferred"] is None or 0 <= f["p_a_preferred"] <= 1
    chk(st, ok, f"{len(ds['featured_comparisons'])} A-vs-B comparisons: per-criterion winners re-derived")
    H = ds.get("headline")
    if chk(st, bool(H), "headline options present"):
        op = H["options"]
        arch = sorted({D[i]["metrics"]["n_robots"] for i in ff if not D[i]["relief_operators_per_shift"]
                       and D[i]["metrics"]["npv_usd"] > 0})
        chk(st, len(op) == min(H["k"], len(arch)) and H["k"] <= 3,
            f"{len(op)} headline options (k={H['k']}, {len(arch)} practical archetypes)")
        chk(st, len({o["archetype_robots"] for o in op}) == len(op), "headline options are distinct archetypes (robot counts)")
        prac = all(o["design"] in ff and not D[o["design"]]["relief_operators_per_shift"]
                   and (D[o["design"]]["metrics"]["npv_usd"] > 0 or "NPV" not in H["practical_rule"]) for o in op)
        chk(st, prac, "every headline option is on the final front, needs no relief operator and has NPV > 0")
        chk(st, all(o["archetype_robots"] == D[o["design"]]["metrics"]["n_robots"] for o in op),
            "headline archetype = robot count of its design")
        cov = sum(o["share_of_weightings"] for o in op)
        chk(st, abs(cov - H["coverage"]) < 1e-6 and 0 < cov <= 1 + 1e-9,
            f"coverage {H['coverage']:.1%} = sum of the shown options' shares")
        pn = {p["name"] for p in ds["personas"]}
        chk(st, all(set(o["personas"]) <= pn for o in op), "headline personas are defined profiles")
    names = {s["name"] for s in al["scenarios"] if s["name"] != "baseline_manual"}
    mapped = {n for d in M for n in d["named_scenarios"]}
    chk(st, names <= mapped, f"every s5 scenario located in the design space ({len(names & mapped)}/{len(names)})")
    rec = ds.get("recommended_design")
    chk(st, rec in D and D[rec].get("simulated"), f"s5 recommendation is design {rec} and was simulated")
    roi = OUT / "s7" / "roi.json"
    if rec in D and roi.exists():
        rr = next((s for s in json.loads(roi.read_text())["scenarios"] if s.get("recommended")), None)
        if rr:
            chk(st, abs(rr["npv_usd"] - D[rec]["metrics"]["npv_usd"]) < 1.0,
                f"recommended design NPV matches s7 ({D[rec]['metrics']['npv_usd']:,.0f} vs {rr['npv_usd']:,.0f})")
    desp = OUT / "s6b" / "des_results.json"
    if rec in D and desp.exists():
        dr = next((s for s in json.loads(desp.read_text())["scenarios"] if s.get("recommended")), None)
        if dr:
            dd = D[rec]["des"]
            gap = abs(dd["throughput_per_hr"] - dr["throughput_per_hr"])
            tol = 2 * math.hypot(dd["ci95_half_width"], dr["ci95_half_width"])
            chk(st, gap <= tol, f"recommended design DES throughput reproduces s6b with other seeds "
                                f"({dd['throughput_per_hr']:.2f} vs {dr['throughput_per_hr']:.2f} /h, tol {tol:.2f})")


# ------------------------------------------------------------------ S8
def check_s8():
    st = "s8"
    p = OUT / "s8" / "report.html"
    if not chk(st, p.exists(), "report.html exists"):
        return
    html = p.read_text()
    chk(st, len(html) > 50_000, f"report size {len(html) // 1024} KB (>50 KB, has embedded figures)")
    ext = re.findall(r'<(?:script|link|img)[^>]+(?:src|href)=["\']https?://', html)
    chk(st, not ext, f"self-contained: {len(ext)} external script/style/img references")
    chk(st, 'id="rcd-data"' in html, "report embeds the stage JSON it was rendered from")
    for k in ["s0", "s1", "s2", "s3", "s4", "s5", "s6a", "s6b", "s7", "s9"]:
        chk(st, f'id="sec-{k}"' in html, f"section sec-{k} present")
    if (OUT / "s5x" / "design_space.json").exists():
        chk(st, 'id="sec-s5x"' in html and 'id="ds-app"' in html, "design-space section and explorer present")
        chk(st, html.count("class='hl-grid'") >= 2, "headline option cards in the summary and the design-space section")
    rfq = list((OUT / "s8" / "rfq").glob("*.md"))
    chk(st, len(rfq) >= 1, f"{len(rfq)} RFQ package(s) generated")


# ------------------------------------------------------------------ S9
def check_s9():
    st = "s9"
    tp = load(st, "task_plan.json", "task_plan")
    al = json.loads((OUT / "s5" / "allocation.json").read_text()) if (OUT / "s5" / "allocation.json").exists() else None
    if not tp:
        return
    prims = set(tp["primitives"])
    for pl in tp["plans"]:
        try:
            ET.fromstring(pl["bt_xml"])
            chk(st, True, f"{pl['station']}: behavior tree XML parses")
        except ET.ParseError as e:
            chk(st, False, f"{pl['station']}: BT XML invalid ({e})")
        chk(st, all(s["primitive"] in prims for s in pl["steps"]), f"{pl['station']}: steps use the fixed primitive vocabulary")
    if al:
        rec = [s for s in al["scenarios"] if s.get("recommended")]
        if rec:
            need = {a["element_key"] for a in rec[0]["assignment"] if a["resource"] not in ("human", "helper")}
            got = {k for pl in tp["plans"] for s in pl["steps"] for k in s.get("covers", [])}
            chk(st, need <= got, f"plan covers every robot-assigned element ({len(need & got)}/{len(need)})")


CHECKS = {"s0": check_s0, "s1": check_s1, "s2": check_s2, "s3": check_s3, "s4": check_s4, "s6a": check_s6a,
          "s5": check_s5, "s6b": check_s6b, "s7": check_s7, "s5x": check_s5x, "s9": check_s9, "s8": check_s8}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", nargs="*", default=list(CHECKS))
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    for s in a.stage:
        try:
            CHECKS[s]()
        except Exception as e:  # a crash in a check is a failure of that stage
            chk(s, False, f"check crashed: {type(e).__name__}: {e}")
    failed = []
    for s in a.stage:
        rows = RESULTS.get(s, [(False, "no checks ran")])
        ok = all(r[0] for r in rows)
        failed += [] if ok else [s]
        print(f"\n[{'PASS' if ok else 'FAIL'}] {s}")
        for good, msg in rows:
            if not good or not ok:
                print(f"   {'ok ' if good else 'XX '} {msg}")
    print(f"\nSUMMARY: {len(a.stage) - len(failed)}/{len(a.stage)} stages pass" + (f"; failing: {failed}" if failed else ""))
    if a.json:
        (OUT / "validation.json").write_text(json.dumps(
            {s: [{"ok": g, "msg": m} for g, m in RESULTS.get(s, [])] for s in a.stage}, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
