"""Generalisation gate. Compares the video-only / SOP-only pipeline against the original
mocap + timed-annotation pipeline on CarDA, and checks a held-out recording.

    RCD_CONFIG=config/demo_regress.yaml   python scripts/run_all.py --config config/demo_regress.yaml   # reference
    python scripts/run_all.py --config config/carda_generic.yaml                                     # candidate
    python tests/validate_generic.py [--g G1 G2 ...]

G1 video 3D pose vs Xsens mocap      G2 SOP-only segmentation vs annotation   G3 end-to-end agreement
G4 generic conveyor detection        G5 no dataset constants (intake-driven) G6 held-out subset B runs + passes
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rcd.common.io import load_config  # noqa: E402

REF = load_config(ROOT / "config/demo_regress.yaml")
GEN = load_config(ROOT / "config/carda_generic.yaml")
RES: dict[str, list[tuple[bool, str]]] = {}


def chk(g, ok, msg):
    RES.setdefault(g, []).append((bool(ok), msg))
    return bool(ok)


def J(cfg, *p):
    f = ROOT / cfg["out_root"] / Path(*p)
    return json.loads(f.read_text()) if f.exists() else None


def cycles(cfg):
    sg = J(cfg, "s1", "scene_graph.json")
    return {(s["station"], c["cycle_id"]): c for s in (sg or {"stations": []})["stations"] for c in s["cycles"]}


def resample(t_src, x, t_dst):
    return np.stack([np.interp(t_dst, t_src, x[:, k]) for k in range(x.shape[1])], 1)


def trunk_flex(P, names):
    v = P[:, names.index("Neck")] - P[:, names.index("Hips")]
    return np.degrees(np.arccos(np.clip(v[:, 2] / np.linalg.norm(v, axis=1), -1, 1)))


def kabsch2d(a, b):
    """Best rigid 2D transform of a onto b; returns residual distances."""
    ma, mb = a.mean(0), b.mean(0)
    H = (a - ma).T @ (b - mb)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    return np.linalg.norm((a - ma) @ R.T + mb - b, axis=1)


# ---------------------------------------------------------------- G1 video pose vs mocap
def g1():
    g = "G1"
    ref, gen = cycles(REF), cycles(GEN)
    chk(g, gen and set(gen) == set(ref), f"video run covers the same {len(ref)} cycles as mocap ({len(gen)})")
    rs, maes, pel, wr, cov, srcs = [], [], [], [], [], set()
    for k in sorted(set(ref) & set(gen)):
        if "video" not in json.dumps(gen[k]["human"]["source"]).lower():
            chk(g, False, f"{k}: human source is not video ({gen[k]['human']['source']})")
            continue
        srcs.add(gen[k]["human"].get("views", "?"))
        a, b = np.load(ROOT / ref[k]["human"]["joints_npz"]), np.load(ROOT / gen[k]["human"]["joints_npz"])
        na, nb = list(a["names"]), list(b["names"])
        ta = np.arange(len(a["joints"])) / float(a["fps"])
        tb = np.arange(len(b["joints"])) / float(b["fps"])
        valid = b["valid"].astype(bool) if "valid" in b.files else np.ones(len(tb), bool)
        cov.append(valid.mean())
        T = tb[valid & (tb <= ta[-1])]
        if len(T) < 30:
            continue
        fa = np.interp(T, ta, trunk_flex(a["joints"], na))
        fb = trunk_flex(b["joints"], nb)[valid & (tb <= ta[-1])]
        rs.append(np.corrcoef(fa, fb)[0, 1])
        maes.append(np.mean(np.abs(fa - fb)))
        pa = resample(ta, a["joints"][:, na.index("Hips"), :2], T)
        pb = b["joints"][valid & (tb <= ta[-1]), nb.index("Hips"), :2]
        pel.append(np.median(kabsch2d(pb, pa)))
        sp = []
        for side in ("Right", "Left"):
            wa = resample(ta, a["joints"][:, na.index(f"{side}Wrist")], T)
            wb = b["joints"][valid & (tb <= ta[-1]), nb.index(f"{side}Wrist")]
            va = np.linalg.norm(np.gradient(wa, T, axis=0), axis=1)
            vb = np.linalg.norm(np.gradient(wb, T, axis=0), axis=1)
            k5 = np.ones(5) / 5
            from scipy.stats import spearmanr
            sp.append(spearmanr(np.convolve(va, k5, "same"), np.convolve(vb, k5, "same")).statistic)
        wr.append(np.nanmean(sp))
    if rs:
        chk(g, np.mean(cov) >= 0.85, f"frames with a valid pose: {np.mean(cov):.0%} (>=85%)")
        chk(g, np.median(rs) >= 0.70, f"trunk flexion vs mocap: median Pearson r {np.median(rs):.2f} (>=0.70)")
        chk(g, np.median(maes) <= 10.0, f"trunk flexion vs mocap: median MAE {np.median(maes):.1f} deg (<=10)")
        chk(g, np.median(pel) <= 0.35, f"pelvis floor path vs mocap (rigid-aligned): median {np.median(pel):.2f} m (<=0.35)")
        chk(g, np.median(wr) >= 0.50, f"wrist speed vs mocap: median Spearman {np.median(wr):.2f} (>=0.50)")
    chk(g, bool(srcs), f"pose views used: {sorted(srcs)}")


# ---------------------------------------------------------------- G2 SOP-only segmentation
def _validator(cfg_rel):
    """Import tests/validate.py bound to a given config (its CFG is read at import time)."""
    import importlib.util
    import os
    old, sys.argv = os.environ.get("RCD_CONFIG"), ["x"]
    os.environ["RCD_CONFIG"] = cfg_rel
    try:
        spec = importlib.util.spec_from_file_location(f"v_{cfg_rel}", ROOT / "tests/validate.py")
        v = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(v)
    finally:
        os.environ.pop("RCD_CONFIG") if old is None else os.environ.__setitem__("RCD_CONFIG", old)
    return v


def boundary_errors(el, ann):
    errs = []
    for c in el["cycles"]:
        for name, s0, e0 in ann.get((c["station"], c["cycle_id"]), []):
            mine = [e for e in c["elements"] if (e.get("subgoal") or "").strip() == name]
            errs += [abs(min(e["start_s"] for e in mine) - s0), abs(max(e["end_s"] for e in mine) - e0)] if mine else [99.0, 99.0]
    return np.asarray(errs)


def frame_agreement(ref, cand):
    agree, tot = 0, 0
    R = {(c["station"], c["cycle_id"]): c for c in ref["cycles"]}
    for c in cand["cycles"]:
        r = R.get((c["station"], c["cycle_id"]))
        if not r:
            continue
        t = np.arange(0, c["duration_s"], 0.1)
        lab = lambda E: np.array([next((e["code"] for e in E if e["start_s"] <= x < e["end_s"]), "-") for x in t])
        a, b = lab(r["elements"]), lab(c["elements"])
        agree += (a == b).sum()
        tot += len(t)
    return agree / tot if tot else 0.0


def g2():
    """G2a: SOP text only (no times at all)        -> reported, not gated (see docs/GENERALIZATION.md)
       G2b: SOP + standard-work-sheet times, emulated leave-one-cycle-out from the OTHER cycles -> gated"""
    g = "G2"
    ann = _validator("config/carda_generic.yaml").annotated_subgoals()
    el = J(GEN, "s2", "elements.json")
    if not chk(g, el is not None, "generic s2 elements exist"):
        return
    meta = el.get("method", {})
    chk(g, meta.get("segmentation") == "sop" and not meta.get("used_annotation_times", True),
        f"segmentation declared SOP-only without annotation timestamps ({meta.get('segmentation')}, used_annotation_times={meta.get('used_annotation_times')})")
    e = boundary_errors(el, ann)
    chk(g, True, f"[G2a info] text-only SOP, video: median boundary error {np.median(e):.1f}s, {np.mean(e <= 5):.0%} within 5s (n={len(e)})")
    elr = J(REF, "s2", "elements.json")
    gm = J(GEN, "s2", "elements_mocap_sop.json")  # SOP-only segmentation on mocap joints (isolates segmentation from pose error)
    if chk(g, gm is not None, "SOP-only segmentation on mocap joints provided (elements_mocap_sop.json)"):
        em = boundary_errors(gm, ann)
        chk(g, True, f"[G2a info] text-only SOP, mocap: median boundary error {np.median(em):.1f}s, {np.mean(em <= 5):.0%} within 5s")
        chk(g, frame_agreement(elr, gm) >= 0.60, f"text-only SOP on mocap: frame agreement with annotation-guided elements {frame_agreement(elr, gm):.0%} (>=60%)")
    for cfg, rel, label in ((GEN, "config/carda_generic.yaml", "A video"), (load_config(ROOT / "config/carda_b.yaml"), "config/carda_b.yaml", "B video (held-out)")):
        es = J(cfg, "s2", "elements_sop_std.json")
        if not chk(g, es is not None, f"[G2b] {label}: SOP + standard-time segmentation provided (elements_sop_std.json)"):
            continue
        chk(g, es.get("method", {}).get("std_times") == "loo", f"[G2b] {label}: standard times are leave-one-cycle-out ({es.get('method', {}).get('std_times')})")
        eb = boundary_errors(es, ann if cfg is GEN else _validator(rel).annotated_subgoals())
        chk(g, len(eb) and np.median(eb) <= 5.0, f"[G2b] {label}: median boundary error {np.median(eb) if len(eb) else -1:.1f}s (<=5s), {np.mean(eb <= 5):.0%} within 5s")
    gms = J(GEN, "s2", "elements_mocap_sop_std.json")
    if gms:
        eb = boundary_errors(gms, ann)
        chk(g, True, f"[G2b info] mocap + std times: median boundary error {np.median(eb):.1f}s, frame agreement {frame_agreement(elr, gms):.0%}")
    rq = J(GEN, "s2", "review_queue.json")
    if chk(g, rq is not None, "IE review queue produced (s2/review_queue.json)"):
        ents = rq.get("entries", rq if isinstance(rq, list) else [])
        flagged = sum(bool(x.get("needs_review")) for x in ents)
        chk(g, ents and 0.1 <= flagged / len(ents) <= 0.4, f"review queue flags {flagged}/{len(ents)} boundaries for IE review")


# ---------------------------------------------------------------- G3 end-to-end agreement
def g3():
    g = "G3"
    a, b = J(REF, "s2", "time_study.json"), J(GEN, "s2", "time_study.json")
    if not chk(g, a and b, "both time studies exist"):
        return
    A = {s["station"]: s for s in a["station_stats"]}
    for s in b["station_stats"]:
        ref = A[s["station"]]["observed_work_mean_s"]
        d = abs(s["observed_work_mean_s"] - ref) / ref
        chk(g, d <= 0.15, f"{s['station']}: observed work content video-only {s['observed_work_mean_s']:.1f}s vs mocap {ref:.1f}s ({d:.0%} <=15%)")
    ra, rb = J(REF, "s3", "automatability.json"), J(GEN, "s3", "automatability.json")
    if ra and rb:
        top = lambda r: [f"{t['station']}:{t['code']}" for t in sorted(r["tasks"], key=lambda t: -t["total"]) if t["allocation"] in ("robot", "human_robot")][:3]
        ov = len(set(top(ra)) & set(top(rb)))
        chk(g, ov >= 2, f"top-3 automation candidates overlap {ov}/3 (>=2): ref {top(ra)} vs video {top(rb)}")
    aa, ab = J(REF, "s5", "allocation.json"), J(GEN, "s5", "allocation.json")
    if aa and ab:
        rec = lambda x: sorted((r["station"], r["model"]) for s in x["scenarios"] if s.get("recommended") for r in s["robots"])
        chk(g, True, f"recommendation ref {rec(aa)} vs video-only {rec(ab)} (informational)")


# ---------------------------------------------------------------- G4 generic conveyor
def g4():
    g = "G4"
    ref, gen = cycles(REF), cycles(GEN)
    errs = []
    for k in sorted(set(ref) & set(gen)):
        m = gen[k]["conveyor"]["method"].lower()
        chk(g, "yellow" not in m, f"{k}: conveyor found without the CarDA yellow-strip rule")
        errs.append(abs(gen[k]["conveyor"]["speed_mm_s"] - ref[k]["conveyor"]["speed_mm_s"]) / ref[k]["conveyor"]["speed_mm_s"])
    chk(g, errs and np.median(errs) <= 0.15 and max(errs) <= 0.30,
        f"conveyor speed vs reference: median {np.median(errs) if errs else -1:.0%} (<=15%), max {max(errs) if errs else -1:.0%} (<=30%)")
    sg = J(GEN, "s1", "scene_graph.json")
    if sg:
        chk(g, all(any(z["name"] == "conveyor" for z in s["zones"]) for s in sg["stations"]), "conveyor zone detected for every station")


# ---------------------------------------------------------------- G5 no dataset constants
def g5():
    g = "G5"
    au = J(GEN, "s3", "automatability.json")
    if au:
        src = [t.get("requirements", {}).get("part_source") for t in au["tasks"] if t["allocation"] in ("robot", "human_robot")]
        chk(g, src and all(s == "intake" for s in src), f"part masses come from the customer intake ({set(src)})")
    for f in ["s1_scene", "s2_elements", "s3_automatability"]:
        code = "\n".join(p.read_text() for p in (ROOT / "src/rcd" / f).glob("*.py"))
        # dataset constants may remain only behind the original-mode branch, never imported by generic modules
        gen_files = [p for p in (ROOT / "src/rcd" / f).glob("*.py") if "generic" in p.name or "video" in p.name or "sop" in p.name]
        bad = [p.name for p in gen_files if re.search(r"speaker|rearview|tie rod|weather strip|yellow|CarDA", p.read_text(), re.I)]
        chk(g, not bad, f"{f}: generic modules free of CarDA-specific constants ({bad or 'ok'})")
    el = J(GEN, "s2", "elements.json")
    if el:
        chk(g, el.get("method", {}).get("intake") == GEN["intake"], "s2 records which intake it used")


# ---------------------------------------------------------------- G6 held-out subset B
def g6():
    g = "G6"
    B = load_config(ROOT / "config/carda_b.yaml")
    out = ROOT / B["out_root"]
    if not chk(g, (out / "s8" / "report.html").exists(), "held-out subset B pipeline produced a report"):
        return
    r = subprocess.run([sys.executable, str(ROOT / "tests/validate.py"), "--stage", "s1", "s2", "s3", "s4", "s6a", "s5", "s6b", "s7", "s9", "s8"],
                       env={**__import__("os").environ, "RCD_CONFIG": "config/carda_b.yaml"}, capture_output=True, text=True)
    last = [l for l in r.stdout.splitlines() if l.startswith("SUMMARY")]
    chk(g, r.returncode == 0, f"standard gate on held-out recordings: {last[0] if last else r.stdout[-300:]}")
    sg = json.loads((out / "s1" / "scene_graph.json").read_text())
    chk(g, all("video" in json.dumps(c["human"]["source"]).lower() for s in sg["stations"] for c in s["cycles"]),
        "held-out run used video only (no mocap exists for subset B)")


G = {"G1": g1, "G2": g2, "G3": g3, "G4": g4, "G5": g5, "G6": g6}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g", nargs="*", default=list(G))
    a = ap.parse_args()
    for k in a.g:
        try:
            G[k]()
        except Exception as e:
            chk(k, False, f"check crashed: {type(e).__name__}: {e}")
    fail = []
    for k in a.g:
        rows = RES.get(k, [(False, "no checks ran")])
        ok = all(r[0] for r in rows)
        fail += [] if ok else [k]
        print(f"\n[{'PASS' if ok else 'FAIL'}] {k}")
        for good, msg in rows:
            print(f"   {'ok ' if good else 'XX '} {msg}")
    print(f"\nSUMMARY: {len(a.g) - len(fail)}/{len(a.g)} generalisation gates pass" + (f"; failing: {fail}" if fail else ""))
    (ROOT / "out_generic").mkdir(exist_ok=True)
    (ROOT / "out_generic" / "validation_generic.json").write_text(json.dumps({k: [{"ok": x, "msg": m} for x, m in RES.get(k, [])] for k in a.g}, indent=2))
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
