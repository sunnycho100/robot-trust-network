"""Simulation packages for the three headline options of s5x (A / B / C), one folder per option.

    python scripts/export_option_packages.py [--config config/demo_regress.yaml]

For every headline option this builds a staging pipeline root out_options/<A|B|C>/ that reads the upstream stage
outputs of the config (s0-s4, s6a via symlinks) and holds its own s5 (the option marked as the recommended scenario)
and s9 (robot task plan regenerated for that option). The existing hand-off exporter (rcd.sim_package.export) is then
run unchanged on that root, followed by the baseline replay of every robot station (worker stays / leaves).
An option without robots gets a worker-only package (observed and to-be worker motion per station).
Writes out_options/manifest.json. Nothing is written to out/.
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rcd.common.io import load_config, load_json  # noqa: E402
from rcd.s5_optimize.allocation import AllocModel  # noqa: E402
from rcd.s5_optimize.data import build  # noqa: E402
from rcd.s5_optimize.inputs import load_input  # noqa: E402
from rcd.s9_vla import run as s9  # noqa: E402
from rcd.sim_package import export as X  # noqa: E402

OUT = ROOT / "out_options"
LINKED = ("s0", "s1", "s2", "s3", "s4", "s6a", "s6b", "s7")


def scenario_for(cfg, alloc, d):
    """The s5 scenario of a design: the named s5 scenario if there is one, else re-solved from its first variant."""
    named = [s for s in alloc["scenarios"] if s["name"] in d["named_scenarios"]]
    if named:
        return copy.deepcopy(named[0])
    v = next(x for x in d["variants"] if "split" in x)
    p = {}
    data = build(cfg, *[load_input(cfg, n, p) for n in ("time_study", "elements", "automatability", "robot_catalog",
                                                         "candidates", "robot_task_times")])
    am = AllocModel(cfg, data)
    cap = 0.95 * data["takt"] if v["pacing_95"] else None
    s = am.solve(d["id"], v["split"], layout=v["layout"], cycle_cap=cap, fix_models=d["arms"], helpers=v["relief"])
    s["name"] = d["id"]
    return s


def worker_only(cfg, sd, scen, rep_cycles):
    """No-robot option: observed and to-be (kitting-compressed) worker motion per station + its timing."""
    el = load_json(ROOT / cfg["out_root"] / "s2" / "elements.json")
    sg = load_json(ROOT / cfg["out_root"] / "s1" / "scene_graph.json")
    walk = {lc["station"]: 1 - (lc.get("reduction") or 0) for lc in scen.get("layout_changes", []) if lc.get("elements")}
    index = []
    for st, cyc in rep_cycles.items():
        d = sd / st
        d.mkdir(parents=True, exist_ok=True)
        c = [c for s in sg["stations"] if s["station"] == st for c in s["cycles"] if c["cycle_id"] == cyc][0]
        npz = np.load(ROOT / c["human"]["joints_npz"])
        J, names, fps = npz["joints"], [str(x) for x in npz["names"]], float(npz["fps"])
        np.savez_compressed(d / "human_asis.npz", joints=J, names=np.array(names), fps=fps)
        els, clip = X._segments(el, st, cyc)
        codes = {a["code"] for a in scen["assignment"] if a["station"] == st and a["resource"] == "human"} - {"UNOBS"}
        Jt, rows = X._human_tobe(J, fps, els, codes, walk.get(st, 1.0))
        np.savez_compressed(d / "human_tobe.npz", joints=Jt, names=np.array(names), fps=fps)
        unobs = [a["time_s"] for a in scen["assignment"] if a["element_key"] == f"{st}:UNOBS"]
        (d / "human_tobe_timeline.json").write_text(json.dumps({
            "station": st, "source_cycle": cyc, "fps": fps, "spliced_duration_s": round(len(Jt) / fps, 2),
            "segments": rows, "walk_time_factor": walk.get(st, 1.0), "unobserved_block_s": unobs[0] if unobs else 0.0,
            "station_times": scen["station_times"][st], "layout_levers": [l for l in scen.get("layout_changes", [])
                                                                          if l["station"] in (st, "LINE")],
            "note": "No robot in this option. Same frame and splice conventions as the robot packages; use the scene of "
                    "option C for this station with the robot, track and trays removed."}, indent=2))
        index.append({"station": st, "model": None, "dir": str(d.relative_to(ROOT))})
    (sd / "index.json").write_text(json.dumps(index, indent=2))


def options_matrix(m) -> str:
    """Run matrix for the three options (the per-package scenario_matrix.yaml keeps the v2 names; this one rules)."""
    import yaml
    takt = m["takt_s"]
    doc = {"note": "Option letters follow the s5x headline (A = no robot, B = 1 robot, C = 3 robots). The "
                   "scenario_matrix.yaml copied inside each package is the v2 file: its A_recommended = option C here, "
                   "its B_min_cost = option B here. SSM parameters, KPI definitions and randomization are unchanged.",
           "takt_s": takt, "station_cycle_cap_s": 0.95 * takt, "draws_per_cell": 30, "options": {}}
    for o in m["options"]:
        robots = {r["station"]: {"model": r["model"], "does": r["tasks"]} for r in o["robots"]}
        cell = {"design": o["design"], "s5_scenario": o["scenario"], "package": o["package"], "robots": robots or "none",
                "layout": sorted({x.split("@")[0] for x in o["layout_levers"]}),
                "station_cycle_s_ie": {st: round(v["cycle_s"], 1) for st, v in o["station_times"].items()},
                "ie_prediction": {"throughput_per_hr": round(o["metrics"]["throughput_per_hr"], 2),
                                  "takt_overrun_pct_station_cycles": round(o["metrics"]["takt_violation_pct"], 1)}}
        if robots:
            cell["vary"] = {"policy": ["P1_parallel_ssm", "P2_zone_exclusive", "P3_sequential"],
                            "worker_after_motion": ["hold", "away"]}
            cell["runs"] = len(robots) * 3 * 2 * 30
        else:
            cell["vary"] = {"walk_time_factor": [0.5, 0.7, 1.0]}
            cell["runs"] = len(o["station_times"]) * 3 * 30
        doc["options"][o["letter"]] = cell
    doc["total_runs"] = sum(c["runs"] for c in doc["options"].values())
    return yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=110)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/demo_regress.yaml")
    a = ap.parse_args()
    cfg = load_config(ROOT / a.config)
    src = ROOT / cfg["out_root"]
    ds = load_json(src / "s5x" / "design_space.json")
    alloc = load_json(src / "s5" / "allocation.json")
    by = {d["id"]: d for d in ds["designs"]}
    if OUT.exists():
        shutil.rmtree(OUT)
    py = sys.executable
    manifest = {"source_config": a.config, "source_out_root": cfg["out_root"], "takt_s": cfg["line"]["takt_s"],
                "headline_coverage": ds["headline"]["coverage"], "options": []}
    opts = sorted(ds["headline"]["options"], key=lambda o: o["archetype_robots"])
    rep_cycles = {}
    for letter, o in zip("ABC", opts):
        d = by[o["design"]]
        scen = scenario_for(cfg, alloc, d)
        for s in alloc["scenarios"]:
            s["recommended"] = False
        scen["recommended"] = True
        stage = OUT / letter
        (stage / "s5").mkdir(parents=True)
        for s in LINKED:
            if (src / s).exists():
                (stage / s).symlink_to(src / s, target_is_directory=True)
        al = {**alloc, "scenarios": [x for x in alloc["scenarios"] if x["name"] != scen["name"]] + [scen]}
        (stage / "s5" / "allocation.json").write_text(json.dumps(al, indent=2, default=float))
        scfg = {**cfg, "out_root": str(stage.relative_to(ROOT))}
        rec = {"letter": letter, "design": d["id"], "scenario": scen["name"], "title": o["title"],
               "share_of_weightings": o["share_of_weightings"], "personas": o["personas"],
               "metrics": o["metrics"], "station_times": scen["station_times"], "robots": d["robots"],
               "layout_levers": d["layout_levers"], "package": f"out_options/{letter}/sim_package", "replay": {}}
        if scen["robots"]:
            s9.run(scfg)
            X.run(scfg)
            for st in [r["station"] for r in scen["robots"]]:
                for extra, tag in ((["--video"], "tobe"), (["--after", "away"], "tobe_away")):
                    subprocess.run([py, "replay.py", st, *extra], cwd=stage / "sim_package", check=True,
                                   capture_output=True)
                    rec["replay"].setdefault(st, {})[tag] = load_json(stage / "sim_package" / st / f"kpi_{tag}.json")
                tl = load_json(stage / "sim_package" / st / "human_tobe_timeline.json")
                rep_cycles[st] = tl["source_cycle"]
        else:
            rec["worker_only"] = True
        manifest["options"].append((rec, scen, scfg))
    for rec, scen, scfg in manifest["options"]:   # worker-only packages reuse the robot packages' representative cycles
        if rec.get("worker_only"):
            sd = ROOT / scfg["out_root"] / "sim_package"
            sd.mkdir(parents=True, exist_ok=True)
            worker_only(cfg, sd, scen, {st: rep_cycles[st] for st in sorted(rep_cycles)})
    manifest["options"] = [r for r, _, _ in manifest["options"]]
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, default=float))
    (OUT / "options_matrix.yaml").write_text(options_matrix(manifest))
    for r in manifest["options"]:
        k = {st: f"SSM {v['tobe']['share_active_time_ssm_violation']:.0%}/{v['tobe_away']['share_active_time_ssm_violation']:.0%}"
             for st, v in r["replay"].items()}
        print(r["letter"], r["design"], r["scenario"], r["title"], k or "worker-only")
    return 0


if __name__ == "__main__":
    sys.exit(main())
