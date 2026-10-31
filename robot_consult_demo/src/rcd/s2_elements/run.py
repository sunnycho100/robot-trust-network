"""Stage s2: work-element decomposition, time study and ergonomics from CarDA mocap + annotation.

Outputs (out/s2/): elements.json, ergonomics.json, time_study.json, figures/*.png.
Methods and citations: docs/IE_METHODS.md.
"""
from __future__ import annotations

import re
from collections import defaultdict

import numpy as np

from rcd.common.intake import mode
from rcd.common.io import ROOT, data_path, out_dir, save_json

from . import ergonomics as ER
from . import timestudy as TS
from .data import discover, parse_annotations
from .elements import segment_cycle
from .features import RATE_HZ, path_length
from .parts import part_for
from .taxonomy import BY_CODE, TAXONOMY, UNOBS


def _element_ergo(e: dict, masks: dict, f: dict) -> None:
    a, b = int(round(e["start_s"] * RATE_HZ)), int(round(e["end_s"] * RATE_HZ))
    b = max(b, a + 1)
    bent = masks["P3_bent_20_60"][a:b] | masks["P4_strongly_bent_gt60"][a:b]
    over = masks["P5_elbow_at_above_shoulder"][a:b] | masks["P6_hands_above_head"][a:b]
    asym = masks["TR_trunk_rotation"][a:b] | masks["LB_lateral_bending"][a:b]
    e["posture_share"] = {"bent": round(float(bent.mean()), 3), "overhead": round(float(over.mean()), 3),
                          "asymmetric": round(float(asym.mean()), 3)}
    e["reba_trunk_mean"] = ER.rula_reba(f, a, b)["reba_trunk"]["mean"]


def _validation_records(cfg: dict, cycles: list[dict], feats: dict) -> list[dict]:
    """SOP mode: the annotation workbook (if the dataset has one) is read ONLY to validate the posture durations
    on the annotators' own subgoal spans (same protocol as the original mode); it never feeds the segmentation."""
    recs = []
    for st in cfg["data"]["stations"]:
        xl = sorted(data_path(cfg, st.lower()).glob("annotation_data_*.xlsx"))
        if not xl:
            continue
        ann = parse_annotations(xl[0])
        for c in cycles:
            if c["station"] != st or c["cycle_id"] not in ann:
                continue
            f = feats[(st, c["cycle_id"])]
            M = f["_masks"]
            for sg in ann[c["cycle_id"]]:
                a = int(round(max(0.0, sg.start) * RATE_HZ))
                b = int(round(min(sg.end, f["n"] / RATE_HZ) * RATE_HZ))
                if b - a < 10:
                    continue
                recs.append({"station": st, "cycle_id": c["cycle_id"], "subgoal": sg.name, "pred": ER.durations(M, a, b),
                             "pred_2s": ER.durations(M, a, b, min_s=2.0), "ann": sg.eaws})
    return recs


def _figures(cfg: dict, cycles: list[dict], feats: dict, spc: list[dict], od) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figdir = od / "figures"
    figdir.mkdir(exist_ok=True)
    made = []
    colors = {"WALK": "#9e9e9e", "GET_PART": "#1f77b4", "INSERT": "#2ca02c", "GET_TOOL": "#ff7f0e",
              "REACH_FASTENER": "#9467bd", "FASTEN": "#d62728", "INSPECT": "#17becf", "SETUP": "#8c564b", "IDLE": "#e0e0e0"}
    by_st = defaultdict(list)
    for c in cycles:
        by_st[c["station"]].append(c)
    for st, cs in by_st.items():
        # Gantt of element sequences
        fig, ax = plt.subplots(figsize=(12, 0.5 + 0.45 * len(cs)))
        for k, c in enumerate(cs):
            for e in c["elements"]:
                ax.barh(k, e["duration_s"], left=e["start_s"], color=colors[e["code"]], edgecolor="white", linewidth=0.3)
        ax.set_yticks(range(len(cs)), [c["cycle_id"] for c in cs])
        ax.set_xlabel("time in clip [s]")
        ax.set_title(f"{st}: work elements (mocap segmentation)")
        ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=v) for v in colors.values()], labels=list(colors),
                  ncol=5, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.35 if len(cs) < 4 else -0.15))
        fig.tight_layout()
        p = figdir / f"gantt_{st}.png"
        fig.savefig(p, dpi=110)
        plt.close(fig)
        made.append(str(p.relative_to(od.parent.parent)))
        # spaghetti diagram (pelvis path, top view), each cycle re-centred on its first long dwell
        fig, ax = plt.subplots(figsize=(5, 5))
        for c in cs:
            f = feats[(st, c["cycle_id"])]
            xy = f["pelvis_xy"] - np.median(f["pelvis_xy"], axis=0)
            ax.plot(xy[:, 0], xy[:, 1], lw=0.8, label=f"{c['cycle_id']} ({c['spaghetti_m']:.0f} m)")
        ax.set_aspect("equal")
        ax.set_title(f"{st}: spaghetti diagram (pelvis path)")
        ax.legend(fontsize=7)
        fig.tight_layout()
        p = figdir / f"spaghetti_{st}.png"
        fig.savefig(p, dpi=110)
        plt.close(fig)
        made.append(str(p.relative_to(od.parent.parent)))
    # I-MR charts
    fig, axs = plt.subplots(len(spc), 1, figsize=(7, 2.2 * len(spc)), squeeze=False)
    for ax, sp in zip(axs[:, 0], spc):
        x = sp["observations_s"]
        ax.plot(range(1, len(x) + 1), x, "o-")
        for v, ls, lab in ((sp["ucl_s"], "--", "UCL"), (sp["xbar_s"], "-", "x-bar"), (sp["lcl_s"], "--", "LCL"),
                           (sp["usl_s"], ":", "USL = takt")):
            ax.axhline(v, ls=ls, color="k" if lab != "USL = takt" else "r", lw=0.8)
            ax.text(len(x) + 0.1, v, lab, fontsize=7, va="center")
        ax.set_title(f"{sp['station']}: Individuals chart, Cpu={sp['cpu']:.2f}" if sp["cpu"] is not None else sp["station"],
                     fontsize=9)
        ax.set_ylabel("s")
    fig.tight_layout()
    p = figdir / "imr_charts.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    made.append(str(p.relative_to(od.parent.parent)))
    return made


def run(cfg: dict) -> None:
    od = out_dir(cfg, "s2")
    tcfg = cfg["time_study"]
    allow, acc = float(tcfg["allowance_pfd"]), float(tcfg["accuracy"])
    full = float(cfg["observation"]["full_cycle_s"])
    takt = cfg["line"]["takt_s"]
    strong = cfg["ergonomics"]["strongly_bent_deg"]

    sop_mode = mode(cfg, "segmentation", "annotation") == "sop"
    if sop_mode:
        from . import sop_run
        taxonomy = [dict(t) for t in TAXONOMY] + [dict(UNOBS, pseudo=True)]
        cycles, feats, masks_by, erg_cycles, sop_meta = sop_run.segment_all(cfg, od, _element_ergo, taxonomy)
        records = _validation_records(cfg, cycles, feats)
        raw_cycles = []
    else:
        raw_cycles = discover(cfg)
        cycles, feats, masks_by = [], {}, {}
        erg_cycles, records = [], []
    if mode(cfg, "parts", "builtin") == "intake":
        from rcd.s3_automatability.generic_parts import IntakeParts
        _ip = IntakeParts(cfg)
        pfor = lambda st, sg: _ip(st, sg)          # noqa: E731
    else:
        pfor = lambda st, sg: part_for(sg)         # noqa: E731
    for c in raw_cycles:
        cyc, f = segment_cycle(c)
        M = ER.posture_masks(f, cfg)
        for e in cyc["elements"]:
            _element_ergo(e, M, f)
        cyc["spaghetti_m"] = round(path_length(f["pelvis_xy"]), 2)
        cyc["walk_distance_m"] = round(sum(e["travel_m"] for e in cyc["elements"] if e["code"] == "WALK"), 2)
        cycles.append(cyc)
        feats[(c.station, c.cycle_id)] = f
        masks_by[(c.station, c.cycle_id)] = M
        sub = []
        for sp in cyc["subgoals"]:
            a, b = int(round(sp["refined"][0] * RATE_HZ)), int(round(sp["refined"][1] * RATE_HZ))
            ann_a, ann_b = int(round(sp["annotated"][0] * RATE_HZ)), int(round(min(sp["annotated"][1], cyc["duration_s"]) * RATE_HZ))
            pred = ER.durations(M, ann_a, ann_b)      # validate on the annotators' own span
            sg = next(s for s in c.subgoals if s.name == sp["name"])
            sub.append({"subgoal": sp["name"], "span_s": sp["refined"], "postures_s": ER.durations(M, a, b),
                        "postures_raw_s": ER.durations(M, a, b, eaws_rule=False), "scores": ER.rula_reba(f, a, b),
                        "annotated_eaws": {k: v for k, v in sg.eaws.items() if v is not None},
                        "annotated_intensity": {k: v for k, v in sg.eaws_intensity.items() if v is not None}})
            records.append({"station": c.station, "cycle_id": c.cycle_id, "subgoal": sp["name"], "pred": pred,
                            "pred_2s": ER.durations(M, ann_a, ann_b, min_s=2.0), "ann": sg.eaws})
        erg_cycles.append({"station": c.station, "cycle_id": c.cycle_id, "duration_s": cyc["duration_s"],
                           "postures_s": ER.durations(M, 0, f["n"]), "postures_raw_s": ER.durations(M, 0, f["n"], False),
                           "scores": ER.rula_reba(f), "subgoals": sub})

    # ---------------------------------------------------------------- ergonomics
    cm = ER.infer_column_mapping(records)
    val = ER.validate(records, cm["mapping"])
    val_2s = ER.validate([dict(r, pred=r["pred_2s"]) for r in records], cm["mapping"])
    val["sensitivity_min_bout_2s"] = {"spearman": val_2s["value"], "mae_s": val_2s["mae_s"], "mae_rel": val_2s["mae_rel"],
                                      "note": "2-s minimum holding time instead of the 4-s EAWS/CarDA rule (not the headline)"}
    ergonomics = {
        "method": "EAWS section-1 posture durations from Xsens mocap (trunk flexion Hips->Neck vs vertical; elbow vs "
                  "shoulder height; wrist vs head; trunk axial rotation and lateral bending > 10 deg per ISO 11226); "
                  "postures counted only in bouts >= 4 s (CarDA annotation rule); RULA upper-arm and REBA trunk scores",
        "thresholds": {"bent_forward_deg": cfg["ergonomics"]["bent_forward_deg"], "strongly_bent_deg": strong,
                       "asymmetry_deg": ER.ASYM_DEG, "min_bout_s": ER.MIN_BOUT_S, "bridge_gap_s": ER.BRIDGE_S},
        "posture_keys": ER.POSTURES,
        "annotation_column_mapping": cm,
        "cycles": erg_cycles,
        "validation": {k: v for k, v in val.items()},
    }
    save_json(ergonomics, od / "ergonomics.json", contract="ergonomics")

    # ---------------------------------------------------------------- time study
    by_st = defaultdict(list)
    for c in cycles:
        by_st[c["station"]].append(c)
    ratings = {}
    for st, cs in by_st.items():
        durs = defaultdict(list)
        for c in cs:
            for sp in c["subgoals"]:
                durs[sp["name"]].append(sp["refined"][1] - sp["refined"][0])
        cvs = [(np.std(v, ddof=1) / np.mean(v), np.mean(v)) for v in durs.values() if len(v) >= 2 and np.mean(v) > 0]
        cv = float(np.average([x for x, _ in cvs], weights=[w for _, w in cvs])) if cvs else \
            float(np.std([c["duration_s"] for c in cs], ddof=1) / np.mean([c["duration_s"] for c in cs]))
        ratings[st] = TS.westinghouse(cv)
        ratings[st]["consistency_basis"] = "time-weighted CV of the same subgoal's duration across cycles"
    # PMTS (MTM-UAS) per element occurrence -> per (station, code) per-cycle mean
    pm_rows = []
    synth = {}
    for st, cs in by_st.items():
        tmu_tot = defaultdict(float)
        obs_tot = defaultdict(float)
        blocks = defaultdict(set)
        for c in cs:
            for e in c["elements"]:
                part = pfor(st, e.get("subgoal"))
                tmu, parts = TS.uas_tmu(e, part["mass_kg"], strong)
                e["pmts_tmu"] = round(tmu, 1)
                tmu_tot[e["code"]] += tmu
                obs_tot[e["code"]] += e["duration_s"]
                blocks[e["code"]].update(re.findall(r"\b(?:[A-Z]{2}\d|KA|KB|VA)\b", " ".join(parts)))
        n = len(cs)
        for code in sorted(tmu_tot):
            tmu = tmu_tot[code] / n
            obs = obs_tot[code] / n
            pm_rows.append({"station": st, "code": code, "tmu": round(tmu, 2), "pmts_s": round(round(tmu, 2) * TS.TMU_S, 4),
                            "observed_s": round(obs, 3),
                            "observed_over_pmts": round(obs / (tmu * TS.TMU_S), 2) if tmu > 0 else None,
                            "uas_blocks": sorted(b for b in blocks[code] if b),
                            "gap_explanation": TS.PMTS_GAP_NOTES.get(code, "")})
        manip = [r for r in pm_rows if r["station"] == st and r["code"] in ("GET_PART", "INSERT", "GET_TOOL", "REACH_FASTENER")]
        s_p = sum(r["pmts_s"] for r in manip)
        s_o = sum(r["observed_s"] for r in manip)
        synth[st] = round(s_p / s_o, 3) if s_o > 0 else None

    # ILO allowances: shares of observed time in EAWS-counted postures (bout rule) and the heaviest part handled
    def _shares(keys):
        n = b = e = o = 0.0
        for k in keys:
            M = masks_by[k]
            N = len(M["P1_standing_walking"])
            n += N
            b += ER.bouts_filter(M["P3_bent_20_60"] | M["P4_strongly_bent_gt60"]).sum()
            e += ER.bouts_filter(M["P5_elbow_at_above_shoulder"]).sum()
            o += ER.bouts_filter(M["P6_hands_above_head"]).sum()
        return b / n, e / n, o / n

    def _maxkg(cs):
        return max([pfor(c["station"], e.get("subgoal"))["mass_kg"] for c in cs for e in c["elements"]
                    if e["code"] == "GET_PART"] or [0.0])
    ilo = TS.ilo_allowances(*_shares(list(masks_by)), _maxkg(cycles))
    allow = ilo["ilo_total"]                      # line-level allowance derived from the data (config is a default)
    ilo_by_station = {st: TS.ilo_allowances(*_shares([(st, c["cycle_id"]) for c in cs]), _maxkg(cs))
                      for st, cs in by_st.items()}
    stats = TS.element_stats(cycles, ratings, allow, acc, full)
    for r in stats:
        r["allowance"] = allow
        r["allowance_station_ilo"] = ilo_by_station[r["station"]]["ilo_total"]

    # SPC: per-cycle station work content = observed non-idle element time + unobserved manual block
    spc = []
    station_stats = []
    for st, cs in sorted(by_st.items()):
        obs = np.array([sum(e["duration_s"] for e in c["elements"] if e["code"] != "IDLE") + max(0.0, full - c["duration_s"])
                        for c in cs])
        sp = TS.imr(obs, float(takt))
        sp["station"] = st
        sp["definition"] = "per-cycle work content = sum of observed non-IDLE element times + UNOBS block (config.observation)"
        frac = float(np.mean([c["duration_s"] for c in cs])) / full
        sp["observed_fraction"] = round(frac, 4)
        sp["basis"] = "measured" if frac >= 0.85 else "assumption (UNOBS block dominates; capability not measured)"
        spc.append(sp)
        rows = [r for r in stats if r["station"] == st]
        std_total = sum(r["standard_s"] for r in rows)
        std_obs = sum(r["standard_s"] for r in rows if r["code"] != "UNOBS")
        station_stats.append({
            "station": st, "n_cycles": len(cs), "cycle_ids": [c["cycle_id"] for c in cs],
            "observed_mean_s": round(float(np.mean([c["duration_s"] for c in cs])), 3),
            "observed_work_mean_s": round(float(np.mean([sum(e["duration_s"] for e in c["elements"] if e["code"] != "IDLE")
                                                         for c in cs])), 3),
            "unobs_mean_s": round(float(np.mean([max(0.0, full - c["duration_s"]) for c in cs])), 3),
            "rating": ratings[st]["rating"], "normal_s": round(sum(r["normal_s"] for r in rows), 3),
            "standard_s": round(std_total, 3), "standard_observed_s": round(std_obs, 3),
            "observed_fraction": round(float(np.mean([c["duration_s"] for c in cs])) / full, 4),
            "overload_basis": "measured" if float(np.mean([c["duration_s"] for c in cs])) / full >= 0.85 else "assumption",
            "takt_s": takt, "utilisation": round(std_total / takt, 4), "standard_over_takt": round(std_total / takt, 4),
            "overload_pct": round(max(0.0, std_total / takt - 1) * 100, 2),
            "spaghetti_m_per_cycle": round(float(np.mean([c["spaghetti_m"] for c in cs])), 2),
            "walk_m_per_cycle": round(float(np.mean([c["walk_distance_m"] for c in cs])), 2),
            "synthetic_rating_pmts_over_observed": synth[st],
            "va_ratio": None,
        })

    # Lean value-stream classification
    codes_used = sorted({e["code"] for c in cycles for e in c["elements"]})
    lean_rows = []
    for code in codes_used:
        t = BY_CODE[code]
        time_share = sum(e["duration_s"] for c in cycles for e in c["elements"] if e["code"] == code) / \
            sum(c["duration_s"] for c in cycles)
        row = {"code": code, "class": t["lean_class"], "waste": t["waste"], "time_share": round(time_share, 4)}
        if code == "WALK":
            walks = [e for c in cycles for e in c["elements"] if e["code"] == "WALK"]
            row["loaded_share"] = round(sum(e["duration_s"] for e in walks if e.get("loaded")) /
                                        max(1e-9, sum(e["duration_s"] for e in walks)), 3)
            row["note"] = "empty walking = Motion; loaded walking (part/tool carried) is Transportation"
        lean_rows.append(row)
    tot = sum(c["duration_s"] for c in cycles)
    va = sum(e["duration_s"] for c in cycles for e in c["elements"] if BY_CODE[e["code"]]["lean_class"] == "VA")
    for ss in station_stats:
        cs = by_st[ss["station"]]
        ss["va_ratio"] = round(sum(e["duration_s"] for c in cs for e in c["elements"]
                                   if BY_CODE[e["code"]]["lean_class"] == "VA") / sum(c["duration_s"] for c in cs), 4)
        ss["nva_ratio"] = round(sum(e["duration_s"] for c in cs for e in c["elements"]
                                    if BY_CODE[e["code"]]["lean_class"] == "NVA") / sum(c["duration_s"] for c in cs), 4)

    rating_overall = TS.westinghouse(float(np.mean([r["factors"]["consistency"]["cv"] for r in ratings.values()])))
    time_study = {
        "params": {"confidence": tcfg["confidence"], "z": TS.Z95, "accuracy": acc, "allowance_pfd": allow,
                   "allowance_pfd_config_default": tcfg["allowance_pfd"],
                   "performance_rating_config": tcfg["performance_rating"], "takt_s": takt, "full_cycle_s": full,
                   "unit": "per-cycle total of an element code in a station (sum of its occurrences in one cycle)"},
        "rating": {"method": "Westinghouse leveling system (skill, effort, conditions, consistency)",
                   "factors": rating_overall["factors"], "rating": rating_overall["rating"],
                   "by_station": ratings,
                   "assumptions": {"skill": "D average: trained line operators, no evidence for above/below average skill",
                                   "effort": "D average: rigidly paced conveyor line (ISyE 510 lecture06) sets the pace; "
                                             "the MTM-UAS synthetic rating is reported as a cross-check, not applied",
                                   "conditions": "C good: clean, well-lit pilot line visible in the CarDA video",
                                   "consistency": "from the coefficient of variation of the same subgoal's duration "
                                                  "across cycles: <=2% A, <=5% B, <=10% C, <=20% D, <=30% E, else F"},
                   "normal_time": "normal_s = mean_s x station rating; IDLE normal 0 (delay is in the PFD allowance); "
                                  "UNOBS normal = mean_s/(1+A) so that its standard equals the observed elapsed residual"},
        "allowance_breakdown": ilo["items"],
        "allowance_detail": {"standard": "ILO rest-allowance table (Introduction to Work Study, 4th ed.)",
                             "basis": "line level, pooled over all observed cycles", "inputs": ilo["inputs"],
                             "ilo_total": ilo["ilo_total"], "config_default_pfd": tcfg["allowance_pfd"],
                             "by_station": ilo_by_station,
                             "note": "all items derived (no balancing item); params.allowance_pfd = the derived total; "
                                     "standard_s on every element row uses this line-level allowance, the station-level "
                                     "ILO totals are reported per row as allowance_station_ilo"},
        "element_stats": stats,
        "station_stats": station_stats,
        "pmts": {"system": "MTM-UAS", "tmu_s": TS.TMU_S, "rows": pm_rows,
                 "distance_classes_cm": ["<=20", "20-50", "50-80"],
                 "basis": "per occurrence: UAS block by element type, distance class from the active wrist's maximum "
                          "displacement in mocap, walking KA per metre of pelvis travel, KB when trunk flexion >= 60 deg; "
                          "per (station, code) = mean over cycles of the per-cycle TMU total",
                 "synthetic_rating_by_station": synth},
        "spc": spc,
        "lean": {"rows": lean_rows, "va_ratio": round(va / tot, 4),
                 "nva_ratio": round(sum(e["duration_s"] for c in cycles for e in c["elements"]
                                        if BY_CODE[e["code"]]["lean_class"] == "NVA") / tot, 4),
                 "basis": "observed clip time (UNOBS excluded)",
                 "spaghetti": [{"station": c["station"], "cycle_id": c["cycle_id"], "pelvis_path_m": c["spaghetti_m"],
                                "walk_m": c["walk_distance_m"]} for c in cycles]},
    }
    try:
        time_study["figures"] = _figures(cfg, cycles, feats, spc, od)
    except Exception as ex:  # figures are a convenience for the report
        time_study["figures_error"] = str(ex)
    save_json(time_study, od / "time_study.json", contract="time_study")

    taxonomy = [dict(t) for t in TAXONOMY] + [dict(UNOBS, pseudo=True)]
    if sop_mode:
        elements = {"method": sop_meta, "taxonomy": taxonomy, "cycles": cycles}
        save_json(elements, od / "elements.json", contract="elements")
        print(f"    s2 (sop): {len(cycles)} cycles, {sum(len(c['elements']) for c in cycles)} elements; "
              f"EAWS validation rho={val['value']} MAE={val['mae_s']}s; VA ratio={time_study['lean']['va_ratio']}")
        return
    elements = {"method": {"segmentation": "standard-work forced alignment (left-to-right HSMM, Viterbi, explicit "
                                           "minimum durations, skippable template steps) of the CarDA activity list "
                                           "inside each annotated subgoal; ergodic sticky HMM outside subgoals; "
                                           "waiting carved out as IDLE",
                           "rate_hz": RATE_HZ},
                "taxonomy": taxonomy, "cycles": cycles}
    save_json(elements, od / "elements.json", contract="elements")
    print(f"    s2: {len(cycles)} cycles, {sum(len(c['elements']) for c in cycles)} elements; "
          f"EAWS validation rho={val['value']} MAE={val['mae_s']}s; VA ratio={time_study['lean']['va_ratio']}")
