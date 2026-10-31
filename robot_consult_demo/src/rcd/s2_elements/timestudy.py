"""Time study: per-cycle element totals -> Westinghouse-rated normal time -> ILO allowances -> standard time,
required sample size, MTM-UAS comparison, I-MR SPC with capability vs takt, lean value analysis.

References
  Niebel & Freivalds, "Methods, Standards, and Work Design" (Westinghouse leveling table; sample size
      n' = (z s / (a xbar))^2; ILO allowance table reproduced from ILO "Introduction to Work Study").
  ILO (1992), Introduction to Work Study, 4th ed., Appendix 3 (rest-allowance table).
  MTM Association, MTM-UAS data card (basic operations A*, P*, H*, B*, Z*, K*, VA); 1 TMU = 0.036 s.
  Montgomery, Introduction to Statistical Quality Control: I-MR chart, UCL = xbar + 3 MRbar/d2 with d2 = 1.128
      (i.e. 2.66 MRbar), sigma_hat = MRbar/1.128; Cpu = (USL - xbar)/(3 sigma_hat)   (ISyE 512 Ch. 6).
  ISyE 510 lecture06: line pacing (rigid pacing on a moving conveyor), total work content, minimum rational
      work element.
  Lean: Ohno (1988) seven wastes (TIMWOOD); VA / NNVA / NVA classification (Hines & Rich 1997).
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from .taxonomy import BY_CODE

Z95 = 1.959963984540054
TMU_S = 0.036

# ---------------------------------------------------------------- Westinghouse leveling system (Niebel & Freivalds)
WH_SKILL = {"A1": 0.15, "A2": 0.13, "B1": 0.11, "B2": 0.08, "C1": 0.06, "C2": 0.03, "D": 0.00, "E1": -0.05,
            "E2": -0.10, "F1": -0.16, "F2": -0.22}
WH_EFFORT = {"A1": 0.13, "A2": 0.12, "B1": 0.10, "B2": 0.08, "C1": 0.05, "C2": 0.02, "D": 0.00, "E1": -0.04,
             "E2": -0.08, "F1": -0.12, "F2": -0.17}
WH_CONDITIONS = {"A": 0.06, "B": 0.04, "C": 0.02, "D": 0.00, "E": -0.03, "F": -0.07}
WH_CONSISTENCY = {"A": 0.04, "B": 0.03, "C": 0.01, "D": 0.00, "E": -0.02, "F": -0.04}
# Operationalisation of the consistency class from the coefficient of variation of the same subgoal's duration
# across cycles (Westinghouse leaves consistency to the analyst's judgement of the spread of readings).
CV_TO_CONSISTENCY = [(0.02, "A"), (0.05, "B"), (0.10, "C"), (0.20, "D"), (0.30, "E"), (math.inf, "F")]

# ---------------------------------------------------------------- MTM-UAS data card (TMU by distance class 1/2/3)
# distance classes: 1 <= 20 cm, 2 > 20-50 cm, 3 > 50-80 cm
UAS = {
    "AA": (20, 35, 50), "AB": (30, 45, 60), "AC": (40, 55, 70),       # get & place <=1 kg, easy grasp: approx/loose/tight
    "AD": (20, 45, 60), "AE": (30, 55, 70), "AF": (40, 65, 80),       # <=1 kg difficult grasp
    "AH": (25, 45, 55), "AJ": (40, 65, 75), "AK": (50, 75, 85),       # >1-8 kg
    "PA": (10, 20, 25), "PB": (20, 30, 35), "PC": (30, 40, 45),       # place approx/loose/tight
    "HA": (25, 45, 65), "HB": (40, 60, 75), "HC": (50, 70, 85),       # handle auxiliary tool
    "BA": (10, 25, 40), "BB": (30, 45, 60),                           # operate simple / composite
    "ZA": (5, 15, 20), "ZB": (10, 30, 40), "ZC": (30, 45, 55),        # motion cycles
}
UAS_KA_PER_M = 25     # walk, per metre
UAS_KB = 60           # bend, stoop, kneel incl. arise
UAS_VA = 15           # visual control


def dist_class(d_m: float) -> int:
    return 0 if d_m <= 0.20 else (1 if d_m <= 0.50 else 2)


def uas_tmu(e: dict, part_mass_kg: float, strong_deg: float) -> tuple[float, list[str]]:
    """TMU of one element occurrence from MTM-UAS building blocks; distances measured in mocap."""
    k = dist_class(e.get("wrist_disp_m", 0.3))
    n_act = max(1, len(str(e.get("activity", "")).split("+"))) if e.get("activity") else 1
    if e.get("step_first_piece") is False:      # continuation of a step split by carved-out waiting
        n_act = 0
    code = e["code"]
    parts: list[str] = []
    tmu = 0.0
    if code == "WALK":
        tmu = UAS_KA_PER_M * e["travel_m"]
        parts.append(f"KA x {e['travel_m']:.2f} m")
    elif code == "GET_PART":
        blk = "AB" if part_mass_kg <= 1.0 else "AJ"
        tmu = UAS[blk][k]
        parts.append(f"{blk}{k + 1}")
    elif code == "INSERT":
        tmu = n_act * UAS["PC"][k]
        parts.append(f"{n_act} x PC{k + 1}")
    elif code == "GET_TOOL":
        tmu = n_act * UAS["HA"][k]
        parts.append(f"{n_act} x HA{k + 1}")
    elif code == "REACH_FASTENER":
        n = int(e.get("fastener_count", 1))
        tmu = n * UAS["AF"][k]
        parts.append(f"{n} x AF{k + 1}")
    elif code == "FASTEN":
        n = int(e.get("fastener_count", 1))
        tmu = n * (UAS["PC"][k] + UAS["BA"][k])
        parts.append(f"{n} x (PC{k + 1} + BA{k + 1})")
    elif code == "INSPECT":
        tmu = n_act * UAS_VA
        parts.append(f"{n_act} x VA")
    elif code == "SETUP":
        tmu = n_act * UAS["BB"][k] + UAS_KA_PER_M * e["travel_m"]
        parts.append(f"{n_act} x BB{k + 1} + KA")
    if code not in ("WALK", "IDLE") and e.get("trunk_flex_deg_max", 0) >= strong_deg:
        tmu += UAS_KB
        parts.append("KB")
    return tmu, parts


PMTS_GAP_NOTES = {
    "WALK": "Observed includes acceleration, path detours around the carrier and hand-over positions; KA assumes straight walking at normal pace.",
    "GET_PART": "Search/select at the cart and part orientation (not in AB/AJ) add time; short when the part is pre-presented.",
    "INSERT": "Clip/channel insertion needs trial-and-error alignment, pressing and regrasps that a single PC (place tight) per activity does not cover.",
    "GET_TOOL": "Includes hand-over to/from the assistive robot and waiting for the arm; HA only covers get/put-aside of the tool.",
    "REACH_FASTENER": "Picking single screws from a pouch and seating them on the bit is a difficult grasp with regrasps.",
    "FASTEN": "Screw/nut run-down is process time of the power tool (not a PMTS motion) plus holding the tool on axis; manual nut starting adds ZC cycles.",
    "INSPECT": "Visual check may include touching/feeling the part.",
    "SETUP": "Cart pushing and workstation height adjustment wait for the actuator.",
    "IDLE": "Waiting has no PMTS motion content (0 TMU).",
}


def westinghouse(consistency_cv: float, skill: str = "D", effort: str = "D", conditions: str = "C") -> dict:
    cons = next(cls for thr, cls in CV_TO_CONSISTENCY if consistency_cv <= thr)
    f = {"skill": {"class": skill, "value": WH_SKILL[skill]}, "effort": {"class": effort, "value": WH_EFFORT[effort]},
         "conditions": {"class": conditions, "value": WH_CONDITIONS[conditions]},
         "consistency": {"class": cons, "value": WH_CONSISTENCY[cons], "cv": round(consistency_cv, 4)}}
    return {"factors": f, "rating": round(1.0 + sum(v["value"] for v in f.values()), 4)}


def ilo_allowances(share_bent: float, share_elbow: float, share_overhead: float, max_part_kg: float) -> dict:
    """ILO rest-allowance items (fractions of normal time), each derived from measured data or a stated judgement:
      constant: personal 5 %, basic fatigue 4 % (ILO table, constant allowances)
      standing 2 %: the operator stands for the whole cycle (EAWS P1 = 100 % of observed time)
      abnormal position: very awkward (stretching up, hands above head) 7 % if >= 10 % of time;
                         awkward (bending / elbow at shoulder) 2 % if bent-forward or elbow-at-shoulder >= 10 %; else 0
      use of force: ILO weight steps (2.5 kg -> 0 %, 5 -> 1 %, 7.5 -> 2 % ...) on the heaviest part handled
      close attention 2 %: fine/exacting (screw starting, clip alignment)   [analyst judgement]
      monotony 1 %: medium (approx. 4-min cycle with several subgoals)      [analyst judgement]
    The 10 %-of-time threshold for posture items is our operationalisation (EAWS treats < 10 % as minor)."""
    items = {"personal": 0.05, "basic_fatigue": 0.04, "standing": 0.02}
    if share_overhead >= 0.10:
        items["abnormal_position"] = 0.07
    elif share_bent >= 0.10 or share_elbow >= 0.10:
        items["abnormal_position"] = 0.02
    else:
        items["abnormal_position"] = 0.0
    kg_steps = [(2.5, 0.0), (5, 0.01), (7.5, 0.02), (10, 0.03), (12.5, 0.04), (15, 0.05)]
    items["use_of_force"] = next((v for kg, v in kg_steps if max_part_kg <= kg), 0.09)
    items["close_attention"] = 0.02
    items["monotony"] = 0.01
    return {"items": {k: round(v, 6) for k, v in items.items()}, "ilo_total": round(sum(items.values()), 6),
            "inputs": {"share_bent": round(share_bent, 4), "share_elbow_at_shoulder": round(share_elbow, 4),
                       "share_hands_above_head": round(share_overhead, 4), "max_part_kg": max_part_kg}}


def imr(x: np.ndarray, usl: float) -> dict:
    x = np.asarray(x, float)
    mr = np.abs(np.diff(x))
    mrbar = float(mr.mean()) if len(mr) else 0.0
    xbar = float(x.mean())
    sigma = mrbar / 1.128
    return {"observations_s": [round(float(v), 4) for v in x], "xbar_s": xbar, "mr_bar_s": mrbar,
            "ucl_s": xbar + 2.66 * mrbar, "lcl_s": max(0.0, xbar - 2.66 * mrbar), "ucl_mr_s": 3.267 * mrbar,
            "sigma_hat_s": sigma, "usl_s": usl,
            "cpu": ((usl - xbar) / (3 * sigma)) if sigma > 0 else None,
            "out_of_control_points": [i for i, v in enumerate(x) if v > xbar + 2.66 * mrbar or v < xbar - 2.66 * mrbar]}


def element_stats(cycles: list[dict], ratings: dict, allow: float, acc: float, full_cycle_s: float) -> list[dict]:
    by_st = defaultdict(list)
    for c in cycles:
        by_st[c["station"]].append(c)
    rows = []
    for st, cs in sorted(by_st.items()):
        codes = sorted({e["code"] for c in cs for e in c["elements"]})
        per = {code: [sum(e["duration_s"] for e in c["elements"] if e["code"] == code) for c in cs] for code in codes}
        per["UNOBS"] = [max(0.0, full_cycle_s - c["duration_s"]) for c in cs]
        for code, tot in per.items():
            x = np.array(tot, float)
            n = len(x)
            mean = float(x.mean())
            std = float(x.std(ddof=1)) if n >= 2 else 0.0
            nreq = math.ceil((Z95 * std / (acc * mean)) ** 2) if (n >= 2 and mean > 0) else 0
            r = 1.0 if code == "UNOBS" else ratings[st]["rating"]
            if code == "IDLE":
                # observed waiting is not work content: unavoidable delay is covered by the D of the PFD allowance
                normal, basis = 0.0, "IDLE: delay -> covered by the PFD allowance, normal time 0"
            elif code == "UNOBS":
                # elapsed residual of a real ~full_cycle_s cycle: already contains its own PF&D share
                normal, basis = mean / (1 + allow), "UNOBS: elapsed residual already includes PF&D -> normal = mean/(1+A)"
            else:
                normal, basis = mean * r, "mean x Westinghouse rating"
            rows.append({"station": st, "code": code, "element_key": f"{st}:{code}", "n": n,
                         "per_cycle_s": [round(v, 3) for v in tot], "mean_s": round(mean, 4), "std_s": round(std, 4),
                         "cv": round(std / mean, 4) if mean > 0 else None,
                         "occurrences_per_cycle": round(float(np.mean([sum(e["code"] == code for e in c["elements"])
                                                                        for c in cs])), 3) if code != "UNOBS" else 1.0,
                         "rating": r, "normal_basis": basis, "normal_s": round(normal, 4), "standard_s": round(normal * (1 + allow), 4),
                         "n_required": nreq, "sample_ok": n >= nreq,
                         "lean_class": None if code == "UNOBS" else BY_CODE[code]["lean_class"]})
    # make the gate arithmetic exact on the rounded numbers
    for r in rows:
        if r["n"] >= 2 and r["mean_s"] > 0:
            r["n_required"] = math.ceil((Z95 * r["std_s"] / (acc * r["mean_s"])) ** 2)
            r["sample_ok"] = r["n"] >= r["n_required"]
        r["standard_s"] = round(r["normal_s"] * (1 + allow), 6)
    return rows
