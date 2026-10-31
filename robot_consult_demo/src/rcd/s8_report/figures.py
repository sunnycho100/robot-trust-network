"""Matplotlib figures for the report. Every function takes stage JSON and returns a base64 PNG data URI."""
from __future__ import annotations

import base64
import io
import math
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

INK = "#1b2a3a"
MUTED = "#6b7785"
GRID = "#e6e9ee"
HUMAN = "#3b6ea8"
ROBOT = "#e07a2f"
INTER = "#9aa5b1"
UNOBS = "#d5dae1"
GOOD = "#1f9d7a"
HELPER = "#9c6ade"
BAD = "#c8443b"
CODE_COLORS = {"WALK": "#9c6ade", "GET_PART": "#3b6ea8", "INSERT": "#1f9d7a", "GET_TOOL": "#7fb3d5",
               "FASTEN": "#e07a2f", "REACH_FASTENER": "#f2b179", "INSPECT": "#c8443b", "IDLE": "#c9ced6",
               "UNOBS": UNOBS}
_extra = ["#2a9d8f", "#e9c46a", "#8d6e63", "#5c6bc0", "#ef476f", "#118ab2", "#06d6a0", "#ffd166", "#73777f"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": "#c3c9d1", "axes.labelcolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "figure.dpi": 110,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.12,
})


def code_color(c):
    if c not in CODE_COLORS:
        CODE_COLORS[c] = _extra[len(CODE_COLORS) % len(_extra)]
    return CODE_COLORS[c]


def uri(fig) -> str:
    b = io.BytesIO()
    fig.savefig(b, format="png", dpi=110)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


# ------------------------------------------------------------------ s2
def gantt(elements, max_cycles=4):
    cyc = elements["cycles"]
    by = defaultdict(list)
    for c in cyc:
        by[c["station"]].append(c)
    stations = sorted(by)
    rows = []
    for st in stations:
        for c in sorted(by[st], key=lambda c: c["cycle_id"])[:max_cycles]:
            rows.append((f"{st} / {c['cycle_id']}", c))
    fig, ax = plt.subplots(figsize=(10.5, 0.36 * len(rows) + 1.3))
    used = set()
    for i, (lab, c) in enumerate(rows):
        for e in c["elements"]:
            ax.barh(i, e["end_s"] - e["start_s"], left=e["start_s"], color=code_color(e["code"]), height=0.72,
                    edgecolor="white", linewidth=0.4)
            used.add(e["code"])
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("time in observed clip [s]")
    ax.set_title("Work-element Gantt per observed cycle (from mocap + video)")
    ax.grid(axis="y", visible=False)
    h = [plt.Rectangle((0, 0), 1, 1, color=code_color(c)) for c in sorted(used)]
    ax.legend(h, sorted(used), ncol=min(8, len(used)), loc="upper center", bbox_to_anchor=(0.5, -0.12 - 0.3 / len(rows)),
              fontsize=8.5)
    return uri(fig)


def spc_chart(ts):
    sp = [s for s in ts.get("spc", []) if len(s.get("observations_s", [])) >= 2]
    if not sp:
        return None
    fig, axes = plt.subplots(1, len(sp), figsize=(3.6 * len(sp), 2.8), squeeze=False)
    for ax, s in zip(axes[0], sp):
        x = np.asarray(s["observations_s"], float)
        ax.plot(range(1, len(x) + 1), x, "o-", color=HUMAN, ms=5)
        ax.axhline(x.mean(), color=MUTED, lw=1, label=f"mean {x.mean():.1f}")
        ax.axhline(s["ucl_s"], color=BAD, ls="--", lw=1, label=f"UCL {s['ucl_s']:.1f}")
        if s.get("usl_s"):
            ax.axhline(s["usl_s"], color=INK, ls=":", lw=1.2, label=f"USL = takt {s['usl_s']:.0f}")
        cpu = s.get("cpu")
        ax.set_title(f"{s['station']}  I-chart" + (f"   Cpu {cpu:.2f}" if cpu is not None else ""), fontsize=10)
        ax.set_xlabel("cycle")
        ax.legend(fontsize=7.5, loc="lower right")
    axes[0][0].set_ylabel("station time [s]")
    return uri(fig)


def element_time_bars(ts):
    rows = [r for r in ts["element_stats"] if r["code"] != "UNOBS"]
    st = sorted({r["station"] for r in rows})
    codes = sorted({r["code"] for r in rows})
    fig, ax = plt.subplots(figsize=(10.5, 3.0))
    w = 0.8 / max(1, len(st))
    for k, s in enumerate(st):
        m = [next((r["mean_s"] for r in rows if r["station"] == s and r["code"] == c), 0) for c in codes]
        sd = [next((r["std_s"] for r in rows if r["station"] == s and r["code"] == c), 0) for c in codes]
        ax.bar(np.arange(len(codes)) + k * w - 0.4 + w / 2, m, w, yerr=sd, label=s, capsize=2,
               color=[HUMAN, GOOD, ROBOT, "#9c6ade"][k % 4], error_kw={"ecolor": MUTED, "lw": 0.8})
    ax.set_xticks(range(len(codes)), codes, rotation=0, fontsize=8)
    ax.set_ylabel("per-cycle total [s]")
    ax.set_title("Time study: mean per-cycle element time (error bar = 1 s.d.)")
    ax.legend()
    return uri(fig)


def ergonomics_chart(erg):
    cyc = erg["cycles"]
    keys = sorted({k for c in cyc for k in c["postures_s"]})
    st = sorted({c["station"] for c in cyc})
    fig, ax = plt.subplots(figsize=(10.5, 2.8))
    w = 0.8 / max(1, len(st))
    for k, s in enumerate(st):
        vals = [np.mean([c["postures_s"].get(p, 0) for c in cyc if c["station"] == s]) for p in keys]
        ax.bar(np.arange(len(keys)) + k * w - 0.4 + w / 2, vals, w, label=s, color=[HUMAN, GOOD, ROBOT, "#9c6ade"][k % 4])
    ax.set_xticks(range(len(keys)), [p.replace("_", " ") for p in keys], fontsize=7.5, rotation=12)
    ax.set_ylabel("seconds per observed cycle")
    ax.set_title("Ergonomics: posture exposure per cycle (EAWS-style bands)")
    ax.legend()
    return uri(fig)


# ------------------------------------------------------------------ s3
def ahp_weights(au):
    cr = au["criteria"]
    fig, ax = plt.subplots(figsize=(5.0, 0.32 * len(cr) + 0.9))
    cr = sorted(cr, key=lambda c: c["weight"])
    ax.barh([c["key"].replace("_", " ") for c in cr], [c["weight"] for c in cr], color=GOOD)
    for i, c in enumerate(cr):
        ax.text(c["weight"], i, f" {c['weight']:.3f}", va="center", fontsize=8, color=INK)
    ax.set_xlabel("weight")
    ax.set_title("Criterion weights")
    ax.grid(axis="y", visible=False)
    return uri(fig)


def automatability_heat(au):
    T = [t for t in au["tasks"] if t["code"] != "UNOBS"]
    T = sorted(T, key=lambda t: -t["total"])
    keys = [c["key"] for c in au["criteria"]]
    M = np.array([[t["scores"].get(k, 0) for k in keys] + [t["total"]] for t in T])
    fig, ax = plt.subplots(figsize=(7.2, 0.27 * len(T) + 1.4))
    im = ax.imshow(M, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(keys) + 1), [k.replace("_", "\n") for k in keys] + ["TOTAL"], fontsize=7)
    ax.set_yticks(range(len(T)), [f"{t['station']}:{t['code']}  [{t['allocation']}]" for t in T], fontsize=7.5)
    for i in range(M.shape[0]):
        ax.text(M.shape[1] - 1, i, f"{M[i, -1]:.2f}", ha="center", va="center", fontsize=7, color="white" if M[i, -1] > .55 else INK)
    ax.grid(False)
    ax.set_title("Automatability scores per work element (weighted total)")
    fig.colorbar(im, ax=ax, fraction=0.03)
    return uri(fig)


# ------------------------------------------------------------------ s6a
def robot_times(rtt):
    E = sorted(rtt["entries"], key=lambda e: (e["group_id"], e["cycle_time_s"]))
    keys = []
    for e in E:
        for k in e["breakdown"]:
            if k not in keys:
                keys.append(k)
    fig, ax = plt.subplots(figsize=(10.5, 0.3 * len(E) + 1.3))
    left = np.zeros(len(E))
    pal = ["#3b6ea8", "#7fb3d5", "#e07a2f", "#f2b179", "#1f9d7a", "#9c6ade", "#c8443b", "#8d6e63", "#73777f"]
    for j, k in enumerate(keys):
        v = np.array([e["breakdown"].get(k, 0) for e in E])
        ax.barh(range(len(E)), v, left=left, color=pal[j % len(pal)], label=k, height=0.7)
        left += v
    for i, e in enumerate(E):
        ax.text(left[i], i, f" {e['cycle_time_s']:.1f}s", va="center", fontsize=7.5)
    ax.set_yticks(range(len(E)), [f"{e['group_id']}  |  {e['model']}" for e in E], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("simulated robot cycle time [s]")
    ax.set_title("Robot cycle time per (model, task group) and its breakdown")
    ax.legend(ncol=min(6, len(keys)), fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.1 - 0.3 / len(E)))
    ax.grid(axis="y", visible=False)
    return uri(fig)


# ------------------------------------------------------------------ s5
def station_loads(al, takt):
    S = al["scenarios"]
    st = sorted({k for s in S for k in s["station_times"]})
    fig, axes = plt.subplots(1, len(S), figsize=(2.3 * len(S) + 0.6, 3.1), sharey=True, squeeze=False)
    for ax, s in zip(axes[0], S):
        x = np.arange(len(st))
        h = [s["station_times"][k]["human_s"] for k in st]
        r = [s["station_times"][k].get("robot_only_s", s["station_times"][k]["robot_s"]) for k in st]
        hp = [s["station_times"][k].get("helper_s", 0.0) for k in st]
        i = [s["station_times"][k]["interaction_s"] for k in st]
        ax.bar(x - 0.19, h, 0.36, color=HUMAN, label="operator")
        ax.bar(x - 0.19, i, 0.36, bottom=h, color=INTER, label="handover")
        ax.bar(x + 0.19, r, 0.36, color=ROBOT, label="robot")
        if any(hp):
            ax.bar(x + 0.19, hp, 0.36, bottom=r, color=HELPER, label="relief operator")
        ax.axhline(takt, color=BAD, ls="--", lw=1)
        ax.set_xticks(x, st, fontsize=8)
        ax.set_title(s["name"].replace("_", " ") + ("  *" if s.get("recommended") else ""), fontsize=9,
                     color=GOOD if s.get("recommended") else INK)
        ax.grid(axis="x", visible=False)
    axes[0][0].set_ylabel("seconds per cycle")
    hs, ls = [], []
    for a in axes[0]:
        for hh, ll in zip(*a.get_legend_handles_labels()):
            if ll not in ls:
                hs.append(hh)
                ls.append(ll)
    fig.legend(hs, ls, fontsize=8, loc="lower center", ncol=len(ls), bbox_to_anchor=(0.5, -0.06))
    axes[0][-1].text(len(st) - 0.5, takt, " takt", color=BAD, fontsize=8, va="bottom", ha="right")
    fig.suptitle("Station loads per allocation scenario (* = recommended)", x=0.01, y=1.06, ha="left", fontsize=11,
                 fontweight="bold", color=INK)
    return uri(fig)


def balance_chart(lb):
    rows = [lb["as_is"]] + lb["heuristics"] + [lb["exact"]]
    fig, ax = plt.subplots(figsize=(10.5, 0.42 * len(rows) + 1.2))
    for i, h in enumerate(rows):
        left = 0
        for k, (tasks, load) in enumerate(zip(h["stations"], h["station_loads_s"])):
            ax.barh(i, load, left=left, color=[HUMAN, GOOD, ROBOT, "#9c6ade", "#7fb3d5"][k % 5], height=0.6,
                    edgecolor="white")
            ax.text(left + load / 2, i, f"S{k + 1}: {load:.0f}s", ha="center", va="center", fontsize=7.5, color="white")
            left += h["cycle_time_s"]
        ax.text(left + 4, i, f"n={h['n_stations']}  E={h['balance_efficiency']:.1%}  SI={h['smoothness_index']:.1f}",
                va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(rows)), [h["name"] for h in rows])
    ax.invert_yaxis()
    ax.set_xlabel(f"stations laid out at cycle time c = {rows[0]['cycle_time_s']:.0f} s")
    ax.set_title("Line balancing (SALBP-1): station loads per method")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(len(h["station_loads_s"]) for h in rows) * rows[0]["cycle_time_s"] * 1.32)
    return uri(fig)


def sweep_chart(lb):
    sw = lb.get("cycle_time_sweep", [])
    if not sw:
        return None
    fig, ax = plt.subplots(figsize=(5.0, 2.8))
    c = [s["cycle_time_s"] for s in sw]
    ax.plot(c, [s["theoretical_min"] for s in sw], "o--", color=MUTED, label="N_t = ceil(sum t / c)")
    ax.plot(c, [s["exact_stations"] for s in sw], "s-", color=GOOD, label="exact SALBP-1")
    ax.plot(c, [s["rpw_stations"] for s in sw], "^:", color=ROBOT, label="RPW")
    ax.set_xlabel("cycle time c [s]")
    ax.set_ylabel("stations")
    ax.set_title("Stations needed vs cycle time")
    ax.legend(fontsize=8)
    return uri(fig)


def worker_machine_chart(wm):
    if not wm:
        return None
    fig, axes = plt.subplots(len(wm), 1, figsize=(10.5, 1.55 * len(wm) + 0.6), squeeze=False,
                             gridspec_kw={"hspace": 0.75})
    for ax, w in zip(axes[:, 0], wm):
        h, r, i, c = w["human_s"], w["robot_s"], w["interaction_s"], w["cycle_s"]
        ax.barh(1, h, color=HUMAN, height=0.55)
        ax.barh(1, c - h - i, left=h, color="white", edgecolor=HUMAN, hatch="///", height=0.55)
        ax.barh(1, i, left=c - i, color=INTER, height=0.55)
        ax.barh(0, i, left=0, color=INTER, height=0.55)
        ax.barh(0, r, left=i, color=ROBOT, height=0.55)
        ax.barh(0, c - r - i, left=i + r, color="white", edgecolor=ROBOT, hatch="///", height=0.55)
        ax.axvline(w["takt_s"], color=BAD, ls="--", lw=1)
        ax.set_yticks([0, 1], [f"{w['model']}", "operator"], fontsize=8)
        ax.set_title(f"{w['station']}: cycle {c:.1f}s = max(H {h:.1f}, R {r:.1f}) + handover {i:.1f}   |   operator idle "
                     f"{w['human_idle_pct']:.0%}, robot idle {w['robot_idle_pct']:.0%}", fontsize=9)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, w["takt_s"] * 1.05)
    axes[-1, 0].set_xlabel("seconds within one takt  (solid = working, hatched = idle, grey = handover)")
    return uri(fig)


# ------------------------------------------------------------------ s6b
def des_chart(des):
    S = des["scenarios"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.0), gridspec_kw={"width_ratios": [1.4, 1]})
    y = np.arange(len(S))
    a1.barh(y, [s["throughput_per_hr"] for s in S], xerr=[s["ci95_half_width"] for s in S],
            color=[GOOD if s.get("recommended") else HUMAN for s in S], capsize=3, height=0.6)
    cap = 3600.0 / des["takt_s"]
    a1.axvline(cap, color=BAD, ls="--", lw=1)
    a1.set_yticks(y, [s["name"].replace("_", " ") for s in S], fontsize=8)
    a1.invert_yaxis()
    lo = min(s["throughput_per_hr"] - s["ci95_half_width"] for s in S)
    a1.set_xlim(lo * 0.97, cap * 1.01)
    a1.set_xlabel("doors / hour (95 % CI)   dashed = 3600 / takt")
    a1.set_title("Simulated throughput")
    a1.grid(axis="y", visible=False)
    a2.barh(y, [s["takt_violations_pct"] for s in S], color=[GOOD if s.get("recommended") else ROBOT for s in S], height=0.6)
    a2.set_yticks(y, [""] * len(S))
    a2.invert_yaxis()
    a2.set_xlabel("% station-cycles longer than takt")
    a2.set_title("Takt overruns")
    a2.grid(axis="y", visible=False)
    return uri(fig)


# ------------------------------------------------------------------ s7
def cashflow_chart(roi):
    S = [s for s in roi["scenarios"] if s["capex_usd"] > 0]
    if not S:
        return None
    n = int(S[0]["crf"]["n"])
    i = S[0]["crf"]["i"]
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    for s in S:
        cum = [-s["capex_usd"]]
        for t in range(1, n + 1):
            cum.append(cum[-1] + s["annual_savings_usd"] / (1 + i) ** t)
        ax.plot(range(n + 1), np.array(cum) / 1000, "o-", ms=4, lw=2.2 if s["recommended"] else 1.2,
                color=GOOD if s["recommended"] else MUTED, label=s["name"].replace("_", " "))
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xlabel("year")
    ax.set_ylabel("cumulative discounted cash flow [k USD]")
    ax.set_title("Discounted cash flow (NPV at year n)")
    ax.legend(fontsize=7.5)
    return uri(fig)


def tornado_chart(roi):
    sens = roi.get("sensitivity", [])
    if not sens:
        return None
    rec = next((s for s in roi["scenarios"] if s["recommended"]), None)
    base = rec["npv_usd"] if rec else 0
    fig, ax = plt.subplots(figsize=(5.4, 0.42 * len(sens) + 1.1))
    for k, r in enumerate(sens[::-1]):
        lo, hi = r["npv_low"], r["npv_high"]
        ax.barh(k, (lo - base) / 1000, left=base / 1000, color=HUMAN, height=0.6)
        ax.barh(k, (hi - base) / 1000, left=base / 1000, color=ROBOT, height=0.6)
    ax.set_yticks(range(len(sens)), [r["param"].replace("_", " ") for r in sens[::-1]], fontsize=8)
    ax.axvline(base / 1000, color=INK, lw=1)
    ax.set_xlabel("NPV [k USD]  (blue = low value, orange = high value)")
    ax.set_title("Tornado: NPV sensitivity (recommended)")
    ax.grid(axis="y", visible=False)
    return uri(fig)


# ------------------------------------------------------------------ s1
BONES = [("Hips", "Spine"), ("Spine", "Spine1"), ("Spine1", "Spine2"), ("Spine2", "Spine3"), ("Spine3", "Neck"),
         ("Spine2", "Neck"), ("Spine1", "Neck"), ("Spine", "Neck"), ("Neck", "Head"),
         ("Neck", "LeftShoulder"), ("LeftShoulder", "LeftArm"), ("LeftArm", "LeftForeArm"), ("LeftForeArm", "LeftHand"),
         ("LeftShoulder", "LeftElbow"), ("LeftElbow", "LeftWrist"),
         ("Neck", "RightShoulder"), ("RightShoulder", "RightArm"), ("RightArm", "RightForeArm"),
         ("RightForeArm", "RightHand"), ("RightShoulder", "RightElbow"), ("RightElbow", "RightWrist"),
         ("Hips", "LeftUpLeg"), ("LeftUpLeg", "LeftLeg"), ("LeftLeg", "LeftFoot"), ("Hips", "LeftHip"),
         ("LeftHip", "LeftKnee"), ("LeftKnee", "LeftAnkle"),
         ("Hips", "RightUpLeg"), ("RightUpLeg", "RightLeg"), ("RightLeg", "RightFoot"), ("Hips", "RightHip"),
         ("RightHip", "RightKnee"), ("RightKnee", "RightAnkle")]


def skeleton_strip(J, names, fps, title, n=6):
    T = J.shape[0]
    idx = np.linspace(0, T - 1, n).astype(int)
    fig, axes = plt.subplots(1, n, figsize=(10.5, 2.6), sharey=True)
    pairs = [(names.index(a), names.index(b)) for a, b in BONES if a in names and b in names]
    zmin = np.nanmin(J[:, :, 2])
    for ax, t in zip(axes, idx):
        P = J[t]
        c = P[:, :2].mean(0)
        for a, b in pairs:
            ax.plot([P[a, 0] - c[0], P[b, 0] - c[0]], [P[a, 2] - zmin, P[b, 2] - zmin], color=HUMAN, lw=1.6)
        ax.scatter(P[:, 0] - c[0], P[:, 2] - zmin, s=5, color=INK, zorder=3)
        ax.set_aspect("equal")
        ax.set_xlim(-0.9, 0.9)
        ax.set_title(f"t = {t / fps:.1f}s", fontsize=8.5)
        ax.set_xticks([])
        ax.grid(False)
    axes[0].set_ylabel("height [m]")
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    return uri(fig)


def trunk_series(J, names, fps, title):
    """Trunk flexion from vertical over the clip (pelvis -> neck vector)."""
    lo = next((n for n in ("Hips", "Pelvis") if n in names), None)
    hi = next((n for n in ("Neck", "Spine3", "Head") if n in names), None)
    if lo is None or hi is None:
        return None
    v = J[:, names.index(hi)] - J[:, names.index(lo)]
    ang = np.degrees(np.arccos(np.clip(v[:, 2] / np.linalg.norm(v, axis=1), -1, 1)))
    t = np.arange(len(ang)) / fps
    fig, ax = plt.subplots(figsize=(10.5, 2.2))
    ax.plot(t, ang, color=HUMAN, lw=1)
    ax.axhspan(20, 60, color=ROBOT, alpha=0.12, label="bent forward 20-60 deg")
    ax.axhspan(60, max(90, ang.max()), color=BAD, alpha=0.10, label="strongly bent > 60 deg")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("trunk flexion [deg]")
    ax.set_title(title)
    ax.legend(fontsize=7.5, loc="upper right")
    return uri(fig)


def overload_chart(al):
    """As-is station cycle at standard time vs takt (with measured/assumption basis) and after the recommendation."""
    base = next(s for s in al["scenarios"] if s["name"] == "baseline_manual")
    rec = next(s for s in al["scenarios"] if s.get("recommended"))
    so = al.get("as_is", {}).get("stations", {})
    st = list(base["station_times"])
    takt = al["params"]["takt_s"]
    fig, ax = plt.subplots(figsize=(10.5, 3.1))
    x = np.arange(len(st))
    b = [base["station_times"][k]["cycle_s"] for k in st]
    r = [rec["station_times"][k]["cycle_s"] for k in st]
    meas = [so.get(k, {}).get("overload_basis") == "measured" for k in st]
    ax.bar(x - 0.2, b, 0.38, color=[BAD if m else "#e8a9a3" for m in meas], label="as-is (standard time)")
    for k, (v, m) in enumerate(zip(b, meas)):
        if v > takt:
            ax.bar(x[k] - 0.2, v - takt, 0.38, bottom=takt, color="none", edgecolor=INK, hatch="////" if not m else None, lw=0)
        ax.text(x[k] - 0.2, v + 2, f"{v / takt - 1:+.1%}\n{'measured' if m else 'assumption'}", ha="center",
                fontsize=8, color=BAD if m else MUTED)
    ax.bar(x + 0.2, r, 0.38, color=GOOD, label=f"after: {rec['name'].replace('_', ' ')}")
    for k, v in enumerate(r):
        ax.text(x[k] + 0.2, v + 2, f"{v:.0f}s", ha="center", fontsize=8, color=GOOD)
    ax.axhline(takt, color=INK, ls="--", lw=1)
    ax.text(len(st) - 0.5, takt + 1, f"takt {takt:.0f} s", ha="right", va="bottom", fontsize=8)
    ax.set_xticks(x, st)
    ax.set_ylim(0, max(b + r) * 1.18)
    ax.set_ylabel("station cycle [s]")
    ax.set_title("As-is station load at standard time vs takt, and after the recommended robots")
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=BAD), plt.Rectangle((0, 0), 1, 1, color="#e8a9a3"),
               plt.Rectangle((0, 0), 1, 1, color=GOOD)],
              ["as-is, measured basis", "as-is, assumption basis (mostly unobserved)",
               f"after: {rec['name'].replace('_', ' ')}"], fontsize=8, loc="lower right", framealpha=0.9, frameon=True)
    ax.grid(axis="x", visible=False)
    return uri(fig)


# ------------------------------------------------------------------ s5x design space
def design_space_chart(ds):
    D = ds["designs"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.6, 4.6), gridspec_kw={"width_ratios": [1.35, 1]})
    dom = [d for d in D if not d["pareto_screen"]]
    scr = [d for d in D if d["pareto_screen"] and not d.get("pareto_final")]
    fin = [d for d in D if d.get("pareto_final")]
    k = lambda xs, m: [d["metrics"][m] / 1e3 for d in xs]  # noqa: E731
    a.scatter(k(dom, "capex_usd"), k(dom, "npv_usd"), s=7, c="#c9ced6", lw=0, label=f"dominated ({len(dom)})")
    a.scatter(k(scr, "capex_usd"), k(scr, "npv_usd"), s=22, c=HUMAN, lw=0, label=f"screen Pareto only ({len(scr)})")
    a.scatter(k(fin, "capex_usd"), k(fin, "npv_usd"), s=46, c=ROBOT, marker="D", edgecolor="white", lw=0.6,
              label=f"final Pareto after DES ({len(fin)})")
    named = sorted([d for d in D if d["named_scenarios"]], key=lambda d: d["metrics"]["capex_usd"])
    for j, d in enumerate(named):   # stagger labels of designs that sit close together
        a.annotate(", ".join(d["named_scenarios"]).replace("_", " "), (d["metrics"]["capex_usd"] / 1e3,
                   d["metrics"]["npv_usd"] / 1e3), xytext=(-60 if d["metrics"]["capex_usd"] > 350e3 else 6, 6 + 11 * (j % 3)),
                   textcoords="offset points", fontsize=7.5, color=INK,
                   arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.5})
    a.axhline(0, color=MUTED, lw=0.8)
    a.set_xlabel("installed cost (k USD)")
    a.set_ylabel("NPV vs as-is (k USD)")
    a.set_title(f"{len(D)} unique designs from {ds['counts']['factor_runs']} factor runs")
    a.legend(loc="lower left", fontsize=8)
    top = sorted(fin, key=lambda d: -(d["smaa"]["rank1"] + d["smaa"]["rank2"] + d["smaa"]["rank3"]))[:10][::-1]
    y = np.arange(len(top))
    r1 = np.array([d["smaa"]["rank1"] for d in top]) * 100
    r2 = np.array([d["smaa"]["rank2"] for d in top]) * 100
    r3 = np.array([d["smaa"]["rank3"] for d in top]) * 100
    b.barh(y, r1, color=ROBOT, label="ranked 1st")
    b.barh(y, r2, left=r1, color="#f2b179", label="2nd")
    b.barh(y, r3, left=r1 + r2, color="#f8dcc0", label="3rd")
    b.set_yticks(y, [d["id"] + ("  *" if d["id"] == ds.get("recommended_design") else "") for d in top])
    b.set_xlabel("share of all possible criterion weightings (%)")
    b.set_title("Rank acceptability (SMAA-2)")
    b.legend(loc="lower right", fontsize=8)
    b.grid(axis="y", visible=False)
    fig.tight_layout()
    return uri(fig)
