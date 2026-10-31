"""s8 - self-contained HTML report (out/s8/report.html) and RFQ packages (out/s8/rfq/<station>.md).

Every number shown is read from the out/ JSON files at render time (they are also embedded verbatim in
<script type="application/json" id="rcd-data">). Figures are inline base64 PNG/JPEG; no external resources.
"""
from __future__ import annotations

import datetime as dt
import html
import json
import traceback
from pathlib import Path

import numpy as np

from rcd.common.io import ROOT, out_dir
from rcd.s5_optimize.inputs import UPSTREAM, load_input
from rcd.s8_report import ds_app as DS
from rcd.s8_report import figures as F
from rcd.s8_report.frames import contact_sheet, station_frames
from rcd.s8_report.rfq import build_rfqs


def e(x):
    return html.escape(str(x))


def n(x, d=1, unit=""):
    if x is None:
        return "-"
    if isinstance(x, (int, np.integer)) and d == 0:
        return f"{x:,}{unit}"
    try:
        return f"{float(x):,.{d}f}{unit}"
    except (TypeError, ValueError):
        return e(x)


def usd(x, d=0):
    if x is None:
        return "-"
    return ("-" if x < 0 else "") + f"${abs(x):,.{d}f}"


def pct(x, d=0):
    return "-" if x is None else f"{100 * x:.{d}f}%"


def table(head, rows, cls=""):
    h = "".join(f"<th>{e(c)}</th>" for c in head)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tw"><table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def img(uri, alt, cap=None, cls="fig"):
    if not uri:
        return ""
    c = f"<figcaption>{cap}</figcaption>" if cap else ""
    return f'<figure class="{cls}"><img src="{uri}" alt="{e(alt)}">{c}</figure>'


def chip(src):
    cls = "real" if src == "real" else ("fix" if "fixture" in str(src) else "miss")
    return f'<span class="chip {cls}">{e(src)}</span>'


def section(sid, kicker, title, lead, body, prov=None):
    pv = ""
    if prov:
        pv = '<div class="prov">source: ' + " ".join(f"{e(k)} {chip(v)}" for k, v in prov.items()) + "</div>"
    return (f'<section id="{sid}"><div class="kicker">{e(kicker)}</div><h2>{e(title)}</h2>'
            f'<p class="lead">{lead}</p>{pv}{body}</section>')


def safe(fn, sid, kicker, title):
    try:
        return fn()
    except Exception as ex:  # a broken upstream must not kill the whole report
        tb = traceback.format_exc(limit=2)
        return section(sid, kicker, title, f"Section could not be rendered: {e(type(ex).__name__)}: {e(ex)}",
                       f"<pre class='err'>{e(tb)}</pre>")


CSS = """
:root{--ink:#1b2a3a;--muted:#5f6b7a;--line:#e3e7ec;--bg:#f6f7f9;--card:#fff;--accent:#1f9d7a;--human:#3b6ea8;
--robot:#e07a2f;--bad:#c8443b;--navy:#0f2235}
*{box-sizing:border-box}html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,
"Helvetica Neue",Arial,sans-serif;-webkit-font-smoothing:antialiased}
header.hero{background:linear-gradient(135deg,#0f2235 0%,#163a55 60%,#1c4f63 100%);color:#fff;padding:56px 24px 40px}
.wrap{max-width:1120px;margin:0 auto;padding:0 24px}
.hero h1{font-size:36px;line-height:1.15;margin:8px 0 10px;letter-spacing:-.02em;max-width:900px}
.hero .sub{color:#c9d6e2;font-size:16px;max-width:820px}
.eyebrow{text-transform:uppercase;letter-spacing:.14em;font-size:12px;color:#8fd3c1;font-weight:600}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin-top:28px}
.kpi{background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.14);border-radius:12px;padding:14px 16px}
.kpi .v{font-size:24px;font-weight:700;letter-spacing:-.01em}.kpi .l{font-size:12px;color:#b9c7d4;margin-top:2px}
nav.toc{position:sticky;top:0;z-index:5;background:rgba(246,247,249,.94);backdrop-filter:blur(6px);
border-bottom:1px solid var(--line)}
nav.toc .wrap{display:flex;gap:4px;overflow-x:auto;padding-top:8px;padding-bottom:8px}
nav.toc a{white-space:nowrap;color:var(--muted);text-decoration:none;font-size:13px;padding:5px 10px;border-radius:999px}
nav.toc a:hover{background:#e8ecf0;color:var(--ink)}
main .wrap>section{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:30px 32px;
margin:22px 0;box-shadow:0 1px 2px rgba(16,34,53,.04)}
h2{font-size:24px;margin:2px 0 8px;letter-spacing:-.01em}h3{font-size:16px;margin:26px 0 8px}
.kicker{font-size:12px;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:var(--accent)}
.lead{color:var(--muted);max-width:880px;margin:0 0 14px}
.prov{font-size:12px;color:var(--muted);margin:-4px 0 16px}
.chip{display:inline-block;font-size:11px;font-weight:600;border-radius:999px;padding:1px 8px;margin-right:6px}
.chip.real{background:#dff3ec;color:#13795d}.chip.fix{background:#fdf0e2;color:#a85a14}.chip.miss{background:#f7e1df;color:#9b2f27}
figure{margin:14px 0}figure img{max-width:100%;border-radius:8px;border:1px solid var(--line);display:block}
figcaption{font-size:12.5px;color:var(--muted);margin-top:6px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:18px}.grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
@media(max-width:820px){.grid2,.grid3{grid-template-columns:1fr}.hero h1{font-size:28px}}
.tw{overflow-x:auto;margin:10px 0 6px}
table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}
th{text-align:left;font-weight:600;color:var(--muted);border-bottom:2px solid var(--line);padding:7px 10px;white-space:nowrap}
td{border-bottom:1px solid var(--line);padding:6px 10px;vertical-align:top}
tr.rec td{background:#eef8f4;font-weight:600}
.callout{border-left:4px solid var(--accent);background:#f1f9f6;padding:14px 18px;border-radius:0 10px 10px 0;margin:14px 0}
.callout.warn{border-color:#d58a2b;background:#fdf6ec}.callout.red{border-color:var(--bad);background:#fbefee}
.formula{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px;background:#f3f5f8;border-radius:8px;
padding:10px 14px;white-space:pre-wrap;color:#24384d}
pre{font-size:11.5px;background:#0f2235;color:#d6e2ee;padding:14px;border-radius:10px;overflow-x:auto;max-height:340px}
pre.err{background:#fbefee;color:#7a231c}
.stat{display:flex;gap:26px;flex-wrap:wrap;margin:12px 0}.stat div{min-width:120px}
.stat .v{font-size:22px;font-weight:700}.stat .l{font-size:12px;color:var(--muted)}
.big{font-size:20px;line-height:1.45;font-weight:600;color:var(--ink)}
ul.tight{margin:6px 0;padding-left:20px}ul.tight li{margin:3px 0}
.cite{font-size:12px;color:var(--muted)}
.frames img{aspect-ratio:16/9;object-fit:cover}
footer{color:var(--muted);font-size:12px;text-align:center;padding:30px 0 50px}
"""


def s2fig(name):
    import base64
    p = ROOT / "out" / "s2" / "figures" / name
    if not p.exists():
        return None
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()


def mounting_block(D, rec, cfg):
    """Robot mounting decision and the reach requirement that drives model choice (from s1, s3, s4, s5)."""
    cand, cat, sg = D.get("candidates"), D.get("robot_catalog"), D.get("scene_graph")
    if not cand or not rec["robots"]:
        return ""
    groups = {g["group_id"]: g for g in cand["task_groups"]}
    robots = {r["model"]: r for r in (cat or {}).get("robots", [])}
    rm = cfg["robots"]["reach_margin"]
    belt, feet = {}, {}
    for st in (sg or {}).get("stations", []):
        be = st.get("belt_edges_y_m")
        if be:
            belt[st["station"]] = abs(be[1] - be[0])
        fr = [c.get("registration", {}).get("check_work_feet_on_belt_frac") for c in st["cycles"]]
        fr = [x for x in fr if x is not None]
        if fr:
            feet[st["station"]] = float(np.mean(fr))
    rows, small = [], []
    for rb in rec["robots"]:
        for gid in rb["groups"]:
            g = groups.get(gid, {})
            pl = g.get("placement", {}) or {}
            req = g.get("requirements", {}) or {}
            ff = (pl.get("fixed_floor_alternative") or {}).get("required_reach_mm")
            r = robots.get(rb["model"], {})
            usable = r.get("reach_mm", 0) * rm
            s3r = (req.get("s3_requirements") or {}).get("reach_mm")
            rows.append([e(gid), e(g.get("mounting", "-")), n(req.get("reach_mm"), 0), n(ff, 0), n(s3r, 0),
                         f"{e(rb['model'])}: {n(usable, 0)}", n(usable - (req.get("reach_mm") or 0), 0),
                         n(pl.get("tracking_window_m"), 2),
                         f"<span class='cite'>{e(pl.get('mounting_reason', ''))}</span>"])
            if s3r:
                pay = req.get("payload_kg", 0) * (1 + cfg["robots"]["safety_margin_payload"])
                q = [m for m, x in robots.items() if x.get("reach_mm", 0) * rm >= s3r and x.get("payload_kg", 0) >= pay
                     and x.get("conveyor_tracking")]
                small.append((gid, s3r, q))
    bw = ", ".join(f"{k} {n(v, 2)} m" for k, v in belt.items())
    fb = ", ".join(f"{k} {pct(v)}" for k, v in feet.items())
    sm = "".join(f"<li>{e(gid)}: from the operator's position the reach need is about {n(r_, 0)} mm, which "
                 f"{len(q)} catalog models meet ({e(', '.join(q[:8]))}{'...' if len(q) > 8 else ''}).</li>"
                 for gid, r_, q in small)
    return (f"<h3>Robot mounting and the reach requirement</h3>"
            f"<p>The operator works standing on the moving belt (belt width from s1: {e(bw)}; share of working "
            f"frames with the feet on the belt: {e(fb)}). A robot cannot stand where the operator stands, so it is "
            f"placed off the belt: on the floor where the tracking window is short, otherwise on a 7th-axis "
            f"linear track synchronised to the conveyor (priced in the capex). The required reach from that base, "
            f"including tool offset and the door's travel during the robot program, is what selects the model.</p>"
            + table(["task group", "mounting", "required reach mm (chosen base)", "fixed-floor alternative mm",
                     "reach from operator position mm (s3)", "usable reach mm", "margin mm", "tracking window m",
                     "why (s4)"], rows)
            + "<div class='callout warn'><b>Future alternative: on-platform mounting.</b> A cobot riding on the "
              "belt platform next to the operator moves with the door, removes the track and the tracking window, "
              f"and needs only the short reach measured from the operator's position:<ul class='tight'>{sm}</ul>"
              "It needs a platform-mounted controller and power/air slip, and a re-validated risk assessment.</div>")


def load_all(cfg):
    prov, D = {}, {}
    for name in UPSTREAM:
        D[name] = load_input(cfg, name, prov, required=False)
    return D, prov


def compact(obj, max_list=3000):
    if isinstance(obj, dict):
        return {k: compact(v, max_list) for k, v in obj.items()}
    if isinstance(obj, list):
        if len(obj) > max_list:
            return [compact(v, max_list) for v in obj[:max_list]] + [f"... truncated {len(obj) - max_list} items"]
        return [compact(v, max_list) for v in obj]
    return obj


def run(cfg: dict) -> None:
    D, prov = load_all(cfg)
    al, des, roi = D["allocation"], D["des_results"], D["roi"]
    takt = cfg["line"]["takt_s"]
    rec = next(s for s in al["scenarios"] if s.get("recommended"))
    base = next(s for s in al["scenarios"] if s["name"] == "baseline_manual")
    rroi = next((s for s in roi["scenarios"] if s["name"] == rec["name"]), None)
    rdes = next((s for s in des["scenarios"] if s["name"] == rec["name"]), None)
    bdes = next((s for s in des["scenarios"] if s["name"] == "baseline_manual"), None)
    P = lambda *ks: {k: prov.get(k) for k in ks}  # noqa: E731
    secs = []

    def headline_html():
        """The 3 options the client chooses between (s5x headline); the full design space stays below."""
        ds = D.get("design_space")
        if not ds or not ds.get("headline"):
            return ""
        H = ds["headline"]
        lab = {k["key"]: k["label"] for k in ds["criteria"]}
        by = {d["id"]: d for d in ds["designs"]}
        fm = lambda k, v: (usd(v) if k.endswith("usd") else str(v) if k == "n_robots"  # noqa: E731
                           else pct(v / 100, 1) if k == "takt_violation_pct" else n(v, 1, " s"))
        cards = []
        for i, o in enumerate(H["options"]):
            d, m = by[o["design"]], o["metrics"]
            rows = "".join(
                f"<tr><td>{e(lab[c['key']])}</td><td style='text-align:right'>"
                + ("<b>" if c["key"] in o["strengths"] else "") + fm(c["key"], m[c["key"]])
                + ("</b> &#9650;" if c["key"] in o["strengths"] else " &#9660;" if c["key"] in o["weaknesses"] else "")
                + "</td></tr>" for c in ds["criteria"])
            does = "".join(f"<li><b>{e(r['model'])}</b> @ {e(r['station'])}: {e(', '.join(r['tasks']))}</li>"
                           for r in d["robots"]) or "<li>no robot: AMR kitting / point-of-use supply and fixtures</li>"
            cards.append(
                f"<div class='hl{' rec' if o['is_recommended'] else ''}'><div class='hl-k'>Option {'ABC'[i]}"
                + (" &middot; current recommendation" if o["is_recommended"] else "") + f"</div><h4>{e(o['title'])}</h4>"
                f"<div class='hl-who'>Best fit for {pct(o['share_of_weightings'])} of all priority weightings"
                + (f"; first choice of: {e(', '.join(o['personas']))}" if o["personas"] else "") + "</div>"
                f"<table>{rows}<tr><td>Payback</td><td style='text-align:right'>"
                f"{n(m['payback_years'], 1, ' yr') if m.get('payback_years') else '-'}</td></tr>"
                f"<tr><td>Throughput (DES)</td><td style='text-align:right'>{n(m.get('throughput_per_hr'), 2, '/h')}</td></tr></table>"
                f"<ul class='tight cite'>{does}</ul><div class='cite'>{e(o['design'])}; similar variants: "
                f"{e(', '.join(x for x in o['variants_in_archetype'] if x != o['design']) or 'none')}</div></div>")
        pr = "".join(f"<li>{e(p['a'])} vs {e(p['b'])}: {e(p['verdict'])}; throughput {p['throughput_diff_per_hr']:+.2f} "
                     f"&plusmn; {p['throughput_diff_ci95']:.2f} doors/h</li>" for p in H["pairs"])
        ns = H["not_shown"]
        return (f"<div class='hl-grid'>{''.join(cards)}</div>"
                f"<p class='cite'>&#9650; best of the three, &#9660; worst of the three. Together the three options contain the "
                f"best practical design for {pct(H['coverage'])} of all priority weightings. Not shown as options: "
                f"{ns['relief_operator']} designs that need an extra relief operator, {ns['negative_npv']} with NPV &le; 0, "
                f"{ns['same_archetype_variants']} near-identical variants"
                + (f", and the {', '.join(str(a) + '-robot' for a in ns['archetypes_not_shown'])} archetype "
                   f"(best for only {pct(1 - H['coverage'])} of weightings; in the explorer)" if ns["archetypes_not_shown"] else "")
                + ".</p><ul class='tight cite'>" + pr + "</ul>")

    # ------------------------------------------------------------------ hero + exec summary
    ob, orc = base["objective"], rec["objective"]
    bent_red = (ob["awkward_posture_s_per_cycle"] - orc["awkward_posture_s_per_cycle"]) / ob["awkward_posture_s_per_cycle"] \
        if ob["awkward_posture_s_per_cycle"] else None
    buys = ", ".join(f"{r['model']} @ {r['station']}" for r in rec["robots"]) or "no robot"
    levers = [l for l in rec.get("layout_changes", []) if l["station"] != "LINE"]
    if levers:
        buys += " + layout levers: " + ", ".join(f"{l['lever'].replace('_', ' ')} @ {l['station']}" for l in levers)
    asis = al.get("as_is", {})
    head_st = asis.get("headline_stations", [])
    so = asis.get("stations", {})
    if head_st:
        w = max(head_st, key=lambda k: so[k]["overload_pct"])
        ov_kpi = (f"+{pct(so[w]['overload_pct'], 1)}", f"{w} over takt today (measured, standard time)")
    else:
        ov_kpi = ("-", "no measured station overload")
    kpis = [ov_kpi,
            (n(orc.get("max_station_cycle_s"), 0, " s"), f"max station cycle after robots (takt {n(takt, 0)} s)"),
            (str(len(rec["robots"])), "cobot cells recommended"),
            (n(ob["human_s_per_cycle"] - orc["human_s_per_cycle"], 1, " s"), "operator work removed per door"),
            (pct(bent_red), "less awkward-posture time"),
            (n(rdes["throughput_per_hr"] - bdes["throughput_per_hr"], 2, "/h") if rdes and bdes else "-",
             "throughput vs today (DES)"),
            (usd(rroi["npv_usd"]) if rroi else "-", f"NPV vs as-is labour ({rroi['crf']['n'] if rroi else '-'} yr)")]
    fixture_used = [k for k, v in prov.items() if v and "fixture" in v]
    hero = f"""<header class="hero"><div class="wrap"><div class="eyebrow">Robot automation assessment</div>
<h1>From factory video to a robot purchase decision</h1>
<div class="sub">CarDA car-door assembly line (Stellantis, Melfi): stations {e(", ".join(base["station_times"]))},
paced conveyor at takt {n(takt, 0)} s. Video and motion capture are turned into work elements, time standards, ergonomics,
automatability, robot screening and simulation, then an optimised human-robot allocation, a line simulation and a business case.</div>
<div class="kpis">{"".join(f'<div class="kpi"><div class="v">{v}</div><div class="l">{e(l)}</div></div>' for v, l in kpis)}</div>
</div></header>"""
    toc_items = [("sec-summary", "Summary"), ("sec-s0", "Video & calibration"), ("sec-s1", "4D human"),
                 ("sec-s2", "Work study"), ("sec-s3", "Automatability"), ("sec-s4", "Robot screening"),
                 ("sec-s6a", "Robot sim"), ("sec-s5", "Allocation & balancing"), ("sec-s6b", "Line simulation"),
                 ("sec-s7", "ROI"), ("sec-s5x", "Design space"), ("sec-s9", "Robot program"), ("sec-rec", "Recommendation"),
                 ("sec-limits", "Limitations")]
    toc = '<nav class="toc"><div class="wrap">' + "".join(f'<a href="#{a}">{e(b)}</a>' for a, b in toc_items) + "</div></nav>"

    def summary():
        fx = ""
        if fixture_used:
            fx = (f'<div class="callout warn"><b>Development fixtures in use</b> for: {e(", ".join(fixture_used))}. '
                  "Numbers from these inputs are placeholders until the real stage outputs exist.</div>")
        econ = ""
        if rroi:
            pb = rroi["payback_years"]
            econ = (f"Installed cost {usd(rroi['capex_usd'])}; net annual saving {usd(rroi['annual_savings_usd'])} "
                    f"({n(rroi['labour_hours_freed_per_yr'], 0)} labour h/yr freed); payback "
                    f"{n(pb, 1, ' yr') if pb else 'not reached'}; NPV {usd(rroi['npv_usd'])} at "
                    f"{pct(rroi['crf']['i'])} over {rroi['crf']['n']} yr.")
        orows = []
        for st, v in so.items():
            orows.append([e(st), n(v["standard_work_s"], 1), n(v["takt_s"], 0), pct(v["utilization"], 1),
                          (f"<b style='color:#c8443b'>+{pct(v['overload_pct'], 1)}</b>" if v["overload_basis"] == "measured"
                           else f"+{pct(v['overload_pct'], 1)}") if v["overload_pct"] > 0 else "-",
                          e(v["overload_basis"]), pct(v.get("observed_fraction")),
                          n(rec["station_times"][st]["cycle_s"], 1),
                          e(", ".join(f"{r['model']}: " + ", ".join(x.split(":")[1] for x in r["elements"])
                                      for r in rec["robots"] if r["station"] == st) or "-")])
        rv = ""
        if rroi and rroi.get("vs_manual_relief"):
            vr = rroi["vs_manual_relief"]
            rv = (f" Against the all-manual fix (a relief operator per shift so every station meets takt at standard "
                  f"time) the robots save {usd(vr['annual_savings_usd'])}/yr, NPV {usd(vr['npv_usd'])}, payback "
                  f"{n(vr['payback_years'], 1, ' yr') if vr['payback_years'] else 'not reached'}.")
        hl = headline_html()
        hl = ("<h3>Three options to choose from</h3><p>Every robot combination was built and scored (see Design space); "
              "these three cover the real choices. Our recommendation is marked, but the trade-off is yours.</p>" + hl) if hl else ""
        body = f"""{fx}<p class="big">Recommended: {e(buys)}.</p>{hl}
<h3>Finding: the as-is line is overloaded at standard time</h3>
<p>Standard time = observed time x performance rating x (1 + PFD allowance), plus the unobserved part of the cycle at elapsed
time. A station is only called overloaded in the headline when its cycle is mostly observed (basis = measured); stations
whose load rests on the unobserved block are shown, but marked as assumption.</p>
{img(F.overload_chart(al), "overload")}
{table(["station", "standard work s", "takt s", "utilisation", "overload", "basis", "observed share of cycle",
        "cycle after robots s", "robot takes"], orows)}
<p>{e(rec.get('justification', ''))}</p><p>{econ}{rv}</p>{mounting_block(D, rec, cfg)}"""
        return section("sec-summary", "Executive summary", "What the data says", "One page answer; every figure "
                       "below is traceable to a stage output embedded in this file.", body,
                       P("allocation", "des_results", "roi"))
    secs.append(safe(summary, "sec-summary", "Executive summary", "What the data says"))

    # ------------------------------------------------------------------ s0
    def s0():
        cal = D["calibration"]
        fr = station_frames(cfg)
        cs = contact_sheet(cfg, station=(fr[-1]["station"] if fr else "WS30"))
        grid = '<div class="grid3 frames">' + "".join(
            img(f["uri"], f"{f['station']} {f['side']}", f"{e(f['station'])} {e(f['side'])}-camera, "
                f"{e(f['file'])}" + (f" @ {n(f['t_s'], 1)} s" if f["t_s"] is not None else "")) for f in fr) + "</div>"
        rows = []
        for c in cal["cameras"] if cal else []:
            rows.append([e(c["id"]), e(c["station"]), e(c["side"]), f"{c['image_size'][0]}x{c['image_size'][1]}",
                         n(c["K"][0][0], 1), n(c["K"][1][1], 1), n(c.get("reproj_rmse_px"), 2),
                         e((c.get("source") or "")[:120] + ("..." if len(c.get("source") or "") > 120 else ""))])
        body = grid + (img(cs["uri"], "contact sheet", f"Frames across one cycle clip ({e(cs['file'])})") if cs else "")
        body += "<h3>Camera calibration</h3>" + table(["camera", "station", "side", "image", "fx", "fy",
                                                       "reproj. RMSE px", "source / verification"], rows)
        if cal:
            body += f'<p class="cite">World frame: {e(cal["world_frame"].get("origin_desc", ""))}</p>'
        return section("sec-s0", "Stage 0 - input", "Factory video and camera calibration",
                       "Two ZED cameras per station (in/out side of the line). Calibration places every camera in one "
                       "metric plant frame so that motion, parts and robot positions share coordinates.", body,
                       P("calibration"))
    secs.append(safe(s0, "sec-s0", "Stage 0", "Video & calibration"))

    # ------------------------------------------------------------------ s1
    def s1():
        sg = D["scene_graph"]
        rows, sk, tr = [], "", ""
        for s in sg["stations"]:
            sp = [c["conveyor"]["speed_mm_s"] for c in s["cycles"]]
            du = [c["duration_s"] for c in s["cycles"]]
            rows.append([e(s["station"]), len(s["cycles"]), n(np.mean(du), 1), n(np.mean(sp), 1),
                         e(s["cycles"][0]["conveyor"].get("method", "")), e(s["cycles"][0]["human"].get("source", ""))])
            if not sk:
                for c in s["cycles"]:
                    p = ROOT / c["human"]["joints_npz"] if c["human"].get("joints_npz") else None
                    if p and p.exists() and p.is_file():
                        d = np.load(p)
                        J = d["joints"]
                        names = list(c["human"]["joint_names"])
                        fps = c["human"]["fps"]
                        sk = F.skeleton_strip(J, names, fps, f"{s['station']} cycle {c['cycle_id']}: 3D skeleton "
                                                             f"(side view, world frame)")
                        tr = F.trunk_series(J, names, fps, f"{s['station']} cycle {c['cycle_id']}: trunk flexion over the clip")
                        break
        body = table(["station", "cycles", "mean clip s", "conveyor mm/s", "conveyor method", "human source"], rows)
        body += img(sk, "skeleton") + img(tr, "trunk flexion")
        return section("sec-s1", "Stage 1 - 4D human & scene", "Worker motion in 3D, over time",
                       "Motion capture (Xsens BVH) is registered into the plant frame; the conveyor speed is measured "
                       "so the robot can be specified for line tracking.", body, P("scene_graph"))
    secs.append(safe(s1, "sec-s1", "Stage 1", "4D human"))

    # ------------------------------------------------------------------ s2
    def s2():
        el, ts, erg = D["elements"], D["time_study"], D["ergonomics"]
        tax = {t["code"]: t for t in el["taxonomy"]}
        body = img(F.gantt(el), "gantt")
        body += "<h3>Element taxonomy</h3>" + table(["code", "name", "therblig", "description"],
                                                     [[e(t["code"]), e(t["name"]), e(t["therblig"]), e(t["description"])]
                                                      for t in el["taxonomy"]])
        body += img(F.element_time_bars(ts), "element times")
        rows = [[e(r["station"]), e(r["code"]), r["n"], n(r["mean_s"], 2), n(r["std_s"], 2), n(r["normal_s"], 2),
                 n(r["standard_s"], 2), r["n_required"]] for r in ts["element_stats"]]
        body += "<h3>Time study (per-cycle totals)</h3>" + table(
            ["station", "code", "n", "mean s", "s.d. s", "normal s", "standard s", "n required"], rows)
        pr = ts["params"]
        body += (f'<div class="formula">normal time = observed x rating ({n(pr.get("performance_rating"), 2)});  '
                 f'standard time = normal x (1 + PFD allowance {pct(pr["allowance_pfd"])});  '
                 f'n required = (z s / (a x mean))^2 with a = {pct(pr["accuracy"])}, confidence {pct(pr["confidence"])}</div>')
        rt = ts.get("rating", {})
        if rt:
            body += "<h3>Performance rating & allowances</h3><div class='grid2'><div>" + table(
                ["Westinghouse factor", "value"], [[e(k), n(v, 3)] for k, v in rt.get("factors", {}).items()]) + \
                "</div><div>" + table(["ILO allowance", "fraction"], [[e(k), n(v, 3)] for k, v in
                                                                       ts.get("allowance_breakdown", {}).items()]) + "</div></div>"
        pm = ts.get("pmts", {})
        if pm:
            body += f"<h3>Predetermined motion times ({e(pm.get('system'))})</h3>" + table(
                ["code", "TMU", "PMTS s", "observed s"],
                [[e(r["code"]), n(r["tmu"], 0), n(r["pmts_s"], 2), n(r.get("observed_s"), 2)] for r in pm["rows"]])
        so_ = (al.get("as_is") or {}).get("stations", {})
        meas = [k for k, v in so_.items() if v.get("overload_basis") == "measured"]
        spc_note = ("Individuals chart of station cycle time per station; UCL = mean + 2.66 x mean moving range; "
                    "Cpu = (takt - mean) / (3 sigma), sigma = MR-bar / 1.128. Capability is meaningful only where the "
                    "cycle is observed: " + (", ".join(meas) if meas else "none") + " (measured basis); the other "
                    "stations' cycle times are dominated by the unobserved block.")
        imr = s2fig("imr_charts.png")
        body += img(imr or F.spc_chart(ts), "SPC", e(spc_note))
        sp = [(stn, s2fig(f"spaghetti_{stn}.png")) for stn in so_ or []]
        sp = [(a_, b_) for a_, b_ in sp if b_]
        if sp:
            body += "<h3>Spaghetti diagrams (operator travel per cycle)</h3><div class='grid3'>" + "".join(
                img(u, f"spaghetti {stn}", f"{e(stn)} (s2 figure)") for stn, u in sp) + "</div>"
        if (ROOT / "docs" / "IE_METHODS.md").exists():
            body += "<p class='cite'>Method notes for the work study: docs/IE_METHODS.md (repository).</p>"
        ln = ts.get("lean", {})
        if ln:
            body += (f"<h3>Lean value-stream view</h3><div class='stat'><div><div class='v'>{pct(ln.get('va_ratio'))}</div>"
                     f"<div class='l'>value-added ratio</div></div></div>" +
                     table(["code", "class", "waste (TIMWOOD)"], [[e(r["code"]), e(r["class"]), e(r.get("waste") or "-")]
                                                                 for r in ln.get("rows", [])]))
        body += "<h3>Ergonomics</h3>" + img(F.ergonomics_chart(erg), "ergonomics")
        v = erg["validation"]
        body += f"<p class='cite'>Method: {e(erg['method'])}. Validation vs annotated EAWS: {e(v['metric'])} = {n(v['value'], 2)} (n = {e(v['n'])}).</p>"
        return section("sec-s2", "Stage 2 - work study", "Work elements, time standards, SPC, lean and ergonomics",
                       "Each cycle is segmented into therblig-like elements; the per-cycle total of every element "
                       "(station:code) is the unit that is scored, assigned and simulated downstream. The part of the "
                       "station cycle that the cropped clips do not show is kept as a manual UNOBS block.", body,
                       P("elements", "time_study", "ergonomics"))
    secs.append(safe(s2, "sec-s2", "Stage 2", "Work study"))

    # ------------------------------------------------------------------ s3
    def s3():
        au = D["automatability"]
        body = "<div class='grid2'><div>" + img(F.ahp_weights(au), "weights") + "</div><div>" + table(
            ["criterion", "weight", "source"], [[e(c["key"]), n(c["weight"], 3), e(c["source"])] for c in au["criteria"]]) + "</div></div>"
        body += img(F.automatability_heat(au), "automatability")
        sens = au.get("sensitivity", {})
        if sens:
            body += f"<p class='cite'>Weight-perturbation rank stability: {n(sens.get('rank_stability'), 2)}.</p>"
        return section("sec-s3", "Stage 3 - automatability", "Which elements a robot should take",
                       "Multi-criteria scoring of every work element; the allocation class (robot / human-robot / "
                       "human / layout change) is what the optimiser is allowed to use.", body, P("automatability"))
    secs.append(safe(s3, "sec-s3", "Stage 3", "Automatability"))

    # ------------------------------------------------------------------ s4
    def s4():
        cat, cand = D["robot_catalog"], D["candidates"]
        rows = [[e(r["model"]), e(r["maker"]), e(r["type"]), n(r["payload_kg"], 1), n(r["reach_mm"], 0),
                 n(r["repeatability_mm"], 3), "yes" if r["collaborative"] else "no", "yes" if r["conveyor_tracking"] else "no",
                 f"{usd(r['price_usd_range'][0])} - {usd(r['price_usd_range'][-1])}"] for r in cat["robots"]]
        body = "<h3>Catalog</h3>" + table(["model", "maker", "type", "payload kg", "reach mm", "repeat. mm", "collab.",
                                           "tracking", "price range"], rows)
        rk = cand.get("ranking", {})
        body += f"<p class='cite'>Ranking: {e(rk.get('method'))}; AHP consistency ratio {n(rk.get('consistency_ratio'), 3)}.</p>"
        tsc = {(gid, r["model"]): r for gid, rows_ in (rk.get("scores") or {}).items() if isinstance(rows_, list)
               for r in rows_ if isinstance(r, dict)}

        def reach(x):
            v = x.get("reachability")
            return v.get("fraction_within_usable_reach") if isinstance(v, dict) else v

        def tops(g, x):
            return x.get("topsis") or tsc.get((g["group_id"], x["model"]), {}).get("topsis")
        for g in cand["task_groups"]:
            ev = sorted(g["evaluated"], key=lambda x: (not x["passed"], -(tops(g, x) or 0)))
            rr = []
            for x in ev:
                cs = x.get("cell_space") or {}
                rr.append([e(x["model"]), "<b style='color:#13795d'>pass</b>" if x["passed"] else "<span style='color:#9b2f27'>fail</span>",
                           n(tops(g, x), 3), n(reach(x), 2),
                           n(cs.get("separation_mm"), 0), n(cs.get("total_m2"), 2),
                           e("; ".join(x["reasons"])[:140])])
            req = g.get("requirements", {})
            body += (f"<h3>{e(g['group_id'])} <span class='cite'>codes {e(', '.join(g['codes']))}; requirements "
                     f"{e(', '.join(f'{k}={v}' for k, v in req.items() if not isinstance(v, (list, dict))))}</span></h3>" +
                     table(["model", "result", "TOPSIS", "reachability", "ISO 13855 S mm", "cell m2", "reasons"], rr))
        return section("sec-s4", "Stage 4 - robot screening", "Commercial robots that can do the job",
                       "Payload (with safety margin), reach, conveyor tracking and reachability screen the catalog; "
                       "AHP-weighted TOPSIS ranks the survivors; ISO 13855 separation distance sizes the cell.", body,
                       P("robot_catalog", "candidates"))
    secs.append(safe(s4, "sec-s4", "Stage 4", "Robot screening"))

    # ------------------------------------------------------------------ s6a
    def s6a():
        rt = D["robot_task_times"]
        body = img(F.robot_times(rt), "robot times")
        body += f"<p class='cite'>Method: {e(rt['method'])}</p>"
        return section("sec-s6a", "Stage 6a - robot simulation", "How long the robot needs",
                       "Kinematic simulation of each candidate on each task group gives the robot cycle time that the "
                       "allocation model uses.", body, P("robot_task_times"))
    secs.append(safe(s6a, "sec-s6a", "Stage 6a", "Robot sim"))

    # ------------------------------------------------------------------ s5
    def s5():
        pa = al["params"]
        rows = []
        for s in al["scenarios"]:
            o = s["objective"]
            rows.append(("rec" if s.get("recommended") else "", [
                e(s["name"]) + (" *" if s.get("recommended") else ""), e(s["status"]),
                e(", ".join(f"{r['model']}@{r['station']}" for r in s["robots"]) or "-"),
                e(", ".join(f"{l['lever']}@{l['station']}" for l in s.get("layout_changes", []) if l["station"] != "LINE") or "-"),
                n(o["human_s_per_cycle"], 1), str(o.get("n_helpers_per_shift", 0)), n(o["awkward_posture_s_per_cycle"], 1),
                usd(o["capex_usd"]),
                usd(o["annual_cost_usd"]), usd(s.get("delta_annual_cost_vs_baseline_usd")),
                " / ".join(n(v["cycle_s"], 1) for v in s["station_times"].values()),
                "yes" if all(s["checks"].values()) else "NO"]))
        head = ["scenario", "status", "robots", "layout levers", "operator s/cycle", "relief ops/shift", "awkward-posture s/cycle", "capex",
                "equiv. annual cost",
                "vs baseline", "station cycles s", "post-checks"]
        tb = '<div class="tw"><table><thead><tr>' + "".join(f"<th>{h}</th>" for h in head) + "</tr></thead><tbody>" + \
             "".join(f'<tr class="{c}">' + "".join(f"<td>{x}</td>" for x in r) + "</tr>" for c, r in rows) + "</tbody></table></div>"
        body = f"""<div class="formula">{e(al.get('model', '').strip())}</div>
<p class='cite'>Parameters: takt {n(pa['takt_s'], 0)} s; {n(pa['annual_cycles'], 0)} doors/yr; loaded wage {usd(pa['loaded_wage_usd_h'], 2)}/h;
A/P({pct(pa['crf']['i'])}, {pa['crf']['n']}) = {n(pa['crf']['value'], 5)}; handover {n(pa['handover_s'], 1)} s per robot element;
ergonomic target: awkward-posture time (trunk bent or arm at/above shoulder) at most {n(pa['ergonomic_cap_s'], 1)} s/cycle ({pct(pa['ergonomic_reduction_target'])} below baseline).</p>
<h3>Allocation scenarios (CP-SAT)</h3>{tb}
<div class="callout"><b>Recommendation rule.</b> {e(al.get('recommendation_rule', ''))}<br>{e(rec.get('justification', ''))}</div>"""
        body += img(F.station_loads(al, takt), "loads")
        lb = al["line_balancing"]
        body += "<h3>Line balancing (SALBP-1)</h3><p>" + e(lb["method"]) + " " + e(lb["interpretation"]) + "</p>"
        body += img(F.balance_chart(lb), "balancing")
        hr = []
        for h in [lb["as_is"]] + lb["heuristics"] + [lb["exact"]]:
            hr.append([e(h["name"]), h["n_stations"], n(h["cycle_time_s"], 0), " / ".join(n(x, 1) for x in h["station_loads_s"]),
                       pct(h["balance_efficiency"], 1), pct(h["balance_delay"], 1), n(h["smoothness_index"], 2),
                       e(h.get("status", h.get("rule", "")))])
        body += table(["method", "stations", "c s", "loads s", "efficiency", "balance delay", "smoothness", "note"], hr)
        s2 = lb.get("salbp2_current_stations", {})
        pa2 = lb.get("post_automation", {})
        body += (f"<div class='formula'>E = sum t / (n c);  d = 1 - E;  SI = sqrt(sum (S_max - S_i)^2);  N_t = ceil(sum t / c) "
                 f"= {lb['theoretical_min_stations']} (work content {n(lb['work_content_s'], 1)} s, {lb['n_tasks']} tasks)\n"
                 f"SALBP-2 with {s2.get('n_stations')} stations: minimum cycle time {n(s2.get('min_cycle_time_s'), 1)} s\n"
                 f"After automation ({e(pa2.get('scenario'))}): human work {n(pa2.get('human_work_s'), 1)} s -> theoretical "
                 f"{pa2.get('theoretical_min_operators')}, exact {pa2.get('exact_operators')} operators per shift</div>")
        body += f"<p class='cite'>Precedence: {e(lb['precedence']['rule'])}</p>"
        sw = F.sweep_chart(lb)
        body += "<div class='grid2'><div>" + img(sw, "sweep") + "</div><div><p class='cite'>" + \
                "<br>".join(e(c) for c in lb["citations"]) + "</p></div></div>"
        wm = al.get("worker_machine", [])
        body += "<h3>Worker-machine (operator + cobot) chart</h3>" + img(F.worker_machine_chart(wm), "wm")
        body += table(["station", "robot", "human s", "robot s", "handover s", "cycle s", "operator idle", "robot idle",
                       "slack to takt s"],
                      [[e(w["station"]), e(w["model"]), n(w["human_s"], 1), n(w["robot_s"], 1), n(w["interaction_s"], 1),
                        n(w["cycle_s"], 1), pct(w["human_idle_pct"]), pct(w["robot_idle_pct"]), n(w["takt_slack_s"], 1)] for w in wm])
        mc = al.get("machine_cluster", {})
        if mc:
            body += f"<h3>Can one cobot serve two stations? (machine cluster)</h3><div class='formula'>{e(mc['formula'])}</div>"
            body += table(["station", "Ts s", "Tm s", "Tr s", "n'", "stations servable"],
                          [[e(r["station"]), n(r["Ts_s"], 1), n(r["Tm_s"], 1), n(r["Tr_s"], 1), n(r["n_prime"], 2),
                            r["max_stations_served"]] for r in mc["per_station"]])
            if mc.get("adjacent_pairs"):
                body += table(["pair", "cobot busy s", "takt s", "feasible"], [[e(" + ".join(p["stations"])),
                              n(p["cobot_busy_s"], 1), n(p["takt_s"], 0), "yes" if p["feasible"] else "no"] for p in mc["adjacent_pairs"]])
            body += f"<p class='cite'>{e(mc.get('caveat', ''))}</p>"
        rc = al.get("rough_cut_capacity", {})
        if rc:
            body += (f"<h3>Rough-cut capacity</h3><p class='cite'>{e(rc['method'])}. Available robot hours/yr "
                     f"{n(rc['available_h_per_robot'], 0)} (availability {pct(rc['availability'])}); cushion {pct(rc['cushion'])}.</p>")
            body += table(["scenario", "station", "model", "required h/yr", "fraction", "robots", "utilisation", "after cushion"],
                          [[e(r["scenario"]), e(r["station"]), e(r["model"]), n(r["required_h"], 0), n(r["fraction"], 3),
                            r["robots"], pct(r["utilization"], 1), r["robots_after_cushion"]] for r in rc["robots"]
                           if r["scenario"] == rec["name"]])
            body += table(["scenario", "operator h/yr", "dedicated operators", "utilisation", "cross-trained (RCA bound)", "utilisation"],
                          [[e(r["scenario"]), n(r["required_h"], 0), r["dedicated_operators"], pct(r["dedicated_utilization"], 1),
                            r["cross_trained_operators"], pct(r["cross_trained_utilization"], 1)] for r in rc["operators"]])
        return section("sec-s5", "Stage 5 - decision", "Optimised human-robot allocation and line balance",
                       "An exact integer program decides, element by element, whether the operator keeps the work or "
                       "a screened robot takes it, and which robots to buy, at minimum equivalent annual cost while "
                       "every station stays within takt.", body, P("allocation"))
    secs.append(safe(s5, "sec-s5", "Stage 5", "Allocation & balancing"))

    # ------------------------------------------------------------------ s6b
    def s6b():
        rows = []
        for s in des["scenarios"]:
            ll = s["littles_law"]
            rows.append([e(s["name"]) + (" *" if s.get("recommended") else ""), e(s.get("mode", "")),
                         n(s["throughput_per_hr"], 3), "&plusmn; " + n(s["ci95_half_width"], 3), n(s["takt_violations_pct"], 1),
                         n(s.get("line_stop_pct"), 1), e(", ".join(f"{k} {pct(v)}" for k, v in s["utilization"].items())),
                         n(ll["avg_wip"], 3), n(ll["throughput_per_s"] * ll["flow_time_s"], 3)])
        v = des["validation"]
        body = img(F.des_chart(des), "des")
        body += table(["scenario", "mode", "doors/h", "95% CI", "% overruns", "% line stops", "utilisation", "WIP L",
                       "TH x W"], rows)
        body += (f"<div class='callout'><b>Validation.</b> Baseline DES {n(v['des_baseline_throughput'], 3)} doors/h vs analytic "
                 f"{e(v['formula'])} = {n(v['analytic_baseline_throughput'], 3)} doors/h ({pct(v.get('rel_error'), 1)} apart). "
                 f"{e(v.get('note', ''))}. Little's law L = TH x W is checked per scenario from independently measured WIP, "
                 f"throughput and flow time.</div>")
        dist = des.get("distributions", {})
        body += (f"<p class='cite'>{e(des['tool'])}; {des['replications']} replications x {n(des['run_length_h'], 0)} h, "
                 f"warm-up {n(des['warmup_s'], 0)} s. Human elements: {e(dist.get('human_element'))}; robot: {e(dist.get('robot'))}; "
                 "UNOBS: CV = " + e(", ".join(f"{k} {n(x['cv'], 3)}" for k, x in dist.get("unobs", {}).items())) + "</p>")
        return section("sec-s6b", "Stage 6b - line simulation", "Does the line still make takt with variability?",
                       "A discrete-event model of the paced conveyor (line stops when any station overruns) replays "
                       "every allocation with stochastic task times.", body, P("des_results"))
    secs.append(safe(s6b, "sec-s6b", "Stage 6b", "Line simulation"))

    # ------------------------------------------------------------------ s7
    def s7():
        body = "<h3>Assumptions and sources</h3>" + table(["assumption", "value", "unit", "source"],
                                                          [[e(a["name"]), e(a["value"]), e(a["unit"]), f"<span class='cite'>{e(a['source'])}</span>"]
                                                           for a in roi["assumptions"]])
        rows = []
        for s in roi["scenarios"]:
            rows.append(("rec" if s["recommended"] else "", [e(s["name"]) + (" *" if s["recommended"] else ""),
                         e(", ".join(s["robots"]) or "-"), usd(s["capex_usd"]), n(s["crf"]["value"], 5),
                         usd(s["annualized_capex_usd"]), n(s["labour_hours_freed_per_yr"], 0), usd(s["gross_labour_savings_usd"]),
                         usd(s["maintenance_usd"] + s.get("relief_operator_cost_usd", 0)), usd(s["annual_savings_usd"]), n(s["payback_years"], 1),
                         usd(s["npv_usd"]), pct(s["irr"], 1) if s.get("irr") is not None else "-",
                         usd(s.get("vs_manual_relief", {}).get("npv_usd")) if s.get("vs_manual_relief") else "-"]))
        head = ["scenario", "robots", "capex", "A/P", "annualised capex", "labour h/yr freed", "gross saving", "maintenance + relief ops",
                "net saving/yr", "payback yr", "NPV vs as-is", "IRR", "NPV vs manual relief"]
        body += "<h3>Business case per scenario</h3>" + '<div class="tw"><table><thead><tr>' + "".join(
            f"<th>{h}</th>" for h in head) + "</tr></thead><tbody>" + "".join(
            f'<tr class="{c}">' + "".join(f"<td>{x}</td>" for x in r) + "</tr>" for c, r in rows) + "</tbody></table></div>"
        body += "<div class='formula'>" + e(roi["method"].strip()) + "</div>"
        body += "<div class='grid2'><div>" + img(F.cashflow_chart(roi), "cash flow") + "</div><div>" + \
                img(F.tornado_chart(roi), "tornado") + "</div></div>"
        be = roi.get("breakeven") or {}
        if be:
            hv = be.get("headcount_view", {})
            body += (f"<div class='callout warn'><b>Break-even.</b> NPV = 0 needs {usd(be['required_annual_savings_usd'])}/yr net, "
                     f"i.e. {n(be['required_human_s_saved_per_cycle'], 1)} s of operator work removed per door "
                     f"(the recommended allocation removes {n(be['actual_human_s_saved_per_cycle'], 1)} s), or a loaded-wage "
                     f"equivalent of {usd((be.get('breakeven_hourly_wage_usd') or 0), 2)}/h base wage. Headcount view: after "
                     f"re-balancing, {hv.get('operators_after_rebalance', '-')} operators for {hv.get('stations', '-')} stations "
                     f"({hv.get('operators_freed_per_shift', '-')} freed per shift, {usd(hv.get('annual_value_usd'))}/yr); counting only "
                     f"whole operators the net saving is {usd(hv.get('annual_net_savings_usd'))}/yr, NPV {usd(hv.get('npv_usd'))}, "
                     f"payback {n(hv.get('payback_years'), 1, ' yr') if hv.get('payback_years') else 'not reached'}.</div>")
        body += "<p class='cite'>Not monetised: " + e("; ".join(roi.get("not_monetised", []))) + "</p>"
        body += table(["parameter", "base", "low", "high", "NPV low", "NPV high", "range"],
                      [[e(r["param"]), n(r["base"], 3), n(r["low"], 3), n(r["high"], 3), usd(r["npv_low"]), usd(r["npv_high"]),
                        f"<span class='cite'>{e(r.get('note', ''))}</span>"] for r in roi["sensitivity"]])
        return section("sec-s7", "Stage 7 - economics", "Return on investment",
                       "Capital recovery factor, NPV, IRR and payback on the labour time freed, with a one-at-a-time "
                       "sensitivity (tornado) on the uncertain inputs.", body, P("roi"))
    secs.append(safe(s7, "sec-s7", "Stage 7", "ROI"))

    # ------------------------------------------------------------------ s5x design space
    def s5x():
        ds = D["design_space"]
        c = ds["counts"]
        by = {d["id"]: d for d in ds["designs"]}
        lab = {k["key"]: k["label"] for k in ds["criteria"]}
        arms = lambda d: " · ".join(f"{st} {d['arms'][st] or '-'}" for st in ds["stations"])  # noqa: E731
        fm = lambda k, v: (usd(v) if k.endswith("usd") else str(v) if k == "n_robots"  # noqa: E731
                           else pct(v / 100, 1) if k == "takt_violation_pct" else n(v, 1, " s"))
        fac = ds["factors"]
        body = "<h3>The three options</h3>" + headline_html()
        body += (f"<h3>How we got there</h3><p>Instead of one answer, every combination the line admits was built and scored: an arm or none at each "
                f"station ({'; '.join(f'{st}: {len(v)} options' for st, v in fac['arm_per_station'].items())}), layout "
                f"levers on/off, three task splits (what the arm does), and planning to takt or to "
                f"{e(fac['pacing'][1])}. The s5 MILP decides the task split of every combination; one that cannot meet "
                f"takt is retried with a relief operator.</p>"
                "<div class='stat'>" + "".join(f"<div><div class='v'>{v:,}</div><div class='l'>{e(l)}</div></div>" for v, l in [
                    (c["arm_combinations"], "arm combinations"), (c["factor_runs"], "factor runs"),
                    (c["milp_solves"], "MILP solves"), (c["unique_designs"], "distinct designs"),
                    (c["screen_pareto"], "non-dominated (screen)"), (c["simulated"], "simulated (DES, CRN)"),
                    (c["final_pareto"], "final Pareto set")]) + "</div>")
        body += img(F.design_space_chart(ds), "design space",
                    "Left: every distinct design; grey ones are beaten on every criterion by at least one other design. "
                    "Right: share of all possible criterion weightings (uniform on the simplex) for which a design ranks "
                    "1st/2nd/3rd. * = s5 recommendation.")
        body += "<h3>Who would pick what</h3>" + table(
            ["stakeholder profile", "weights"] + ["best", "runner-up", "third"],
            [[e(p["name"]), e(", ".join(f"{lab[k]} {pct(w)}" for k, w in p["weights"].items()))] +
             [f"<b>{e(x)}</b> <span class='cite'>{e(arms(by[x]))}</span>" for x in p["top3"]] for p in ds["personas"]])
        rows = []
        for f in ds["featured_comparisons"]:
            a, b = by[f["a"]], by[f["b"]]
            win = " ".join(f"{lab[r['criterion']]}: <b>{'A' if r['winner'] == 'a' else 'B' if r['winner'] == 'b' else '='}</b>"
                           for r in f["rows"])
            rows.append([f"<b>{e(f['a'])}</b> vs <b>{e(f['b'])}</b><div class='cite'>B = {e(arms(b))}"
                         + (f" ({e(', '.join(b['named_scenarios']))})" if b["named_scenarios"] else "") + "</div>",
                         win, pct(f["p_a_preferred"]) if f["p_a_preferred"] is not None else "-",
                         f"{f['throughput_diff_per_hr']:+.2f} ± {f['throughput_diff_ci95']:.2f}",
                         e(f["verdict"]) + (f"<div class='cite'>{e(f['tradeoff'])}</div>" if f["tradeoff"] else "")])
        body += ("<h3>Head-to-head against the recommendation</h3>" +
                 table(["A vs B", "winner per criterion", "P(A preferred) over all weightings",
                        "throughput A-B doors/h (paired, 95 % CI)", "verdict"], rows))
        body += "<h3>Explore it yourself</h3><p class='cite'>Weights, budget and filters re-rank the simulated shortlist " \
                "live; pick any two designs to compare them side by side. Values are scaled 0-1 between the worst and best " \
                "design of the current shortlist.</p>" + DS.HTML
        fr = sorted([d for d in ds["designs"] if d.get("pareto_final")], key=lambda d: -d["smaa"]["rank1"])
        body += "<h3>Final Pareto set</h3>" + table(
            ["design", "arms", "extras"] + [lab[k["key"]] for k in ds["criteria"]] + ["ranked 1st", "typical weights of a client who picks it", "s5 name"],
            [[("<b>" + e(d["id"]) + "</b>" + (" *" if d["id"] == ds.get("recommended_design") else "")), e(arms(d)),
              e(", ".join([f"{k} x{v}" for k, v in sorted(__import__("collections").Counter(
                  x.split("@")[0].replace("_", " ") for x in d["layout_levers"] if not x.endswith("@LINE")).items())]
                          + ([f"{d['relief_operators_per_shift']} relief op"] if d["relief_operators_per_shift"] else [])) or "-")]
             + [fm(k["key"], d["metrics"][k["key"]]) for k in ds["criteria"]]
             + [pct(d["smaa"]["rank1"], 1),
                e(", ".join(f"{lab[k]} {pct(w)}" for k, w in (d["smaa"]["central_weights"] or {}).items())) or "-",
                e(", ".join(d["named_scenarios"]))] for d in fr])
        eff = [[e(x["station"]), e(x["arm"] or "none"), x["n_designs"], usd(x.get("best_npv_usd")) + f" ({e(x.get('best_npv_design', ''))})",
                usd(x.get("min_capex_usd")), n(x.get("min_awkward_s"), 1), x.get("on_final_front", 0)] for x in ds["main_effects"]]
        body += ("<h3>If you insist on a given arm</h3><p class='cite'>Best attainable outcome with one factor fixed "
                 "(main effects over the full factorial). A model with 0 designs on the final front is beaten on every "
                 "criterion whatever else is chosen.</p>" +
                 table(["station", "arm", "designs", "best NPV", "lowest installed cost", "least awkward posture s",
                        "on final front"], eff))
        body += "<div class='formula'>" + e(ds["method"].strip()) + "</div>"
        body += "<p class='cite'>" + e("; ".join(ds["citations"].values())) + "</p>"
        return section("sec-s5x", "Design space", "Every option we evaluated, and how to choose between them",
                       "Not one answer: the full set of robot combinations, the ones that cannot be beaten, and which "
                       "of them fits which priorities.", body, P("design_space"))
    if D.get("design_space"):
        secs.append(safe(s5x, "sec-s5x", "Design space", "Design space"))

    # ------------------------------------------------------------------ s9
    def s9():
        tp = D["task_plan"]
        body = f"<p class='cite'>Primitive vocabulary: {e(', '.join(tp['primitives']))}</p>"
        for p in tp["plans"]:
            rows = [[k + 1, e(s["primitive"]), e(", ".join(s.get("covers", []))),
                     e(", ".join(f"{a}={b}" for a, b in s.items() if a not in ("primitive", "covers") and not isinstance(b, (dict, list)))[:120])]
                    for k, s in enumerate(p["steps"][:60])]
            body += f"<h3>{e(p['station'])} - {e(p['model'])}</h3>" + table(["#", "primitive", "covers", "parameters"], rows)
            body += f"<pre>{e(p['bt_xml'][:6000])}</pre>"
        return section("sec-s9", "Stage 9 - robot task plan", "From allocation to a draft robot program",
                       "The robot-assigned elements become a behaviour tree over a fixed primitive vocabulary, the "
                       "starting point for the integrator's program.", body, P("task_plan"))
    secs.append(safe(s9, "sec-s9", "Stage 9", "Robot program"))

    # ------------------------------------------------------------------ recommendation + limitations
    def recsec():
        items = []
        for r in rec["robots"]:
            st = rec["station_times"][r["station"]]
            items.append(f"<li><b>Buy 1 x {e(r['model'])} for {e(r['station'])}</b> to take "
                         f"{e(', '.join(x.split(':')[1] for x in r['elements']))}: operator load {n(base['station_times'][r['station']]['human_s'], 1)} "
                         f"-> {n(st['human_s'], 1)} s, station cycle {n(st['cycle_s'], 1)} s within takt {n(takt, 0)} s; "
                         f"installed cost {usd(r['capex_usd'])}.</li>")
        for l in rec.get("layout_changes", []):
            items.append(f"<li><b>{e(l['lever'].replace('_', ' '))} at {e(l['station'])}</b>"
                         + (f" for {e(', '.join(x.split(':')[1] for x in l['elements']))} (-{pct(l['reduction'])} time)"
                            if l["elements"] else "") + f": {usd(l['capex_usd'])}. <span class='cite'>{e(l['desc'])}</span></li>")
        why = []
        if bent_red is not None:
            why.append(f"awkward-posture exposure falls {pct(bent_red)} ({n(ob['awkward_posture_s_per_cycle'], 1)} -> "
                       f"{n(orc['awkward_posture_s_per_cycle'], 1)} s/door)")
        if rdes and bdes:
            why.append(f"takt overruns fall from {n(bdes['takt_violations_pct'], 1)}% to {n(rdes['takt_violations_pct'], 1)}% of "
                       f"station-cycles (DES)")
        if rroi:
            why.append(f"NPV {usd(rroi['npv_usd'])}, payback {n(rroi['payback_years'], 1) + ' yr' if rroi['payback_years'] else 'not reached'}")
        rfq = sorted(p.name for p in (out_dir(cfg, "s8") / "rfq").glob("*.md"))
        body = (f"<ul class='tight'>{''.join(items) or '<li>No robot purchase is justified by the current data.</li>'}</ul>"
                f"<p class='big'>Why: {e('; '.join(why))}.</p>"
                f"<p>RFQ packages: {e(', '.join('out/s8/rfq/' + x for x in rfq) or 'none')}.</p>")
        return section("sec-rec", "Recommendation", "What to buy, where, and why", e(al.get("recommendation_rule", "")),
                       body, P("allocation", "des_results", "roi"))

    def limits():
        L = []
        if fixture_used:
            L.append(f"Inputs still on development fixtures: {', '.join(fixture_used)}.")
        u = [x for x in al["elements"] if x["code"] == "UNOBS"]
        if u:
            share = sum(x["time_s"] for x in u) / (takt * len(u))
            L.append(f"The CarDA clips are cropped: {pct(share)} of the station cycle is unobserved (UNOBS) and is "
                     "treated as manual, non-automatable work; automation potential inside it is unknown.")
        so_ = (al.get("as_is") or {}).get("stations", {})
        asm = [k for k, v in so_.items() if v.get("overload_basis") != "measured" and v.get("overload_pct", 0) > 0]
        if asm:
            L.append(f"As-is overload at {', '.join(asm)} rests on the unobserved block (basis 'assumption'); only "
                     f"measured stations are headlined.")
        L += [f"Element times use {al.get('time_basis')} (observed x rating x (1 + PFD allowance)); small samples "
              "per station (see n and n required in the time study).",
              "Layout levers (AMR kitting / point-of-use supply, fixtures) use an assumed time reduction and cost; "
              "they are lean improvements, not robot-arm work.",
              "The relief-operator alternative (manual_relief) is costed as whole headcount per shift.",
              "Labour savings are valued as freed hours x loaded wage; cash savings require redeploying that time "
              "(headcount changes only via re-balancing).",
              "Robot prices are catalog ranges; integration, EOAT, safety and maintenance are labelled assumptions "
              "with sensitivity ranges (tornado).",
              "Ergonomic and quality benefits are reported but not monetised.",
              "Precedence across stations is assumed sequential (door moves WS10 -> WS20 -> WS30)."]
        L += [f"s5 note: {x}" for x in al.get("notes", [])[:6]]
        body = "<div class='callout red'><ul class='tight'>" + "".join(f"<li>{e(x)}</li>" for x in L) + "</ul></div>"
        return section("sec-limits", "Read before quoting", "Limitations and assumptions", "", body)

    # RFQs first (the recommendation section lists them)
    rfq_dir = out_dir(cfg, "s8") / "rfq"
    rfq_dir.mkdir(parents=True, exist_ok=True)
    for p in rfq_dir.glob("*.md"):
        p.unlink()
    for st, md in build_rfqs(D).items():
        (rfq_dir / f"{st}.md").write_text(md)
    secs.append(safe(recsec, "sec-rec", "Recommendation", "Recommendation"))
    secs.append(safe(limits, "sec-limits", "Limitations", "Limitations"))

    data = {k: compact(v) for k, v in D.items() if v is not None and k != "design_space"}
    if D.get("design_space"):   # full designs only for the simulated shortlist; the rest as summary rows (file has all)
        ds = D["design_space"]
        slim = ("id", "arms", "metrics", "pareto_screen", "dominated_by", "named_scenarios", "relief_operators_per_shift")
        data["design_space"] = {**{k: v for k, v in ds.items() if k not in ("designs", "runs")},
                                "designs": [d if d.get("simulated") else {k: d[k] for k in slim} for d in ds["designs"]],
                                "runs_omitted": f"{len(ds.get('runs', []))} factor runs: see out/s5x/design_space.json"}
    data["_provenance"] = prov
    blob = json.dumps(data, ensure_ascii=False, default=float).replace("</", "<\\/")
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    ds_js = f"<script>{DS.JS}</script>" if D.get("design_space") else ""
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Robot automation assessment - CarDA door line</title>
<style>{CSS}{DS.CSS}</style></head><body>{hero}{toc}<main><div class="wrap">{''.join(secs)}</div></main>
<footer>Generated {e(stamp)} by rcd.s8_report from out/*.json - inputs: {" ".join(f"{e(k)} {chip(v)}" for k, v in prov.items())}</footer>
<script type="application/json" id="rcd-data">{blob}</script>{ds_js}</body></html>"""
    p = out_dir(cfg, "s8") / "report.html"
    p.write_text(doc)
    print(f"  s8: report {len(doc) // 1024} KB, {len(list(rfq_dir.glob('*.md')))} RFQ(s); fixtures used: {fixture_used}")
