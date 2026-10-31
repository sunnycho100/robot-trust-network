"""Render docs/SIM_OPTIONS_ABC.md into docs/SIM_OPTIONS_ABC.pdf (simulation request for the three headline options).

    python scripts/export_option_packages.py     # first: out_options/ packages + manifest
    python scripts/build_options_pdf.py

Same print style as the v2 hand-off (scripts/build_handoff_pdf.py, imported read-only), with a cover comparing the
three options and a figure appendix (station cycles per option, closest robot-worker approach in option C, previews).
Everything is embedded; headless Chrome prints the PDF.
"""
from __future__ import annotations

import base64
import importlib.util
import io
import json
import subprocess
import sys
from pathlib import Path

import cv2
import markdown
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OPT = ROOT / "out_options"
_spec = importlib.util.spec_from_file_location("handoff_v2", ROOT / "scripts" / "build_handoff_pdf.py")
V2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V2)
CSS = V2.CSS.replace("Simulation team handoff v2", "Simulation request v3 · three-option comparison") + """
.cards { grid-template-columns: repeat(3, 1fr); }
th strong { color: #fff; }
.card.rec { border: 1.5px solid #f6a26b; }
.card small { display: block; font-size: 8pt; color: #f6a26b; margin-bottom: 1.5mm; letter-spacing: .06em; }
.card .row { display: flex; justify-content: space-between; font-size: 8.4pt; color: #dfe6f0; border-top: 1px solid rgba(255,255,255,.12);
  padding: 1mm 0; }
"""
FONT = ["Helvetica Neue", "Arial", "sans-serif"]


def png(fig) -> str:
    b = io.BytesIO()
    fig.savefig(b, format="png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def cycle_chart(m) -> str:
    plt.rcParams.update({"font.family": FONT, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    sts = sorted(m["options"][0]["station_times"])
    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    w = 0.26
    cols = {"A": "#9aa5b1", "B": "#1c5d99", "C": "#d9480f"}
    for i, o in enumerate(m["options"]):
        x = np.arange(len(sts)) + (i - 1) * w
        hum = [o["station_times"][s]["human_s"] for s in sts]
        cyc = [o["station_times"][s]["cycle_s"] for s in sts]
        ax.bar(x, cyc, w * 0.92, color=cols[o["letter"]], label=f"Option {o['letter']} · " + {"A": "layout only", "B": "1 robot", "C": "3 robots"}[o["letter"]])
        for xi, c in zip(x, cyc):
            ax.text(xi, c + 3, f"{c:.0f}", ha="center", fontsize=7.5, color="#18212f")
    ax.axhline(m["takt_s"], color="#18212f", lw=1, label=f"Takt {m['takt_s']:.0f} s")
    ax.axhline(0.95 * m["takt_s"], color="#d9480f", lw=1, ls="--", label=f"Planning cap {0.95 * m['takt_s']:.0f} s")
    ax.set_xticks(np.arange(len(sts)), sts)
    ax.set_ylim(0, 270)
    ax.set_xlim(-0.55, len(sts) - 0.45)
    ax.set_ylabel("Station cycle (s, standard time)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=8, frameon=False, ncol=5)
    ax.grid(axis="y", color="#e3e7ee")
    return png(fig)


def closest(st_dir: Path):
    R = np.loadtxt(st_dir / "trace_tobe.csv", delimiter=",")
    act = R[:, 5] > 0.5
    k = int(np.argmin(np.where(act, R[:, 1], 9e9)))
    cap = cv2.VideoCapture(str(st_dir / "replay_tobe.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(R[k, 0] / 0.2)))
    ok, fr = cap.read()
    return fr, R[k, 0], R[k, 1]


def fix_md(md: str) -> str:
    """python-markdown needs a blank line before a table / list that directly follows a paragraph line."""
    out = []
    for ln in md.splitlines():
        prev = out[-1].lstrip() if out else ""
        if ln.lstrip().startswith("|") and prev and not prev.startswith("|"):
            out.append("")
        if ln.startswith(("- ", "1. ")) and prev and not prev.startswith(("- ", "1. ", "2. ", "3. ", "4. ", "|", ">")) \
                and not out[-1].startswith("  "):
            out.append("")
        if ln.startswith("  - "):   # python-markdown nests lists at 4 spaces
            ln = "  " + ln
        if ln.startswith("> - ") and prev.startswith(">") and not prev.startswith("> - ") and prev != ">":
            out.append(">")
        out.append(ln)
    return "\n".join(out)


def main() -> int:
    m = json.loads((OPT / "manifest.json").read_text())
    O = {o["letter"]: o for o in m["options"]}
    body = markdown.markdown(fix_md((ROOT / "docs" / "SIM_OPTIONS_ABC.md").read_text()),
                             extensions=["tables", "fenced_code", "sane_lists"])
    c10 = OPT / "C" / "sim_package" / "WS10"
    fr, t10, d10 = closest(c10)
    kC = {st: v["tobe"]["share_active_time_ssm_violation"] for st, v in O["C"]["replay"].items()}
    kCa = O["C"]["replay"]["WS30"]["tobe_away"]["share_active_time_ssm_violation"]
    kB = O["B"]["replay"]["WS20"]["tobe"]["share_active_time_ssm_violation"]
    usd = lambda v: f"${v / 1000:,.0f}k"  # noqa: E731

    def card(o, rec=False):
        mm = o["metrics"]
        rows = [("Capex", usd(mm["capex_usd"])), ("NPV 5 yr", usd(mm["npv_usd"])),
                ("Takt overrun (DES)", f"{mm['takt_violation_pct']:.0f}%"), ("Awkward posture", f"{mm['awkward_s']:.0f} s/door"),
                ("Best-option share", f"{o['share_of_weightings']:.0%}")]
        name = {"A": "Layout only", "B": "1 robot · WS20", "C": "3 robots · all stations"}[o["letter"]]
        return (f"<div class='card{' rec' if rec else ''}'><small>Option {o['letter']}{' · earlier recommendation' if rec else ''}</small>"
                f"<b>{name}</b>" + "".join(f"<div class='row'><span>{a}</span><span>{b}</span></div>" for a, b in rows) + "</div>")
    cover = f"""
<section class="cover">
  <div class="kicker">ROBOT SIMULATION REQUEST · v3</div>
  <h1>Three options for the customer,<br>please simulate each one</h1>
  <div class="sub">Three options picked after scoring 1,219 robot combinations. Which one a customer picks depends on their priorities,
  so all three must be validated on the same basis (same random numbers, same KPIs) for a fair comparison.</div>
  <div class="hero"><img src="{V2.b64img(fr)}"></div>
  <div class="cards">{card(O['A'])}{card(O['B'])}{card(O['C'], True)}</div>
  <div class="meta"><span>Author: industrial engineering · Data: CarDA (CC BY 4.0) · 990 runs</span>
  <span>Packages: out_options/ · Doc: docs/SIM_OPTIONS_ABC.md</span></div>
</section>"""
    figs = f"""
<section class="figs">
<h2>Appendix A. The three options in figures</h2>
<figure><img src="{cycle_chart(m)}"><figcaption><b>Station cycle per option (IE prediction, standard time).</b>
In option A all three stations exceed the 228 s planning cap, and WS30 sits right at the 240 s takt (239.9 s). Option B clears only WS20;
WS10 and WS30 are the same as in option A. Only option C keeps all three stations under the cap. The simulation checks whether these bars
still hold once real motion, safety policies, and randomization are added.</figcaption></figure>
<figure class="alert"><img src="{V2.b64img(fr)}"><figcaption><b>Option C WS10, closest approach (t = {t10:.1f} s, minimum distance {d10:.2f} m).</b>
The robot drives screws into the door where the worker is inserting the glass. {kC['WS10']:.0%} of robot motion time is inside the SSM
protective separation distance, which makes this the main item to validate for option C.</figcaption></figure>
<div class="grid2">
<figure><img src="{V2.b64img(OPT / 'B' / 'sim_package' / 'WS20' / 'preview.png')}"><figcaption><b>Options B and C, WS20 (same cell).</b>
3 mounting points and 9 screws, tool changer. The robot at 150.2 s is the bottleneck, and SSM violation is {kB:.0%} (overlap only in the first ~27 s).</figcaption></figure>
<figure><img src="{V2.b64img(OPT / 'C' / 'sim_package' / 'WS30' / 'preview.png')}"><figcaption><b>Option C WS30.</b>
After the worker inserts the speaker, the robot drives 4 screws. SSM violation is {kC['WS30']:.0%} if the worker stays and {kCa:.0%} if the worker leaves.</figcaption></figure>
</div>
</section>"""
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>CarDA simulation request v3 (three options)</title>
<style>{CSS}</style></head><body>{cover}<main>{body}</main>{figs}</body></html>"""
    out_html = OPT / "SIM_OPTIONS_print.html"
    out_html.write_text(html)
    pdf = ROOT / "docs" / "SIM_OPTIONS_ABC.pdf"
    subprocess.run([V2.CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    "--run-all-compositor-stages-before-draw", f"--print-to-pdf={pdf}", out_html.as_uri()],
                   check=True, capture_output=True)
    print(pdf)
    return 0


if __name__ == "__main__":
    sys.exit(main())
