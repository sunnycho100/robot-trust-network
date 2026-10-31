"""Render docs/SIM_HANDOFF.md into a designed PDF (docs/SIM_HANDOFF.pdf) via headless Chrome.

    python scripts/build_handoff_pdf.py

Adds a cover page with the key facts and a figure page (CarDA frames + the three simulation scenes, and the WS10
moment of closest robot-worker approach) taken from out/sim_package. Everything is embedded; no network access.
"""
from __future__ import annotations

import base64
import json
import subprocess
import sys
from pathlib import Path

import cv2
import markdown
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "out" / "sim_package"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def b64img(path_or_arr, ext="jpg", max_w=1400) -> str:
    img = cv2.imread(str(path_or_arr)) if not isinstance(path_or_arr, np.ndarray) else path_or_arr
    if img.shape[1] > max_w:
        img = cv2.resize(img, (max_w, int(img.shape[0] * max_w / img.shape[1])))
    ok, buf = cv2.imencode(f".{ext}", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return f"data:image/{'jpeg' if ext == 'jpg' else ext};base64," + base64.b64encode(buf).decode()


def closest_approach_frame(st: str):
    """Video frame of the smallest robot-worker distance while the robot is active (replay records a frame every 0.2 s)."""
    R = np.loadtxt(PKG / st / "trace_tobe.csv", delimiter=",")
    act = R[:, 5] > 0.5
    k = int(np.argmin(np.where(act, R[:, 1], 9e9)))
    t, dmin = R[k, 0], R[k, 1]
    cap = cv2.VideoCapture(str(PKG / st / "replay_tobe.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t / 0.2)))
    ok, fr = cap.read()
    return fr, t, dmin


def kpi(st, tag="tobe"):
    return json.loads((PKG / st / f"kpi_{tag}.json").read_text())


CSS = """
@page { size: A4; margin: 18mm 16mm 20mm 16mm;
  @bottom-right { content: counter(page) " / " counter(pages); font: 8.5pt "Helvetica Neue", Arial, sans-serif; color: #8a94a6; }
  @bottom-left { content: "CarDA cobot deployment · Simulation team handoff v2"; font: 8.5pt "Helvetica Neue", Arial, sans-serif; color: #8a94a6; } }
@page :first { margin: 0; @bottom-right { content: none; } @bottom-left { content: none; } }
:root { --ink:#18212f; --muted:#5b6678; --line:#e3e7ee; --accent:#d9480f; --accent2:#1c5d99; --soft:#f6f8fb; }
* { box-sizing: border-box; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { font-family: "Helvetica Neue", Arial, sans-serif; color: var(--ink); font-size: 9.6pt; line-height: 1.62;
  margin: 0; word-break: keep-all; }
.cover { height: 297mm; padding: 26mm 20mm 18mm; background: linear-gradient(160deg, #0f1b2d 0%, #16304f 55%, #1c5d99 100%);
  color: #fff; position: relative; page-break-after: always; }
.cover .kicker { letter-spacing: .18em; font-size: 9pt; color: #f6a26b; font-weight: 700; }
.cover h1 { font-size: 27pt; line-height: 1.25; margin: 6mm 0 4mm; font-weight: 800; }
.cover .sub { font-size: 11.5pt; color: #cfd8e6; max-width: 150mm; }
.cover .hero { margin-top: 10mm; border-radius: 4mm; overflow: hidden; border: 1px solid rgba(255,255,255,.18); }
.cover .hero img { width: 100%; display: block; }
.cards { display: grid; grid-template-columns: repeat(4, 1fr); gap: 4mm; margin-top: 8mm; }
.card { background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.16); border-radius: 3mm; padding: 4mm; }
.card b { display: block; font-size: 17pt; color: #fff; line-height: 1.1; }
.card span { font-size: 8.3pt; color: #cfd8e6; }
.cover .meta { position: absolute; bottom: 16mm; left: 20mm; right: 20mm; font-size: 8.5pt; color: #a9b6c9;
  border-top: 1px solid rgba(255,255,255,.18); padding-top: 4mm; display: flex; justify-content: space-between; }
main h1 { display: none; }
h2 { font-size: 14.5pt; margin: 9mm 0 3mm; padding-bottom: 2mm; border-bottom: 2px solid var(--ink); page-break-after: avoid; }
h3 { font-size: 11.2pt; margin: 6mm 0 2mm; color: var(--accent2); page-break-after: avoid; }
p { margin: 1.6mm 0 2.4mm; }
ul, ol { margin: 1mm 0 3mm; padding-left: 6mm; }
li { margin: .8mm 0; }
strong { color: #0d1726; }
table { width: 100%; border-collapse: collapse; margin: 2.5mm 0 4mm; font-size: 8.5pt; page-break-inside: avoid; }
th { background: var(--ink); color: #fff; text-align: left; font-weight: 700; padding: 1.8mm 2.2mm; }
td { padding: 1.6mm 2.2mm; border-bottom: 1px solid var(--line); vertical-align: top; }
tr:nth-child(even) td { background: var(--soft); }
code { font-family: 'SF Mono', Menlo, monospace; font-size: 8.1pt; background: #eef1f6; padding: .2mm 1.2mm; border-radius: 1mm; }
pre { background: #0f1b2d; color: #e6edf6; padding: 3.5mm 4mm; border-radius: 2.5mm; font-size: 8.1pt; line-height: 1.5;
  page-break-inside: avoid; overflow: hidden; }
pre code { background: none; color: inherit; padding: 0; }
blockquote { margin: 3mm 0; padding: 3mm 4mm; background: #fff4ec; border-left: 3px solid var(--accent); border-radius: 0 2mm 2mm 0;
  color: #3b2b20; }
blockquote p { margin: 1mm 0; }
hr { border: none; height: 0; margin: 2mm 0; }
.figs { page-break-before: always; }
.figs h2 { margin-top: 0; }
.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 4mm; }
figure { margin: 0 0 4mm; page-break-inside: avoid; }
figure img { width: 100%; border-radius: 2mm; border: 1px solid var(--line); display: block; max-height: 95mm; object-fit: cover; }
figcaption { font-size: 8.2pt; color: var(--muted); margin-top: 1.4mm; line-height: 1.45; }
.alert { border: 1.5px solid var(--accent); }
"""


def main() -> int:
    md = (ROOT / "docs" / "SIM_HANDOFF.md").read_text()
    # python-markdown needs a blank line before a table that follows a paragraph line
    lines, fixed = md.splitlines(), []
    for ln in lines:
        prev = fixed[-1].lstrip() if fixed else ""
        if ln.lstrip().startswith("|") and prev and not prev.startswith("|"):
            fixed.append("")
        # same for a list that directly follows a paragraph line (e.g. "**KPI**" then "- ...")
        is_item = ln.startswith(("- ", "1. ")) and not ln.startswith("  ")
        if is_item and prev and not prev.startswith(("- ", "1. ", "2. ", "3. ", "4. ", "5. ", "6. ", "|", ">")) \
                and not fixed[-1].startswith("  "):
            fixed.append("")
        fixed.append(ln)
    md = "\n".join(fixed)
    body = markdown.markdown(md, extensions=["tables", "fenced_code", "sane_lists"])
    # figures
    frame10, t10, d10 = closest_approach_frame("WS10")
    k = {s: kpi(s) for s in ("WS10", "WS20", "WS30")}
    kaway = kpi("WS30", "tobe_away")
    sheet = ROOT.parent / "carda_samples" / "ws30_cycle002_contact_sheet.jpg"
    figs = f"""
<section class="figs">
<h2>Appendix A. The scenario in figures</h2>
<figure><img src="{b64img(sheet)}"><figcaption><b>Input data (CarDA WS30, cycle 002).</b> Top row is the IN camera, bottom row the OUT camera,
both at 3, 21, 38, and 56 s. Shot with fixed ZED2 cameras on 1.85 m posts. The checkerboard and the ArUco markers on the cart are the calibration references.</figcaption></figure>
<figure class="alert"><img src="{b64img(frame10)}"><figcaption><b>WS10 closest approach (t = {t10:.1f} s, min distance {d10:.2f} m).</b>
While the robot (M0617, on the track) drives screws, the worker is still inserting glass at the same door after automation.
{k['WS10']['share_active_time_ssm_violation']:.0%} of the robot's motion time is inside the SSM protective separation distance, so the task sequence needs a redesign (P2/P3).</figcaption></figure>
<div class="grid2">
<figure><img src="{b64img(PKG / 'WS20' / 'preview.png')}"><figcaption><b>WS20.</b> The 12 red dots are 3 placements and 9 screws. Green marks the part and screw supply trays.
The robot is the bottleneck (150.2 s), and the pick-and-place reach margin is 8 mm. SSM violation {k['WS20']['share_active_time_ssm_violation']:.0%}.</figcaption></figure>
<figure><img src="{b64img(PKG / 'WS30' / 'preview.png')}"><figcaption><b>WS30.</b> After the worker inserts the speaker, the robot drives 4 screws.
SSM violation is {k['WS30']['share_active_time_ssm_violation']:.0%} if the worker stays at the door and {kaway['share_active_time_ssm_violation']:.0%} if the worker leaves.</figcaption></figure>
</div>
</section>"""
    cover = f"""
<section class="cover">
  <div class="kicker">ROBOT SIMULATION HAND-OFF · v2</div>
  <h1>CarDA car-door assembly line<br>Cobot deployment scenario</h1>
  <div class="sub">The recommendation derived from video and motion capture of the Stellantis Melfi experimental line (WS10·WS20·WS30), packaged so the simulation team can reproduce and verify it as is.</div>
  <div class="hero"><img src="{b64img(frame10)}"></div>
  <div class="cards">
    <div class="card"><b>3 robots</b><span>Doosan M0617<br>1 per station, linear track</span></div>
    <div class="card"><b>211 s</b><span>Max station cycle<br>(takt 240 s, cap 228 s)</span></div>
    <div class="card"><b>17 screws</b><span>fastened<br>+ 3 part placements</span></div>
    <div class="card"><b>{k['WS10']['share_active_time_ssm_violation']:.0%}</b><span>WS10 SSM violation time<br>→ main item to verify</span></div>
  </div>
  <div class="meta"><span>Author: industrial engineering team · Data: CarDA (CC BY 4.0)</span><span>Package: out/sim_package · Document: docs/SIM_HANDOFF.md</span></div>
</section>"""
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>CarDA simulation handoff v2</title>
<style>{CSS}</style></head><body>{cover}<main>{body}</main>{figs}</body></html>"""
    out_html = ROOT / "out" / "SIM_HANDOFF_print.html"
    out_html.write_text(html)
    pdf = ROOT / "docs" / "SIM_HANDOFF.pdf"
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--run-all-compositor-stages-before-draw",
                    f"--print-to-pdf={pdf}", out_html.as_uri()], check=True, capture_output=True)
    print(pdf)
    return 0


if __name__ == "__main__":
    sys.exit(main())
