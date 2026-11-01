"""Fetch the CC0 PBR texture sets look_hq.py uses (ambientCG, 2K JPG) into ~/robot/door_team/tex_cc0.
Usage (on the render host, outside the container): python3 fetch_tex.py [OUT_DIR]
Sets: Metal009 (brushed steel), Metal038 (dark steel), Metal027 (paint normals), Concrete048 (floor),
Rubber004 (suction cups, grips), Plastic013A (driver body)."""
import io
import pathlib
import sys
import urllib.request
import zipfile

SETS = ("Metal009", "Metal038", "Metal027", "Concrete048", "Rubber004", "Plastic013A")
out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else pathlib.Path.home() / "robot/door_team/tex_cc0")
for s in SETS:
    d = out / s
    if (d / f"{s}_2K-JPG_Color.jpg").exists():
        print("[tex] have", s)
        continue
    url = f"https://ambientcg.com/get?file={s}_2K-JPG.zip"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 door-team-render"})
    d.mkdir(parents=True, exist_ok=True)
    zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(req, timeout=120).read())).extractall(d)
    print("[tex] got", s, flush=True)
