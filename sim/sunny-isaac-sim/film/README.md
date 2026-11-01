# Demo films

Video tools that turn our simulation runs into short films in the website's look. Fonts are the site's (Google Fonts, OFL) and are not committed. Fetch them once:

```bash
mkdir -p sim/sunny-isaac-sim/film/fonts && cd sim/sunny-isaac-sim/film/fonts
curl -sSfLo Archivo.ttf "https://github.com/google/fonts/raw/main/ofl/archivo/Archivo%5Bwdth,wght%5D.ttf"
curl -sSfLo IBMPlexMono-Medium.ttf "https://github.com/google/fonts/raw/main/ofl/ibmplexmono/IBMPlexMono-Medium.ttf"
```

| File | What |
|---|---|
| `render_process_film.py` | Site-to-simulation film from our scan `floor4_static3`: video frames, camera poses, point cloud, surface mesh, collider, then a G1 walking in the clean collider room (MuJoCo). The docstring lists which parts are pipeline output and which are styling. |
| `ws30_process_film.py` | Process film v4 on the real CarDA WS30 station: CarDA camera clips, the frame frozen with labels on the key objects, a wipe into the rebuilt station (`--station`), then robots doing the door job (`--robots`). Missing inputs render as placeholder cards. |
| `steel_plate_intro.py` | Seohee's factory overview, a zoom into the sheet cell, "Lift this steel plate", then a CooHOI pair clip from `coohoi_lab/play.py --plates 8`. |
