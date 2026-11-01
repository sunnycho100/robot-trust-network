# Door team scene: how to rebuild and render it

The final car-door scene (v3c) puts two UR20 arms on rails and two Unitree G1 humanoids inside the CarDA WS30 station. The arms take the door off its trolley on the floor belt and carry it to a stand at WS2. G1 B sets the window glass with a suction lifter while G1 A drives a latch screw from the other face of the door. One orchestrator drives everything on events, and each custody step is signed. This file covers rebuilding it from a fresh machine, in run order.

**What is physics and what is replay.** The G1 legs and root come from Unitree's published walking policy, stepped in MuJoCo by `orchestrate.py`. Everything else is replayed kinematically in Isaac Sim: the UR20 joints (IK in `rig.py`), the G1 arms (IK to the hand targets), the door, glass, lifter and tools, and the background loops. PhysX is not simulating anything at render time.

## Files

| File | Role |
|---|---|
| `orchestrate.py` | The cell: tasks, team dispatch and signed custody events (flexible_factory vocabulary), agent programs, the G1 MuJoCo walks. Writes the plan. |
| `rig.py` | Joint trees read from the stock USD robots: UR20 tip IK, G1 hand IK. Used by the render and the checks. |
| `door_team.py` | Isaac Sim 6.2 render of a plan inside the WS30 station. Medium mode (RTX real time, 1280 x 720, PNG) or `--hq` (1080p path tracing, JPEG). |
| `plant.py` | Background around the station: body line, fenced weld cell, neighbouring pick-and-place station, racks, KLT carts, AMR. |
| `look_hq.py` | HQ looks: textured materials, OmniGlass, HDRI dome, soft area lights. |
| `overlay.py` | Event timeline, labels and footer drawn onto rendered frames (PNG or JPEG, any size). |
| `export_dots.py` | Section 03 dots data (`today.bin`, `team.bin`, `manifest.json`, `dots_data.js`), and `--check`, the clearance check. |
| `test_rig.py` | IK check: every arm tip and G1 hand lands on its target over the whole plan. |
| `run_v3c.sh` | Medium render runner: chunks in parallel containers, clean mp4, overlay mp4, contact sheet. |
| `run_hq.sh` | HQ render runner, with a disk guard. |
| `finish_hq.sh` | After the HQ render: clean mp4, overlays in post, overlay mp4, contact sheet. |
| `kill_mine.sh` | Stops only this scene's render containers, leaving other jobs on a shared GPU alone. |
| `fetch_tex.py`, `fetch_assets.sh` | Texture sets for HQ, and local robot USDs for the laptop-side checks. |
| `../carda/kin.py` | Shared FK and IK math (`Kin`, `tf`, `rot`, `minjerk`). |

**Not in git, regenerated:**
- plan directories (`plan.npz`, about 1.2 MB, plus the json files)
- frames and mp4s
- the local asset copies and textures
- the dots binaries and `dots_data.js`, made by `export_dots.py`; the copies in `docs/evidence/web/dots_team/` are outputs

## 1. Environment

**Laptop side (plan, checks, dots export):**
- Python 3.12 and ffmpeg.
- The MuJoCo venv: `cd sim/mujoco_houses && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`. It needs mujoco, torch and cryptography.
- The policy submodule: `git submodule update --init sim/mujoco_houses/third_party/unitree_rl_gym`. The G1 walking policy is its `deploy/pre_train/g1/motion.pt`.
- A small USD venv for the checks: `python3 -m venv usdenv && usdenv/bin/pip install usd-core numpy`.
- Signing uses `network/protocol.py` (Ed25519) through `unitree_demo/trust_session.py`. Keys are generated fresh per run and only the public registry is written.

**Render host:**
- An NVIDIA RTX GPU with at least 24 GB (we used an L40S, 48 GB).
- Docker with the NVIDIA container toolkit.
- The Isaac Sim 6.2.0 container image from NVIDIA NGC (`isaac-sim:6.2.0`; the runners name it in `ISAAC_SIM_IMAGE`).
- `sim/sunny-isaac-sim/brev_setup.sh` sets this up. It sparse-clones the IsaacSim repo at v6.2.0 to `~/IsaacSim` for `tools/docker/docker-compose.yml`, and installs the container.
- Copy `sim/sunny-isaac-sim/docker-compose.override.yml` to `~/robot/isaac-sim/`. It mounts `~/robot` at `/robot` inside the container. The runners use both compose files.

## 2. Assets and where they come from

| Asset | Source | How it is fetched |
|---|---|---|
| UR20 | `/Isaac/Robots/UniversalRobots/ur20/ur20.usd` on the NVIDIA Isaac Sim 6.2 asset root | Streamed by Isaac Sim at render time; `fetch_assets.sh` makes local copies for the checks |
| Unitree G1 (29 DoF, three-finger hands) | `/Isaac/Robots/Unitree/G1/g1.usd` | Same |
| Fanuc CRX-10iA (neighbouring station) | `/Isaac/Robots/Fanuc/crx10ia/crx10ia.usd` | Same |
| Kuka KR210 L150 (weld cell) | `/Isaac/Robots/Kuka/KR210_L150/kr210_l150.usd` | Streamed |
| AMR (iw.hub) | `/Isaac/Robots/Idealworks/iwhub/iw_hub_static.usd` | Streamed |
| WS30 station, trolley and door | `ws30_station.usd`, built from the CarDA WS30 video by `sim/sunny-isaac-sim/ws30_station/ws30_station.py` (layout in `layout.json`). A built copy is in git at `sim/sunny-isaac-sim/door_team/inputs/ws30_station.usd` | Copy to `~/robot/ws30_station/out/ws30_station.usd` on the render host, or pass `--station` |
| Real worker clip (split screen, left) | `ws30_human_alone.mp4`, cut from the CarDA dataset (Zenodo 14644367, CC BY 4.0), WS30 cycle 003 | Not in git (local evidence only); optional, only the split screen uses it. Ask Sunghwan for the clip |
| HDRI | `/NVIDIA/Assets/Skies/Indoor/autoshop_01_4k.hdr` on the same asset root (Poly Haven, CC0) | Streamed |
| PBR textures | ambientCG CC0: Metal009, Metal038, Metal027, Concrete048, Rubber004, Plastic013A | `python3 fetch_tex.py` on the render host (to `~/robot/door_team/tex_cc0`). If another job already fetched the same sets, a symlink named `tex_cc0` to that folder works too |

The asset root is the public one Isaac Sim 6.2 uses by default (what `get_assets_root_path()` returns); `fetch_assets.sh` and `look_hq.py` name it.

## 3. Plan and signed events (laptop)

```bash
cd sim/mujoco_houses
.venv/bin/python ../sunny-isaac-sim/door_team/orchestrate.py ~/door_out/plan_v3c
```

**Outputs:**
- `plan.npz`: per-frame poses at 30 fps.
- `plan.json`: labels, events, layout.
- `events.jsonl`, `signed_events.json` and `registry.json` (public keys).

It verifies every signature and that a tampered copy is rejected before it exits. v3c runs 1,580 frames (52.6 s), and STATION 2 COMPLETE comes at 42.8 s. The walks are deterministic on CPU, but the signatures differ every run because the keys are new.

To verify a saved log later, use the message time as `now`:

```python
protocol.Verifier(reg["receiver"], reg["registry"]).check(r, now=r["time"])
```

## 4. Checks and the dots export (laptop)

```bash
bash fetch_assets.sh ~/door_out/assets
usdenv/bin/python test_rig.py ~/door_out/plan_v3c ~/door_out/assets
usdenv/bin/python export_dots.py ~/door_out/plan_v3c ~/door_out/assets \
    sim/sunny-isaac-sim/door_team/inputs/ws30_station.usd sim/sunny-isaac-sim/door_team/inputs/worker_plant.npz /tmp/x --check
usdenv/bin/python export_dots.py ~/door_out/plan_v3c ~/door_out/assets \
    sim/sunny-isaac-sim/door_team/inputs/ws30_station.usd sim/sunny-isaac-sim/door_team/inputs/worker_plant.npz docs/evidence/web/dots_team
```

**`test_rig.py`** should report UR20 tip error under 1 mm. G1 hands should be under 3 cm, except about 0.4 s while A loads the screw onto the bit.

**`--check`** prints the worst clearance per robot pair over the whole timeline. It uses 90 surface dots per part, every frame at 15 fps. Intended contacts are left out: grippers on the door, and G1 B's hands on the lifter and glass. The aisle G1 stays more than 3 m away and is not checked. Target: at least 5 cm everywhere. v3c's minimum is 8.6 cm, G1 B's elbow to the glass.

**The dots export** writes about 1.3 MB of binaries, plus `dots_data.js` for the site.

## 5. Render host layout

```
~/robot/door_team/   door_team.py rig.py plant.py overlay.py look_hq.py kin.py run_v3c.sh run_hq.sh finish_hq.sh kill_mine.sh fetch_tex.py
~/robot/door_team/plan_v3c/plan.npz, plan.json
~/robot/door_team/tex_cc0/          (HQ only: python3 fetch_tex.py)
~/robot/ws30_station/out/ws30_station.usd
```

## 6. Medium render (clean and with overlays)

```bash
cd ~/robot/door_team && OUTN=v3c DUR=52.7 JOBS=2 setsid nohup bash run_v3c.sh > run_v3c.log 2>&1 &
```

- **Outputs:** `door_team_v3c_clean.mp4` (no text), `door_team_v3c.mp4` (event timeline, labels, footer) and `door_team_v3c_sheet.jpg`.
- **Time:** about 2.5 minutes of stage loading and IK precompute per process. Then roughly 5 frames per second per process on a shared L40S; the whole job is about 8 to 10 minutes.
- **Disk:** about 1.5 MB per PNG frame, twice over with the overlay pass, so about 5 GB at peak. It skips the overlay if the disk falls under 8 GB free.
- **Split screen** (real worker on the left on the same cycle clock; the clip covers cycle 6 to 30 s):

```bash
ffmpeg -i docs/evidence/ws30_station/ws30_human_alone.mp4 -i door_team_v3c.mp4 -filter_complex \
 "[0:v]fps=30,tpad=start_mode=clone:start_duration=6:stop_mode=clone:stop_duration=80,scale=960:540,setsar=1[h];\
  [1:v]scale=960:540,setsar=1[t];[h][t]hstack=inputs=2:shortest=1" -c:v libx264 -pix_fmt yuv420p -crf 21 door_split_v3c.mp4
```

## 7. HQ render

```bash
cd ~/robot/door_team && python3 fetch_tex.py
NAME=v3c_hq JOBS=2 DUR=52.7 setsid nohup bash run_hq.sh --spp 8 --subframes 2 --mblur 2 --dof 400 > run_hq.log 2>&1 &
```

**Settings (the final render):**
- 1920 x 1080, PathTracing with the OptiX denoiser, texture streaming off.
- 8 samples per pixel per subframe, 2 subframes, 2 motion-blur time samples over a 180 degree shutter: 32 samples per pixel per frame before the denoiser.
- Depth of field focused on each shot's target. The stage is in metres, so the USD camera's f-stop scales differently from a real lens: `--dof 400` is a subtle blur, and single digits look like a miniature.
- Two chunk processes (`JOBS=2`) next to another path-traced job on the same GPU.
- Frames are JPEG quality 95, about 0.5 MB each, so about 0.8 GB for the clip.
- The runner stops its renders if the disk drops under 5 GB free.

**Timing:** a still test at 64 samples per pixel ran about 20 s per frame for one process on a shared L40S, so the final render uses 32. The measured time per frame for the final settings is in `docs/evidence/door_team/v3c_hq/HQ_RENDER.txt`. Budget about 4 hours for the 1,580 frames.

**Encode and overlays afterwards:** `finish_hq.sh` waits for `run_hq.sh` to end and does all of the following (clean mp4, contact sheet, overlays per chunk folder, overlay mp4); start it right after the render with `NAME=v3c_hq setsid nohup bash finish_hq.sh > finish_hq.log 2>&1 &`. By hand:

```bash
ffmpeg -framerate 30 -pattern_type glob -i "v3c_hq/rec*/frame_*.jpg" -c:v libx264 -pix_fmt yuv420p -crf 18 door_team_v3c_hq_clean.mp4
# overlays in post, inside the container, once per chunk folder (overlay.py lays the text out at 1280 x 720 and scales it):
#   ./python.sh /robot/door_team/overlay.py /robot/door_team/v3c_hq/rec0 /robot/door_team/plan_v3c /robot/door_team/v3c_hq/ov
#   ./python.sh /robot/door_team/overlay.py /robot/door_team/v3c_hq/rec1 /robot/door_team/plan_v3c /robot/door_team/v3c_hq/ov
ffmpeg -framerate 30 -pattern_type glob -i "v3c_hq/ov/frame_*.jpg" -c:v libx264 -pix_fmt yuv420p -crf 18 door_team_v3c_hq.mp4
```

## 8. Resume, and sharing a GPU

- **Resume:** frames are named by their global index (`frame_01234`), and every chunk writes its own `rec<i>/meta.json`.
  1. Re-render a missing range into a new folder, with later `--start` and `--secs` flags; the last value wins. For HQ, also give a new `NAME`, because the runner wipes its own folder when it starts.
  2. Run the overlay once per folder.
  3. Encode with a glob over all folders.
- **Sharing a GPU:** at most two processes at once. `kill_mine.sh` stops only containers whose arguments name `/robot/door_team/door_team.py`.
- **VRAM:** a medium process uses about 4 to 5 GB; HQ at 1080p uses more, about 8 to 10 GB.

## 9. Camera shots (v3c, times in seconds)

| From | Shot |
|---|---|
| 0.0 | Overhead of the station, below the hall lights |
| 3.0 | Zoom onto the trolley as the arms grasp the moving door (3 s blend) |
| 6.3 | Low, along the OUT rail |
| 9.0 | Following the door along the rails |
| DOOR DELIVERED + 0.3 (19.6) | G1s arriving at WS2 |
| B approaching + 2 (27.4) | Both G1s at work on the two faces of the door, high, from past the downstream end |
| A driving - 0.4 (36.1) | Screw close-up on the outer face |
| screw seated + 0.4 (41.6) | Glass and the suction holder |
| STATION 2 COMPLETE - 0.5 (42.3) | Closing view of WS2 |

Every cut blends over 1.2 s. The shot list lives in `door_team.py` (search for `SHOTS`).
