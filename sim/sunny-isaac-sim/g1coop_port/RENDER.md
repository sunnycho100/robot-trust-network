# Rendering the metal shop (g1coop_port)

Seohee's MuJoCo factory (`sim/mujoco_houses`, the 20-robot plant with the 1-, 2- and 3-G1 carries) replayed in
NVIDIA Isaac Sim 6.2 and rendered with RTX path tracing. MuJoCo does all the physics; Isaac Sim only renders the
recorded poses. Run the steps below in order on a fresh GPU host.

## 0. What you need

- An NVIDIA RTX GPU with at least 24 GB (tested on an L40S, 48 GB, driver 565) on Ubuntu 22.04, with Docker and
  the NVIDIA container toolkit. RT cores matter: A100 and H100 have none and render slower than an L40S.
- Disk: about 21 GB for the Isaac Sim image, 2 GB for the inputs (step 2), and up to 1 GB per 1080p clip of
  frames while it renders. The HQ runner stops itself below 5 GB free.

## 1. Environment

1. Isaac Sim 6.2 container: follow `sim/sunny-isaac-sim/README.md` ("Order of operations": `brev_setup.sh`, then
   `ACCEPT_EULA=Y bash isaac_up.sh up` once to pull the image and compile shaders). Renders here run in
   throwaway containers through `in_isaac.sh`, which mounts `~/robot` at `/robot`, so the work folder must be
   `~/robot/g1coop` (the scripts call `/robot/g1coop/...` inside the container). Stop the streaming app
   (`isaac_up.sh down`) while batch rendering.
2. MuJoCo side, from a checkout of this repo:
   ```bash
   bash sim/sunny-isaac-sim/g1coop_port/setup_vm.sh
   ```
   It copies these scripts and Seohee's sim (`sim/mujoco_houses/src`, `factory`, `assets`) into `~/robot/g1coop`,
   clones roqsim at its pinned commit `6e6999be07e56ebf442b2f20225f33b97c3a9f70`, fetches the G1 model and
   walking policy (unitree_rl_gym `276801e`, the files `scripts/fetch_assets` pins), and makes a Python 3.13
   venv with `mujoco==3.17.0`, CPU torch, `usd-core`, numpy and pillow.

## 2. Inputs (once, about 5 minutes)

```bash
cd ~/robot/g1coop && bash record_v3.sh
```

- Records 45 s of the plant with `realistic_layout_v3.json` (side-squeeze crate and T-frame carries from
  `side_grip.py`, wide placement surfaces) into `rec_v3/` and prints her pass checks: all three carries should be
  `AVAILABLE pass True`, the dual-arm panel `True`, QC `DONE`, both Spots walking.
- Runs `check_penetration.py` (how deep any G1 mesh goes into a load; 0.0 cm expected).
- Builds `usd_v3/factory.usd` (every body animated by time samples, one camera per shot from `shots.py`).
- Fetches the CC0 texture sets into `tex_cc0/` (`fetch_tex.py`, ambientCG) and writes the look layer
  `usd_v3/look_hq.usda` (`look_hq.py`: textured materials, HDRI, key and fill lights, chamfered pallets,
  strap seals, welds, background props). The HDRI and NVIDIA base materials load from the Isaac asset root at
  render time.

After editing `shots.py`, rebuild the USD so the cameras change:
`venv/bin/python build_usd.py rec_v3 usd_v3 && venv/bin/python look_hq.py usd_v3`.

## 3. Shots

`shots.py` holds every camera (sim start and end, caption, tracked body, lens):

- Film: `Overview` (the "Flexible Metalworks" wide shot), `ThreeG1`, `TwoG1`, `OneG1`.
- Close-ups per section: `ThreeG1Close`, `TwoG1Close`, `OneG1Close`, `CNCArm`, `CleaningArm`, `AssemblyArm`,
  `QCPanda`, `PackagingArm`, `TwoArmPanel`, `CourierLanes`, `SpotInspection`, `SpotQuality`, `Shelving`, `Dock`.

Framing check, one medium still per shot (the middle frame, with caption) into `shot_stills/SHOT.png`:
```bash
bash stills.sh                 # or: bash stills.sh CNCArm Dock
```

## 4. Medium renders (720p, captions and credit strip, path tracing 64 spp)

One shot:
```bash
bash in_isaac.sh dock.log /robot/g1coop/render_video.py /robot/g1coop/usd_v3/factory.usd /robot/g1coop/dock \
  --mode PathTracing --spp 16 --subframes 4 --shots Dock
ffmpeg -framerate 30 -i dock/frame_%05d.png -c:v libx264 -pix_fmt yuv420p -crf 18 dock.mp4
```

Film plus the three carry close-ups, as parallel chunks of at most 7 s (`shot:start:end`):
```bash
TAG=v3 FILM="Overview:0:5 Overview:5:10 ThreeG1:5:11 ThreeG1:11:17 TwoG1:7:12.5 TwoG1:12.5:18 OneG1:0:5" \
CLIPS="ThreeG1Close:4:11 ThreeG1Close:11:18 TwoG1Close:5:12 TwoG1Close:12:19 OneG1Close:0:5 OneG1Close:5:10" \
bash run_parallel.sh usd_v3 v3 --mode PathTracing --spp 16 --subframes 4
```
Concurrency comes from `v3/maxjobs` (default 4, editable while it runs). Kit processes start one at a time
(simultaneous cold starts stalled for minutes), then render in parallel. Outputs: `v3/g1coop_factory_v3.mp4`,
`v3/close_*_v3.mp4`, a contact sheet each, and `v3/gpu.log`. Any shot from section 3 can go in `CLIPS`.

## 5. HQ renders (1080p, look pass, no overlay)

```bash
bash hq_clip.sh TwoG1Close   close_2g1_sheets_v3_hq
bash hq_clip.sh ThreeG1Close close_3g1_tframe_v3_hq
bash hq_clip.sh Overview     g1coop_overview_v3_hq
```
Settings: `usd_v3/look_hq.usda`, path tracing 32 spp x 2 subframes x 3 motion-blur samples (180 degree
shutter), depth of field `--dof 60` (on this metre-scale stage Kit reads the f-stop about 100 times wider, so 60
gives a mild, natural depth of field), subtle bloom and vignette (`--post`), full-resolution textures (`--hq`),
no title bar, time stamp or credit strip (`--no-overlay`). Credits and texture licences go in a sidecar, for
example `docs/evidence/g1coop_port/close_2g1_sheets_v3_hq_credits.txt`.

A 4K still of one moment: add `--size 3840 2160 --spp 64 --span 16.7 16.73` to the `render_video.py` call.

Several `hq_clip.sh` runs can share one GPU (about 5 GB VRAM each), but path tracing keeps the L40S at 100%, so
running them together does not finish sooner. A second GPU halves the time: give each GPU part of the shot with
`--span A B --first-index N` on `render_video.py`, then encode the joined frame folder.

**Resume after a crash:** run the same `hq_clip.sh` command again. It deletes a truncated last frame, counts the
frames on disk and continues from there.

## 6. Timings on the L40S (one process alone unless noted)

| Render | Time |
|---|---|
| Record, check, build USD, look layer (step 2) | about 5 min |
| Medium 720p, 64 spp | about 1.3 s per frame; four processes together about 0.7 to 0.8 frames per second in total; the full v3 set (2,280 frames) took 54 min, peak 22 GB VRAM |
| HQ 1080p look pass | about 15.6 s per frame; a 14 s close-up (420 frames) about 1 h 50 min on one GPU, about 55 min on two |
| HQ 4K still (look pass, 3 motion samples) | about 90 s |
| Framing still, 720p | about 10 s each plus about 30 s Kit start |

## 7. Where outputs land

All under `~/robot/g1coop`: `rec_v3/` (recording), `usd_v3/` (USD, textures, look layer), `tex_cc0/`, `v3/`
(medium chunks and videos), `hq_NAME/` frames and `NAME.mp4` plus `NAME_sheet.png` (HQ), `shot_stills/`. Copy the
finished videos and sheets into `docs/evidence/g1coop_port/` (medium) and `docs/evidence/g1coop_port/hq/` (HQ),
then delete the frame folders.

## 8. Regenerated, not stored in git

- `rec_v3/` (`plant.mjb` about 1.1 GB, `poses.npz`, `plant.xml`, `report.json`): `record_v3.sh`
- `usd_v3/` (about 85 MB) and `look_hq.usda`: `build_usd.py`, `look_hq.py`
- `tex_cc0/` (ambientCG texture sets): `fetch_tex.py`
- the roqsim clone (about 480 MB), the G1 model and policy, the venv: `setup_vm.sh`
- the Isaac Sim image (about 21 GB), the HDRI and NVIDIA base materials (pulled by Isaac Sim at render time)
- frame folders and intermediate chunk videos

## Notes

- `side_grip.py` adds the `crate_side` and `fixture_side` scenarios beside her code at runtime (her files are
  not modified). `tote_side`, a level two-palm tote grip, is experimental and not used: the tote rolled between
  the palms or the robot never finished stepping back. The tote uses her `sofa_side` grip.
- `sweep_grip.sh` runs grip probes in parallel (one process per variant).
- To browse the result, open `usd_v3/look_hq.usda` (or `factory.usd`) in Isaac Sim and scrub the timeline; the
  cameras are under `/World/Cameras`.
