# Render handoff: the final high-quality videos

Two scenes are final. No more simulation changes, only high-quality (HQ) renders in NVIDIA Isaac Sim 6.2 with RTX path tracing.

| Scene | What it is | Full guide |
|---|---|---|
| Car-door station (v3c) | Two UR20 arms on rails carry a door to WS2; G1 B sets the glass with a suction lifter while G1 A drives a screw | [`door_team/RENDER.md`](door_team/RENDER.md) |
| Metal shop | Seohee's 20-robot MuJoCo factory, replayed in Isaac Sim | [`g1coop_port/RENDER.md`](g1coop_port/RENDER.md) |

## Render queue

Ask Sunghwan which clips are already done before starting; some may be rendering on our own GPU.

| # | Clip | Command | Output name |
|---|---|---|---|
| 1 | Car door, full clip, clean | `door_team/RENDER.md` section 7 | `door_team_v3c_hq_clean.mp4` |
| 2 | Car door, with event overlays (post pass on the frames from 1) | `door_team/RENDER.md` section 7 | `door_team_v3c_hq.mp4` |
| 3 | Metal shop, 2-G1 sheet carry close-up | `bash hq_clip.sh TwoG1Close close_2g1_sheets_v3_hq` | `close_2g1_sheets_v3_hq.mp4` |
| 4 | Metal shop, 3-G1 T-frame carry close-up | `bash hq_clip.sh ThreeG1Close close_3g1_tframe_v3_hq` | `close_3g1_tframe_v3_hq.mp4` |
| 5 | Metal shop, global view | `bash hq_clip.sh Overview g1coop_overview_v3_hq` | `g1coop_overview_v3_hq.mp4` |
| 6 | Metal shop, one zoom-in per section (optional) | `bash hq_clip.sh SHOT NAME` for each section shot in `g1coop_port/RENDER.md` section 3 | `SHOT_v3_hq.mp4` |

Check framing with medium stills first (`bash stills.sh` for the metal shop) before committing hours to an HQ clip.

## Decided HQ settings

Both scenes: 1920 x 1080, path tracing with the OptiX denoiser, CC0 textures from ambientCG, an HDRI dome plus soft area lights, a 180 degree motion-blur shutter, mild depth of field, and no text burned in (credits go in a sidecar file).

| | Car door | Metal shop |
|---|---|---|
| Samples | 8 spp x 2 subframes x 2 motion samples (32 per pixel) | 32 spp x 2 subframes x 3 motion samples |
| Depth of field | `--dof 400` (subtle on this stage) | `--dof 60` |
| Extras | OmniGlass on the window pane | Bloom and vignette (`--post`), full-resolution textures (`--hq`) |
| Time on an L40S | about 4 h for 1,580 frames, shared GPU | about 15.6 s per frame; a 14 s close-up is about 1 h 50 min |

The two scenes' f-stop values differ because the stages are built at different scales; each guide explains its own.

## Hardware and environment

- An NVIDIA RTX GPU with RT cores and at least 24 GB (we used an L40S, 48 GB). A100 and H100 have no RT cores and path-trace slower.
- Ubuntu 22.04, Docker and the NVIDIA container toolkit, the Isaac Sim 6.2.0 container. `brev_setup.sh` and `isaac_up.sh` in this folder set it up (see `README.md`, "Order of operations"). `isaac_aws.sh` runs the same stack on AWS.
- Disk: about 21 GB for the image, plus up to 1 GB of frames per clip. The HQ runners stop themselves below 5 GB free.
- Several processes can share one GPU (about 5 to 10 GB VRAM each), but path tracing keeps it at 100%, so only a second GPU makes the queue finish sooner. Each guide shows how to split a clip across GPUs by frame range.

## What comes back

The finished mp4s and their contact sheets. Videos are not stored in git; send them to Sunghwan, who keeps them in `docs/evidence/door_team/v3c_hq/` and `docs/evidence/g1coop_port/hq/`.
