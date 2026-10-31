# Simulations so far

Every simulation we have run, with what it shows, the result, where it ran and where the code lives. Videos are kept out of git (`docs/evidence/` stays local); the paths below are where they sit on Sunghwan's Mac, and the website share link has the main clips.

Where it ran: **Brev** is the NVIDIA Brev L40S VM, **Colab** a Colab L4, **Mac** the laptop. "Kinematic replay" means poses are computed or recorded and drawn frame by frame, with no physics engine deciding the motion; everything else is physics with a controller or policy in the loop.

## 1. Real places into simulation

| Simulation | What it shows | Result | Ran on | Code | Video |
|---|---|---|---|---|---|
| Scan to MuJoCo (`floor4_static3`) | A scanned room (RGB-D frames, LiDAR poses, TSDF mesh) turned into a MuJoCo scene: floor plane, wall boxes and convex hulls as the collider, the scan mesh for looks. A G1 walks in it on Unitree's published policy. | Walk test to open and near-obstacle goals and into every wall, against the same goals on a bare floor; collider checked against the scan mesh. Numbers in the folder's README. | Mac | `sim/sunny-scene-recon/` | |
| Process film | The same scan as one story: video frames, camera poses, point cloud, surface mesh, collider, then a G1 fading in and walking in the cleaned collider room. | 38 s film. The clean walls, camera moves and fade-in are styling; the docstring lists which parts are pipeline output. | Mac | `sim/sunny-isaac-sim/film/render_process_film.py` | `process/process_film.mp4` |
| Our scan in Isaac Sim | `floor4_static3` imported into Isaac Sim 6.2 with an H1 on the stock walking policy. | Loads with its texture; the H1 walks but the room is small (4.5 x 4.1 m) and it ends against a wall. | Brev | `demos/import_our_scene.py`, `demos/record_ours.sh` | `isaac/h1_our_scene.mp4`, `isaac/h1_our_center_walk.mp4` |
| Real loading area (ETH3D `delivery_area`) | A public DSLR capture of a covered loading area: 3DGRUT splat for looks, PatchMatch stereo mesh as the invisible collider, no ground plane. An H1 plans an A* path on a map built from the collider. | H1 walked about 11 m to the goal in 17.6 s, no fall. The collider is within 5 cm of the laser scan on 79% of pixels; the splat sits 54 cm off, so it is for looks only. | Colab | `demos/multi_robot_scene.py`, `colab/` recipes | `overnight/fleet/dah1b.mp4`, `dah1c.mp4` |
| Three robots share one spot in that scan | Two H1s and a Go2 take one hand-off spot in turn (2 s each), then cross the floor. | All arrived (11, 18 and 22 s), closest 1.09 m, no falls. On a crack-filled TSDF collider the H1s also finish; the Go2 stalls 0.45 m short (cause not found). | Colab | `demos/fleet.py`, `colab/dafleet.sh`, `dafill_fleet.sh` | `overnight/fleet/dafleet.mp4`, `dafill.mp4` |
| H1 in a photoreal scan (NVIDIA NuRec wormhole) | NVIDIA's own scan of a room, rendered as a particle field, with a ground plane for physics. | H1 walks across the room. The scan is NVIDIA's (CC BY 4.0), not ours. | Brev | `demos/h1_in_scene.py`, `demos/record_more.sh` | `isaac/h1_wormhole_walk.mp4` |

## 2. Stock Isaac Sim robots and fleets

| Simulation | What it shows | Result | Ran on | Code | Video |
|---|---|---|---|---|---|
| Stock policy demos | Spot, Go2, ANYmal, H2 walking; Franka; UR10e pick and place (cuMotion); UR10 palletizing; three H1s in the warehouse. | Recorded headless, one clip each. | Brev | `demos/record_all.sh`, `record_more.sh`, `h1_warehouse.sh` | `isaac/*.mp4` |
| Fleet in the warehouse | H1, Go2 and ANYmal with goals, giving way to each other, A* around shelves, one station used in turn. | Without avoidance robots collide (closest 0.01 m); with it none do (closest 1.15 m), at 6% to 87% more time. | Colab, Brev | `demos/multi_robot_scene.py`, `fleet.py`, `navmap.py` | `overnight/colab_multi.mp4`, `overnight/fleet/*.mp4` |

## 3. Two humanoids carrying one object

| Simulation | What it shows | Result | Ran on | Code | Video |
|---|---|---|---|---|---|
| CooHOI in Isaac Gym | The published CooHOI checkpoint (NeurIPS 2024) in its own simulator. | Runs, but each humanoid lives in its own env with a copied box, so one camera sees one humanoid. | Brev | `sim/sunny-isaac-sim/coohoi/` | `isaac/coohoi_twoagent.mp4` |
| CooHOI ported to Isaac Lab 3.0 | The same weights driving two humanoids around one real shared box in Isaac Sim 6.2 physics, no retraining. | Single 81%, pair 88% success (32 episodes each), from 0% before fixing the humanoid asset (knee and elbow drives, masses, root link). | Brev | `sim/sunny-isaac-sim/coohoi_lab/` | `isaac/coohoi_lab_pair_cut.mp4` |
| Steel plate story | Seohee's factory overview, zoom into the sheet cell, "Lift this steel plate", then the pair carrying a strapped steel plate stack. | The stack is CooHOI's trained box drawn as steel plates (`--plates 8`). A real thin plate failed every time: 0% at 8 to 25 cm thick, also on trestles. | Brev, Mac | `coohoi_lab/play.py`, `film/steel_plate_intro.py` | `isaac/steel_plate_story.mp4` |
| Pair carry in the loading area scan | The CooHOI pair inside the rebuilt `delivery_area`. | Paused. The scan has no straight 11 m of free floor, so CooHOI's start puts the humanoids at walls and both seeds fell. `play_scene.py --home` shortens the start; state is on the VM in `/data/da/`. | Brev | `coohoi_lab/play_scene.py`, `colab/brev_*.sh`, `colab/da_coohoi_scene.py` | |
| Two-agent training check (MAPPO) | Isaac Lab 3.0's multi-agent trainer on NVIDIA's two-hand handover, the path for fine-tuning CooHOI if needed. | Success 0.7% to 47% in 21 minutes. | Colab | `colab/lab_mappo.sh` | |

## 4. Car-door line (CarDA)

| Simulation | What it shows | Result | Ran on | Code | Video |
|---|---|---|---|---|---|
| WS30 cobot cell, MuJoCo rebuild | The SIM_HANDOFF v2 study's WS30 baseline rebuilt from its PDF, Doosan's M0617 URDF and CarDA's motion capture (CC BY 4.0): a cobot on a track drives screws while the worker's recorded motion plays. Also with our G1 in the worker's place. | Matches the study: closest approach -0.16 m (theirs -0.15 m), inside the safety distance 86% of moving time. With the G1, the arm waits 2.5 s and keeps at least 0.76 m away. | Colab | `sim/handoff/`, analysis in `docs/research/sim_handoff_analysis.md` | `handoff/ws30_*.mp4`, `handoff/g1_ws30_*.mp4` |
| Option C line in Isaac Sim | The v3 request's option C as a picture: three stations, one cobot per station on a track that follows the belt, screw counts from the study (4, 9 with a tool change, 4), an AMR bringing kits, a worker fitting glass at the WS10 door with a separation ring that turns red. | One 32 s kinematic replay. Not the study's analysis: no P1 to P3 policies, no repeated runs, no cycle times. The UR10e is a visual stand-in for the M0617 and failed the study's own s4 screening at WS10 (reach and payload), so reach or cycle-time checks need the M0617. WS30 shows only the "worker leaves" case. Inside the protective distance 81% of the time is a demo number, not comparable to the study's 71%. | Colab | `sim/sunny-isaac-sim/carda/` | `carda/carda_line.mp4` |

## 5. Teammates' simulations

| Simulation | What it shows | Result | Ran on | Code | Video |
|---|---|---|---|---|---|
| G1 carries and duo (Seohee) | G1 and Go1 on published walking policies talking through signed MQTT requests: a WASD G1, a leader and follower duo, a two-G1 cooperative lift, two RoboCasa houses. | See the folder's README. | Mac | `sim/mujoco_houses/` | |
| 20-robot factory (Seohee, PR 10) | Six G1s, four arm cells, a two-arm panel, six couriers and two Spots in roqsim's Depot warehouse, with a production scheduler. | Her own checks pass. Under review in PR 10. | Mac, Brev | branch `g1_cooperative_task` | her video |
| That factory in Isaac Sim | Her MuJoCo run replayed in Isaac Sim's RTX renderer: the USD is built from her compiled model (4,513 geoms, 13 textures) and every body follows her recorded poses. | 39 s video. Visual port only: MuJoCo decides all motion. The USD is on the VM at `~/robot/g1coop/usd/factory.usd`. | Brev | `sim/sunny-isaac-sim/g1coop_port/` | `g1coop_port/g1coop_factory.mp4` |
| G1 manipulation (Sam) | Articulated G1 with grippers, balance, arm IK and camera-guided manipulation. | See its review doc. | Mac | `sim/sam-g1-manipulation/` | |

How to run Isaac Sim on the VM after `brev shell isaac-sim`: [`sim/sunny-isaac-sim/README.md`](../sim/sunny-isaac-sim/README.md), then [`demos/RUN.md`](../sim/sunny-isaac-sim/demos/RUN.md) and [`RUNSHEET.md`](../sim/sunny-isaac-sim/RUNSHEET.md).
