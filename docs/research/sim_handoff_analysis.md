# Simulation handoff (CarDA cobot cell): analysis and Colab run (2026-10-31)

Input: `~/Downloads/SIM_HANDOFF.pdf` (Korean, 10 pages, "CarDA 차 문 조립 라인 협동로봇 도입 시나리오, 시뮬레이션 팀 전달서 v2"), received over KakaoTalk, plus two mp4 files in `~/Downloads`.
Code from this session: `sim/handoff/` (uncommitted). Evidence: `docs/evidence/handoff/`.

## 0. Short answer

- The handoff is an **industrial engineering study, not a robot learning or fleet project**: three Doosan M0617 cobots on linear tracks taking over screwdriving on a moving car-door line (CarDA data, Stellantis Melfi experimental line), checked in **MuJoCo as a kinematic replay** (IK waypoints, joint-space interpolation, a mocap worker as capsules) against takt time and ISO/TS 15066 speed and separation monitoring (SSM).
- **It runs on MuJoCo, as Sunghwan thought**, but the runnable package it describes (`out/sim_package/`, `replay.py`) was **not in the PDF and is not on the Mac**. The PDF has no links. The two mp4 files are **our own Isaac Sim videos**, byte for byte (section 1).
- So I **rebuilt its WS30 baseline from the PDF's numbers and public data** (Doosan's URDF, CarDA's mocap for the same representative cycle) and ran it on a Colab L4. It reproduces the handoff's as-is numbers closely (min distance -0.16 m against their -0.15 m, SSM violation 86% against 86%).
- **The teammate's MuJoCo video** (section 9) is a whole-factory overview: 19 robots of several types doing scripted work orders in a synthetic depot. It shows breadth where the handoff shows depth on one station; ranked expansion options are in 9.4.
- **Overlap with us is thin but useful**: same simulator, same "real data to sim" idea, same need to measure separation and share a station in turn. It adds what our demos lack: **people**, a **fixed industrial arm at a station**, and a **safety metric** (SSM) with interlock policies. I put our G1 in the worker's place as a first robot-to-robot hand-off of the door.

## 1. What was received

| Item | What it is |
|---|---|
| `SIM_HANDOFF.pdf` | The handoff document v2. Describes a package (`out/sim_package/`: per-station `scene.xml`, `targets.json`, `robot_reference.json`, `human_asis.npz`, `human_tobe.npz`, `motion.json`, KPI JSON, trace CSV, `replay.py`, `scenario_matrix.yaml`) and the pipeline that made it (`scripts/export_sim_package.py`, `scripts/run_all.py`, stages s0 to s9). **None of these files came with it**; no URL or repo name is in the PDF (checked the text and the PDF's link annotations), and Spotlight finds none of the file names on the Mac. |
| `7tsUwdqdvUHW9KLjJ10hpg.mp4` (20 s) | **Our** `docs/evidence/overnight/colab_multi.mp4` (same MD5, `aac882cf...`): five robots in NVIDIA's warehouse, Isaac Sim 6.2 on Colab. |
| `Yy9HyEobRVOCItcC-sIAlA.mp4` (45 s) | **Our** `docs/evidence/overnight/fleet/dafleet.mp4` (same MD5, `31483cdc...`): two H1s and a Go2 sharing a hand-off spot in the reconstructed ETH3D delivery_area. |

Both videos carry a "Claude" download tag (13:06 and 13:08 today), so they were most likely saved from a chat, not sent with the PDF. The handoff's own demo videos (for example `WS10/replay_tobe.mp4`) are not here. Its figures (pages 1 and 10) show MuJoCo's default renderer: grey capsule arm on a pedestal and rail, orange capsule worker, black door box with red target dots, green tray (`docs/evidence/handoff/handoff_pdf_ws30_figure.png`).

## 2. What the handoff simulation is

**Question it answers** (PDF section 0): does the recommended layout (one M0617 per station, WS10, WS20, WS30, each on a linear track that follows the conveyor) hold takt 240 s (cap 228 s); how should robot and worker share the same door under ISO/TS 15066 SSM (P1 parallel with SSM, P2 zone exclusive: robot moves only while the worker is at least 1.0 m from the active target, P3 sequential); and do the resulting waits keep the industrial engineering conclusions (robot count, ROI). Results feed back into their pipeline through `robot_task_times.json` (stages s5 to s9 recompute allocation, line DES, ROI, RFQ).

| Aspect | Handoff |
|---|---|
| Robots | 3 Doosan M0617 (6 kg, 1.7 m reach), pedestal 0.9 to 1.0 m, linear track following the belt (about 24 mm/s). WS20 swaps a Robotiq 2F-85 gripper and an OnRobot screwdriver through a tool changer. |
| Task | Car-door assembly on a moving line: 17 screws plus 3 part placements across three stations. Workers keep the flexible parts (glass, weatherstrip, speaker insertion). |
| Simulator | MuJoCo, one MJCF per station, in a "belt frame" that moves with the conveyor (door, worker and tracking carriage are static in it). Isaac is mentioned only as an option for a plant-frame version. |
| Controller | **Scripted, no learning.** IK joint solutions at waypoints (s6a, all succeed: 42/42, 129/129, 42/42), joined by joint-space linear interpolation. Sequence in a BehaviorTree.CPP v4 XML, each screw wrapped in Retry x2, then hand the screw to the worker. The PDF itself warns that the moves between waypoints are not collision checked and should be replaced by cuRobo or MoveIt. |
| Human | CarDA XSens mocap (60 fps, 23 segments) as 22 bone capsules. As-is (observed) and to-be (observed segments stitched, walking halved for AMR kitting) timelines. Unobserved time bounded by `hold` (stays at the door) and `away` (leaves). |
| Geometry | Screw points are clusters of where the worker's wrist dwelt (about plus or minus 5 cm), not CAD. The door is a box fitted to the deepest point; other points float in front of it by `depth_offset_m` (WS30 up to 0.36 m). |
| Inputs | CarDA mocap and calibrated ZED2 cameras, a time study (ILO allowance 0.16), Doosan and OnRobot catalog values (joint speeds 100/100/150/225/225/225 deg/s, assumed 1.4 rad/s2, TCP cap 1.0 m/s, rundown 2.51 s, stop time 0.326 s). |
| Outputs | KPIs per run (station and robot cycle, robot wait, SSM slowdown, min robot-worker distance, IK failures, collisions, tracking error), p95 over at least 30 randomized runs, pass criteria (station p95 at most 228 s, zero collisions, tracking error at most 1 mm, payload check). |
| Dependencies | `mujoco`, `numpy`, `opencv`, `pyyaml`. Nothing proprietary, no GPU. |
| Their baseline finding | WS10 does not work in parallel (71% of robot time inside the SSM distance, worker and robot at the same door). WS30 depends on whether the worker leaves after inserting the speaker (100% if they stay, 0% if they leave). WS20 is acceptable (10%) but has 8 mm of reach margin. |

## 3. How it relates to our work

| | Handoff | Ours |
|---|---|---|
| Simulator | MuJoCo, kinematic replay only (no dynamics, no contacts) | MuJoCo with physics and RL walking policies (`sim/mujoco_houses`), Isaac Sim 6.2 with stock policies (`sim/sunny-isaac-sim`) |
| Robots | One fixed industrial arm per station, on a track | Mobile legged robots: G1, Go1, H1, Go2, Spot, ANYmal; two G1s carrying a box; a G1 with grippers (`sam-g1-manipulation`) |
| Control | IK waypoints and a behaviour tree | Published walking policies, our planner (A*, priority yielding, station claims in `demos/fleet.py`), PD arms |
| Real-world source | Mocap and calibrated cameras of a real line, geometry fitted to wrist points (about 5 cm) | Photos and LiDAR of real sites, reconstructed into a collider (3 mm median against laser) and a 3DGRUT splat |
| People | Yes: a mocap worker is the whole point | None in any scene |
| Shared station | Robot and worker share one door; P1, P2, P3 interlocks; robot hands a failed screw to the worker | Robots take one hand-off spot in turn (`fleet.Jobs`, 2 s dwell), closest approach logged |
| Metric | ISO/TS 15066 SSM distance, takt, ROI | Arrival time, closest approach, falls, collider accuracy |

**Overlap:** MuJoCo; turning a real place into a sim scene; measuring separation between agents; a station several agents use in turn. **Differences:** fixed arm against mobile robots, scripted replay against physics with policies, people against no people, business KPIs against demo visuals. The "hand-off" in the PDF title means the document is a handover to a simulation team; inside the sim the closest thing to our hand-off is P2 (the worker hands the door over to the robot) and the BT fallback (the robot hands a screw back to the worker).

Nothing in it overlaps CooHOI (two humanoids, one box) or the GR00T and VLA plans.

## 4. What ran on Colab

### 4.1 Rebuild

`sim/handoff/carda_cell.py` rebuilds the WS30 cell. What came from where:

- **From the PDF:** pedestal 0.9 m, track, joint speed limits, 1.4 rad/s2, TCP cap 1.0 m/s, 0.06 m approach at 0.25 m/s, 0.3 s tracking sync, 2.51 s rundown, track return 0.75 m at 0.5 m/s, 0.30 m screwdriver TCP, robot start 11.0 s, the SSM formula (v_h 1.6 m/s, T_r 0.1 s, T_s 0.326 s, C 0.12 m; Z_d and Z_r are not given, so 0), P2's 1.0 m, worker capsule radii 0.05 to 0.13 m, the "door box fitted to the deepest point" rule, the hold and away bounds.
- **From public data:** M0617 joint origins and limits from Doosan's URDF (`doosan-robotics/doosan-robot`, `dsr_description/urdf/m0617.urdf`, Apache-2.0), links as capsules as in the handoff. Worker motion from CarDA on Zenodo (record 14644367, CC BY 4.0), WS30 cycle `xsens_003_WS30_2023_10_04`, which is the handoff's representative WS30 cycle "003_1004". It covers 54.4 s, matching the annotated 53 s clip (the PDF's 22% of a 4 min cycle). The archive also has an EAWS annotation sheet, but with no per-step times.
- **Ours (choices the PDF does not pin down):** screw targets are the 4 longest right-wrist dwell clusters after 11 s (dwell under 5 cm/s for 0.6 s, merged within 8 cm); the door normal is from the worker's hips to those points; the base is picked from 6 candidate spots by IK success and distance from the worker (their base coordinates are in a belt frame we do not have); the feeder tray sits 0.5 m in front of the base (J1 is limited to plus or minus 150 deg, so a tray behind the robot is unreachable); 0.5 s per screw pick at the feeder; a min-jerk shape on the trapezoid duration; the XSens world frame is used as the belt frame (XSens integrates footsteps, so a worker standing on the belt does not see its motion; the 0.36 to 0.45 m depth spread of the points suggests some drift, as in the handoff). The end of INSERT is when the worker first leaves the door, 7.0 s in this clip (the worker then fetches the screwdriver).
- `--selfcheck` covers the BVH parser (axes, units, channel order), IK to 1 mm, and the move-time formula.

Program for the 4 screws: **45.4 s** (feeder 16.1, reach 10.7, fasten 14.3, return 4.3) against the handoff's **49.0 s** (17.0, 19.0, 13.0). Reach from the shoulder 1.12 to 1.38 m (handoff: 1,286 mm needed). IK 0 failures. Depth offsets 0.01 to 0.45 m (handoff: up to 0.36 m).

### 4.2 Results against the handoff's section 7

"Moving" fractions are over the time the robot program advances; SSM violation means the capsule distance is below S_p = v_h(T_r + T_s) + v_r T_r + C. P1s is P1 with a protective stop while inside S_p (an upper bound on the slowdown; real SSM slows down first).

| Worker | Policy | Robot done at | Robot wait | Min distance | Within 0.5 m | SSM violation | Handoff (same worker case) |
|---|---|---|---|---|---|---|---|
| as-is | P1 | 56.5 s | 0 | **-0.16 m** | **61%** | **86%** | as-is: -0.15 m, 64%, 86% |
| as-is | P1s | 97.5 s | 41.0 s | 0.40 m | 0% | 0.1% | |
| as-is | P2 | 98.2 s | 41.7 s | 0.40 m | 0% | 5% | |
| as-is | P3 | 102.3 s | 45.9 s | 0.40 m | 0% | 0% | |
| hold (stays after INSERT) | P1 | 56.5 s | 0 | -0.18 m | 100% | **100%** | to-be hold: 0.11 m, 51%, 100% |
| hold | P1s, P2, P3 | not by 228 s | 217 s | 0.41 m | | | |
| away (leaves after INSERT) | any | 56.5 s | 0 | 2.5 m | 0% | **0%** | to-be away: over 3 m, 0%, 0% |

What this says:
- **The rebuild agrees with the handoff where the cases match** (as-is, and the SSM column for hold and away). The hold minimum differs because our hold freezes the worker with hands on the speaker, while theirs freezes the last pose of a stitched to-be timeline.
- **WS30 lives or dies on the worker leaving**, as they wrote: under P2 or a protective stop the robot never finishes if the worker stays, and finishes at 56 s if the worker leaves.
- **P2 is not SSM**: with the worker 1.0 m from the active target, 5% of robot motion is still inside S_p (about 0.8 m), because other arm links pass closer to the worker than the target is.
- In the as-is case the worker does the screws, so every interlock costs about 41 to 46 s of robot wait; the robot is only useful if the worker's screw work is removed (to-be).

### 4.3 Our G1 in the worker's place

`sim/handoff/g1_cell.py` drops the 23-DoF G1 from `sim/mujoco_houses` (Unitree's published `motion.pt` on the legs, our PD on the arms, attached with `MjSpec.attach` as in `coop_lift.py`) into the same cell. The G1 walks about 2.6 m to the door, turns to face it, holds its hands at the door for 4 s (the worker's INSERT), then walks off. The arm runs the same program from 11.0 s.

| Policy | G1 at the door | Robot wait | Robot program done | Closest arm to G1 | Inside S_p while moving | G1 falls |
|---|---|---|---|---|---|---|
| P2 (zone, 1.0 m) | 7.1 s | 2.5 s | about 58.9 s | 0.76 m | 0.2% | none (pelvis never below 0.76 m) |
| P1 (fixed start) | 7.1 s | 0 | 56.5 s | 0.51 m | 5.4% | none |

This is a **robot-to-robot hand-off of a work zone**: the humanoid brings the job to the station and leaves, the arm waits for the zone to clear, then works. The cost of the interlock is the 2.5 s the G1 needs to clear 1.0 m. Videos: `g1_ws30_P2.mp4`, `g1_ws30_P1.mp4`. The G1's hands only pose at the door; nothing is carried or inserted.

### 4.4 Joint-space moves hit the door

The handoff warns that its waypoints are joined by unchecked joint-space moves. The rebuild shows the effect: in every run where the arm completes, the tool passes up to 3 cm into the door box for 52 frames (1.7 s), during joint moves to and from the deepest screw. Adding a waypoint 0.25 m in front of each screw (`--standoff 0.25`, joint move there, then a straight move in and out) removes all of them (0 frames) and costs **6.2 s** (program 45.4 to 51.6 s). That is a cheap stand-in for the cuRobo or MoveIt step in their plan, and a first number for "how much longer the cycle gets" (their section 6).

## 5. What it could add to our demos

1. **People in our scenes.** None of our demos has a human. CarDA's mocap is CC BY 4.0 and `carda_cell.py` already turns any BVH into moving capsules. A worker walking through the reconstructed loading area while the Isaac fleet runs would make "robots give way" mean something.
2. **A safety number next to closest approach.** The SSM distance S_p is about 10 lines. Our fleet already logs 50 Hz trajectories (`demos/traj_check.py`), so we can score robot-robot and robot-human separation with the same formula the handoff uses, and show P1, P2, P3 style interlocks at the shared hand-off spot.
3. **A fixed arm at the station.** Our hand-off spot in `dafleet` has robots dwelling 2 s with nothing happening. An arm (M0617 from this rebuild, or the UR10e we already ran with cuMotion in `docs/evidence/isaac/ur10e_pick_place_cumotion.mp4`) working at that spot, interlocked by P2 against the arriving robots, is the MuJoCo prototype in 4.3 moved into Isaac.
4. **A behaviour tree with a human fallback.** Their BT (Retry x2, then hand the screw to the worker) maps onto our trust network: a failed step becomes a signed request for help, and the policy decides who may take it.
5. **Their KPI discipline.** p95 over 30 randomized runs with pass criteria is a stronger claim than one good video, and our runs are already scripted enough to repeat.

## 6. Can it run inside our reconstructed scenes?

**Yes in principle, not done yet.** The cell is one self-contained MJCF with no contact needs (a kinematic replay), and our reconstructions are MJCF too (`sunny-scene-recon/build_scene.py`: floor plane, wall boxes, hulls, textured mesh), so `MjSpec.attach` with a frame places it, the same way 4.3 attaches the G1. Three conditions:
- **Free floor of about 3 m by 3 m** for the door cart, pedestal and worker. The only reconstructed MuJoCo scene kept locally, `data/floor4_static3` (4.5 m by 4.1 m, 0.2 m clearance at the spawn, no open waypoints), is too cramped. ETH3D delivery_area (a real loading area, 4.7 m clearance at the spawn) fits, but only its `scene_info.json` is local; its `scene.xml` would have to be rebuilt on Colab with `run_eth3d.sh`.
- **Frame.** The handoff works in a belt frame that moves with the conveyor. In a static scan, either drop the conveyor (render in the belt frame, as here) or drive the `track` joint with the belt displacement, which is what the PDF asks for in plant-frame simulators.
- **Isaac Sim.** Doosan's URDF imports with the URDF importer; the cell needs no ground plane (fixed base), which suits our `da_world.usda` (collider only, no plane). The worker would be capsules or an animated character driven by the BVH. Not tried.

## 7. Problems and next steps

Problems:
- **The handoff's package is missing**, so this is a rebuild, not their code. Their targets, base pose and to-be worker timeline are unknown; ours come from our own clustering and search, so numbers agree in pattern and in the as-is case, not by construction.
- **Frames.** We treat XSens world as the belt frame; the 0.45 m depth spread of the screw points suggests some drift (theirs is 0.36 m, so they likely hit the same thing).
- **Simplifications:** P1s stops instead of slowing; SSM uses the TCP speed for v_r and Z_d = Z_r = 0; min-jerk shape on trapezoid durations (peak acceleration about 1.44 times the limit); feeder tray placement and its 0.5 s pick are ours; WS10 and WS20 are not rebuilt.
- **Rendering:** the G1 videos show shadow speckle on the floor (shadow map over a larger scene). Cosmetic.
- **The two mp4 files were ours**, not the handoff's; the handoff's own videos were not shared.

Next steps, cheapest first:
1. Ask the sender for `out/sim_package/` (or the repo with `scripts/export_sim_package.py`). Their `replay.py` needs only `mujoco`, `numpy`, `opencv`, `pyyaml` and runs on the Colab CPU in minutes; then compare it with this rebuild.
2. Score `dafleet_traj.csv` with the SSM formula (robots as obstacles), no new sim needed.
3. Put a CarDA worker (BVH as capsules or an animated character) into the delivery_area fleet run in Isaac.
4. Arm at the hand-off spot in Isaac with a P2 interlock (4.3 moved to Isaac); cuMotion for collision-free moves instead of the standoff waypoint.
5. If the handoff team wants it: 30 randomized runs per scenario and policy, returned in their section 10 format.

## 8. Files, commands, units

- Code (uncommitted): `sim/handoff/carda_cell.py`, `sim/handoff/g1_cell.py`, `sim/handoff/colab_job.sh`, `sim/handoff/README.md`.
- Evidence: `docs/evidence/handoff/`
  - Videos: `ws30_asis_P1.mp4` (worker as observed, robot in parallel), `ws30_asis_P2.mp4` (zone rule, robot waits for the worker), `ws30_away_P1.mp4` (worker leaves after INSERT), `g1_ws30_P2.mp4`, `g1_ws30_P1.mp4`, each with a `_sheet.jpg`. All under 2 MB.
  - `kpi/`: KPI JSON and 30 Hz trace CSV for every run (`kpi_<worker>_<policy>_<standoff>.json`), G1 runs as `g1_P1` and `g1_P2`.
  - `logs/`: Colab job logs. `ws30_cell.xml`: the WS30 scene MJCF. `carda003_*_sheet.jpg`: CarDA's two camera views of the cycle used. `handoff_pdf_ws30_figure.png`: the handoff's own WS30 render. `teammate_sim_*`: frames from the teammate's video (section 9).
- Colab: one L4 session, `handoff`, about 70 minutes including the 922 MB CarDA download; stopped. **Units: 16.79 before, 14.53 after (2.26 used).**

## 9. The teammate's MuJoCo video

Input: `~/Downloads/KakaoTalk_Video_2026-10-31-14-16-31.mp4` (60 s, 1280 by 720, 24 fps), analyzed from frames only (`docs/evidence/handoff/teammate_sim_sheet.jpg`, `teammate_sim_t8.png`, `teammate_sim_hud_timeline.png`). No code was shared, so section 9.2 is an informed guess.

### 9.1 What it shows

- A top-down view of a factory floor titled **"FLEXIBLE METALWORKS | 19 ROBOTS | 6 MOBILE LOGISTICS RESOURCES"**, subtitle "Cell work orders + contact carries". The credits line names the building ("Depot" by Rafi Abdullah and Louise Poubel, Open Robotics, CC BY 4.0, a Gazebo world model), the robots ("roqsim / Menagerie", so most models come from MuJoCo Menagerie) and "original equipment / cartons".
- **Four arm cells** with a status line each, as TASK / PHASE: `cnc_arm` (LOAD_MACHINE, then UNLOAD_MACHINE), `deburr_arm` (LOAD_CLEANING_FIXTURE, then CLEANED_OUTPUT), `assembly_arm` (INSERT_COMPONENT_A, then B), `packaging_arm` (PALLET_POSITION_1, then 2). Phases run APPROACH, GRASP, LIFT, TRANSFER, PLACE. All four read COMPLETE / DONE by about t = 19 s; the remaining 40 s the arms are idle.
- **Humanoid "contact carries"**: two humanoids holding an orange box between them, and a group of three around a table with a box. Over the 60 s they shift slightly but stay inside their marked areas.
- **Mobile resources**: a yellow quadruped (Spot-like), small dark four-wheel rovers driving the aisle, blue carts with cartons, plus racks, pallets and safety lines.
- No people, no numbers beyond the task states, no real-world data.

### 9.2 How it likely works (plain version)

- **One scene file.** The building is loaded as static meshes; each robot is a copy of a ready-made Menagerie model placed at a station; machines and cartons are simple boxes. MuJoCo loads it all as one world.
- **A loop with a to-do list per arm.** Each arm has a list of work orders. Each order is cut into phases (go above the part, close the gripper, lift, carry over, put down). For each phase the code picks a target position for the hand, works out joint angles that put the hand there (inverse kinematics), and the arm's motors drive toward those angles. When the hand gets there, the phase advances. The HUD simply prints which order and phase each arm is in. This is the same idea as the handoff's robot program and our `Program` class in `carda_cell.py`.
- **Grasping is probably simplified** (the part attached to the gripper on GRASP), which is how most scripted MuJoCo demos keep picks reliable. Not verifiable from video.
- **"Contact carries"** suggests the humanoids hold the box by pressing their hands on it, as our `coop_lift` does, rather than gluing it. Whether they walk on a learned policy is not visible; the small in-place motion suggests posed or replayed bodies.
- **Mobile robots most likely glide along scripted paths** (their base position is set each frame); Menagerie provides robot models, not walking controllers. Not verified.

### 9.3 How it relates

| | Teammate | SIM_HANDOFF | Ours |
|---|---|---|---|
| Scope | Whole factory, many robot types | One station, one arm, one worker | Several mobile robots, one shared station, real scans |
| Logic | Work orders per arm (task, phase) | BT steps per arm, interlock with a worker | Station claims, A*, yielding (`fleet.py`); trust policy over MQTT |
| Humanoids | Scripted contact carries | None | G1 walking on a published policy; two G1s carrying a box by friction (`coop_lift`); CooHOI port planned |
| Realism | Synthetic depot | Real worker motion, safety standard, takt | Real site scans, real walking policies |
| Output | Overview video | KPIs and pass criteria | Videos plus trajectory logs |

The three are pieces of one picture: the teammate has the **factory layout and cell logic**, the handoff has **people and safety numbers**, we have **real places and robots that actually walk**.

### 9.4 Expansion options, ranked

| Rank | Option | Effort | Payoff |
|---|---|---|---|
| 1 | **One job format.** Their work orders, the handoff's BT steps and our `fleet.Jobs` routes are the same idea (station, task, phases, who may enter, interlock). One small JSON schema would let cells and fleets run from one file, and each order could go through our signed requests and trust policy. | Low, about 1 day | High: the demo story becomes "signed work orders across a mixed fleet" |
| 2 | **Put a person and a safety number in their factory.** Attach `carda_cell.py`'s cell (MJCF, CarDA worker as capsules) as a fifth cell and print wait and SSM time on their HUD. | Low, half a day | High: the only human in any of the three sims |
| 3 | **Real walking for their mobile robots.** Drive their humanoids and quadrupeds with our policies (G1 `motion.pt`, Go1 ONNX) and `fleet.py`'s yielding and station claims (pure numpy, outputs velocity commands). | Medium, 1 to 2 days | High: gliding robots become walking ones; risk of falls in tight aisles |
| 4 | **Swap their carries for `coop_lift`.** Two G1s carrying by friction while walking. | Medium, about 1 day | Medium: more real, risk of dropping the box |
| 5 | **Their factory inside a real scan.** Place their cells in ETH3D delivery_area (rebuild its MuJoCo scene, or Isaac with `da_world.usda` and our fleet). | Medium to high, 2 to 4 days | High: real site, their cell logic, our robots |
| 6 | **Learned carries.** Replace scripted carries with the CooHOI port (`coohoi_port_plan.md`). | High, GPU training days | Medium until the port works |

Before any of this, it is worth asking the teammate three things: whether grasps are physical or attached, whether any robot walks on a policy, and whether the scene code can be shared.

