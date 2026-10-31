# Simulation team handoff v2: cobot deployment scenario for the CarDA car-door assembly line

Author: industrial engineering team · Data: CarDA (Stellantis Melfi experimental line, CC BY 4.0) · Basis: pipeline results in `out/`

> **Changes from v1**: v1 only stated conclusions, which was not enough to start simulating. It was missing the door geometry, the worker's to-be motion, the frame conventions, how the human and robot share space, and a runnable scene.
> v2 consists of **a package you can open and run directly** (`out/sim_package/`) and this document. The baseline replay already shows the main issue in this scenario (spatial conflict between robot and worker).

---

## 0. Questions the simulation must answer

1. **Does the recommended setup (3 Doosan M0617, 1 per station, on linear tracks) actually run while meeting the 240 s takt (228 s cap)?**
2. **How should the robot and worker share the same door** so that the safety requirement (ISO/TS 15066 SSM) is met and the cycle holds? (P1 parallel / P2 zone exclusion / P3 sequential)
3. Do the industrial engineering conclusions (optimal robot count, ROI) still hold once the added wait and slowdown time is included? When you return results, s5→s8 are recomputed automatically.

---

## 1. Package contents and how to run

```bash
python scripts/export_sim_package.py        # rebuild the package from out/ results
cd out/sim_package
python replay.py WS30 --video               # baseline replay: KPIs, trace CSV, preview image, mp4
python replay.py WS30 --human asis          # replay with the observed (current) worker motion
python replay.py WS30 --after away          # assume the worker leaves the station after the motion data ends
```
Required packages: `mujoco`, `numpy`. Video output needs `opencv`, and reading scenario parameters needs `pyyaml`.

| File (per station) | Contents |
|---|---|
| `scene.xml` | MuJoCo MJCF scene (belt frame, see section 2)<br>Belt, cart and door panel (**carrier body**), target sites (red), linear track rail, **carriage** (slide joint `track`), pedestal, screw and part supply trays (green), **M0617** (manufacturer ROS kinematics, links as capsules), tool capsule, 22 worker bone capsules (mocap), markers for the work zone and the line-side zone |
| `targets.json` | All robot targets: `id`, `operation` (screw / place / pick_screw / pick_part), `order`, `pos_m`, `approach_axis`, `approach_offset_m`, `tolerance_mm`, `depth_offset_m`, observed dwell time |
| `robot_reference.json` | Robot program timeline (s6a) and **IK joint angles per step** (`q_end_rad`)<br>Joint names, base position and yaw, pedestal height, tool, joint velocity and acceleration limits, TCP speed cap, robot start time relative to the worker, BT XML |
| `human_asis.npz` | Observed worker trajectories for 23 joints (m, z-up, 60 fps, belt frame) |
| `human_tobe.npz` / `human_tobe_timeline.json` | **Worker motion after automation**<br>Observed segments of the work elements that stay with the worker, concatenated in order (0.3 s blending). WALK is time-compressed ×0.5, assuming AMR kitting |
| `motion.json` | Frame conventions, conveyor speed and direction, takt and cap, station cycle (s5), representative cycle, belt width, tracking range, notes on the geometry approximation |
| `kpi_tobe.json`, `kpi_tobe_away.json`, `kpi_asis.json`, `trace_*.csv`, `preview.png`, `replay_tobe.mp4` | Baseline replay results (section 7) |
| Shared: `scenario_matrix.yaml`, `replay.py`, `index.json` | Scenarios, policies, randomization, SSM parameters, KPIs, acceptance criteria, return format |

---

## 2. Frame and time conventions (align these first)

- **Plant frame** (s0): floor at z=0, **+x = line flow direction (WS10→WS30)**, +y = toward the IN camera, units in m
- **Belt frame**: equal to the plant frame at the cycle reference time `t_ref`. **The door (carrier), the worker (standing on the belt), and the robot carriage while tracking all move with the belt**, so they are stationary relative to each other in this frame. All coordinates in the package are in this frame.
  - Conversion to plant coordinates: `x_world(t) = x_belt + v·d·(t − t_ref)`, v ≈ 24 mm/s (WS10 24.7, WS20 24.6, WS30 23.9), d = +x
- **Cycle clock**: t=0 is when the door enters the station. The robot starts at `robot_start_after_human_s` (WS10 0 s, WS20 4.5 s, WS30 11.0 s). This is an interlock set by s5. For example, at WS30 the robot starts after the worker inserts the speaker.
- **Track**: during the robot program the carriage is synchronized with the belt (stationary in the belt frame). When the program ends, it returns at 0.5 m/s. The return time is included in the s6a results. In a plant-frame simulator (Isaac etc.), drive the `track` slide joint directly with the belt displacement.

---

## 3. Process and human/robot split per station

Times are **standard times** per work element (observed × rating × (1 + ILO allowance 0.16)), summed over one cycle. Sources: `out/s2/time_study.json`, `out/s5/allocation.json`.

### WS10: front door glass, weatherstrip, rear vertical cover
| Assigned to | Work elements (s/cycle) |
|---|---|
| **Robot M0617** | GET_TOOL 16.7 · REACH_FASTENER 17.9 · FASTEN 13.0 → **4 screws, 47.6 s** |
| Worker | INSERT (glass and weatherstrip, flexible parts) 23.2 · WALK 42.2→21.1 (kitting) · GET_PART 3.0 · SETUP 1.9 · INSPECT 0.7 · **unobserved 145.8** |
| Station cycle (s5) | 201.7 s = human 195.7 + handover 6 (in parallel with robot 47.6) |

### WS20: side mirror, tie rod, plugs, carrier (highest robot load)
| Assigned to | Work elements |
|---|---|
| **Robot M0617** | With gripper: GET_PART 17.1 · INSERT 14.8 (**3 placements**) → tool changer (5 s) → with screwdriver: GET_TOOL 38.2 · REACH_FASTENER 40.9 · FASTEN 29.3 (**9 screws**) → tool changer (5 s) = **150.2 s** |
| Worker | SETUP 4.5 · WALK 58.5→29.2 · unobserved 10.3 → **44.0 s** (utilization 18%: room to consider sharing this worker with another station) |
| Station cycle | 160.2 s (**robot is the bottleneck**) |

### WS30: speaker
| Assigned to | Work elements |
|---|---|
| **Robot M0617** | GET_TOOL 17.0 · REACH_FASTENER 19.0 · FASTEN 13.0 → **4 screws, 49.0 s** |
| Worker | GET_PART (speaker) 2.2 · INSERT (carrier slot) 8.8 · WALK 13.5→6.8 · **unobserved 187.4** |
| Station cycle | 211.1 s (human is the bottleneck). Robot starts after the worker's INSERT ends (11.0 s) |

> **Unobserved block**: the CarDA clips cover only part of each 4-minute cycle (WS10 39%, WS30 22%, WS20 about 100%). **The data does not say where the worker is during the unobserved time.** The replay therefore bounds it two ways. `hold` assumes the worker stays at the door in the last pose (upper bound on conflict). `away` assumes the worker leaves the station (lower bound on conflict).

---

## 4. Cell specification

| Item | WS10 | WS20 | WS30 | Source |
|---|---|---|---|---|
| Robot | Doosan M0617 (6 kg, 1,700 mm, ±0.1 mm, 34 kg) | Same | Same | s4 catalog |
| Base (belt frame, m) | (3.592, 1.531), yaw 1.678 rad | (10.169, 4.273), yaw −1.671 | (16.302, 4.255), yaw −1.748 | s4 inverse reachability search |
| Pedestal height | 1.0 m | 1.0 m | 0.9 m | s4 |
| Mounting | Linear track, tracking range 0.78 m | Track, pick-and-place 0.46 m / fastening 1.71 m | Track, 0.75 m | s4 (floor mounting would need a reach of 1,359 / 1,656 / 1,402 mm, so a track is required) |
| Required reach / effective reach | 1,209 / 1,530 mm | Pick-and-place **1,522** / 1,530 (margin 8 mm) · fastening 1,385 | 1,286 / 1,530 | s4 |
| Tool | OnRobot Screwdriver 2.5 kg, TCP 0.30 m | Robotiq 2F-85 0.925 kg, TCP 0.16 m ↔ Screwdriver (tool changer) | Screwdriver | s4 `tool` |
| Fastening | Rundown 2.51 s (340 rpm × 12.5 turns + torque-up 0.3 s), 0.15–5 Nm, M1.6–M6 | Same | Same | OnRobot published values |
| Joint speed limits | J1–J6: 100/100/150/225/225/225 °/s | Same | Same | Doosan datasheet |
| Joint acceleration | 1.4 rad/s² (**assumption**: not published by the manufacturer; URScript default) | Same | Same | s4 process.py |
| TCP speed cap / linear approach speed | 1.0 m/s / 0.25 m/s, approach offset 0.06 m | Same | Same | s4 |
| Tracking sync time | 0.3 s (Doosan `tracking_conveyor` default) | Same | Same | Doosan manual |
| Stopping time | 0.326 s (Cat 0, J2, maximum conditions) | Same | Same | Doosan manual |
| ISO 13855 separation distance S = K·T + C | 1,714 mm | 1,714 mm | 1,714 mm | s4 `cell_space` |
| Cell area (incl. 10% aisle allowance) | 8.5 m² | 7.3 / 9.1 m² | 8.5 m² | s4 |

---

## 5. Targets and geometry: know the accuracy limits

- **Screw and placement positions** do not come from CAD. They are **clusters of positions where the worker's wrist dwelled (motion capture)** (representative cycles: WS10 003_0921, WS20 004_1003, WS30 003_1004). Position accuracy is roughly ±5 cm.
- **The door panel and cart are boxes fitted to those points.** The inner door trim is not flat, so the box face is aligned with the **deepest target**. The other targets float in front of the face, and that distance is `depth_offset_m` (maximum: WS10 0.14 m, WS30 0.36 m, **WS20 0.74 m**). The WS20 points in particular are spread over several parts (mirror, carrier, plugs), so a single box does not represent them well.
- Use this geometry **to check reach, cycle time, and spatial interference with the worker**. **The actual cell design requires the door CAD and the fastener position table.** We have also requested them in the RFQ (`out/s8/rfq/`).
- Cart and door position variation between car models is not in the data, so it is randomized at ±10 mm, ±1° (`scenario_matrix.yaml`). In practice a vision-based position correction step may be needed.

---

## 6. Robot program reference

- `robot_reference.json → waypoints`: each step has `op` (move_j / move_l / screw_pick / rundown / grip / track_start·stop / tool_change / track_return), `target`, `t_start_s`, `dur_s`, `code` (work element), `q_end_rad` (IK solution), and `ik_ok`. IK succeeded for every step: WS10 42/42, WS20 129/129, WS30 42/42.
- The action sequence is in the BT XML (`bt_xml`, BehaviorTree.CPP v4). Each screw is wrapped in `Retry×2 → hand over to worker on failure`.
- **Caution**: s6a checked collisions only at waypoints. Linear joint-space interpolation between waypoints **is not path planning.** It has to be replaced with collision-free paths (cuRobo/MoveIt or similar), and the cycle may grow as a result.

---

## 7. Baseline replay results (the main issue in this scenario)

The `replay.py` baseline connects the robot waypoints with joint-space interpolation and replays the worker motion alongside. Minimum human-robot distance is computed as capsule-to-capsule distance (negative = overlap).

| Station | Worker motion | Min distance while robot is moving | Time share within 0.5 m | SSM violation time share* |
|---|---|---|---|---|
| WS10 | to-be (`hold` and `away` identical) | **−0.17 m (overlap)** | 49% | **71%** |
| WS10 | as-is | −0.17 m | 32% | 52% |
| WS20 | to-be | −0.18 m (roughly the first 27 s) | 7% | 10% |
| WS20 | as-is | −0.17 m | 56% | 75% |
| WS30 | to-be, `hold` (stays at the door) | 0.11 m | 51% | **100%** |
| WS30 | to-be, `away` (leaves after INSERT) | 3 m or more | 0% | **0%** |
| WS30 | as-is | −0.15 m | 64% | 86% |

\* Share of time in which the ISO/TS 15066 protective separation distance S_p = v_h(T_r + T_s) + v_r·T_r + C + Z_d + Z_r (v_h 1.6 m/s, T_r 0.1 s, T_s 0.326 s, C 0.12 m) is not met, so the robot has to slow down or stop.

**Interpretation**
- **At WS10, parallel operation (P1) does not work with the current layout and sequence.** While the robot drives screws, the worker is inserting glass at the same door (scene video `WS10/replay_tobe.mp4`). This needs either a different task sequence (worker inserts the glass → robot fastens while the worker goes to get the next part) or the P2 policy.
- **WS30 depends entirely on whether the worker leaves the door after inserting the speaker.** If the worker leaves, interference is 0. If the worker stays, the robot has to slow down for its entire motion time. What the worker does during the 187 s unobserved block has to be checked on the shop floor.
- **WS20 is in relatively good shape.** The worker has little to do, so they overlap only in the first 27 s. However, its reach margin is the smallest, at 8 mm.
- This is the **largest uncertainty** in the s5 model, which assumes "the human and robot simply work in parallel (cycle = max + handover)". Once the simulation gives wait and slowdown times, we feed them into s5 and recompute.

---

## 8. Scenario matrix (`scenario_matrix.yaml`)

**Scenarios**
| ID | Description |
|---|---|
| **A** Recommended | 3 M0617, 3 tracks, WS20 tool changer, AMR kitting |
| **B** Minimum cost | 1 robot at WS20 only. WS30 sits right at takt at 239.9 s (28% over takt in DES) |
| **C** Platform-mounted | Robot rides on the belt platform next to the worker → no track or tracking needed, required reach 625 to 811 mm<br>Candidates: UR10e, UR15, CRX-10iA(/L), M1013, M0617, TM12S, HC10DTP |
| **D** 1 robot for 2 stations | Per the machine-cluster calculation, 1 robot can cover WS10+WS20 or WS20+WS30 (busy 196 to 199 s of 240 s) |

**Human-robot space sharing policies**
- **P1** Parallel + SSM
- **P2** Zone exclusion: the robot moves only when the worker is at least 1.0 m from the active target
- **P3** Sequential (worst case)

**Randomization** (at least 30 runs per scenario × policy)
- Conveyor speed ±5%
- Door position ±10 mm, ±1°
- Worker work-element times: lognormal (s2 mean and standard deviation, CV capped at 0.5)
- Joint acceleration 1.0 to 2.0 rad/s²
- Rundown 2.0 to 3.5 s

---

## 9. KPIs and acceptance criteria

**KPIs** (record per run; summarize per scenario as mean, p95, and worst value)
- station_cycle_s, robot_cycle_s, **robot_wait_s**, **ssm_slowdown_s**
- min_robot_worker_distance_m
- IK failure count, collision count (door, cart, pedestal, trays)
- TCP error while tracking (mm)
- Takt overrun rate

**Acceptance criteria** (scenario A must meet all of them under its best policy)
- Station cycle p95 ≤ 228 s
- Robot cycle p95 ≤ WS10 222 s / WS20 218 s / WS30 222 s
- 0 collisions, 0 IK failures
- Tracking TCP error ≤ 1.0 mm
- Payload: (tool + part) × 1.25 ≤ 6 kg

---

## 10. What to send back

1. Fill in `out/s6a/robot_task_times.json` following `contracts/robot_task_times.schema.json`.
   - Required: `model`, `group_id`, `station`, `cycle_time_s`, `breakdown`, **`breakdown_by_code`**, `ik_success_rate`, `collision_free_rate`, `sim`
   - Then run the two commands below. Allocation optimization, line simulation, ROI, and the report are recomputed automatically.
     - `python scripts/run_all.py --stage s6a s5 s6b s7 s9 s8`
     - `python tests/validate.py`
2. A table per (scenario, policy, station): station_cycle_s p50/p95, robot_wait_s, ssm_slowdown_s. We use it to calibrate the s5 handover and wait model.
3. Demo videos: one per station, showing the recommended setup with its best policy

---

## 11. Suggested order of work

1. Reproduce the scenes and numbers with `replay.py WS10/WS20/WS30 --video`, then move to the plant frame and implement track synchronization.
2. Replace the paths between waypoints with collision-free path planning (cuRobo/MoveIt) and measure the change in cycle time.
3. **Redesign the WS10 task sequence**: find a sequence where the glass INSERT and robot fastening do not overlap, and compare P1, P2, P3.
4. **Gain reach margin at WS20**: try moving the base 0.1 to 0.2 m closer or changing the pedestal height. If that does not work, look at scenario C.
5. Get KPI distributions from at least 30 randomized runs and check them against the acceptance criteria.
6. Compare scenarios B, C, D and return the results in the section 10 format.

---

## 12. Assumptions and limitations (in order of replacement priority)

| Item | Current value | Real data to replace it |
|---|---|---|
| Door geometry and fastener positions | Box fitted to motion-capture contact points (±5 cm) | Door CAD, fastener position table |
| Worker position during unobserved time | Unknown (bounded by `hold` / `away`) | On-site observation, full-cycle video |
| To-be worker motion | Concatenated observed segments, WALK ×0.5 | Measurements after kitting is in place |
| Joint acceleration | 1.4 rad/s² | Actual Doosan value / DRL default profile |
| Tool TCP offset | Estimated from drawing scale (0.30 m / 0.16 m) | Manufacturer CAD |
| Paths between waypoints | Linear joint-space interpolation (no collision check) | Collision-free path planning |
| PFL slowdown, payload dynamics | Not modeled | Manufacturer controller simulation |
| Human model | 22 capsules (radius 0.05 to 0.13 m) | SMPL mesh if needed |
| Tool changer time | 5 s per change (OnRobot published) | Spec of the selected changer |
