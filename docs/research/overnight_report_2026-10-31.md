# Morning report (overnight Isaac Sim prep, 2026-10-31)

## 1. What ran and the results
- **Isaac Sim 6.2 runs headless on a Colab L4 through pip**, so we can test without Brev (2 min install, recipe in `overnight/colab/isaac_run.sh`). Videos in `docs/evidence/overnight/`: two H1s on the stock course, and five robots (two H1s, Spot, Go2, ANYmal) walking together in the warehouse.
- **Several robots coordinating in Isaac Sim 6.2** (`demos/fleet.py`, videos in `docs/evidence/overnight/fleet/`):
  - `fleet.mp4`: four robots to four goals, no falls.
  - `job.mp4`: three robots share one loading station in turn (H1 leaves 8.5 s, Go2 15.2 s, all done by 30.0 s).
  - `plan.mp4`: A* paths on a top-down map built from the scene's meshes. All four arrive in 9 to 13 s, closest 1.06 m.
  - `plan2`: A* sends the H1 and Go2 out of their aisle, round the shelf end and into the next one; all three arrive by 21.9 s, closest 3.2 m.
  - Without avoidance, robots collide in the simple stand-in (closest 0.01 m); with it, none (closest 1.15 m), at a time cost of 6% to 87%.
  - `demos/task_plan.py`: describe the job in English, Claude assigns the stations, the stock policies walk them. Offline checks pass; the live call needs your key (a few cents), not run.
- **An H1 walked through a real reconstruction** (ETH3D delivery_area, a real loading area): 3DGRUT splat for looks, PatchMatch mesh as the invisible collider, no ground plane. A* on a map built from the collider, 7 waypoints, about 11 m, goal reached in 17.6 s, no fall. First video had the robot black and blurry floaters; with a dome light and the camera at a photo spot it looks like the photo (`dah1b.mp4`, `dah1c.mp4` in `docs/evidence/overnight/fleet/`). The critic's catch: this proves the plumbing, not collisions (the walk stayed in open floor).
- **Three robots shared a job in that real scan**: two H1s and a Go2 take one hand-off spot in turn (2 s each), then cross to the far side. All arrived (11, 18 and 22 s), closest 1.09 m, no falls in the 50 Hz trajectory log (`dafleet.mp4`, `dafleet_sheet.png`). Command in RUNSHEET step 4.
- **The real-scan collider stops robots at walls.** Five H1s sent from 1 m straight at walls and obstacles in delivery_area: none got more than 0.17 m past the face; three stopped against it, two fell on contact (the stock walking policy is blind). Video is unusable (camera inside splat floaters); the trajectory log is the evidence.
- **A TSDF collider from the same stereo depth is more accurate but has holes**: 88% of laser rays within 5 cm (Poisson mesh 79%), similar F1 (0.83 against 0.84 at 5 cm), but no surface on 30% of pixels (Poisson 3%). **They do**: the same hand-off job on the TSDF collider dropped all three robots through it in 10 to 30 s, none arrived. The Poisson mesh stays the collider here, the opposite of the warehouse, where Poisson erased the floor and TSDF kept it. The H1s fell over narrow cracks (16 and 9 cm wide under the feet), not big holes. **Filling only those cracks from the Poisson mesh fixes the H1s**: on the filled collider both finished the job with no falls, as fast as on Poisson (11.2 and 17.9 s). The Go2 still sank to a 9 cm base height and stalled 0.45 m short, as on plain TSDF (it arrived on Poisson). No crack or bump under it; cause not found yet (`dafill.mp4`, `fill_cracks.py`).
- **The planner no longer walks robots into holes.** Before, any patch with no geometry counted as walkable; on delivery_area that was 1,300 of 1,700 m² (outside the scan, or under a roof with no floor reconstructed), and the first walk passed under a metre from one. Now a cell needs collider floor under it; no real floor was lost (`docs/evidence/overnight/nav/da_support_audit.png`).
- **Splat and collider disagree by over 10 cm on 69% of pixels** in that scene, and the laser scan says the collider is right: where they disagree, the collider is within 5 cm of the laser on 79% of pixels, the splat on 0.3%. Median error: collider 3 mm, splat 54 cm too deep. So the splat is for looks only; robots touch the stereo mesh.
- **3DGRUT on ETH3D delivery_area** (30k steps, 48 min on the L4): looks good (held-out PSNR 23.0, vid2scene got 19.9), geometry poor (F1 0.16 at 5 cm against the laser scan, PatchMatch 0.88). **Splat for looks, stereo depth for the collider.** The NuRec USDZ with a collider packed in works (168 MB).
- **Grading a reconstruction the way a robot would** (`score_robot.py` on NVIDIA's warehouse, our own walkthrough capture, true geometry known): floor error, floor holes, walkable area wrongly free or wrongly blocked, obstacles found. Perfect depth scores 0.9999 IoU, so the scorer is trustworthy.
  - **The mesher mattered more than the reconstruction.** Poisson loses 94% of the floor even from perfect depth; a TSDF keeps 96% with a 3 mm floor error. Switched to TSDF.
  - **The splat's depth is wrong only on the floor** (2.1 times too deep, likely the shiny floor's reflections). Walls, racks and boxes are right (median 1.01).
  - **Fix: clip the splat depth at the floor height found from SfM points.** Floor error 201 mm to 1.3 mm, walkable area 0% to 66% IoU, 93% of obstacles found, almost nothing wrongly open.
  - Dropping small floating mesh pieces raises walkable area to 96%, but it also deletes real obstacles (2 of 14 gone), so the collider stays unfiltered.
- **Isaac Lab 3.0 two-agent training (MAPPO) runs on a Colab L4** with no Isaac Sim (Newton physics): success 0% to 47% in 21 min on NVIDIA's two-hand handover. This is the training path for a CooHOI port (`docs/research/coohoi_port_plan.md`: same humanoid, same 30 Hz, one shared policy, so first try loading `TwoAgent.pth` with no training).

## 2. What failed and why
- **PatchMatch on the warehouse (decides the collider) crashed twice**: no output folder, then no neighbour images, because our exact-pose model has no matched points. `colab/pm_sources.py` now picks neighbours from the poses. Rerunning, about 20 s per image, results around 14:45 UTC.
- **Robots on a collider made from the splat's own depth fall straight through** (holes under the spawn). Stereo depth or the floor clip is required.
- **3DGRUT's NuRec export writes a broken transform** (the translation lands in the projective part, so the splat would be distorted in Isaac Sim). Patched locally; still unfixed upstream. Issue draft: `docs/research/grut_nurec_transform_issue.md`.
- **Floor guard**: v1 holed 82% of the shiny floor. v2 (hole only where SfM sees points well below the floor) costs almost nothing here (walkable IoU 0.64 against 0.66) but misses its bar by a hair, so it stays an option for sites with pits.
- **CooHOI still shows one humanoid.** The two-camera depth merge is unit tested but needs Brev (Isaac Gym).
- **2DGS (the one-model-for-looks-and-collision challenger) collapsed twice at the same step** (3,200 to 3,300), with the distortion-loss weight at gsplat's default and at 2.6 times lower, so the weight is not the cause. Step 3,000 is also gsplat's first opacity reset. The control with no distortion loss collapsed too, so the reset is the trigger. **With the reset off it trains stably** (L1 0.018 at step 7,400, the collapsed runs sat at 0.3 to 0.44). Depth score around 16:00 UTC.
- Colab setup traps, all fixed in the scripts: Python 3.12 venv, a Vulkan driver file, `MPLBACKEND=Agg`, a missing driver folder, Open3D 0.20's TSDF returning empty meshes (pinned 0.19), gsplat's 2DGS backward needing a one-line patch.

## 3. Research findings
- `docs/research/coordination_methods_2026-10-31.md`: nothing released combines a VLM planner, RL policies and several robots in Isaac Lab. Best path: stock 6.2 walking policies plus our planner. Next: GR00T N1.6 on a G1 in Isaac Lab-Arena (alpha, non-commercial weights, one G1 per env).
- `docs/research/factory_reconstruction_2026-10-31.md`: 3DGRUT for the photoreal USDZ, PatchMatch for collision. Only 3DGRUT, gsplat and nerfstudio allow commercial use.
- Critic reviews in `docs/research/critic_*.md` (reconstruction plan, coordination, 3DGRUT result, scoring, TSDF collider, floor clip, hybrid walk). Each one's fixes were applied before the runs.

## 4. Tomorrow's first commands on Brev
In order, from `sim/isaac-sim/RUNSHEET.md`:
1. CooHOI with both humanoids: `bash run_coohoi.sh 20`
2. Multi-robot scene, streamed: `multi_robot_scene.py --stream`, then the fleet modes (goals, `--plan`, the station job, the aisle layout)
3. Warehouse walkthrough capture
4. Robots in our reconstruction: `multi_robot_scene.py --scene walk_pm_world.usda --plan` and `--scene da_world.usda --plan` (if the Colab runs finish)
5. GR00T N1.6 driving a G1 in Isaac Lab-Arena
6. Isaac Lab 3.0 two-agent MAPPO training

## 5. Colab units
34.01 at start, 18.73 at 14:27 UTC. Session `reconB` (L4) is still running: warehouse PatchMatch (141 of 240 depth maps), the 2DGS rerun, then two robot walks on our own warehouse colliders. Hard stop at 16:30 UTC, about 14.7 left.
