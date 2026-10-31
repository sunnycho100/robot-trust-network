# 3D reconstruction to MuJoCo

Turn a scanned real space (a factory floor, a construction site) into a MuJoCo scene that the Unitree G1 can walk in. The visual mesh comes from the scan. Collisions use simple shapes fitted to it, because a raw scan mesh makes a bad collision surface.

```
images (+ depth, LiDAR or known poses)
  -> camera poses           lidar_poses.py, or known poses, or global SfM
  -> dense surface          reconstruct.py (RGB-D TSDF), or COLMAP PatchMatch on Colab
  -> MuJoCo scene           build_scene.py: floor plane, wall boxes, CoACD hulls for clutter, textured visual mesh
  -> checks                 check_collisions.py (proxies vs mesh), walk_in_scene.py (G1 walk test vs a bare floor)
```

| File | What it does |
|---|---|
| `extract_bag.py` | ROS bag to a frames folder (color, depth, intrinsics) |
| `lidar_poses.py` | LiDAR ICP odometry and the camera to LiDAR extrinsic, for when camera odometry drifts |
| `reconstruct.py` | RGB-D TSDF fusion, with known poses or chained RGB-D odometry |
| `build_scene.py` | Levels the mesh, fits collision proxies, writes `scene.xml` and `scene_info.json` |
| `check_collisions.py` | Wall boxes and heightfield vs the scan mesh, surface IoU of the proxies at 0.5 m and 1.0 m above the ground |
| `walk_in_scene.py` | G1 walking policy to reachable open and near-obstacle goals, the same goals on a bare plane, and walking into every wall |
| `export_scene.py` | Isaac-ready export (URDF, meshes, terrain OBJ) and a CooHOI two-humanoid box-carry layout on free scanned ground |
| `test_outdoor.py` | Self-check of the outdoor path on a synthetic site (slope, curb, crate, canopy) |
| `colab/` | Remote GPU runs: `setup.sh`; `run_eth3d.sh` and `mesh_ab.sh` (ETH3D DSLR); `run_rig.sh` (ETH3D rig video); `compare_splat.sh` (vid2scene comparison); `run_outdoor.sh` (outdoor scene, export, preview); `run_hilti.sh` (Hilti 2022 construction site); `pull_results.sh` (copies results to the Mac during a run) |

## Running on Colab

Heavy steps run on a Colab L4, not locally. With the Colab CLI connected:

```bash
colab upload -s recon colab/setup.sh /content/setup.sh
colab exec -s recon   # then: bash /content/setup.sh pipes delivery_area
bash run_eth3d.sh delivery_area   # dense stereo, mesh, ETH3D score, MuJoCo scene, checks, walk test
bash compare_splat.sh             # vid2scene method vs ours on the same poses and views
```

The code is expected at `/content/robosim` (this repo's `sim/sunny-scene-recon` and the G1 files from `sim/mujoco_houses`).

The VM's disk is gone when the session ends, so keep `bash colab/pull_results.sh recon` running on the Mac during a run. Every 10 minutes it copies the run logs, scores and reports into `data/colab_results`.

## Results so far

All scores are against a laser scan (ETH3D evaluator), F1 at 2 / 5 cm. Meshes are sampled at 1 cm for scoring.

**Tailored vs realistic capture**, same scene (ETH3D `delivery_area`), same pipeline, the dataset's own poses:

| Capture | Points | Mesh |
|---|---|---|
| Tripod DSLR, 44 photos, 3200 px | 0.78 / 0.88 | 0.74 / 0.87 (Delaunay) |
| Handheld 4-camera rig video, 948 grayscale frames | 0.38 / 0.54 | 0.37 / 0.50 (Poisson) |

**Meshers** on the DSLR points: Open3D Poisson 0.71 / 0.84, COLMAP Poisson (depth 12) 0.72 / 0.83, COLMAP Delaunay 0.74 / 0.87.

**vid2scene vs ours**, same 1920 px images, same GLOMAP-SIFT poses (44/44 registered, 2.3 cm median position error), same 38 training views, 6 held-out views:

| | PSNR | LPIPS | Geometry | MuJoCo scene |
|---|---|---|---|---|
| vid2scene splats | 19.9 | 0.32 | n/a | none |
| Splat depth, meshed like ours | 11.2 | 0.70 | 0.07 / 0.16 | no floor found |
| Ours (PatchMatch) | 13.7 | 0.63 | 0.30 / 0.56 | ground 1.6 cm, 0 falls |

Splats win on looks and lose on geometry, so collisions come from the MVS mesh and splats are only a visual layer.

**NVIDIA 3DGRUT (3DGUT, MCMC, 1M Gaussians, 30k steps)** on the DSLR photos at 2069 px with ETH3D's poses, every 8th photo held out, 48 min on a Colab L4: PSNR 23.0, SSIM 0.89, LPIPS 0.36. Its depth (`colab/grut_points.py`, multi-view consistency like ours) scores 0.07 / 0.16 as points and 0.06 / 0.15 as a mesh, against 0.78 / 0.88 for PatchMatch on the same poses. The fused points sit a median 12 cm from the scan, so it is spread error, not an offset. Same verdict: 3DGRUT gives the Isaac Sim USDZ, PatchMatch gives the collider (`colab/grut_collider.py` puts both in one frame). In that hybrid the two disagree by over 10 cm on 69% of opaque pixels (`colab/splat_vs_collider.py`, 38 training views, camera centres match the stereo poses to under 1 mm), mostly with the collider nearer. The collider is the one on real geometry: 82% of those collider hits lie within 5 cm of a PatchMatch point against 2% of the splat's surface points. So a robot collides where the walls really are, but the splat's own depth sits behind them, which can show up as wrong occlusion where a robot meets the splat (not checked in Isaac).

**Synthetic warehouse, scored the way a robot needs it** (`colab/score_robot.py`, NVIDIA's full warehouse rendered along a 240-frame walkthrough by `isaac-sim/demos/capture_walkthrough.py`, 5 cm grid, ground truth from the training frames' depth). Free IoU is walkable floor that the collider also leaves walkable; false free is blocked ground the collider leaves open (collision danger). The TSDF ignores depth beyond 12 m, which adds some holes:

| Depth source | Mesher | Floor error median / p95 | Floor holes | Free IoU | False free | Obstacle recall |
|---|---|---|---|---|---|---|
| Ground truth | none (points) | 0.0 / 0.1 mm | 0% | 1.00 | 0.000 | 1.00 |
| Ground truth | Poisson (`eth3d_prep.py`) | 2.7 / 186 mm | 94% | 0.00 | 0.000 | 0.96 |
| Ground truth | TSDF 2 cm (`colab/tsdf_mesh.py`) | 1.1 / 2.9 mm | 5% | 0.96 | 0.001 | 0.93 |
| 3DGUT render | TSDF 2 cm | 201 / 287 mm | 92% | 0.00 | 0.000 | 0.93 |
| 3DGUT render clipped at the SfM floor (`colab/floor_clip.py`) | TSDF 2 cm | 1.3 / 2.6 mm | 5% | 0.66 | 0.001 | 0.93 |
| Same, pit guard (`floor_clip.py --guard`) | TSDF 2 cm | 3.1 / 16 mm | 4% | 0.64 | 0.001 | 0.93 |
| Same, pieces under 2000 triangles dropped (`tsdf_mesh.py ... 0.02 2000`) | TSDF 2 cm | 1.3 / 2.6 mm | 6% | 0.96 | 0.009 | 0.85 |
| Same, ground-truth-free piece rule (`colab/collider_audit.py`) | TSDF 2 cm | | 5% | 0.76 | 0.002 | 0.88 |

Poisson as configured in `eth3d_prep.py` (depth 10, 12% density trim, cluster filter) loses 94% of the floor cells, so nothing is left walkable. It removes the floor rather than covering it: casting down from 1.85 m over each true floor cell (`score_robot.py lid`), 93% of rays hit nothing, 6% hit something 0.1 to 1.8 m up and 1% hit the floor (TSDF: 89% floor, 6% nothing, 5% shelves above the floor). TSDF carves the space the cameras saw, so colliders from depth maps use TSDF. The 3DGUT splat (held-out PSNR 27.7) renders the floor far behind its true depth, so its depth cannot make a collider even through TSDF. Split by pixel (`colab/depth_ratio.py`, 210 views paired by pose, about 98M floor and 95M other pixels): rendered over true depth on walls, racks and boxes has median 1.01 (p10 0.87, p90 1.52), but on the floor median 2.1 (p10 1.10, p90 5.8), and only 2% of floor pixels are within 5 cm. The floor error fits reflections of the glossy floor rendered as geometry below it (likely, not tested). Clipping every splat pixel that lands under the floor onto the floor plane, with the floor height taken from the SfM points only (-0.003 m, truth 0.005 m), fixes the floor; the remaining gap was space wrongly blocked (false occupied 0.34). Dropping connected mesh pieces under 2000 triangles (floaters; obstacles standing on the floor stay joined to it) brings free IoU to 0.96, level with ground-truth depth, at the cost of obstacle recall (0.93 to 0.85; 20000 triangles: 0.79). The threshold was picked from two tries scored against ground truth, so check it on a second scene before trusting 2000. **The audit says do not trust it** (`colab/collider_audit.py`, critic_floor_clip.md actions 1 to 3 and 6): the 2000 filter deletes whole obstacles (5 of the 14 obstacles over 0.1 m2 fall under 50% recall, two of them to 0) and opens 5 wrongly free blobs over 0.1 m2 (largest 0.59 m2), while the unfiltered clip has none over 0.04 m2 and no obstacle under 50%. A ground-truth-free rule (keep a piece that reaches the floor or exceeds 1 m2, drop the rest only if fewer than 3 cameras see it or it is within 1 m of a camera) keeps false free at 0.002 but still drops 4 obstacles entirely (boxes on shelves next to the camera path) and only reaches free IoU 0.76. Connected pieces cannot tell a floater from a box on a shelf, so the splat-only collider stays unfiltered (safe, but 34% of walkable floor wrongly blocked) and PatchMatch remains the collider. The clip invents flat floor over any pit, so `--guard` leaves a hole instead wherever SfM sees a point over 10 cm below the floor with none on it (a first version that required SfM floor support holed 82% of this glossy floor). The warehouse has no pits, so its 54 guard cells are reflections; the guard costs 0.013 free IoU and a 16 mm floor p95 here, just over the 0.01 bar, so it stays opt-in. Open3D 0.20's legacy TSDF returns empty meshes; pin 0.19.

**Per-pixel depth** against ETH3D's laser depth maps (`colab/depth_check.py`): on the training photos, ours has 66% of laser pixels within 5 cm (median error 3.5 cm), and 49% (3.8 cm) on the held-out ones. vid2scene's rendered splat depth has 6% within 5 cm and a median error of 1.5 m even on the photos it was trained on, mostly placing surfaces behind the real ones.

**Collision checks** (`check_collisions.py`), outdoor path on `delivery_area`: heightfield vs mesh ground 0.9 cm (95th percentile). Against the laser scan's own ground, where the camera saw the ground (278 m2): median 0.25 cm, 95th percentile 1.0 cm. Ground the mesh filled in (40 m2): median 0.4 cm, 95th percentile 64 cm, 5 m2 off by over 20 cm. Under the roof, where the laser has no ground either, our heightfield sits on the ceiling in about 100 m2, which these scores leave out.

Object boxes on a 10 cm grid vs the mesh, both ways, within 5 / 10 cm:

| Height above ground | Mesh covered by boxes | Box faces on the mesh | Surface IoU |
|---|---|---|---|
| 0.5 m | 0.75 / 0.95 | 0.55 / 0.77 | 0.50 |
| 1.0 m | 0.67 / 0.88 | 0.54 / 0.77 | 0.48 |

G1 walk test: 20 of 20 goals reached, 0 falls, same as on a bare plane. Walking straight at a tall object from 0.5 m, 10 runs: 9 touched it, 0 passed through.

## Choices so far

- Indoors: a plane for the floor, boxes for walls and convex hulls for clutter. No heightfield indoors, since it erases thin walls and can't represent overhangs.
- Outdoors and on sites (`--outdoor`): a heightfield from cloth-simulation ground points (curbs, slopes), and everything standing on it made solid on a 5 cm grid and covered by merged boxes on a 10 cm grid (`--box-cell`). Overhangs keep the gap below them. CoACD was dropped here because its error bound scales with the part, so a 50 m facade got meters of slack.
- LiDAR poses when available. Camera-only odometry on the HCIC construction bag drifted into doubled walls.
- ETH3D for scoring, because it has a laser-scan ground truth.
