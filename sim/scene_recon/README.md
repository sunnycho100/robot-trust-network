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
| `colab/` | Remote GPU runs: `setup.sh`; `run_eth3d.sh` and `mesh_ab.sh` (ETH3D DSLR); `run_rig.sh` (ETH3D rig video); `compare_splat.sh` (vid2scene comparison); `run_outdoor.sh` (outdoor scene, export, preview); `run_hilti.sh` (Hilti 2022 construction site) |

## Running on Colab

Heavy steps run on a Colab L4, not locally. With the Colab CLI connected:

```bash
colab upload -s recon colab/setup.sh /content/setup.sh
colab exec -s recon   # then: bash /content/setup.sh pipes delivery_area
bash run_eth3d.sh delivery_area   # dense stereo, mesh, ETH3D score, MuJoCo scene, checks, walk test
bash compare_splat.sh             # vid2scene method vs ours on the same poses and views
```

The code is expected at `/content/robosim` (this repo's `sim/scene_recon` and the G1 files from `sim/mujoco_houses`).

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

**Per-pixel depth** against ETH3D's laser depth maps (`colab/depth_check.py`): on the training photos, ours has 66% of laser pixels within 5 cm (median error 3.5 cm), and 49% (3.8 cm) on the held-out ones. vid2scene's rendered splat depth has 6% within 5 cm and a median error of 1.5 m even on the photos it was trained on, mostly placing surfaces behind the real ones.

**Collision checks** (`check_collisions.py`), outdoor path on `delivery_area`: heightfield vs mesh ground 1.0 cm (95th percentile). Against the laser scan's own ground, where the camera saw the ground: median 0.26 cm, 95th percentile 2.7 cm. Ground the mesh filled in is not scored yet, since the laser reference still picks up the ceiling in about 10 m2.

Object boxes on a 10 cm grid vs the mesh, both ways, within 5 / 10 cm:

| Height above ground | Mesh covered by boxes | Box faces on the mesh | Surface IoU |
|---|---|---|---|
| 0.5 m | 0.74 / 0.95 | 0.51 / 0.75 | 0.48 |
| 1.0 m | 0.63 / 0.88 | 0.52 / 0.76 | 0.47 |

G1 walk test: 20 of 20 goals reached, 0 falls, same as on a bare plane. Walking straight at a tall object from 0.5 m, 10 runs: 7 touched it, 0 passed through.

## Choices so far

- Indoors: a plane for the floor, boxes for walls and convex hulls for clutter. No heightfield indoors, since it erases thin walls and can't represent overhangs.
- Outdoors and on sites (`--outdoor`): a heightfield from cloth-simulation ground points (curbs, slopes), and everything standing on it made solid on a 5 cm grid and covered by merged boxes on a 10 cm grid (`--box-cell`). Overhangs keep the gap below them. CoACD was dropped here because its error bound scales with the part, so a 50 m facade got meters of slack.
- LiDAR poses when available. Camera-only odometry on the HCIC construction bag drifted into doubled walls.
- ETH3D for scoring, because it has a laser-scan ground truth.
