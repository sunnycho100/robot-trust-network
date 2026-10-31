# Factory reconstruction for Isaac Sim 6.2: datasets, synthetic route, method pick

Date: 2026-10-31. Goal: a photoreal splat plus a collision mesh of a large open indoor space, loaded in Isaac Sim 6.2, scored against ground truth, on one Colab L4 (24 GB) in about 3 hours.

Everything marked **verified** was checked on the live page or by an HTTP request today. Anything else is marked **unverified**.

## Short answer

- **Dataset tonight:** Tanks and Temples **Church** (large open indoor hall, laser ground truth, 0.69 GB images plus 1.7 GB scan, no signup). Use **Meetingroom** from the same set as the calibration scene, since 2DGS and PGSR publish F1 on it.
- **Method:** **3DGRUT (3DGUT config, MCMC)** for the appearance, exported as a ParticleField USDZ. **Collision from a separate mesh**, not the splat: our COLMAP PatchMatch mesh first, then attached with 3DGRUT's `add_mesh_to_usdz --set_collision`, plus an Isaac Sim Ground Plane. Optional splat-derived mesh: gsplat 2DGS plus TSDF on rendered depth, reusing `reconstruct.py`.
- **License flag:** 2DGS, PGSR, GOF, SuGaR and MILo are all non-commercial (verified on their LICENSE files). For a paid consulting tool, only 3DGRUT, gsplat and nerfstudio (Apache-2.0) are clean. Most datasets below are non-commercial too, which is fine for benchmarking but not for shipping their assets.

## Part 1: datasets

### Table

| Dataset | Link | Size we need | License | Gated | Images | GT | Factory-like and open |
|---|---|---|---|---|---|---|---|
| Tanks and Temples, Church | tanksandtemples.org/download | 0.69 GB images, 1.7 GB scan (verified) | Non-commercial research only (verified, the page says CC BY 4.0 but the grant forbids commercial use) | No. Public Google Drive, anonymous download confirmed | unverified (several hundred frames from 4K video) | Laser scan, plus alignment and crop files | High: big open indoor hall |
| Tanks and Temples, Meetingroom | same | 0.40 GB images, 1.1 GB scan (verified) | same | No | unverified (about 370 from memory) | Laser scan | Medium: one room, table and chairs |
| Tanks and Temples, Courthouse, Barn | same | Courthouse 1.1 plus 1.5 GB, Barn 0.50 plus 0.33 GB (verified) | same | No | unverified | Laser scan | Low: outdoor |
| T and T Auditorium, Ballroom | same | not checked | same | No | unverified | **None public** (advanced test set, scored by server only) | High, but unscorable |
| ETH3D high-res, delivery_area | eth3d.net/datasets | 0.48 GB images, 0.12 GB scan (verified) | CC BY-NC-SA 4.0 (verified) | No | 44 DSLR (verified) | Laser scan | Medium: covered loading area. Our baseline already exists |
| ETH3D terrains, office, pipes | same | 0.46, 0.29, 0.15 GB images (verified) | same | No | 42, 26, 14 (verified) | Laser scan | Low: small rooms. Pipes is industrial but 14 images |
| ETH3D exhibition_hall, lecture_room | same | not checked | same | No | unverified | No public GT (test set) | High, but unscorable |
| Hilti 2022, exp04 upper level | huggingface.co/datasets/Hilti-Research/hilti-slam-challenge-2022 | 10.47 GB bag plus 0.83 GB e57 scan (verified) | CC BY-NC-SA 3.0 (verified) | No (verified, gated=False) | 5 cameras, 720x540, **greyscale** (verified) | TLS scan, surveyed points | High: concrete floors, columns. Pipeline exists (`run_hilti.sh`) |
| Hilti 2022, Voxel51 MCAP copy | huggingface.co/datasets/Voxel51/Hilti-SLAM-Challenge-2022 | 1.52 GB for exp04 (verified) | same | No | JPEG, reduced frame rate (verified); laser scans not included | Survey points only | same |
| Hilti 2021, Parking_1, Basement | huggingface.co/datasets/Hilti-Research/hilti-slam-challenge-2021 | 29.3 GB, 6.7 to 21.7 GB (verified) | CC BY-NC-SA 3.0 | No | greyscale (unverified) | Trajectory only, no scan in the repo listing | High (a parking garage), but too big and no geometry GT |
| Hilti x Trimble 2026 | huggingface.co/datasets/Hilti-Research/hilti-trimble-slam-challenge-2026 | 2.6 to 7 GB per run (verified) | CC BY-NC-SA 3.0 | No | Dual fisheye RGB, 1472x1440, 30 Hz (verified) | Trajectories and floor plans; raw LiDAR **not included** (verified) | High: active construction site. No geometry GT |
| IRIS (Industrial Room In Saclay) | zenodo.org/records/13777859 | about 2.2 GB per panorama part, 2.5 GB per scan station; 52 GB total (verified via page) | CC BY-NC-ND 4.0 (verified) | No (open) | 312 panoramas at 16384x8192 plus 67 TLS panoramas (verified) | Station TLS scans, CAD model | **Highest**: 530 m2 room with pipes, pumps, valves. Panoramas need splitting into perspective views |
| NVIDIA PhysicalAI-Robotics-NuRec | huggingface.co/datasets/nvidia/PhysicalAI-Robotics-NuRec | 79.9 GB total; e.g. nova_carter-galileo raw images 8.4 GB | CC BY 4.0 (verified) | **Yes**: HTTP 401 GatedRepo without login (verified). No HF token on this machine | COLMAP models and images for zh_lounge, zh_fourth_floor_iphone, bluefield_iphone | USDZ splats, meshes, occupancy maps | Medium: offices, cafe, lounge. No warehouse scene in v0.2 (verified file list) |
| ScanNet++ | scannetpp.mlsg.cit.tum.de | not given | ScanNet++ Terms of Use | **Yes**: account plus approved application (verified) | 33 MP DSLR | Sub-mm laser | Medium: rooms, some large |
| Matterport3D | niessner.github.io/Matterport | not checked | Signed terms of use | **Yes** (unverified today, known requirement) | panoramas | Mesh | Low: homes |
| Replica | github.com/facebookresearch/Replica-Dataset | 33.9 GB, one split archive (verified) | Research terms, non-commercial (verified) | Click-through only | **No photos**, meshes only | Mesh | Low: apartments, offices |
| LaMAR | lamar.ethz.ch | not given | Research | **Yes**: request form (verified) | AR device captures | Laser scans, meshes | Medium: CAB and HGE have big halls |
| Zip-NeRF | smerf-3d.github.io | 10.8 to 22.6 GB per scene (verified) | not checked | No | about 1000 to 2000 per scene (unverified) | None | Low: houses |
| Mip-NeRF 360 | jonbarron.info/mipnerf360 | 12.5 GB plus 4.5 GB (verified) | not checked | No | 100 to 300 per scene (unverified) | None | Low: object-centric |
| TUM RGB-D fr2 pioneer_360, pioneer_slam | cvg.cit.tum.de/data/datasets/rgbd-dataset | 0.73 to 1.82 GB (verified) | CC BY 4.0 (verified) | No | 640x480 Kinect RGB-D | Mocap poses only | Medium: wheeled robot in a big empty hall, but low-res and no geometry GT |
| Newer College, Oxford Spires | ori-drs.github.io/newer-college-dataset | not checked | not checked | No (Hugging Face) | RealSense, Alphasense | TLS map | Low: mostly outdoor courtyards |
| WHU | not resolved | | | | | | Name matches several datasets (mostly TLS point clouds without photos). Not checked |
| IndustReal | arxiv.org/abs/2310.17323 | | | | | | Not relevant: egocentric assembly videos of a toy car |
| FactoryNet | icml.cc 2026 | | | | | | Not relevant: industrial time series |

New in 2025 to 2026: IRIS (Feb 2025) is the only open, truly industrial interior with geometry GT found. Hilti x Trimble 2026 has no geometry GT. A 2024 factory-hall SLAM paper (arXiv 2404.17215) did not publish data.

### Top 3 for Colab tonight

Rules: no signup, under about 10 GB, color photos, laser GT, as open and factory-like as possible.

1. **Tanks and Temples Church.** Biggest open indoor space with public laser GT. 2.4 GB total. Color frames suit splats far better than ETH3D's 44 DSLR shots or Hilti's greyscale. Grab the alignment and crop files from the "Results on Training Data" table so the official T&T evaluator works. Download: https://www.tanksandtemples.org/download/. Images `Church.zip`: https://drive.google.com/file/d/0B-ePgl6HF260SmhXM0czaHJ3SU0/view?resourcekey=0-Gsw54gkOI-Crg5S-9wJZpg . Scan `Church.ply`: https://drive.google.com/file/d/0B-ePgl6HF260ZWhubWM4TWNvS3c/view?resourcekey=0-uhj_yYICuKLgLp0k7UPbcg (both verified anonymous, `gdown` handles the confirm page).
2. **Tanks and Temples Meetingroom.** 1.5 GB. Less open, but 2DGS, PGSR and GOF all report F1 on it, so it tells us whether our setup reproduces published numbers before we trust Church.
3. **ETH3D delivery_area.** 0.6 GB, direct URLs (`https://www.eth3d.net/data/delivery_area_dslr_undistorted.7z`, `..._dslr_scan_eval.7z`, both verified). Our PatchMatch and vid2scene scores already exist, so 3DGRUT drops straight into the same table.

Runner-ups: Hilti 2022 exp04 (most factory-like, pipeline ready, but 11.3 GB and greyscale, so geometry only). IRIS (best match for a customer factory; plan a dedicated session to split panoramas and pick 2 or 3 scan stations; the ND license means we cannot share derived models).

## Part 2: synthetic route in Isaac Sim

### Stock environment paths (Isaac Sim 6.2)

All return HTTP 200 under `https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.2` (verified today):

- `/Isaac/Environments/Simple_Warehouse/full_warehouse.usd` (best match: big, open, shelving)
- `/Isaac/Environments/Simple_Warehouse/warehouse.usd`
- `/Isaac/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd`
- `/Isaac/Environments/Simple_Warehouse/warehouse_with_forklifts.usd`
- `/Isaac/Environments/Digital_Twin_Warehouse/small_warehouse_digital_twin.usd`
- `/Isaac/Environments/Hospital/hospital.usd`
- `/Isaac/Environments/Office/office.usd`
- `/Isaac/Environments/Simple_Room/simple_room.usd`

`sim/isaac-sim/demos/h1_warehouse.sh` already loads `full_warehouse.usd` on our cloud box.

### Capture: Replicator APIs

- Camera and render product: `rep.functional.create.camera(position=..., look_at=..., parent="/World")` and `rep.create.render_product(cam, (W, H))` (verified, 6.1 tutorial).
- `BasicWriter.initialize(rgb=True, camera_params=True, distance_to_image_plane=True, ...)` writes one JSON of camera params per frame (verified for `rgb` and `camera_params`; the `distance_to_image_plane` flag is listed among Isaac writer parameters, its exact BasicWriter keyword is unverified).
- Use **`distance_to_image_plane`** (z-depth, what TSDF and COLMAP expect), not `distance_to_camera` (ray length). Units are scene units; scale by `metersPerSceneUnit` from camera params (unverified field name).
- The camera params JSON carries a world-to-camera view matrix and projection, focal length and aperture (field names `cameraViewTransform`, `cameraFocalLength`, `cameraAperture`, `renderProductResolution` unverified). There is **no COLMAP writer** (verified list: DataVisualization, DOPE, Pose, Pytorch, YCBVideo). We need a short converter: fx = focal_length x width / horizontal_aperture, then flip the camera Y and Z axes (USD cameras look down minus Z with Y up; COLMAP looks down plus Z with Y down) and write `cameras.txt` and `images.txt` (convention from memory, unverified; check by reprojecting one depth map).
- Ground-truth mesh: read every `UsdGeom.Mesh` with its world transform via `pxr` and write a PLY. Score only surfaces seen by the cameras, as `hilti_run.py` already does for the laser scan.

Plan: about 300 frames on a 1.6 m walk loop (humanoid head height) plus a 0.4 m loop (wheeled robot), hold out every 8th frame. Render on our Isaac Sim cloud GPU, not Colab (running Isaac Sim on Colab is unverified).

### Scoring

Geometry: Chamfer and F1 at 2, 5 and 10 cm against the USD mesh, our existing scorer. Image: PSNR, SSIM, LPIPS on held-out views. Isaac Sim 6.2 also ships `nurec_render.py` (renders a TUM trajectory) and `test_nurec_render_vs_gt.py` (PSNR, SSIM against GT images) (verified on the 6.1 page), which covers the round trip: render the trained USDZ inside Isaac Sim from the held-out poses and score it against the Replicator frames.

### 3DGRUT round trip (verified on the README and export README)

- Train on COLMAP: `python train.py --config-name apps/colmap_3dgut_mcmc.yaml path=<scene> out_dir=runs experiment_name=<name> dataset.downsample_factor=2 export_usd.enabled=true`
- Standalone export: `python -m threedgrut.export.scripts.export_usd --checkpoint ckpt.pt --output scene.usdz --dataset <scene>` with `--format {standard,nurec}`. Prefer `standard` (ParticleField); NuRec USDZ targets Isaac Sim 5.0 to 6.0 and is being deprecated.
- `--up-axis {y,z}` only writes metadata and does not rotate the scene. Our COLMAP frames must be leveled to Z-up first (`build_scene.py` already levels).
- Collision: `python -m threedgrut.export.scripts.add_mesh_to_usdz --input_usdz in.usdz --output_usdz out.usdz --mesh_ply mesh.ply --set_collision`.
- Isaac Sim 6.2 docs say NuRec scenes are ParticleField stages rendered natively by RTX, and point to 3DGRUT for training (verified). Known issue: opening a USDZ as the root stage and adding references fails; open a `.usda` root and reference the USDZ (verified).
- 3DGRUT is Apache-2.0 (verified).

### How NVIDIA makes collision for NuRec

- The NuRec stereo pipeline runs **nvblox** on dense depth plus RGB, producing a triangle mesh and an occupancy map (verified).
- In Isaac Sim, select the mesh prim and use Add, Physics, Collider Preset (verified).
- Floors are not guaranteed at z = 0; the docs say to add a **Ground Plane** or shift the scene root (verified).
- The HF dataset README says the same: meshes give collision, Gaussians give rendering (verified).

This matches our own finding: splats for looks, a real mesh for collision.

### Splat to mesh tools

| Tool | How | License | Notes |
|---|---|---|---|
| 3DGRUT | No documented mesh extraction (verified absent from README) | Apache-2.0 | Can attach any mesh as a collider |
| gsplat 2DGS (`simple_trainer_2dgs.py`) | Render depth, TSDF fuse (our `reconstruct.py`) | Apache-2.0 | 15.7 min, 3.7 GB on a 4090 for Mip-NeRF 360 (verified, gsplat eval page) |
| nerfstudio splatfacto | Depth render plus TSDF or Poisson export | Apache-2.0 | Built on gsplat |
| 2DGS (official) | TSDF, bounded and unbounded | **Non-commercial** (Inria license, verified) | |
| PGSR | TSDF from planar depth | **Non-commercial** (ZJU license, verified) | T&T mean F1 0.51, about 45 min to 1.2 h per T&T scene on their GPU (verified README) |
| GOF | Tetrahedra, marching | **Non-commercial** (verified) | |
| SuGaR | Poisson on aligned Gaussians | **Non-commercial** (verified) | |
| MILo | Mesh in the training loop, Delaunay | **Non-commercial** (verified) | 250K to 14M vertices by config (verified); time and VRAM not documented |

## Part 3: recommendation

**Pick: 3DGRUT 3DGUT with MCMC for appearance, PatchMatch mesh for collision, gsplat 2DGS plus TSDF as the challenger mesh.**

Why 3DGRUT over gsplat for the splat: it is the only one with a first-party ParticleField USDZ export, a collider-attach script, and PPISP color correction, all built for Isaac Sim. Image quality is on par (3DGUT averages 27.4 PSNR on Mip-NeRF 360, gsplat default 29.0 at 30k steps, but those are different eval settings, so treat them as equal; verified numbers, comparison unverified). It also handles fisheye natively, which matters for real factory captures.

Why not 2DGS or PGSR for the mesh: license blocks commercial use, and our own test showed splat-rendered depth scoring F1 0.07 and 0.16 at 2 cm and 5 cm against 0.30 and 0.56 for PatchMatch on delivery_area. PGSR is the most accurate splat mesher on T&T, so it is worth one internal comparison run only.

### Expected cost on one L4 (estimates, unverified)

L4 has about a sixth of the RTX 5090's memory bandwidth, and splat training is bandwidth-bound, so scale published times by about 4 to 6.

| Step | Published | L4 estimate | VRAM estimate |
|---|---|---|---|
| COLMAP GPU SIFT plus GLOMAP, about 400 frames | | 20 to 30 min | under 8 GB |
| 3DGRUT 3DGUT, 30k iters | 7 to 16 min per Mip-NeRF 360 scene on a 5090 (verified) | 40 to 70 min | 8 to 14 GB with MCMC capped at 1 to 2M Gaussians |
| gsplat 2DGS, 30k iters, plus TSDF | 15.7 min, 3.7 GB on a 4090 (verified) | 30 to 45 min | 4 to 8 GB |
| PatchMatch plus fusion at half resolution | | 40 to 60 min | under 10 GB |

That is 2.5 to 3.5 hours in total, so tonight run Church at `downsample_factor=2` with 3DGRUT plus the PatchMatch mesh, and leave 2DGS for the next session if time runs out. Avoid the 3DGUT NHT config: on an L40 it took 76 min per scene on average (verified), so several hours on an L4.

### Run order tonight

1. Download Church images, scan, alignment and crop files (2.4 GB) Check: file sizes match the Drive pages.
2. COLMAP or GLOMAP poses, level to Z-up Check: registered fraction and reprojection error.
3. 3DGRUT train with `export_usd.enabled=true` Check: held-out PSNR, SSIM, LPIPS.
4. PatchMatch mesh, score with the T&T evaluator Check: F1 against the laser scan.
5. `add_mesh_to_usdz --set_collision`, load a `.usda` root in Isaac Sim 6.2, add a Ground Plane Check: H1 walks without falling through.

## Sources

- 3DGRUT README and export README: https://github.com/nv-tlabs/3dgrut
- Isaac Sim 6.2 NuRec page: https://docs.isaacsim.omniverse.nvidia.com/6.2.0/assets/usd_assets_nurec.html
- Isaac Sim environments: https://docs.isaacsim.omniverse.nvidia.com/6.1.0/assets/usd_assets_environments.html
- NuRec utils: https://docs.isaacsim.omniverse.nvidia.com/6.1.0/assets/nurec_utils.html
- Replicator getting started: https://docs.isaacsim.omniverse.nvidia.com/6.1.0/replicator_tutorials/tutorial_replicator_getting_started.html
- Isaac Sim writers: https://docs.isaacsim.omniverse.nvidia.com/6.0.0/py/source/extensions/isaacsim.replicator.writers/docs/index.html
- NuRec stereo pipeline (nvblox, collider, ground plane): https://docs.nvidia.com/nurec/robotics/neural_reconstruction_stereo.html
- NuRec dataset: https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-NuRec
- Tanks and Temples download and license: https://www.tanksandtemples.org/download/ and https://www.tanksandtemples.org/license/
- ETH3D: https://www.eth3d.net/datasets
- Hilti: https://huggingface.co/Hilti-Research and https://hilti-challenge.com/dataset-2022.html
- IRIS: https://zenodo.org/records/13777859
- ScanNet++: https://scannetpp.mlsg.cit.tum.de/scannetpp/
- LaMAR: https://lamar.ethz.ch/lamar/
- Replica: https://github.com/facebookresearch/Replica-Dataset
- TUM RGB-D: https://cvg.cit.tum.de/data/datasets/rgbd-dataset
- gsplat eval: https://docs.gsplat.studio/main/tests/eval.html
- PGSR: https://github.com/zju3dv/PGSR, 2DGS: https://github.com/hbb1/2d-gaussian-splatting, MILo: https://github.com/Anttwo/MILo
