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
| `check_collisions.py` | Wall box to mesh distance, free-space IoU at 0.1 m and 1.0 m |
| `walk_in_scene.py` | G1 walking policy to open and near-obstacle goals, the same goals on a bare plane, and walking into every wall |
| `colab/` | Remote GPU runs: `setup.sh`, `run_eth3d.sh`, `compare_splat.sh` (vid2scene comparison), `run_openmvs.sh` |

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

ETH3D `pipes`, COLMAP PatchMatch on ETH3D's poses (2000 px), scored against the laser scan:

| | Accuracy 2 / 5 cm | Completeness 2 / 5 cm | F1 2 / 5 cm |
|---|---|---|---|
| Fused points | 97.1 / 98.1 % | 22.1 / 34.3 % | 0.36 / 0.51 |
| Poisson mesh | 79.8 / 87.1 % | 34.0 / 40.4 % | 0.48 / 0.55 |

Points are accurate but sparse, so the next runs use a higher resolution and a looser fusion filter, on `delivery_area`, a larger scene.

## vid2scene comparison

[vid2scene](https://github.com/samuelm2/vid2scene) turns a phone video into Gaussian splats (GLOMAP poses, then its gsplat fork). Splats render well but carry no mesh, metric scale or gravity direction, so MuJoCo can't use them directly. `colab/compare_splat.sh` runs both methods on the same 1920 px images, poses and 38 training views. For the splats, it renders their depth and fuses it into a mesh. Both meshes then go through the same MuJoCo pipeline. It reports pose error, ETH3D F1, held-out view PSNR, SSIM and LPIPS, floor flatness, collision checks and the walk test. Features are SIFT instead of vid2scene's ALIKED and LightGlue, so the row is "GLOMAP-SIFT + vid2scene trainer".

## Choices so far

- A plane for the floor, boxes for walls and convex hulls for clutter. No heightfield indoors, since it erases thin walls and can't represent overhangs.
- LiDAR poses when available. Camera-only odometry on the HCIC construction bag drifted into doubled walls.
- ETH3D for scoring, because it has a laser-scan ground truth.
