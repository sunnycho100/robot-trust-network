# Isaac Sim 6.2 demos

<!-- Usage: run the blocks below in order, after isaac_up.sh has the web viewer working (see ../README.md). -->

Four demos: three H1s in the warehouse, the NuRec cafe download, one H1 in the cafe, one H1 in our reconstructed scene. Each demo is its own Isaac Sim app streaming to the same web viewer.

| On the VM | In the container | What |
|---|---|---|
| `~/robot/isaac-sim/demos/` | `/robot/isaac-sim/demos/` | This folder |
| `~/robot/our_scene/` | `/robot/our_scene/` | Our scene export (step 0) |
| `~/robot/nurec/` | `/robot/nurec/` | NuRec cafe (step 1) |

## 0. On the Mac: pack our scene

```bash
cd ~/Documents/projects/robot
tar --no-xattrs -czf ~/Desktop/our_scene.tgz -C sim/sunny-scene-recon/data/floor4_static3/mujoco export
```

With the Jupyter file browser, upload `our_scene.tgz` (5 MB) to `~/robot/` and the files of this folder to `~/robot/isaac-sim/demos/`.

## 1. On the VM (Jupyter terminal)

```bash
mkdir -p ~/robot/our_scene && tar xzf ~/robot/our_scene.tgz -C ~/robot/our_scene --strip-components 1
sudo setfacl -R -m u:1234:rwX ~/robot   # the container user writes our_scene.usd there (as isaac_up.sh does)
bash ~/robot/isaac-sim/demos/get_nurec_cafe.sh
```

`get_nurec_cafe.sh` stops and says what to do if you haven't accepted the dataset terms or run `hf auth login`.

The streaming app that `isaac_up.sh` starts holds the WebRTC ports, so stop it and open a shell in a fresh container from the same compose files. It keeps the `/robot` mount and the caches, and the web viewer keeps running:

```bash
export ISAAC_SIM_IMAGE=nvcr.io/nvidia/isaac-sim:6.2.0 ISAACSIM_HOST=$(curl -4 -s ifconfig.me)
C="docker compose -p isim -f ${REPO:-$HOME/IsaacSim}/tools/docker/docker-compose.yml -f $HOME/robot/isaac-sim/docker-compose.override.yml"
$C stop isaac-sim
$C run --rm --entrypoint bash isaac-sim
```

`ISAACSIM_HOST` must be the address `isaac_up.sh` printed, because the viewer was built with it.

## 2. In the container

After starting each demo, wait for it to load (the first run downloads assets and compiles shaders), then reload the viewer tab. Ctrl+C ends a demo.

Three H1s walking a course in the full warehouse:

```bash
bash /robot/isaac-sim/demos/h1_warehouse.sh
```

One H1 in the NuRec cafe (spawns at the origin; move it with `--pos X Y Z` if that isn't open floor):

```bash
/isaac-sim/python.sh /robot/isaac-sim/demos/h1_in_scene.py /robot/nurec/nova_carter-cafe/particle_sh_optimized.usdz --stream
```

One H1 in our scene. The import is headless and writes `/robot/our_scene/our_scene.usd`; the spawn is the most open spot from `build_scene.py`:

```bash
/isaac-sim/python.sh /robot/isaac-sim/demos/import_our_scene.py /robot/our_scene
/isaac-sim/python.sh /robot/isaac-sim/demos/h1_in_scene.py /robot/our_scene/our_scene.usd --pos 0.95 0.14 1.05 --stream
```

The room is 4.5 x 4.1 m and cluttered (0.2 m clearance at that spot), so the H1 reaches clutter or a wall within a few steps. `--command 0 0 0` makes it stand, `--command 0.45 0 0.7` walks a left arc.

Several robots with their stock policies in one scene (two H1s, Spot, Go2 and ANYmal walking circles in the full warehouse; `--robot KIND X Y YAW VX VY WZ` repeated replaces them, `--scene` takes a local file):

```bash
/isaac-sim/python.sh /robot/isaac-sim/demos/multi_robot_scene.py --stream
```

Render a camera walkthrough of a stock environment for reconstruction tests. It writes images, a COLMAP model with the exact poses, 16-bit depth, the ground-truth mesh and a held-out list (headless, no viewer needed; `python test_capture_walkthrough.py` checks the pose math):

```bash
/isaac-sim/python.sh /robot/isaac-sim/demos/capture_walkthrough.py /robot/walk/full_warehouse --frames 240
```

## 3. Back to the normal viewer

```bash
exit
ACCEPT_EULA=Y bash ~/robot/isaac-sim/isaac_up.sh up
```
