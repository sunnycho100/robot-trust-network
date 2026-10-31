# Run sheet for the next Brev session

<!-- Usage: run top to bottom on the Brev L40S after README.md's "Order of operations" has the viewer up. -->

Upload this folder to `~/robot/isaac-sim/` first (Jupyter file browser), then:

1. CooHOI, both humanoids in the video (Isaac Gym, outside the Isaac Sim container; see `coohoi/RUN.md`):
   ```bash
   cd ~/robot/isaac-sim/coohoi && python test_record_coohoi.py && bash setup_coohoi.sh && bash run_coohoi.sh 20
   ```
   Check the first-frame log line for the frame choice. One humanoid only: `--partner_shift env` or `none` (edit `run_coohoi.sh`).

2. Several robots in one warehouse, streamed (in the container shell from `demos/RUN.md` step 1):
   ```bash
   /isaac-sim/python.sh /robot/isaac-sim/demos/multi_robot_scene.py --stream
   ```
   Recorded instead: `./python.sh /robot/isaac-sim/demos/record_demo.py /robot/recordings/multi 30 /robot/isaac-sim/demos/multi_robot_scene.py`

   The fleet modes (`demos/fleet.py`), same layouts as the Colab recordings. Add `--stream` to watch live:
   ```bash
   R="--robot h1 -4 -3 0 0 0 0 --robot go2 -4 -1 0 0 0 0 --robot h1 -4 1 0 0 0 0 --robot anymal -4 3 0 0 0 0"
   G="--goal 3 -1.6 --goal 3 0 --goal 3 1.6 --goal 4.6 0"
   /isaac-sim/python.sh /robot/isaac-sim/demos/multi_robot_scene.py $R $G            # goals, robots give way
   /isaac-sim/python.sh /robot/isaac-sim/demos/multi_robot_scene.py $R $G --plan     # plus A* around shelves
   /isaac-sim/python.sh /robot/isaac-sim/demos/multi_robot_scene.py \
     --robot h1 -4 0 0 0 0 0 --robot go2 0 -4 90 0 0 0 --robot anymal 4 -3 135 0 0 0 \
     --stop 0 0 0 3 --stop 0 3 3 0 --stop 1 0 0 2 --stop 1 -3 3 0 --stop 2 0 0 2 --stop 2 -3 -3 0   # one station, in turn
   # the layout above has a clear corridor; this one makes A* route round the shelves (H1 and Go2 leave one aisle
   # and enter the next, 12 m instead of 4.9 m through a shelf), camera south of the aisles
   /isaac-sim/python.sh /robot/isaac-sim/demos/multi_robot_scene.py --plan \
     --robot h1 -17.9 13 -90 0 0 0 --robot go2 -8 13 -90 0 0 0 --robot anymal -3 5 180 0 0 0 \
     --goal -22.8 13 --goal -12.9 13 --goal -20.4 5 --eye -12.9 -5 13
   ```
   Language model on top (`demos/task_plan.py`, Claude writes the station routes, fleet.Jobs runs them; needs
   `pip install anthropic` and Sunghwan's key or `ant auth login`, costs a few cents; live call not tested yet):
   ```bash
   S=$(python3 /robot/isaac-sim/demos/task_plan.py "The H1 and the Go2 each fetch a load from the dock; the H1 takes it to shelf A, the Go2 to shelf B" \
     --robot h1 -4 0 0 --robot go2 0 -4 90 --robot anymal 4 -3 135 --station dock 0 0 3 --station shelfA 0 3 0 --station shelfB -3 3 0)
   /isaac-sim/python.sh /robot/isaac-sim/demos/multi_robot_scene.py $S --stream
   ```
   `grep "\[fleet\]"` in the output gives arrivals, closest approach and base heights (a fall shows as base z dropping).

3. Synthetic reconstruction source (headless, about 240 frames):
   ```bash
   /isaac-sim/python.sh /robot/isaac-sim/demos/capture_walkthrough.py /robot/walk/full_warehouse --frames 240
   ```
   Then copy `~/robot/walk/full_warehouse` to wherever 3DGRUT runs; `sparse/0` is a COLMAP text model with exact poses.
   Already done once on Colab (`overnight/colab/walk_queue.sh`); its outputs are in `docs/evidence/overnight/grut/` if the run finished.

4. Robots in OUR reconstruction, with its own collider and no ground plane. Use the PatchMatch TSDF collider:
   copy `walk_pm_world.usda` and `walk_pm_collider.usdz` into one folder (the splat-depth colliders,
   `walk_world.usda` and `walk_tsdf_world.usda`, have no usable floor: on the TSDF one an H1 and a Go2 fell
   straight through at spawn, 83% of the map blocked). The wrapper references the usdz by
   relative path and undoes the exporter's normalizing transform, so the stage is in the warehouse frame (metres,
   Z up). Starts and goals are in `grep ARGS walk_pm.log` (unverified until tonight's walk_pm run ends):
   ```bash
   /isaac-sim/python.sh /robot/isaac-sim/demos/multi_robot_scene.py --scene /robot/recon/walk_pm_world.usda --plan $ARGS --stream
   ```
   Same for the REAL ETH3D delivery_area (3DGUT looks, PatchMatch collider): copy `da_world.usda` and
   `da_full_hybrid.usdz` together; `sim/sunny-scene-recon/colab/level_scene.py` levelled the floor to z = 0 and
   picked the start, goal and camera (`grep ARGS dah1c.log`). For a new capture, run level_scene.py on its collider
   mesh, its `images.txt` and its usdz first (3dgrut venv). Ran on Colab: the H1 walked the 7-waypoint A* path
   (about 11 m) to the goal in 17.6 s with no fall (`docs/evidence/overnight/fleet/dah1.mp4`). That first run had no
   light (the robot rendered black) and a free camera (blurry floaters). The scene script now adds a dome light for
   `--scene` (verified: `dah1b.mp4`, robot lit, splat sharp from a photo spot), and level_scene.py picks the longest
   walk inside one photo's view and prints that camera as `--eye` and `--target` (3.6 m on delivery_area: its
   tripod shots face the walls). With `--scene` the planner also blocks cells with no collider floor under them (a
   hole is a fall with no ground plane); on delivery_area that removed 1,300 m2 of void the old map called free, and
   no real floor. Add `--log /robot/out/NAME` to keep NAME_navmap.npz and a 50 Hz NAME_traj.csv of every robot.
   ```bash
   /isaac-sim/python.sh /robot/isaac-sim/demos/multi_robot_scene.py --scene /robot/recon/da_world.usda --plan $ARGS --stream
   ```
   Three robots sharing one hand-off spot in the same scan (job mode, 2 s dwell each, then across the floor). Ran on
   Colab: H1s arrived at 11.1 and 18.2 s, the Go2 at 22.2 s, closest 1.09 m, no falls in the trajectory
   (`dafleet.mp4`; check any run with `python3 demos/traj_check.py /robot/out/dafleet_traj.csv`):
   ```bash
   /isaac-sim/python.sh /robot/isaac-sim/demos/multi_robot_scene.py --scene /robot/recon/da_world.usda \
     --robot h1 -4.61 2.78 -139 0 0 0 --robot h1 -4.23 -0.19 154 0 0 0 --robot go2 -8.88 0.72 7 0 0 0 \
     --stop 0 -6.65 1.01 2.0 --stop 0 -8.56 -1.76 0 --stop 1 -6.65 1.01 2.0 --stop 1 -9.20 3.20 0 \
     --stop 2 -6.65 1.01 2.0 --stop 2 -4.42 1.30 0 --eye -0.45 1.81 1.47 --target -6.89 0.98 1.10 \
     --log /robot/out/dafleet --stream
   ```

5. VLA over a trained RL controller: GR00T N1.6 drives a G1 whole-body controller in Isaac Lab-Arena (`main` targets
   Isaac Sim 6.2.0; alpha; weights are non-commercial, internal evaluation only). From the Arena `main` docs
   (`docs/pages/example_workflows/locomanipulation/step_5_evaluation.rst`), not run yet. One G1 per env; `--num_envs 5`
   runs five in parallel, but there is no multi-G1 option in this task. On the host, outside the Isaac Sim container:
   ```bash
   git config --global url."https://github.com/".insteadOf git@github.com:   # .gitmodules uses ssh URLs
   git clone https://github.com/isaac-sim/IsaacLab-Arena.git ~/IsaacLab-Arena && cd ~/IsaacLab-Arena
   git submodule update --init --recursive
   mkdir -p ~/datasets ~/models ~/eval        # run_docker.sh only mounts folders that exist
   export MODELS_DIR=$HOME/models/isaaclab_arena/locomanipulation_tutorial
   hf download --revision gn1_6 nvidia/GN1x-Tuned-Arena-G1-Loco-Manipulation \
     --local-dir $MODELS_DIR/checkpoint-20000 --exclude optimizer.pt   # gn1_6 is N1.6 (default branch is N1.5); skips 10 GB
   # terminal A, policy server on the host (needs uv)
   cd ~/IsaacLab-Arena/submodules/Isaac-GR00T && uv run python gr00t/eval/run_gr00t_server.py \
     --modality-config-path ../../isaaclab_arena_gr00t/embodiments/g1/g1_sim_wbc_data_config.py \
     --model-path $MODELS_DIR/checkpoint-20000 --embodiment-tag NEW_EMBODIMENT --device cuda --host 127.0.0.1 --port 5555
   # terminal B: builds isaaclab_arena:latest on first run (FROM nvcr.io/nvidia/isaac-sim:6.2.0), then inside it
   cd ~/IsaacLab-Arena && ./docker/run_docker.sh
   python isaaclab_arena/evaluation/policy_runner.py \
     --policy_type isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy.Gr00tRemoteClosedloopPolicy \
     --policy_config_yaml_path isaaclab_arena_gr00t/policy/config/g1_locomanip_gr00t_closedloop_config.yaml \
     --remote_host 127.0.0.1 --remote_port 5555 --num_steps 1500 --enable_cameras \
     --record_viewport_video --record_camera_video --output_base_dir /eval/arena \
     galileo_g1_locomanip_pick_and_place --object brown_box --embodiment g1_wbc_joint
   ```
   Expect `Metrics: {success_rate: 1.0, num_episodes: 1}`. Headless is inferred (the docs add `--viz kit` for the GUI);
   the two video flags exist in `policy_runner_cli.py` but are untested on this task. Parallel: add
   `--num_envs 5 --device cuda --policy_device cuda` and `--num_steps 1200`.

6. Multi-agent training scaffold for the CooHOI port (`docs/research/coohoi_port_plan.md`): Isaac Lab 3.0 skrl MAPPO,
   two agents, kit-less Newton physics (no Isaac Sim needed). Ran on a Colab L4 tonight (`overnight/colab/lab_mappo.sh`):
   7 min kernel compile, then about 22k env steps per second; success rate 0.007 to 0.47 (levelling off) and mean reward 1.4 to 41 in the
   first 7200 of 36000 steps (21 min of training; 40 min time box). The full run is about 110 min on an L4. On the host:
   ```bash
   git clone --depth 1 --branch v3.0.0 https://github.com/isaac-sim/IsaacLab ~/IsaacLab && cd ~/IsaacLab
   UV_SYSTEM_PYTHON=false GIT_LFS_SKIP_SMUDGE=1 uv sync --extra skrl
   uv run --extra skrl isaaclab train --rl_library skrl --task Isaac-Shadow-Handover-Direct --algorithm MAPPO \
     --num_envs 4096 physics=newton_mjwarp
   ```
