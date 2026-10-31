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
   ```
   `grep "\[fleet\]"` in the output gives arrivals, closest approach and base heights (a fall shows as base z dropping).

3. Synthetic reconstruction source (headless, about 240 frames):
   ```bash
   /isaac-sim/python.sh /robot/isaac-sim/demos/capture_walkthrough.py /robot/walk/full_warehouse --frames 240
   ```
   Then copy `~/robot/walk/full_warehouse` to wherever 3DGRUT runs; `sparse/0` is a COLMAP text model with exact poses.
   Already done once on Colab (`overnight/colab/walk_queue.sh`); its outputs are in `docs/evidence/overnight/grut/` if the run finished.

4. Robots in OUR reconstruction, with its own collider and no ground plane. Copy `walk_world.usda` and
   `walk_nurec_collider.usdz` into one folder (the wrapper references the usdz by relative path and undoes the
   exporter's normalizing transform, so the stage is in the warehouse frame, metres, Z up). Starts and goals are in
   `grep ARGS walk_h1.log`:
   ```bash
   /isaac-sim/python.sh /robot/isaac-sim/demos/multi_robot_scene.py --scene /robot/recon/walk_world.usda --plan $ARGS --stream
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
