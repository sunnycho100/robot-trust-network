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
