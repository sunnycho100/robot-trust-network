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

3. Synthetic reconstruction source (headless, about 240 frames):
   ```bash
   /isaac-sim/python.sh /robot/isaac-sim/demos/capture_walkthrough.py /robot/walk/full_warehouse --frames 240
   ```
   Then copy `~/robot/walk/full_warehouse` to wherever 3DGRUT runs; `sparse/0` is a COLMAP text model with exact poses.
