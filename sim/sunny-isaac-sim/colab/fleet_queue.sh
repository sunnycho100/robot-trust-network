#!/bin/bash
# Colab B and 7 on one L4: install Isaac Sim 6.2 pip while 3DGUT trains, then record the fleet demo once training is done.
bash /content/isaac_pip2.sh > /content/isaac_pip2.log 2>&1
while ! grep -q RUN_DONE /content/grut_full.log 2>/dev/null; do sleep 30; done
bash /content/isaac_run.sh fleet 40 /content/examples/multi_robot_scene.py \
  --robot h1 -4 -3 0 0 0 0 --robot go2 -4 -1 0 0 0 0 --robot h1 -4 1 0 0 0 0 --robot anymal -4 3 0 0 0 0 \
  --goal 3 -1.6 --goal 3 0 --goal 3 1.6 --goal 4.6 0 > /content/isaac_run_fleet.log 2>&1
echo FLEET_DONE
