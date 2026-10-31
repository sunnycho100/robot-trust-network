#!/bin/bash
# Re-record the --goal fleet run with PYTHONUNBUFFERED (the first run's [fleet] lines sat in Python's buffer and died
# with the app). Runs after the whole chain so it never shares the GPU. Usage: nohup bash fleet2_queue.sh > /content/fleet2_queue.log 2>&1 &   (ends FLEET2_DONE)
while ! grep -q TRI_DONE /content/walk_tri.log 2>/dev/null; do sleep 30; done
bash /content/isaac_run.sh fleet2 40 /content/examples/multi_robot_scene.py \
  --robot h1 -4 -3 0 0 0 0 --robot go2 -4 -1 0 0 0 0 --robot h1 -4 1 0 0 0 0 --robot anymal -4 3 0 0 0 0 \
  --goal 3 -1.6 --goal 3 0 --goal 3 1.6 --goal 4.6 0 > /content/isaac_run_fleet2.log 2>&1
grep "\[fleet\]\|\[multi\]\|frames" /content/isaac_run_fleet2.log | tail -20
echo FLEET2_DONE
