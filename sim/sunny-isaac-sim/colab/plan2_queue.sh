#!/bin/bash
# Replaces fleet2 (the plan run already logged the goal layout). After the whole chain: an obstacle-routing layout
# where the straight line goes through a shelf. H1 and Go2 start inside aisles and must walk out, around the shelf
# end and into the next aisle (A* 12 m against 4.9 m straight); the ANYmal crosses the open floor below.
# Stand-in (overnight/scratch/uturn_standin.py): all arrive by 22 s, closest pair 3.2 m, H1 centre 0.42 m from a shelf.
# Usage: nohup bash plan2_queue.sh > /content/plan2_queue.log 2>&1 &   (ends PLAN2_DONE)
while ! grep -q TRI_DONE /content/walk_tri.log 2>/dev/null; do sleep 30; done
bash /content/isaac_run.sh plan2 45 /content/examples/multi_robot_scene_plan2.py --plan \
  --robot h1 -17.9 13 -90 0 0 0 --robot go2 -8 13 -90 0 0 0 --robot anymal -3 5 180 0 0 0 \
  --goal -22.8 13 --goal -12.9 13 --goal -20.4 5 --eye -12.9 -5 13 > /content/isaac_run_plan2.log 2>&1
grep "\[fleet\]\|\[multi\]\|frames" /content/isaac_run_plan2.log | tail -25
echo PLAN2_DONE
