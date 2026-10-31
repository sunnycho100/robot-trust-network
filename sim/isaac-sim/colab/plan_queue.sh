#!/bin/bash
# After the job recording: the fleet layout again with --plan (navmap.py A* around the warehouse's meshes).
# Prints each robot's path and saves the occupancy grid to $SP/navmap.npz.
while ! grep -q JOB_DONE /content/job_queue.log 2>/dev/null; do sleep 30; done
bash /content/isaac_run.sh plan 40 /content/examples/multi_robot_scene_plan.py --plan \
  --robot h1 -4 -3 0 0 0 0 --robot go2 -4 -1 0 0 0 0 --robot h1 -4 1 0 0 0 0 --robot anymal -4 3 0 0 0 0 \
  --goal 3 -1.6 --goal 3 0 --goal 3 1.6 --goal 4.6 0 > /content/isaac_run_plan.log 2>&1
cp /content/isim/lib/python3.12/site-packages/isaacsim/navmap.npz /content/rec/ 2>/dev/null
echo PLAN_DONE
