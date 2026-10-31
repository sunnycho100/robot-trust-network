#!/bin/bash
# Critic hybrid-walk action 8: three robots (2 H1, Go2) share one hand-off spot in the REAL delivery_area hybrid,
# filmed from training camera 41 (layout from overnight/scratch/da_fleet_standin.py). Waits for the laser check.
# Usage: setsid nohup bash dafleet.sh > /content/dafleet.log 2>&1 &   (ends DAFLEET_DONE; run log /content/isaac_run_dafleet.log)
while ! grep -q LASER_DONE /content/laser_check.log 2>/dev/null; do sleep 30; done
set -x
bash /content/isaac_run.sh dafleet 45 /content/examples/multi_robot_scene_plan2.py --scene /content/work/pm_da/da_world.usda \
  --robot h1 -4.61 2.78 -139 0 0 0 --robot h1 -4.23 -0.19 154 0 0 0 --robot go2 -8.88 0.72 7 0 0 0 --stop 0 -6.65 1.01 2.0 --stop 0 -8.56 -1.76 0 --stop 1 -6.65 1.01 2.0 --stop 1 -9.20 3.20 0 --stop 2 -6.65 1.01 2.0 --stop 2 -4.42 1.30 0 --eye -0.45 1.81 1.47 --target -6.89 0.98 1.10 --log /content/rec/dafleet > /content/isaac_run_dafleet.log 2>&1
grep "\[fleet\]\|\[multi\]\|frames" /content/isaac_run_dafleet.log | tail -25
wc -l /content/rec/dafleet_traj.csv
echo DAFLEET_DONE
