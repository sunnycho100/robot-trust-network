#!/bin/bash
# Critic hybrid-walk action 7 (wall half): five H1s walk straight into collider walls from 1 m in the REAL
# delivery_area hybrid (sites from wall_sites.py pick, goals 1.5 m behind each wall). Bar: no pass-through.
# Usage: setsid nohup bash dawall.sh > /content/dawall.log 2>&1 &   (ends DAWALL_DONE; run log /content/isaac_run_dawall.log)
while ! grep -q DAFLEET_DONE /content/dafleet.log 2>/dev/null; do sleep 30; done
set -x
W=/content/work/pm_da; PY=/content/3dgrut/.venv/bin/python
ARGS=$(cd /content/code && $PY wall_sites.py pick $W/mesh_usdframe.ply \
  /content/eth3d/delivery_area/dslr_calibration_undistorted/images.txt $W/da_world.usda $W/wall_sites.json 6)
echo "ARGS $ARGS"
bash /content/isaac_run.sh dawall 25 /content/examples/multi_robot_scene_plan2.py --scene $W/da_world.usda $ARGS \
  --log /content/rec/dawall > /content/isaac_run_dawall.log 2>&1
grep "\[fleet\]\|\[multi\]\|frames" /content/isaac_run_dawall.log | tail -20
(cd /content/code && $PY wall_sites.py check $W/wall_sites.json /content/rec/dawall_traj.csv)
echo DAWALL_DONE
