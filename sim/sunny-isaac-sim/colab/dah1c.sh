#!/bin/bash
# Third delivery_area run: level_scene.py now picks the walk inside one training camera's view cone and puts the
# viewer at that camera (dah1b had the light but a viewer that saw only part of the 10 m walk). Waits for dah1b.
# Usage: setsid nohup bash dah1c.sh > /content/dah1c.log 2>&1 &   (ends DAH1C_DONE; run log /content/isaac_run_dah1c.log)
while ! grep -q DAH1B_DONE /content/dah1b.log 2>/dev/null; do sleep 30; done
set -x
W=/content/work/pm_da
ARGS=$(/content/3dgrut/.venv/bin/python /content/code/level_scene.py $W/mesh_usdframe.ply \
  /content/eth3d/delivery_area/dslr_calibration_undistorted/images.txt $W/da_full_hybrid.usdz $W/da_world.usda)
echo "ARGS $ARGS"
[ -z "$ARGS" ] && { echo DAH1C_DONE; exit 1; }
bash /content/isaac_run.sh dah1c 40 /content/examples/multi_robot_scene_plan2.py --scene $W/da_world.usda --plan $ARGS \
  > /content/isaac_run_dah1c.log 2>&1
grep "\[fleet\]\|\[multi\]\|frames" /content/isaac_run_dah1c.log | tail -20
echo DAH1C_DONE
