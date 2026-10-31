#!/bin/bash
# Rerun of da_h1 with the dome light (the H1 rendered black) and level_scene's training-camera viewer (the free
# viewpoint showed blurry floaters). Waits for walk_tsdf's Isaac run.
# Usage: setsid nohup bash dah1b.sh > /content/dah1b.log 2>&1 &   (ends DAH1B_DONE; run log /content/isaac_run_dah1b.log)
while ! grep -q WALKT_DONE /content/walk_tsdf.log 2>/dev/null; do sleep 30; done
set -x
W=/content/work/pm_da
ARGS=$(/content/3dgrut/.venv/bin/python /content/code/level_scene.py $W/mesh_usdframe.ply \
  /content/eth3d/delivery_area/dslr_calibration_undistorted/images.txt $W/da_full_hybrid.usdz $W/da_world.usda)
echo "ARGS $ARGS"
[ -z "$ARGS" ] && { echo DAH1B_DONE; exit 1; }
bash /content/isaac_run.sh dah1b 40 /content/examples/multi_robot_scene_plan2.py --scene $W/da_world.usda --plan $ARGS \
  > /content/isaac_run_dah1b.log 2>&1
grep "\[fleet\]\|\[multi\]\|frames" /content/isaac_run_dah1b.log | tail -20
echo DAH1B_DONE
