#!/bin/bash
# After pm_da.sh and plan2: an H1 walks an A* path in the REAL ETH3D delivery_area, 3DGUT NuRec looks plus the
# PatchMatch mesh as the invisible collider (da_full_hybrid.usdz), no ground plane. level_scene.py levels the floor
# to z = 0 and picks the start, goal and camera on the PatchMatch free floor.
# Usage: nohup bash da_h1.sh > /content/da_h1.log 2>&1 &   (ends DAH1_DONE; run log /content/isaac_run_dah1.log)
while ! grep -q PM_DONE /content/pm_da.log 2>/dev/null || ! grep -q PLAN2_DONE /content/plan2_queue.log 2>/dev/null; do sleep 30; done
set -x
W=/content/work/pm_da
[ -f $W/da_full_hybrid.usdz ] || { echo "no hybrid usdz"; echo DAH1_DONE; exit 1; }
ARGS=$(/content/3dgrut/.venv/bin/python /content/code/level_scene.py $W/mesh_usdframe.ply \
  /content/eth3d/delivery_area/dslr_calibration_undistorted/images.txt $W/da_full_hybrid.usdz $W/da_world.usda)
echo "ARGS $ARGS"
[ -z "$ARGS" ] && { echo DAH1_DONE; exit 1; }
cp /content/code/navmap.py /content/examples/navmap.py
bash /content/isaac_run.sh dah1 40 /content/examples/multi_robot_scene_plan2.py --scene $W/da_world.usda --plan $ARGS \
  > /content/isaac_run_dah1.log 2>&1
grep "\[fleet\]\|\[multi\]\|frames" /content/isaac_run_dah1.log | tail -20
echo DAH1_DONE
