#!/bin/bash
# The dafleet job on the TSDF collider with its cracks filled from Poisson (fill_cracks.py). datsdf dropped all
# three robots (H1s over 16 cm and 9 cm cracks under the feet). Pass: dafleet arrivals, no falls in traj_check.
# Usage: setsid nohup bash dafill_fleet.sh > /content/dafill_fleet.log 2>&1 &   (ends DAFILL_FLEET_DONE)
set -x
G=/content/grut_work/da_full; W=/content/work/pm_da; F=$W/fill; mkdir -p $F
/content/o3d19/bin/python /content/code/fill_cracks.py $W/tsdf/mesh_usdframe.ply $W/mesh_usdframe.ply $F/mesh_usdframe.ply
cd /content/3dgrut && . .venv/bin/activate
python -m threedgrut.export.scripts.add_mesh_to_usdz --input_usdz $G/da_full_nurec.usdz \
  --output_usdz $F/da_fill_hybrid.usdz --mesh_ply $F/mesh_usdframe.ply --set_collision --set_invisible
python /content/code/level_scene.py $F/mesh_usdframe.ply \
  /content/eth3d/delivery_area/dslr_calibration_undistorted/images.txt $F/da_fill_hybrid.usdz $F/da_fill_world.usda
python /content/code/fallsite.py /content/rec/datsdf_traj.csv $F/mesh_usdframe.ply $F/da_fill_world.usda
deactivate
bash /content/isaac_run.sh dafill 45 /content/examples/multi_robot_scene_plan2.py --scene $F/da_fill_world.usda \
  --robot h1 -4.61 2.78 -139 0 0 0 --robot h1 -4.23 -0.19 154 0 0 0 --robot go2 -8.88 0.72 7 0 0 0 --stop 0 -6.65 1.01 2.0 --stop 0 -8.56 -1.76 0 --stop 1 -6.65 1.01 2.0 --stop 1 -9.20 3.20 0 --stop 2 -6.65 1.01 2.0 --stop 2 -4.42 1.30 0 --eye -0.45 1.81 1.47 --target -6.89 0.98 1.10 --log /content/rec/dafill > /content/isaac_run_dafill.log 2>&1
grep "\[fleet\]\|\[multi\]\|frames" /content/isaac_run_dafill.log | tail -12
python3 /content/examples/traj_check.py /content/rec/dafill_traj.csv
echo DAFILL_FLEET_DONE
