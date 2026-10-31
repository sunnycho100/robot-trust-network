#!/bin/bash
# The dafleet job (2 H1 + Go2 share one hand-off spot, then cross) on the delivery_area TSDF collider instead of the
# Poisson one: same 3DGUT splat, same robots and stops. The TSDF is closer to the laser (88% vs 79% within 5 cm,
# F1 0.83 at 5 cm vs 0.84) but leaves 30% of pixels without a hit (Poisson 3%), so this checks whether its holes
# drop a robot on straight job-mode legs (no planner). Pass: same arrivals as dafleet, no falls in traj_check.
# Usage: setsid nohup bash datsdf_fleet.sh > /content/datsdf_fleet.log 2>&1 &   (ends DATSDF_FLEET_DONE)
set -x
G=/content/grut_work/da_full; W=/content/work/pm_da; C=$W/tsdf
cd /content/3dgrut && . .venv/bin/activate
python /content/code/grut_collider.py $G/da_full_nurec.usdz $C/tsdf_mesh.ply $C/mesh_usdframe.ply
python -m threedgrut.export.scripts.add_mesh_to_usdz --input_usdz $G/da_full_nurec.usdz \
  --output_usdz $C/da_tsdf_hybrid.usdz --mesh_ply $C/mesh_usdframe.ply --set_collision --set_invisible
python /content/code/level_scene.py $C/mesh_usdframe.ply \
  /content/eth3d/delivery_area/dslr_calibration_undistorted/images.txt $C/da_tsdf_hybrid.usdz $C/da_tsdf_world.usda
deactivate
bash /content/isaac_run.sh datsdf 45 /content/examples/multi_robot_scene_plan2.py --scene $C/da_tsdf_world.usda \
  --robot h1 -4.61 2.78 -139 0 0 0 --robot h1 -4.23 -0.19 154 0 0 0 --robot go2 -8.88 0.72 7 0 0 0 --stop 0 -6.65 1.01 2.0 --stop 0 -8.56 -1.76 0 --stop 1 -6.65 1.01 2.0 --stop 1 -9.20 3.20 0 --stop 2 -6.65 1.01 2.0 --stop 2 -4.42 1.30 0 --eye -0.45 1.81 1.47 --target -6.89 0.98 1.10 --log /content/rec/datsdf > /content/isaac_run_datsdf.log 2>&1
grep "\[fleet\]\|\[multi\]\|frames" /content/isaac_run_datsdf.log | tail -12
python3 /content/examples/traj_check.py /content/rec/datsdf_traj.csv
echo DATSDF_FLEET_DONE
