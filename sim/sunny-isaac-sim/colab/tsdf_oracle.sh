#!/bin/bash
# Mesher head to head on the GT depth of the warehouse capture: TSDF (2 cm) vs eth3d_prep's Poisson (score_oracle_mesh.json).
# Usage: nohup bash tsdf_oracle.sh > /content/tsdf_oracle.log 2>&1 &   (ends TSDF_DONE)
set -x
[ -x /content/o3d19/bin/python ] || { uv venv -q /content/o3d19 --python 3.12 && uv pip install -q --python /content/o3d19/bin/python open3d==0.19.0 scipy pillow; }
/content/o3d19/bin/python /content/code/tsdf_mesh.py test || { echo TSDF_DONE; exit 1; }
WK=/content/walk/full_warehouse; O=/content/grut_work/walk_oracle; W=/content/grut_work/walk
cd /content/code; PY=/content/o3d19/bin/python  # Open3D 0.20 legacy TSDF returns empty meshes (self-check fails), 0.19 works
python3 -c "
import numpy as np, open3d as o3d
v=np.asarray(o3d.io.read_triangle_mesh('$O/frames/mesh.ply').vertices); print('poisson oracle vertex z percentiles', np.percentile(v[:,2],[1,10,25,50,75,90,99]).round(2), len(v))"
t=$(date +%s); nice $PY tsdf_mesh.py $WK $O/tsdf_mesh.ply 0.02; echo "TIME tsdf $(( $(date +%s) - t ))"
$PY score_robot.py $O/tsdf_mesh.ply $WK $W/score_oracle_tsdf.json --poses $WK/sparse/0/images.txt
echo TSDF_DONE
