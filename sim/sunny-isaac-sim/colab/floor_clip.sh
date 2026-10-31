#!/bin/bash
# Splat depth clipped at the SfM floor plane -> 2 cm TSDF -> score_robot, against splat TSDF (free IoU 0) and PM TSDF.
# Usage: setsid nohup bash floor_clip.sh > /content/floor_clip.log 2>&1 < /dev/null &   (ends FCLIP_DONE)
set -ex
W=/content/grut_work/walk; WK=/content/walk/full_warehouse; PY=/content/o3d19/bin/python; C=/content/grut_work/walk_fc
cd /content/code
nice $PY floor_clip.py /content/grut_work/walk_gd /content/walk_tri/sparse/0/points3D.txt $C
nice $PY tsdf_mesh.py $C $W/fclip_tsdf_mesh.ply 0.02
nice $PY score_robot.py $W/fclip_tsdf_mesh.ply $WK $W/score_fclip_tsdf.json --poses /content/walk_rand/sparse/0/images.txt
echo FCLIP_DONE
