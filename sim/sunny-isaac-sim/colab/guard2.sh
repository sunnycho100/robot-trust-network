#!/bin/bash
# Pit guard v2 (negative evidence: hole only in 0.5 m cells where SfM shows a point over 10 cm below the floor and
# none on it). Unfiltered TSDF, scored against the unguarded fclip TSDF (score_fclip_tsdf.json; bar: free IoU and
# holes within 0.01). Ends GUARD2_DONE in /content/guard2.log.
set -ex
W=/content/grut_work/walk; WK=/content/walk/full_warehouse; PY=/content/o3d19/bin/python; C=/content/grut_work/walk_fcg2
cd /content/code
rm -rf $C
nice $PY floor_clip.py /content/grut_work/walk_gd /content/walk_tri/sparse/0/points3D.txt $C --guard
nice $PY tsdf_mesh.py $C $W/fclipg2_tsdf.ply 0.02
nice $PY score_robot.py $W/fclipg2_tsdf.ply $WK $W/score_fclipg2_tsdf.json --poses /content/walk_rand/sparse/0/images.txt | grep -E "floor_err|holes|free_iou|false_|recall"
grep -E "floor_err|holes|free_iou|false_|recall" $W/score_fclip_tsdf.json
echo GUARD2_DONE
