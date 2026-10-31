#!/bin/bash
# critic_floor_clip action 5: plane fit plus floor-support guard, same TSDF and k2000 filter, score against the
# unguarded fclip_k2000 (bar: within 0.01). Waits for AUDIT_DONE. Ends GUARD_DONE in /content/guard.log.
set -ex
until grep -q AUDIT_DONE /content/audit.log; do sleep 30; done
W=/content/grut_work/walk; WK=/content/walk/full_warehouse; PY=/content/o3d19/bin/python; C=/content/grut_work/walk_fcg
cd /content/code
rm -rf $C
nice $PY floor_clip.py /content/grut_work/walk_gd /content/walk_tri/sparse/0/points3D.txt $C --guard
nice $PY tsdf_mesh.py $C $W/fclipg_k2000.ply 0.02 2000
nice $PY score_robot.py $W/fclipg_k2000.ply $WK $W/score_fclipg_k2000.json --poses /content/walk_rand/sparse/0/images.txt | grep -E "floor_err|holes|free_iou|false_|recall"
grep -E "floor_err|holes|free_iou|false_|recall" $W/score_fclip_k2000.json
echo GUARD_DONE
