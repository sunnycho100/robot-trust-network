#!/bin/bash
# Rerun walk_queue.sh's post-training steps after grut_points.py crashed on a view with too few points
# (now skipped), then walk_h1 and walk_tri in order. Log /content/walk_fix.log, ends WALKFIX_DONE.
while ! grep -q WALK_DONE /content/walk_queue.log 2>/dev/null; do sleep 20; done
while pgrep -f grut_work/walk_dbg >/dev/null; do sleep 20; done  # the diagnostic run already fuses the points
set -x
WK=/content/walk/full_warehouse; R=/content/walk_rand; W=/content/grut_work/walk; NAME=walk
cd /content/3dgrut && . .venv/bin/activate
CKPT=$(ls -t /content/runs/$NAME/*/ckpt_last.pt | head -1)
if [ -s /content/grut_work/walk_dbg/fused_xyz.ply ]; then cp /content/grut_work/walk_dbg/fused*.ply $W/
else python /content/code/grut_points.py $CKPT $W 2>&1 | tail -3; fi
python /content/code/eth3d_prep.py $R $W sparse/0 2>&1 | tail -3
for f in fused_xyz.ply frames/mesh.ply; do
  python /content/code/score_robot.py $W/$f $WK $W/score_$(basename $f .ply).json --poses $R/sparse/0/images.txt
done
python /content/code/score_robot.py $W/frames/mesh.ply $WK/gt_mesh.ply $W/score_mesh_vs_gtmesh.json --poses $R/sparse/0/images.txt
[ -f $W/${NAME}_nurec.usdz ] || python -m threedgrut.export.scripts.export_usd -c $CKPT -o $W/${NAME}_nurec.usdz --format nurec --no-transform 2>&1 | tail -5
python /content/code/grut_collider.py $W/${NAME}_nurec.usdz $W/frames/mesh.ply $W/mesh_usdframe.ply
python -m threedgrut.export.scripts.add_mesh_to_usdz --input_usdz $W/${NAME}_nurec.usdz \
  --output_usdz $W/${NAME}_nurec_collider.usdz --mesh_ply $W/mesh_usdframe.ply --set_collision --set_invisible
ls -la $W; deactivate
mv /content/walk_h1.log /content/walk_h1_fail1.log; mv /content/walk_tri.log /content/walk_tri_fail1.log
nohup bash /content/walk_tri.sh > /content/walk_tri.log 2>&1 &  # tri_init on CPU now, trains after WALKH1_DONE
bash /content/walk_h1.sh > /content/walk_h1.log 2>&1
echo WALKFIX_DONE
