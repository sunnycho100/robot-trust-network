#!/bin/bash
# After grut_run.sh: NuRec USDZ (the format Isaac Sim 6.2 rendered for xgrid-wormhole) plus the mesh as an
# invisible collider in the same frame. grut_run.sh's own collider step targets the standard ParticleField export,
# which add_mesh_to_usdz does not support and whose transform it ignores.
# Usage: nohup bash grut_post.sh NAME > /content/grut_post.log 2>&1 &
NAME=$1; W=/content/grut_work/$NAME
until grep -q RUN_DONE /content/grut_full.log; do sleep 30; done
set -x
cd /content/3dgrut && . .venv/bin/activate
CKPT=$(ls -t /content/runs/$NAME/*/ckpt_last.pt | head -1)
t=$(date +%s)
python -m threedgrut.export.scripts.export_usd -c $CKPT -o $W/${NAME}_nurec.usdz --format nurec --no-transform 2>&1 | tail -15
python /content/code/grut_collider.py $W/${NAME}_nurec.usdz $W/frames/mesh.ply $W/mesh_usdframe.ply
python -m threedgrut.export.scripts.add_mesh_to_usdz --input_usdz $W/${NAME}_nurec.usdz \
  --output_usdz $W/${NAME}_nurec_collider.usdz --mesh_ply $W/mesh_usdframe.ply --set_collision --set_invisible
echo "TIME post $(( $(date +%s) - t ))" | tee -a $W/timing.txt
ls -la $W
echo POST_DONE
