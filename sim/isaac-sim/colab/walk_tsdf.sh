#!/bin/bash
# The warehouse splat's collider from rendered splat depth through a 2 cm TSDF (Poisson closed 99.8% of the free floor,
# even on GT depth), scored, packed into the NuRec usdz, then the H1 and Go2 walk again on it after DAH1_DONE.
# Usage: nohup bash walk_tsdf.sh > /content/walk_tsdf.log 2>&1 &   (ends WALKT_DONE; run log /content/isaac_run_walkh1t.log)
while ! grep -q WALKFIX_DONE /content/walk_fix.log 2>/dev/null; do sleep 30; done
set -ex  # stop at the first failure (a failed render left junk downstream)
WK=/content/walk/full_warehouse; R=/content/walk_rand; W=/content/grut_work/walk; G=/content/grut_work/walk_gd
PY=/content/o3d19/bin/python
cd /content/3dgrut && . .venv/bin/activate
CKPT=$(ls -t /content/runs/walk/*/ckpt_last.pt | head -1)
python /content/code/grut_depth.py $CKPT $G 2>&1 | tail -2
cd /content/code
$PY tsdf_mesh.py $G $W/tsdf_mesh.ply 0.02
$PY score_robot.py $W/tsdf_mesh.ply $WK $W/score_splat_tsdf.json --poses $R/sparse/0/images.txt
cd /content/3dgrut
python /content/code/grut_collider.py $W/walk_nurec.usdz $W/tsdf_mesh.ply $W/mesh_tsdf_usdframe.ply
python -m threedgrut.export.scripts.add_mesh_to_usdz --input_usdz $W/walk_nurec.usdz \
  --output_usdz $W/walk_tsdf_collider.usdz --mesh_ply $W/mesh_tsdf_usdframe.ply --set_collision --set_invisible
deactivate
while ! grep -q DAH1_DONE /content/da_h1.log 2>/dev/null; do sleep 30; done  # one Isaac Sim at a time
sed -e 's/walk_nurec_collider.usdz/walk_tsdf_collider.usdz/g; s/walk_world.usda/walk_tsdf_world.usda/g' \
    -e 's/walkh1 40/walkh1t 40/; s/isaac_run_walkh1.log/isaac_run_walkh1t.log/g; s/navmap_walkh1/navmap_walkh1t/' \
    /content/walk_h1.sh > /content/walk_h1t.sh
bash /content/walk_h1t.sh
echo WALKT_DONE
