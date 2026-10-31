#!/bin/bash
# After walk_h1: the same warehouse capture, but 3DGUT starts from SfM points triangulated on the exact poses
# (tri_init.py: SIFT, sequential matching, COLMAP point_triangulator) instead of random points. Head to head with
# walk_queue.sh's random init on the same scorer. Usage: nohup bash walk_tri.sh > /content/walk_tri.log 2>&1 & (ends TRI_DONE)
while ! grep -q WALK_DONE /content/walk_queue.log 2>/dev/null; do sleep 30; done
set -x
WK=/content/walk/full_warehouse; R=/content/walk_tri; W=/content/grut_work/walk_tri; NAME=walk_tri
[ -f $WK/test.txt ] || { echo "no capture"; echo TRI_DONE; exit 1; }
rm -rf $W /content/runs/$NAME; mkdir -p $W
t=$(date +%s); /content/pcm/bin/python /content/code/tri_init.py $WK $R 2>&1 | grep "\[tri\]\|Error\|error" | tail -5
echo "TIME tri $(( $(date +%s) - t ))" | tee -a $W/timing.txt
[ -s $R/sparse/0/points3D.txt ] || { echo TRI_DONE; exit 1; }
while ! grep -q WALKH1_DONE /content/walk_h1.log 2>/dev/null; do sleep 30; done  # GPU free
cd /content/3dgrut && . .venv/bin/activate
t=$(date +%s)
python train.py --config-name apps/colmap_3dgut_mcmc.yaml path=$R out_dir=/content/runs experiment_name=$NAME \
  dataset.downsample_factor=1 dataset.load_exif=false n_iterations=30000 num_workers=8 \
  "checkpoint.iterations=[30000]" export_usd.enabled=true 2>&1 | grep -v "it/s\]" | tail -40
echo "TIME train $(( $(date +%s) - t ))" | tee -a $W/timing.txt
set -o pipefail; CKPT=$(ls -t /content/runs/$NAME/*/ckpt_last.pt | head -1); set +o pipefail; echo "CKPT $CKPT"
[ -z "$CKPT" ] && { echo TRI_DONE; exit 1; }
python render.py --checkpoint $CKPT --out-dir $(dirname $CKPT)/render 2>&1 | grep -i "psnr\|ssim\|lpips" | tail -4 | tee -a $W/metrics_$NAME.txt
t=$(date +%s); python /content/code/grut_points.py $CKPT $W 2>&1 | tail -3; echo "TIME points $(( $(date +%s) - t ))" | tee -a $W/timing.txt
python /content/code/eth3d_prep.py $R $W sparse/0 2>&1 | tail -3
for f in fused_xyz.ply frames/mesh.ply; do
  python /content/code/score_robot.py $W/$f $WK $W/score_$(basename $f .ply).json --poses $R/sparse/0/images.txt
done
python -m threedgrut.export.scripts.export_usd -c $CKPT -o $W/${NAME}_nurec.usdz --format nurec --no-transform 2>&1 | tail -5
python /content/code/grut_collider.py $W/${NAME}_nurec.usdz $W/frames/mesh.ply $W/mesh_usdframe.ply
python -m threedgrut.export.scripts.add_mesh_to_usdz --input_usdz $W/${NAME}_nurec.usdz \
  --output_usdz $W/${NAME}_nurec_collider.usdz --mesh_ply $W/mesh_usdframe.ply --set_collision --set_invisible
ls -la $W
echo TRI_DONE
