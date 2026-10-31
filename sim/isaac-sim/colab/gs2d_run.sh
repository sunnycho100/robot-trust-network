#!/bin/bash
# Critic action 2: gsplat 2DGS (Apache 2.0) on ETH3D delivery_area against 3DGUT da_full. Same 2069 px images,
# ETH3D poses and sparse points, every 8th view held out. 30k steps with normal and distortion losses,
# then median depth -> 1 cm TSDF -> ETH3D eval. Bar: F1 at 5 cm >= 0.75 and PSNR >= 22.04.
# Runs in the system python (gsplat built there). Started by gs2d_queue.sh.
# Usage: bash gs2d_run.sh [STEPS] [DIST_LAMBDA] [TRAINER FLAGS...]   (ends GS2D_DONE)
# gsplat's dist_lambda 1e-2 is tuned for the normalized world; the distortion is linear in depth and this run is
# metric (6.37 vs 2.43 scene scale), so 1e-2 diverged at step 3000 (L1 0.026 -> 0.4). 0.0038 is the same weight.
# 0.0038 diverged at the same step with the same curve, so the trigger may be the step-3000 opacity reset instead.
set -x
N=${1:-20000}; L=${2:-0.0038}; K=$((N-1))
D=/content/eth3d/delivery_area; G=/content/gs_da; W=/content/gs2d_work
EVAL=/content/multi-view-evaluation/build/ETH3DMultiViewEvaluation
mkdir -p $G/sparse $W
[ -e $G/sparse/0 ] || cp -r /content/grut_da/sparse/0 $G/sparse/0
[ -e $G/images ] || ln -s $D/images $G/images
[ -e $G/images_3 ] || ln -s /content/grut_da/images_3 $G/images_3
# gsplat resizes the full JPEGs to PNG but does not create the subfolder in the image names
mkdir -p $G/images_3_png/dslr_images_undistorted
cd /content/gsplat/examples
t=$(date +%s)
python3 simple_trainer_2dgs.py --data_dir $G --data_factor 3 --result_dir $W --no-normalize_world_space --disable_viewer \
  --normal_loss --dist_loss --dist_lambda $L --max_steps $N --eval_steps $N --save_steps $N "${@:3}" 2>&1 | grep -v "it/s\]" | tail -30
echo "TIME train $(( $(date +%s) - t ))" | tee -a $W/timing.txt
cat $W/stats/val_step$K.json
t=$(date +%s)
python3 /content/code/gs2d_depth.py render $W/ckpts/ckpt_$K.pt $G $W 2>&1 | grep -v "^view" | tail -8
/content/o3d19/bin/python /content/code/gs2d_depth.py fuse $W 0.01
echo "TIME tsdf $(( $(date +%s) - t ))" | tee -a $W/timing.txt
$EVAL --reconstruction_ply_path $W/mesh_points.ply --ground_truth_mlp_path $D/dslr_scan_eval/scan_alignment.mlp \
  --tolerances 0.02,0.05 > $W/eval_mesh_points.txt 2>&1; tail -4 $W/eval_mesh_points.txt
ls -la $W
echo GS2D_DONE
