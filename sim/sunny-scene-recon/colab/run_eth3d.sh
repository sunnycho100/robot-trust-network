#!/bin/bash
# Full ETH3D run on the Colab VM: dense stereo, mesh, score against the laser scan, MuJoCo scene, G1 walk test.
# Usage: bash run_eth3d.sh SCENE   (e.g. pipes; data in /content/eth3d/SCENE, code in /content/robosim)
set -eu
SCENE=${1:-pipes}
MAX_SIZE=${MAX_SIZE:-3200}  # critic round 2: higher resolution, wider window, more iterations,
MIN_PIXELS=${MIN_PIXELS:-3} # and keep surfaces seen by 3 views (the default 5 drops floor and far walls)
D=/content/eth3d/$SCENE
W=/content/work/${SCENE}_${MAX_SIZE}_${MIN_PIXELS}
CODE=/content/robosim/sim/sunny-scene-recon
EVAL=/content/multi-view-evaluation/build/ETH3DMultiViewEvaluation
mkdir -p $W
export MUJOCO_GL=egl

if [ ! -f $W/fused.ply ]; then  # dense stereo is the slow part; reruns reuse it
echo "== 1. dense stereo (COLMAP CUDA patch-match on ETH3D's metric poses)"
colmap image_undistorter --image_path $D/images --input_path $D/dslr_calibration_undistorted \
  --output_path $W/dense --output_type COLMAP --max_image_size $MAX_SIZE
colmap patch_match_stereo --workspace_path $W/dense --workspace_format COLMAP --PatchMatchStereo.geom_consistency true \
  --PatchMatchStereo.max_image_size $MAX_SIZE --PatchMatchStereo.window_radius 7 --PatchMatchStereo.num_iterations 7
colmap stereo_fusion --workspace_path $W/dense --workspace_format COLMAP --input_type geometric --output_path $W/fused.ply \
  --StereoFusion.min_num_pixels $MIN_PIXELS --StereoFusion.max_image_size $MAX_SIZE
fi

echo "== 2. mesh"
python $CODE/colab/eth3d_prep.py $D $W

echo "== 3. score against the laser scan (ETH3D frame, before leveling)"
for name in fused mesh_points; do
  echo "-- $name"
  $EVAL --reconstruction_ply_path $W/$name.ply --ground_truth_mlp_path $D/dslr_scan_eval/scan_alignment.mlp \
    --tolerances 0.02,0.05 | tee $W/eval_$name.txt | tail -4
done

echo "== 4. MuJoCo scene"
python $CODE/build_scene.py $W/frames --crop-height 2.2

echo "== 5. collision proxies against the visual mesh"
python $CODE/check_collisions.py $W/frames/mujoco

echo "== 6. G1 walk test: open and near-obstacle goals vs a bare plane, then walking into walls"
python $CODE/walk_in_scene.py $W/frames/mujoco --episodes 20 --video $W/walk.mp4
echo "== done"
