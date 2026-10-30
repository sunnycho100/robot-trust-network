#!/bin/bash
# Same scene, realistic capture: ETH3D's handheld 4-camera rig video of delivery_area (948 grayscale 0.4 MP
# frames) on its given poses, scored on the same laser scans as the DSLR run. The F1 gap is the capture alone.
# Usage: bash run_rig.sh [defaults]   (defaults: COLMAP's own PatchMatch and fusion settings instead of the ones
# tuned on the DSLR photos, to check the gap isn't our tuning; waits for the GPU)
set -eu
D=/content/eth3d/delivery_area
MODE=${1:-tuned}
W=/content/work/delivery_area_rig$([ $MODE = defaults ] && echo _defaults)
PM="--PatchMatchStereo.window_radius 7 --PatchMatchStereo.num_iterations 7"; FU="--StereoFusion.min_num_pixels 3"
[ $MODE = defaults ] && PM= && FU=
CODE=/content/robosim/sim/sunny-scene-recon
EVAL=/content/multi-view-evaluation/build/ETH3DMultiViewEvaluation
mkdir -p $W
if [ ! -d $D/rig_calibration_undistorted ]; then
  cd /content/eth3d && wget -q https://www.eth3d.net/data/delivery_area_rig_undistorted.7z https://www.eth3d.net/data/delivery_area_rig_scan_eval.7z
  7z x -y delivery_area_rig_undistorted.7z > /dev/null && 7z x -y delivery_area_rig_scan_eval.7z > /dev/null
fi
until ! pgrep -f "patch_match_stereo|stereo_fusion" > /dev/null; do sleep 60; done  # one PatchMatch at a time on the GPU
t=$(date +%s)
colmap image_undistorter --image_path $D/images --input_path $D/rig_calibration_undistorted --output_path $W/dense --output_type COLMAP
colmap patch_match_stereo --workspace_path $W/dense --workspace_format COLMAP --PatchMatchStereo.geom_consistency true $PM
colmap stereo_fusion --workspace_path $W/dense --workspace_format COLMAP --input_type geometric --output_path $W/fused.ply $FU
echo "TIME rig_dense $(( $(date +%s) - t ))"
python $CODE/colab/eth3d_prep.py $D $W rig_calibration_undistorted
for name in fused mesh_points; do
  echo "-- $name"
  $EVAL --reconstruction_ply_path $W/$name.ply --ground_truth_mlp_path $D/rig_scan_eval/scan_alignment.mlp \
    --tolerances 0.02,0.05,0.1 | tee $W/eval_$name.txt | tail -3
done
echo "== done"
