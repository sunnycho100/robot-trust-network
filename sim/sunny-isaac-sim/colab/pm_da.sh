#!/bin/bash
# PatchMatch on ETH3D delivery_area at 3200 px (run_eth3d.sh steps 1 to 3, reconB paths), then the hybrid:
# 3DGUT NuRec looks plus the PatchMatch mesh as the invisible collider, in the same frame.
# Usage: nohup bash pm_da.sh > /content/pm_da.log 2>&1 &   (ends PM_DONE)
set -x
COLMAP=$(command -v colmap || echo /content/colmap/build/src/colmap/exe/colmap)
D=/content/eth3d/delivery_area; W=/content/work/pm_da; G=/content/grut_work/da_full
EVAL=/content/multi-view-evaluation/build/ETH3DMultiViewEvaluation
mkdir -p $W; t=$(date +%s)
if [ ! -f $W/fused.ply ]; then
$COLMAP image_undistorter --image_path $D/images --input_path $D/dslr_calibration_undistorted \
  --output_path $W/dense --output_type COLMAP --max_image_size 3200
$COLMAP patch_match_stereo --workspace_path $W/dense --workspace_format COLMAP --PatchMatchStereo.geom_consistency true \
  --PatchMatchStereo.max_image_size 3200 --PatchMatchStereo.window_radius 7 --PatchMatchStereo.num_iterations 7 2>&1 | grep -v "^Processing\|^  " | tail -20
$COLMAP stereo_fusion --workspace_path $W/dense --workspace_format COLMAP --input_type geometric --output_path $W/fused.ply \
  --StereoFusion.min_num_pixels 3 --StereoFusion.max_image_size 3200 2>&1 | tail -5
fi
echo "TIME patchmatch $(( $(date +%s) - t ))" | tee -a $W/timing.txt
python /content/code/eth3d_prep.py $D $W 2>&1 | tail -3
for name in fused mesh_points; do
  $EVAL --reconstruction_ply_path $W/$name.ply --ground_truth_mlp_path $D/dslr_scan_eval/scan_alignment.mlp \
    --tolerances 0.02,0.05 > $W/eval_$name.txt 2>&1; tail -4 $W/eval_$name.txt
done
cd /content/3dgrut && . .venv/bin/activate
python /content/code/grut_collider.py $G/da_full_nurec.usdz $W/frames/mesh.ply $W/mesh_usdframe.ply
python -m threedgrut.export.scripts.add_mesh_to_usdz --input_usdz $G/da_full_nurec.usdz \
  --output_usdz $W/da_full_hybrid.usdz --mesh_ply $W/mesh_usdframe.ply --set_collision --set_invisible
ls -la $W
echo PM_DONE
