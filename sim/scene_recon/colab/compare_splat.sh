#!/bin/bash
# Head-to-head on ETH3D delivery_area at 1920 px (vid2scene's video frame size). Both methods get the same
# GLOMAP-style poses (COLMAP SIFT + global_mapper + model_orientation_aligner) and the same 38 training views:
#   A. vid2scene's gsplat fork, its default MCMC settings       B. ours: COLMAP PatchMatch
# Both depths are fused into points and meshed with the same Poisson step, so only the depth source differs.
# Then both go through the same MuJoCo pipeline. Reference row: our 3200 px run on ETH3D's poses (run_eth3d.sh).
# Usage: bash compare_splat.sh   (needs setup_v2s.sh done and run_eth3d.sh delivery_area finished)
set -eu
D=/content/eth3d/delivery_area
V=/content/v2s/delivery_area
W=/content/work/delivery_area_3200_3
CODE=/content/robosim/sim/scene_recon
EVAL=/content/multi-view-evaluation/build/ETH3DMultiViewEvaluation
export MUJOCO_GL=egl
mkdir -p $V/images $V/sparse $V/sparse_txt $V/mvs
touch $V/timing.txt  # reruns skip finished stages and keep their times
clock() { echo "TIME $1 $(( $(date +%s) - $2 ))" | tee -a $V/timing.txt; }

echo "== 0. frames and poses (shared)"
if [ ! -f $V/sparse_txt/images.txt ]; then
t=$(date +%s)
python - <<'EOF'
from pathlib import Path
from PIL import Image
src, dst = Path("/content/eth3d/delivery_area/images/dslr_images_undistorted"), Path("/content/v2s/delivery_area/images")
for p in sorted(src.glob("*.JPG")):
    im = Image.open(p)
    s = 1920 / max(im.size)
    im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS).save(dst / p.name, quality=95)
EOF
colmap feature_extractor --database_path $V/db.db --image_path $V/images --ImageReader.camera_model PINHOLE --ImageReader.single_camera 1
colmap exhaustive_matcher --database_path $V/db.db
colmap global_mapper --database_path $V/db.db --image_path $V/images --output_path $V/sparse
mkdir -p $V/sparse/aligned
colmap model_orientation_aligner --input_path $V/sparse/0 --output_path $V/sparse/aligned --image_path $V/images
mv $V/sparse/0 $V/sparse/unaligned && mv $V/sparse/aligned $V/sparse/0
colmap model_converter --input_path $V/sparse/0 --output_path $V/sparse_txt --output_type TXT
clock sfm $t
fi

echo "== A. splats"
if ! ls $V/gs/ckpts/ckpt_*_rank0.pt > /dev/null 2>&1; then
until grep -q LIBJXL /content/setup_v2s.log; do sleep 30; done
grep -q IMPORT_OK /content/setup_v2s.log || { echo "splat stack failed to install"; exit 1; }
t=$(date +%s)
(cd /content/v2s_gsplat/examples && python simple_trainer.py mcmc --data_dir $V --data_factor 1 --result_dir $V/gs \
  --save_ply_path $V/gs/splat.ply --use_bilateral_grid --use_fused_bilagrid --disable_eval --random_bkgd \
  --mcmc_max_num_gaussians 1000000 --steps_scaler 1.0 --disable_viewer)
clock splat_train $t
fi

echo "== B. ours on the same poses and training views"
if [ ! -f $V/mvs/fused_local.ply ]; then
until grep -q RUN_EXIT /content/run_da.log; do sleep 30; done  # one PatchMatch at a time on the GPU
t=$(date +%s)
python -c "import sys; sys.path.insert(0, '$CODE/colab'); from compare_eval import test_names; from pathlib import Path; print('\n'.join(test_names(Path('$V'))))" > $V/test_names.txt
mkdir -p $V/sparse_train
colmap image_deleter --input_path $V/sparse/0 --output_path $V/sparse_train --image_names_path $V/test_names.txt
colmap image_undistorter --image_path $V/images --input_path $V/sparse_train --output_path $V/mvs/dense --output_type COLMAP --max_image_size 1920
colmap patch_match_stereo --workspace_path $V/mvs/dense --workspace_format COLMAP --PatchMatchStereo.geom_consistency true \
  --PatchMatchStereo.max_image_size 1920 --PatchMatchStereo.window_radius 7 --PatchMatchStereo.num_iterations 7
colmap stereo_fusion --workspace_path $V/mvs/dense --workspace_format COLMAP --input_type geometric --output_path $V/mvs/fused_local.ply \
  --StereoFusion.min_num_pixels 3 --StereoFusion.max_image_size 1920
clock ours_dense $t
fi

echo "== poses vs ETH3D, Sim3 into the laser-scan frame, splat depth -> points"
t=$(date +%s)
python $CODE/colab/compare_eval.py geometry $V $D
clock splat_depth $t
for M in $V/mvs $V/splat; do  # same Poisson meshing for both
  t=$(date +%s)
  python $CODE/colab/eth3d_prep.py $D $M
  clock mesh_$(basename $M) $t
done

for M in $V/mvs $V/splat; do
  echo "== $M: score, MuJoCo scene, collision check, walk"
  for name in fused_xyz mesh_points; do
    $EVAL --reconstruction_ply_path $M/$name.ply --ground_truth_mlp_path $D/dslr_scan_eval/scan_alignment.mlp \
      --tolerances 0.02,0.05 | tee $M/eval_$name.txt | tail -1
  done
  python $CODE/build_scene.py $M/frames --outdoor --crop-height 2.2 || { echo "no MuJoCo scene for $M"; continue; }
  python $CODE/check_collisions.py $M/frames/mujoco
  python $CODE/walk_in_scene.py $M/frames/mujoco --episodes 20
done

echo "== report"
python $CODE/colab/compare_eval.py report $V $D $W
echo "== done"
