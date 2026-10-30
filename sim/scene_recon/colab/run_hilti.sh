#!/bin/bash
# Hilti 2022 construction site (exp04): LiDAR map and camera MVS on LiDAR poses, both scored against the laser scan.
# Usage: bash run_hilti.sh [SEQUENCE]   (data in /content/hilti from Hugging Face Hilti-Research/hilti-slam-challenge-2022)
set -eu
R=/content/hilti
S=${1:-exp04_construction_upper_level}
W=/content/work/hilti_$S
CODE=/content/robosim/sim/scene_recon
python $CODE/colab/hilti_run.py $R $S $W prepare
if [ ! -f $W/db.db ]; then
  colmap feature_extractor --database_path $W/db.db --image_path $W/images --ImageReader.single_camera_per_image 1 \
    --ImageReader.camera_model PINHOLE
  colmap sequential_matcher --database_path $W/db.db --SequentialMatching.overlap 20
fi
python $CODE/colab/hilti_run.py $R $S $W model
rm -f $W/triangulation.txt
for m in $W/sparse_*; do
  mkdir -p $m/tri
  [ -f $m/tri/points3D.bin ] || colmap point_triangulator --database_path $W/db.db --image_path $W/images --input_path $m \
    --output_path $m/tri > $m/tri.log 2>&1
  err=$(colmap model_analyzer --path $m/tri 2>&1 | grep -i "mean reprojection error" | grep -o "[0-9.]*px" | tr -d px)
  pts=$(colmap model_analyzer --path $m/tri 2>&1 | grep -i "points:" | head -1 | grep -o "[0-9]*$")
  echo "$(basename $m) ${err:-99} ${pts:-0}" | tee -a $W/triangulation.txt
done
# lowest reprojection error among the offsets that keep at least half the most points (bad poses shed points)
best=$W/$(python -c "
rows = [l.split() for l in open('$W/triangulation.txt')]
most = max(int(r[2]) for r in rows)
print(min((float(r[1]), r[0]) for r in rows if int(r[2]) >= most / 2)[1])")
echo "== poses: $(basename $best)"
until ! pgrep -f "patch_match_stereo" > /dev/null; do sleep 60; done  # one PatchMatch at a time on the GPU
colmap image_undistorter --image_path $W/images --input_path $best/tri --output_path $W/dense --output_type COLMAP
colmap patch_match_stereo --workspace_path $W/dense --workspace_format COLMAP --PatchMatchStereo.geom_consistency true \
  --PatchMatchStereo.window_radius 7 --PatchMatchStereo.num_iterations 7
colmap stereo_fusion --workspace_path $W/dense --workspace_format COLMAP --input_type geometric --output_path $W/fused_map.ply \
  --StereoFusion.min_num_pixels 3
ln -sf $W/fused_map.ply $W/fused.ply
python $CODE/colab/hilti_run.py $R $S $W score
echo "== done"
