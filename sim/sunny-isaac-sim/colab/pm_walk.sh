#!/bin/bash
# The real-pipeline collider on the synthetic warehouse: COLMAP PatchMatch depth (exact poses, tri_init points for
# the depth range) -> 2 cm TSDF -> score_robot, against the oracle TSDF (GT depth, free IoU 0.957) and splat depth.
# PatchMatch sees all 240 frames; the TSDF skips the 30 held-out ones (small leak through stereo neighbours).
# Usage: nohup bash pm_walk.sh > /content/pm_walk.log 2>&1 &   (ends PMW_DONE; score /content/grut_work/walk/score_pm_tsdf.json)
until grep -q PM_DONE /content/pm_da.log 2>/dev/null && [ -s /content/walk_tri/sparse/0/points3D.txt ]; do sleep 30; done
set -ex
COLMAP=$(command -v colmap || echo /content/colmap/build/src/colmap/exe/colmap)
WK=/content/walk/full_warehouse; D=/content/work/pm_walk/dense; PY=/content/o3d19/bin/python
mkdir -p $D; t=$(date +%s)
$COLMAP image_undistorter --image_path $WK/images --input_path /content/walk_tri/sparse/0 --output_path $D \
  --output_type COLMAP --max_image_size 1280
# the known-pose model has no 2D tracks, so COLMAP's __auto__ sources and depth ranges are empty: set both from poses
(cd /content/code && $PY pm_sources.py /content/walk_tri/sparse/0/images.txt $D/stereo/patch-match.cfg 10)
$COLMAP patch_match_stereo --workspace_path $D --workspace_format COLMAP --PatchMatchStereo.geom_consistency true \
  --PatchMatchStereo.max_image_size 1280 --PatchMatchStereo.depth_min 0.2 --PatchMatchStereo.depth_max 25 2>&1 | grep -v "^Processing\|^  " | tail -5
echo "TIME patchmatch $(( $(date +%s) - t ))"
cd /content/code
$PY colmap_depth.py $D $WK /content/work/pm_walk/cap
$PY tsdf_mesh.py /content/work/pm_walk/cap /content/grut_work/walk/pm_tsdf_mesh.ply 0.02
$PY score_robot.py /content/grut_work/walk/pm_tsdf_mesh.ply $WK /content/grut_work/walk/score_pm_tsdf.json --poses $WK/sparse/0/images.txt
echo PMW_DONE
