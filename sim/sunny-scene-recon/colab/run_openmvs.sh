#!/bin/bash
# OpenMVS dense stage on the same ETH3D poses (critic: higher F1 than COLMAP fusion, and texturing from
# the full-resolution photos). Reuses the COLMAP undistorted workspace made by run_eth3d.sh.
# Usage: bash run_openmvs.sh SCENE COLMAP_WORK_DIR
set -eu
SCENE=$1
CW=$2                               # e.g. /content/work/delivery_area_3200_3
D=/content/eth3d/$SCENE
W=/content/work/${SCENE}_openmvs
CODE=/content/robosim/sim/sunny-scene-recon
EVAL=/content/multi-view-evaluation/build/ETH3DMultiViewEvaluation
BIN=/usr/local/bin/OpenMVS
mkdir -p $W && cd $W
export MUJOCO_GL=egl

echo "== 1. OpenMVS: import poses, densify, mesh, refine, texture"
$BIN/InterfaceCOLMAP -i $CW/dense -o scene.mvs --image-folder $CW/dense/images
$BIN/DensifyPointCloud scene.mvs --resolution-level 1
$BIN/ReconstructMesh scene_dense.mvs
$BIN/RefineMesh scene_dense_mesh.mvs --resolution-level 1
$BIN/TextureMesh scene_dense_mesh_refine.mvs --export-type obj

echo "== 2. score against the laser scan"
python - <<'EOF'
import numpy as np, open3d as o3d
m = o3d.io.read_triangle_mesh("scene_dense_mesh_refine.ply")
n = int(min(m.get_surface_area() / 0.01 ** 2, 40e6))  # 1 cm spacing, as in eth3d_prep.py
o3d.t.io.write_point_cloud("mesh_points.ply", o3d.t.geometry.PointCloud(np.asarray(m.sample_points_uniformly(n).points, dtype=np.float32)))
p = o3d.io.read_point_cloud("scene_dense.ply")
o3d.t.io.write_point_cloud("dense_points.ply", o3d.t.geometry.PointCloud(np.asarray(p.points, dtype=np.float32)))
EOF
for name in dense_points mesh_points; do
  echo "-- $name"
  $EVAL --reconstruction_ply_path $W/$name.ply --ground_truth_mlp_path $D/dslr_scan_eval/scan_alignment.mlp \
    --tolerances 0.02,0.05 | tee $W/eval_$name.txt | tail -4
done

echo "== 3. MuJoCo scene (OpenMVS texture as the visual, refined mesh for leveling and collisions)"
mkdir -p frames && cp scene_dense_mesh_refine.ply frames/mesh.ply && cp $CW/frames/poses.txt frames/poses.txt
python $CODE/build_scene.py frames --crop-height 3.0 --visual-obj $W/scene_dense_mesh_refine_texture.obj
python $CODE/check_collisions.py frames/mujoco
python $CODE/walk_in_scene.py frames/mujoco --episodes 20 --video $W/walk.mp4
echo "== done"
