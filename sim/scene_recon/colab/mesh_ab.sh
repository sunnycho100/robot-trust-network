#!/bin/bash
# Which mesher keeps the 2 cm detail of the fused points on a large scene: COLMAP Poisson at depth 12, or Delaunay
# (uses the per-point visibility in fused.ply.vis). Scored like run_eth3d.sh. Usage: bash mesh_ab.sh SCENE WORK_DIR
set -eu
D=/content/eth3d/$1
W=$2
EVAL=/content/multi-view-evaluation/build/ETH3DMultiViewEvaluation
cd $W
[ -f dense/fused.ply ] || ln -sf $W/fused.ply dense/fused.ply && ln -sf $W/fused.ply.vis dense/fused.ply.vis
t=$(date +%s)
[ -f mesh_poisson12.ply ] || colmap poisson_mesher --input_path fused.ply --output_path mesh_poisson12.ply --PoissonMeshing.depth 12 \
  --PoissonMeshing.trim 7 --PoissonMeshing.num_threads 4
echo "TIME poisson12 $(( $(date +%s) - t ))"
t=$(date +%s)
[ -f mesh_delaunay.ply ] || colmap delaunay_mesher --input_path dense --input_type dense --output_path mesh_delaunay.ply
echo "TIME delaunay $(( $(date +%s) - t ))"
# frames/mesh.ply is eth3d_prep's Open3D Poisson; its score file keeps run_eth3d.sh's name
for pair in frames/mesh:mesh_points mesh_poisson12:mesh_poisson12_points mesh_delaunay:mesh_delaunay_points; do
  m=${pair%%:*}; pts=${pair##*:}
  python -c "
import numpy as np, open3d as o3d
m = o3d.io.read_triangle_mesh('$m.ply'); print('$m', len(m.triangles), 'triangles')
n = int(min(m.get_surface_area() / 0.01 ** 2, 40e6))  # 1 cm spacing, as in eth3d_prep.py
o3d.t.io.write_point_cloud('$pts.ply', o3d.t.geometry.PointCloud(np.asarray(m.sample_points_uniformly(n).points, dtype=np.float32)))"
  echo "-- $m"
  eval_name=$([ $m = frames/mesh ] && echo eval_mesh_points || echo eval_$m)
  $EVAL --reconstruction_ply_path $W/$pts.ply --ground_truth_mlp_path $D/dslr_scan_eval/scan_alignment.mlp \
    --tolerances 0.02,0.05 | tee $W/$eval_name.txt | tail -3
done
echo "== done"
