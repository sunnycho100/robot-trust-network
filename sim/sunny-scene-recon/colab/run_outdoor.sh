#!/bin/bash
# delivery_area with the outdoor collision path: best mesher by F1 at 2 cm, heightfield ground, solid object hulls,
# checks, G1 walk test, then the Isaac and CooHOI export and a preview sheet.
# Usage: bash run_outdoor.sh   (after run_eth3d.sh delivery_area and mesh_ab.sh)
set -eu
W=/content/work/delivery_area_3200_3
F=$W/frames_outdoor
CODE=/content/robosim/sim/sunny-scene-recon
export MUJOCO_GL=egl
until grep -q AB_EXIT /content/mesh_ab.log; do sleep 60; done
mkdir -p $F && cp $W/frames/poses.txt $F/
best=$(python - <<EOF
from pathlib import Path
f1 = lambda p: float(next(l for l in Path(p).read_text().splitlines() if l.startswith("F1")).split()[1])
runs = {"$W/frames/mesh.ply": "$W/eval_mesh_points.txt", "$W/mesh_poisson12.ply": "$W/eval_mesh_poisson12.txt",
        "$W/mesh_delaunay.ply": "$W/eval_mesh_delaunay.txt"}
scores = {m: f1(e) for m, e in runs.items() if Path(e).exists()}
print(max(scores, key=scores.get))
EOF
)
echo "== mesh: $best"
python - <<EOF  # COLMAP's Delaunay mesh has no vertex colors: take them from the nearest fused point
import numpy as np, open3d as o3d
from scipy.spatial import cKDTree
m = o3d.io.read_triangle_mesh("$best")
if not m.has_vertex_colors():
    pts = o3d.io.read_point_cloud("$W/fused.ply")
    nearest = cKDTree(np.asarray(pts.points)).query(np.asarray(m.vertices))[1]
    m.vertex_colors = o3d.utility.Vector3dVector(np.asarray(pts.colors)[nearest])
o3d.io.write_triangle_mesh("$F/mesh.ply", m)
EOF
python $CODE/build_scene.py $F --outdoor --crop-height 2.2
python $CODE/check_collisions.py $F/mujoco /content/eth3d/delivery_area/dslr_scan_eval/scan_alignment.mlp
python $CODE/walk_in_scene.py $F/mujoco --episodes 20 --video $W/walk_outdoor.mp4
python $CODE/export_scene.py $F/mujoco --coohoi-humanoid /content/coohoi/amp_humanoid.xml
python $CODE/colab/render_preview.py $F/mujoco /content/delivery_outdoor.jpg
echo "== done"
