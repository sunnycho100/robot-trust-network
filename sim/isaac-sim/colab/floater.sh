#!/bin/bash
# Floater test on the floor-clipped splat TSDF: drop connected triangle clusters under N triangles (obstacles that
# stand on the floor stay joined to it), score again. Ends FLOAT_DONE in /content/floater.log.
set -ex
W=/content/grut_work/walk; WK=/content/walk/full_warehouse; PY=/content/o3d19/bin/python
cd /content/code
for N in 2000 20000; do
nice $PY - $N <<'PY'
import sys, numpy as np, open3d as o3d
n = int(sys.argv[1]); W = "/content/grut_work/walk"
m = o3d.io.read_triangle_mesh(f"{W}/fclip_tsdf_mesh.ply")
idx, count, _ = m.cluster_connected_triangles()
idx, count = np.asarray(idx), np.asarray(count)
drop = count[idx] < n
print(f"N {n}: {len(count)} clusters, largest {count.max()}, dropping {drop.sum()} of {len(idx)} triangles")
m.remove_triangles_by_mask(drop); m.remove_unreferenced_vertices()
o3d.io.write_triangle_mesh(f"{W}/fclip_k{n}.ply", m)
PY
nice $PY score_robot.py $W/fclip_k$N.ply $WK $W/score_fclip_k$N.json --poses /content/walk_rand/sparse/0/images.txt | grep -E "floor_err|holes|free_iou|false_|recall"
done
echo FLOAT_DONE
