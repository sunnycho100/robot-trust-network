#!/bin/bash
# Critic hybrid-walk action 9: TSDF collider from the delivery_area PatchMatch GEOMETRIC depth (3200 px, ETH3D poses)
# against the Poisson collider we walked on. Same scoring as Poisson: ETH3D F1 (eval_mesh_points.txt) and the laser
# ray check (splat_vs_collider_laser.json). Bar (critic section D): F1 at 5 cm within 0.02 of Poisson, collider
# within 5 cm of the laser on more pixels than Poisson's 79%. CPU apart from the last render, so it runs alongside Isaac.
# Usage: setsid nohup bash da_tsdf.sh > /content/da_tsdf.log 2>&1 &   (ends DATSDF_DONE; out in /content/work/pm_da/tsdf)
set -x
W=/content/work/pm_da; C=$W/tsdf; PY=/content/o3d19/bin/python
COLMAP=$(command -v colmap || echo /content/colmap/build/src/colmap/exe/colmap)
EVAL=/content/multi-view-evaluation/build/ETH3DMultiViewEvaluation
mkdir -p $C/sparse/0 $C/images $C/depth
$COLMAP model_converter --input_path $W/dense/sparse --output_path $C/sparse/0 --output_type TXT
# capture-style folder for tsdf_mesh.py: flat PNG names (images.txt names carry a subfolder and .JPG, and Open3D
# picks the reader from the extension), color and uint16 mm depth under the same name
(cd /content/code && $PY - <<'PY'
import re, numpy as np
from pathlib import Path
from PIL import Image
from colmap_depth import read_array
W, C = Path("/content/work/pm_da"), Path("/content/work/pm_da/tsdf")
txt = (C / "sparse/0/images.txt").read_text().splitlines()
out, n = [], 0
for i, l in enumerate(txt):
    f = l.split()
    if l.startswith("#") or len(f) < 10 or not f[-1].lower().endswith(".jpg"):
        out.append(l); continue
    name = f[-1]; flat = Path(name).stem + ".png"
    z = np.nan_to_num(read_array(W / f"dense/stereo/depth_maps/{name}.geometric.bin")).clip(0, 60)
    rgb = Image.open(W / f"dense/images/{name}").convert("RGB")
    assert rgb.size == z.shape[::-1], (rgb.size, z.shape)
    rgb.save(C / "images" / flat); Image.fromarray((z * 1000).round().astype(np.uint16)).save(C / "depth" / flat)
    out.append(" ".join(f[:-1] + [flat])); n += 1
(C / "sparse/0/images.txt").write_text("\n".join(out) + "\n"); (C / "test.txt").write_text("")
print(n, "views,", "valid depth share of last view", float((z > 0).mean()))
PY
)
head -4 $C/sparse/0/cameras.txt
t=$(date +%s)
(cd /content/code && $PY tsdf_mesh.py $C $C/tsdf_mesh.ply 0.02)
echo "TIME tsdf $(( $(date +%s) - t ))"
(cd /content/code && $PY -c "
import numpy as np, score_robot as sr
p = sr.load_points('$C/tsdf_mesh.ply', 0.01).astype(np.float32)  # the ETH3D evaluator scores Open3D's double ply as 0
f = open('$C/mesh_points.ply', 'wb'); f.write(f'ply\\nformat binary_little_endian 1.0\\nelement vertex {len(p)}\\nproperty float x\\nproperty float y\\nproperty float z\\nend_header\\n'.encode())
f.write(p.tobytes()); print(len(p), 'points')")
$EVAL --reconstruction_ply_path $C/mesh_points.ply --ground_truth_mlp_path /content/eth3d/delivery_area/dslr_scan_eval/scan_alignment.mlp \
  --tolerances 0.02,0.05 > $C/eval_mesh_points.txt 2>&1
echo "POISSON"; tail -4 $W/eval_mesh_points.txt; echo "TSDF"; tail -4 $C/eval_mesh_points.txt
cd /content/3dgrut && . .venv/bin/activate
CKPT=$(ls -t /content/runs/da_full/*/ckpt_last.pt | head -1)
nice python /content/code/splat_vs_collider.py $CKPT $C/tsdf_mesh.ply $C/splat_vs_tsdf_laser.json \
  $W/fused.ply /content/eth3d/delivery_area/dslr_scan_eval 2>&1 | tail -12
echo DATSDF_DONE
