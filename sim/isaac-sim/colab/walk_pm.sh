#!/bin/bash
# The pipeline we would sell, on the synthetic warehouse: 3DGUT splat for looks plus the PatchMatch TSDF mesh
# (pm_walk.sh) as the invisible collider, then the H1 and Go2 walk the same planned paths as walk_h1 and walk_tsdf.
# Prints the critic's pm_walk bar first; walks either way (a failed bar is still a useful video).
# Usage: nohup bash walk_pm.sh > /content/walk_pm.log 2>&1 &   (ends WALKP_DONE; run log /content/isaac_run_walkh1p.log)
until grep -q PMW_DONE /content/pm_walk.log 2>/dev/null && grep -q WALKT_DONE /content/walk_tsdf.log 2>/dev/null; do sleep 30; done
set -ex
W=/content/grut_work/walk
/content/o3d19/bin/python - <<'PY'
import json
s = json.load(open("/content/grut_work/walk/score_pm_tsdf.json"))
bar = {"floor_err_p95_m": ("<=", 0.02), "floor_holes": ("<=", 0.15), "free_iou": (">=", 0.85),
       "false_free": ("<=", 0.02), "obstacle_recall": (">=", 0.85)}
for k, (op, v) in bar.items():
    ok = s[k] is not None and (s[k] <= v if op == "<=" else s[k] >= v)
    print(f"BAR {k} {s[k]} {op} {v} {'PASS' if ok else 'FAIL'}")
PY
cd /content/3dgrut && . .venv/bin/activate
python /content/code/grut_collider.py $W/walk_nurec.usdz $W/pm_tsdf_mesh.ply $W/mesh_pm_usdframe.ply
python -m threedgrut.export.scripts.add_mesh_to_usdz --input_usdz $W/walk_nurec.usdz \
  --output_usdz $W/walk_pm_collider.usdz --mesh_ply $W/mesh_pm_usdframe.ply --set_collision --set_invisible
deactivate
sed -e 's/walk_nurec_collider.usdz/walk_pm_collider.usdz/g; s/walk_world.usda/walk_pm_world.usda/g' \
    -e 's/walkh1 40/walkh1p 40/; s/isaac_run_walkh1.log/isaac_run_walkh1p.log/g; s/navmap_walkh1/navmap_walkh1p/' \
    /content/walk_h1.sh > /content/walk_h1p.sh
bash /content/walk_h1p.sh
echo WALKP_DONE
