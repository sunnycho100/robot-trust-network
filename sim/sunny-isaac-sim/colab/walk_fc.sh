#!/bin/bash
# Splat-only collider: 3DGUT depth clipped at the SfM floor, 2 cm TSDF, unfiltered (collider_audit.py: the 2000
# triangle filter deletes whole obstacles), packed into the same NuRec usdz; then the H1 and Go2 walk as in walk_pm.sh.
# Prints the critic's pm_walk bar for this mesh first; walks either way.
# Usage: nohup bash walk_fc.sh > /content/walk_fc.log 2>&1 &   (ends WALKF_DONE; run log /content/isaac_run_walkh1f.log)
until grep -q WALKP_DONE /content/walk_pm.log 2>/dev/null; do sleep 30; done  # one Isaac Sim at a time
set -ex
W=/content/grut_work/walk
/content/o3d19/bin/python - <<'PY'
import json
s = json.load(open("/content/grut_work/walk/score_fclip_tsdf.json"))
bar = {"floor_err_p95_m": ("<=", 0.02), "floor_holes": ("<=", 0.15), "free_iou": (">=", 0.85),
       "false_free": ("<=", 0.02), "obstacle_recall": (">=", 0.85)}
for k, (op, v) in bar.items():
    ok = s[k] is not None and (s[k] <= v if op == "<=" else s[k] >= v)
    print(f"BAR {k} {s[k]} {op} {v} {'PASS' if ok else 'FAIL'}")
PY
cd /content/3dgrut && . .venv/bin/activate
python /content/code/grut_collider.py $W/walk_nurec.usdz $W/fclip_tsdf_mesh.ply $W/mesh_fc_usdframe.ply
python -m threedgrut.export.scripts.add_mesh_to_usdz --input_usdz $W/walk_nurec.usdz \
  --output_usdz $W/walk_fc_collider.usdz --mesh_ply $W/mesh_fc_usdframe.ply --set_collision --set_invisible
deactivate
sed -e 's/walk_nurec_collider.usdz/walk_fc_collider.usdz/g; s/walk_world.usda/walk_fc_world.usda/g' \
    -e 's/walkh1 40/walkh1f 40/; s/isaac_run_walkh1.log/isaac_run_walkh1f.log/g; s/navmap_walkh1/navmap_walkh1f/' \
    /content/walk_h1.sh > /content/walk_h1f.sh
bash /content/walk_h1f.sh
echo WALKF_DONE
