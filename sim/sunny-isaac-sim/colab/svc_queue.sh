#!/bin/bash
# Critic item 5 on the ETH3D hybrid: splat (3DGUT da_full) vs collider (PatchMatch mesh) disagreement over 10 cm.
# Usage: nohup bash svc_queue.sh > /content/svc_queue.log 2>&1 &   (ends SVC_DONE; result /content/work/pm_da/splat_vs_collider.json)
while ! grep -q PM_DONE /content/pm_da.log 2>/dev/null; do sleep 30; done
set -x
cd /content/3dgrut && . .venv/bin/activate
CKPT=$(ls -t /content/runs/da_full/*/ckpt_last.pt | head -1)
python /content/code/splat_vs_collider.py $CKPT /content/work/pm_da/frames/mesh.ply /content/work/pm_da/splat_vs_collider.json 2>&1 | tail -12
echo SVC_DONE
