#!/bin/bash
# Critic hybrid-walk action 5: splat vs collider vs the ETH3D laser scan along the same rays (not circular).
# Usage: setsid nohup bash laser_check.sh > /content/laser_check.log 2>&1 &   (ends LASER_DONE; result splat_vs_collider_laser.json)
set -x
cd /content/3dgrut && . .venv/bin/activate
python /content/code/splat_vs_collider.py test
CKPT=$(ls -t /content/runs/da_full/*/ckpt_last.pt | head -1)
nice python /content/code/splat_vs_collider.py $CKPT /content/work/pm_da/frames/mesh.ply /content/work/pm_da/splat_vs_collider_laser.json \
  /content/work/pm_da/fused.ply /content/eth3d/delivery_area/dslr_scan_eval 2>&1 | tail -30
echo LASER_DONE
