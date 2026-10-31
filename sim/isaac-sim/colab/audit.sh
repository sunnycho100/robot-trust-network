#!/bin/bash
# critic_floor_clip actions 1, 2, 3, 6 on the warehouse splat collider. Ends AUDIT_DONE in /content/audit.log.
set -ex
W=/content/grut_work/walk; PY=/content/o3d19/bin/python
cd /content/code
nice $PY collider_audit.py $W/fclip_tsdf_mesh.ply $W/fclip_k2000.ply /content/walk/full_warehouse \
  /content/walk_rand/sparse/0/images.txt /content/walk_tri/sparse/0/points3D.txt $W/audit_fclip.json
echo AUDIT_DONE
