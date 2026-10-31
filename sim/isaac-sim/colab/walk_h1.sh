#!/bin/bash
# After walk_queue.sh: an H1 and a Go2 walk planned paths in OUR reconstruction of the warehouse (NuRec splat plus
# the invisible collider, no ground plane), with starts and goals picked on the GT free space so a failure is the
# reconstruction's fault. A wrapper .usda undoes the exporter's normalizing transform, so the stage is back in the
# warehouse frame (metres, Z up, floor near 0) where the stock spawn heights apply.
# Usage: nohup bash walk_h1.sh > /content/walk_h1.log 2>&1 &   (ends WALKH1_DONE; log of the run: /content/isaac_run_walkh1.log)
while ! grep -q WALK_DONE /content/walk_queue.log 2>/dev/null; do sleep 30; done
set -x
W=/content/grut_work/walk; WK=/content/walk/full_warehouse
[ -f $W/walk_nurec_collider.usdz ] || { echo "no collider usdz"; echo WALKH1_DONE; exit 1; }
cd /content/3dgrut && . .venv/bin/activate
ARGS=$(python - <<'PY'
import sys
import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdVol
from scipy import ndimage
sys.path.insert(0, "/content/code")
import score_robot as sr

W, WK = "/content/grut_work/walk", "/content/walk/full_warehouse"
src = Usd.Stage.Open(f"{W}/walk_nurec_collider.usdz")
vol = next(p for p in src.Traverse() if p.IsA(UsdVol.Volume))
m = np.array(UsdGeom.Xformable(vol).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
st = Usd.Stage.CreateNew(f"{W}/walk_world.usda")
UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(st, 1.0)
x = UsdGeom.Xform.Define(st, "/Recon")
x.GetPrim().GetReferences().AddReference("./walk_nurec_collider.usdz")  # relative: copy both files together
x.AddTransformOp().Set(Gf.Matrix4d(*np.linalg.inv(m).flatten().tolist()))
st.SetDefaultPrim(x.GetPrim()); st.GetRootLayer().Save()
print("wrapper inverse of\n", np.round(m, 4), file=sys.stderr)

gt = sr.load_points(f"{WK}/gt_mesh.ply")
cams = sr.camera_centres(f"{WK}/sparse/0/images.txt")
lo, hi = cams[:, :2].min(0) - 3, cams[:, :2].max(0) + 3
shape = tuple(np.ceil((hi - lo) / sr.CELL).astype(int))
import json
zf = json.load(open(f"{W}/score_fused_xyz.json"))["z_floor"]  # the GT floor height score_robot found
free = ~np.isnan(sr.floor_map(gt, lo, shape, zf, 0.05)) & ~sr.occupied_map(gt, lo, shape, zf, (0.1, 1.8))
clear = ndimage.distance_transform_edt(free) * sr.CELL
ok = np.argwhere(clear > 1.0)
used = []
def near(p):  # nearest well-clear cell at least 2 m from points already used (the capture path revisits spots)
    xy = (ok + 0.5) * sr.CELL + lo
    far = np.all([np.hypot(*(xy - u).T) > 2.0 for u in used], 0) if used else np.ones(len(xy), bool)
    q = xy[far][np.argmin(np.hypot(*(xy[far] - p).T))]
    used.append(q)
    return q
out = []
for kind, a, b in (("h1", 0, 30), ("go2", 120, 150)):
    s, g = near(cams[a, :2]), near(cams[b, :2])
    yaw = np.degrees(np.arctan2(*(g - s)[::-1]))
    out.append((kind, s, g, yaw))
    print(f"{kind} start {s.round(2)} goal {g.round(2)} distance {np.hypot(*(g - s)):.1f} m", file=sys.stderr)
print(" ".join(f"--robot {k} {s[0]:.2f} {s[1]:.2f} {y:.0f} 0 0 0" for k, s, g, y in out),
      " ".join(f"--goal {g[0]:.2f} {g[1]:.2f}" for k, s, g, y in out))
PY
)
echo "ARGS $ARGS"
[ -z "$ARGS" ] && { echo WALKH1_DONE; exit 1; }
deactivate
cp /content/code/navmap.py /content/examples/navmap.py  # the version that reads invisible colliders
bash /content/isaac_run.sh walkh1 40 /content/examples/multi_robot_scene_plan.py --scene $W/walk_world.usda --plan $ARGS \
  > /content/isaac_run_walkh1.log 2>&1
SP=/content/isim/lib/python3.12/site-packages/isaacsim
cp $SP/fleet_navmap.npz /content/rec/navmap_walkh1.npz; cp $SP/fleet_traj.csv /content/rec/navmap_walkh1_traj.csv
grep "\[fleet\]\|\[multi\]\|frames" /content/isaac_run_walkh1.log | tail -20
echo WALKH1_DONE
