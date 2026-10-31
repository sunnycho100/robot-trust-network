#!/bin/bash
# After the plan recording: synthetic route end to end. Capture the stock full warehouse (240 frames, exact poses,
# GT mesh), train 3DGUT MCMC from RANDOM points (the capture's points3D.txt is GT depth, which would leak geometry),
# fuse depth, mesh, score with score_robot.py against the GT mesh, export the NuRec USDZ plus collider.
# Usage: nohup bash walk_queue.sh > /content/walk_queue.log 2>&1 &   (ends WALK_DONE)
while ! grep -q PLAN_DONE /content/plan_queue.log 2>/dev/null; do sleep 30; done
set -x
mkdir -p /usr/share/vulkan/icd.d  # missing on some Colab VMs (the echo then fails and Vulkan finds no driver)
echo '{"file_format_version":"1.0.1","ICD":{"library_path":"/usr/lib64-nvidia/libEGL_nvidia.so.0","api_version":"1.4.312"}}' > /usr/share/vulkan/icd.d/nvidia_icd.json
export MPLBACKEND=Agg OMNI_KIT_ACCEPT_EULA=YES XDG_RUNTIME_DIR=/tmp/xdg
mkdir -p /tmp/xdg
WK=/content/walk/full_warehouse; R=/content/walk_rand; W=/content/grut_work/walk; NAME=walk
rm -rf $WK $R $W /content/runs/$NAME; mkdir -p $W  # never score a stale capture (critic 9)
md5sum /content/examples/capture_walkthrough.py
t=$(date +%s)
(cd /content/isim/lib/python3.12/site-packages/isaacsim && LD_LIBRARY_PATH=/usr/lib64-nvidia:$LD_LIBRARY_PATH \
  timeout 2400 /content/isim/bin/python /content/examples/capture_walkthrough.py $WK --frames 240 > /content/walk.log 2>&1)
echo "walk exit $?"; echo "TIME capture $(( $(date +%s) - t ))" | tee -a $W/timing.txt
ls $WK/images | wc -l; grep -q "\[walk\] done" /content/walk.log || { tail -20 /content/walk.log; echo WALK_DONE; exit 1; }
mkdir -p $R/sparse/0 && ln -sfn $WK/images $R/images && cp $WK/sparse/0/cameras.txt $WK/sparse/0/images.txt $R/sparse/0/
cd /content/3dgrut && . .venv/bin/activate
python - <<'PY'
# uninformed init: 100k points uniform in the GT's 2nd to 98th percentile box (critic 7: the camera box leaves out
# the walls). Knowing the room's extent is fair: a real capture's SfM points give it. 3DGS-MCMC trains from random init.
import numpy as np, open3d as o3d
v = np.asarray(o3d.io.read_triangle_mesh("/content/walk/full_warehouse/gt_mesh.ply").vertices)
lo, hi = np.percentile(v, 2, axis=0), np.percentile(v, 98, axis=0)
p = np.random.default_rng(0).uniform(lo, hi, (100_000, 3))
with open("/content/walk_rand/sparse/0/points3D.txt", "w") as f:
    for i, (x, y, z) in enumerate(p, 1):
        f.write(f"{i} {x:.4f} {y:.4f} {z:.4f} 128 128 128 0\n")
print("random init box", lo.round(2), hi.round(2))
PY
t=$(date +%s)
python train.py --config-name apps/colmap_3dgut_mcmc.yaml path=$R out_dir=/content/runs experiment_name=$NAME \
  dataset.downsample_factor=1 dataset.load_exif=false n_iterations=30000 num_workers=8 \
  "checkpoint.iterations=[30000]" export_usd.enabled=true 2>&1 | grep -v "it/s\]" | tail -40
echo "TIME train $(( $(date +%s) - t ))" | tee -a $W/timing.txt
set -o pipefail; CKPT=$(ls -t /content/runs/$NAME/*/ckpt_last.pt | head -1); set +o pipefail; echo "CKPT $CKPT"
[ -z "$CKPT" ] && { echo WALK_DONE; exit 1; }
# held-out PSNR, SSIM, LPIPS (every 8th frame, same split as capture's test.txt), plus the ETH3D run's for the report
for c in $CKPT $(ls -t /content/runs/da_full/*/ckpt_last.pt | head -1); do
  python render.py --checkpoint $c --out-dir $(dirname $c)/render 2>&1 | grep -i "psnr\|ssim\|lpips" | tail -4 | tee -a $W/metrics_$(basename $(dirname $(dirname $c))).txt
done
t=$(date +%s); python /content/code/grut_points.py $CKPT $W 2>&1 | tail -3; echo "TIME points $(( $(date +%s) - t ))" | tee -a $W/timing.txt
python /content/code/eth3d_prep.py $R $W sparse/0 2>&1 | tail -3
# GT = the training frames' depth (only what the cameras saw, critic 3); the mesh GT once for reference
for f in fused_xyz.ply frames/mesh.ply; do
  python /content/code/score_robot.py $W/$f $WK $W/score_$(basename $f .ply).json --poses $R/sparse/0/images.txt
done
python /content/code/score_robot.py $W/frames/mesh.ply $WK/gt_mesh.ply $W/score_mesh_vs_gtmesh.json --poses $R/sparse/0/images.txt
# oracle (critic 8): GT depth through the same Poisson mesher, no training. Checks the scorer and gives the mesher's ceiling
O=/content/grut_work/walk_oracle; rm -rf $O; mkdir -p $O
python /content/code/score_robot.py oracle $WK $O/fused.ply && python /content/code/eth3d_prep.py $R $O sparse/0 2>&1 | tail -2
python /content/code/score_robot.py $O/fused.ply $WK $W/score_oracle_points.json --poses $R/sparse/0/images.txt
python /content/code/score_robot.py $O/frames/mesh.ply $WK $W/score_oracle_mesh.json --poses $R/sparse/0/images.txt
python -m threedgrut.export.scripts.export_usd -c $CKPT -o $W/${NAME}_nurec.usdz --format nurec --no-transform 2>&1 | tail -5
python /content/code/grut_collider.py $W/${NAME}_nurec.usdz $W/frames/mesh.ply $W/mesh_usdframe.ply
python -m threedgrut.export.scripts.add_mesh_to_usdz --input_usdz $W/${NAME}_nurec.usdz \
  --output_usdz $W/${NAME}_nurec_collider.usdz --mesh_ply $W/mesh_usdframe.ply --set_collision --set_invisible
ls -la $W
echo WALK_DONE
