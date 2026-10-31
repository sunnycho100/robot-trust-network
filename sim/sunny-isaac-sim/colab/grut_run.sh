#!/bin/bash
# Colab B step 2: 3DGUT (MCMC) on ETH3D delivery_area at 2069 px (3DGRUT needs an exact integer downsample of 6208x4135) with ETH3D's own poses (laser-scan frame),
# 38 train and 6 test views (every 8th). Exports a ParticleField USDZ, fuses rendered depth into points,
# meshes them with eth3d_prep.py, scores against the laser scan, and packs the mesh as a collider into the USDZ.
# Usage: bash grut_run.sh ITERS NAME
ITERS=$1; NAME=$2
set -x
D=/content/eth3d/delivery_area; G=/content/grut_da; W=/content/grut_work/$NAME
CODE=/content/code; EVAL=/content/multi-view-evaluation/build/ETH3DMultiViewEvaluation
mkdir -p $W
clock() { echo "TIME $1 $(( $(date +%s) - $2 ))" | tee -a $W/timing.txt; }
cd /content/3dgrut && . .venv/bin/activate
if [ ! -d $G/images_3 ]; then
  mkdir -p $G/sparse/0 $G/images_3/dslr_images_undistorted
  cp $D/dslr_calibration_undistorted/*.txt $G/sparse/0/
  python - <<'PY'
from pathlib import Path
from PIL import Image
src, dst = Path("/content/eth3d/delivery_area/images/dslr_images_undistorted"), Path("/content/grut_da/images_3/dslr_images_undistorted")
for p in sorted(src.glob("*.JPG")):
    im = Image.open(p)
    im.resize((round(im.width / 3), round(im.height / 3)), Image.LANCZOS).save(dst / p.name, quality=95)
print("resized", len(list(dst.glob("*.JPG"))))
PY
fi
t=$(date +%s)
python train.py --config-name apps/colmap_3dgut_mcmc.yaml path=$G out_dir=/content/runs experiment_name=$NAME \
  dataset.downsample_factor=3 dataset.load_exif=false n_iterations=$ITERS num_workers=8 \
  "checkpoint.iterations=[$ITERS]" export_usd.enabled=true 2>&1 | grep -v "it/s\]" | tail -60
clock train $t
nvidia-smi --query-gpu=memory.used --format=csv
CKPT=$(ls -t /content/runs/*/ckpt_last.pt /content/runs/*/*/ckpt_last.pt 2>/dev/null | head -1); echo "CKPT $CKPT"
[ -z "$CKPT" ] && { echo "no checkpoint"; echo RUN_DONE; exit 1; }
find /content/runs -newer $G/sparse/0/images.txt \( -name "*.usdz" -o -name "*.usd*" \) | head
t=$(date +%s); python $CODE/grut_points.py $CKPT $W; clock points $t
t=$(date +%s); python $CODE/eth3d_prep.py $D $W; clock mesh $t
for name in fused_xyz mesh_points; do
  [ -x $EVAL ] && $EVAL --reconstruction_ply_path $W/$name.ply --ground_truth_mlp_path $D/dslr_scan_eval/scan_alignment.mlp \
    --tolerances 0.02,0.05 | tee $W/eval_$name.txt | tail -3
done
USDZ=$(find $(dirname $CKPT) -name "*.usdz" | head -1)
[ -n "$USDZ" ] && python -m threedgrut.export.scripts.add_mesh_to_usdz --input_usdz $USDZ --output_usdz $W/${NAME}_collider.usdz \
  --mesh_ply $W/frames/mesh.ply --set_collision --set_invisible
ls -la $W
echo RUN_DONE
