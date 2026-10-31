#!/bin/bash
# Brev L40S version of grut_setup.sh, grut_setup2.sh, grut_run.sh (training only) and grut_post.sh (NuRec export):
# 3DGRUT bb203b1 on the system CUDA 12.6 toolkit, minus Kaolin and tiny-cuda-nn, the export_usd no-transform patch,
# 3DGUT MCMC on delivery_area at 2069 px with ETH3D's poses, then a NuRec USDZ in the dataset (laser-scan) frame.
# Usage: setsid nohup bash brev_grut.sh ITERS > /data/da/grut.log 2>&1 < /dev/null &   (ends GRUT_DONE)
ITERS=${1:-30000}; NAME=da_full
set -x
export PATH=$HOME/.local/bin:/usr/local/cuda-12.6/bin:$PATH CUDA_HOME=/usr/local/cuda-12.6 TORCH_CUDA_ARCH_LIST="8.9" MAX_JOBS=8
D=/data/da/eth3d/delivery_area; G=/data/da/grut_da; R=/data/da/grut_runs; OUT=/data/da/grut_work
cd /data/da/3dgrut
if [ ! -x .venv/bin/python ]; then
  sed -i 's#^INSTALL_TCNN_WITH_UV=1 bash#echo skip tcnn \#&#; s#^if \[ "${CUDA_MAJOR_TARGET:-0}" -le 12 \]; then#if true; then#; s#^uv pip install -e .\[playground\]#uv pip install pygltflib#; s#^python -c "import kaolin.*##' install_env_uv.sh
  sed -i '/^if importlib.util.find_spec("tinycudann")/,/^    print("tiny-cuda-nn: ready")/d' install_env_uv.sh
  time bash install_env_uv.sh 3dgrut
fi
. .venv/bin/activate
python -c "import ppisp; from fused_ssim import fused_ssim" 2>/dev/null || {
  time uv pip install --no-cache --no-build-isolation -r requirements_extra.txt; uv pip install pygltflib; bash scripts/install_slangc.sh; }
python - <<'PY'
p = "/data/da/3dgrut/threedgrut/export/scripts/export_usd.py"
s = open(p).read()
k = '    if args.format == "nurec":\n'
if "LOCAL_PATCH_NO_TRANSFORM" not in s:
    s = s.replace(k, k + "        if args.no_transform: conf.export_usd.apply_normalizing_transform = False  # LOCAL_PATCH_NO_TRANSFORM\n", 1)
    open(p, "w").write(s)
print("export_usd no-transform patch", "LOCAL_PATCH_NO_TRANSFORM" in open(p).read())
PY
python -c "import torch, threedgrut; print('GRUT_OK', torch.__version__, torch.cuda.is_available())" || { echo GRUT_DONE; exit 1; }
if [ ! -d $G/images_3 ]; then
  mkdir -p $G/sparse/0 $G/images_3/dslr_images_undistorted
  cp $D/dslr_calibration_undistorted/*.txt $G/sparse/0/
  python - <<'PY'
from pathlib import Path
from PIL import Image
src, dst = Path("/data/da/eth3d/delivery_area/images/dslr_images_undistorted"), Path("/data/da/grut_da/images_3/dslr_images_undistorted")
for p in sorted(src.glob("*.JPG")):
    im = Image.open(p)
    im.resize((round(im.width / 3), round(im.height / 3)), Image.LANCZOS).save(dst / p.name, quality=95)
print("resized", len(list(dst.glob("*.JPG"))))
PY
fi
echo SETUP_DONE
t=$(date +%s)
python train.py --config-name apps/colmap_3dgut_mcmc.yaml path=$G out_dir=$R experiment_name=$NAME \
  dataset.downsample_factor=3 dataset.load_exif=false n_iterations=$ITERS num_workers=8 \
  "checkpoint.iterations=[$ITERS]" export_usd.enabled=false 2>&1 | grep -v "it/s\]" | tail -40
echo "TIME train $(( $(date +%s) - t ))"
CKPT=$(ls -t $R/$NAME/*/ckpt_last.pt | head -1); echo "CKPT $CKPT"
mkdir -p $OUT
python -m threedgrut.export.scripts.export_usd -c $CKPT -o $OUT/${NAME}_nurec.usdz --format nurec --no-transform 2>&1 | tail -15
ls -la $OUT
echo GRUT_DONE
