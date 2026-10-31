#!/bin/bash
# Colab B step 1b: the installer exits at the Kaolin find-links URL (no cu130 page), so run its steps 8 and 9 by hand.
set -x
cd /content/3dgrut && . .venv/bin/activate
export TORCH_CUDA_ARCH_LIST="8.9" MAX_JOBS=8
time uv pip install --no-cache --no-build-isolation -r requirements_extra.txt
uv pip install pygltflib
bash scripts/install_slangc.sh
python -c "import ppisp; from fused_ssim import fused_ssim; print('EXTRAS_OK')"
echo SETUP2_DONE
# Local patch (3dgrut bb203b1): the nurec branch of export_usd ignores --no-transform (it reads
# conf.export_usd.apply_normalizing_transform), and its Volume matrix is authored column-major, which USD reads
# as projective. With --no-transform the Volume is identity and the splat stays in the dataset (COLMAP) frame.
python - <<'PY'
p = "/content/3dgrut/threedgrut/export/scripts/export_usd.py"
s = open(p).read()
k = '    if args.format == "nurec":\n'
if "LOCAL_PATCH_NO_TRANSFORM" not in s:
    s = s.replace(k, k + "        if args.no_transform: conf.export_usd.apply_normalizing_transform = False  # LOCAL_PATCH_NO_TRANSFORM\n", 1)
    open(p, "w").write(s)
print("export_usd no-transform patch", "LOCAL_PATCH_NO_TRANSFORM" in open(p).read())
PY
