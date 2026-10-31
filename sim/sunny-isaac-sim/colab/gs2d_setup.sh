#!/bin/bash
# gsplat 2DGS venv on reconB, separate from the 3DGRUT venv. torch 2.9.1 cu130 matches Colab's nvcc 13.0.
# Usage: nohup bash gs2d_setup.sh > /content/gs2d_setup.log 2>&1 &   (ends GSSETUP_DONE)
set -x
cd /content/gsplat && git submodule update --init --recursive --depth 1
uv venv -p 3.11 /content/gs && . /content/gs/bin/activate
uv pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu130
uv pip install pycolmap viser "git+https://github.com/nerfstudio-project/nerfview@4538024fe0d15fd1a0e4d760f3695fc44ca72787" \
  "imageio[ffmpeg]" "numpy>=2,<3" scipy scikit-learn tqdm torchmetrics==1.8.2 opencv-python-headless \
  "tyro>=0.8.8,!=1.0.9,!=1.0.10" Pillow piexif tensorboard tensorly pyyaml matplotlib splines open3d ninja setuptools wheel
TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=4 nice uv pip install --no-build-isolation -e /content/gsplat
python -c "import gsplat; from gsplat.rendering import rasterization_2dgs; print('GS_OK', gsplat.__version__)"
echo GSSETUP_DONE
