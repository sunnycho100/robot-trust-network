#!/bin/bash
# Colab B step 1: 3DGRUT on a Colab L4 (CUDA 13.0 toolkit, Ubuntu 24.04) plus ETH3D delivery_area.
# Uses the repo's own uv installer minus Kaolin and tiny-cuda-nn (playground and NHT only). Logs to /content/grut_setup.log
set -x
export TORCH_CUDA_ARCH_LIST="8.9" MAX_JOBS=8 TCNN_CUDA_ARCHITECTURES=89
mkdir -p /content/eth3d
(cd /content/eth3d && wget -q https://www.eth3d.net/data/delivery_area_dslr_undistorted.7z https://www.eth3d.net/data/delivery_area_dslr_scan_eval.7z && \
  7z x -y delivery_area_dslr_undistorted.7z > /dev/null && 7z x -y delivery_area_dslr_scan_eval.7z > /dev/null; echo "DATA_DONE $?") > /content/data.log 2>&1 &
cd /content && rm -rf 3dgrut && git clone -q --recursive https://github.com/nv-tlabs/3dgrut.git && cd 3dgrut
git log -1 --format='%h %cd'
# ponytail: drop the tcnn step and the kaolin build; the playground extra is the only kaolin user
sed -i 's#^INSTALL_TCNN_WITH_UV=1 bash#echo skip tcnn \#&#; s#^if \[ "${CUDA_MAJOR_TARGET:-0}" -le 12 \]; then#if true; then#; s#^uv pip install -e .\[playground\]#uv pip install pygltflib#; s#^python -c "import kaolin.*##' install_env_uv.sh
sed -i '/^if importlib.util.find_spec("tinycudann")/,/^    print("tiny-cuda-nn: ready")/d' install_env_uv.sh
time bash install_env_uv.sh 3dgrut
. .venv/bin/activate && python -c "import torch, threedgrut; print('GRUT_OK', torch.__version__, torch.cuda.is_available())"
wait; cat /content/data.log
echo SETUP_DONE
