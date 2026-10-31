#!/bin/bash
# One-time CooHOI setup on the Brev VM: Python 3.8 conda env with torch 2.2.2+cu121, Isaac Gym Preview 4, CooHOI at 3183d99.
# Usage: bash setup_coohoi.sh   (upload IsaacGym_Preview_4_Package.tar.gz to $HOME first; safe to rerun)
set -eu
TGZ=$HOME/IsaacGym_Preview_4_Package.tar.gz
COOHOI=${COOHOI:-$HOME/CooHOI}
ENV=coohoi
# unverified: Brev's Jupyter terminal runs as ubuntu with passwordless sudo, on Ubuntu 22.04 (24.04 should also work).

if [ ! -f "$TGZ" ]; then
  echo "ERROR: $TGZ not found." >&2
  echo "Download it from https://developer.nvidia.com/isaac-gym (NVIDIA login and license accept), upload it to the VM," >&2
  echo "and move it to \$HOME. A Jupyter upload may land in ~/workspace: mv ~/workspace/IsaacGym_Preview_4_Package.tar.gz ~/" >&2
  exit 1
fi

echo "== 1. system packages: g++ and ninja build gymtorch, ffmpeg writes the mp4, vulkaninfo checks rendering"
. /etc/os-release; echo "$PRETTY_NAME"
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq build-essential ninja-build ffmpeg git curl vulkan-tools

echo "== 2. conda env $ENV (Python 3.8; Miniforge if conda is missing)"
CONDA=$(command -v conda || echo "$HOME/miniforge3/bin/conda")
if [ ! -x "$CONDA" ]; then
  curl -fsSL -o /tmp/miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash /tmp/miniforge.sh -b -p "$HOME/miniforge3"
fi
eval "$("$CONDA" shell.bash hook)"
# conda-forge only: avoids the Terms of Service prompt of Anaconda's default channels
conda env list | grep -q "^$ENV " || conda create -y -n $ENV -c conda-forge --override-channels python=3.8
set +u; conda activate $ENV; set -u  # conda's activate scripts trip over set -u

echo "== 3. LD_LIBRARY_PATH for libpython3.8 (Isaac Gym's gym_38.so), set on every 'conda activate $ENV'"
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}' > "$CONDA_PREFIX/etc/conda/activate.d/libpython.sh"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

echo "== 4. torch 2.2.2+cu121 and numpy 1.23.5 (the Coohoi_ran fork's pins; torch 1.8.1 does not support the L40S)"
# numpy < 1.24: Isaac Gym's torch_utils uses np.float. setuptools 69.5.1: also the fork's pin; setuptools 70
# broke torch.utils.cpp_extension (which builds gymtorch) on older torch.
pip install torch==2.2.2+cu121 torchvision==0.17.2+cu121 numpy==1.23.5 setuptools==69.5.1 \
  --extra-index-url https://download.pytorch.org/whl/cu121

echo "== 5. Isaac Gym Preview 4"
[ -d "$HOME/isaacgym" ] || tar -xzf "$TGZ" -C "$HOME"
pip install -e "$HOME/isaacgym/python"

echo "== 6. CooHOI at 3183d99 (TwoAgent.pth is in the repo), requirements minus torch"
[ -d "$COOHOI" ] || git clone https://github.com/Winston-Gu/CooHOI.git "$COOHOI"
git -C "$COOHOI" checkout -q 3183d99
grep -v '^torch' "$COOHOI/requirements.txt" > /tmp/coohoi_req.txt
pip install -r /tmp/coohoi_req.txt numpy==1.23.5

echo "== 7. checks"
cd "$COOHOI"
# importing gymtorch compiles it with g++ and ninja the first time
python -c "import isaacgym; from isaacgym import gymtorch; import torch, numpy; print('torch', torch.__version__, '| numpy', numpy.__version__, '| cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
command -v g++ ninja ffmpeg
[ -f /usr/share/vulkan/icd.d/nvidia_icd.json ] || [ -f /etc/vulkan/icd.d/nvidia_icd.json ] || echo "WARNING: no NVIDIA \
Vulkan ICD file. Offscreen rendering needs the driver's graphics libs (the full driver, not a headless or compute-only package)."
vulkaninfo --summary 2>/dev/null | grep -E "deviceName|driverName" || echo "WARNING: vulkaninfo lists no GPU, so camera rendering will likely fail."
echo "Setup done. Next: bash run_coohoi.sh 20"
