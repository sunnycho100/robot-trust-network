#!/bin/bash
# Colab A: does Isaac Sim 6.2 run headless on a Colab L4 via pip? Logs to /content/isaac_pip.log
set -x
cd /content
date; nvidia-smi; python3 --version; ldd --version | head -1
ls /usr/share/vulkan/icd.d /etc/vulkan/icd.d 2>&1; ls /usr/lib64-nvidia 2>&1 | head -50
apt-get -qq update && apt-get -qq install -y vulkan-tools libvulkan1 libglu1-mesa libxt6 libsm6 libxrender1 >/dev/null 2>&1
vulkaninfo --summary 2>&1 | head -40
pip install -q uv
for PY in 3.12 3.11; do
  rm -rf /content/isim; uv venv -p $PY /content/isim && . /content/isim/bin/activate
  time uv pip install "isaacsim[all,extscache]==6.2.0" --extra-index-url https://pypi.nvidia.com > /content/uv_$PY.log 2>&1 && { echo "INSTALLED on $PY"; break; }
  tail -15 /content/uv_$PY.log; deactivate
done
pip list 2>/dev/null | grep -i isaacsim; uv pip list 2>/dev/null | grep -i isaacsim | head
df -h /content
echo PIP_DONE
