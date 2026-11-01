#!/bin/bash
# Colab L4 setup for photo_recon: Vulkan libs, then Isaac Sim 6.2 pip (reuses colab/isaac_pip2.sh). Launch: setsid nohup bash colab_setup.sh > setup.log 2>&1 &
apt-get -qq update && apt-get -qq install -y vulkan-tools libvulkan1 libglu1-mesa libxt6 libsm6 libxrender1 ffmpeg >/dev/null 2>&1
pip install -q uv
bash /content/isaac_pip2.sh
echo SETUP_DONE
