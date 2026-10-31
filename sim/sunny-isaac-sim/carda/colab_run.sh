#!/bin/bash
# Run an Isaac Sim 6.2 (pip, /content/isim) script headless on a Colab L4. Usage: bash colab_run.sh SCRIPT.py [args]
# Needs colab/isaac_pip.sh first. Same environment fixes as colab/isaac_run.sh.
mkdir -p /usr/share/vulkan/icd.d
echo '{"file_format_version":"1.0.1","ICD":{"library_path":"/usr/lib64-nvidia/libEGL_nvidia.so.0","api_version":"1.4.312"}}' > /usr/share/vulkan/icd.d/nvidia_icd.json
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg OMNI_KIT_ACCEPT_EULA=YES XDG_RUNTIME_DIR=/tmp/xdg LD_LIBRARY_PATH=/usr/lib64-nvidia:$LD_LIBRARY_PATH
mkdir -p /tmp/xdg
cd /content/isim/lib/python3.12/site-packages/isaacsim
date; timeout 2400 /content/isim/bin/python "$@"; echo "EXIT $?"; date
echo RUN_DONE
