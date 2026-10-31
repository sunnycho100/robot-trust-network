#!/bin/bash
# Colab queue 1: multi-robot scene recording, then the walkthrough capture of the full warehouse.
bash /content/isaac_run.sh multi 20 /content/examples/multi_robot_scene.py > /content/isaac_run_multi.log 2>&1
export MPLBACKEND=Agg OMNI_KIT_ACCEPT_EULA=YES XDG_RUNTIME_DIR=/tmp/xdg LD_LIBRARY_PATH=/usr/lib64-nvidia:$LD_LIBRARY_PATH
cd /content/isim/lib/python3.12/site-packages/isaacsim
date; timeout 2400 /content/isim/bin/python /content/examples/capture_walkthrough.py /content/walk/full_warehouse --frames 240 > /content/walk.log 2>&1; echo "walk exit $?" >> /content/walk.log; date
echo QUEUE1_DONE
