#!/bin/bash
# Records CooHOI's two-agent box carry (TwoAgent.pth) to an mp4 through an offscreen camera; no display needed.
# Usage: bash run_coohoi.sh [SECONDS]   (default 20 = one full episode; writes ~/workspace/coohoi_twoagent.mp4, or $OUT)
set -eu
DUR=${1:-20}
OUT=${OUT:-$HOME/workspace/coohoi_twoagent.mp4}
HERE=$(cd "$(dirname "$0")" && pwd)
CONDA=$(command -v conda || echo "$HOME/miniforge3/bin/conda")
eval "$("$CONDA" shell.bash hook)"
set +u; conda activate coohoi; set -u  # the env's activate hook sets LD_LIBRARY_PATH for libpython3.8
unset DISPLAY  # render offscreen: Xvfb or X forwarding crash Isaac Gym's Vulkan in create_sim
mkdir -p "$(dirname "$OUT")"
cd "${COOHOI:-$HOME/CooHOI}"
python "$HERE/record_coohoi.py" --seconds "$DUR" --video "$OUT" \
  --test --task ShareHumanoidCarryObject --num_envs 16 --headless \
  --cfg_env coohoi/data/cfg/share_humanoid_carrybox.yaml \
  --cfg_train coohoi/data/cfg/train/share_humanoid_task_coohoi.yaml \
  --motion_file coohoi/data/motions/coohoi_data/coohoi_data.yaml \
  --checkpoint coohoi/data/models/TwoAgent.pth
