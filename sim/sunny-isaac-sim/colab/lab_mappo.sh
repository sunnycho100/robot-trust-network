#!/bin/bash
# Isaac Lab 3.0 multi-agent smoke on a Colab L4: skrl MAPPO on Isaac-Shadow-Handover-Direct (two hands, one ball),
# kit-less Newton physics, no Isaac Sim. Scaffold check for a CooHOI port. Log /content/lab_mappo.log, ends MAPPO_DONE.
set -x
export UV_SYSTEM_PYTHON=false GIT_LFS_SKIP_SMUDGE=1
cd /content && [ -d IsaacLab ] || git clone -q --depth 1 --branch v3.0.0 https://github.com/isaac-sim/IsaacLab
cd /content/IsaacLab
date; time uv sync --extra skrl 2>&1 | tail -5
date; timeout 2400 uv run --extra skrl isaaclab train --rl_library skrl --task Isaac-Shadow-Handover-Direct \
  --algorithm MAPPO --num_envs ${NENV:-4096} physics=newton_mjwarp 2>&1 | grep -v '^\s*[0-9]*%|' | tail -60
echo "exit $?"; date
find /content/IsaacLab/logs -name "events.out*" | head; echo MAPPO_DONE
