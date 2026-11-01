#!/bin/bash
# One-time setup of the MuJoCo half on a fresh GPU host, from a checkout of this repo (Seohee's sim from main).
# Usage: bash sim/sunny-isaac-sim/g1coop_port/setup_vm.sh   Env: G1COOP (default ~/robot/g1coop), VENV (default $G1COOP/venv)
# The Isaac Sim container is set up separately with sim/sunny-isaac-sim/README.md (brev_setup.sh, isaac_up.sh).
set -ex
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../../.." && pwd)
G1COOP=${G1COOP:-$HOME/robot/g1coop}; VENV=${VENV:-$G1COOP/venv}
mkdir -p "$G1COOP/code/mujoco_houses"
cp "$HERE"/*.py "$HERE"/*.sh "$HERE"/*.json "$G1COOP/"
cp -r "$REPO/sim/mujoco_houses/src" "$REPO/sim/mujoco_houses/factory" "$REPO/sim/mujoco_houses/assets" "$G1COOP/code/mujoco_houses/"
cd "$G1COOP/code"
[ -d roqsim_factory_upstream ] || git clone -q --filter=blob:none https://github.com/cps-test-lab/roqsim.git roqsim_factory_upstream
git -C roqsim_factory_upstream checkout -q 6e6999be07e56ebf442b2f20225f33b97c3a9f70
T=mujoco_houses/third_party/unitree_rl_gym  # the G1 model and walking policy, as scripts/fetch_assets pins them
if [ ! -d $T/.git ]; then
  mkdir -p $T && cd $T && git init -q && git remote add origin https://github.com/unitreerobotics/unitree_rl_gym.git
  git sparse-checkout set --no-cone /deploy/pre_train/g1/motion.pt /deploy/deploy_mujoco/configs/g1.yaml /resources/robots/g1_description/
  git fetch -q --depth 1 --filter=blob:none origin 276801e46c5d433564f24658bac64f254b7d2d4b && git checkout -q 276801e46c5d433564f24658bac64f254b7d2d4b
  cd -
fi
command -v uv || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=$HOME/.local/bin:$PATH
[ -x "$VENV/bin/python" ] || uv venv "$VENV" --python 3.13
uv pip install --python "$VENV/bin/python" mujoco==3.17.0 numpy pillow tqdm pyyaml usd-core
uv pip install --python "$VENV/bin/python" torch --index-url https://download.pytorch.org/whl/cpu
"$VENV/bin/python" -c "import mujoco, torch, pxr; print('setup ok', mujoco.__version__, torch.__version__)"
