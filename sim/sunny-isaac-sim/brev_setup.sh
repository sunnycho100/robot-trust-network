#!/bin/bash
# One-time Isaac Sim 6.2 prep on the Brev VM: driver, Docker, NVIDIA Container Toolkit, compose files, cache folders, image. Usage: bash brev_setup.sh
# Safe to rerun; every step skips itself when already done. Never asks for or stores an NGC key.
set -eu
NEED=595.58.03  # Linux driver Isaac Sim 6.2 was tested on (6.2.0/installation/requirements.html)
IMAGE=nvcr.io/nvidia/isaac-sim:6.2.0
REPO=${REPO:-$HOME/IsaacSim}
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
# unverified: Brev's Jupyter terminal runs as the VM user (ubuntu) with passwordless sudo.

echo "== 1. OS and driver"
. /etc/os-release
echo "$PRETTY_NAME, kernel $(uname -r)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
df -h / "$HOME"
# unverified: which driver Brev's Crusoe L40S image ships. The check below handles any version.
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)
if [ -z "$DRV" ] || [ "$(printf '%s\n' "$NEED" "$DRV" | sort -V | head -1)" != "$NEED" ]; then
  echo "Driver '${DRV:-none}' is older than $NEED. Installing $NEED from NVIDIA's CUDA apt repo."
  case $VERSION_ID in 22.04|24.04) ;; *) echo "Ubuntu $VERSION_ID is not supported by Isaac Sim 6.2 (22.04 or 24.04)."; exit 1;; esac
  DISTRO=ubuntu${VERSION_ID/./}  # ubuntu2204 or ubuntu2404
  # Steps from NVIDIA's Driver Installation Guide (Ubuntu, network repo). Branch 590+ packages drop the
  # branch number; nvidia-driver-pinning-<version> selects the version. Both names checked in the repo index.
  sudo apt-get update
  sudo apt-get install -y "linux-headers-$(uname -r)"
  curl -fsSL -o "$TMP/cuda-keyring.deb" "https://developer.download.nvidia.com/compute/cuda/repos/$DISTRO/x86_64/cuda-keyring_1.1-1_all.deb"
  sudo dpkg -i "$TMP/cuda-keyring.deb"
  sudo apt-get update
  apt-mark showhold | grep -i nvidia && echo "Warning: the held NVIDIA packages above can block the upgrade."
  # nvidia-open = full driver with open kernel modules (graphics libs included, which Isaac Sim's RTX renderer needs)
  # unverified: an older driver from Ubuntu's own packages (nvidia-driver-5xx-server) is replaced cleanly.
  sudo apt-get install -y "nvidia-driver-pinning-$NEED" nvidia-open || {
    echo "apt could not install the driver. Check the conflict above; removing the old driver first usually fixes it:"
    echo "  sudo apt-get purge 'nvidia-driver-*' 'libnvidia-*' 'nvidia-dkms-*' && rerun this script"
    exit 1; }
  echo
  echo "Driver $NEED installed. A REBOOT IS NEEDED: run 'sudo reboot', wait about 2 minutes,"
  echo "reopen the Jupyter terminal, then run this script again."
  exit 1
fi
echo "Driver $DRV is new enough (>= $NEED)."

echo "== 2. Docker, Compose and NVIDIA Container Toolkit"
# Brev documents VM Mode as shipping Docker, Docker Compose and the NVIDIA Container Toolkit; install only if missing.
if ! command -v docker >/dev/null || ! docker compose version >/dev/null 2>&1; then
  echo "Installing Docker with get.docker.com (the method in the Isaac Sim 6.2 container docs)."
  curl -fsSL https://get.docker.com -o "$TMP/get-docker.sh"
  sudo sh "$TMP/get-docker.sh"
fi
docker --version; docker compose version
if ! docker info >/dev/null 2>&1; then
  id -nG | grep -qw docker && { echo "Docker daemon is not reachable. Check: sudo systemctl status docker"; exit 1; }
  sudo usermod -aG docker "$USER"
  echo "Added $USER to the docker group. Run 'newgrp docker' in this terminal, then rerun this script."
  exit 1
fi
if ! command -v nvidia-ctk >/dev/null; then
  # From tools/docker/README.md at v6.2.0
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list |
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' |
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  sudo apt-get update
  sudo apt-get install -y nvidia-container-toolkit
fi
nvidia-ctk --version | head -1
# The compose file uses "runtime: nvidia", so Docker must know that runtime
if ! docker info --format '{{json .Runtimes}}' | grep -q '"nvidia"'; then
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
fi
docker run --rm --runtime=nvidia --gpus all nvcr.io/nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi -L

echo "== 3. Isaac Sim compose files (v6.2.0, tools/docker only)"
command -v git >/dev/null && command -v setfacl >/dev/null || { sudo apt-get update; sudo apt-get install -y git acl; }
if [ ! -d "$REPO/.git" ]; then
  git clone --depth 1 --branch v6.2.0 --filter=blob:none --sparse https://github.com/isaac-sim/IsaacSim "$REPO"
  git -C "$REPO" sparse-checkout set tools/docker
fi
ls "$REPO/tools/docker/docker-compose.yml"

echo "== 4. Cache and log folders owned by uid 1234 (the container user)"
mkdir -p ~/docker/isaac-sim/{cache/main,cache/computecache,cache/kit,config,data,logs,pkg}
mkdir -p ~/.cache/ov/hub
sudo chown -R 1234:1234 ~/docker ~/.cache/ov/hub
mkdir -p "$HOME/robot"  # our demos and scenes; isaac_up.sh mounts it at /robot

echo "== 5. Pull $IMAGE (about 11 GB compressed)"
# The image is public: its manifest and config read anonymously from nvcr.io on 2026-10-30, and the 6.2 docs list no login step.
if ! docker pull "$IMAGE"; then
  echo
  echo "Pull failed. If the error says 'unauthorized' or 'denied', log in yourself and rerun this script:"
  echo "  docker login nvcr.io     (username: \$oauthtoken, password: your NGC API key)"
  echo "If it says 'Please accept license on the browser', this Docker is already logged in to an NGC org"
  echo "that has not accepted the terms. Either accept them on the NGC page or run 'docker logout nvcr.io'."
  exit 1
fi
echo
echo "Setup done. Next: read the license, then  ACCEPT_EULA=Y bash $(dirname "$0")/isaac_up.sh up"
