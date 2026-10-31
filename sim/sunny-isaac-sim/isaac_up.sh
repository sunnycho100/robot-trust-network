#!/bin/bash
# Isaac Sim 6.2 plus its web viewer on the Brev VM, with Docker Compose as in the 6.2 docs. Usage: ACCEPT_EULA=Y bash isaac_up.sh up|down|logs [service]|status
set -eu
REPO=${REPO:-$HOME/IsaacSim}  # cloned by brev_setup.sh
HERE=$(cd "$(dirname "$0")" && pwd)
export ISAAC_SIM_IMAGE=nvcr.io/nvidia/isaac-sim:6.2.0
C=(docker compose -p isim -f "$REPO/tools/docker/docker-compose.yml" -f "$HERE/docker-compose.override.yml")
LICENSE=https://docs.isaacsim.omniverse.nvidia.com/6.2.0/common/NVIDIA_Omniverse_License_Agreement.html
docker info >/dev/null 2>&1 || { echo "Docker is not usable by $USER. Run brev_setup.sh first."; exit 1; }
# The Brev guide uses curl -s ifconfig.me; -4 keeps it IPv4 because the viewer is built with this address.
# unverified: the address ifconfig.me reports is the one Brev's exposed ports answer on (the Brev guide assumes so).
pubip() { echo "${ISAACSIM_HOST:-$(curl -4 -s ifconfig.me)}"; }
health() { docker inspect -f '{{.State.Health.Status}}' "$("${C[@]}" ps -q isaac-sim)" 2>/dev/null || true; }

case ${1:-} in
up)
  # The compose file hard-codes ACCEPT_EULA=Y for the container, so the decision has to be made here, by you.
  if [ "${ACCEPT_EULA:-}" != Y ]; then
    echo "Refusing to start. Starting the container accepts the NVIDIA Omniverse License Agreement:"
    echo "  $LICENSE"
    echo "If you accept it, run:  ACCEPT_EULA=Y bash $0 up"
    echo "(The compose file sets no PRIVACY_CONSENT, so data collection stays off.)"
    exit 1
  fi
  export ISAACSIM_HOST; ISAACSIM_HOST=$(pubip)
  echo "ISAACSIM_HOST=$ISAACSIM_HOST"
  # The container runs as uid 1234 and ':rw' does not grant host write access; ACLs as in tools/docker/README.md
  R=$HOME/robot; mkdir -p "$R"
  sudo setfacl -R -m u:1234:rwX "$R"
  sudo find "$R" -type d -exec setfacl -d -m "u:1234:rwx,u:$(id -u):rwx" {} +
  # --build: the viewer bakes ISAACSIM_HOST in at build time, and the public IP can change after a stop.
  # Compose waits for Isaac Sim's healthcheck before starting the viewer; a long first start (shader
  # compilation) can outlast that check and make 'up' fail, so wait below and start the viewer again.
  # unverified: that Compose aborts 'up' on an unhealthy dependency here, and how long the first start takes on an L40S.
  "${C[@]}" up --build -d || echo "Isaac Sim is still loading. Waiting for it."
  [ -n "$("${C[@]}" ps -q isaac-sim)" ] || { echo "The isaac-sim container did not start. See: bash $0 logs"; exit 1; }
  # Ready = the compose healthcheck found " App is loaded." in the newest Kit log
  # (the 6.2 docs' ready line: "Isaac Sim Full Streaming App is loaded.")
  for i in $(seq 180); do  # 180 x 20 s = 60 min
    [ "$(health)" = healthy ] && break
    [ $((i % 3)) = 0 ] && echo "still loading ($((i / 3)) min): $("${C[@]}" logs --tail 1 isaac-sim 2>&1 | cut -c1-150)"
    sleep 20
  done
  [ "$(health)" = healthy ] || { echo "Isaac Sim is not ready after 60 min. See: bash $0 logs isaac-sim"; exit 1; }
  "${C[@]}" logs isaac-sim 2>&1 | grep 'App is loaded' | tail -1 || true
  "${C[@]}" up -d  # starts the web viewer if the first 'up' gave up waiting
  for i in $(seq 60); do curl -sf http://localhost:8210/ >/dev/null && break; sleep 5; done
  curl -sf http://localhost:8210/ >/dev/null || { echo "Web viewer is not answering. See: bash $0 logs web-viewer"; exit 1; }
  echo
  echo "Isaac Sim is ready. Open in Chrome (one tab only):  http://$ISAACSIM_HOST:8210"
  ;;
down) "${C[@]}" down ;;
logs) shift; "${C[@]}" logs -f --tail 100 "$@" ;;
status)
  "${C[@]}" ps
  echo "isaac-sim health: $(health)"
  echo "Viewer: http://$(pubip):8210"
  ;;
*) sed -n 2p "$0"; exit 1 ;;
esac
