#!/bin/bash
# Three H1 humanoids walking in the full warehouse (stock 6.2 example), streamed to the web viewer. Usage (inside the container): bash h1_warehouse.sh
set -eu
# Livestream flags: the address and ports the container's runheadless.sh passes, from the compose environment.
# h1_standalone.py and SimulationApp both pass unknown arguments on to Kit.
LS=--/exts/omni.kit.livestream.app/primaryStream
STREAM=(--enable omni.kit.livestream.app
  "$LS/publicIp=${ISAACSIM_HOST:-127.0.0.1}" "$LS/signalPort=${ISAACSIM_SIGNAL_PORT:-49100}"
  "$LS/streamPort=${ISAACSIM_STREAM_PORT:-47998}" "$LS/streamType=webrtc"
  # unverified: this later hideUi wins over the hideUi=1 SimulationApp adds without a display (UI shown, not just the viewport)
  --/app/window/hideUi=false --/app/window/drawMouse=true)
cd /isaac-sim
exec ./python.sh standalone_examples/api/isaacsim.robot.policy.examples/h1_standalone.py \
  --num-robots 3 --env-url /Isaac/Environments/Simple_Warehouse/full_warehouse.usd "${STREAM[@]}" "$@"
