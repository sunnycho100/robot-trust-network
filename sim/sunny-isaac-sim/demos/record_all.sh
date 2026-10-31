#!/bin/bash
# Record a list of stock Isaac Sim 6.2 examples to mp4 plus a contact sheet, one container per demo. Usage (on the VM): bash record_all.sh [SECONDS]
set -u
SECS=${1:-20}
export ISAAC_SIM_IMAGE=nvcr.io/nvidia/isaac-sim:6.2.0
C="docker compose -p isim -f ${REPO:-$HOME/IsaacSim}/tools/docker/docker-compose.yml -f $HOME/robot/isaac-sim/docker-compose.override.yml"
R=$HOME/robot/recordings
P=standalone_examples/api/isaacsim.robot.policy.examples
DEMOS=(
  "spot|$P/spot_standalone.py"
  "go2|$P/go2_standalone.py"
  "anymal|$P/anymal_standalone.py"
  "h2|$P/h2_standalone.py"
  "franka_policy|$P/franka_standalone.py"
  "ur10e_pick_place_cumotion|standalone_examples/tutorials/manipulation/tutorial_9_pick_place_cumotion.py"
  "ur10_palletizing|standalone_examples/tutorials/state_machine/ur10_palletizing_transitions.py"
  "franka_stacking|standalone_examples/tutorials/state_machine/franka_stacking.py"
)
mkdir -p "$R"
for d in "${DEMOS[@]}"; do
  name=${d%%|*}; cmd=${d#*|}
  [ -f "$R/$name.mp4" ] && { echo "$name: already recorded"; continue; }
  echo "$(date +%H:%M) $name: recording ${SECS}s"
  sudo setfacl -R -m u:1234:rwX "$R"
  docker rm -f "rec_$name" >/dev/null 2>&1
  $C run --name "rec_$name" --entrypoint bash isaac-sim -c \
    "cd /isaac-sim && timeout 900 ./python.sh /robot/isaac-sim/demos/record_demo.py /robot/recordings/$name $SECS $cmd" \
    > "$R/$name.log" 2>&1
  sudo chown -R "$(id -u):$(id -g)" "$R/$name" 2>/dev/null
  n=$(ls "$R/$name" 2>/dev/null | wc -l)
  if [ "$n" -gt 0 ]; then
    ffmpeg -loglevel error -y -framerate 30 -pattern_type glob -i "$R/$name/frame_*.png" -c:v libx264 -pix_fmt yuv420p -crf 22 "$R/$name.mp4" &&
      ffmpeg -loglevel error -y -i "$R/$name.mp4" -vf 'select=not(mod(n\,90)),scale=640:-1,tile=3x2' -frames:v 1 "$R/${name}_sheet.png" &&
      rm -rf "${R:?}/$name"
    echo "$(date +%H:%M) $name: $n frames -> $name.mp4"
  else
    echo "$(date +%H:%M) $name: FAILED, no frames. Last errors:"; grep -E "Error|Traceback" "$R/$name.log" | grep -v -E "hoops|OmniHub|NVST" | tail -3
  fi
  docker rm -f "rec_$name" >/dev/null 2>&1
done
echo "$(date +%H:%M) all done"
