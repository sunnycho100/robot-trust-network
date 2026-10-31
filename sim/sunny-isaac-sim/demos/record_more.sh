#!/bin/bash
# Extra takes: 60 s palletizing, the H1 in our scanned floor (room center), the H1 in the NuRec wormhole scan. Usage (on the VM): bash record_more.sh
set -u
export ISAAC_SIM_IMAGE=nvcr.io/nvidia/isaac-sim:6.2.0
C="docker compose -p isim -f ${REPO:-$HOME/IsaacSim}/tools/docker/docker-compose.yml -f $HOME/robot/isaac-sim/docker-compose.override.yml"
R=$HOME/robot/recordings; S=/robot/scenes/floor4_static3
rec() {  # name seconds script args...
  name=$1; secs=$2; shift 2
  echo "$(date +%H:%M) $name: recording ${secs}s"
  sudo setfacl -R -m u:1234:rwX "$R"
  $C run --rm --entrypoint bash isaac-sim -c "cd /isaac-sim && timeout 1200 ./python.sh /robot/isaac-sim/demos/record_demo.py /robot/recordings/$name $secs $*" > "$R/$name.log" 2>&1
  sudo chown -R "$(id -u):$(id -g)" "$R/$name" 2>/dev/null
  n=$(ls "$R/$name" 2>/dev/null | wc -l); echo "$(date +%H:%M) $name: $n frames"
  [ "$n" -gt 0 ] && ffmpeg -loglevel error -y -framerate 30 -pattern_type glob -i "$R/$name/frame_*.png" -c:v libx264 -pix_fmt yuv420p -crf 22 "$R/$name.mp4" && ffmpeg -loglevel error -y -i "$R/$name.mp4" -vf 'select=not(mod(n\,300)),scale=640:-1,tile=3x2' -frames:v 1 "$R/${name}_sheet.png" && rm -rf "${R:?}/$name"
}
rec ur10_palletizing_60s 60 standalone_examples/tutorials/state_machine/ur10_palletizing_transitions.py
# room center from scene_bbox.py; the export's suggested spawn (0.95 0.14) sits against a wall
rec h1_our_center_walk 15 /robot/isaac-sim/demos/h1_in_scene.py $S/our_scene.usd --pos -1.0 0.25 1.05 --command 0.3 0 0.4
# wormhole.usdz from nvidia/PhysicalAI-Robotics-NuRec (xgrid-wormhole); spawn = most open cell of its occupancy map
rec h1_wormhole_walk 15 /robot/isaac-sim/demos/h1_in_scene.py /robot/nurec/xgrid-wormhole/wormhole.usdz --pos 0.48 7.78 1.05 --command 0.4 0 0.2
echo "$(date +%H:%M) all done"
