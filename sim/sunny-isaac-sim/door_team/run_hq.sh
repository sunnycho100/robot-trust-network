#!/bin/bash
# v3c high-quality render on the Brev VM (1080p path tracing, denoiser, textured look, DOF, motion blur, JPEG q95).
# Usage (on the render host): cd ~/robot/door_team && NAME=v3c_hq JOBS=1 DUR=52.7 setsid nohup bash run_hq.sh [door_team.py args] > run_hq.log 2>&1 &
# Later --start/--secs in the args override the chunk range (argparse keeps the last value): that is how to resume.
# With --stills in the args it only renders those times (per-frame timing test). Stops if the disk drops under 5 GB.
set -u
export ISAAC_SIM_IMAGE=nvcr.io/nvidia/isaac-sim:6.2.0
C="docker compose -p isim -f $HOME/IsaacSim/tools/docker/docker-compose.yml -f $HOME/robot/isaac-sim/docker-compose.override.yml"
R=$HOME/robot/door_team; N=${NAME:-v3c_hq}; PLAN=${PLAN:-plan_v3c}; O=$R/$N; J=${JOBS:-1}; DUR=${DUR:-52.7}; X="$*"; cd $R
free_gb() { df -BG --output=avail / | tail -1 | tr -dc 0-9; }
echo "free before: $(free_gb) GB"; [ "$(free_gb)" -lt 5 ] && { echo "LOW DISK, stop"; exit 1; }
sudo setfacl -R -m u:1234:rwX "$R"; sudo find "$R" -type d -exec setfacl -d -m "u:1234:rwx,u:$(id -u):rwx" {} +
py() { $C run --rm --entrypoint bash isaac-sim -c "cd /isaac-sim && ./python.sh $*"; }
sudo rm -rf $O; mkdir -p $O
for i in $(seq 0 $((J - 1))); do
  a=$(python3 -c "print(round($DUR * $i / $J, 4))"); b=$(python3 -c "print(round($DUR * ($i + 1) / $J, 4))")
  py /robot/door_team/door_team.py /robot/door_team/$PLAN /robot/door_team/$N/rec$i --hq --start $a --secs $b $X > $O/r$i.log 2>&1 &
  while [ -z "$(sudo ls $O/rec$i 2>/dev/null | head -1)" ] && [ "$(jobs -r | wc -l)" -gt 0 ]; do sleep 5; done
done
( while sleep 60; do [ "$(free_gb)" -lt 5 ] && { echo "LOW DISK $(date +%T): stopping renders"; pkill -f "[r]un_hq.sh"; }; done ) &
WATCH=$!
wait $(jobs -p | grep -v $WATCH) 2>/dev/null; kill $WATCH 2>/dev/null
sudo chown -R "$(id -u):$(id -g)" $O
grep -h -E "Traceback|Error:|\[hq\]|\[render\]|\[plant\]" $O/r*.log | grep -v -E "OmniHub|hoops|NVST" | tail -20
ls $O/rec* | head; echo "free at end: $(free_gb) GB"; echo HQ_DONE
