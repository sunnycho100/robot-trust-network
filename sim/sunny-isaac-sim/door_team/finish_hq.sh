#!/bin/bash
# After run_hq.sh: waits for it to end, then the clean mp4, the overlays in post (inside the container, once per
# chunk folder, scaled to the frame size), the overlay mp4 and a contact sheet. Skips the overlay pass under 6 GB free.
# Usage (on the render host): cd ~/robot/door_team && NAME=v3c_hq setsid nohup bash finish_hq.sh > finish_hq.log 2>&1 &
set -u
export ISAAC_SIM_IMAGE=nvcr.io/nvidia/isaac-sim:6.2.0
C="docker compose -p isim -f $HOME/IsaacSim/tools/docker/docker-compose.yml -f $HOME/robot/isaac-sim/docker-compose.override.yml"
R=$HOME/robot/door_team; N=${NAME:-v3c_hq}; PLAN=${PLAN:-plan_v3c}; O=$R/$N; cd $R
free_gb() { df -BG --output=avail / | tail -1 | tr -dc 0-9; }
py() { $C run --rm --entrypoint bash isaac-sim -c "cd /isaac-sim && ./python.sh $*"; }
while pgrep -f "[r]un_hq.sh" > /dev/null; do sleep 60; done
sudo chown -R "$(id -u):$(id -g)" $O
NF=$(ls $O/rec*/ | grep -c frame_)
echo "$(date +%T) frames: $NF, free $(free_gb) GB"
ffmpeg -loglevel error -y -framerate 30 -pattern_type glob -i "$O/rec*/frame_*.jpg" -c:v libx264 -pix_fmt yuv420p -crf 18 $R/door_team_${N}_clean.mp4
ffmpeg -loglevel error -y -i $R/door_team_${N}_clean.mp4 -vf "select=not(mod(n\,$((NF / 12 + 1)))),scale=640:-1,tile=4x3" -frames:v 1 $R/door_team_${N}_sheet.jpg
if [ "$(free_gb)" -ge 6 ]; then
  sudo setfacl -R -m u:1234:rwX "$O"; sudo find "$O" -type d -exec setfacl -d -m "u:1234:rwx,u:$(id -u):rwx" {} +
  for d in $O/rec*; do py /robot/door_team/overlay.py /robot/door_team/$N/$(basename $d) /robot/door_team/$PLAN /robot/door_team/$N/ov >> $O/ov.log 2>&1; done
  sudo chown -R "$(id -u):$(id -g)" $O
  ffmpeg -loglevel error -y -framerate 30 -pattern_type glob -i "$O/ov/frame_*.jpg" -c:v libx264 -pix_fmt yuv420p -crf 18 $R/door_team_${N}.mp4
else
  echo "LOW DISK: overlay pass skipped"
fi
ls -la $R/door_team_${N}*; echo "$(date +%T) free $(free_gb) GB"; echo FINISH_DONE
