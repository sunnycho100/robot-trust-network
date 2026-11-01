#!/bin/bash
# Door team v3c on the Brev VM: the plan rendered as JOBS parallel Isaac Sim 6.2 containers (time chunks, staggered
# stage loads); the clean mp4 straight from the frames, then the overlay mp4 and its contact sheet. Stops before the
# overlay if the disk would drop under 5 GB free.
# Usage (on the VM): cd ~/robot/door_team && OUTN=v3c DUR=52.7 JOBS=2 setsid nohup bash run_v3c.sh > run_v3c.log 2>&1 &
set -u
export ISAAC_SIM_IMAGE=nvcr.io/nvidia/isaac-sim:6.2.0
C="docker compose -p isim -f $HOME/IsaacSim/tools/docker/docker-compose.yml -f $HOME/robot/isaac-sim/docker-compose.override.yml"
R=$HOME/robot/door_team; N=${OUTN:-v3c}; O=$R/$N; J=${JOBS:-2}; DUR=${DUR:-60}; cd $R
free_gb() { df -BG --output=avail / | tail -1 | tr -dc 0-9; }
echo "free before: $(free_gb) GB"
sudo setfacl -R -m u:1234:rwX "$R"; sudo find "$R" -type d -exec setfacl -d -m "u:1234:rwx,u:$(id -u):rwx" {} +
py() { $C run --rm --entrypoint bash isaac-sim -c "cd /isaac-sim && ./python.sh $*"; }
sudo rm -rf $O; mkdir -p $O
for i in $(seq 0 $((J - 1))); do
  a=$(python3 -c "print(round($DUR * $i / $J, 4))"); b=$(python3 -c "print(round($DUR * ($i + 1) / $J, 4))")
  py /robot/door_team/door_team.py /robot/door_team/plan_$N /robot/door_team/$N/rec$i --start $a --secs $b > $O/r$i.log 2>&1 &
  while [ -z "$(sudo ls $O/rec$i 2>/dev/null | head -1)" ] && [ "$(jobs -r | wc -l)" -gt 0 ]; do sleep 5; done
done
wait
sudo chown -R "$(id -u):$(id -g)" $O
ffmpeg -loglevel error -y -framerate 30 -pattern_type glob -i "$O/rec[0-9]/frame_*.png" -c:v libx264 -pix_fmt yuv420p -crf 21 $R/door_team_${N}_clean.mp4
echo "free after render: $(free_gb) GB"
if [ "$(free_gb)" -lt 8 ]; then echo "LOW DISK: overlay skipped"; else
  sudo setfacl -R -m u:1234:rwX "$O"
  for i in $(seq 0 $((J - 1))); do py /robot/door_team/overlay.py /robot/door_team/$N/rec$i /robot/door_team/plan_$N /robot/door_team/$N/ov >> $O/ov.log 2>&1; done
  sudo chown -R "$(id -u):$(id -g)" $O
  NF=$(ls $O/ov | wc -l)
  ffmpeg -loglevel error -y -framerate 30 -pattern_type glob -i "$O/ov/frame_*.png" -c:v libx264 -pix_fmt yuv420p -crf 21 $R/door_team_$N.mp4
  ffmpeg -loglevel error -y -i $R/door_team_$N.mp4 -vf "select=not(mod(n\,$((NF / 12 + 1)))),scale=640:-1,tile=4x3" -frames:v 1 $R/door_team_${N}_sheet.jpg
fi
for d in $O/rec* $O/ov; do echo "$d $(ls $d | grep -c frame_) frames, small $(find $d -name 'frame_*' -size -100k | wc -l)"; done
grep -h -E "Traceback|Error:|\[camera\]|\[precompute\]|\[ws30\]|\[plant\]|render\] done" $O/r*.log | grep -v -E "OmniHub|hoops|NVST" | sort | uniq -c | head -20
ls -la $R/door_team_${N}*; echo "free at end: $(free_gb) GB"; echo ALL_DONE
