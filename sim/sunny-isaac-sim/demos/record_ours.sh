#!/bin/bash
# After record_all.sh: import our scanned floor (HCIC floor4_static3) and record an H1 walking in it. Usage (on the VM): bash record_ours.sh
set -u
export ISAAC_SIM_IMAGE=nvcr.io/nvidia/isaac-sim:6.2.0
C="docker compose -p isim -f ${REPO:-$HOME/IsaacSim}/tools/docker/docker-compose.yml -f $HOME/robot/isaac-sim/docker-compose.override.yml"
R=$HOME/robot/recordings; S=/robot/scenes/floor4_static3
until grep -q "all done" "$R/batch.log" 2>/dev/null; do sleep 20; done
sudo setfacl -R -m u:1234:rwX $HOME/robot
echo "$(date +%H:%M) importing our scene"
$C run --rm --entrypoint bash isaac-sim -c "cd /isaac-sim && timeout 900 ./python.sh /robot/isaac-sim/demos/import_our_scene.py $S" > "$R/import_ours.log" 2>&1
ls -la $HOME/robot/scenes/floor4_static3/our_scene.usd || { echo "import FAILED"; grep -E "Error|Traceback" "$R/import_ours.log" | grep -v -E "hoops|OmniHub" | tail -5; exit 1; }
name=h1_our_scene
echo "$(date +%H:%M) recording $name"
sudo setfacl -R -m u:1234:rwX "$R"
$C run --rm --entrypoint bash isaac-sim -c "cd /isaac-sim && timeout 900 ./python.sh /robot/isaac-sim/demos/record_demo.py /robot/recordings/$name 15 /robot/isaac-sim/demos/h1_in_scene.py $S/our_scene.usd --pos 0.95 0.14 1.05 --command 0.45 0 0.7" > "$R/$name.log" 2>&1
sudo chown -R "$(id -u):$(id -g)" "$R/$name" 2>/dev/null
n=$(ls "$R/$name" 2>/dev/null | wc -l); echo "$(date +%H:%M) $name: $n frames"
[ "$n" -gt 0 ] && ffmpeg -loglevel error -y -framerate 30 -pattern_type glob -i "$R/$name/frame_*.png" -c:v libx264 -pix_fmt yuv420p -crf 22 "$R/$name.mp4" && ffmpeg -loglevel error -y -i "$R/$name.mp4" -vf 'select=not(mod(n\,90)),scale=640:-1,tile=3x2' -frames:v 1 "$R/${name}_sheet.png" && rm -rf "${R:?}/$name"
grep -E "Error|Traceback" "$R/$name.log" | grep -v -E "hoops|OmniHub|NVST" | tail -3
echo "$(date +%H:%M) ours done"
