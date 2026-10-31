#!/bin/bash
# Render ~/robot/g1coop/usd/factory.usd in the Isaac Sim 6.2 container, then make the mp4 and a contact sheet.
# Usage (on the VM): bash run_render.sh NAME [render_video.py options, e.g. --every 30 --mode PathTracing]
set -u
export ISAAC_SIM_IMAGE=nvcr.io/nvidia/isaac-sim:6.2.0
C="docker compose -p isim -f $HOME/IsaacSim/tools/docker/docker-compose.yml -f $HOME/robot/isaac-sim/docker-compose.override.yml"
name=$1; shift
R=$HOME/robot/g1coop; O=$R/$name
rm -rf "$O"
sudo setfacl -R -m u:1234:rwX "$R"
sudo find "$R" -type d -exec setfacl -d -m "u:1234:rwx,u:$(id -u):rwx" {} +
$C run --rm --entrypoint bash isaac-sim -c "cd /isaac-sim && ./python.sh /robot/g1coop/render_video.py /robot/g1coop/usd/factory.usd /robot/g1coop/$name $*" > "$R/$name.log" 2>&1
grep -E "g1coop|Error|Traceback|error:" "$R/$name.log" | grep -v -E "OmniHub|hoops|NVST" | tail -20
sudo chown -R "$(id -u):$(id -g)" "$O"
n=$(ls "$O" | wc -l); echo "$name: $n frames"
[ "$n" -gt 0 ] || exit 1
ffmpeg -loglevel error -y -framerate 30 -i "$O/frame_%05d.png" -c:v libx264 -pix_fmt yuv420p -crf 18 "$R/$name.mp4"
step=$(( n / 12 > 0 ? n / 12 : 1 ))
ffmpeg -loglevel error -y -i "$R/$name.mp4" -vf "select=not(mod(n\,$step)),scale=640:-1,tile=4x3" -frames:v 1 "$R/${name}_sheet.png"
ls -la "$R/$name.mp4" "$R/${name}_sheet.png"
