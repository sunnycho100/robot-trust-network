#!/bin/bash
# Look-pass HQ clip of one shot at 1080p, no overlay; resumes into an existing frame folder after a crash.
# Usage (on the host): bash hq_clip.sh SHOT NAME   e.g. bash hq_clip.sh TwoG1Close close_2g1_sheets_v3_hq
# Faster (32 samples per pixel instead of 192): SPP=8 MBLUR=2 bash hq_clip.sh SHOT NAME
# Needs usd_v3/look_hq.usda (record_v3.sh). Stops if under 5 GB disk is free. Frames: hq_NAME/, video: NAME.mp4
cd "$(dirname "$0")"
shot=$1; name=$2; F=hq_$name; PY=${VENV:-$PWD/venv}/bin/python
free=$(df --output=avail -BG / | tail -1 | tr -dc 0-9); echo "$(date +%T) free ${free} GB"
[ "$free" -ge 5 ] || { echo "STOP: under 5 GB free"; exit 1; }
mkdir -p $F
last=$(ls $F | tail -1)
if [ -n "$last" ] && ! $PY -c "from PIL import Image; Image.open('$F/$last').load()" 2>/dev/null; then rm -f "$F/$last"; fi
n=$(ls $F | wc -l)
read -r a b <<< "$($PY -c "import shots; s = next(x for x in shots.SHOTS if x[0] == '$shot'); print(s[1] + $n / shots.FPS, s[2])")"
echo "$(date +%T) $name: $n frames on disk, rendering sim $a to $b s"
t=$(date +%s)
bash in_isaac.sh $F.log /robot/g1coop/render_video.py /robot/g1coop/usd_v3/look_hq.usda /robot/g1coop/$F --mode PathTracing \
  --shots $shot --span $a $b --first-index $n --hq --no-overlay --dof 60 --mblur ${MBLUR:-3} --post --size 1920 1080 --spp ${SPP:-32} --subframes ${SUB:-2}
echo "$(date +%T) $name wall $(( $(date +%s) - t )) s"; grep "g1coop\]" $F.log | tail -1
sudo chown -R "$(id -u):$(id -g)" $F
n=$(ls $F | wc -l); echo "frames $n"
ffmpeg -loglevel error -y -framerate 30 -i $F/frame_%05d.png -c:v libx264 -preset slow -crf 14 -pix_fmt yuv420p $name.mp4
ffmpeg -loglevel error -y -i $name.mp4 -vf "select=not(mod(n\,$(( n / 12 > 0 ? n / 12 : 1 )))),scale=640:-1,tile=4x3" -frames:v 1 ${name}_sheet.png
ls -la $name.mp4 ${name}_sheet.png
echo "CLIP_DONE $name"
