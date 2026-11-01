#!/bin/bash
# 16 s demo cut of the WS30 car-door story, no text (1920x1080, 30 fps). Run from the repo root.
#   1 record the place (CarDA OUT camera)  2 wipe to the 3D rebuild from the same camera
#   3 robots in the rebuilt station        4 the worker doing the job (CarDA IN camera)
#   5 the robots doing that job (G1s on screw and glass)
# Inputs are local evidence (CarDA footage CC BY 4.0, Isaac Sim renders); output: docs/evidence/process/demo_steps_16s.mp4
set -eu
REAL=docs/evidence/carda/ws30_real/xsens_003_WS30_2023_10_04
OUT_CAM=${REAL}_out_camera_cropped.mp4; IN_CAM=${REAL}_in_camera_cropped.mp4
SBS=docs/evidence/ws30_station/side_by_side_out_09.0.jpg          # left: real at 9 s, right: rebuild, same camera
DOOR=docs/evidence/door_team/v3c_hq/door_team_v3c_hq_clean.mp4
OUT=${1:-docs/evidence/process/demo_steps_16s.mp4}
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
ffmpeg -v error -y -ss 9.0 -i "$OUT_CAM" -frames:v 1 "$TMP/real9.png"
# Same crop on the real frame and the rebuild half (drops the label strip on the side-by-side), then 1080p.
CROP="crop=1191:670:44:50,scale=1920:1080,setsar=1"
N="fps=30,format=yuv420p,settb=AVTB"
ffmpeg -v error -y \
  -ss 5.6 -t 3.4 -i "$OUT_CAM" \
  -loop 1 -t 2.0 -i "$TMP/real9.png" \
  -loop 1 -t 2.6 -i "$SBS" \
  -ss 0.5 -t 3.4 -i "$DOOR" \
  -ss 39.0 -t 3.4 -i "$IN_CAM" \
  -ss 31.2 -t 4.0 -i "$DOOR" \
  -filter_complex "
    [0:v]$CROP,$N[s1];
    [1:v]$CROP,$N[ra];
    [2:v]crop=1280:720:1280:0,$CROP,$N[rb];
    [ra][rb]xfade=transition=wiperight:duration=1.2:offset=0.8[s2];
    [3:v]scale=1920:1080,setsar=1,$N[s3];
    [4:v]scale=1920:1080,setsar=1,$N[s4];
    [5:v]scale=1920:1080,setsar=1,$N,fade=t=out:st=3.5:d=0.5[s5];
    [s1][s2]xfade=transition=fade:duration=0.4:offset=3.0[x1];
    [x1][s3]xfade=transition=fade:duration=0.4:offset=6.0[x2];
    [x2][s4]xfade=transition=fade:duration=0.4:offset=9.0[x3];
    [x3][s5]xfade=transition=fade:duration=0.4:offset=12.0[v]" \
  -map "[v]" -an -c:v libx264 -preset slow -crf 16 -pix_fmt yuv420p -movflags +faststart "$OUT"
ffprobe -v error -show_entries format=duration:stream=width,height -of csv=p=0 "$OUT"
