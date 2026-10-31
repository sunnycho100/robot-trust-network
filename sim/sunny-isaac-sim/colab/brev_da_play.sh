#!/bin/bash
# CooHOI pair carry in the rebuilt delivery_area: da_coohoi_scene.py builds da_world.usda from the PatchMatch mesh (and
# the splat when given), then play_scene.py runs one episode per seed with the eye at a training photo spot, each to
# its own mp4 and contact sheet. One episode per run keeps each video to one carry.
# Usage: setsid nohup bash brev_da_play.sh NAME "0 1 2" HOME_X [--splat USDZ] [--floor_plane Z] [--fill M] > ~/da_NAME.log 2>&1 < /dev/null &
#   (ends PLAY_DONE; Isaac Lab output in ~/da_NAME_sSEED.play.log)
NAME=$1; SEEDS=$2; H=$3; shift 3
set -x
S=/data/da/scene_$NAME; L=$HOME/robot/isaac-sim/coohoi_lab
EYE=$(/data/da/o3d/bin/python $HOME/robot/recon_code/da_coohoi_scene.py /data/da/pm_da/frames/mesh.ply \
  /data/da/eth3d/delivery_area/dslr_calibration_undistorted/images.txt $S --home $H "$@")
echo "EYE $EYE"
[ -z "$EYE" ] && { echo PLAY_DONE; exit 1; }
cd $HOME/IsaacLab
for SEED in $SEEDS; do
  N="$NAME"_s$SEED; F=$HOME/frames_$N; rm -rf $F
  OMNI_KIT_ACCEPT_EULA=YES COOHOI_HUMANOID_USD=$L/humanoid28_pelvis_root.usda timeout 900 .venv/bin/python $L/play_scene.py \
    --mode pair --episodes 1 --num_envs 1 --effort unlimited --seed $SEED --scene $S/da_world.usda --video $F $EYE --home $H > $HOME/da_$N.play.log 2>&1
  grep "\[play\] episode" $HOME/da_$N.play.log
  ffmpeg -loglevel error -y -framerate 30 -pattern_type glob -i "$F/frame_*.png" -c:v libx264 -pix_fmt yuv420p -crf 22 $HOME/da_$N.mp4 && \
    ffmpeg -loglevel error -y -i $HOME/da_$N.mp4 -vf 'select=not(mod(n\,100)),scale=640:-1,tile=3x2' -frames:v 1 $HOME/da_"$N"_sheet.png
done
echo PLAY_DONE
