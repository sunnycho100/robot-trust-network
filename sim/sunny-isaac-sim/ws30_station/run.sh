#!/bin/bash
# WS30 station on the Brev VM: CarDA cycle 003 data, BVH registration, then ONE Isaac Sim process for the station USD,
# stills and the clip, then finish.py (side-by-side stills, captioned mp4). Code: ~/robot/ws30_station/code, repo layout.
# Usage: cd ~/robot/ws30_station && setsid nohup bash code/sim/sunny-isaac-sim/ws30_station/run.sh [CLIP|""] > run.log 2>&1 &
#   CLIP is cycle seconds A:B (default 6:30); "" builds the USD and stills only. STILLS=view@t,... overrides the stills.
set -u
R=$HOME/robot/ws30_station; CODE=$R/code/sim/sunny-isaac-sim/ws30_station; O=$R/out; CLIP=${1-6:30}
X=$R/data/ws30-svo-mp4-bvh/ws30; CY=xsens_003_WS30_2023_10_04
cd $R/data && { [ -f "$X/xsens/${CY}_cropped_fixed.bvh" ] || bsdtar -xf ws30.rar --exclude '*.svo'; }
mkdir -p $O && cp $CODE/layout.json $R/cameras_ws30.json $O/   # hand-off: USD, layout and cameras side by side
$R/venv/bin/python $CODE/register_bvh.py $X/xsens/${CY}_cropped_fixed.bvh $X/mp4/${CY}_in_camera_cropped.mp4 \
  $X/mp4/${CY}_out_camera_cropped.mp4 $R/cameras_ws30.json $O
sudo setfacl -R -m u:1234:rwX $R; sudo find $R -type d -exec setfacl -d -m "u:1234:rwx,u:$(id -u):rwx" {} +
export ISAAC_SIM_IMAGE=nvcr.io/nvidia/isaac-sim:6.2.0
C="docker compose -p isim -f $HOME/IsaacSim/tools/docker/docker-compose.yml -f $HOME/robot/isaac-sim/docker-compose.override.yml"
V=/robot/ws30_station; VC=$V/code/sim/sunny-isaac-sim/ws30_station
rm -rf $O/clip
$C run --rm --entrypoint bash isaac-sim -c "cd /isaac-sim && ./python.sh $VC/ws30_station.py $VC/layout.json $V/out \
  --cams $V/cameras_ws30.json --worker $V/out/worker_plant.npz --stills ${STILLS:-in@36,out@9,orbit@20} ${CLIP:+--clip $CLIP}" \
  > $O/isaac.log 2>&1
sudo chown -R "$(id -u):$(id -g)" $O
$R/venv/bin/python $CODE/finish.py $O $X/mp4/${CY}_in_camera_cropped.mp4 $X/mp4/${CY}_out_camera_cropped.mp4
grep -E "\[ws30\]|Traceback|Error" $O/isaac.log | grep -v -E "OmniHub|hoops|NVST" | tail -20
echo RUN_DONE
