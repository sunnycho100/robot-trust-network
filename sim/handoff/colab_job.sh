#!/bin/bash
# Usage (Colab L4, after the CarDA WS30 BVH is unpacked and mujoco + opencv are installed, see README.md):
#   cd /content; setsid nohup bash handoff/colab_job.sh > job.log 2>&1 &      # everything
#   cd /content; setsid nohup bash handoff/colab_job.sh kpi > kpi.log 2>&1 &  # KPI tables only
# KPI matrix (3 worker modes x 4 policies, plus 0.25 m standoff waypoints), three replay videos,
# then our G1 in the worker's place (P2 and P1).
set -x
B=/content/carda/bvh/ws30-svo-mp4-bvh/ws30/xsens/xsens_003_WS30_2023_10_04_cropped_fixed.bvh
O=/content/out
export MUJOCO_GL=egl OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
cd /content/handoff
for m in asis hold away; do for p in P1 P1s P2 P3; do echo "$m $p 0"; done; done | cat - <(echo "asis P1 0.25"; echo "away P1 0.25") |
  xargs -P 6 -n 3 sh -c 'python carda_cell.py '"$B"' --mode $0 --policy $1 --standoff $2 --out '"$O"'/kpi_$0_$1_$2.json --trace '"$O"'/trace_$0_$1_$2.csv > '"$O"'/run_$0_$1_$2.log 2>&1'
[ "$1" = kpi ] && { echo DONE; exit; }
python carda_cell.py $B --mode asis --policy P1 --video $O/ws30_asis_P1.mp4 --xml $O/ws30_cell.xml > $O/vid_asis_P1.log 2>&1 &
python carda_cell.py $B --mode away --policy P1 --video $O/ws30_away_P1.mp4 > $O/vid_away_P1.log 2>&1 &
python carda_cell.py $B --mode asis --policy P2 --video $O/ws30_asis_P2.mp4 > $O/vid_asis_P2.log 2>&1 &
export PYTHONPATH=/content/robosim/mujoco_houses/src
python g1_cell.py $B --policy P2 --video $O/g1_ws30_P2.mp4 --out $O/g1_P2.json --trace $O/g1_P2.csv > $O/g1_P2.log 2>&1 &
python g1_cell.py $B --policy P1 --video $O/g1_ws30_P1.mp4 --out $O/g1_P1.json --trace $O/g1_P1.csv > $O/g1_P1.log 2>&1 &
wait
cd $O && for f in *.mp4; do ffmpeg -v error -y -i $f -vf "fps=1/6,scale=640:-1,tile=5x3" -frames:v 1 ${f%.mp4}_sheet.jpg; done
echo DONE
