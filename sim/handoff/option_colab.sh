#!/bin/bash
# Colab job for SIM_OPTIONS_ABC option C (sim/handoff/option_runner.py). Needs on the VM:
#   /content/robot_consult_demo   the teammate's pipeline (branch line-video-to-robot-cell, read only)
#   /content/option_runner.py     this folder's runner
# Run as: cd /content; setsid nohup bash option_colab.sh > job.log 2>&1 &
set -x
R=/content/robot_consult_demo
cd $R
pip install -q $(grep -v mediapipe requirements.txt)
apt-get -qq install -y libarchive-tools aria2 > /dev/null
# CarDA subset A (Zenodo 14644367, CC BY 4.0), laid out as scripts/fetch_carda.sh does; aria2 in parallel is
# about 10x faster than its sequential curl from Colab
RAW=data/_raw; OUT=data/carda; B="https://zenodo.org/records/14644367/files"
mkdir -p $RAW $OUT
curl -fsSL -o $OUT/cameras_calibration.xlsx "$B/cameras_calibration.xlsx?download=1"
for ws in ws10 ws20 ws30; do
  aria2c -q -c -x 8 -s 8 -k 20M -d $RAW -o $ws.rar "$B/${ws}%20-%20svo%20-%20mp4%20-%20bvh.rar?download=1" &
done
wait
for ws in ws10 ws20 ws30; do
  mkdir -p "$RAW/$ws" && bsdtar -xf "$RAW/$ws.rar" -C "$RAW/$ws" --exclude '*.svo'
  src=$(dirname "$(find "$RAW/$ws" -name 'annotation*.xlsx' | head -1)")
  mkdir -p "$OUT/$ws"
  ln -sfn "$(cd "$src" && pwd)/$(ls "$src" | grep -E '^xsens' | head -1)" "$OUT/$ws/xsens"
  ln -sfn "$(cd "$src" && pwd)/$(ls "$src" | grep -E 'mp4$' | head -1)" "$OUT/$ws/mp4"
  ln -sfn "$(cd "$src" && pwd)/$(ls "$src" | grep -E '^annotation.*xlsx$')" "$OUT/$ws/"
done
export OMP_NUM_THREADS=2 MUJOCO_GL=egl
python scripts/run_all.py --config config/demo_regress.yaml      # out_regress/, about 7 min
ln -sfn out_regress out                                          # export.py copies human_asis from out/s1
python scripts/export_option_packages.py                         # out_options/A|B|C, their baseline replays
python tests/validate.py --config config/demo_regress.yaml

cd /content; export OMP_NUM_THREADS=1
O=/content/final; mkdir -p $O
python option_runner.py selfcheck > $O/selfcheck.log 2>&1
python option_runner.py baseline --rcd $R --option C --out $O/baseline_check.json
python option_runner.py run --rcd $R --option C --jobs 11 --out $O > $O/run.log 2>&1   # 540 runs, about 8 min
python option_runner.py contrast --rcd $R --option C --out $O > $O/contrast.log 2>&1 &
export OMP_NUM_THREADS=2
python option_runner.py video --rcd $R --station WS10 --policy P1 --after stays --draw 0 --out $O/ws10_stays_P1.mp4 &
python option_runner.py video --rcd $R --station WS10 --policy P3 --after stays --draw 0 --out $O/ws10_stays_P3.mp4 &
python option_runner.py video --rcd $R --station WS10 --policy P3 --after leaves --draw 0 --out $O/ws10_leaves_P3.mp4 &
python option_runner.py sidebyside --rcd $R --station WS10 --policy P3 --after leaves --draw 0 --every 0.2 --t-end 110 \
  --out $O/ws10_human_vs_cell.mp4 &
wait
tar czf /content/final.tgz -C /content final
