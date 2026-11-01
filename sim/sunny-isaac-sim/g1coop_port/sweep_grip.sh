#!/bin/bash
# Run side_grip.py probes in parallel on the VM, one process per carry and variant. Usage: bash sweep_grip.sh OUT "scenario:squeeze:height:top[:reach[:table cx,cy,hx,hy]]" ...
# A field left empty keeps the default. Reports land in OUT/<scenario>_<squeeze>_<height>_<top>.txt
set -u
cd "$(dirname "$0")"; O=$1; shift; mkdir -p "$O"
for v in "$@"; do
  IFS=: read -r sc sq h t r tb <<< "$v"
  a="--scenario $sc --duration 30"
  [ -n "$sq" ] && a="$a --squeeze $sq"; [ -n "$h" ] && a="$a --height $h"; [ -n "$t" ] && a="$a --top $t"; [ -n "$r" ] && a="$a --reach $r"; [ -n "$tb" ] && a="$a --table ${tb//,/ }"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=code/mujoco_houses/src /data/g1coop_venv/bin/python side_grip.py $a > "$O/${sc}_${sq}_${h}_${t}_${r}_${tb}.txt" 2>&1 &
done
wait
for f in "$O"/*.txt; do echo "$(basename "$f") $(grep -E '"state"|pass_probe|transit_supported|lowest|max_pen|tilt|displacement' "$f" | tr -d ' \n')"; done
