#!/bin/bash
# Run on the Mac while Colab jobs run: every few minutes, pack the small results on the VM (run logs, laser scores,
# check and walk reports, previews) and download them, so a dropped session can't wipe them.
# Usage: bash pull_results.sh SESSION [MINUTES]   (into data/colab_results, which git ignores)
set -u
S=$1
OUT="$(cd "$(dirname "$0")/.." && pwd)/data/colab_results"
mkdir -p "$OUT"
PACK='cd /content && find . -maxdepth 1 \( -name "*.log" -o -name "*.json" -o -name "*.jpg" \) > /tmp/pull.txt; find work -maxdepth 4 \( -name "eval_*.txt" -o -name "triangulation.txt" -o -name "timing.txt" -o -name "*.json" \) 2>/dev/null | grep -v -e /dense/ -e /images/ >> /tmp/pull.txt; tar czf results.tgz -T /tmp/pull.txt'
limit() { perl -e 'alarm shift; exec @ARGV' "$@"; }  # a busy VM can leave a call hanging
while true; do
  printf 'import subprocess\nsubprocess.run(r%s, shell=True)\nprint("PACKED")\n' "'$PACK'" | limit 180 colab exec -s "$S" 2>&1 | grep -q PACKED &&
    limit 600 colab download -s "$S" /content/results.tgz "$OUT/results.tgz" > /dev/null 2>&1 && tar xzf "$OUT/results.tgz" -C "$OUT" &&
    echo "$(date +%H:%M) pulled $(tar tzf "$OUT/results.tgz" | wc -l | tr -d ' ') files" || echo "$(date +%H:%M) pull failed"
  sleep $(( ${2:-10} * 60 ))
done
