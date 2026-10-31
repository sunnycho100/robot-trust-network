#!/usr/bin/env bash
# Download CarDA subset A (Zenodo 10.5281/zenodo.14644367, CC BY 4.0) and lay it out as data/carda/<ws>/{xsens,mp4,annotation*.xlsx}.
# ~5.1 GB download; .svo stereo files are skipped on extraction (ZED SDK needed to read them).
set -euo pipefail
cd "$(dirname "$0")/.."
RAW=data/_raw; OUT=data/carda
mkdir -p "$RAW" "$OUT"
B="https://zenodo.org/records/14644367/files"
curl -fSL -o "$OUT/cameras_calibration.xlsx" "$B/cameras_calibration.xlsx?download=1"
for ws in ws10 ws20 ws30; do
  curl -fSL -o "$RAW/$ws.rar" "$B/${ws}%20-%20svo%20-%20mp4%20-%20bvh.rar?download=1"
  mkdir -p "$RAW/$ws" && bsdtar -xf "$RAW/$ws.rar" -C "$RAW/$ws" --exclude '*.svo' && rm "$RAW/$ws.rar"
  src=$(dirname "$(find "$RAW/$ws" -name 'annotation*.xlsx' | head -1)")
  mkdir -p "$OUT/$ws"
  ln -sfn "$(cd "$src" && pwd)/$(ls "$src" | grep -E '^xsens' | head -1)" "$OUT/$ws/xsens"
  ln -sfn "$(cd "$src" && pwd)/$(ls "$src" | grep -E 'mp4$' | head -1)" "$OUT/$ws/mp4"
  ln -sfn "$(cd "$src" && pwd)/$(ls "$src" | grep -E '^annotation.*xlsx$')" "$OUT/$ws/"
done
echo "CarDA ready under $OUT"
