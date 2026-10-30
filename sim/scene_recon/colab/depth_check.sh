#!/bin/bash
# Downloads ETH3D's laser depth maps and the distorted DSLR calibration for delivery_area, then runs depth_check.py.
set -eu
cd /content/eth3d
[ -f depth.ok ] || { wget -q https://www.eth3d.net/data/delivery_area_dslr_depth.7z && 7z x -y delivery_area_dslr_depth.7z > /dev/null && touch depth.ok; }
[ -f delivery_area/dslr_calibration_jpg/cameras.txt ] || { wget -q https://www.eth3d.net/data/delivery_area_dslr_jpg.7z && 7z x -y delivery_area_dslr_jpg.7z 'delivery_area/dslr_calibration_jpg/*' > /dev/null; }
head -4 delivery_area/dslr_calibration_jpg/cameras.txt | tail -1 | cut -c1-80
python /content/robosim/sim/scene_recon/colab/depth_check.py
