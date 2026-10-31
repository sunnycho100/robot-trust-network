#!/bin/bash
# Full CarDA line recording on a Colab L4: render, overlay, mp4, contact sheet. Usage: bash colab_job.sh [extra carda_line.py args]
# Needs colab/isaac_pip2.sh first; carda_line.py, overlay.py and colab_run.sh in /content.
cd /content && rm -rf rec ov && bash colab_run.sh /content/carda_line.py /content/rec "$@"
python3 overlay.py rec ov
ffmpeg -loglevel error -y -framerate 30 -pattern_type glob -i "ov/frame_*.png" -c:v libx264 -pix_fmt yuv420p -crf 20 carda_line.mp4
ffmpeg -loglevel error -y -i carda_line.mp4 -vf "select=not(mod(n\,120)),scale=640:-1,tile=4x2" -frames:v 1 carda_line_sheet.jpg
cp rec/meta.json carda_line_meta.json; ls -la carda_line*; echo JOB_DONE
