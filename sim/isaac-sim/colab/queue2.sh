#!/bin/bash
# Colab queue 2: after queue 1, rerun the multi-robot recording with the gcd physics dt.
while ! grep -q QUEUE1_DONE /content/queue1.log; do sleep 15; done
rm -rf /content/rec/multi
bash /content/isaac_run.sh multi 20 /content/examples/multi_robot_scene.py > /content/isaac_run_multi.log 2>&1
echo QUEUE2_DONE
