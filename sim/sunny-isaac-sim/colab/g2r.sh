#!/bin/bash
# Kills the diverged 2DGS run, reruns gs2d_run.sh at the metric-equivalent dist_lambda; if L1 is still over 0.1
# at step 4500 (past the step-3000 dist start and opacity reset), kills it and reruns with no distortion loss.
# Usage: setsid nohup bash /content/g2r.sh > /content/g2r.log 2>&1 &   (gs2d output in /content/gs2d_queue.log, ends GS2D_DONE)
pkill -f gs2d_queue.sh; pkill -f gs2d_run.sh; pkill -f simple_trainer_2dgs; sleep 10
mv /content/gs2d_work /content/gs2d_work_div1; mv /content/gs2d_queue.log /content/gs2d_queue_div1.log
setsid nohup bash /content/gs2d_run.sh 20000 0.0038 > /content/gs2d_queue.log 2>&1 &
l1() { python3 - <<'PY'
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
f = sorted(glob.glob("/content/gs2d_work/tb/events*"))
if not f: print("0 1"); raise SystemExit
e = EventAccumulator(f[-1]); e.Reload()
s = e.Scalars("train/l1loss") if "train/l1loss" in e.Tags()["scalars"] else []
print(s[-1].step if s else 0, max([x.value for x in s if x.step >= 4300] or [0]))
PY
}
while read st v < <(l1); [ "$st" -lt 4500 ]; do sleep 60; done
echo "step $st max L1 since 4300: $v"
if python3 -c "import sys; sys.exit(0 if $v > 0.1 else 1)"; then
  echo "DIVERGED again, rerun with dist_lambda 0"
  pkill -f gs2d_run.sh; pkill -f simple_trainer_2dgs; sleep 10
  mv /content/gs2d_work /content/gs2d_work_div2; mv /content/gs2d_queue.log /content/gs2d_queue_div2.log
  setsid nohup bash /content/gs2d_run.sh 20000 0 > /content/gs2d_queue.log 2>&1 &
else echo "STABLE"; fi
echo G2R_DONE
