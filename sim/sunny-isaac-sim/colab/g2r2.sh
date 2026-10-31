#!/bin/bash
# 2DGS divergence, second test. Runs 1 (dist_lambda 1e-2) and 2 (0.0038) collapsed at the same step (3200 to 3300) with
# the same L1 curve, so the weight is not the cause; step 3000 is both dist_start_iter and the first opacity reset.
# g2r.sh's fallback (dist_lambda 0) is the control. If it also collapses by step 4500, the reset is the trigger:
# kill it and rerun at 0.0038 with the reset off (--reset_every 100000, refinement still stops at 15k).
# Usage: setsid nohup bash /content/g2r2.sh > /content/g2r2.log 2>&1 &   (gs2d output in /content/gs2d_queue.log)
l1() { python3 - <<'PY'
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
f = sorted(glob.glob("/content/gs2d_work/tb/events*"))
if not f: print("0 1"); raise SystemExit
e = EventAccumulator(f[-1]); e.Reload()
s = e.Scalars("train/l1loss") if "train/l1loss" in e.Tags()["scalars"] else []
print(s[-1].step if s else 0, max([x.value for x in s if x.step >= 3300] or [0]))
PY
}
while read st v < <(l1 2>/dev/null); [ "$st" -lt 4500 ]; do sleep 60; done
echo "dist 0 run: step $st max L1 since 3300: $v"
if python3 -c "import sys; sys.exit(0 if $v > 0.1 else 1)"; then
  echo "COLLAPSED with no distortion loss too: opacity reset is the trigger, rerun with the reset off"
  pkill -f gs2d_run; pkill -f simple_trainer_2dgs; sleep 10
  mv /content/gs2d_work /content/gs2d_work_div3; mv /content/gs2d_queue.log /content/gs2d_queue_div3.log
  setsid nohup bash /content/gs2d_run_v2.sh 20000 0.0038 --reset_every 100000 > /content/gs2d_queue.log 2>&1 &
else echo "STABLE with no distortion loss: the distortion term is the trigger, keep this run"; fi
echo G2R2_DONE
