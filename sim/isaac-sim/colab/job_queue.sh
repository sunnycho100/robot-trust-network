#!/bin/bash
# After the fleet recording: job mode (fleet.Jobs station locks). Three robots each load at the station (0, 0) in turn,
# then carry to their own drop-off. Uses a separate copy of the scene so the queued --goal run kept its tested file.
while ! grep -q FLEET_DONE /content/fleet_queue.log 2>/dev/null; do sleep 30; done
bash /content/isaac_run.sh job 50 /content/examples/multi_robot_scene_job.py \
  --robot h1 -4 0 0 0 0 0 --robot go2 0 -4 90 0 0 0 --robot anymal 4 -3 135 0 0 0 \
  --stop 0 0 0 3 --stop 0 3 3 0 --stop 1 0 0 2 --stop 1 -3 3 0 --stop 2 0 0 2 --stop 2 -3 -3 0 > /content/isaac_run_job.log 2>&1
echo JOB_DONE
