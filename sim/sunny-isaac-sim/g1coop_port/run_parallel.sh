#!/bin/bash
# Render the film and the three close-up clips as parallel headless Kit processes (one container each, shots split
# into chunks of at most 6 s), then join the chunks with ffmpeg. Logs GPU use and load to OUT/gpu.log.
# Usage (on the VM): bash run_parallel.sh USD_DIR OUT [render_video.py options]   (both relative to ~/robot/g1coop)
# Concurrency is read from OUT/maxjobs (default 4) before each launch; raise it while running to add a process.
set -u
export ISAAC_SIM_IMAGE=nvcr.io/nvidia/isaac-sim:6.2.0
C="docker compose -p isim -f $HOME/IsaacSim/tools/docker/docker-compose.yml -f $HOME/robot/isaac-sim/docker-compose.override.yml"
R=$HOME/robot/g1coop; U=$1; O=$2; shift 2
mkdir -p "$R/$O"; cd "$R/$O" || exit 1
[ -f maxjobs ] || echo 4 > maxjobs
sudo setfacl -R -m u:1234:rwX "$R"
sudo find "$R" -type d -exec setfacl -d -m "u:1234:rwx,u:$(id -u):rwx" {} +
(while true; do echo "$(date +%T) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits) load $(cut -d' ' -f1 /proc/loadavg)"; sleep 5; done) >> gpu.log &
MON=$!
# film: Overview ThreeG1 TwoG1 OneG1; clips: *Close. shot:start:end in sim seconds
FILM="Overview:0:6 Overview:6:12 ThreeG1:0:6 ThreeG1:6:11 TwoG1:3:8 TwoG1:8:13 OneG1:0:6"
CLIPS="ThreeG1Close:0:6 ThreeG1Close:6:12 TwoG1Close:1:7.5 TwoG1Close:7.5:14 OneG1Close:0:6 OneG1Close:6:11"
t0=$(date +%s); pids=()
running() { local n=0; for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && n=$((n + 1)); done; echo $n; }
for j in $CLIPS $FILM; do
  IFS=: read -r s a b <<< "$j"; d=${s}_$a
  [ "$(ls "$d" 2>/dev/null | wc -l)" -ge "$(awk "BEGIN{print int(($b - $a) * 30 + .5)}")" ] && continue  # chunk already rendered
  while [ "$(running)" -ge "$(cat maxjobs)" ]; do sleep 5; done
  $C run --rm --entrypoint bash isaac-sim -c "cd /isaac-sim && ./python.sh /robot/g1coop/render_video.py /robot/g1coop/$U/factory.usd /robot/g1coop/$O/$d --shots $s --span $a $b $*" > "$d.log" 2>&1 &
  pids+=($!); echo "$(date +%T) started $d ($(running) running)"
  # Kit processes that load the stage at the same time stalled for minutes (CPU-bound, no frames), so each job
  # starts only after the previous one has rendered its first frame; rendering itself runs in parallel.
  while kill -0 $! 2>/dev/null && [ -z "$(ls "$d" 2>/dev/null)" ]; do sleep 5; done
done
wait "${pids[@]}"; kill $MON
echo "render wall time: $(( $(date +%s) - t0 )) s"
sudo chown -R "$(id -u):$(id -g)" .
grep -h -E "Traceback|Error:" ./*.log | grep -v -E "OmniHub|hoops|NVST" | sort | uniq -c | head
join() {  # join chunk folders, in order, into one mp4
  out=$1; shift; : > list.txt
  for d in "$@"; do
    ffmpeg -loglevel error -y -framerate 30 -i "$d/frame_%05d.png" -c:v libx264 -pix_fmt yuv420p -crf 18 "$d.mp4" && echo "file '$d.mp4'" >> list.txt
  done
  ffmpeg -loglevel error -y -f concat -safe 0 -i list.txt -c copy "$out"
  echo "$out: $(ffprobe -v error -show_entries format=duration -of csv=p=0 "$out") s"
}
join g1coop_factory_v2.mp4 Overview_0 Overview_6 ThreeG1_0 ThreeG1_6 TwoG1_3 TwoG1_8 OneG1_0
join close_3g1_tframe.mp4 ThreeG1Close_0 ThreeG1Close_6
join close_2g1_sheets.mp4 TwoG1Close_1 TwoG1Close_7.5
join close_1g1_tote.mp4 OneG1Close_0 OneG1Close_6
for v in g1coop_factory_v2 close_3g1_tframe close_2g1_sheets close_1g1_tote; do
  n=$(ffprobe -v error -count_frames -select_streams v -show_entries stream=nb_read_frames -of csv=p=0 $v.mp4)
  ffmpeg -loglevel error -y -i $v.mp4 -vf "select=not(mod(n\,$(( n / 12 )))),scale=640:-1,tile=4x3" -frames:v 1 ${v}_sheet.png
done
echo "peak GPU util and memory (MiB): $(cut -d' ' -f2 gpu.log | tr -d , | sort -n | tail -1) $(cut -d' ' -f3 gpu.log | sort -n | tail -1)"
echo "ALL DONE in $(( $(date +%s) - t0 )) s"
