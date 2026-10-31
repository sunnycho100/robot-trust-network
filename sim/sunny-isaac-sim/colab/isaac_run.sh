#!/bin/bash
# Colab A step 3: headless Isaac Sim 6.2 (pip) on the L4, recording the stock H1 example. Usage: bash isaac_run.sh NAME SECONDS SCRIPT [args]
NAME=$1; SECS=$2; shift 2
mkdir -p /usr/share/vulkan/icd.d  # missing on some Colab VMs (the echo then fails and Vulkan finds no driver)
echo '{"file_format_version":"1.0.1","ICD":{"library_path":"/usr/lib64-nvidia/libEGL_nvidia.so.0","api_version":"1.4.312"}}' > /usr/share/vulkan/icd.d/nvidia_icd.json
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg OMNI_KIT_ACCEPT_EULA=YES XDG_RUNTIME_DIR=/tmp/xdg LD_LIBRARY_PATH=/usr/lib64-nvidia:$LD_LIBRARY_PATH
mkdir -p /tmp/xdg /content/rec
SP=/content/isim/lib/python3.12/site-packages/isaacsim
cd $SP
date; timeout 1500 /content/isim/bin/python /content/record_demo.py /content/rec/$NAME $SECS "$@" &
PID=$!; WANT=$(python3 -c "print(int(float('$SECS')*30))")
# ponytail: post_quit can leave the app spinning on Colab, so stop it once every frame is on disk
while kill -0 $PID 2>/dev/null; do [ "$(ls /content/rec/$NAME 2>/dev/null | wc -l)" -ge "$WANT" ] && { sleep 15; kill $PID; }; sleep 5; done
echo "exit"; date
n=$(ls /content/rec/$NAME 2>/dev/null | wc -l); echo "frames $n"
[ "$n" -gt 0 ] && ffmpeg -loglevel error -y -framerate 30 -pattern_type glob -i "/content/rec/$NAME/frame_*.png" -c:v libx264 -pix_fmt yuv420p -crf 22 /content/rec/$NAME.mp4 && \
  ffmpeg -loglevel error -y -i /content/rec/$NAME.mp4 -vf 'select=not(mod(n\,60)),scale=640:-1,tile=3x2' -frames:v 1 /content/rec/${NAME}_sheet.png
echo RUN_DONE
