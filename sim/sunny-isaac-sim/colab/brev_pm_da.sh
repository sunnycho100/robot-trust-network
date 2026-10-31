#!/bin/bash
# Brev L40S version of pm_da.sh steps 1 to 3: COLMAP 4.2.1 PatchMatch from the colmap/colmap docker image (no build),
# then eth3d_prep.py's Poisson mesh in a small open3d venv (/data/da/o3d). Data from grut_setup.sh's two 7z files.
# Usage: setsid nohup bash brev_pm_da.sh > /data/da/pm.log 2>&1 < /dev/null &   (ends PM_DONE)
set -x
D=/data/da/eth3d/delivery_area; W=/data/da/pm_da; PY=/data/da/o3d/bin/python; CODE=$HOME/robot/recon_code
mkdir -p $W; t=$(date +%s)
# ponytail: 1600 px, window 5, 5 iterations (pm_da.sh: 3200, 7, 7). 3200 px ran 8 s per iteration here, about 80 min for
# the 44 images; this is about 8x less work. Raise S for a sharper collider when time allows.
S=1600
# The image is CUDA 12.9 with PTX only, which the 565 driver (CUDA 12.7) cannot JIT: skip the image's CUDA check and
# load its forward-compatibility libcuda (575) from /usr/local/cuda/compat (datacenter GPUs only; the L40S is one)
COLMAP="docker run --rm --gpus all -e NVIDIA_DISABLE_REQUIRE=1 -e LD_LIBRARY_PATH=/usr/local/cuda/compat:/usr/local/cuda/lib64 -u $(id -u):$(id -g) -v /data/da:/data/da colmap/colmap:latest colmap"
if [ ! -f $W/fused.ply ]; then
$COLMAP image_undistorter --image_path $D/images --input_path $D/dslr_calibration_undistorted \
  --output_path $W/dense --output_type COLMAP --max_image_size 3200
$COLMAP patch_match_stereo --workspace_path $W/dense --workspace_format COLMAP --PatchMatchStereo.geom_consistency true \
  --PatchMatchStereo.max_image_size $S --PatchMatchStereo.window_radius 5 --PatchMatchStereo.num_iterations 5 2>&1 | grep -v "^Processing\|^  " | tail -20
$COLMAP stereo_fusion --workspace_path $W/dense --workspace_format COLMAP --input_type geometric --output_path $W/fused.ply \
  --StereoFusion.min_num_pixels 3 --StereoFusion.max_image_size $S 2>&1 | tail -5
fi
echo "TIME patchmatch $(( $(date +%s) - t ))" | tee -a $W/timing.txt
until grep -q VENV_DONE /data/da/venv.log 2>/dev/null; do sleep 10; done
t=$(date +%s); $PY $CODE/eth3d_prep.py $D $W 2>&1 | tail -3
echo "TIME mesh $(( $(date +%s) - t ))" | tee -a $W/timing.txt
ls -la $W $W/frames
echo PM_DONE
