#!/bin/bash
# Medium-quality framing stills, one per shot (its middle frame), 720p with captions, into shot_stills/SHOT.png.
# Usage (on the host): bash stills.sh [SHOT ...]   Env: USD (USD folder, default usd_v3), VENV
cd "$(dirname "$0")"
PY=${VENV:-$PWD/venv}/bin/python; U=${USD:-usd_v3}
S=${*:-Overview ThreeG1Close TwoG1Close OneG1Close CNCArm CleaningArm AssemblyArm QCPanda PackagingArm TwoArmPanel CourierLanes SpotInspection SpotQuality Shelving Dock}
rm -rf shot_stills_tmp; mkdir -p shot_stills  # stills of shots not named here are kept
bash in_isaac.sh shot_stills.log /robot/g1coop/render_video.py /robot/g1coop/$U/factory.usd /robot/g1coop/shot_stills_tmp \
  --mode PathTracing --spp 16 --subframes 4 --mid --shots $S
sudo chown -R "$(id -u):$(id -g)" shot_stills_tmp
i=0  # frames come out in shots.py order
for s in $($PY -c "import shots; print(' '.join(n for n, *_ in shots.SHOTS if n in '$S'.split()))"); do
  mv shot_stills_tmp/frame_$(printf %05d $i).png shot_stills/$s.png; i=$((i + 1))
done
rm -rf shot_stills_tmp
ls shot_stills
