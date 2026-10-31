#!/bin/bash
# NuRec cafe scene (Nova Carter capture, particle_sh_optimized.usdz, 628 MB) from Hugging Face. Usage (on the VM, not in the container): bash get_nurec_cafe.sh [DEST, default ~/robot/nurec]
# The dataset is gated. You accept its terms on the website and run `hf auth login` yourself; this script never reads or passes a token.
set -eu
REPO=nvidia/PhysicalAI-Robotics-NuRec
DEST=${1:-$HOME/robot/nurec}
FILE=nova_carter-cafe/particle_sh_optimized.usdz
LOGIN="Accept the terms at https://huggingface.co/datasets/$REPO (signed in), run 'hf auth login', then rerun this script."

# unverified: pip may refuse on Ubuntu 24.04 (externally managed Python); pipx install huggingface_hub works there.
command -v hf >/dev/null || { echo "The hf CLI is missing. Install it (the dataset card's way): python3 -m pip install -U huggingface_hub"; exit 1; }
# Recent hf exits 1 when logged out; older versions print "Not logged in" and exit 0, so check both.
if ! who=$(hf auth whoami 2>&1) || grep -qi "not logged in" <<<"$who"; then
  echo "Not logged in to Hugging Face. $LOGIN"
  exit 1
fi
echo "$who"

# The dataset card loads particle_sh_optimized.usdz directly with no extra launch flags, and only asks for whole
# folders for the XGRIDS and Zurich scenes (external .nurec files), so this one file is the whole cafe.
# unverified: that the cafe .usdz has no outside references (inferred from the card; the files are gated).
hf download "$REPO" "$FILE" --repo-type dataset --revision main --local-dir "$DEST" ||
  { echo "Download failed. A 403 or 'gated' error means this account has not accepted the terms. $LOGIN"; exit 1; }
ls -lh "$DEST/$FILE"
