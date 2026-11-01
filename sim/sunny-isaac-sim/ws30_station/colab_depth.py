# Metric depth for the WS30 station frames on a Colab L4 (Depth Anything V2 Metric Indoor Large, HF transformers).
# Usage: colab upload depth_in.tgz, then colab exec -f colab_depth.py; writes /content/depth_out.tgz (float16 .npy, metres).
import glob
import os
import subprocess

import numpy as np
import torch
from PIL import Image
from transformers import pipeline

os.chdir("/content")
subprocess.run("tar xzf depth_in.tgz", shell=True, check=True)
pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
                device=0, torch_dtype=torch.float16)
os.makedirs("depth_out", exist_ok=True)
for f in sorted(glob.glob("depth_in/*.png")):
    im = Image.open(f).convert("RGB")
    d = pipe(im)["predicted_depth"].squeeze().float().cpu().numpy()
    if d.shape != (im.height, im.width):
        d = np.array(Image.fromarray(d).resize(im.size, Image.BILINEAR))
    np.save("depth_out/" + os.path.basename(f)[:-4] + ".npy", d.astype(np.float16))
    print(os.path.basename(f), d.shape, float(d.min()), float(np.median(d)), float(d.max()))
subprocess.run("tar czf depth_out.tgz depth_out", shell=True, check=True)
print("DEPTH_DONE")
