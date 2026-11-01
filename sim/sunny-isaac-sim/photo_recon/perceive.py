"""Single-photo geometry on a Colab L4: Depth Pro metric depth, Grounding DINO + SAM 2 masks, floor plane, per-object 3D stats.
Usage: python perceive.py photo.jpg OUTDIR   (photo is the 1840 px copy; EXIF focal 17 mm on a 35.9 mm wide D800 sensor)
Writes OUTDIR/perception.json, seg_overlay.jpg, depth.jpg, topdown.png.
"""
import json
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

PHOTO, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
dev = "cuda"
img = Image.open(PHOTO).convert("RGB")
W, H = img.size
F_EXIF = 17.0 / 35.9 * W   # px, from EXIF (Nikon D800, 17 mm)

# ---------------------------------------------------------------- depth (Depth Pro)
from transformers import DepthProForDepthEstimation, DepthProImageProcessorFast

dp = DepthProImageProcessorFast.from_pretrained("apple/DepthPro-hf")
dm = DepthProForDepthEstimation.from_pretrained("apple/DepthPro-hf", torch_dtype=torch.float16).to(dev)
with torch.no_grad():
    o = dm(**dp(images=img, return_tensors="pt").to(dev, torch.float16))
pp = dp.post_process_depth_estimation(o, target_sizes=[(H, W)])[0]
depth = pp["predicted_depth"].float().cpu().numpy()
f_pred = float(pp["focal_length"])
# ponytail: Depth Pro depth scales linearly with focal, so swap in the EXIF focal (ground truth) and keep both numbers
depth_exif = depth * (F_EXIF / f_pred)
print(f"focal: Depth Pro {f_pred:.1f}px, EXIF {F_EXIF:.1f}px, ratio {F_EXIF / f_pred:.3f}")
del dm
torch.cuda.empty_cache()

# ---------------------------------------------------------------- detection + masks
LABELS = ["forklift", "pallet of orange sacks", "traffic cone", "yellow guardrail", "yellow staircase", "modular office",
          "white wall", "brick wall", "roll-up door", "column", "yellow floor marking", "floor", "ceiling"]
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, Sam2Model, Sam2Processor

gp = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
gm = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to(dev)
KMAX = {"traffic cone": 4, "yellow guardrail": 6, "column": 3, "roll-up door": 3, "white wall": 3, "brick wall": 3}


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix + 1e-9)


dets = []
for lab in LABELS:   # one prompt per label: avoids phrase bleed between labels
    inp = gp(images=img, text=lab + ".", return_tensors="pt").to(dev)
    with torch.no_grad():
        go = gm(**inp)
    r = gp.post_process_grounded_object_detection(go, inp.input_ids, threshold=0.3, text_threshold=0.25, target_sizes=[(H, W)])[0]
    keep = []   # greedy NMS, at most KMAX boxes per label
    for b, s in sorted(zip(r["boxes"].tolist(), r["scores"].tolist()), key=lambda t: -t[1]):
        if s >= 0.3 and len(keep) < KMAX.get(lab, 2) and all(iou(b, k[0]) < 0.5 for k in keep):
            keep.append((b, s))
    dets += [dict(label=lab, score=round(s, 3), box=[round(v, 1) for v in b]) for b, s in keep]
del gm
sp = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-large")
sm = Sam2Model.from_pretrained("facebook/sam2.1-hiera-large").to(dev)
print(len(dets), "detections", flush=True)
masks = []
for i in range(0, len(dets), 8):
    inp = sp(images=img, input_boxes=[[d["box"] for d in dets[i:i + 8]]], return_tensors="pt").to(dev)
    with torch.no_grad():
        so = sm(**inp, multimask_output=False)
    # ponytail: SAM 2 resizes straight to 1024x1024, so upsample the logits ourselves (post_process_masks ran out of RAM here)
    lg = torch.nn.functional.interpolate(so.pred_masks[0].float(), size=(H, W), mode="bilinear", align_corners=False)
    masks += list((lg[:, 0] > 0).cpu().numpy())
masks = np.stack(masks)   # (n, H, W) bool

# ---------------------------------------------------------------- lift to 3D, floor plane, world frame
u, v = np.meshgrid(np.arange(W), np.arange(H))
P = np.stack([(u - W / 2) / F_EXIF * depth_exif, (v - H / 2) / F_EXIF * depth_exif, depth_exif], -1)   # OpenCV camera frame
floor_m = np.zeros((H, W), bool)
for d, m in zip(dets, masks):
    if d["label"] in ("floor", "yellow floor marking"):
        floor_m |= m
for d, m in zip(dets, masks):   # objects standing on the floor are not floor
    if d["label"] not in ("floor", "yellow floor marking"):
        floor_m &= ~m
fp = P[floor_m & (depth_exif < 25)]
rng = np.random.default_rng(0)
best = (0, None)
for _ in range(500):   # RANSAC plane
    a, b, c = fp[rng.choice(len(fp), 3, replace=False)]
    n = np.cross(b - a, c - a)
    if np.linalg.norm(n) < 1e-9:
        continue
    n /= np.linalg.norm(n)
    inl = np.abs((fp - a) @ n) < 0.04
    if inl.sum() > best[0]:
        best = (inl.sum(), inl)
q = fp[best[1]]
c0 = q.mean(0)
n = np.linalg.svd(q - c0, full_matrices=False)[2][2]   # full_matrices=True builds an N x N U and runs out of RAM
if n[1] > 0:   # up is -y in OpenCV
    n = -n
d0 = -n @ c0
h = float(d0)   # camera height above the floor
zw = n
yw = np.array([0, 0, 1.0]) - n[2] * n
yw /= np.linalg.norm(yw)
xw = np.cross(yw, zw)
R = np.stack([xw, yw, zw])   # world-from-camera rotation (rows are world axes in camera coords)
Pw = (P.reshape(-1, 3) @ R.T).reshape(H, W, 3)
Pw[..., 2] += h
pitch = float(np.degrees(np.arctan2(R[2, 2], R[1, 2])))   # optical axis vs horizon, negative = looking down
roll = float(np.degrees(np.arctan2(R[2, 0], R[0, 0])))
# USD camera looks down -Z with +Y up: columns are cam x, -cam y, -cam z in world
T = np.eye(4)
T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = R[:, 0], -R[:, 1], -R[:, 2], [0, 0, h]
print(f"camera height {h:.2f} m, pitch {pitch:.1f} deg, roll {roll:.1f} deg, floor inliers {best[0]}/{len(fp)}")

objs = []
for i, (d, m) in enumerate(zip(dets, masks)):
    pts = Pw[m & (depth_exif < 60)]
    if len(pts) < 50:
        continue
    lo, med, hi = np.percentile(pts, [5, 50, 95], axis=0)
    xy = pts[:, :2] - med[:2]
    ev, evec = np.linalg.eigh(np.cov(xy.T))
    ax = evec[:, 1]
    yaw = float(np.degrees(np.arctan2(ax[1], ax[0])))
    ext = np.percentile(xy @ evec, 95, axis=0) - np.percentile(xy @ evec, 5, axis=0)
    objs.append(dict(id=i, **d, n=int(m.sum()), med=med.round(2).tolist(), p5=lo.round(2).tolist(), p95=hi.round(2).tolist(),
                     zmax=round(float(np.percentile(pts[:, 2], 98)), 2), yaw_major=round(yaw, 1),
                     ext_major_minor=[round(float(ext[1]), 2), round(float(ext[0]), 2)]))

json.dump(dict(W=W, H=H, f_exif=F_EXIF, f_depthpro=f_pred, cam_height=h, pitch=pitch, roll=roll,
               R_world_from_cam=R.tolist(), usd_cam_to_world=T.tolist(), objects=objs), open(f"{OUT}/perception.json", "w"), indent=1)
np.savez_compressed(f"{OUT}/pointcloud.npz", xyz=Pw[::4, ::4].astype(np.float16), rgb=np.asarray(img)[::4, ::4])

# ---------------------------------------------------------------- evidence images
ov = np.asarray(img).astype(float)
pal = rng.integers(40, 255, (len(dets), 3))
for i, m in enumerate(masks):
    ov[m] = ov[m] * 0.5 + pal[i] * 0.5
ovi = Image.fromarray(ov.astype(np.uint8))
dr = ImageDraw.Draw(ovi)
for i, d in enumerate(dets):
    dr.rectangle(d["box"], outline=tuple(int(c) for c in pal[i]), width=3)
    dr.text((d["box"][0] + 4, d["box"][1] + 4), f"{i}:{d['label']} {d['score']:.2f}", fill=(255, 255, 255))
ovi.save(f"{OUT}/seg_overlay.jpg", quality=90)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.imsave(f"{OUT}/depth.jpg", np.clip(depth_exif, 0, 40), cmap="turbo")
fig, axp = plt.subplots(figsize=(10, 10))
s = Pw[::6, ::6].reshape(-1, 3)
cc = np.asarray(img)[::6, ::6].reshape(-1, 3) / 255
k = (depth_exif[::6, ::6].reshape(-1) < 40)
axp.scatter(s[k, 0], s[k, 1], c=cc[k], s=1)
for o in objs:
    axp.text(o["med"][0], o["med"][1], f"{o['id']}:{o['label']}", fontsize=7)
axp.plot([0], [0], "r^")
axp.set_aspect("equal")
axp.set_xlabel("x (m, right)")
axp.set_ylabel("y (m, forward)")
axp.set_title("Top-down, floor frame (camera at red triangle)")
fig.savefig(f"{OUT}/topdown.png", dpi=110)
print("PERCEIVE_DONE")
