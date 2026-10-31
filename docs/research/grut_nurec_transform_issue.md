# Draft issue for nv-tlabs/3dgrut (not posted)

Checked against upstream main on 2026-10-31 (head 2026-10-28, #277): still present. Post it only if you want to.

**Title:** NuRec USDZ export writes the normalizing transform transposed (translation lands in the projective column)

`threedgrut/export/usd/nurec/serializer.py`, line 142:

```python
matrix_op.Set(Gf.Matrix4d(*corrected_matrix.flatten()))
```

`corrected_matrix` is a numpy 4x4 in the column-vector convention (translation in the last column), but
`Gf.Matrix4d` reads row-major in USD's row-vector convention (translation in the last row). With a non-identity
normalizing transform, as `train.py` exports by default for COLMAP data, the translation ends up in the
projective column. On ETH3D delivery_area the exported Volume's last column was (0.94, 3.17, -3.48), the mean
camera centre, so the splat renders distorted and a collision mesh in the dataset frame does not line up.

Also, `--no-transform` is ignored on the NuRec path (it only reaches the standard ParticleField writer).

Fix: `Gf.Matrix4d(*corrected_matrix.T.flatten())` (or the repo's own numpy-to-Gf helper), and pass the
`--no-transform` flag through to the NuRec serializer.

Repro: train any COLMAP scene, `python -m threedgrut.export.scripts.export_usd -c ckpt_last.pt -o out.usdz --format nurec`,
then read the Volume prim's `xformOp:transform`.
