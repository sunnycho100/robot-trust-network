# gsplat 512d366 2DGS backward: "grad.depths must be contiguous" on torch 2.9 (the 2DGS trainer's depth loss passes
# a strided gradient). Make the projection backward grads contiguous. Idempotent.
from pathlib import Path
p = Path("/content/gsplat/gsplat/cuda/_wrapper.py")
s = p.read_text()
fix = "        v_means2d, v_depths, v_ray_transforms, v_normals = (t.contiguous() for t in (v_means2d, v_depths, v_ray_transforms, v_normals))\n"
if fix not in s:
    a = "    def backward(cls, ctx, v_radii, v_means2d, v_depths, v_ray_transforms, v_normals):\n"
    assert s.count(a) == 1
    s = s.replace(a, a + fix)
    b = "        v_batch_ids,\n        v_camera_ids,\n        v_gaussian_ids,\n        v_indptr,\n        v_radii,\n        v_means2d,\n        v_depths,\n        v_ray_transforms,\n        v_normals,\n    ):\n"
    assert s.count(b) == 1
    s = s.replace(b, b + fix)
    p.write_text(s)
print("PATCHED", s.count(fix))
