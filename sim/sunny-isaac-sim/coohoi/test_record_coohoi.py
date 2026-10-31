# Checks the depth composite in record_coohoi.py with numpy only (no Isaac Gym). Usage: python test_record_coohoi.py
import numpy as np
import record_coohoi as r

r.H, r.W = 2, 3
base = np.zeros((2, 12), np.uint8); base[:] = 10                # base render: flat colour 10
part = np.zeros((2, 12), np.uint8); part[:] = 200               # partner render: flat colour 200
ground = -5.0 * np.ones((2, 3), np.float32)                     # both see the same ground at 5 m
d_base, d_part = ground.copy(), ground.copy()
d_part[0, 0] = -2.0                                             # partner humanoid, nearer than the ground
d_part[0, 1] = -2.0                                             # partner ghost box, nearer but masked
d_base[1, 2] = -1.0; d_part[1, 2] = -2.0                        # base humanoid in front of the partner's
d_base[1, 0] = -np.inf; d_part[1, 0] = -3.0                     # base sees sky, partner sees something
seg = np.zeros((2, 3), np.uint32); seg[0, 1] = r.GHOST
out = r.composite(base, d_base, part, d_part, seg)
want = np.array([[200, 10, 10], [200, 10, 10]])
assert (out[..., 0] == want).all(), out[..., 0]
assert out.shape == (2, 3, 4) and out.dtype == np.uint8
print('composite ok')
