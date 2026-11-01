"""How deep any G1 mesh vertex goes inside each payload's collision boxes, over the first 20 s of a recording.
Usage: python check_penetration.py REC_DIR   (from record_mujoco.py). Collision meshes count too; 0 cm = no overlap."""
import sys, numpy as np, mujoco
rec = sys.argv[1]
m = mujoco.MjModel.from_binary_path(rec + '/plant.mjb'); P = np.load(rec + '/poses.npz')
names = list(P['body_names']); xp, xq = P['xpos'], P['xquat']
def R(q): r = np.zeros(9); mujoco.mju_quat2Mat(r, q.astype(float)); return r.reshape(3, 3)
for g in ('sofa', 'crate', 'frame'):
    pb = names.index(g + '_payload')
    boxes = [i for i in range(m.ngeom) if m.geom_bodyid[i] == pb]
    worst = {}
    for i in range(m.ngeom):
        b = m.geom_bodyid[i]; bn = m.body(b).name
        if not bn.startswith(g + '_') or bn == g + '_payload' or m.geom_type[i] != 7 or m.geom_group[i] > 2: continue
        boxes_live = [k for k in boxes if m.geom_contype[k] or m.geom_conaffinity[k]]  # skip removed geoms
        d = m.geom_dataid[i]; v = m.mesh_vert[m.mesh_vertadr[d]:m.mesh_vertadr[d] + m.mesh_vertnum[d]]
        gl = R(m.geom_quat[i]); v = v @ gl.T + m.geom_pos[i]
        for f in range(0, 600, 10):
            w = v @ R(xq[f, b]).T + xp[f, b]
            loc = (w - xp[f, pb]) @ R(xq[f, pb])
            for k in boxes_live:
                c = loc - m.geom_pos[k]; s = m.geom_size[k]
                depth = np.min(s - np.abs(c), axis=1)  # >0 inside
                if 'rubber_hand' in bn:  # a hand over the box: inside its footprint, up to 10 cm above the top
                    over = (np.abs(c[:, :2]) < s[:2]).all(axis=1) & (c[:, 2] > s[2]) & (c[:, 2] < s[2] + .1)
                    if over.any():
                        key_o = bn.replace(g + '_', '') + ' OVER ' + m.geom(k).name.replace(g + '_', '')
                        worst[key_o] = max(worst.get(key_o, (0, 0, 0)), (int(over.sum()) / 1000, f / 30, 'vertices/1000'))
                dmax = depth.max()
                key = bn.replace(g + '_', '') + ' in ' + (m.geom(k).name.replace(g + '_', ''))
                if dmax > worst.get(key, (0,))[0]: worst[key] = (dmax, f / 30, (m.geom_contype[i], m.geom_conaffinity[i]))
    for k, (dmax, t, cc) in sorted(worst.items(), key=lambda x: -x[1][0])[:8]:
        print(g, f'{k:55s} depth {dmax*100:5.1f} cm at t={t:4.1f}s  ct/ca {cc}')
