"""python traj_check.py TRAJ.csv: per robot, base z range and the first time z drops below half its standing height (median over 0.5 to 2 s; a fall),
plus the closest centre-to-centre gap between any pair, from multi_robot_scene.py --log output."""
import sys, numpy as np
d = np.genfromtxt(sys.argv[1], delimiter=",", names=True)
t_all = np.unique(d["t"])
for r in np.unique(d["robot"]).astype(int):
    x = d[d["robot"] == r]
    med = np.median(x["z"][(x["t"] >= 0.5) & (x["t"] <= 2)]); fall = x["t"][x["z"] < 0.5 * med]  # 0.6 flagged an upright Go2 (it walks with its base 0.14 to 0.23 m up)
    print(f"robot {r}: {len(x)} rows over {x['t'].max():.1f} s, z p1 {np.percentile(x['z'], 1):.2f} median {med:.2f}, "
          f"fall at {fall[0]:.1f} s" if len(fall) else f"robot {r}: {len(x)} rows over {x['t'].max():.1f} s, z p1 {np.percentile(x['z'], 1):.2f} median {med:.2f}, no fall",
          f"end ({x['x'][-1]:.2f}, {x['y'][-1]:.2f})")
gap = np.inf
for t in t_all:
    p = np.c_[d["x"], d["y"]][d["t"] == t]
    for i in range(len(p)):
        for j in range(i):
            gap = min(gap, float(np.hypot(*(p[i] - p[j]))))
print(f"closest pair {gap:.2f} m over {len(t_all)} ticks")
