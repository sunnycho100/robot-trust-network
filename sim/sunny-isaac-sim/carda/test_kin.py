"""Checks kin.py on the UR10e joint frames printed by probe_assets.py (Isaac 6.2 ur10e.usd). Usage: python3 test_kin.py"""
import math

import numpy as np

from kin import Kin, rot, rot_between, tf, two_bone

H = math.sqrt(0.5)


def quat(w, x, y, z):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


# (name, b0, b1, localPos0, localRot0 wxyz, limit deg); localPos1 = 0, localRot1 = 1, axis Z for all
UR10E = [("pan", "base", "shoulder", (0, 0, 0.1807), (0, 0, 0, 1), 360), ("lift", "shoulder", "upper", (0, 0, 0), (H, H, 0, 0), 360),
         ("elbow", "upper", "fore", (-0.6127, 0, 0), (1, 0, 0, 0), 180), ("w1", "fore", "w1", (-0.57155, 0, 0.17415), (1, 0, 0, 0), 360),
         ("w2", "w1", "w2", (0, -0.11985, 0), (H, H, 0, 0), 360), ("w3", "w2", "w3", (0, 0.11655, 0), (H, -H, 0, 0), 360)]


def ur10e():
    return Kin([dict(name=n, b0=a, b1=b, kind="rev", axis=2, J0=tf(quat(*r), p), J1i=np.eye(4), lo=-math.radians(l), hi=math.radians(l))
                for n, a, b, p, r, l in UR10E], "base")


def main(targets=None):
    k = ur10e()
    # tool axis points away from wrist 2, and the tip sits TOOL beyond the flange
    T = k.fk(np.zeros(6))
    assert np.allclose(np.linalg.norm(k.axis_l), 1.0)
    # rot_between and two_bone
    for a, b in [((1, 0, 0), (0, 1, 0)), ((0, 0, 1), (0, 0, -1)), ((1, 2, 3), (-2, 0.5, 1))]:
        a, b = np.array(a, float), np.array(b, float)
        assert np.allclose(rot_between(a, b) @ (a / np.linalg.norm(a)), b / np.linalg.norm(b), atol=1e-9)
    S, Tg = np.zeros(3), np.array([0.3, 0.2, -0.25])
    E, Wr = two_bone(S, 0.3, 0.28, Tg, np.array([0, 0, -1.0]))
    assert abs(np.linalg.norm(E - S) - 0.3) < 1e-9 and abs(np.linalg.norm(Wr - E) - 0.28) < 1e-9 and np.linalg.norm(Wr - Tg) < 1e-9
    # IK: tool down and tool horizontal targets of the kind carda_line.py uses (root frame)
    q = np.array([0, -1.2, 1.6, -1.5, -1.57, 0.0])
    for name, p, a in targets or [("home", (0.45, 0.1, 0.55), (0, 0, -1)), ("door", (0.79, -0.4, 0.2), (1, 0, 0)),
                                  ("door low", (0.79, -0.3, 0.05), (1, 0, 0)), ("feeder", (-0.1, 0.42, -0.04), (0, 0, -1))]:
        q2, ok = k.ik(np.array(p, float), np.array(a, float), q)
        pc, ac = k.tip(q2)
        print(f"{name:10s} ok {ok}  err {np.linalg.norm(pc - p) * 1000:.2f} mm  q {np.round(np.degrees(q2)).astype(int)}")
        assert ok, name
        q = q2
    print("kin OK")


if __name__ == "__main__":
    main()
