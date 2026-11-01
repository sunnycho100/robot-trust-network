"""Runnable check for rig.py on a laptop: usd-core stage with the stock UR20 and G1, the arm and hand IK run over a
plan from orchestrate.py, and every solved tip must land on its target.
  python test_rig.py PLAN_DIR ASSET_DIR   (ASSET_DIR holds UniversalRobots/ur20/... and Unitree/G1/... as on the asset root)
"""
import json
import math
import os
import sys

import numpy as np
from pxr import Usd, UsdGeom

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rig  # noqa: E402
from kin import rot, rot_between, tf  # noqa: E402

plan_dir, assets = sys.argv[1:3]
P = np.load(os.path.join(plan_dir, "plan.npz"))
J = json.load(open(os.path.join(plan_dir, "plan.json")))
L = J["layout"]
st = Usd.Stage.CreateInMemory()
UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z)
for path, url in [("/World/ArmL", "UniversalRobots/ur20/ur20.usd"), ("/World/G1", "Unitree/G1/g1.usd")]:
    st.DefinePrim(path, "Xform").GetReferences().AddReference(os.path.join(assets, url))
arm = rig.Arm(st, "/World/ArmL", tf(None, (0, 0, L["PED"])), None)
G = rot_between(np.array([0, 0, 1.0]), arm.axis_l)
arm.finger_l = G[:3, 0]
assert arm.fk0_err < 1e-4
worst = 0.0
for k in range(0, len(P["t"]), 3):    # ARM L, the longer reach from the OUT rail
    Tc = tf(None, (P["ARM_L.cx"][k], L["RAILS"]["ARM_L"], 0))
    q, ok = arm.solve(Tc, P["ARM_L.tip"][k], P["ARM_L.axis"][k])
    tip, ax = arm.tip(q)
    Tr = Tc @ arm.T_mount
    err = np.linalg.norm(Tr[:3, :3] @ tip + Tr[:3, 3] - P["ARM_L.tip"][k])
    worst = max(worst, err)
print(f"UR20 ARM_L tip error max {worst * 1000:.1f} mm over the plan")
assert worst < 0.01
armr = rig.Arm(st, "/World/ArmL", tf(None, (0, 0, L["PED"])), arm.finger_l)   # same asset, IN rail targets
worst = 0.0
for k in range(0, len(P["t"]), 3):
    Tc = tf(None, (P["ARM_R.cx"][k], L["RAILS"]["ARM_R"], 0))
    q, ok = armr.solve(Tc, P["ARM_R.tip"][k], P["ARM_R.axis"][k])
    tip, ax = armr.tip(q)
    Tr = Tc @ armr.T_mount
    worst = max(worst, np.linalg.norm(Tr[:3, :3] @ tip + Tr[:3, 3] - P["ARM_R.tip"][k]))
print(f"UR20 ARM_R tip error max {worst * 1000:.1f} mm over the plan")
assert worst < 0.01
g1 = rig.G1(st, "/World/G1", J["joints"][:13])
assert g1.fk0_err < 1e-3
for s in "lr":
    print("relaxed hand", s, g1.hand(g1.fk(g1.relax), s).round(3))
for n in J["humanoids"]:
    errs = []
    for k in range(0, len(P["t"]), 3):
        T_root, q, err = g1.solve(P[n + ".root"][k], P[n + ".q"][k][:13], P[n + ".hand"][k], P[n + ".w"][k])
        errs.append(err)
    errs = np.array(errs)
    print(f"{n} hand IK error: median {np.median(errs[errs > 0]) * 1000:.1f} mm, max {errs.max() * 1000:.1f} mm, "
          f"frames over 3 cm {(errs > 0.03).sum()} of {len(errs)}")
    bad = np.where(errs > 0.03)[0][:5] * 3
    print("   first bad frames t =", [round(P["t"][b], 2) for b in bad])
# feet on the floor: lowest ankle roll link in world while standing at t = 0
T_root, q, _ = g1.solve(P[J["humanoids"][0] + ".root"][0], P[J["humanoids"][0] + ".q"][0][:13], np.zeros((2, 3)), [0, 0])
T = g1.fk(q)
print("ankle z at t=0:", [round(float((T_root @ T[l])[2, 3]), 3) for l in g1.links if "ankle_roll" in l])
