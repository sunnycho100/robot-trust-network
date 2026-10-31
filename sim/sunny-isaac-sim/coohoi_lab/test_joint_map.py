"""python test_joint_map.py: checks joint_map.py against CooHOI's hard-coded tables, hand-computed action ranges, and a
breadth-first (PhysX-style) reordering of the same humanoid."""
import math
import xml.etree.ElementTree as ET
from collections import deque

import joint_map as jm

assert len(jm.BODIES) == 15 and len(jm.JOINTS) == 28
# CooHOI share_humanoid.py:208-210 for mjcf/amp_humanoid.xml
assert jm.dof_offsets() == [0, 3, 6, 9, 10, 13, 14, 17, 18, 21, 24, 25, 28]
assert [jm.BODIES.index(jm.JOINTS[o]["body"]) for o in jm.dof_offsets()[:-1]] == [1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14]
# the order Isaac Lab's own AMP motion files store (humanoid_walk.npz dof_names, v3.0.0) is this same depth-first order
assert jm.JOINT_NAMES[:4] == ["abdomen_x", "abdomen_y", "abdomen_z", "neck_x"] and jm.JOINT_NAMES[14:17] == ["right_hip_x", "right_hip_y", "right_hip_z"]

off, sc = jm.pd_action_offset_scale()
r = math.radians
cases = {"abdomen_x": (0, 1.2 * r(90)), "neck_z": (0, 1.2 * r(60)), "right_shoulder_y": (0, math.pi),
         "right_elbow": (r(-80), 0.7 * r(160)), "right_knee": (r(80), 0.7 * r(160)), "left_hip_z": (0, 1.2 * r(140)),
         "left_ankle_x": (0, 1.2 * r(55))}
for name, (o, s) in cases.items():
    i = jm.JOINT_NAMES.index(name)
    assert abs(off[i] - o) < 1e-9 and abs(sc[i] - s) < 1e-9, (name, off[i], sc[i])
kp, kd, gear = jm.gains("stiffness"), jm.gains("damping"), jm.gains("gear")
assert (kp["abdomen_y"], kd["abdomen_y"], kp["neck_x"], kp["right_knee"], kd["left_ankle_z"]) == (1000, 100, 100, 500, 40)
assert (gear["right_hip_y"], gear["left_knee"], gear["neck_z"]) == (200, 150, 50)

# breadth-first order of the same tree, the kind of order a USD articulation can come back in
root = ET.parse(jm.mjcf_path()).getroot().find("worldbody").find("body")
bfs, q = [], deque([root])
while q:
    b = q.popleft()
    bfs.append(b)
    q.extend(b.findall("body"))
lab_bodies = [b.get("name") for b in bfs] + ["some_dummy_link"]
lab_joints = [j.get("name") for b in bfs for j in b.findall("joint")]
assert lab_joints != jm.JOINT_NAMES
m = jm.JointMap(lab_joints, lab_bodies)
gym_vals = list(range(100, 128))
lab_vals = [gym_vals[i] for i in m.joints_lab_from_gym]
assert [lab_vals[i] for i in m.joints_gym_from_lab] == gym_vals, "round trip"
assert all(lab_joints[k] == jm.JOINT_NAMES[g] for k, g in enumerate(m.joints_lab_from_gym))
assert [lab_bodies[i] for i in m.bodies_gym_from_lab] == jm.BODIES
assert "differs" in m.report() and "some_dummy_link" in m.report()
assert "identical" in jm.JointMap(jm.JOINT_NAMES, jm.BODIES).report()
for bad in (lab_joints[:-1], lab_joints[:-1] + ["abdomen_q"]):
    try:
        jm.JointMap(bad)
        raise AssertionError("a wrong joint list must raise")
    except ValueError:
        pass
print("joint_map ok\n" + m.report().splitlines()[0])
