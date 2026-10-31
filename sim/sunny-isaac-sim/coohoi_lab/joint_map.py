"""CooHOI humanoid: MJCF order (Isaac Gym, depth first) vs Isaac Lab articulation order, matched by name, plus
CooHOI's PD action mapping (offset + scale * action) and the MJCF PD gains.
Usage: python joint_map.py                         # print the MJCF table (order, limits, gains, action map)
       python joint_map.py --lab names.json        # check an Isaac Lab name list: {"joints": [...], "bodies": [...]}
       python joint_map.py --lab "abdomen_x,abdomen_y,..."   # or a comma-separated joint-name list
The MJCF is read from the CooHOI checkout ($COOHOI, else ~/CooHOI, else ../../../../CooHOI next to this repo).
"""
import argparse
import json
import math
import os
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))


def coohoi_root():
    for p in (os.environ.get("COOHOI"), "~/CooHOI", os.path.join(HERE, "../../../../CooHOI")):
        if p and os.path.isdir(os.path.expanduser(p)):
            return os.path.abspath(os.path.expanduser(p))
    raise FileNotFoundError("CooHOI repo not found: set COOHOI to its root")


def mjcf_path():
    return os.path.join(coohoi_root(), "coohoi/data/assets/mjcf/amp_humanoid.xml")


def parse_mjcf(path=None):
    """Bodies and hinge joints in Isaac Gym order: bodies depth first, joints in XML order within each body.
    Angles in radians (MJCF default compiler angle is degrees)."""
    root = ET.parse(path or mjcf_path()).getroot()
    defaults = {}
    for d in root.iter("default"):
        j = d.find("joint")
        if d.get("class") == "body" and j is not None:
            defaults = dict(j.attrib)
    gear = {m.get("joint"): float(m.get("gear")) for m in root.find("actuator")}
    bodies, joints = [], []

    def visit(body):
        bodies.append(body.get("name"))
        for j in body.findall("joint"):  # freejoint is its own tag, so the root's 6 DoF are skipped
            a = {**defaults, **j.attrib}
            lo, hi = (math.radians(float(x)) for x in a["range"].split())
            joints.append(dict(name=a["name"], body=body.get("name"), lower=lo, upper=hi,
                               stiffness=float(a["stiffness"]), damping=float(a["damping"]),
                               armature=float(a["armature"]), gear=gear[a["name"]]))
        for child in body.findall("body"):
            visit(child)

    for b in root.find("worldbody").findall("body"):
        visit(b)
    return bodies, joints


BODIES, JOINTS = parse_mjcf()
JOINT_NAMES = [j["name"] for j in JOINTS]


def dof_offsets(joints=JOINTS):
    """CooHOI's _dof_offsets: start index of each body's joint group, plus the total."""
    out = [0]
    for i in range(1, len(joints) + 1):
        if i == len(joints) or joints[i]["body"] != joints[i - 1]["body"]:
            out.append(i)
    return out


def pd_action_offset_scale(joints=JOINTS):
    """Port of ShareHumanoid._build_pd_action_offset_scale (share_humanoid.py:381), gym joint order.
    3-DoF joints: symmetric, 1.2 x the largest |limit| of the group, capped at pi. 1-DoF joints: centre +- 0.7 x range."""
    lo = [j["lower"] for j in joints]
    hi = [j["upper"] for j in joints]
    offs = dof_offsets(joints)
    for a, b in zip(offs, offs[1:]):
        if b - a == 3:
            s = min(1.2 * max(max(abs(x) for x in lo[a:b]), max(abs(x) for x in hi[a:b])), math.pi)
            lo[a:b], hi[a:b] = [-s] * 3, [s] * 3
        elif b - a == 1:
            mid, s = 0.5 * (hi[a] + lo[a]), 0.7 * (hi[a] - lo[a])
            lo[a], hi[a] = mid - s, mid + s
    offset = [0.5 * (h + l) for h, l in zip(hi, lo)]
    scale = [0.5 * (h - l) for h, l in zip(hi, lo)]
    return offset, scale


def gains(key, joints=JOINTS):
    """{joint name: value} for key in stiffness, damping, armature, gear (the gear is the motor's effort scale)."""
    return {j["name"]: j[key] for j in joints}


class JointMap:
    """Index lists between the two orders. With x_gym in MJCF order and x_lab in Isaac Lab order:
    x_lab = x_gym[..., joints_lab_from_gym],  x_gym = x_lab[..., joints_gym_from_lab],
    body_gym = body_lab[:, bodies_gym_from_lab] (Isaac Lab may have extra bodies; they are left out)."""

    def __init__(self, lab_joints, lab_bodies=None, gym_joints=None, gym_bodies=None):
        gj, gb = gym_joints or JOINT_NAMES, gym_bodies or BODIES
        missing, extra = [n for n in gj if n not in lab_joints], [n for n in lab_joints if n not in gj]
        if missing or extra or len(lab_joints) != len(gj):
            raise ValueError(f"joint names differ from the MJCF: missing in Isaac Lab {missing}, extra {extra}")
        self.joints_lab_from_gym = [gj.index(n) for n in lab_joints]
        self.joints_gym_from_lab = [list(lab_joints).index(n) for n in gj]
        self.bodies_gym_from_lab = None
        if lab_bodies is not None:
            missing = [n for n in gb if n not in lab_bodies]
            if missing:
                raise ValueError(f"MJCF bodies missing in Isaac Lab: {missing}")
            self.bodies_gym_from_lab = [list(lab_bodies).index(n) for n in gb]
        self.lab_joints, self.lab_bodies, self.gym_joints, self.gym_bodies = list(lab_joints), lab_bodies, gj, gb

    def report(self):
        same = self.joints_lab_from_gym == list(range(len(self.gym_joints)))
        lines = [f"joints: {len(self.gym_joints)} match by name; order {'identical' if same else 'differs'}"]
        if not same:
            lines += [f"  lab[{i:2d}] {n:18s} <- gym[{g:2d}]" for i, (n, g) in enumerate(zip(self.lab_joints, self.joints_lab_from_gym))]
        if self.lab_bodies is not None:
            extra = [n for n in self.lab_bodies if n not in self.gym_bodies]
            same = self.bodies_gym_from_lab == list(range(len(self.gym_bodies)))
            lines.append(f"bodies: {len(self.gym_bodies)} MJCF bodies found; order {'identical' if same else 'differs'}"
                         f"; extra Isaac Lab bodies (ignored): {extra}")
            if not same:
                lines.append("  gym -> lab: " + ", ".join(f"{n}:{i}" for n, i in zip(self.gym_bodies, self.bodies_gym_from_lab)))
        return "\n".join(lines)


def table():
    off, sc = pd_action_offset_scale()
    rows = ["idx name               body             range deg       kp     kd   armature  gear  pd offset  pd scale (rad)"]
    for i, (j, o, s) in enumerate(zip(JOINTS, off, sc)):
        rows.append(f"{i:3d} {j['name']:18s} {j['body']:16s} {math.degrees(j['lower']):7.1f} {math.degrees(j['upper']):6.1f}"
                    f" {j['stiffness']:7.0f} {j['damping']:6.0f} {j['armature']:8.3f} {j['gear']:5.0f} {o:9.4f} {s:9.4f}")
    rows.append(f"bodies (depth first): {BODIES}")
    rows.append(f"dof offsets: {dof_offsets()}")
    return "\n".join(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lab", help="JSON file with joints (and bodies), or a comma-separated joint-name list")
    a = ap.parse_args()
    if not a.lab:
        print(f"MJCF: {mjcf_path()}\n{table()}")
    else:
        names = json.load(open(a.lab)) if os.path.isfile(a.lab) else {"joints": a.lab.split(",")}
        print(JointMap(names["joints"], names.get("bodies")).report())
