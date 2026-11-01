"""Kinematic rigs for door_team.py, read from the stock USD assets: a joint tree per robot (FK from its UsdPhysics
joints, as carda_line.py does), tip IK for the UR20 on its carriage, and hand IK for the G1 arms. Pure pxr + numpy, so
test_rig.py runs it on a laptop with usd-core against the same asset files Isaac Sim loads."""
import math
import os
import sys

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "carda"))
from kin import Kin, lerp, tf  # noqa: E402


def gf2np(m):
    return np.array(m, dtype=float).T          # Gf is row-vector; numpy here is column-vector


def np2gf(T):
    return Gf.Matrix4d(*[float(v) for v in T.T.reshape(-1)])


def quat_R(q):
    w, (x, y, z) = q.GetReal(), q.GetImaginary()
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def quat_wxyz_R(q):
    return quat_R(Gf.Quatd(float(q[0]), Gf.Vec3d(*map(float, q[1:]))))


class Chain(Kin):
    """Kin built from a robot asset's UsdPhysics joints; poses are written to the link prims."""

    def __init__(self, stage, root_path):
        self.stage = stage
        self.root = stage.GetPrimAtPath(root_path)
        xc = UsdGeom.XformCache()
        Wi = np.linalg.inv(gf2np(xc.GetLocalToWorldTransform(self.root)))
        self.rel = lambda p: Wi @ gf2np(xc.GetLocalToWorldTransform(p))  # noqa: E731
        edges = []
        for p in Usd.PrimRange(self.root):
            if not p.IsA(UsdPhysics.Joint):
                continue
            j = UsdPhysics.Joint(p)
            b0, b1 = j.GetBody0Rel().GetTargets(), j.GetBody1Rel().GetTargets()
            if not b0 or not b1:
                continue
            kind = "rev" if p.IsA(UsdPhysics.RevoluteJoint) else "pri" if p.IsA(UsdPhysics.PrismaticJoint) else "fix"
            ax = p.GetAttribute("physics:axis").Get() if kind != "fix" else "X"
            lo, hi = p.GetAttribute("physics:lowerLimit").Get(), p.GetAttribute("physics:upperLimit").Get()
            k = math.pi / 180 if kind == "rev" else 1.0
            edges.append(dict(name=p.GetName(), b0=str(b0[0]), b1=str(b1[0]), kind=kind, axis="XYZ".index(ax),
                              J0=tf(quat_R(j.GetLocalRot0Attr().Get()), j.GetLocalPos0Attr().Get()),
                              J1i=np.linalg.inv(tf(quat_R(j.GetLocalRot1Attr().Get()), j.GetLocalPos1Attr().Get())),
                              lo=(lo if lo is not None else -360) * k, hi=(hi if hi is not None else 360) * k))
        kids = {e["b1"] for e in edges}
        base = next(e["b0"] for e in edges if e["b0"] not in kids)
        known, keep = {base}, []
        while True:                               # drop sub-trees that do not hang off the base (hand articulations)
            nxt = [e for e in edges if e not in keep and e["b0"] in known]
            if not nxt:
                break
            keep += nxt
            known |= {e["b1"] for e in nxt}
        super().__init__(keep, base, self.rel(stage.GetPrimAtPath(base)))
        self.parent, self.static, self.ops = {}, {}, {}
        for link in self.links:
            par = stage.GetPrimAtPath(link).GetParent()
            self.parent[link] = str(par.GetPath())
            if self.parent[link] not in self.links and self.parent[link] != self.base:
                self.static[self.parent[link]] = self.rel(par)
        T = self.fk(np.zeros(len(self.dofs)))
        self.fk0_err = max(np.abs(T[link] - self.rel(stage.GetPrimAtPath(link))).max() for link in self.links)
        self.idx = {e["name"]: i for i, e in enumerate(self.dofs)}
        print(f"[chain] {root_path}: {len(self.dofs)} dofs, FK(0) vs authored pose max diff {self.fk0_err:.2e}")

    def write(self, q):
        T = self.fk(q)
        for link in self.links:
            if link not in self.ops:
                self.ops[link] = UsdGeom.Xformable(self.stage.GetPrimAtPath(link)).MakeMatrixXform()
            par = self.parent[link]
            Tp = T[par] if par in T else self.static.get(par, np.eye(4))
            self.ops[link].Set(np2gf(np.linalg.inv(Tp) @ T[link]))
        return T


class Arm(Chain):
    """UR on a carriage: tip and tool-axis IK, frame to frame from the last solution."""

    def __init__(self, stage, root_path, T_mount, finger_l):
        super().__init__(stage, root_path)
        self.T_mount = T_mount                    # robot root in the carriage frame
        self.finger_l = finger_l                  # finger opening direction in the flange frame
        self.q = None

    def solve(self, T_carriage, tip, axis, finger_w=np.array([1.0, 0, 0])):
        Tr = T_carriage @ self.T_mount
        Ri = Tr[:3, :3].T
        p, a = Ri @ (tip - Tr[:3, 3]), Ri @ axis
        if self.q is None:                        # elbow-up start, upper arm above the shoulder
            best = None
            for pan in np.radians([-180, -90, 0, 90, 180]):
                q0 = np.array([pan, -1.2, 1.6, -1.5, -1.57, 0.0])
                q, ok = self.ik(p, a, q0)
                if ok:
                    elbow = self.fk(q)[self.dofs[2]["b1"]][2, 3]
                    if best is None or elbow > best[0]:
                        best = (elbow, q)
            assert best, "no start IK"
            self.q = best[1]
        q, ok = self.ik(p, a, self.q, iters=200)
        # roll the last joint so the fingers close across the door's thickness (world finger_w)
        T = self.fk(q)[self.tool]
        aw, fw = Tr[:3, :3] @ T[:3, :3] @ self.axis_l, Tr[:3, :3] @ T[:3, :3] @ self.finger_l
        d = finger_w - np.dot(finger_w, aw) * aw
        ang = math.atan2(np.dot(np.cross(fw, d), aw), np.dot(fw, d))
        ang = (ang + math.pi / 2) % math.pi - math.pi / 2      # either finger may lead
        for sgn in (1, -1):
            qq = q.copy()
            qq[-1] += sgn * ang
            T = self.fk(qq)[self.tool]
            if abs(np.dot(Tr[:3, :3] @ T[:3, :3] @ self.finger_l, d / max(np.linalg.norm(d), 1e-9))) > 0.99:
                q = qq
                break
        self.q = q
        return q, ok


class G1(Chain):
    """G1 replay: root, leg and waist yaw angles from MuJoCo (mj_names), arms by position IK to the hand targets."""
    ARM = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow"]

    def __init__(self, stage, root_path, mj_names):
        super().__init__(stage, root_path)
        self.mj = [self.idx[n] for n in mj_names]
        self.arm = {"l": [self.idx[f"left_{j}_joint"] for j in self.ARM], "r": [self.idx[f"right_{j}_joint"] for j in self.ARM]}
        self.wrist = {s: f"{root_path}/{n}_wrist_yaw_link" for s, n in (("l", "left"), ("r", "right"))}
        self.palm = {}
        for s, n in (("l", "left"), ("r", "right")):
            pp = stage.GetPrimAtPath(f"{self.wrist[s]}/{n}_hand_palm_link")
            off = np.linalg.inv(self.rel(stage.GetPrimAtPath(self.wrist[s]))) @ self.rel(pp) if pp.IsValid() else tf()
            self.palm[s] = off @ tf(None, (0.06, 0, 0))          # grip point a little past the palm frame
        self.q = np.zeros(len(self.dofs))
        self.relax = self.q.copy()
        for s, y in (("l", 0.21), ("r", -0.21)):      # hands hanging beside the hips
            self.relax = self.arm_ik(self.relax, s, np.array([0.06, y, -0.08]), self.relax, iters=200)
        self.last = {s: self.relax.copy() for s in "lr"}

    def hand(self, T, s):
        return (T[self.wrist[s]] @ self.palm[s])[:3, 3]

    def arm_ik(self, q, s, target, q_nom, iters=12):
        """Position IK on shoulder pitch, roll, yaw and elbow; the spare freedom drifts to the nominal pose."""
        ids = self.arm[s]
        q = q.copy()
        for _ in range(iters):
            p = self.hand(self.fk(q), s)
            e = target - p
            if np.linalg.norm(e) < 2e-3:
                break
            J = np.zeros((3, len(ids)))
            for c, i in enumerate(ids):
                dq = q.copy()
                dq[i] += 1e-5
                J[:, c] = (self.hand(self.fk(dq), s) - p) / 1e-5
            Jp = J.T @ np.linalg.inv(J @ J.T + 1e-3 * np.eye(3))
            step = Jp @ e + (np.eye(len(ids)) - Jp @ J) @ (0.2 * (q_nom[ids] - q[ids]))
            q[ids] = np.clip(q[ids] + np.clip(step, -0.25, 0.25), self.lo[ids], self.hi[ids])
        return q

    def solve(self, root7, q_mj, hands, w):
        """World root pose of the robot prim and the full joint vector for one frame."""
        T_pel = tf(quat_wxyz_R(root7[3:7]), root7[:3])
        T_root = T_pel @ np.linalg.inv(self.T_base)
        q = self.relax.copy()
        q[self.mj] = q_mj
        Ti = np.linalg.inv(T_root)
        err = 0.0
        for c, s in enumerate("lr"):
            ids = self.arm[s]
            if w[c] <= 0:
                self.last[s][ids] = self.relax[ids]
                continue
            tgt = (Ti @ np.r_[hands[c], 1])[:3]
            seed = q.copy()
            seed[ids] = self.last[s][ids]
            qs = self.arm_ik(seed, s, tgt, self.relax)
            self.last[s][ids] = qs[ids]
            err = max(err, float(np.linalg.norm(self.hand(self.fk(qs), s) - tgt)) * (w[c] >= 1))
            q[ids] = lerp(self.relax[ids], qs[ids], w[c])
        return T_root, q, err
