"""Pure-numpy kinematics for carda_line.py: forward kinematics of a joint tree (as read from a USD robot asset),
damped least-squares IK for a tool tip and its axis, and a two-bone arm solver for the worker. Checked by test_kin.py."""
import math

import numpy as np

TOOL = 0.25   # m, flange to screwdriver bit tip (handoff: 0.30 m)


def tf(R=None, p=(0, 0, 0)):
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    T[:3, 3] = p
    return T


def rot(axis, a):
    c, s = math.cos(a), math.sin(a)
    i, j = [(1, 2), (2, 0), (0, 1)][axis]
    R = np.eye(3)
    R[i, i] = R[j, j] = c
    R[i, j], R[j, i] = -s, s
    return R


def rot_between(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(np.dot(a, b))
    if c < -0.999999:
        p = np.cross(a, [1, 0, 0]) if abs(a[0]) < 0.9 else np.cross(a, [0, 1, 0])
        p /= np.linalg.norm(p)
        return 2 * np.outer(p, p) - np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1 + c)


def minjerk(s):
    s = min(max(s, 0.0), 1.0)
    return s * s * s * (10 - 15 * s + 6 * s * s)


def lerp(a, b, s):
    return np.asarray(a, float) + (np.asarray(b, float) - np.asarray(a, float)) * s



def two_bone(S, a, b, T, pole):
    d = T - S
    L = np.linalg.norm(d)
    n = d / max(L, 1e-6)
    L = min(max(L, abs(a - b) + 1e-3), a + b - 1e-3)
    x = (a * a - b * b + L * L) / (2 * L)
    h = math.sqrt(max(a * a - x * x, 0.0))
    m = pole - np.dot(pole, n) * n
    m /= max(np.linalg.norm(m), 1e-6)
    E = S + n * x + m * h
    Wr = E + (T - E) / max(np.linalg.norm(T - E), 1e-6) * b
    return E, Wr


class Kin:
    """Joint tree from edges {b0, b1, kind rev|pri|fix, axis 0-2, J0 (joint frame in b0), J1i (inverse joint frame in b1), lo, hi}."""

    def __init__(self, edges, base, T_base=np.eye(4)):
        order, known = [], {base}
        while len(order) < len(edges):
            nxt = [e for e in edges if e not in order and e["b0"] in known]
            assert nxt, "joint tree is not connected"
            order += nxt
            known |= {e["b1"] for e in nxt}
        self.base, self.T_base, self.edges = base, T_base, order
        self.dofs = [e for e in order if e["kind"] != "fix"]
        self.lo = np.array([e["lo"] for e in self.dofs])
        self.hi = np.array([e["hi"] for e in self.dofs])
        self.links = [e["b1"] for e in order]
        # tool: the last moving link; its spin axis points out of the flange, away from the joint before it
        last, prev = self.dofs[-1], self.dofs[-2]
        self.tool = last["b1"]
        T = self.fk(np.zeros(len(self.dofs)))
        a = (T[last["b0"]] @ last["J0"])[:3, last["axis"]]
        p_prev = (T[prev["b0"]] @ prev["J0"])[:3, 3]
        a = a * (np.sign(np.dot(a, T[self.tool][:3, 3] - p_prev)) or 1.0)
        self.axis_l = T[self.tool][:3, :3].T @ a
        self.tip_l = self.axis_l * TOOL

    def fk(self, q):
        T = {self.base: self.T_base}
        i = 0
        for e in self.edges:
            M = e["J0"]
            if e["kind"] == "rev":
                M = M @ tf(rot(e["axis"], q[i]))
                i += 1
            elif e["kind"] == "pri":
                d = np.zeros(3)
                d[e["axis"]] = q[i]
                M = M @ tf(None, d)
                i += 1
            T[e["b1"]] = T[e["b0"]] @ M @ e["J1i"]
        return T

    def tip(self, q, T=None):
        T = self.fk(q)[self.tool] if T is None else T[self.tool]
        return T[:3, :3] @ self.tip_l + T[:3, 3], T[:3, :3] @ self.axis_l

    def _ik(self, p, a, q, iters):
        def resid(q):
            pc, ac = self.tip(q)
            return np.r_[p - pc, -0.5 * np.cross(ac, a)], ac
        r, ac = resid(q)
        lam = 1e-2
        for _ in range(iters):
            if np.linalg.norm(r[:3]) < 5e-4 and np.dot(ac, a) > 0.9998:
                return q, True
            J = np.zeros((6, len(q)))
            pc = p - r[:3]
            for i in range(len(q)):
                dq = q.copy()
                dq[i] += 1e-6
                pp, aa = self.tip(dq)
                J[:3, i] = (pp - pc) / 1e-6
                J[3:, i] = 0.5 * (np.cross(aa, a) - np.cross(ac, a)) / 1e-6
            step = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(6), r)
            qn = np.clip(q + np.clip(step, -0.3, 0.3), self.lo, self.hi)
            rn, acn = resid(qn)
            if np.linalg.norm(rn) < np.linalg.norm(r):        # Levenberg-Marquardt: accept and trust more
                q, r, ac, lam = qn, rn, acn, max(lam * 0.3, 1e-6)
            else:
                lam = min(lam * 4, 10.0)
        return q, bool(np.linalg.norm(r[:3]) < 2e-3 and np.dot(ac, a) > 0.999)

    def ik(self, p, a, q0, iters=300, seeds=()):
        """Tip at p with the tool axis along a. Tries q0, then the seeds, then q0 with each joint nudged."""
        p, a = np.asarray(p, float), np.asarray(a, float) / np.linalg.norm(a)
        q0 = np.asarray(q0, float)
        rng = np.random.default_rng(0)
        starts = [q0, *map(np.asarray, seeds)] + [q0 + rng.normal(0, 0.6, len(q0)) for _ in range(10)]
        for s in starts:
            q, ok = self._ik(p, a, np.clip(s, self.lo, self.hi), iters)
            if ok:
                break
        # same pose, shortest joint move: wrap each joint to within half a turn of the start where the limits allow
        w = q0 + (q - q0 + np.pi) % (2 * np.pi) - np.pi
        q = np.where((w >= self.lo) & (w <= self.hi), w, q)
        return q, ok

    def points(self, T):
        """Points along the arm for the separation check (root frame)."""
        js = [(T[e["b0"]] @ e["J0"])[:3, 3] for e in self.dofs] + [self.tip(None, T)[0]]
        return np.array([lerp(a, b, s) for a, b in zip(js[:-1], js[1:]) for s in (0, 0.5)] + [js[-1]])

