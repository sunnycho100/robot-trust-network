"""Top-down occupancy grid and A* paths from scene geometry, so fleet.py robots route around obstacles.

Pure numpy, tested on the Mac (test_navmap.py). Input is world points (z up) from any mesh: a stock Isaac scene
(stage_points below, inside Isaac Sim) or our reconstruction's collision mesh. A cell is blocked when any point sits
between `step` and `height` above the floor, then the blocked cells grow by the robot radius, so paths are for the
robot's centre. Cells outside the points' extent count as blocked.
"""
import heapq
import math

import numpy as np


class NavMap:
    def __init__(self, points, radius=0.4, res=0.1, step=0.1, height=1.8, floor_z=None):
        p = np.asarray(points, float)
        if floor_z is None:
            # ponytail: the densest 5 cm z slab in the lowest 30% of points; pass floor_z for multi-level scenes
            low = p[p[:, 2] <= np.percentile(p[:, 2], 30), 2]
            edges = np.arange(low.min(), low.max() + 0.1, 0.05)
            e = edges[np.histogram(low, bins=edges)[0].argmax()]
            floor_z = float(np.median(low[(low >= e) & (low < e + 0.05)]))
        self.floor_z, self.res = floor_z, res
        self.lo = np.percentile(p[:, :2], 1, axis=0)  # ignore a far-out skybox or stray splats
        hi = np.percentile(p[:, :2], 99, axis=0)
        self.shape = tuple(np.ceil((hi - self.lo) / res).astype(int) + 1)
        h = p[:, 2] - floor_z
        ob = p[(h > step) & (h < height), :2]
        ij = np.floor((ob - self.lo) / res).astype(int)
        ij = ij[(ij >= 0).all(1) & (ij < self.shape).all(1)]
        hit = np.zeros(self.shape, bool)
        hit[ij[:, 0], ij[:, 1]] = True
        r = int(math.ceil(radius / res))
        pad = np.pad(hit, r)
        self.blocked = np.zeros(self.shape, bool)
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                if di * di + dj * dj <= (radius / res) ** 2:
                    self.blocked |= pad[r + di:r + di + self.shape[0], r + dj:r + dj + self.shape[1]]

    def cell(self, x, y):
        return tuple(np.floor((np.asarray([x, y]) - self.lo) / self.res).astype(int))

    def free(self, x, y):
        i, j = self.cell(x, y)
        return 0 <= i < self.shape[0] and 0 <= j < self.shape[1] and not self.blocked[i, j]

    def clear(self, a, b):
        """Straight line from a to b stays in free cells (sampled every half cell)."""
        n = int(math.dist(a, b) / (self.res / 2)) + 2
        return all(self.free(x, y) for x, y in np.linspace(a, b, n))

    def path(self, start, goal):
        """Corner waypoints from start to goal (both kept exactly), or None when the goal is blocked or sealed off.
        ponytail: 8-connected A* in pure Python, a second or so on a 100 x 100 m map at 10 cm; scipy or a coarser
        res beyond that."""
        start, goal = tuple(start), tuple(goal)
        if not self.free(*goal):
            return None
        if self.clear(start, goal):
            return [start, goal]
        s, g = self.cell(*start), self.cell(*goal)
        h = lambda c: math.hypot(c[0] - g[0], c[1] - g[1])
        came, cost, todo = {s: None}, {s: 0.0}, [(h(s), s)]
        while todo:
            _, c = heapq.heappop(todo)
            if c == g:
                break
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                n = (c[0] + di, c[1] + dj)
                if not (0 <= n[0] < self.shape[0] and 0 <= n[1] < self.shape[1]) or self.blocked[n]:
                    continue
                nc = cost[c] + math.hypot(di, dj)
                if nc < cost.get(n, math.inf):
                    cost[n], came[n] = nc, c
                    heapq.heappush(todo, (nc + h(n), n))
        if g not in came:
            return None
        cells = []
        while g is not None:
            cells.append(g)
            g = came[g]
        pts = [start] + [tuple(float(v) for v in self.lo + (np.asarray(c) + 0.5) * self.res)
                         for c in reversed(cells[1:-1])] + [goal]
        out = [pts[0]]  # keep only corners: jump to the farthest point still in line of sight
        k = 0
        while k < len(pts) - 1:
            k = next(m for m in range(len(pts) - 1, k, -1) if m == k + 1 or self.clear(pts[k], pts[m]))
            out.append(pts[k])
        return out


def stage_points(root):
    """World points of every visible Mesh under a USD prim (inside Isaac Sim), same traversal as
    capture_walkthrough.py, plus invisible Meshes with collision (a splat scene's collider from add_mesh_to_usdz).
    ponytail: vertices only; a big flat floor with few vertices still shows up because only obstacles need points,
    and cells with no points at all are free inside the scene's extent."""
    from pxr import Usd, UsdGeom, UsdPhysics

    out = []
    for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not p.IsA(UsdGeom.Mesh):
            continue
        if UsdGeom.Imageable(p).ComputeVisibility() == UsdGeom.Tokens.invisible and not p.HasAPI(UsdPhysics.CollisionAPI):
            continue
        pts = UsdGeom.Mesh(p).GetPointsAttr().Get()
        if pts:
            xf = np.array(UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
            out.append(np.asarray(pts, float) @ xf[:3, :3].T + xf[:3, 3])
    return np.concatenate(out)
