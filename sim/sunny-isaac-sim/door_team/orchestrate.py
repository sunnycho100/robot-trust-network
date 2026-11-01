"""Door team cell: one scripted orchestrator drives two rail arms and two G1 humanoids through a car-door job.

The task, team and custody vocabulary is flexible_factory's (Job, Task, Resource, Dispatcher, EventBus): the
Dispatcher forms the arm team, and custody moves only on measured pickup and delivery acknowledgements, signed per
robot by CustodySigner (network/protocol.py). Every agent is a small program that waits on bus events, never on a
fixed time, so each one can be replaced by a policy that reads the same events and sends the same acknowledgements.
The G1 legs and root come from Unitree's published walking policy, stepped in MuJoCo here; arms, door, glass and
tools are scripted targets that door_team.py solves by IK and renders in Isaac Sim.

  cd sim/mujoco_houses && .venv/bin/python ../sunny-isaac-sim/door_team/orchestrate.py OUT_DIR [--solo]

Writes OUT_DIR/plan.npz (per-frame poses), plan.json (labels and events), events.jsonl, signed_events.json and
registry.json (public keys). --solo: one G1 does every step alone, same clock and layout, no team and no arms.
"""
import argparse
import base64
import json
import math
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1] / "mujoco_houses" / "src"), str(HERE.parent / "carda")]
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey  # noqa: E402
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat  # noqa: E402
from kin import lerp, minjerk, rot, tf  # noqa: E402
from unitree_demo.coop_lift import demo_config  # noqa: E402
from unitree_demo.flexible_factory.custody_signer import CustodySigner  # noqa: E402
from unitree_demo.flexible_factory.dispatch import Dispatcher  # noqa: E402
from unitree_demo.flexible_factory.event_bus import EventBus  # noqa: E402
from unitree_demo.flexible_factory.job import Job, Priority, Task  # noqa: E402
from unitree_demo.flexible_factory.resource_selector import Resource  # noqa: E402
from unitree_demo.g1_policy import LEG_JOINT_NAMES, UPPER_BODY_JOINT_NAMES, G1Controller, G1UpperBodyController, gait_phase  # noqa: E402
from unitree_demo.trust_session import load_trust  # noqa: E402

FPS = 30
DT = 1.0 / FPS

# ------------------------------------------------------------------ layout: the CarDA WS30 station (plant frame, m)
# sim/sunny-isaac-sim/ws30_station/layout.json: x = belt travel, y towards the IN camera, z up, floor = belt = 0
BELT = 0.0239                # m/s, the WS30 floor belt
BELT_X, BELT_Y = (0.0, 19.1), (1.95, 3.83)   # a standing robot on the belt rides it
CARRIER_X0, CARRIER_Y = 13.093, 3.145        # door trolley at t = 0, rides the belt
DOOR_DX, DOOR_DY, DOOR_Z0 = -0.19, -0.05, 0.75   # door centre on the trolley (layout door.local)
X1 = CARRIER_X0 + DOOR_DX    # door centre x at t = 0, on the trolley
X2 = 19.85                   # WS2 stand, just past the belt end (19.1) so it does not ride the belt
DOOR_Y = CARRIER_Y + DOOR_DY
FIX_Z = 0.38                 # WS2 stand top: door bottom edge there (the trolley holds it at 0.75)
LIFT = 0.15
# rail anchors from layout.json automation_anchors_proposed, both extended to x 20.7 to reach WS2; the OUT rail moved
# from y 1.25 to 1.70 (still on the walkway, off the belt edge strip) because 1.85 m is past a UR20's reach
RAILS = {"ARM_L": 1.70, "ARM_R": 4.45}
RAIL_X = (11.5, 20.7)
PED = 0.80                   # pedestal top on the carriage (arm base height)
GRASP_W = 0.30               # grasp height on the door end faces, door frame
DOOR_L = 1.2
R_DOOR = rot(2, math.pi)     # door frame: x along the door (plant -x), y through it, front (inner side, worker side) +y


def door_T(x, z):
    return tf(R_DOOR, (x, DOOR_Y, z))


def cross2(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


def window_opening(gap=0.03):
    """The station door's window opening in our door frame (x, z), from the frame polyline of ws30_station.py
    (layout door.local), shrunk by half the 35 mm frame tube plus an edge gap."""
    dx0, dx1 = -0.79, 0.41
    zl, zt = 1.30, 1.93
    fr = [(dx0 + 0.02, zl), (dx0 + 0.02, zt - 0.12), (dx0 + 0.16, zt), (dx1 - 0.42, zt), (dx1 - 0.02, zl + 0.2), (dx1 - 0.02, zl)]
    P = np.array([(-(x - DOOR_DX), z - DOOR_Z0) for x, z in fr])     # trolley frame -> door frame (x flips)
    if cross2(P[1] - P[0], P[2] - P[1]) < 0:
        P = P[::-1]                                                  # counter-clockwise
    n = len(P)
    lines = []
    for i in range(n):                                               # each edge moved inwards by gap
        a, b = P[i], P[(i + 1) % n]
        d = (b - a) / np.linalg.norm(b - a)
        nin = np.array([-d[1], d[0]])
        lines.append((a + nin * gap, d))
    out = []
    for i in range(n):                                               # corners: consecutive offset edges meet
        (p1, d1), (p2, d2) = lines[i - 1], lines[i]
        t = cross2(p2 - p1, d2) / cross2(d1, d2)
        out.append(p1 + d1 * t)
    return np.array(out)


GLASS_OPEN = window_opening()
GLASS_Z0 = float(GLASS_OPEN[:, 1].min())
GLASS_IN_DOOR = tf(None, (0.0, 0.0, GLASS_Z0))                  # glass frame: on the bottom edge, x along, y normal
GLASS_PTS = [(float(x), 0, float(z - GLASS_Z0)) for x, z in GLASS_OPEN]
_GA = np.array([(p[0], p[2]) for p in GLASS_PTS])
GLASS_C = _GA.mean(0)                                           # lifter goes on the middle of the pane
# handheld suction-cup lifter: handle bar across the glass, two cups on the face toward the G1 (glass -y)
LIFT_IN_GLASS = tf(None, (float(GLASS_C[0]), -0.10, float(GLASS_C[1])))
LIFT_GRIPS = {"l": np.array([-0.10, 0.0, 0.0]), "r": np.array([0.10, 0.0, 0.0])}
# the latch screw on the door's OUTER face, downstream half: A works it from the OUT side while B sets the glass
SCREW_IN_DOOR = np.array([-0.45, 0.06, 0.42])
TIPV = np.array([0.20, 0.0, 0.075])                            # pistol driver: origin in the grip, x along the bit

# G1 spots (pelvis x, y, yaw): A takes the driver from the station's own line-side rack (as the real worker does),
# B takes the glass from a rack past the belt end; both wait clear of the rails and the moving arms
RACK = (15.22, 5.30)                                           # WS30 screwdriver rack (layout line_side)
GR = (23.00, 4.60)                                             # glass rack (ours), well past the rails
S_A = (RACK[0], 4.76, math.pi / 2)
GLASS_RACK_T = tf(None, (GR[0], GR[1], 0.80))
_LR = GLASS_RACK_T @ LIFT_IN_GLASS                              # lifter handle on the racked pane
S_B = (float(_LR[0, 3]), float(_LR[1, 3]) - 0.30, math.pi / 2)
A_VIA = (16.2, 5.15, 0.0)                                      # between the gantry post and the IN camera rod
A_VIA2 = (21.9, 5.05, -math.pi / 2)                             # past the rail ends, then down to the OUT side
# waiting spots outside the arms' swept volume at WS2 (ARM R works x 20.2 to 21.0, y 3.0 to 4.7 there)
W_A, W_B = (21.90, 2.40, math.pi), (22.90, 3.50, math.pi)
# work spots: B on the inner side centred on the lifter, A on the outer side at the downstream latch screw (the
# door between them), both clear of ARM L at the upstream door end; each comes in square from a point out front
P_B = (float((door_T(X2, FIX_Z) @ GLASS_IN_DOOR @ LIFT_IN_GLASS)[0, 3]), DOOR_Y + 0.42, -math.pi / 2)
P_A = (X2 - SCREW_IN_DOOR[0] - 0.22, DOOR_Y - 0.45, math.pi / 2)
VIA_A, VIA_B = (P_A[0], DOOR_Y - 0.95, math.pi / 2), (P_B[0], DOOR_Y + 0.95, -math.pi / 2)
DRIVER_SHELF_T = tf(rot(2, math.pi / 2), (RACK[0] + 0.15, 5.12, 0.93))     # standing on its battery, right hand
SCREW_BIN = np.array([RACK[0] - 0.15, 5.12, 0.94])
LIFT_CARRY = tf(rot(2, -math.pi / 2), (0.27, 0.0, 0.32))    # lifter handle in the G1 heading frame (pelvis xy, yaw only)
DRIVER_CARRY = tf(None, (0.24, -0.12, 0.05))                  # right hand
BUFFERS = {"LINE_CARRIER": (X1, DOOR_Y), "WS2_FIXTURE": (X2, DOOR_Y), "PARTS_SHELF": S_A[:2], "GLASS_RACK": S_B[:2]}
RUNDOWN = 2.51               # s, screw rundown (sim/handoff)

HEADLINES = [("TEAM_FORMED", "COOPERATIVE_DOOR", "TEAM FORMED (ARM L, ARM R)"),
             ("MEASURED_PICKUP", "COOPERATIVE_DOOR", "DOOR PICKED"),
             ("MEASURED_DELIVERY", "COOPERATIVE_DOOR", "DOOR DELIVERED TO WS2"),
             ("MEASURED_DELIVERY", "GLASS_DELIVERY", "GLASS PLACED"),
             ("MEASURED_DELIVERY", "SCREW_INSTALL", "SCREW INSTALLED"),
             ("MEASURED_DELIVERY", "FIXTURE_HOLD", "STATION 2 COMPLETE")]
SOLO_HEADLINES = [("MEASURED_DELIVERY", "GLASS_DELIVERY", "GLASS PLACED"),
                  ("MEASURED_DELIVERY", "SCREW_INSTALL", "SCREW INSTALLED"),
                  ("MEASURED_DELIVERY", "STATION_RELEASE", "STATION COMPLETE")]


# ------------------------------------------------------------------ pose helpers
def log_R(R):
    c = max(-1.0, min(1.0, (np.trace(R) - 1) / 2))
    a = math.acos(c)
    if a < 1e-6:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if a > math.pi - 1e-3:          # half turn: axis from the symmetric part
        v = np.sqrt(np.maximum((np.diag(R) + 1) / 2, 0))
        v = v * np.sign(w + 1e-12)
        return v / np.linalg.norm(v) * a
    return w / (2 * math.sin(a)) * a


def exp_R(w):
    a = np.linalg.norm(w)
    if a < 1e-9:
        return np.eye(3)
    k = w / a
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


def blend(Ta, Tb, s):
    """Pose between Ta and Tb: straight line for the position, shortest rotation for the frame."""
    R = Ta[:3, :3] @ exp_R(s * log_R(Ta[:3, :3].T @ Tb[:3, :3]))
    return tf(R, lerp(Ta[:3, 3], Tb[:3, 3], s))


def door_profile(tau, x0, lift=1.8, slide=11.0, lower=2.4):
    """Door bottom-centre (x, z) and phase 0 lift off the trolley, 1 carry along the rails, 2 lower onto the WS2
    stand, 3 placed, tau seconds after the team grasp at x0."""
    zc = DOOR_Z0 + LIFT
    if tau < lift:
        return x0, DOOR_Z0 + LIFT * minjerk(tau / lift), 0
    if tau < lift + slide:
        return lerp(x0, X2, minjerk((tau - lift) / slide)), zc, 1
    if tau < lift + slide + lower:
        return X2, zc + (FIX_Z - zc) * minjerk((tau - lift - slide) / lower), 2
    return X2, FIX_Z, 3


def ev(x):
    return x() if callable(x) else x


def wrap(a):
    return math.remainder(a, math.tau)


# ------------------------------------------------------------------ the cell: engine.py custody rules
class Cell:
    """Tasks move custody only on a measured acknowledgement; teams come from flexible_factory's Dispatcher."""

    def __init__(self, resources, backend, process=("FIXTURE_HOLD", "fixture_hold")):
        self.process = process
        self.resources = {r.id: r for r in resources}
        self.backend, self.bus, self.dispatcher = backend, EventBus(), Dispatcher()
        self.jobs, self.tasks, self.serial, self.time = {}, {}, 0, 0.0

    def add_job(self, job):
        self.jobs[job.id] = job
        self.bus.emit(self.time, "MATERIAL_REGISTERED", job=job.id, product=job.product, location=job.location)

    def submit(self, kind, job_id, source, destination, capability, agents=1, profile=None, after=()):
        self.serial += 1
        tid = f"TASK-{self.serial:05d}"
        self.tasks[tid] = Task(tid, kind, job_id, source, destination, Priority.WIP, frozenset([capability]),
                               required_agents=agents, profile=profile, created=self.time, metadata=dict(after=list(after)))
        self.bus.emit(self.time, "TASK_CREATED", task=tid, job=job_id, kind=kind, priority=int(Priority.WIP))
        return self.tasks[tid]

    def acknowledge_pickup(self, tid):
        task = self.tasks[tid]
        if task.status != "ASSIGNED" or task.picked:
            raise ValueError("Invalid or repeated pickup")
        task.picked, task.status = True, "PICKED"
        self.jobs[task.job_id].location = "ON:" + ",".join(task.resources)
        self.bus.emit(self.time, "MEASURED_PICKUP", task=tid, job=task.job_id, kind=task.kind, resources=task.resources)

    def acknowledge_delivery(self, tid):
        task = self.tasks[tid]
        if task.status != "PICKED":
            raise ValueError("No delivery before physical pickup")
        missing = [k for k in task.metadata["after"] if not any(t.kind == k and t.status == "DONE" for t in self.tasks.values())]
        if missing:
            raise ValueError(f"{task.kind} cannot complete before {missing}")
        job = self.jobs[task.job_id]
        job.location, task.status = task.destination, "DONE"
        job.history.append(dict(t=self.time, source=task.source, destination=task.destination, task=tid))
        self.bus.emit(self.time, "MEASURED_DELIVERY", task=tid, job=job.id, kind=task.kind, destination=task.destination,
                      resources=task.resources)
        self.dispatcher.release(task, self.resources, self.time, self.bus)

    def generate(self):
        # the station process starts once the door is on the WS2 fixture: someone has to hold it until the job ends
        if not self.process:
            return
        kind, cap = self.process
        if self.jobs["DOOR-1"].location == "WS2_FIXTURE" and not any(t.kind == kind for t in self.tasks.values()):
            self.submit(kind, "WS2-PROCESS-1", "WS2_FIXTURE", "WS2_DONE", cap, after=("SCREW_INSTALL", "GLASS_DELIVERY"))

    def tick(self, t):
        self.time = t
        for tid, signal in self.backend.poll():
            (self.acknowledge_pickup if signal == "PICKED" else self.acknowledge_delivery)(tid)
        self.generate()
        self.dispatcher.assign(self.tasks, self.jobs, self.resources, BUFFERS, self.backend, self.time, self.bus)


class Backend:
    """The Dispatcher's backend: start() hands a task to the agents, poll() returns acknowledgements a team agrees on."""

    def __init__(self):
        self.agents, self.reports, self.out = {}, {}, []

    def can_execute(self, task, names):
        return all(n in self.agents for n in names)

    def start(self, task, names):
        return True

    def report(self, name, task, signal):
        got = self.reports.setdefault((task.id, signal), set())
        got.add(name)
        if got == set(task.resources):          # a team acknowledgement needs every member's own measurement
            self.out.append((task.id, signal))

    def poll(self):
        out, self.out = self.out, []
        return out


# ------------------------------------------------------------------ agents
class Agent:
    """A program (generator, one step per frame) that reads the bus and reports measured signals to the backend."""

    def __init__(self, name, cell, world):
        self.name, self.cell, self.world, self.cursor, self.label = name, cell, world, 0, "idle"
        self.prog = self.program()

    def step(self):
        next(self.prog, None)

    def events(self):
        return self.cell.bus.events

    def wait_for(self, pred, label=None):
        if label:
            self.label = label
        while True:
            for row in self.events()[self.cursor:]:
                self.cursor += 1
                if pred(row):
                    return row
            yield

    def wait_task(self, kind):
        row = yield from self.wait_for(lambda r: r["event"] in ("TASK_ASSIGNED", "TEAM_FORMED") and self.name in r["resources"]
                                       and self.cell.tasks[r["task"]].kind == kind)
        return self.cell.tasks[row["task"]]

    def report(self, task, signal):
        self.cell.backend.report(self.name, task, signal)

    def over(self, dur, fn):
        """Call fn(s) for s in (0, 1] across dur seconds, one frame per step."""
        n = max(1, int(round(dur * FPS)))
        for i in range(1, n + 1):
            fn(minjerk(i / n))
            yield


class Arm(Agent):
    """UR20 on a carriage riding its rail. Holds the door through an assisted grasp (a fixed joint in the render)."""

    def __init__(self, name, cell, world, end):
        self.end = end                         # door end it grips: +1 the door's local +x end (upstream), -1 the other
        self.rail_y = RAILS[name]
        self.axis = R_DOOR @ np.array([-end, 0, 0.0])   # tool axis, into that end face
        self.cx, self.grip, self.track = X1 - end * DOOR_L / 2, 0.0, False
        self.tip = self.home_tip()
        super().__init__(name, cell, world)

    def step(self):
        if self.track:                         # the carriage rides along level with its door end
            self.cx = float(self.grasp_point()[0])
        super().step()

    def home_tip(self):
        toward = 1.0 if DOOR_Y > self.rail_y else -1.0
        return np.array([self.cx, self.rail_y + toward * 0.85, PED + 0.45]) - self.axis * 0.35

    def grasp_point(self):
        return (self.world.door() @ np.r_[self.end * DOOR_L / 2, 0, GRASP_W, 1])[:3]

    def move_tip(self, a, b, dur):
        a = ev(a)
        yield from self.over(dur, lambda s: setattr(self, "tip", lerp(a, ev(b), s)))

    def follow_door(self):
        self.tip = self.grasp_point()

    def program(self):
        task = yield from self.wait_task("COOPERATIVE_DOOR")
        self.label = "approach door on the moving line"
        self.track = True
        yield from self.move_tip(self.tip, lambda: self.grasp_point() - self.axis * 0.15, 2.4)
        yield from self.move_tip(self.tip, self.grasp_point, 1.0)
        self.label = "grasp (assisted)"
        yield from self.over(0.6, lambda s: setattr(self, "grip", s))
        self.cell.bus.emit(self.cell.time, "GRASP_CLOSED", task=task.id, resources=(self.name,))
        # the lift starts on the partner's grasp event, so both arms run the same door profile
        self.label = "waiting for partner grasp"
        n = 0
        while n < len(task.resources):
            yield from self.wait_for(lambda r: r["event"] == "GRASP_CLOSED" and r["task"] == task.id)
            n += 1
        t_e = max(r["t"] for r in self.events() if r["event"] == "GRASP_CLOSED" and r["task"] == task.id)
        x0 = X1 + BELT * t_e                    # where the line had brought the door when the team closed on it
        self.track, picked = False, False
        while True:   # both arms evaluate the same door profile on the shared clock, so the grippers stay in step
            x, z, phase = door_profile(self.cell.time - t_e, x0)
            self.world.door = lambda T=door_T(x, z): T
            self.follow_door()
            self.cx = float(self.tip[0])
            self.label = ["team lift", "team carry along the rails", "team place on WS2 fixture", "placed"][phase]
            if phase >= 1 and not picked:
                self.report(task, "PICKED")
                picked = True
            if phase == 3:
                break
            yield
        self.report(task, "DELIVERED")
        yield from self.wait_for(lambda r: r["event"] == "TEAM_DISSOLVED" and r["task"] == task.id, "placed")
        hold = [r for r in self.events() if r["event"] == "TASK_ASSIGNED" and r["resources"] == (self.name,)
                and self.cell.tasks[r["task"]].kind == "FIXTURE_HOLD"]
        if hold:
            hold = self.cell.tasks[hold[0]["task"]]
            self.report(hold, "PICKED")       # it never let go: the hold starts with the door already in its gripper
            self.label = "holding door steady"
            while not all(any(t.kind == k and t.status == "DONE" for t in self.cell.tasks.values())
                          for k in ("SCREW_INSTALL", "GLASS_DELIVERY")):
                yield
            yield from self.release()
            self.report(hold, "DELIVERED")
            yield from self.move_tip(self.tip, self.home_tip, 1.6)
            self.label = "done"
            return
        yield from self.release()
        yield from self.move_tip(self.tip, self.home_tip, 1.6)
        self.label = "clearing the G1 path"
        c0 = self.cx
        yield from self.over(3.0, lambda s: (setattr(self, "cx", lerp(c0, X2 - 2.2, s)), setattr(self, "tip", self.home_tip())))
        self.cell.bus.emit(self.cell.time, "ZONE_CLEAR", zone="WS2_IN_SIDE", resources=(self.name,))
        self.label = "idle"

    def release(self):
        self.label = "release"
        yield from self.over(0.6, lambda s: setattr(self, "grip", 1 - s))
        yield from self.move_tip(self.tip, self.tip - self.axis * 0.15, 0.9)


class G1Body:
    """One G1 in its own MuJoCo world. Idle, it stands still on a fixed standing pose (no physics stepping, no
    marching on the spot). To go somewhere it turns towards the target, walks straight with Unitree's walking policy
    (physics), stops, turns to face the work, and stands again."""
    JOINTS = LEG_JOINT_NAMES + UPPER_BODY_JOINT_NAMES

    def __init__(self, x, y, yaw, cfg):
        spec = mujoco.MjSpec.from_string('<mujoco><option timestep="0.002"/><worldbody>'
                                         '<geom type="plane" size="40 40 0.1"/></worldbody></mujoco>')
        spec.attach(mujoco.MjSpec.from_file(str(cfg.robot_xml_23dof)), prefix="g1_", frame=spec.worldbody.add_frame())
        self.m = spec.compile()
        self.m.opt.timestep = cfg.simulation_dt
        self.d = mujoco.MjData(self.m)
        self.cfg = cfg
        self.legs, self.upper = G1Controller(self.m, cfg, "g1_"), G1UpperBodyController(self.m, "g1_")
        self.qadr = [self.m.jnt_qposadr[self.m.joint("g1_" + n).id] for n in self.JOINTS]
        self.q_stand = np.r_[cfg.default_angles, np.zeros(len(UPPER_BODY_JOINT_NAMES))]
        # standing height: default leg angles, lowest foot collision point on the floor
        b = self.legs.base_qpos
        self.d.qpos[b:b + 7] = [0, 0, 1.0, 1, 0, 0, 0]
        self.d.qpos[self.qadr] = self.q_stand
        mujoco.mj_kinematics(self.m, self.d)
        feet = [g for g in range(self.m.ngeom) if "ankle_roll" in self.m.body(self.m.geom_bodyid[g]).name and self.m.geom_contype[g]]
        low = min(self.d.geom_xpos[g][2] - (self.m.geom_size[g][0] if self.m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE else 0) for g in feet)
        self.z_stand = 1.0 - low
        self.mode, self.goal, self.phase, self.t_off, self.now = "stand", None, None, 0.0, 0.0
        self.root, self.q = self.stand_root(x, y, yaw), self.q_stand.copy()
        self.blend = None

    @staticmethod
    def yaw_of(root):
        w, z = root[3], root[6]
        return 2 * math.atan2(z, w)

    def stand_root(self, x, y, yaw):
        return np.array([x, y, self.z_stand, math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])

    def xy(self):
        return self.root[:2].copy()

    def yaw(self):
        return self.yaw_of(self.root)

    def heading(self):
        return tf(rot(2, self.yaw()), self.root[:3])

    def goto(self, goal):
        """Start a walk from the standing pose: physics resumes exactly at that pose with zero velocity."""
        self.goal, self.phase = goal, "turn"
        if self.mode == "stand":
            b, bd = self.legs.base_qpos, self.legs.base_dof
            self.d.qpos[b:b + 7] = self.root
            self.d.qpos[self.qadr] = self.q_stand
            self.d.qvel[:] = 0
            self.legs.action[:] = 0
            self.legs.target_q = self.cfg.default_angles.copy()
            self.upper.target_q[:] = 0
            mujoco.mj_forward(self.m, self.d)
            self.t_off = self.d.time - self.now
            self.mode, self.blend = "walk", None

    def walk_cmd(self):
        g = ev(self.goal)
        b = self.legs.base_qpos
        xy, yaw = self.d.qpos[b:b + 2], self.legs.yaw(self.d)
        dvec = np.array(g[:2]) - xy
        dist = float(np.linalg.norm(dvec))
        head = math.atan2(dvec[1], dvec[0])
        if self.phase == "turn":
            err = wrap(head - yaw)
            if dist < 0.1 or abs(err) < 0.12:
                self.phase = "go" if dist >= 0.1 else "face"
            else:
                return np.array([0, 0, np.clip(2.0 * err, -1.0, 1.0)], np.float32)
        if self.phase == "go":
            err = wrap(head - yaw)
            if dist < 0.07 or (dist < 0.25 and abs(err) > 1.2):
                self.phase = "face"
            else:
                v = float(np.clip(1.0 * dist, 0.2, 0.75)) * max(0.0, math.cos(err))
                return np.array([v, 0, np.clip(2.0 * err, -1.0, 1.0)], np.float32)
        yerr = wrap(g[2] - yaw)
        if abs(yerr) < 0.08:
            self.phase = "done"
            return np.zeros(3, np.float32)
        return np.array([0, 0, np.clip(2.0 * yerr, -1.0, 1.0)], np.float32)

    def advance(self, t):
        dt, self.now = t - self.now, t
        if self.mode == "stand":
            if BELT_X[0] < self.root[0] < BELT_X[1] and BELT_Y[0] < self.root[1] < BELT_Y[1]:   # on the belt: ride it
                self.root[0] += BELT * dt
                if self.blend:
                    self.stand_target[0] += BELT * dt
            if self.blend:                     # settle from the last walking pose onto the standing pose
                r0, q0, k = self.blend
                s = minjerk(k / 12)
                self.root = np.r_[lerp(r0[:3], self.stand_target[:3], s), self.stand_target[3:]]
                self.q = lerp(q0, self.q_stand, s)
                self.blend = (r0, q0, k + 1) if k < 12 else None
            return
        while self.d.time - self.t_off < t:
            if self.step_i % self.cfg.control_decimation == 0:
                cmd = self.walk_cmd()
                self.legs.update_policy(self.d, cmd, gait_phase(self.d))
            self.legs.apply(self.d)
            self.upper.apply(self.d)
            mujoco.mj_step(self.m, self.d)
            self.step_i += 1
        b = self.legs.base_qpos
        self.root, self.q = self.d.qpos[b:b + 7].copy(), self.d.qpos[self.qadr].copy()
        if self.phase == "done":
            self.mode = "stand"
            self.stand_target = self.stand_root(self.root[0], self.root[1], self.yaw())
            self.blend = (self.root.copy(), self.q.copy(), 1)

    step_i = 0


class Humanoid(Agent):
    """G1, legs and root replayed from the MuJoCo walking policy, hands by IK in the render."""

    def __init__(self, name, cell, world, start, cfg):
        self.body = G1Body(*start, cfg)
        self.hand = {"l": None, "r": None}     # callables -> world point
        self.w = {"l": 0.0, "r": 0.0}           # 0: relaxed arm, 1: hand on its IK target
        super().__init__(name, cell, world)

    def H(self, local):
        return lambda: self.body.heading() @ local

    def walk(self, goal, label=None):
        """Turn to the target, walk straight, stop, face the work. goal: (x, y, yaw) or a callable for a moving target."""
        if label:
            self.label = label
        self.body.goto(goal)
        n = 0
        while self.body.mode == "walk" and n < 40 * FPS:
            n += 1
            yield

    def hold(self, obj, grips, sides=("l", "r")):
        for sd in sides:
            self.hand[sd] = (lambda sd=sd: (self.world.obj[obj]() @ np.r_[grips[sd], 1])[:3])

    def move_obj(self, obj, a, b, dur):
        """Object pose from a to b (matrices or callables, evaluated live), hands staying on their grips."""
        def f(s):
            self.world.obj[obj] = lambda: blend(ev(a), ev(b), s)
        yield from self.over(dur, f)
        self.world.obj[obj] = b if callable(b) else (lambda T=b: T)

    def reach(self, sides, dur, out=False):
        yield from self.over(dur, lambda s: [self.w.__setitem__(sd, 1 - s if out else s) for sd in sides])

    def wait(self, dur, label):
        self.label = label
        yield from self.over(dur, lambda s: None)

    # ---- the steps, shared by the team members and the solo worker
    def pick_glass(self, task):
        """Grip the suction lifter already on the racked pane, lift, bring it to the carry pose; the pane hangs on the cups."""
        self.label = "picking glass with the suction lifter"
        g = self.world.obj
        g["glass"] = lambda: g["lifter"]() @ np.linalg.inv(LIFT_IN_GLASS)
        self.hold("lifter", LIFT_GRIPS)
        yield from self.reach("lr", 1.4)
        l0 = GLASS_RACK_T @ LIFT_IN_GLASS
        yield from self.move_obj("lifter", l0, l0 @ tf(None, (0, 0, 0.06)), 0.8)
        self.report(task, "PICKED")
        yield from self.move_obj("lifter", g["lifter"], self.H(LIFT_CARRY), 1.2)

    def place_glass(self, task):
        """Set the pane in the window frame, release the cups, lift the holder off and away."""
        self.label = "setting glass in the window frame"
        g, door = self.world.obj, self.world.door
        seat = lambda: door() @ GLASS_IN_DOOR @ LIFT_IN_GLASS  # noqa: E731
        above = lambda: seat() @ tf(None, (0, -0.12, 0.06))  # noqa: E731
        yield from self.move_obj("lifter", g["lifter"], above, 1.4)
        yield from self.move_obj("lifter", g["lifter"], seat, 1.2)
        g["glass"] = lambda: door() @ GLASS_IN_DOOR                # the frame holds the pane now
        self.label = "releasing the suction cups"
        yield from self.over(0.6, lambda s: None)
        self.report(task, "DELIVERED")
        self.label = "lifting the holder away"
        yield from self.move_obj("lifter", g["lifter"], lambda: seat() @ tf(None, (0, -0.16, 0.10)), 1.0)
        yield from self.move_obj("lifter", g["lifter"], self.H(LIFT_CARRY), 1.2)

    def pick_kit(self, task):
        self.label = "picking screwdriver and screw"
        g = self.world.obj
        self.hold("driver", {"r": np.zeros(3)}, "r")
        yield from self.reach("r", 1.3)
        yield from self.move_obj("driver", DRIVER_SHELF_T, DRIVER_SHELF_T @ tf(None, (0, 0, 0.06)), 0.6)
        self.hand["l"] = lambda: SCREW_BIN
        yield from self.reach("l", 1.2)
        # the screw rides the right hand onto the magnetic bit while the driver comes to the carry pose
        tip = lambda: (g["driver"]() @ np.r_[TIPV, 1])[:3]  # noqa: E731
        bit = lambda: g["driver"]() @ tf(None, TIPV)  # noqa: E731  screw frame: head, x along the shank
        g["screw"] = lambda: tf(g["driver"]()[:3, :3], self.hand["l"]())
        start = SCREW_BIN.copy()
        a0 = g["driver"]()
        yield from self.over(1.2, lambda s: (g.__setitem__("driver", lambda: blend(a0, self.H(DRIVER_CARRY)(), s)),
                                             self.hand.__setitem__("l", lambda: lerp(start, tip(), s))))
        g["driver"] = self.H(DRIVER_CARRY)
        g["screw"] = bit
        yield from self.reach("l", 0.6, out=True)
        self.report(task, "PICKED")

    def drive_screw(self, task):
        self.label = "driving the screw"
        g, door = self.world.obj, self.world.door
        face = -1.0 if SCREW_IN_DOOR[1] > 0 else 1.0     # +1 inner face (door -y side), -1 outer face
        R_in = door()[:3, :3] @ rot(2, face * math.pi / 2)  # driver x into that face
        out = -face * door()[:3, 1]                    # out of the face
        hole = lambda: (door() @ np.r_[SCREW_IN_DOOR, 1])[:3]  # noqa: E731
        # driver pose with the screw head d above its seated position (bit tip on the head)
        engage = lambda d: (lambda: tf(R_in, hole() + out * (0.004 + d) - R_in @ TIPV))  # noqa: E731
        yield from self.move_obj("driver", g["driver"], engage(0.08), 1.4)
        yield from self.move_obj("driver", g["driver"], engage(0.024), 0.8)
        self.world.screw_state = 1
        yield from self.over(RUNDOWN, lambda s: (g.__setitem__("driver", engage(0.024 * (1 - s))),
                                                 setattr(self.world, "spin", self.world.spin + 0.9)))
        local = np.linalg.inv(door()) @ tf(R_in, hole() + out * 0.004)
        g["screw"] = lambda: door() @ local
        self.world.screw_state = 2
        self.report(task, "DELIVERED")
        yield from self.wait(0.6, "screw seated")
        yield from self.move_obj("driver", g["driver"], self.H(DRIVER_CARRY), 1.2)


def delivered(**match):
    return lambda r: r["event"] == "MEASURED_DELIVERY" and all(r.get(k) == v for k, v in match.items())


class GlassCarrier(Humanoid):
    def program(self):
        task = yield from self.wait_task("GLASS_DELIVERY")
        yield from self.pick_glass(task)
        yield from self.walk(W_B, "carrying glass to WS2")
        yield from self.wait_for(delivered(job="DOOR-1"), "waiting for DOOR DELIVERED")
        yield from self.wait_for(lambda r: r["event"] == "ZONE_CLEAR", "waiting for ARM R to clear")
        yield from self.walk(VIA_B, "approach WS2")
        yield from self.walk(P_B)
        yield from self.place_glass(task)
        yield from self.walk((P_B[0], DOOR_Y + 0.95, -math.pi / 2), "stepping back with the holder")
        self.label = "done"


class ScrewDriver(Humanoid):
    def program(self):
        task = yield from self.wait_task("SCREW_INSTALL")
        yield from self.pick_kit(task)
        yield from self.walk(A_VIA, "carrying screw and driver to WS2")
        yield from self.walk(A_VIA2)
        yield from self.walk(W_A)
        # the OUT side is clear of ARM R, so A goes in on DOOR DELIVERED and works while B sets the glass
        yield from self.wait_for(delivered(job="DOOR-1"), "waiting for DOOR DELIVERED")
        yield from self.walk(VIA_A, "approach WS2 (outer side)")
        yield from self.walk(P_A)
        yield from self.drive_screw(task)
        yield from self.walk(VIA_A, "stepping back")
        self.label = "done"


class SoloWorker(Humanoid):
    """The station as one person does it today (a G1 stands in for the worker): the door keeps moving on the line
    conveyor while the worker fetches the glass, fits it, fetches the screw and the driver, and drives the screw."""

    def at_door(self, dx, y):
        door = self.world.door
        return lambda: (float(door()[0, 3]) + dx + BELT * 1.0, y, math.pi)   # aim a second ahead of the line

    def program(self):
        task = yield from self.wait_task("GLASS_DELIVERY")
        yield from self.walk(S_B, "walking to the glass rack")
        yield from self.pick_glass(task)
        yield from self.walk(self.at_door(P_B[0] - X2, P_B[1]), "carrying glass to the moving door")
        yield from self.place_glass(task)
        yield from self.wait(2.0, "pressing the glass into its clips")
        yield from self.reach("lr", 0.8, out=True)
        self.report(task, "DELIVERED")
        task = yield from self.wait_task("SCREW_INSTALL")
        yield from self.walk(S_A, "walking to the parts shelf")
        yield from self.pick_kit(task)
        yield from self.walk(self.at_door(P_A[0] - X2, P_A[1]), "carrying screw and driver to the door")
        yield from self.drive_screw(task)
        task = yield from self.wait_task("STATION_RELEASE")
        self.report(task, "PICKED")
        yield from self.wait(1.5, "checking the door")
        self.report(task, "DELIVERED")
        self.label = "done"


class BackgroundWalker(Humanoid):
    """A G1 walking an aisle on the far side of the station: line background, not part of the job."""

    def program(self):
        self.label = "walking the aisle"
        while True:
            yield from self.walk((21.0, AISLE_Y, 0.0))
            yield from self.walk((9.0, AISLE_Y, math.pi))


AISLE_Y = -1.85                    # OUT-side aisle between the supervision desks and the body line


class World:
    """Passive objects: each pose is a callable, so a held object follows whoever holds it."""

    def __init__(self):
        self.t = 0.0
        self.door = lambda: door_T(X1 + BELT * self.t, DOOR_Z0)    # on its trolley, riding the WS30 belt
        self.obj = {"glass": lambda: GLASS_RACK_T, "driver": lambda: DRIVER_SHELF_T, "lifter": lambda: GLASS_RACK_T @ LIFT_IN_GLASS,
                    "screw": lambda: tf(None, SCREW_BIN)}
        self.screw_state, self.spin = 0, 0.0


# ------------------------------------------------------------------ run
def build(solo):
    cfg = demo_config()
    backend = Backend()
    world = World()
    if solo:
        # ponytail: the worker stands in for a person, so its payload is a person's, not a G1's
        resources = [Resource("WORKER", frozenset(["glass_handling", "screw_driving", "station_release"]), (1.5, -1.0), 30.0)]
        cell = Cell(resources, backend, process=None)
        agents = [SoloWorker("WORKER", cell, world, (1.6, -1.1, math.pi / 2), cfg)]
    else:
        # only the OUT-side arm can hold the door at WS2 without reaching across the G1 work side
        resources = [Resource("ARM_L", frozenset(["team_transport", "fixture_hold"]), (X1, RAILS["ARM_L"]), 20.0, frozenset(["two_end"])),
                     Resource("ARM_R", frozenset(["team_transport"]), (X1, RAILS["ARM_R"]), 20.0, frozenset(["two_end"])),
                     Resource("HUMANOID_A", frozenset(["screw_driving"]), S_A[:2], 3.0),
                     Resource("HUMANOID_B", frozenset(["glass_handling"]), S_B[:2], 3.0)]
        cell = Cell(resources, backend)
        agents = [Arm("ARM_L", cell, world, +1), Arm("ARM_R", cell, world, -1),
                  ScrewDriver("HUMANOID_A", cell, world, S_A, cfg), GlassCarrier("HUMANOID_B", cell, world, S_B, cfg),
                  BackgroundWalker("HUMANOID_C", cell, world, (9.0, AISLE_Y, 0.0), cfg)]
    backend.agents = {a.name: a for a in agents}
    cell.add_job(Job("DOOR-1", "CAR_DOOR", (), "LINE_CARRIER", mass_kg=25.0, dimensions=(1.2, 0.1, 1.05), required_grasp_points=2))
    cell.add_job(Job("GLASS-1", "WINDOW_GLASS", (), "GLASS_RACK", mass_kg=2.5))
    cell.add_job(Job("SCREW-1", "SCREW_AND_DRIVER", (), "PARTS_SHELF", mass_kg=0.6))
    # the station process on the door: the fixture (or the carrier) carries the weight, a hold only steadies it
    cell.add_job(Job("WS2-PROCESS-1", "STATION_PROCESS", (), "WS2_FIXTURE", mass_kg=0.0))
    if solo:
        cell.submit("GLASS_DELIVERY", "GLASS-1", "GLASS_RACK", "LINE_DOOR", "glass_handling")
        cell.submit("SCREW_INSTALL", "SCREW-1", "PARTS_SHELF", "LINE_DOOR", "screw_driving", after=("GLASS_DELIVERY",))
        cell.submit("STATION_RELEASE", "WS2-PROCESS-1", "LINE_DOOR", "LINE_DONE", "station_release", after=("SCREW_INSTALL",))
    else:
        # a single UR20 (20 kg) cannot take the 25 kg door, so the Dispatcher has to form a team of two
        cell.submit("COOPERATIVE_DOOR", "DOOR-1", "LINE_CARRIER", "WS2_FIXTURE", "team_transport", agents=2, profile="two_end")
        cell.submit("GLASS_DELIVERY", "GLASS-1", "GLASS_RACK", "WS2_DOOR", "glass_handling", after=("COOPERATIVE_DOOR",))
        cell.submit("SCREW_INSTALL", "SCREW-1", "PARTS_SHELF", "WS2_DOOR", "screw_driving", after=("COOPERATIVE_DOOR",))
    return cell, world, agents, SOLO_HEADLINES if solo else HEADLINES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--solo", action="store_true")
    ap.add_argument("--tail", type=float, default=3.0, help="seconds to keep recording after the station completes")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    cell, world, agents, headlines = build(a.solo)
    protocol, _ = load_trust()
    keys = {name: Ed25519PrivateKey.generate() for name in cell.resources}
    registry = {n: {"public_key": base64.b64encode(k.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)).decode(),
                    "status": "active"} for n, k in keys.items()}
    signer = CustodySigner(keys, receiver="ledger@door-team")
    cell.bus.subscribers.append(signer)
    arms = [x for x in agents if isinstance(x, Arm)]
    hums = [x for x in agents if isinstance(x, Humanoid)]
    rec = {k: [] for k in ("t", "door", "glass", "driver", "screw", "lifter", "screw_state", "spin")}
    for x in arms:
        for k in ("cx", "tip", "axis", "grip"):
            rec[f"{x.name}.{k}"] = []
    for x in hums:
        for k in ("root", "q", "hand", "w"):
            rec[f"{x.name}.{k}"] = []
    labels = {x.name: [] for x in agents}
    fi, done_t, wall = 0, None, time.time()
    while fi < 120 * FPS:
        t = round(fi * DT, 4)
        world.t = t
        cell.tick(t)
        for x in agents:
            x.step()
        for x in hums:
            x.body.advance(t)
        rec["t"].append(t)
        rec["door"].append(world.door())
        for k in ("glass", "driver", "screw", "lifter"):
            rec[k].append(world.obj[k]())
        rec["screw_state"].append(world.screw_state)
        rec["spin"].append(world.spin)
        for x in arms:
            rec[f"{x.name}.cx"].append(x.cx)
            rec[f"{x.name}.tip"].append(x.tip.copy())
            rec[f"{x.name}.axis"].append(x.axis)
            rec[f"{x.name}.grip"].append(x.grip)
        for x in hums:
            rec[f"{x.name}.root"].append(x.body.root.copy())
            rec[f"{x.name}.q"].append(x.body.q.copy())
            rec[f"{x.name}.hand"].append([x.hand[s]() if x.hand[s] else np.zeros(3) for s in "lr"])
            rec[f"{x.name}.w"].append([x.w[s] for s in "lr"])
        for x in agents:
            if not labels[x.name] or labels[x.name][-1][1] != x.label:
                labels[x.name].append((round(t, 3), x.label))
        if done_t is None and any(r["event"] == "MEASURED_DELIVERY" and r["kind"] in ("FIXTURE_HOLD", "STATION_RELEASE") for r in cell.bus.events[-8:]):
            done_t = t
        if done_t is not None and t > done_t + a.tail and all(x.label in ("done", "idle") for x in agents
                                                               if not isinstance(x, BackgroundWalker)):
            break
        fi += 1
    print(f"[orchestrate] {len(rec['t'])} frames, {t:.1f} s sim, station complete at {done_t}, wall {time.time() - wall:.0f} s")
    for x in hums:
        z = np.array(rec[f"{x.name}.root"])[:, 2]
        print(f"[orchestrate] {x.name} pelvis z {z.min():.3f} to {z.max():.3f}")
        assert z.min() > 0.55, f"{x.name} fell"
    # signed custody log: check every record against the public registry right away
    verifier = protocol.Verifier(signer.receiver, registry)
    checks = [verifier.check(r) for r in signer.records]
    assert all(c == (True, "ok") for c in checks), checks
    tamper = dict(signer.records[-1], destination="SCRAP")
    assert verifier.check(tamper) == (False, "bad signature")
    events = []
    for row in cell.bus.events:
        sigs = [r for r in signer.records if r["request"] == row["event"] and r.get("t") == row["t"] and r.get("task") == row.get("task")]
        kind = row.get("kind") or (cell.tasks[row["task"]].kind if row.get("task") in cell.tasks else None)
        head = next((h for e, k, h in headlines if row["event"] == e and kind == k), None)
        events.append(dict({k: (list(v) if isinstance(v, tuple) else v) for k, v in row.items()}, headline=head,
                           signed_by=[r["from"] for r in sigs], sig=[r["sig"][:12] for r in sigs]))
    with open(a.out / "events.jsonl", "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
    json.dump(signer.records, open(a.out / "signed_events.json", "w"), indent=1)
    json.dump(dict(receiver=signer.receiver, registry=registry, verified=len(checks)), open(a.out / "registry.json", "w"), indent=1)
    arrays = {k: np.asarray(v, float) for k, v in rec.items()}
    np.savez_compressed(a.out / "plan.npz", **arrays)
    json.dump(dict(fps=FPS, solo=a.solo, joints=list(G1Body.JOINTS), arms=[x.name for x in arms], humanoids=[x.name for x in hums],
                   labels=labels, events=events, done_t=done_t,
                   layout=dict(X1=X1, X2=X2, FIX_Z=FIX_Z, RAILS=RAILS, RAIL_X=RAIL_X, PED=PED, DOOR_L=DOOR_L, BELT=BELT,
                               BELT_X=BELT_X, BELT_Y=BELT_Y, DOOR_Y=DOOR_Y, CARRIER=(CARRIER_X0, CARRIER_Y),
                               DOOR_ON_CARRIER=(DOOR_DX, DOOR_DY, DOOR_Z0), GLASS_PTS=GLASS_PTS, RACK=RACK, GR=GR,
                               SCREW_IN_DOOR=list(SCREW_IN_DOOR), SCREW_BIN=list(SCREW_BIN), WS30=True)),
              open(a.out / "plan.json", "w"), indent=1)
    for e in events:
        if e["headline"]:
            print(f"  {e['t']:6.2f}  {e['headline']:28s} signed by {', '.join(e['signed_by'])}")
    print(f"[orchestrate] {len(signer.records)} signed records, all verified; tampered copy rejected")


if __name__ == "__main__":
    main()
