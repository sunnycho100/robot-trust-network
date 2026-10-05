"""MuJoCo sim driven by a vision-language model through skill calls.

Run:  .venv/bin/mjpython vla_sim.py --robot panda     (or --robot go2)
Type a command like "put the red block on the blue block" and watch it move.
"""
import argparse, base64, io, json, math, os, time

import mujoco, mujoco.viewer
import numpy as np
from dotenv import load_dotenv
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
MENAGERIE = os.path.join(HERE, "menagerie")
COLORS = {"red": [0.9, 0.15, 0.15, 1], "green": [0.15, 0.8, 0.2, 1], "blue": [0.15, 0.3, 0.9, 1]}


def build(scene_xml, add_objects):
    spec = mujoco.MjSpec.from_file(scene_xml)
    for k in list(spec.keys):  # menagerie keyframes break once we add free bodies
        spec.delete(k)
    add_objects(spec.worldbody)
    model = spec.compile()
    return model, mujoco.MjData(model)


class Robot:
    """Shared plumbing: viewer sync, camera image, running a skill by name."""
    skills = []  # OpenAI-style tool schemas, set by subclasses

    def __init__(self, model, data, viewer=None):
        self.m, self.d, self.viewer = model, data, viewer
        self.renderer = mujoco.Renderer(model, 480, 640)
        self.cam = mujoco.MjvCamera()

    def sync(self, sim_dt):
        if self.viewer:
            self.viewer.sync()
            time.sleep(sim_dt)  # ponytail: sleep-based pacing, fine for a demo viewer

    def image_png(self):
        self.renderer.update_scene(self.d, camera=self.cam)
        buf = io.BytesIO()
        Image.fromarray(self.renderer.render()).save(buf, format="PNG")
        return buf.getvalue()

    def run(self, name, args):
        try:
            return getattr(self, "skill_" + name)(**args)
        except Exception as e:  # report bad calls back to the model instead of crashing
            return f"error: {e}"

    def pos(self, name):
        return self.d.body(name).xpos.copy()


# ---------------------------------------------------------------- Franka Panda
def tool(name, desc, **params):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": params, "required": list(params)}}}


OBJ = {"type": "string", "enum": [f"{c}_block" for c in COLORS]}
NUM = {"type": "number"}


class Panda(Robot):
    HOME = [0, 0, 0, -1.57079, 0, 1.57079, -0.7853]
    TCP = 0.1034  # hand origin to fingertip center, along the hand z axis
    skills = [
        tool("pick", "Grasp a block and lift it.", object=OBJ),
        tool("place_on", "Place the held block on top of another block.", target=OBJ),
        tool("place_at", "Place the held block on the floor at (x, y) metres. Reach is about 0.3 to 0.75 from the robot base at (0, 0).", x=NUM, y=NUM),
        tool("move_to", "Move the gripper tip to (x, y, z) metres, gripper pointing down.", x=NUM, y=NUM, z=NUM),
        tool("open_gripper", "Open the gripper."),
        tool("home", "Return the arm to its rest pose."),
        tool("done", "Call when the command is finished, with a one sentence summary.", summary={"type": "string"}),
    ]
    OBJECTS = "red_block, green_block, blue_block (4 cm cubes on the floor in front of the arm)"
    LIMITS = [(-2.897, 2.897), (-1.763, 1.763), (-2.897, 2.897), (-3.072, -0.07), (-2.897, 2.897), (-0.018, 3.752), (-2.897, 2.897)]
    JOINT_SKILLS = [
        tool("set_joints", "Move all 7 arm joints to these angles in radians (base j1 to wrist j7), then set the gripper. "
             "Rest pose is j1..j7 = [0, 0, 0, -1.571, 0, 1.571, -0.785], fingertip at (0.554, 0, 0.521) pointing down. "
             "The gripper opens 8 cm wide; to grasp, open it, put the fingertip at the block centre, then close it.",
             gripper={"type": "string", "enum": ["open", "closed"]},
             **{f"j{i + 1}": {"type": "number", "minimum": lo, "maximum": hi} for i, (lo, hi) in enumerate(LIMITS)}),
        tool("done", "Call when the command is finished, with a one sentence summary.", summary={"type": "string"}),
    ]

    @classmethod
    def make(cls):
        def add(world):
            for (c, rgba), xy in zip(COLORS.items(), [(0.5, 0.2), (0.6, 0.0), (0.45, -0.2)]):
                b = world.add_body(name=f"{c}_block", pos=[*xy, 0.02])
                b.add_freejoint()
                b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.02] * 3, rgba=rgba, mass=0.05, friction=[1.5, 0.05, 0.01])
        m, d = build(os.path.join(MENAGERIE, "franka_emika_panda", "scene.xml"), add)
        return m, d

    def __init__(self, m, d, viewer=None):
        super().__init__(m, d, viewer)
        d.qpos[:7] = d.ctrl[:7] = self.HOME
        d.qpos[7:9] = 0.04
        d.ctrl[7] = 255
        mujoco.mj_forward(m, d)
        self.hand = m.body("hand").id
        self.down = d.xquat[self.hand].copy()  # keep the home gripper orientation
        self.cam.lookat[:] = [0.45, 0, 0.1]
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 1.4, 180, -35
        self.held = None

    def wait(self, seconds):
        steps = int(seconds / self.m.opt.timestep)
        for i in range(steps):
            mujoco.mj_step(self.m, self.d)
            if i % 8 == 0:
                self.sync(self.m.opt.timestep * 8)

    def ik(self, target):
        """Damped least squares on a scratch copy; returns 7 joint angles."""
        d = mujoco.MjData(self.m)
        d.qpos[:] = self.d.qpos
        jp, jr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
        for _ in range(200):
            mujoco.mj_kinematics(self.m, d)
            mujoco.mj_comPos(self.m, d)
            R = d.xmat[self.hand].reshape(3, 3)
            tip = d.xpos[self.hand] + R[:, 2] * self.TCP
            neg, dq, rot = np.zeros(4), np.zeros(4), np.zeros(3)
            mujoco.mju_negQuat(neg, d.xquat[self.hand])
            mujoco.mju_mulQuat(dq, self.down, neg)
            mujoco.mju_quat2Vel(rot, dq, 1)
            err = np.concatenate([np.asarray(target) - tip, rot])
            if np.linalg.norm(err) < 1e-4:
                break
            mujoco.mj_jac(self.m, d, jp, jr, tip, self.hand)
            J = np.vstack([jp, jr])[:, :7]
            d.qpos[:7] += J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), err)
            d.qpos[:7] = np.clip(d.qpos[:7], self.m.jnt_range[:7, 0], self.m.jnt_range[:7, 1])
        if np.linalg.norm(err[:3]) > 0.01:
            raise ValueError(f"target {np.round(target, 3).tolist()} is out of reach")
        return d.qpos[:7].copy()

    def go(self, target, seconds=1.2):
        self.go_q(self.ik(target), seconds)

    def go_q(self, q1, seconds=1.2):
        q0 = self.d.ctrl[:7].copy()
        steps = int(seconds / self.m.opt.timestep)
        for i in range(steps):
            s = (1 - math.cos(math.pi * (i + 1) / steps)) / 2  # smooth start and stop
            self.d.ctrl[:7] = q0 + s * (q1 - q0)
            mujoco.mj_step(self.m, self.d)
            if i % 8 == 0:
                self.sync(self.m.opt.timestep * 8)
        self.wait(0.3)

    def gripper(self, open_):
        self.d.ctrl[7] = 255 if open_ else 0
        self.wait(0.6)

    def skill_set_joints(self, gripper, **joints):
        q = np.array([joints[f"j{i}"] for i in range(1, 8)], float)
        lo, hi = self.m.jnt_range[:7, 0], self.m.jnt_range[:7, 1]
        clipped = [f"j{i + 1}" for i in range(7) if not lo[i] <= q[i] <= hi[i]]
        self.go_q(np.clip(q, lo, hi), 1.5)
        self.d.ctrl[7] = 255 if gripper == "open" else 0
        self.wait(0.6)
        note = f" (clipped to limits: {', '.join(clipped)})" if clipped else ""
        return f"moved{note}. fingertip now at {np.round(self.tip(), 3).tolist()}"

    def tip(self):
        return self.d.xpos[self.hand] + self.d.xmat[self.hand].reshape(3, 3)[:, 2] * self.TCP

    def skill_move_to(self, x, y, z):
        self.go([x, y, z])
        return f"gripper at {[x, y, z]}"

    def skill_open_gripper(self):
        self.gripper(True)
        self.held = None
        return "gripper open"

    def skill_home(self):
        self.go(self.d.xpos[self.hand] + [0, 0, 0.1])
        q0 = self.d.ctrl[:7].copy()
        for i in range(600):
            self.d.ctrl[:7] = q0 + (i + 1) / 600 * (np.array(self.HOME) - q0)
            mujoco.mj_step(self.m, self.d)
            if i % 8 == 0:
                self.sync(self.m.opt.timestep * 8)
        return "arm at rest pose"

    def skill_pick(self, object):
        if self.held:
            return f"error: already holding {self.held}, place it first"
        p = self.pos(object)
        self.gripper(True)
        self.go(p + [0, 0, 0.12])
        self.go(p + [0, 0, 0.005], 0.8)
        self.gripper(False)
        self.go(p + [0, 0, 0.15], 0.8)
        lifted = self.pos(object)[2] > p[2] + 0.05
        self.held = object if lifted else None
        return f"picked {object}" if lifted else f"failed to grasp {object}, it slipped"

    def _place(self, x, y, z):
        if not self.held:
            return "error: not holding anything"
        self.go([x, y, z + 0.1])
        self.go([x, y, z + 0.005], 0.8)
        obj = self.held
        self.skill_open_gripper()
        self.go([x, y, z + 0.12], 0.6)
        return f"placed {obj}, now at {np.round(self.pos(obj), 3).tolist()}"

    def skill_place_on(self, target):
        p = self.pos(target)
        return self._place(p[0], p[1], p[2] + 0.04)

    def skill_place_at(self, x, y):
        return self._place(x, y, 0.02)

    def state(self):
        if self.skills is self.JOINT_SKILLS:
            return f"joints {np.round(self.d.qpos[:7], 3).tolist()}; fingertip at {np.round(self.tip(), 3).tolist()}; " \
                   f"gripper {'open' if self.d.ctrl[7] > 127 else 'closed'}; " + \
                   "; ".join(f"{c}_block at {np.round(self.pos(c + '_block'), 3).tolist()}" for c in COLORS)
        return "; ".join(f"{c}_block at {np.round(self.pos(c + '_block'), 2).tolist()}" for c in COLORS) + \
               f"; holding: {self.held or 'nothing'}"


# ---------------------------------------------------------------- Unitree Go2
class Go2(Robot):
    """Kinematic Go2: the base glides and the legs play a trot animation.

    ponytail: no physics walking. A real gait needs a trained locomotion policy;
    swap skill_walk_to for one if you want contact-accurate motion.
    """
    STAND = [0, 0.9, -1.8] * 4
    SIT = [0, 0.9, -1.8, 0, 0.9, -1.8, 0, 2.2, -2.6, 0, 2.2, -2.6]
    TARGETS = {"red": (2.0, 1.5), "green": (2.5, -1.5), "blue": (-1.5, 2.0)}
    skills = [
        tool("walk_to", "Walk to a coloured marker.", target={"type": "string", "enum": [f"{c}_marker" for c in COLORS]}),
        tool("walk_to_xy", "Walk to floor point (x, y) metres.", x=NUM, y=NUM),
        tool("turn", "Turn in place, positive is left, in degrees.", degrees=NUM),
        tool("sit", "Sit down."),
        tool("stand", "Stand up."),
        tool("done", "Call when the command is finished, with a one sentence summary.", summary={"type": "string"}),
    ]
    OBJECTS = "red_marker, green_marker, blue_marker (coloured posts on the floor)"

    @classmethod
    def make(cls):
        def add(world):
            for c, rgba in COLORS.items():
                world.add_geom(name=f"{c}_marker", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.12, 0.25, 0],
                               pos=[*cls.TARGETS[c], 0.25], rgba=rgba, contype=0, conaffinity=0)
        return build(os.path.join(MENAGERIE, "unitree_go2", "scene.xml"), add)

    def __init__(self, m, d, viewer=None):
        super().__init__(m, d, viewer)
        self.x = self.y = self.yaw = 0.0
        self.h, self.legs = 0.27, np.array(self.STAND, float)
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 6.5, -150, -40
        self.pose()

    def pose(self, phase=None):
        q = self.d.qpos
        q[:3] = self.x, self.y, self.h
        q[3:7] = math.cos(self.yaw / 2), 0, 0, math.sin(self.yaw / 2)
        q[7:19] = self.legs
        if phase is not None:  # diagonal pairs (FL+RR, FR+RL) swing in opposite phase
            for leg, sign in enumerate([1, -1, -1, 1]):
                s = math.sin(phase) * sign
                q[8 + 3 * leg] += 0.3 * s
                q[9 + 3 * leg] -= 0.4 * max(s, 0)
        mujoco.mj_forward(self.m, self.d)
        self.cam.lookat[:] = [self.x, self.y, 0.2]
        self.sync(1 / 60)

    def move(self, x=None, y=None, yaw=None, speed=0.6):
        x0, y0, a0 = self.x, self.y, self.yaw
        x, y, yaw = (x0 if x is None else x), (y0 if y is None else y), (a0 if yaw is None else yaw)
        da = (yaw - a0 + math.pi) % (2 * math.pi) - math.pi
        frames = max(1, int(60 * max(math.hypot(x - x0, y - y0) / speed, abs(da) / 1.5)))
        for i in range(frames):
            s = (i + 1) / frames
            self.x, self.y, self.yaw = x0 + s * (x - x0), y0 + s * (y - y0), a0 + s * da
            self.pose(phase=i * 0.35)
        self.pose()

    def skill_walk_to_xy(self, x, y):
        if self.h < 0.2:
            return "error: sitting, call stand first"
        self.move(yaw=math.atan2(y - self.y, x - self.x))
        self.move(x, y)
        return f"at ({x:.2f}, {y:.2f})"

    def skill_walk_to(self, target):
        tx, ty = self.TARGETS[target.split("_")[0]]
        d = math.hypot(tx - self.x, ty - self.y)
        stop = max(d - 0.55, 0) / d if d else 0  # stop in front of the post, not inside it
        return self.skill_walk_to_xy(self.x + stop * (tx - self.x), self.y + stop * (ty - self.y)) + f", next to {target}"

    def skill_turn(self, degrees):
        self.move(yaw=self.yaw + math.radians(degrees))
        return f"facing {math.degrees(self.yaw) % 360:.0f} degrees"

    def _blend(self, legs, h):
        l0, h0 = self.legs.copy(), self.h
        for i in range(40):
            s = (i + 1) / 40
            self.legs, self.h = l0 + s * (np.array(legs) - l0), h0 + s * (h - h0)
            self.pose()

    def skill_sit(self):
        self._blend(self.SIT, 0.18)
        return "sitting"

    def skill_stand(self):
        self._blend(self.STAND, 0.27)
        return "standing"

    def state(self):
        return f"robot at ({self.x:.2f}, {self.y:.2f}) facing {math.degrees(self.yaw) % 360:.0f} degrees, " \
               f"{'sitting' if self.h < 0.2 else 'standing'}; markers: " + \
               ", ".join(f"{c}_marker at {xy}" for c, xy in self.TARGETS.items())


# ---------------------------------------------------------------- the model loop
SYSTEM = """You control a robot in a MuJoCo simulation. Each turn you get a camera image and the robot state.
Objects: {objects}. Coordinates are metres, x forward from the robot base, y to the left, z up.
Use the tools to carry out the user's command one step at a time, look at the new image after each step,
and call done when the command is finished or impossible."""


def data_url(png):
    return "data:image/png;base64," + base64.b64encode(png).decode()


def ask_model(robot, command, client, model_name, max_turns=10):
    def observe(text):
        t = time.time()
        png = robot.image_png()
        with open(os.path.join(HERE, "last_view.png"), "wb") as f:  # exactly what the model sees this turn
            f.write(png)
        timing["image"] += time.time() - t
        return {"role": "user", "content": [{"type": "text", "text": f"{text}\nState: {robot.state()}"},
                                            {"type": "image_url", "image_url": {"url": data_url(png)}}]}

    timing = {"think": 0.0, "move": 0.0, "image": 0.0}
    msgs = [{"role": "system", "content": SYSTEM.format(objects=robot.OBJECTS)}, observe(f"Command: {command}")]
    extra = {"reasoning_effort": os.environ["VLA_REASONING_EFFORT"]} if os.environ.get("VLA_REASONING_EFFORT") else {}
    try:
        for turn in range(max_turns):
            print("  thinking...", end="", flush=True)
            t = time.time()
            resp = client.chat.completions.create(model=model_name, messages=msgs, tools=robot.skills, **extra)
            dt = time.time() - t
            timing["think"] += dt
            u = resp.usage
            print(f" {dt:.1f}s (sent {u.prompt_tokens} tokens, got {u.completion_tokens})" if u else f" {dt:.1f}s")
            reply = resp.choices[0].message
            if reply.content:
                print(f"  model: {reply.content.strip()}")
            if not reply.tool_calls:
                return
            msgs.append(reply.model_dump(exclude_none=True))
            for call in reply.tool_calls:
                args = json.loads(call.function.arguments or "{}")
                if call.function.name == "done":
                    print(f"  done: {args.get('summary', '')}")
                    return
                print(f"  -> {call.function.name}({args})")
                t = time.time()
                result = robot.run(call.function.name, args)
                timing["move"] += time.time() - t
                print(f"     {result}")
                msgs.append({"role": "tool", "tool_call_id": call.id, "content": result})
            msgs.append(observe("After those actions."))  # images go in a user turn; not every provider takes them in tool results
        print("  stopped: hit max_turns")
    finally:
        total = sum(timing.values())
        print("  time: " + ", ".join(f"{k} {v:.1f}s ({v / total:.0%})" for k, v in timing.items() if total))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--robot", choices=["panda", "go2"], default="panda")
    p.add_argument("--joints", action="store_true", help="panda only: the model sets raw joint angles instead of calling skills")
    p.add_argument("--manual", action="store_true", help="drive the joints yourself with the viewer's Control sliders")
    args = p.parse_args()
    cls = {"panda": Panda, "go2": Go2}[args.robot]
    if args.joints and cls is not Panda:
        p.error("--joints only works with --robot panda")
    if args.manual:  # the full viewer steps physics itself, so the sliders move the robot
        m, d = cls.make()
        cls(m, d)  # start from the robot's rest pose
        mujoco.viewer.launch(m, d)
        return
    load_dotenv(os.path.join(HERE, ".env"), override=True)  # .env wins over stale shell exports
    from openai import OpenAI
    client = OpenAI(base_url=os.environ["VLA_BASE_URL"], api_key=os.environ["VLA_API_KEY"], timeout=120, max_retries=1)
    model_name = os.environ["VLA_MODEL"]

    m, d = cls.make()
    with mujoco.viewer.launch_passive(m, d) as viewer:
        robot = cls(m, d, viewer)
        if args.joints:
            robot.skills = Panda.JOINT_SKILLS
        print(f"{args.robot} ready, model {model_name}. Type a command (Ctrl-D to quit).")
        while viewer.is_running():
            try:
                command = input("> ").strip()
            except EOFError:
                break
            if command:
                try:
                    ask_model(robot, command, client, model_name)
                except Exception as e:  # a bad model name or network blip shouldn't close the sim
                    print(f"  error: {e}")


if __name__ == "__main__":
    main()
