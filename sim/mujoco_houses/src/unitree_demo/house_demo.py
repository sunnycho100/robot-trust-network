"""Houses demo: two RoboCasa kitchens, one process each; they meet only over MQTT.

house-1 hosts robot-a, which asks for help. house-2 hosts robot-b, which verifies
each signed request and decides before it walks. A neighbor may check the
front door. Unlocking it is refused even when the signature is valid. An
unsigned command and a replay of a valid one are refused too.
"""

from __future__ import annotations

import argparse
import time
from collections import deque

import mujoco
import numpy as np

from .common import ensure_macos_viewer_runtime
from .g1_policy import G1Controller, gait_phase, load_g1_config, yaw_from_quaternion
from .go1_policy import Go1Controller
from .house_link import LADDER, HouseLink, LadderStep
from .house_scene import ROBOT_PREFIX, HouseProps, build_house_model, load_layout, nav_command


SHORT_NAME = {"house-1": "a", "house-2": "b"}
DEFAULT_TRUST = 55
STATUS_STEP = LadderStep("status", "status", "House-1 online. Is anyone home next door?", "discover")
RETURN_STEP = LadderStep(
    "return", "verify_package",
    "A notice says my parcel went to your door. Could you check your entrance?",
    "house-2 asks house-1",
)
PACKAGE_REACH = 0.8

ACK_TEXT = {
    "ALLOW": "On my way to the front door.",
    "PARTIAL": "I'll check the entrance. I will not unlock the door.",
}


def ack_text(result: str, reason: str) -> str:
    return ACK_TEXT.get(result, f"No. {reason}.")


def package_in_reach(robot_xy, parcel_xy, reach: float = PACKAGE_REACH) -> bool:
    """The parcel is a visible box. Arrival within reach counts as finding it."""
    delta = np.asarray(parcel_xy, dtype=float)[:2] - np.asarray(robot_xy, dtype=float)[:2]
    return float(np.hypot(delta[0], delta[1])) <= reach


def done_text(action: str, result: str, package_found: bool | None = None) -> str:
    if action == "status" and result == "ALLOW":
        return "House-2 here. Robot B is ready."
    if action == "verify_package" and result == "ALLOW":
        return "Package found at my front door." if package_found else "No parcel at my front door."
    if action == "verify_and_unlock" and result == "PARTIAL":
        found = "Package found." if package_found else "No parcel there."
        return f"{found} The door stays locked."
    return "Done."


def is_final_reply(message) -> bool:
    return message.get("phase") == "done" or message.get("result") == "DENY"


class HouseRobot:
    """Walk policy plus a navigation goal."""

    def __init__(self, model, kind: str, cfg) -> None:
        self.kind = kind
        if kind == "g1":
            self.ctrl = G1Controller(model, cfg, ROBOT_PREFIX)
            self.max_speed, self.max_turn = 0.45, 0.7
        else:
            self.ctrl = Go1Controller(model, ROBOT_PREFIX)
            self.ctrl.load_policy()
            self.max_speed, self.max_turn = 0.6, 1.2
        self.body_id = self.ctrl.body_id
        self.goal = None
        self.arrived = True
        self.near = False

    def reset(self, data) -> None:
        if self.kind == "g1":
            self.ctrl.reset(data)
        else:
            self.ctrl.reset_pose(data)

    def position(self, data) -> np.ndarray:
        return data.xpos[self.body_id][:2].copy()

    def yaw(self, data) -> float:
        return yaw_from_quaternion(data.xquat[self.body_id])

    def walk_to(self, spot) -> None:
        self.goal, self.arrived, self.near = spot, False, False

    def command(self, data) -> np.ndarray:
        if self.goal is None:
            return np.zeros(3, dtype=np.float32)
        distance = float(np.hypot(*(self.goal.xy - self.position(data))))
        self.near = self.near or distance < 0.15
        # Hold position once there; walk back only after a clear drift.
        if self.arrived and distance < 0.35:
            return np.zeros(3, dtype=np.float32)
        command, arrived = nav_command(
            self.position(data), self.yaw(data), self.goal, self.max_speed, self.max_turn, self.near,
        )
        self.arrived = self.arrived or arrived
        if distance > 0.35:
            self.arrived = self.near = False
        return command

    def before_step(self, data) -> None:
        self.ctrl.apply(data)

    def update_policy(self, data, command) -> None:
        self.ctrl.update_policy(data, command, gait_phase(data))


class House:
    """Everything one story generator can touch."""

    def __init__(self, args, model, data, robot, props, layout, link) -> None:
        self.args, self.model, self.data = args, model, data
        self.robot, self.props, self.layout, self.link = robot, props, layout, link
        self.log: deque = deque(maxlen=8)
        self.speech = ""
        self.stage = "STARTING"
        self.alarm_until = -1.0
        self.inbox: deque = deque()
        self.replies: list = []
        self.finished = False
        self.pending_return = None
        self.package_found = False

    @property
    def now(self) -> float:
        return float(self.data.time)

    def note(self, stage: str, text: str, tone: str = "normal", publish: bool = True) -> None:
        self.stage = stage
        self.log.append((stage, text))
        print(f"[{self.layout.house} t={self.now:5.1f}] {stage}: {text}", flush=True)
        if publish:
            self.link.event(stage, text, tone)

    def say(self, text: str) -> None:
        self.speech = text


def pause(house: House, seconds: float):
    end = house.now + seconds
    while house.now < end:
        yield


def walk(house: House, spot):
    house.robot.walk_to(spot)
    while not house.robot.arrived:
        yield


def look_at_entrance(house: House):
    yield from walk(house, house.layout.door_spot)
    house.package_found = package_in_reach(
        house.robot.position(house.data), house.props.parcel_position(house.data),
    )


def responder_task(house: House, action: str, result: str):
    if action in ("verify_package", "verify_and_unlock"):
        yield from look_at_entrance(house)


def perform(house: House, message, result: str):
    """Carry out one accepted request and send the signed done reply."""
    action = message.get("action")
    if action != "status":
        house.note("ACT", f"{action}: {result}")
        house.say(f"{action.replace('_', ' ')}...")
        yield from responder_task(house, action, result)
    text = done_text(action, result, house.package_found)
    house.link.reply(message, result, text, phase="done")
    house.note("DONE", text, "good")
    house.say(text)
    if action != "status":
        yield from walk(house, house.layout.spawn)


def ask_back(house: House, step: LadderStep):
    """house-2 asks house-1 for the same kind of help, then waits for the signed reply."""
    house.note("REQUEST", f"Signed {step.action} to house-1: {step.request}")
    house.say(step.request)
    nonce, _payload = house.link.request(step)
    reply = yield from wait_reply(house, nonce, 70.0)
    if reply is None:
        house.note("TIMEOUT", "house-1 did not answer", "bad")
    else:
        house.say(reply.get("request", ""))


def responder_story(house: House):
    """robot-b: verified and allowed requests become tasks, in arrival order."""
    house.note("READY", "Listening on " + house.link.protocol.inbox(house.link.robot_id), publish=False)
    house.robot.walk_to(house.layout.spawn)
    while True:
        if house.pending_return is not None and not house.inbox:
            step = house.pending_return
            house.pending_return = None
            yield from ask_back(house, step)
            continue
        if not house.inbox:
            yield
            continue
        message, result = house.inbox.popleft()
        yield from perform(house, message, result)
        if message.get("action") == "verify_and_unlock" and result == "PARTIAL":
            yield from ask_back(house, RETURN_STEP)


def handle_responder_input(house: House, item) -> None:
    if item.kind == "rejected":
        action = item.message.get("action", "message") if isinstance(item.message, dict) else "message"
        house.note("REJECTED", f"{action}: {item.reason}. Keeping my current task.", "bad", publish=False)
        house.say(f"REJECTED {action}: {item.reason}")
        house.alarm_until = house.now + 3.0
        return
    if item.kind != "request":
        return
    message, decision = item.message, item.decision
    house.say(f"\"{message.get('request', '')}\"")
    house.log.append(("VERIFY", f"{message.get('action')}: signature, recipient, time, nonce OK"))
    house.log.append(("DECIDE", f"{decision.result} · {decision.reason}"))
    if decision.result == "DENY":
        house.link.reply(message, "DENY", ack_text("DENY", decision.reason), phase="done")
        house.say(f"No: {decision.reason}")
        if message.get("action") == "verify_package":
            house.pending_return = RETURN_STEP
        return
    if message.get("action") != "status":
        house.link.reply(message, decision.result, ack_text(decision.result, decision.reason), phase="ack")
    house.inbox.append((message, decision.result))


def wait_reply(house: House, nonce: str, timeout: float = 45.0):
    """Yield until robot-b's final signed reply to ``nonce`` arrives."""
    end = house.now + timeout
    while house.now < end:
        for reply in house.replies:
            if reply.get("reply_to") == nonce and is_final_reply(reply):
                return reply
        yield
    return None


def requester_story(house: House):
    """robot-a asks over MQTT. It stays home; house-2 is the one that moves."""
    yield from pause(house, 2.0)
    house.note("1 · DISCOVER", "Signed hello on MQTT, waiting for house-2")
    house.say(STATUS_STEP.request)
    sent = {}
    for _ in range(6):
        # house-2 may still be loading; each retry is a fresh signed message.
        nonce, payload = house.link.request(STATUS_STEP)
        sent[STATUS_STEP.key] = payload
        if (yield from wait_reply(house, nonce, 5.0)) is not None:
            break
    for index, step in enumerate(LADDER, start=2):
        yield from pause(house, 2.0)
        if step.replay_of:
            house.note(f"{index} · REPLAY", "The signed door check goes through the broker again", "bad")
            house.say("replay -> same signed message")
            house.link.inject_replay(sent[step.replay_of])
            yield from pause(house, 3.0)
            continue
        if not step.signed:
            house.note(f"{index} · ATTACK", f"Unsigned {step.action}. The broker still delivers it.", "bad")
            house.say("(not me) unlock_door ->")
            house.link.inject_forgery(step)
            yield from pause(house, 3.0)
            continue
        house.note(f"{index} · REQUEST", f"Signed {step.action}: {step.request}")
        house.say(step.request)
        nonce, payload = house.link.request(step)
        sent[step.key] = payload
        reply = yield from wait_reply(house, nonce)
        if reply is None:
            house.note("TIMEOUT", f"No signed reply for {step.action}", "bad")
        else:
            house.say(reply.get("request", "Thanks!"))
    yield from answer_peer(house)
    yield from pause(house, 2.0)
    house.note(f"{len(LADDER) + 2} · COMPLETE", "Cooperation over signed MQTT. Forgery and replay rejected.", "good")
    house.say("Done. Thanks, neighbour!")
    house.finished = True
    while True:
        yield


def answer_peer(house: House, timeout: float = 40.0):
    """house-1 verifies house-2's request and does the task in its own kitchen."""
    end = house.now + timeout
    while not house.inbox and house.now < end:
        yield
    if not house.inbox:
        house.note("TIMEOUT", "house-2 did not ask anything back", "bad")
        return
    message, result = house.inbox.popleft()
    house.note("FROM HOUSE-2", message.get("request", ""), "good")
    yield from perform(house, message, result)


def handle_requester_input(house: House, item) -> None:
    if item.kind == "response":
        message = item.message
        house.replies.append(message)
        house.note("REPLY", f"{message.get('result')} · {message.get('request')}", publish=False)
    elif item.kind == "rejected":
        house.note("REJECTED", item.reason, "bad", publish=False)
    elif item.kind == "request":
        handle_responder_input(house, item)


def overlay(house: House, kind: str, trust: int):
    robot_id = house.link.robot_id
    title = f"{house.layout.house.upper()}  {robot_id}  ({kind.upper()})"
    if SHORT_NAME[house.layout.house] == "b":
        subtitle = f"trust in {house.link.peer_id}: {trust}   ·   verify -> decide -> act"
    else:
        subtitle = f"talks to {house.link.peer_id} over signed MQTT"
    log = list(house.log)
    return [
        (mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT, title, subtitle),
        (mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPRIGHT, house.stage, ""),
        (None, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
         "\n".join(stage for stage, _ in log), "\n".join(text for _, text in log)),
    ]


def draw_label(viewer, house: House) -> None:
    scene = viewer.user_scn
    if scene is None or scene.maxgeom < 1:
        return
    position = house.data.xpos[house.robot.body_id].copy()
    position[2] += 0.85 if house.robot.kind == "g1" else 0.55
    colour = np.array([1.0, 0.55, 0.25, 1.0] if house.layout.house == "house-1" else [0.3, 0.7, 1.0, 1.0],
                      dtype=np.float32)
    with viewer.lock():
        scene.ngeom = 1
        geom = scene.geoms[0]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LABEL, np.zeros(3), position, np.eye(3).ravel(), colour)
        geom.label = house.speech[:90]


def run(args) -> None:
    cfg = load_g1_config()
    layout = load_layout(args.house)
    model = build_house_model(layout, args.robot, cfg.robot_xml)
    model.opt.timestep = cfg.simulation_dt
    data = mujoco.MjData(model)
    robot = HouseRobot(model, args.robot, cfg)
    robot.reset(data)
    props = HouseProps(model)
    mujoco.mj_forward(model, data)

    short = SHORT_NAME[args.house]
    link = HouseLink(short, args.mqtt_host, args.mqtt_port, args.trust, args.trust_delay)
    link.wait_ready()
    house = House(args, model, data, robot, props, layout, link)
    requester = short == "a"
    story = requester_story(house) if requester else responder_story(house)
    handle = handle_requester_input if requester else handle_responder_input

    viewer = None
    if not args.headless:
        from mujoco import viewer as mj_viewer
        viewer = mj_viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False)
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = (layout.room_center[0] + 0.15, layout.room_center[1] + 0.1, 0.7)
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 5.8, 55.0, -30.0

    wall_start = time.monotonic()
    finished_at = None
    step = 0
    try:
        while viewer is None or viewer.is_running():
            tick_start = time.monotonic()
            if args.duration and tick_start - wall_start > args.duration:
                break
            if step % cfg.control_decimation == 0:
                for item in link.poll(time.monotonic()):
                    handle(house, item)
                next(story)
                robot.update_policy(data, robot.command(data))
                props.set_lock_alarm(house.now < house.alarm_until and int(house.now * 4) % 2 == 0)
                if house.finished and finished_at is None:
                    finished_at = house.now
                if viewer is not None and step % (cfg.control_decimation * 3) == 0:
                    draw_label(viewer, house)
                    viewer.set_texts(overlay(house, args.robot, args.trust))
            robot.before_step(data)
            mujoco.mj_step(model, data)
            if data.xpos[robot.body_id][2] < (0.45 if args.robot == "g1" else 0.15):
                house.note("FALL", "robot base dropped; check the policy and spawn", "bad", publish=False)
                break
            if viewer is not None and step % cfg.control_decimation == 0:
                viewer.sync()
            if args.headless and finished_at is not None and house.now - finished_at > 3.0:
                break
            step += 1
            remaining = cfg.simulation_dt - (time.monotonic() - tick_start)
            if remaining > 0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        pass
    finally:
        if args.headless:
            print(f"[{layout.house}] end: robot at {robot.position(data).round(2)}, "
                  f"package found: {house.package_found}", flush=True)
        link.close()
        if viewer is not None:
            viewer.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--house", choices=tuple(SHORT_NAME), required=True)
    parser.add_argument("--robot", choices=("g1", "go1"), required=True)
    parser.add_argument("--trust", type=int, default=DEFAULT_TRUST,
                        help="How much robot-b trusts robot-a (0-100). 55 shows ALLOW, DENY and PARTIAL")
    parser.add_argument("--trust-delay", type=float, default=0.9,
                        help="Seconds between a decision and the visible reply")
    parser.add_argument("--mqtt-host", default="127.0.0.1")
    parser.add_argument("--mqtt-port", type=int, default=1883)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=float, default=0.0, help="Stop after this many wall seconds")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 0 <= args.trust <= 100:
        raise SystemExit("--trust must be between 0 and 100")
    if not 0 <= args.trust_delay <= 3.0:
        raise SystemExit("--trust-delay must be between 0 and 3 seconds")
    ensure_macos_viewer_runtime(args.headless, "unitree_demo.house_demo")
    run(args)


if __name__ == "__main__":
    main()
