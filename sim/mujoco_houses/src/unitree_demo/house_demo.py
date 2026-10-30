"""Houses demo: two RoboCasa kitchens, one process each; they meet only over MQTT.

house-1 hosts robot-a, which asks for help. house-2 hosts robot-b, which verifies
each signed request and decides before it walks. A neighbor may check the
front door. Unlocking it is refused even when the signature is valid. An
unsigned command and a replay of a valid one are refused too.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path
from uuid import uuid4

import mujoco
import numpy as np

from .common import ensure_macos_viewer_runtime, viewer_label
from .g1_policy import G1Controller, gait_phase, load_g1_config, yaw_from_quaternion
from .go1_policy import Go1Controller
from .house_link import LADDER, HouseLink, LadderStep
from .house_scene import ROBOT_PREFIX, HouseProps, build_house_model, load_layout, nav_command
from .trust_session import action_label


SHORT_NAME = {"house-1": "a", "house-2": "b"}
DEFAULT_TRUST = 55
STATUS_STEP = LadderStep("status", "status", "House-1 online. Is anyone home next door?", "discover")
RETURN_STEP = LadderStep(
    "return", "verify_package",
    "A notice says my parcel went to your door. Could you check your entrance?",
    "house-2 asks house-1",
)
PACKAGE_REACH = 0.8
EXECUTABLE = {"status": "ALLOW", "verify_package": "ALLOW", "verify_and_unlock": "PARTIAL"}

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
        return "Robot is ready."
    if action == "verify_package" and result == "ALLOW":
        return "Package found at my front door." if package_found else "No parcel at my front door."
    if action == "verify_and_unlock" and result == "PARTIAL":
        found = "Package found." if package_found else "No parcel there."
        return f"{found} The door stays locked."
    raise ValueError(f"No executor for {action}: {result}")


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
        self.outstanding: dict[str, str] = {}
        self.replies: dict[str, dict] = {}
        self.events: deque = deque(maxlen=256)
        self.run_id = uuid4().hex
        self.event_log = getattr(args, "event_log", None)
        self.failures: list[str] = []
        self.outcome = None
        self.finished = False
        self.pending_return = None
        self.package_found = False
        self.return_request = None
        self.return_closed = False
        self.return_deadline = None
        self.active_request = None

    @property
    def now(self) -> float:
        return float(self.data.time)

    def note(self, stage: str, text: str, tone: str = "normal", publish: bool = True) -> None:
        self.stage = stage
        self.log.append((stage, text))
        self.record("note", stage=stage, text=text, tone=tone)
        line = f"[{self.layout.house} t={self.now:5.1f}] {stage}: {text}"
        encoding = sys.stdout.encoding or "utf-8"
        print(line.encode(encoding, errors="backslashreplace").decode(encoding), flush=True)
        if publish:
            self.link.event(stage, text, tone)

    def say(self, text: str) -> None:
        self.speech = text

    def record(self, event: str, **fields) -> None:
        record = {"run_id": self.run_id, "house": self.layout.house, "sim_time": self.now,
                  "wall_time": time.time(), "monotonic_time": time.monotonic(),
                  "event": event, **fields}
        self.events.append(record)
        if self.event_log is not None:
            with self.event_log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    def request(self, step: LadderStep):
        nonce, payload = self.link.request(step)
        self.outstanding[nonce] = step.action
        self.record("request_submitted", nonce=nonce, action=step.action, peer=self.link.peer_id)
        return nonce, payload

    def fail(self, reason: str) -> None:
        self.failures.append(reason)
        self.outcome = "FAILED"
        self.note("FAILED", reason, "bad")

    def exit_status(self, requester: bool) -> int:
        return int(bool(self.failures) or self.has_pending_work
                   or (requester and self.outcome != "COMPLETE"))

    @property
    def has_pending_work(self) -> bool:
        return bool(self.outstanding or self.inbox or self.active_request is not None
                    or self.pending_return is not None or self.link.inbox.deciding)


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
    if EXECUTABLE.get(action) != result:
        refuse_execution(house, message, "unsupported action or permission")
        return False
    house.active_request = message["nonce"]
    house.record("execution_started", nonce=message["nonce"], action=action, result=result)
    if action != "status":
        house.note("ACT", f"{action}: {result}")
        house.say(f"{action.replace('_', ' ')}...")
        yield from responder_task(house, action, result)
    text = done_text(action, result, house.package_found)
    house.link.reply(message, result, text, phase="done")
    house.record("execution_done", nonce=message["nonce"], action=action, result=result,
                 evidence="simulator_ground_truth", package_found=house.package_found if action != "status" else None)
    house.note("DONE", text, "good")
    house.say(text)
    if action != "status":
        yield from walk(house, house.layout.spawn)
    house.active_request = None
    return True


def refuse_execution(house: House, message, reason: str) -> None:
    house.link.reply(message, "DENY", f"No. {reason}.", phase="done")
    house.record("execution_denied", nonce=message["nonce"], action=message.get("action"), reason=reason)
    house.note("DENY", f"{message.get('action')}: {reason}", "bad")
    house.say(f"No: {reason}")


def ask_back(house: House, step: LadderStep):
    """house-2 asks house-1 for the same kind of help, then waits for the signed reply."""
    house.note("REQUEST", f"Signed {step.action} to house-1: {step.request}")
    house.say(step.request)
    nonce, _payload = house.request(step)
    reply = yield from wait_reply(house, nonce, 70.0)
    if reply is None:
        house.note("TIMEOUT", "house-1 did not answer", "bad")
        house.fail("Return request timed out")
    elif reply["result"] != EXECUTABLE[step.action]:
        house.fail(f"Return request received {reply['result']}")
    else:
        house.say(reply.get("request", ""))
        house.note("RETURN CONFIRMED", "House-1 reported the return task complete")


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
    if item.kind == "response":
        handle_reply(house, item.message)
        return
    if item.kind == "rejected":
        action = action_label(item.message)
        house.note("REJECTED", f"{action}: {item.reason}. Keeping my current task.", "bad", publish=False)
        house.say(f"REJECTED {action}: {item.reason}")
        house.alarm_until = house.now + 3.0
        house.record("authentication_rejected", reason=item.reason)
        return
    if item.kind != "request":
        return
    message, decision = item.message, item.decision
    house.say(f"\"{message.get('request', '')}\"")
    house.log.append(("VERIFY", f"{message.get('action')}: signature, recipient, time, nonce OK"))
    house.log.append(("DECIDE", f"{decision.result} · {decision.reason}"))
    house.record("policy_decision", nonce=message["nonce"], action=message.get("action"),
                 peer=message["from"], result=decision.result, reason=decision.reason)
    if decision.result == "DENY":
        refuse_execution(house, message, decision.reason)
        if message.get("action") == "verify_package" and house.layout.house == "house-2":
            house.pending_return = RETURN_STEP
        return
    if EXECUTABLE.get(message.get("action")) != decision.result:
        refuse_execution(house, message, "unsupported action or permission")
        return
    if message.get("action") != "status":
        house.link.reply(message, decision.result, ack_text(decision.result, decision.reason), phase="ack")
    house.inbox.append((message, decision.result))


def wait_reply(house: House, nonce: str, timeout: float = 45.0):
    """Yield until robot-b's final signed reply to ``nonce`` arrives."""
    end = house.now + timeout
    try:
        while house.now < end:
            reply = house.replies.pop(nonce, None)
            if reply is not None:
                house.record("request_outcome", nonce=nonce, outcome=reply["result"], evidence="peer_report")
                return reply
            yield
        house.record("request_outcome", nonce=nonce, outcome="TIMEOUT", evidence="local")
        return None
    finally:
        house.outstanding.pop(nonce, None)
        house.replies.pop(nonce, None)


def handle_reply(house: House, message) -> None:
    """Only an expected peer's matching final reply can finish an outstanding request."""
    nonce = message.get("reply_to")
    expected = house.outstanding.get(nonce)
    if (expected is None or message.get("from") != house.link.peer_id
            or message.get("action") != expected
            or message.get("phase") not in ("ack", "done")
            or message.get("result") not in ("ALLOW", "PARTIAL", "DENY")
            or nonce in house.replies):
        house.record("reply_ignored", reply_to=nonce, reason="unexpected, invalid or duplicate reply")
        return
    if is_final_reply(message):
        house.replies[nonce] = message
    house.record("reply_received", reply_to=nonce, action=expected, result=message["result"],
                 phase=message["phase"], evidence="peer_report")
    house.note("REPLY", f"{message['result']} · {message.get('request', '')}", publish=False)


def finish_story(house: House):
    while house.inbox:
        message, _ = house.inbox.popleft()
        refuse_execution(house, message, "story ended before this request could execute")
    house.outcome = "FAILED" if house.failures else "COMPLETE"
    text = ("Story incomplete: " + "; ".join(house.failures) if house.failures else
            "Neighbor reported the requested work complete; local return task completed. "
            "Attack outcomes must be checked in the receiving house's log.")
    house.note(house.outcome, text, "bad" if house.failures else "good")
    house.say("Story incomplete." if house.failures else "Cooperation tasks complete.")
    house.finished = True
    while True:
        yield


def requester_story(house: House):
    """robot-a asks over MQTT. It stays home; house-2 is the one that moves."""
    yield from pause(house, 2.0)
    house.note("1 · DISCOVER", "Signed hello on MQTT, waiting for house-2")
    house.say(STATUS_STEP.request)
    sent = {}
    for _ in range(6):
        # house-2 may still be loading; each retry is a fresh signed message.
        nonce, payload = house.request(STATUS_STEP)
        sent[STATUS_STEP.key] = payload
        reply = yield from wait_reply(house, nonce, 5.0)
        if reply is not None and reply["result"] == "ALLOW":
            break
    else:
        house.fail("No successful discovery reply from house-2")
        yield from finish_story(house)
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
        nonce, payload = house.request(step)
        sent[step.key] = payload
        reply = yield from wait_reply(house, nonce)
        if reply is None:
            house.note("TIMEOUT", f"No signed reply for {step.action}", "bad")
            house.fail(f"{step.action} timed out")
        elif reply["result"] != EXECUTABLE[step.action]:
            house.fail(f"{step.action} received {reply['result']}")
        else:
            house.say(reply.get("request", "Thanks!"))
    if not (yield from answer_peer(house)):
        house.fail("Return task did not complete")
    yield from pause(house, 2.0)
    yield from finish_story(house)


def answer_peer(house: House, timeout: float = 40.0):
    """house-1 verifies house-2's request and does the task in its own kitchen."""
    end = house.now + timeout
    house.return_deadline = end
    while house.now < end:
        for message, result in house.inbox:
            if message.get("from") == house.link.peer_id and message.get("action") == RETURN_STEP.action:
                house.inbox.remove((message, result))
                house.note("FROM HOUSE-2", message.get("request", ""), "good")
                return (yield from perform(house, message, result))
        yield
    house.return_closed = True
    house.note("TIMEOUT", "house-2 did not ask for the return inspection", "bad")
    return False


def handle_requester_input(house: House, item) -> None:
    if item.kind == "request":
        message = item.message
        if (house.finished or house.return_closed or house.return_request is not None
                or (house.return_deadline is not None and house.now >= house.return_deadline)
                or message.get("from") != house.link.peer_id
                or message.get("action") != RETURN_STEP.action):
            house.record("policy_decision", nonce=message["nonce"], action=message.get("action"),
                         peer=message["from"], result=item.decision.result, reason=item.decision.reason)
            refuse_execution(house, message, "requester is not accepting this task")
            return
    handle_responder_input(house, item)
    if item.kind == "request" and house.inbox:
        house.return_request = house.inbox[-1][0]["nonce"]


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
        geom.label = viewer_label(house.speech)


def run(args) -> int:
    if args.event_log is not None:
        args.event_log.parent.mkdir(parents=True, exist_ok=True)
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
    exit_code = 0
    try:
        while viewer is None or viewer.is_running():
            tick_start = time.monotonic()
            if args.duration and tick_start - wall_start > args.duration:
                if (requester and not house.finished) or house.has_pending_work:
                    house.fail("Run duration expired before pending work completed")
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
                exit_code = 1
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
        exit_code = 1
    finally:
        if args.headless:
            print(f"[{layout.house}] end: robot at {robot.position(data).round(2)}, "
                  f"package found: {house.package_found}", flush=True)
        link.close()
        if viewer is not None:
            viewer.close()
    exit_code = max(exit_code, house.exit_status(requester))
    house.record("run_ended", outcome=house.outcome or "STOPPED", exit_code=exit_code)
    return exit_code


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
    parser.add_argument("--event-log", type=Path, help="Append local structured events to a JSONL file")
    parser.add_argument("--duration", type=float, default=0.0, help="Stop after this many wall seconds")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 0 <= args.trust <= 100:
        raise SystemExit("--trust must be between 0 and 100")
    if not 0 <= args.trust_delay <= 3.0:
        raise SystemExit("--trust-delay must be between 0 and 3 seconds")
    ensure_macos_viewer_runtime(args.headless, "unitree_demo.house_demo")
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
