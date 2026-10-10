"""Duo demo: a G1 leader and a G1 or Go1 follower linked only through an MQTT broker.

The leader publishes its pose and velocity intent. The follower uses its own
copy of a published walk policy and follows only after a signed request has
been verified and allowed by the trust policy.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import ssl
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from .common import HumanCommandInput, add_run_arguments, check_run_arguments, ensure_macos_viewer_runtime
from .g1_policy import G1Config, G1Controller, gait_phase, load_g1_config
from .go1_policy import Go1Controller, load_go1_spec, set_policy_gains
from .trust_session import TrustMailbox, motion_from_grant, open_pose, sign_pose


FOLLOWERS = ("g1", "go1")
# Body-frame (vx, vy, yaw rate) limits each published policy walks well inside.
COMMAND_LIMITS = {"g1": (0.5, 0.3, 0.6), "go1": (1.0, 0.5, 1.2)}
LEADER_TINT = np.array([0.95, 0.35, 0.22])
FOLLOWER_TINT = np.array([0.20, 0.55, 1.00])


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


@dataclass(frozen=True)
class LeaderMessage:
    sequence: int
    sent_at_ns: int
    position_xy: np.ndarray
    yaw: float
    command: np.ndarray

    def to_payload(self) -> bytes:
        return json.dumps(self.to_state(), separators=(",", ":")).encode("utf-8")

    def to_state(self) -> dict:
        return {
            "schema": "unitree.formation.v1",
            "robot_id": "g1-leader",
            "seq": self.sequence,
            "sent_at_ns": self.sent_at_ns,
            "pose": {"x": float(self.position_xy[0]), "y": float(self.position_xy[1]), "yaw": float(self.yaw)},
            "command": {"vx": float(self.command[0]), "vy": float(self.command[1]), "yaw_rate": float(self.command[2])},
        }

    @classmethod
    def from_payload(cls, payload: bytes) -> "LeaderMessage":
        if len(payload) > 4096:
            raise ValueError("MQTT formation payload is too large")
        return cls.from_state(json.loads(payload))

    @classmethod
    def from_state(cls, raw: dict) -> "LeaderMessage":
        if raw.get("schema") != "unitree.formation.v1" or raw.get("robot_id") != "g1-leader":
            raise ValueError("Unexpected MQTT formation schema or robot_id")
        pose, command = raw["pose"], raw["command"]
        values = np.asarray(
            [pose["x"], pose["y"], pose["yaw"], command["vx"], command["vy"], command["yaw_rate"]],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("MQTT formation payload contains non-finite values")
        return cls(
            sequence=int(raw["seq"]),
            sent_at_ns=int(raw["sent_at_ns"]),
            position_xy=values[:2].astype(np.float32),
            yaw=float(values[2]),
            command=values[3:].astype(np.float32),
        )


@dataclass(frozen=True)
class StoryBeat:
    name: str
    leader_command: np.ndarray
    leader_text: str
    follower_text: str
    action: str | None = None
    attack: bool = False


STORY_TITLE = "STORY  team up · block the forged command"


def story_beat(elapsed: float) -> StoryBeat:
    """Signed imitation, a forged door command, then a shared carry."""
    if elapsed < 2.0:
        return StoryBeat("1 · DISCOVER", np.zeros(3, dtype=np.float32),
                         "I found another robot", "Exploring on my own")
    if elapsed < 5.0:
        return StoryBeat("2 · REQUEST", np.array([0.28, 0.0, 0.0], dtype=np.float32),
                         "Copy my route — signed", "Checking signature and trust",
                         action="mirror_motion")
    if elapsed < 9.0:
        return StoryBeat("3 · IMITATE", np.array([0.22, 0.0, 0.28], dtype=np.float32),
                         "Copy this turn", "Matching the verified motion",
                         action="mirror_motion")
    if elapsed < 11.0:
        return StoryBeat("4 · ATTACK", np.array([0.24, 0.0, 0.18], dtype=np.float32),
                         "That door command is not mine", "Rejected unlock_door — still following",
                         attack=True)
    if elapsed < 16.0:
        yaw = 0.22 if int((elapsed - 11.0) // 2.0) % 2 == 0 else -0.22
        return StoryBeat("5 · COOPERATE", np.array([0.26, 0.0, yaw], dtype=np.float32),
                         "Take the other side", "Holding my side of the load",
                         action="carry_together")
    return StoryBeat("6 · COMPLETE", np.zeros(3, dtype=np.float32),
                     "Task complete", "Signed cooperation finished")


def select_delayed_message(history, now: float, delay: float, max_age: float):
    """Select the newest packet old enough to make communication visually legible."""
    if not history or now - history[-1][1] > max_age:
        return None, math.inf
    cutoff = now - delay
    for message, received_at in reversed(history):
        if received_at <= cutoff:
            return message, now - received_at
    return None, math.inf


class MqttFormationLink:
    """Leader, follower and attacker as distinct MQTT clients linked only through the broker."""

    def __init__(self, args) -> None:
        try:
            import paho.mqtt.client as mqtt
        except ImportError as exc:
            raise RuntimeError("MQTT support is missing. Run: python -m pip install -r requirements.txt") from exc

        self.mqtt = mqtt
        self.topic = f"unitree/formations/{args.formation_id}/leader/state"
        self.status_prefix = f"unitree/formations/{args.formation_id}/status"
        self.trust = TrustMailbox(args.trust, max(args.trust_delay, args.follower_start_delay))
        self._lock = threading.Lock()
        self._history = deque(maxlen=512)
        self._last_sequence = -1
        self.pose_rejection: str | None = None
        self._leader_ready = threading.Event()
        self._follower_ready = threading.Event()
        self._follower_subscribed = threading.Event()
        self._error: str | None = None

        self.leader = self._new_client(
            "g1-leader", args, args.mqtt_leader_cert, args.mqtt_leader_key,
            os.environ.get("G1_LEADER_MQTT_USERNAME"), os.environ.get("G1_LEADER_MQTT_PASSWORD"),
        )
        self.follower = self._new_client(
            "follower", args, args.mqtt_follower_cert, args.mqtt_follower_key,
            os.environ.get("FOLLOWER_MQTT_USERNAME"), os.environ.get("FOLLOWER_MQTT_PASSWORD"),
        )
        self.attacker = self._new_client("attacker", args, None, None, None, None)
        self.leader.on_connect = self._on_leader_connect
        self.leader.on_message = self._on_leader_message
        self.follower.on_connect = self._on_follower_connect
        self.follower.on_subscribe = self._on_follower_subscribe
        self.follower.on_message = self._on_message
        self.leader.on_disconnect = self._on_disconnect
        self.follower.on_disconnect = self._on_disconnect

        for client in (self.follower, self.leader, self.attacker):
            client.connect_async(args.mqtt_host, args.mqtt_port, keepalive=30)
            client.loop_start()
        for event, what in ((self._follower_ready, "Follower could not connect to"),
                            (self._follower_subscribed, "Follower could not subscribe on"),
                            (self._leader_ready, "Leader could not connect to")):
            if not event.wait(args.mqtt_connect_timeout):
                self.close()
                raise RuntimeError(self._error or f"{what} MQTT broker {args.mqtt_host}:{args.mqtt_port}")

    def _new_client(self, role, args, cert, key, username, password):
        mqtt = self.mqtt
        client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id=f"{role}-{os.getpid()}",
            protocol=mqtt.MQTTv5,
        )
        client.reconnect_delay_set(min_delay=1, max_delay=10)
        client.will_set(f"{self.status_prefix}/{role}", "offline", qos=1, retain=True)
        if username:
            client.username_pw_set(username, password)
        if args.mqtt_tls:
            context = ssl.create_default_context(cafile=str(args.mqtt_ca) if args.mqtt_ca else None)
            context.minimum_version = ssl.TLSVersion.TLSv1_2
            if cert or key:
                if not cert or not key:
                    raise ValueError(f"Both certificate and key are required for {role}")
                context.load_cert_chain(str(cert), str(key))
            client.tls_set_context(context)
        return client

    def _on_leader_connect(self, client, _userdata, _flags, reason_code, _properties):
        if reason_code.is_failure:
            self._error = f"Leader MQTT connection rejected: {reason_code}"
            return
        client.publish(f"{self.status_prefix}/g1-leader", "online", qos=1, retain=True)
        client.subscribe([(self.trust.leader_inbox, 1)])
        self._leader_ready.set()

    def _on_follower_connect(self, client, _userdata, _flags, reason_code, _properties):
        if reason_code.is_failure:
            self._error = f"Follower MQTT connection rejected: {reason_code}"
            return
        client.subscribe([(self.topic, 1), (self.trust.follower_inbox, 1)])
        client.publish(f"{self.status_prefix}/follower", "online", qos=1, retain=True)
        self._follower_ready.set()

    def _on_follower_subscribe(self, _client, _userdata, _mid, reason_codes, _properties):
        if any(code.is_failure for code in reason_codes):
            self._error = f"Follower MQTT subscription rejected: {reason_codes}"
            return
        self._follower_subscribed.set()

    def _on_disconnect(self, _client, _userdata, _flags, reason_code, _properties):
        if reason_code.is_failure:
            self._error = f"MQTT connection lost: {reason_code}"

    def _on_message(self, _client, _userdata, mqtt_message):
        if mqtt_message.topic == self.trust.follower_inbox:
            self.trust.ingest_request(mqtt_message.payload, time.monotonic())
            return
        state, reason = open_pose(self.trust.pose_verifier, mqtt_message.payload)
        if state is None:
            if reason != self.pose_rejection:
                print(f"Ignored pose: {reason}", flush=True)
            self.pose_rejection = reason
            return
        try:
            message = LeaderMessage.from_state(state)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            print(f"Ignored invalid MQTT message: {exc}", flush=True)
            return
        with self._lock:
            if message.sequence <= self._last_sequence:
                return
            self._last_sequence = message.sequence
            self._history.append((message, time.monotonic()))

    def _on_leader_message(self, _client, _userdata, mqtt_message):
        if mqtt_message.topic == self.trust.leader_inbox:
            self.trust.ingest_reply(mqtt_message.payload)

    def _publish(self, client, topic, payload, what: str) -> None:
        result = client.publish(topic, payload, qos=1, retain=False)
        if result.rc != self.mqtt.MQTT_ERR_SUCCESS:
            self._error = f"MQTT {what} failed with rc={result.rc}"

    def publish(self, message: LeaderMessage) -> None:
        trust = self.trust
        payload = sign_pose(trust.protocol, trust.leader_key, trust.robot_a, trust.robot_b, message.to_state())
        self._publish(self.leader, self.topic, payload, "pose publish")

    def publish_request(self, action: str) -> None:
        self._publish(self.leader, *self.trust.request_payload(action), "signed request")

    def publish_attack(self) -> None:
        self._publish(self.attacker, *self.trust.attack_payload(), "attack publish")
        # Also try to steer the follower with an unsigned pose 5 m off to the side.
        fake = LeaderMessage(10**9, time.time_ns(), np.array([0.0, 5.0], dtype=np.float32), 0.0, np.zeros(3, dtype=np.float32))
        self._publish(self.attacker, self.topic, fake.to_payload(), "attack publish")

    def publish_due_reply(self, now: float) -> None:
        due = self.trust.take_reply(now)
        if due is not None:
            self._publish(self.follower, *due, "signed reply")

    def latest(self, max_age: float, delay: float = 0.0) -> tuple[LeaderMessage | None, float]:
        now = time.monotonic()
        with self._lock:
            return select_delayed_message(tuple(self._history), now, delay, max_age)

    def close(self) -> None:
        for client, role in ((getattr(self, "leader", None), "g1-leader"),
                             (getattr(self, "follower", None), "follower"),
                             (getattr(self, "attacker", None), "attacker")):
            if client is not None:
                try:
                    client.publish(f"{self.status_prefix}/{role}", "offline", qos=1, retain=True).wait_for_publish(1.0)
                    client.disconnect()
                    client.loop_stop()
                except Exception:
                    pass


def follower_command(message, follower_position_xy, follower_yaw, follow_distance, limits=COMMAND_LIMITS["g1"]):
    """Body-frame command that holds the slot ``follow_distance`` behind the leader."""
    leader_forward = np.array([math.cos(message.yaw), math.sin(message.yaw)])
    leader_left = np.array([-math.sin(message.yaw), math.cos(message.yaw)])
    target_xy = message.position_xy - follow_distance * leader_forward
    position_error = target_xy - follower_position_xy
    follower_forward = np.array([math.cos(follower_yaw), math.sin(follower_yaw)])
    follower_left = np.array([-math.sin(follower_yaw), math.cos(follower_yaw)])
    desired_world_velocity = message.command[0] * leader_forward + message.command[1] * leader_left
    desired_world_velocity += 0.8 * position_error
    command = np.array(
        [np.dot(desired_world_velocity, follower_forward),
         np.dot(desired_world_velocity, follower_left),
         message.command[2] + 1.2 * wrap_angle(message.yaw - follower_yaw)],
        dtype=np.float32,
    )
    limit = np.asarray(limits, dtype=np.float32)
    command[:] = np.clip(command, -limit, limit)
    if np.linalg.norm(position_error) < 0.08 and np.linalg.norm(message.command[:2]) < 0.03:
        command[:2] = 0
    if abs(wrap_angle(message.yaw - follower_yaw)) < 0.06 and abs(message.command[2]) < 0.03:
        command[2] = 0
    return command


def build_duo_model(cfg: G1Config, follow_distance: float, follower: str) -> mujoco.MjModel:
    arena = mujoco.MjSpec.from_string("""
      <mujoco model="Unitree MQTT leader follower">
        <option timestep="0.002"/>
        <visual><headlight diffuse="0.6 0.6 0.6" ambient="0.2 0.2 0.2" specular="0.8 0.8 0.8"/>
          <rgba haze="0.15 0.25 0.35 1"/><global azimuth="-120" elevation="-18"/></visual>
        <asset><texture type="skybox" builtin="flat" rgb1="0 0 0" rgb2="0 0 0" width="512" height="3072"/>
          <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4"
            rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
          <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="12 12" reflectance="0.2"/></asset>
        <worldbody><light pos="0 0 4" dir="0 0 -1" directional="true"/>
          <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/></worldbody>
      </mujoco>
    """)
    arena.attach(mujoco.MjSpec.from_file(str(cfg.robot_xml)), prefix="leader_",
                 frame=arena.worldbody.add_frame(name="leader_spawn"))
    # The follower starts facing across the leader's path, so its own walk and
    # the later turn into formation are easy to see.
    frame = arena.worldbody.add_frame(name="follower_spawn", pos=(-follow_distance, 0, 0),
                                      quat=(math.sqrt(0.5), 0, 0, -math.sqrt(0.5)))
    robot = mujoco.MjSpec.from_file(str(cfg.robot_xml)) if follower == "g1" else load_go1_spec()
    arena.attach(robot, prefix="follower_", frame=frame)
    model = arena.compile()
    model.opt.timestep = cfg.simulation_dt
    for geom_id, body_id in enumerate(model.geom_bodyid):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(body_id)) or ""
        tint = LEADER_TINT if body_name.startswith("leader_") else FOLLOWER_TINT if body_name.startswith("follower_") else None
        if tint is not None:
            model.geom_rgba[geom_id, :3] = 0.25 * model.geom_rgba[geom_id, :3] + 0.75 * tint
    if follower == "go1":
        set_policy_gains(model, "follower_")
    return model


def make_follower(model, cfg: G1Config, follower: str, data):
    if follower == "g1":
        return G1Controller(model, cfg, "follower_")
    go1 = Go1Controller(model, "follower_")
    go1.load_policy()
    go1.reset_pose(data)
    return go1


def update_speech_labels(viewer, data, leader, follower, leader_text, follower_text):
    """Draw speech-style 3-D labels that remain above each robot."""
    scene = viewer.user_scn
    if scene is None or scene.maxgeom < 2:
        return
    labels = (
        (leader, leader_text, np.array([1.0, 0.55, 0.25, 1.0], dtype=np.float32)),
        (follower, follower_text, np.array([0.3, 0.7, 1.0, 1.0], dtype=np.float32)),
    )
    with viewer.lock():
        scene.ngeom = 2
        for index, (robot, text, colour) in enumerate(labels):
            position = data.xpos[robot.body_id].copy()
            position[2] += 0.85
            geom = scene.geoms[index]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LABEL, np.zeros(3), position, np.eye(3).ravel(), colour)
            geom.label = text


def run(args) -> None:
    cfg = load_g1_config()
    model = build_duo_model(cfg, args.follow_distance, args.follower)
    data = mujoco.MjData(model)
    leader = G1Controller(model, cfg, "leader_")
    follower = make_follower(model, cfg, args.follower, data)
    mujoco.mj_forward(model, data)
    limits = COMMAND_LIMITS[args.follower]
    name = args.follower.upper()
    mqtt_link = MqttFormationLink(args)
    fixed_command = np.asarray(args.command, dtype=np.float32)
    human_input = None if args.headless else HumanCommandInput(f"G1 leader + {name} follower", duo=True)
    viewer = None
    if not args.headless:
        from mujoco import viewer as mj_viewer
        viewer = mj_viewer.launch_passive(model, data)
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = leader.body_id
        viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = 4.0, -18.0, 135.0

    print(f"MQTT connected: {args.mqtt_host}:{args.mqtt_port}  topic={mqtt_link.topic}", flush=True)
    print(f"Trust gate: score={args.trust} decision pause={mqtt_link.trust.follower.reaction_delay:.2f}s "
          f"inbox={mqtt_link.trust.follower_inbox}", flush=True)
    print(f"G1 leader, {name} follower at {args.follow_distance:.2f}m", flush=True)
    if args.demo_mode == "story":
        print(STORY_TITLE, flush=True)
    elif human_input:
        print(f"WASD/QE drives the LEADER. The {name} follows only after a signed request is allowed.", flush=True)
    counter = sequence = 0
    sent_token = ""
    last_report = 0.0
    follower_cmd = np.zeros(3, dtype=np.float32)
    first_motion_at: float | None = None
    first_received_at: float | None = None
    leader_text = "LEADER: Ready - hold W to go"
    follower_text = "FOLLOWER: Going my own way..."
    wall_start = time.monotonic()
    try:
        while ((args.duration <= 0 or time.monotonic() - wall_start < args.duration)
               and (viewer is None or viewer.is_running())
               and (human_input is None or human_input.running)):
            step_start = time.monotonic()
            beat = story_beat(time.monotonic() - wall_start) if args.demo_mode == "story" else None
            operator_command = human_input.poll() if human_input else fixed_command
            leader_cmd = beat.leader_command if beat else operator_command
            leader.apply(data)
            follower.apply(data)
            mujoco.mj_step(model, data)
            counter += 1
            if counter % cfg.control_decimation == 0:
                now = time.monotonic()
                sequence += 1
                mqtt_link.publish(LeaderMessage(sequence, time.time_ns(), leader.position_xy(data), leader.yaw(data), leader_cmd.copy()))
                received, _ = mqtt_link.latest(args.mqtt_stale_timeout, args.message_delay)
                leader_is_moving = np.linalg.norm(leader_cmd) > 0.03
                if leader_is_moving and first_motion_at is None:
                    first_motion_at = now
                if first_motion_at is not None and received is not None and first_received_at is None:
                    first_received_at = now
                token = "attack" if beat and beat.attack else (beat.action if beat else None)
                if token is None and leader_is_moving:
                    token = "mirror_motion"
                if token and token != sent_token:
                    if token == "attack":
                        mqtt_link.publish_attack()
                    else:
                        mqtt_link.publish_request(token)
                    sent_token = token
                mqtt_link.publish_due_reply(now)
                trust_state = mqtt_link.trust.follower.snapshot()
                grant = trust_state["grant"]
                approved = grant is not None and grant.result != "DENY"

                if approved and received is not None:
                    follower_cmd = motion_from_grant(follower_command(
                        received, follower.position_xy(data), follower.yaw(data), args.follow_distance, limits,
                    ), grant)
                    if grant.result == "PARTIAL":
                        follower_text = "FOLLOWER: On the route, not taking the load"
                    else:
                        follower_text = (f"FOLLOWER: {beat.follower_text}" if beat
                                         else "FOLLOWER: Matching the signed motion")
                elif approved:
                    follower_cmd = np.zeros(3, dtype=np.float32)
                    follower_text = ("FOLLOWER: MQTT stale - STOP" if first_received_at is not None
                                     else "FOLLOWER: Pose stream in transit...")
                elif first_motion_at is None and grant is None and not trust_state["pending"]:
                    follower_cmd = np.array([args.distracted_speed, 0.0, 0.0], dtype=np.float32)
                    follower_text = f"FOLLOWER: {beat.follower_text}" if beat else "FOLLOWER: Going my own way..."
                else:
                    follower_cmd = np.zeros(3, dtype=np.float32)
                    if grant is not None and grant.result == "DENY":
                        follower_text = "FOLLOWER: Refused — trust is too low"
                    elif trust_state["pending"]:
                        follower_text = "FOLLOWER: Verifying signature and trust"
                    else:
                        follower_text = "FOLLOWER: Waiting for a signed decision"
                if beat and beat.attack and trust_state["rejection"]:
                    follower_text = ("FOLLOWER: Rejected unlock_door — still following" if approved
                                     else "FOLLOWER: Rejected unlock_door")
                    leader_text = "LEADER: That door command is not mine"
                else:
                    leader_text = (f"LEADER: {beat.leader_text}" if beat else
                                   ("LEADER: Follow me!" if leader_is_moving else "LEADER: Not calling yet"))
                phase = gait_phase(data)
                leader.update_policy(data, leader_cmd, phase)
                follower.update_policy(data, follower_cmd, phase)
            if data.time - last_report >= 1.0:
                received, broker_age = mqtt_link.latest(args.mqtt_stale_timeout, args.message_delay)
                separation = np.linalg.norm(leader.position_xy(data) - follower.position_xy(data))
                state = f"seq={received.sequence} age={broker_age * 1000:.0f}ms" if received else "STALE → STOP"
                print(f"t={data.time:5.1f}s MQTT[{state}] {mqtt_link.trust.status_line()} "
                      f"distance={separation:.2f}m height={data.xpos[follower.body_id][2]:.2f} "
                      f"follower_cmd=({follower_cmd[0]:+.2f},{follower_cmd[1]:+.2f},{follower_cmd[2]:+.2f})", flush=True)
                last_report = data.time
            if human_input and counter % cfg.control_decimation == 0:
                separation = np.linalg.norm(leader.position_xy(data) - follower.position_xy(data))
                received, broker_age = mqtt_link.latest(args.mqtt_stale_timeout, args.message_delay)
                human_input.show_status({
                    "connected": received is not None,
                    "sequence": received.sequence if received else "-",
                    "age_ms": round(broker_age * 1000) if received else "-",
                    "distance_m": round(float(separation), 2),
                    "leader_command": [round(float(value), 2) for value in leader_cmd],
                    "stage": beat.name if beat else "MANUAL CONTROL",
                    "demo_title": STORY_TITLE if beat else "MANUAL  signed request, then follow",
                    "follower_title": f"{name} FOLLOWER",
                    "message_delay_ms": round(args.message_delay * 1000),
                    "trust_delay_ms": round(mqtt_link.trust.follower.reaction_delay * 1000),
                    "trust_text": mqtt_link.trust.status_line(),
                    "ack_text": mqtt_link.trust.reply_text or "waiting for signed reply",
                    "leader_text": leader_text.removeprefix("LEADER: "),
                    "follower_text": follower_text.removeprefix("FOLLOWER: "),
                })
            if viewer:
                update_speech_labels(viewer, data, leader, follower, leader_text, follower_text)
                viewer.sync()
                remaining = cfg.simulation_dt - (time.monotonic() - step_start)
                if remaining > 0:
                    time.sleep(remaining)
    except KeyboardInterrupt:
        print("\nStopped by user.", flush=True)
    finally:
        if viewer:
            viewer.close()
        if human_input:
            human_input.close()
        mqtt_link.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--follower", choices=FOLLOWERS, default="g1", help="Robot that follows the G1 leader")
    add_run_arguments(parser)
    parser.add_argument("--demo-mode", choices=("manual", "story"), default="manual",
                        help="manual: WASD drives the leader. story: scripted request, attack and carry")
    parser.add_argument("--follow-distance", type=float, default=1.2, help="Formation distance in metres")
    parser.add_argument("--follower-start-delay", type=float, default=1.0,
                        help="Seconds the follower waits after the leader first moves")
    parser.add_argument("--distracted-speed", type=float, default=0.18,
                        help="Follower's own walking speed before it is asked to follow (m/s)")
    parser.add_argument("--trust", type=int, default=70,
                        help="Follower trust in the leader, 0-100. 45 makes the carry PARTIAL, 10 denies following")
    parser.add_argument("--trust-delay", type=float, default=0.9,
                        help="Seconds after verification before the follower acts")
    parser.add_argument("--message-delay", type=float, default=0.0,
                        help="Extra lag on the pose stream after MQTT delivery, for a legible reaction")
    parser.add_argument("--mqtt-host", default="127.0.0.1")
    parser.add_argument("--mqtt-port", type=int, default=1883)
    parser.add_argument("--formation-id", default="demo-1")
    parser.add_argument("--mqtt-connect-timeout", type=float, default=8.0)
    parser.add_argument("--mqtt-stale-timeout", type=float, default=0.5)
    parser.add_argument("--mqtt-tls", action="store_true")
    parser.add_argument("--mqtt-ca", type=Path)
    parser.add_argument("--mqtt-leader-cert", type=Path)
    parser.add_argument("--mqtt-leader-key", type=Path)
    parser.add_argument("--mqtt-follower-cert", type=Path)
    parser.add_argument("--mqtt-follower-key", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    check_run_arguments(args)
    if args.follow_distance <= 0:
        raise SystemExit("--follow-distance must be positive")
    if args.follower_start_delay < 0:
        raise SystemExit("--follower-start-delay cannot be negative")
    if not 0 <= args.message_delay <= 3.0:
        raise SystemExit("--message-delay must be between 0 and 3 seconds")
    if not 0 <= args.trust_delay <= 3.0:
        raise SystemExit("--trust-delay must be between 0 and 3 seconds")
    if not 0 <= args.trust <= 100:
        raise SystemExit("--trust must be between 0 and 100")
    if not 0 <= args.distracted_speed <= 0.4:
        raise SystemExit("--distracted-speed must be between 0 and 0.4")
    if args.mqtt_connect_timeout <= 0 or args.mqtt_stale_timeout <= 0:
        raise SystemExit("MQTT timeouts must be positive")
    ensure_macos_viewer_runtime(args.headless, "unitree_demo.duo")
    run(args)


if __name__ == "__main__":
    main()
