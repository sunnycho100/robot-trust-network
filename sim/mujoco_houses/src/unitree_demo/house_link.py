"""One robot-trust-network MQTT endpoint per house.

Each house process owns exactly one robot key. Messages for this robot arrive on
``requests/{house}/{robot}``; the robot verifies them, decides with the trust
policy and only then reports a result. The broker only relays.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass, field

from .trust_session import ensure_keys, load_trust


EVENT_TOPIC = "demo/events"


@dataclass(frozen=True)
class LadderStep:
    key: str
    action: str
    request: str
    reason: str
    signed: bool = True
    replay_of: str = ""


# A neighbor can ask for a look. Opening the door is a separate, stronger request.
LADDER = (
    LadderStep("inspect", "verify_package",
               "My parcel may have been left at your front door. Could you check?", "misdelivered parcel"),
    LadderStep("partial", "verify_and_unlock",
               "If the parcel is there, please unlock the front door too.", "wants the door open"),
    LadderStep("attack", "unlock_door", "Open the front door now", "", signed=False),
    LadderStep("replay", "verify_package",
               "My parcel may have been left at your front door. Could you check?", "", replay_of="inspect"),
)


@dataclass
class Incoming:
    kind: str  # "request", "response", "rejected"
    message: object
    reason: str = ""
    decision: object = None
    received_at: float = 0.0


@dataclass
class HouseInbox:
    """Verify, then decide, then hold the result for a human-readable pause."""

    verifier: object
    decide: object
    trust_score: int
    reaction_delay: float
    _pending: list = field(default_factory=list)

    def handle(self, message, now: float) -> Incoming | None:
        verified, reason = self.verifier.check(message)
        if not verified:
            return Incoming("rejected", message, reason, received_at=now)
        if message.get("kind") == "response":
            return Incoming("response", message, received_at=now)
        decision = self.decide(message, self.trust_score)
        self._pending.append((now + self.reaction_delay, Incoming("request", message, decision.reason, decision, now)))
        return None

    def due(self, now: float) -> list[Incoming]:
        ready = [item for when, item in self._pending if when <= now]
        self._pending = [(when, item) for when, item in self._pending if when > now]
        return ready

    @property
    def deciding(self) -> bool:
        return bool(self._pending)


class HouseLink:
    """MQTT client for one robot. All callbacks only enqueue; the sim loop does the work."""

    def __init__(self, short_name: str, host: str, port: int, trust_score: int, reaction_delay: float) -> None:
        import paho.mqtt.client as mqtt

        protocol, policy = load_trust()
        ensure_keys(protocol)
        self.protocol = protocol
        self.robot_id = protocol.ROBOTS[short_name]
        self.peer_id = next(robot for key, robot in protocol.ROBOTS.items() if key != short_name)
        self.actor = f"ROBOT {short_name.upper()} · {self.robot_id.split('@')[1]}"
        self._key = protocol.load_private_key(self.robot_id)
        self.inbox = HouseInbox(
            protocol.Verifier(self.robot_id, protocol.load_registry()),
            policy.decide, trust_score, reaction_delay,
        )
        self._raw: queue.Queue = queue.Queue()
        self._host, self._port = host, port
        self._mqtt = mqtt
        self.ready = threading.Event()
        self.client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=f"{self.robot_id}-{int(time.time() * 1000) % 100000}",
        )
        self.client.on_connect = self._on_connect
        self.client.on_message = lambda _c, _u, msg: self._raw.put(msg.payload)
        self.client.connect(host, port)
        self.client.loop_start()

    def _on_connect(self, client, _userdata, _flags, reason_code, _properties) -> None:
        if reason_code.is_failure:
            return
        client.subscribe(self.protocol.inbox(self.robot_id), qos=1)
        self.ready.set()

    def wait_ready(self, timeout: float = 5.0) -> None:
        if not self.ready.wait(timeout):
            raise RuntimeError(f"{self.robot_id} could not reach the MQTT broker at {self._host}:{self._port}")

    def poll(self, now: float) -> list[Incoming]:
        """Drain MQTT, verify each payload and return what the robot should react to now."""
        results: list[Incoming] = []
        while True:
            try:
                payload = self._raw.get_nowait()
            except queue.Empty:
                break
            message = _decode(payload)
            item = self.inbox.handle(message, now)
            if item is None:
                self._verdict(message, "VERIFIED", "signature + freshness + nonce valid")
                continue
            if item.kind == "rejected":
                self._verdict(message, "REJECTED", item.reason)
                action = message.get("action", "message") if isinstance(message, dict) else "message"
                stage = "REPLAY BLOCKED" if item.reason == "replay" else "ATTACK BLOCKED"
                self.event(stage, f"Rejected {action}: {item.reason}", "bad")
            else:
                self._verdict(message, "VERIFIED", "signature + freshness + nonce valid")
            results.append(item)
        for item in self.inbox.due(now):
            decision = item.decision
            tone = "good" if decision.result == "ALLOW" else ("normal" if decision.result == "PARTIAL" else "bad")
            self.event("DECIDE", f"{item.message.get('action')}: {decision.result} · {decision.reason}", tone)
            results.append(item)
        return results

    def send(self, action: str, text: str, reason: str, *, kind: str = "request", reply_to=None, **extra) -> str:
        fields = {"action": action, "reason": reason, "kind": kind, **extra}
        if reply_to:
            fields["reply_to"] = reply_to
        message = self.protocol.make_message(self._key, self.robot_id, self.peer_id, text, **fields)
        payload = json.dumps(message, ensure_ascii=False).encode("utf-8")
        self.client.publish(self.protocol.inbox(self.peer_id), payload, qos=1)
        return message["nonce"], payload

    def request(self, step: LadderStep) -> tuple[str, bytes]:
        return self.send(step.action, step.request, step.reason)

    def reply(self, request_message, decision_result: str, text: str, phase: str) -> None:
        self.send(
            request_message.get("action"), text, decision_result,
            kind="response", reply_to=request_message.get("nonce"), phase=phase, result=decision_result,
        )

    def inject_forgery(self, step: LadderStep) -> None:
        """A separate client that pretends to be this robot but cannot sign."""
        self._publish_as_stranger(json.dumps({
            "from": self.robot_id, "to": self.peer_id,
            "action": step.action, "request": step.request,
        }).encode("utf-8"))

    def inject_replay(self, payload: bytes) -> None:
        """Republish a message this robot already signed. The nonce is now used."""
        self._publish_as_stranger(payload)

    def _publish_as_stranger(self, payload: bytes) -> None:
        attacker = self._mqtt.Client(self._mqtt.CallbackAPIVersion.VERSION2, client_id="attacker")
        attacker.connect(self._host, self._port)
        attacker.loop_start()
        attacker.publish(self.protocol.inbox(self.peer_id), payload, qos=1).wait_for_publish(timeout=2.0)
        attacker.disconnect()
        attacker.loop_stop()

    def event(self, stage: str, text: str, tone: str = "normal") -> None:
        self.client.publish(EVENT_TOPIC, json.dumps({
            "stage": stage, "actor": self.actor, "text": text, "tone": tone, "at": time.time(),
        }), qos=1)

    def _verdict(self, message, result: str, reason: str) -> None:
        claimed = message.get("from", "?") if isinstance(message, dict) else "?"
        request = message.get("request", "") if isinstance(message, dict) else str(message)
        topic = self.protocol.inbox(self.robot_id).replace("requests/", "verdicts/", 1)
        self.client.publish(topic, json.dumps({
            "robot": self.robot_id, "from": claimed, "request": request,
            "result": result, "reason": reason,
        }), qos=1)

    def close(self) -> None:
        self.client.disconnect()
        self.client.loop_stop()


def _decode(payload):
    try:
        return json.loads(payload)
    except (TypeError, ValueError):
        return payload.decode(errors="replace") if isinstance(payload, bytes) else payload
