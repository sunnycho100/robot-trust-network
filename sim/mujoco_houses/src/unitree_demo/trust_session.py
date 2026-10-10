"""Gate G1 motion on a robot-trust-network signed request.

The pose stream still carries position and velocity. It is applied only after the
follower has verified the request and the trust policy has allowed it. A failed
check does not cancel an approval that is already in effect.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path

import numpy as np


PARTIAL_LINEAR_SCALE = np.float32(0.35)
ACTIONS = {
    "mirror_motion": ("Copy my left turn and match my pace.", "inspect together"),
    "carry_together": ("Take the other side and move with me.", "shared delivery"),
}


def network_root() -> Path:
    return Path(__file__).resolve().parents[4] / "network"


def load_trust():
    """Import the trust-network protocol from this repo's network/ folder."""
    root = network_root()
    if not (root / "protocol.py").is_file():
        raise RuntimeError(f"robot-trust-network protocol was not found at {root}")
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    import policy
    import protocol

    return protocol, policy


def ensure_keys(protocol) -> None:
    registry_path = os.path.join(protocol.KEYS_DIR, "registry.json")
    if os.path.exists(registry_path):
        return
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    import base64

    os.makedirs(protocol.KEYS_DIR, exist_ok=True)
    registry = {}
    for robot_id in protocol.ROBOTS.values():
        key = Ed25519PrivateKey.generate()
        path = os.path.join(protocol.KEYS_DIR, robot_id.split("@")[0] + ".key")
        with open(path, "wb") as handle:
            handle.write(key.private_bytes(
                serialization.Encoding.Raw,
                serialization.PrivateFormat.Raw,
                serialization.NoEncryption(),
            ))
        os.chmod(path, 0o600)
        public = key.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        )
        registry[robot_id] = {
            "public_key": base64.b64encode(public).decode(),
            "status": "active",
        }
    with open(registry_path, "w") as handle:
        json.dump(registry, handle, indent=2)
    print(f"Created robot trust keys in {protocol.KEYS_DIR}", flush=True)


def response_text(message, decision) -> str:
    if decision.result == "DENY":
        return f"No. {decision.reason}."
    if decision.result == "PARTIAL":
        return "Partial permission. I can follow the route, but I will not handle the load."
    action = message.get("action")
    if action == "mirror_motion":
        return "Accepted. I copied the turn and matched your pace."
    if action == "carry_together":
        return "Accepted. I am holding my side; moving together now."
    return f"{decision.result.title()}. {decision.reason}."


def motion_from_grant(command, grant) -> np.ndarray:
    """Return the MuJoCo command a verified decision is allowed to drive."""
    if grant is None or grant.result == "DENY":
        return np.zeros(3, dtype=np.float32)
    scaled = np.asarray(command, dtype=np.float32).copy()
    if grant.result == "PARTIAL":
        scaled[:2] *= PARTIAL_LINEAR_SCALE
    return scaled


def sign_pose(protocol, key, sender, receiver, state: dict) -> bytes:
    """Wrap one leader pose in a signed envelope, like any other request."""
    message = protocol.make_message(key, sender, receiver, "pose", kind="pose", state=state)
    return json.dumps(message, ensure_ascii=False).encode("utf-8")


def open_pose(verifier, payload):
    """Return (state, "ok") for a pose the leader really signed, else (None, reason)."""
    if len(payload) > 4096:
        return None, "too large"
    message = _decode(payload)
    verified, reason = verifier.check(message)
    if not verified:
        return None, reason
    if message.get("kind") != "pose" or not isinstance(message.get("state"), dict):
        return None, "not a pose"
    return message["state"], "ok"


class TrustFollower:
    """Verify, then decide. Rejection leaves the current approval in place."""

    def __init__(self, verifier, trust_score, reaction_delay, decide) -> None:
        self.verifier = verifier
        self.trust_score = trust_score
        self.reaction_delay = reaction_delay
        self._decide = decide
        self._lock = threading.Lock()
        self.grant = None
        self.grant_action = None
        self.grant_at = None
        self.pending = None
        self.rejection = None
        self.rejected_action = None
        self.rejection_at = None

    def receive(self, message, now: float) -> str:
        with self._lock:
            verified, reason = self.verifier.check(message)
            if not verified:
                self.rejection = reason
                self.rejected_action = message.get("action") if isinstance(message, dict) else None
                self.rejection_at = now
                return "rejected"
            if isinstance(message, dict) and message.get("kind") == "response":
                return "response"
            decision = self._decide(message, self.trust_score)
            self.pending = (now + self.reaction_delay, message, decision)
            return "pending"

    def poll(self, now: float):
        with self._lock:
            if self.pending is None or now < self.pending[0]:
                return None
            _, message, decision = self.pending
            self.pending = None
            self.grant = decision
            self.grant_action = message.get("action")
            self.grant_at = now
            return message, decision

    def snapshot(self):
        with self._lock:
            return {
                "grant": self.grant,
                "action": self.grant_action,
                "pending": self.pending is not None,
                "rejection": self.rejection,
                "rejected_action": self.rejected_action,
                "rejection_at": self.rejection_at,
                "grant_at": self.grant_at,
            }


class TrustMailbox:
    """Signed request and reply envelopes for the two G1 MQTT clients."""

    def __init__(self, trust_score: int, reaction_delay: float) -> None:
        protocol, policy = load_trust()
        ensure_keys(protocol)
        self.protocol = protocol
        self.robot_a = protocol.ROBOTS["a"]
        self.robot_b = protocol.ROBOTS["b"]
        self.leader_inbox = protocol.inbox(self.robot_a)
        self.follower_inbox = protocol.inbox(self.robot_b)
        registry = protocol.load_registry()
        self.leader_key = protocol.load_private_key(self.robot_a)
        self.follower_key = protocol.load_private_key(self.robot_b)
        self.leader_verifier = protocol.Verifier(self.robot_a, registry)
        # Own nonce memory for the 50 Hz pose stream, apart from the request inbox.
        self.pose_verifier = protocol.Verifier(self.robot_b, registry)
        self.follower = TrustFollower(
            protocol.Verifier(self.robot_b, registry),
            trust_score,
            reaction_delay,
            policy.decide,
        )
        self.reply_text = ""

    def request_payload(self, action: str) -> tuple[str, bytes]:
        try:
            request, reason = ACTIONS[action]
        except KeyError as exc:
            raise ValueError(f"Unsupported trust action: {action}") from exc
        message = self.protocol.make_message(
            self.leader_key, self.robot_a, self.robot_b, request,
            action=action, reason=reason, kind="request",
        )
        return self.follower_inbox, json.dumps(message, ensure_ascii=False).encode("utf-8")

    def attack_payload(self) -> tuple[str, bytes]:
        message = {
            "from": self.robot_a,
            "to": self.robot_b,
            "action": "unlock_door",
            "request": "Open the front door now",
        }
        return self.follower_inbox, json.dumps(message).encode("utf-8")

    def ingest_request(self, payload, now: float) -> str:
        message = _decode(payload)
        return self.follower.receive(message, now)

    def take_reply(self, now: float):
        promoted = self.follower.poll(now)
        if promoted is None:
            return None
        message, decision = promoted
        text = response_text(message, decision)
        signed = self.protocol.make_message(
            self.follower_key, self.robot_b, self.robot_a, text,
            action=message.get("action"), reason=decision.reason,
            kind="response", reply_to=message.get("nonce"),
        )
        return self.leader_inbox, json.dumps(signed, ensure_ascii=False).encode("utf-8")

    def ingest_reply(self, payload) -> None:
        message = _decode(payload)
        verified, reason = self.leader_verifier.check(message)
        if not verified:
            self.reply_text = f"Ignored reply: {reason}"
            return
        self.reply_text = message.get("request", "") if isinstance(message, dict) else ""

    def status_line(self) -> str:
        state = self.follower.snapshot()
        if state["pending"]:
            return "VERIFYING · signature, recipient, time, nonce"
        rejection_at = state["rejection_at"] if state["rejection_at"] is not None else -1.0
        grant_at = state["grant_at"] if state["grant_at"] is not None else -1.0
        if state["rejection"] and rejection_at >= grant_at:
            action = state["rejected_action"] or "message"
            return f"REJECTED {action} · {state['rejection']}"
        if state["grant"] is not None:
            return f"{state['grant'].result} · {state['action']}"
        return "waiting for a signed request"


def _decode(payload):
    try:
        return json.loads(payload)
    except (TypeError, ValueError, json.JSONDecodeError):
        if isinstance(payload, bytes):
            return payload.decode(errors="replace")
        return payload
