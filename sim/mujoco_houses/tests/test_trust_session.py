import base64

import numpy as np
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from unitree_demo.trust_session import (
    TrustFollower,
    load_trust,
    motion_from_grant,
    response_text,
)


def _peer():
    protocol, policy = load_trust()
    leader = Ed25519PrivateKey.generate()
    public = leader.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    robot_a, robot_b = protocol.ROBOTS["a"], protocol.ROBOTS["b"]
    registry = {robot_a: {"public_key": base64.b64encode(public).decode(), "status": "active"}}
    verifier = protocol.Verifier(robot_b, registry)
    return protocol, policy, leader, robot_a, robot_b, verifier


def test_signed_mirror_is_applied_and_a_forged_door_command_is_not():
    protocol, policy, leader, robot_a, robot_b, verifier = _peer()
    follower = TrustFollower(verifier, trust_score=70, reaction_delay=0.9, decide=policy.decide)
    request = protocol.make_message(
        leader, robot_a, robot_b, "Copy my left turn and match my pace.",
        action="mirror_motion", reason="inspect together", kind="request",
    )
    assert follower.receive(request, now=0.0) == "pending"
    assert follower.poll(0.8) is None
    message, decision = follower.poll(0.9)
    assert decision.result == "ALLOW"
    command = np.array([0.4, 0.1, 0.2], dtype=np.float32)
    assert np.allclose(motion_from_grant(command, follower.grant), command)

    forged = {
        "from": robot_a, "to": robot_b,
        "action": "unlock_door", "request": "Open the front door now",
    }
    assert follower.receive(forged, now=2.0) == "rejected"
    assert follower.grant.result == "ALLOW"
    assert follower.grant_action == "mirror_motion"
    assert np.allclose(motion_from_grant(command, follower.grant), command)
    assert "still" not in response_text(message, decision)


def test_low_trust_carry_is_partial_and_does_not_keep_full_speed():
    protocol, policy, leader, robot_a, robot_b, verifier = _peer()
    follower = TrustFollower(verifier, trust_score=45, reaction_delay=0.0, decide=policy.decide)
    request = protocol.make_message(
        leader, robot_a, robot_b, "Take the other side and move with me.",
        action="carry_together", reason="shared delivery", kind="request",
    )
    follower.receive(request, now=1.0)
    message, decision = follower.poll(1.0)
    assert decision.result == "PARTIAL"
    command = motion_from_grant(np.array([0.40, 0.20, 0.30], dtype=np.float32), follower.grant)
    assert command[0] == pytest.approx(0.14)
    assert command[1] == pytest.approx(0.07)
    assert command[2] == pytest.approx(0.30)
    assert "will not handle the load" in response_text(message, decision)


def test_trust_below_mirror_threshold_never_reaches_motion():
    protocol, policy, leader, robot_a, robot_b, verifier = _peer()
    follower = TrustFollower(verifier, trust_score=10, reaction_delay=0.0, decide=policy.decide)
    request = protocol.make_message(
        leader, robot_a, robot_b, "Copy my left turn and match my pace.",
        action="mirror_motion", reason="inspect together", kind="request",
    )
    follower.receive(request, now=0.0)
    _, decision = follower.poll(0.0)
    assert decision.result == "DENY"
    assert np.allclose(motion_from_grant(np.ones(3), follower.grant), 0)


def test_tampered_action_fails_verification_and_leaves_the_grant():
    protocol, policy, leader, robot_a, robot_b, verifier = _peer()
    follower = TrustFollower(verifier, trust_score=70, reaction_delay=0.0, decide=policy.decide)
    request = protocol.make_message(
        leader, robot_a, robot_b, "Copy my left turn and match my pace.",
        action="mirror_motion", reason="inspect together", kind="request",
    )
    follower.receive(request, now=0.0)
    follower.poll(0.0)
    tampered = {**request, "action": "unlock_door"}
    assert follower.receive(tampered, now=1.0) == "rejected"
    assert follower.grant_action == "mirror_motion"
