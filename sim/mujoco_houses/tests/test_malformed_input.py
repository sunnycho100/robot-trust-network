"""Untrusted payloads must not terminate a sim loop or MQTT callback."""

import base64
import json
import io
import queue
import threading
from collections import deque
from types import SimpleNamespace

import pytest
import mujoco

from test_trust_session import _peer
from test_house_flow import make_pair
from unitree_demo.house_demo import handle_responder_input
from unitree_demo.duo import LeaderMessage, MqttFormationLink
from unitree_demo.common import viewer_label
from unitree_demo.house_link import HouseInbox, HouseLink, Incoming
from unitree_demo.trust_session import TrustFollower, TrustMailbox


BAD_FIELDS = [
    ("from", []), ("to", {}), ("nonce", []), ("sig", 12), ("action", []),
    ("request", {}), ("reason", []), ("kind", []), ("kind", "other"),
    ("reply_to", []), ("phase", []), ("result", []), ("time", "now"),
    ("time", None), ("time", True), ("time", float("nan")),
    ("time", float("inf")), ("time", 10 ** 400),
]


@pytest.mark.parametrize("field,value", BAD_FIELDS)
@pytest.mark.parametrize("boundary", ["house", "duo", "duo_reply"])
def test_malformed_signed_envelopes_are_contained_and_valid_traffic_recovers(field, value, boundary):
    protocol, policy, key, a, b, verifier = _peer()
    valid = protocol.make_message(key, a, b, "please", action="mirror_motion")
    malformed = {**valid, field: value}
    # Even an authenticated peer must not crash the application with wrong field types.
    if field != "sig":
        malformed["sig"] = base64.b64encode(key.sign(protocol.canonical(malformed))).decode()
    if boundary == "house":
        inbox = HouseInbox(verifier, policy.decide, 70, 0)
        assert inbox.handle(malformed, 0).kind == "rejected"
        assert inbox.handle(valid, 1) is None
        assert inbox.due(1)[0].decision.result == "ALLOW"
    elif boundary == "duo":
        follower = TrustFollower(verifier, 70, 0, policy.decide)
        follower.receive(valid, 0)
        follower.poll(0)
        grant = follower.grant
        assert follower.receive(malformed, 1) == "rejected"
        assert follower.grant is grant
        fresh = protocol.make_message(key, a, b, "please", action="carry_together")
        assert follower.receive(fresh, 2) == "pending"
        assert follower.poll(2)[1].result == "ALLOW"
    else:
        mailbox = TrustMailbox.__new__(TrustMailbox)
        mailbox.leader_verifier = verifier
        mailbox.ingest_reply(json.dumps(malformed).encode())
        assert mailbox.reply_text.startswith("Ignored reply:")
        mailbox.ingest_reply(json.dumps(valid).encode())
        assert mailbox.reply_text == "please"


def test_house_poll_survives_invalid_json_deep_json_and_bad_sender_then_accepts_valid_message():
    protocol, policy, key, a, b, verifier = _peer()
    link = HouseLink.__new__(HouseLink)
    link.protocol, link.robot_id, link.actor = protocol, b, "test"
    link.client = SimpleNamespace(publish=lambda *args, **kwargs: None)
    link.inbox = HouseInbox(verifier, policy.decide, 55, 0)
    link._raw = queue.Queue()
    valid = protocol.make_message(key, a, b, "inspect", action="verify_package")
    for payload in (b"{", b"\xff", b"[" * 2000 + b"]" * 2000,
                    json.dumps({**valid, "from": []}).encode(), json.dumps(valid).encode()):
        link._raw.put(payload)
    results = link.poll(0)
    assert [item.kind for item in results] == ["rejected"] * 4 + ["request"]
    assert results[-1].decision.result == "ALLOW"


@pytest.mark.parametrize("field", ["action", "request", "reason", "reply_to"])
def test_lone_surrogate_cannot_authenticate_or_reach_executor(field):
    protocol, policy, key, a, b, verifier = _peer()
    valid = protocol.make_message(key, a, b, "inspect", action="verify_package")
    malformed = {**valid, field: "\ud800"}
    # The shared canonical UTF-8 format cannot represent a lone surrogate.
    with pytest.raises(UnicodeEncodeError):
        protocol.canonical(malformed)
    inbox = HouseInbox(verifier, policy.decide, 55, 0)
    item = inbox.handle(json.loads(json.dumps(malformed)), 0)
    assert item.kind == "rejected" and item.reason == "bad signature"
    assert not inbox.due(0)
    assert inbox.handle(valid, 1) is None
    assert inbox.due(1)[0].kind == "request"


@pytest.mark.parametrize("payload", [b"[]", b"null", b"1", b"{", b"[" * 2000 + b"]" * 2000])
def test_duo_pose_callback_survives_malformed_json(payload):
    link = MqttFormationLink.__new__(MqttFormationLink)
    link.trust = SimpleNamespace(follower_inbox="signed")
    link._lock = threading.Lock()
    link._history, link._last_sequence = deque(), -1
    link._on_message(None, None, SimpleNamespace(topic="pose", payload=payload))
    good = LeaderMessage(1, 0, [0, 0], 0, [0, 0, 0]).to_payload()
    link._on_message(None, None, SimpleNamespace(topic="pose", payload=good))
    assert link._last_sequence == 1
    assert len(link._history) == 1


@pytest.mark.parametrize("encoding,action", [("utf-8", "\ud800"), ("cp1252", "中→"), ("cp1252", "中" * 10000)],
                         ids=["surrogate", "legacy-unicode", "long-label"])
def test_rejected_action_is_safe_in_house_output_and_duo_status(monkeypatch, encoding, action):
    stream = io.BytesIO()
    output = io.TextIOWrapper(stream, encoding=encoding, errors="strict")
    monkeypatch.setattr("sys.stdout", output)
    _, house = make_pair()
    malformed = {"action": action}
    handle_responder_input(house, Incoming("rejected", malformed, "no signature"))
    assert house.speech.encode("utf-8")
    protocol, policy, _, _, _, verifier = _peer()
    mailbox = TrustMailbox.__new__(TrustMailbox)
    mailbox.follower = TrustFollower(verifier, 70, 0, policy.decide)
    assert mailbox.follower.receive(malformed, 0) == "rejected"
    assert len(mailbox.follower.rejected_action) <= 43
    print(mailbox.status_line())
    output.flush()
    assert b"\\u" in stream.getvalue()


def test_house_console_can_report_valid_unicode_on_legacy_stdout(monkeypatch):
    stream = io.BytesIO()
    output = io.TextIOWrapper(stream, encoding="cp1252", errors="strict")
    monkeypatch.setattr("sys.stdout", output)
    a, _ = make_pair()
    a.note("REPLY", "中→")
    output.flush()
    assert b"\\u4e2d" in stream.getvalue()
    assert a.log[-1][1] == "中→"


@pytest.mark.parametrize("text", ["中" * 90, "😀" * 90, "\ud800" * 90], ids=["cjk", "emoji", "surrogate"])
def test_viewer_labels_fit_mujoco_byte_limit(text):
    label = viewer_label(text)
    assert len(label.encode("utf-8")) <= 90
    scene = mujoco.MjvScene(mujoco.MjModel.from_xml_string("<mujoco/>"), maxgeom=1)
    scene.geoms[0].label = label
    assert scene.geoms[0].label == label


@pytest.mark.parametrize("field,value", [("x", "中"), ("x", True), ("x", "1e3"), ("x", 1e100),
                                         ("seq", 1.5), ("seq", "7"), ("sent_at_ns", False)])
def test_invalid_pose_types_cannot_crash_legacy_stdout_callback(monkeypatch, field, value):
    stream = io.BytesIO()
    output = io.TextIOWrapper(stream, encoding="cp1252", errors="strict")
    monkeypatch.setattr("sys.stdout", output)
    link = MqttFormationLink.__new__(MqttFormationLink)
    link.trust = SimpleNamespace(follower_inbox="signed")
    link._lock = threading.Lock()
    link._history, link._last_sequence = deque(), -1
    valid = LeaderMessage(1, 0, [0, 0], 0, [0, 0, 0]).to_payload()
    malformed = json.loads(valid)
    if field == "x":
        malformed["pose"][field] = value
    else:
        malformed[field] = value
    link._on_message(None, None, SimpleNamespace(topic="pose", payload=json.dumps(malformed).encode()))
    assert not link._history
    link._on_message(None, None, SimpleNamespace(topic="pose", payload=valid))
    assert link._last_sequence == 1
