"""Exercise both story generators with real signed envelopes and deterministic fake motion."""

import base64
import json
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from unitree_demo.house_demo import (
    House, RETURN_STEP, STATUS_STEP, answer_peer, ask_back, handle_requester_input,
    handle_responder_input, perform, requester_story, responder_story, wait_reply, finish_story,
)
from unitree_demo.house_link import HouseInbox, Incoming, LadderStep
from unitree_demo.house_scene import Spot
from unitree_demo.trust_session import load_trust


class LocalLink:
    def __init__(self, short, key, registry, trust):
        self.protocol, policy = load_trust()
        self.robot_id = self.protocol.ROBOTS[short]
        self.peer_id = self.protocol.ROBOTS["b" if short == "a" else "a"]
        self.key = key
        self.inbox = HouseInbox(self.protocol.Verifier(self.robot_id, registry), policy.decide, trust, 0)
        self.raw = deque()
        self.sent = []
        self.peer = None
        self.drop = lambda message: False

    def transmit(self, message):
        self.sent.append(message)
        if self.peer is not None and not self.drop(message):
            self.peer.raw.append(json.dumps(message).encode())

    def request(self, step):
        message = self.protocol.make_message(self.key, self.robot_id, self.peer_id, step.request,
                                             action=step.action, reason=step.reason, kind="request")
        self.transmit(message)
        return message["nonce"], json.dumps(message).encode()

    def reply(self, message, result, text, phase):
        self.transmit(self.protocol.make_message(
            self.key, self.robot_id, self.peer_id, text, action=message["action"],
            reason=result, kind="response", reply_to=message["nonce"], result=result, phase=phase,
        ))

    def inject_forgery(self, step):
        self.transmit({"from": self.robot_id, "to": self.peer_id, "action": step.action})

    def inject_replay(self, payload):
        self.transmit(json.loads(payload))

    def event(self, *_args):
        pass

    def poll(self, now):
        items = []
        while self.raw:
            item = self.inbox.handle(json.loads(self.raw.popleft()), now)
            if item is not None:
                items.append(item)
        return items + self.inbox.due(now)


class FakeRobot:
    def __init__(self):
        self.arrived = True
        self.xy = np.zeros(2)
        self.visits = []

    def walk_to(self, spot):
        self.xy = spot.xy
        self.visits.append(spot)

    def position(self, _data):
        return self.xy


def make_pair(trust_b=55, event_log=None):
    protocol, _ = load_trust()
    keys = {short: Ed25519PrivateKey.generate() for short in ("a", "b")}
    registry = {protocol.ROBOTS[short]: {
        "public_key": base64.b64encode(key.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw)).decode(), "status": "active",
    } for short, key in keys.items()}
    # A third registered identity lets correlation be tested independently of authentication.
    registry["robot-c@house-3"] = registry[protocol.ROBOTS["b"]]
    links = [LocalLink(short, keys[short], registry, 55 if short == "a" else trust_b) for short in ("a", "b")]
    links[0].peer, links[1].peer = links[1], links[0]
    houses = []
    for index, link in enumerate(links, 1):
        layout = SimpleNamespace(house=f"house-{index}", spawn=Spot(0, 0, 0), door_spot=Spot(2, 0, 0))
        props = SimpleNamespace(parcel_position=lambda data: (2.25, -.5, .12))
        houses.append(House(SimpleNamespace(event_log=event_log if index == 1 else None), None,
                            SimpleNamespace(time=0.), FakeRobot(), props, layout, link))
    return houses


def drive_pair(houses, limit=200):
    stories = [requester_story(houses[0]), responder_story(houses[1])]
    for _ in range(int(limit * 10)):
        for house, story, handler in zip(houses, stories, (handle_requester_input, handle_responder_input)):
            for item in house.link.poll(house.now):
                handler(house, item)
            next(story)
            house.data.time += .1
        if houses[0].finished:
            return
    raise AssertionError("requester did not terminate")


def deliver_reply(house, peer, nonce, **changes):
    fields = {"action": house.outstanding.get(nonce, "status"), "kind": "response", "reply_to": nonce,
              "phase": "done", "result": "ALLOW"}
    fields.update(changes)
    sender = fields.pop("sender", peer.robot_id)
    message = peer.protocol.make_message(peer.key, sender, house.link.robot_id, "done", **fields)
    item = house.link.inbox.handle(message, house.now)
    assert item.kind == "response"
    handle_responder_input(house, item)
    return message


def test_signed_two_house_story_completes_return_and_records_evidence(tmp_path):
    path = tmp_path / "events.jsonl"
    a, b = make_pair(event_log=path)
    drive_pair([a, b])
    assert a.outcome == "COMPLETE"
    assert not a.failures and not b.failures
    assert any(stage == "RETURN CONFIRMED" for stage, _ in b.log)
    assert not a.outstanding and not b.outstanding
    assert a.layout.door_spot in a.robot.visits and b.layout.door_spot in b.robot.visits
    rejected = [e["reason"] for e in b.events if e["event"] == "authentication_rejected"]
    assert rejected == ["no signature", "replay"]
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len({e["run_id"] for e in records}) == 1
    assert any(e.get("evidence") == "peer_report" for e in records)
    assert any(e.get("evidence") == "simulator_ground_truth" for e in records)
    assert "Forgery and replay rejected" not in a.log[-1][1]


def test_discovery_timeout_fails_instead_of_claiming_complete():
    a, b = make_pair()
    a.link.drop = lambda message: True
    drive_pair([a, b])
    assert a.outcome == "FAILED"
    assert not a.outstanding and not a.replies
    assert a.robot.visits == []
    assert not any(e.get("stage") == "COMPLETE" for e in a.events)


@pytest.mark.parametrize("failure", ["inspection_timeout", "denied", "no_return"])
def test_failed_cooperation_never_finishes_green(failure):
    a, b = make_pair(trust_b=10 if failure == "denied" else 55)
    if failure == "inspection_timeout":
        b.link.drop = lambda m: m.get("kind") == "response" and m.get("action") == "verify_package"
    if failure == "no_return":
        b.link.drop = lambda m: m.get("kind") == "request"
    drive_pair([a, b])
    assert a.outcome == "FAILED"
    assert a.failures
    assert not any(e.get("stage") == "COMPLETE" for e in a.events)


@pytest.mark.parametrize("changes", [
    {"sender": "robot-c@house-3"}, {"action": "open_door"}, {"reply_to": "unrelated"},
    {"phase": "unknown"}, {"result": "SUCCESS"},
])
def test_authenticated_but_mismatched_replies_do_not_finish_request(changes):
    a, b = make_pair()
    nonce, _ = a.request(STATUS_STEP)
    deliver_reply(a, b.link, nonce, **changes)
    assert not a.replies
    valid = deliver_reply(a, b.link, nonce)
    with pytest.raises(StopIteration) as finished:
        next(wait_reply(a, nonce))
    assert finished.value.value == valid
    assert not a.outstanding and not a.replies


def test_ack_duplicate_and_late_reply_cannot_replace_final_reply():
    a, b = make_pair()
    nonce, _ = a.request(STATUS_STEP)
    deliver_reply(a, b.link, nonce, phase="ack")
    assert not a.replies
    final = deliver_reply(a, b.link, nonce)
    deliver_reply(a, b.link, nonce, result="DENY")
    assert a.replies[nonce] == final
    with pytest.raises(StopIteration):
        next(wait_reply(a, nonce))
    deliver_reply(a, b.link, nonce)
    assert not a.replies


def test_timeout_retires_correlation_and_late_reply_is_ignored():
    a, b = make_pair()
    nonce, _ = a.request(STATUS_STEP)
    wait = wait_reply(a, nonce, timeout=1)
    next(wait)
    a.data.time = 2
    with pytest.raises(StopIteration) as finished:
        next(wait)
    assert finished.value.value is None
    deliver_reply(a, b.link, nonce)
    assert not a.outstanding and not a.replies


@pytest.mark.parametrize("action", ["open_door", "turn_on_light", "mirror_motion", "deliver_parcel", "carry_together"])
def test_policy_approval_without_executor_is_denied_before_ack_or_motion(action):
    a, b = make_pair(trust_b=100)
    a.request(LadderStep("unsupported", action, "please", "test"))
    for item in b.link.poll(0):
        handle_responder_input(b, item)
    assert not b.inbox and not b.robot.visits
    assert [(m["result"], m["phase"]) for m in b.link.sent] == [("DENY", "done")]
    assert any(e["event"] == "policy_decision" and e["result"] == "ALLOW" for e in b.events)
    assert not any(e["event"] == "execution_done" for e in b.events)


def test_executor_defensively_refuses_unsupported_permission():
    a, _ = make_pair()
    message = {"nonce": "request", "action": "verify_and_unlock"}
    with pytest.raises(StopIteration) as finished:
        next(perform(a, message, "ALLOW"))
    assert finished.value.value is False
    assert not a.robot.visits
    assert a.link.sent[-1]["result"] == "DENY"


def test_unrelated_inbound_request_does_not_count_as_return_task():
    a, b = make_pair()
    b.request(STATUS_STEP)
    for item in a.link.poll(0):
        handle_requester_input(a, item)
    task = answer_peer(a, timeout=1)
    next(task)
    a.data.time = 2
    with pytest.raises(StopIteration) as finished:
        next(task)
    assert finished.value.value is False


def test_responder_records_matching_return_response():
    a, b = make_pair()
    task = ask_back(b, RETURN_STEP)
    next(task)
    nonce = next(iter(b.outstanding))
    deliver_reply(b, a.link, nonce)
    with pytest.raises(StopIteration):
        next(task)
    assert not b.failures
    assert b.log[-1][0] == "RETURN CONFIRMED"
    assert b.exit_status(requester=False) == 0


@pytest.mark.parametrize("failure", ["timeout", "denied"])
def test_responder_return_failure_sets_nonzero_exit_status(failure):
    a, b = make_pair()
    task = ask_back(b, RETURN_STEP)
    next(task)
    if failure == "timeout":
        b.data.time = 71
    else:
        deliver_reply(b, a.link, next(iter(b.outstanding)), result="DENY")
    with pytest.raises(StopIteration):
        next(task)
    assert b.failures and b.outcome == "FAILED"
    assert b.exit_status(requester=False) == 1
    assert not b.outstanding


@pytest.mark.parametrize("step,finished", [(STATUS_STEP, False), (RETURN_STEP, True)])
def test_requester_refuses_work_outside_its_script_before_ack(step, finished):
    a, b = make_pair()
    a.finished = finished
    b.request(step)
    for item in a.link.poll(0):
        handle_requester_input(a, item)
    assert not a.inbox
    assert [(m["result"], m["phase"]) for m in a.link.sent] == [("DENY", "done")]


def test_failed_story_finally_denies_queued_return_work_and_rejects_late_requests():
    a, b = make_pair()
    b.request(RETURN_STEP)
    for item in a.link.poll(0):
        handle_requester_input(a, item)
    assert len(a.inbox) == 1
    a.fail("discovery failed")
    next(finish_story(a))
    assert not a.inbox and not a.robot.visits
    assert [(m["result"], m["phase"]) for m in a.link.sent] == [("ALLOW", "ack"), ("DENY", "done")]
    b.request(RETURN_STEP)
    for item in a.link.poll(0):
        handle_requester_input(a, item)
    assert not a.inbox
    assert a.link.sent[-1]["result"] == "DENY"


def test_requester_admits_only_one_return_inspection():
    a, b = make_pair()
    first, _ = b.request(RETURN_STEP)
    b.request(RETURN_STEP)
    for item in a.link.poll(0):
        handle_requester_input(a, item)
    assert len(a.inbox) == 1
    assert a.return_request == first
    assert [(m["result"], m["phase"]) for m in a.link.sent] == [("ALLOW", "ack"), ("DENY", "done")]


def test_return_request_after_wait_timeout_is_denied_before_story_finishes():
    a, b = make_pair()
    task = answer_peer(a, timeout=1)
    next(task)
    a.data.time = 2
    with pytest.raises(StopIteration):
        next(task)
    assert not a.finished
    b.request(RETURN_STEP)
    for item in a.link.poll(a.now):
        handle_requester_input(a, item)
    assert not a.inbox
    assert [(m["result"], m["phase"]) for m in a.link.sent] == [("DENY", "done")]


def test_return_request_at_deadline_is_denied_before_generator_advances():
    a, b = make_pair()
    task = answer_peer(a, timeout=1)
    next(task)
    a.data.time = 1
    b.request(RETURN_STEP)
    for item in a.link.poll(a.now):
        handle_requester_input(a, item)
    assert not a.inbox
    assert [(m["result"], m["phase"]) for m in a.link.sent] == [("DENY", "done")]
    with pytest.raises(StopIteration):
        next(task)


def test_stopping_responder_during_decision_delay_is_not_success():
    a, b = make_pair()
    b.link.inbox.reaction_delay = 1
    a.request(RETURN_STEP)
    assert b.link.poll(0) == []
    assert not b.inbox and not b.outstanding
    assert b.has_pending_work and b.exit_status(requester=False) == 1


def test_requester_policy_denial_does_not_schedule_responder_only_work():
    a, b = make_pair()
    a.link.inbox.trust_score = 0
    b.request(RETURN_STEP)
    for item in a.link.poll(0):
        handle_requester_input(a, item)
    assert a.link.sent[-1]["result"] == "DENY"
    assert a.pending_return is None and not a.inbox


def test_stopping_responder_while_waiting_for_return_is_not_success():
    a, b = make_pair()
    task = ask_back(b, RETURN_STEP)
    next(task)
    assert not b.failures
    assert b.has_pending_work and b.exit_status(requester=False) == 1
    deliver_reply(b, a.link, next(iter(b.outstanding)))
    with pytest.raises(StopIteration):
        next(task)
    assert not b.has_pending_work and b.exit_status(requester=False) == 0


def test_stopping_responder_during_execution_or_return_walk_is_not_success(monkeypatch):
    a, b = make_pair()
    a.request(RETURN_STEP)
    for item in b.link.poll(0):
        handle_responder_input(b, item)
    assert b.has_pending_work and b.exit_status(requester=False) == 1
    message, result = b.inbox.popleft()
    walk_to = b.robot.walk_to
    def start_walk(spot):
        walk_to(spot)
        b.robot.arrived = False
    monkeypatch.setattr(b.robot, "walk_to", start_walk)
    task = perform(b, message, result)
    next(task)
    assert b.active_request == message["nonce"]
    assert b.exit_status(requester=False) == 1
    b.robot.arrived = True
    next(task)
    assert b.link.sent[-1]["phase"] == "done"
    assert b.robot.visits[-1] == b.layout.spawn
    assert b.has_pending_work and b.exit_status(requester=False) == 1
    b.robot.arrived = True
    with pytest.raises(StopIteration):
        next(task)
    assert not b.has_pending_work and b.exit_status(requester=False) == 0
