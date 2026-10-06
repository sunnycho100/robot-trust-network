import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from unitree_demo.house_demo import (
    ack_text, done_text, is_final_reply, package_in_reach, responder_task,
)
from unitree_demo.house_link import LADDER, HouseInbox
from unitree_demo.house_scene import Spot, layout_from_manifest, nav_command, wrap_angle
from test_trust_session import _peer


MANIFEST = {
    "house": "house-2",
    "xml": "house-2.xml",
    "fixtures": [
        {"name": "floor_room", "class": "Floor", "pos": [2.75, -1.5, -0.02], "size": [1.54, 2.79, 0.02]},
        {"name": "wall_right_room", "class": "Wall", "pos": [5.52, -1.5, 1.5], "size": [1.54, 1.5, 0.02]},
    ],
}


def test_layout_spots_sit_on_the_open_floor_inside_the_room():
    layout = layout_from_manifest(MANIFEST, Path("/tmp"))
    assert layout.xml_path == Path("/tmp/house-2.xml")
    for spot in (layout.spawn, layout.door_spot):
        assert 0.2 < spot.x < 5.3
        assert -2.8 < spot.y < -0.7
    assert layout.door_spot.yaw == 0.0
    assert layout.front_door_xy[0] == pytest.approx(5.49)
    assert layout.parcel_xy[0] < layout.front_door_xy[0]


def test_navigation_reaches_the_spot_and_faces_its_heading():
    position, yaw = np.array([2.8, -1.85]), math.pi / 2
    target = Spot(4.06, -1.25, math.pi / 2)
    dt = 0.02
    for _ in range(3000):
        command, arrived = nav_command(position, yaw, target, 0.45, 0.7)
        if arrived:
            break
        yaw += command[2] * dt
        position = position + command[0] * dt * np.array([math.cos(yaw), math.sin(yaw)])
    assert arrived
    assert np.hypot(*(target.xy - position)) < 0.15
    assert abs(wrap_angle(target.yaw - yaw)) < 0.15


def test_navigation_near_the_spot_turns_in_place_instead_of_walking_again():
    command, arrived = nav_command((4.3, -1.25), math.pi / 2 - 0.5, Spot(4.06, -1.25, math.pi / 2), 0.45, 0.7, near=True)
    assert not arrived
    assert command[0] == 0.0
    assert command[2] > 0.0


def _inbox(trust):
    protocol, policy, leader, robot_a, robot_b, verifier = _peer()
    return protocol, leader, robot_a, robot_b, HouseInbox(verifier, policy.decide, trust, reaction_delay=0.9)


def test_trust_55_walks_the_whole_ladder():
    protocol, leader, robot_a, robot_b, inbox = _inbox(55)
    results = {}
    sent = {}
    for step in LADDER:
        if step.replay_of:
            item = inbox.handle(sent[step.replay_of], now=2.0)
            assert item.kind == "rejected"
            assert item.reason == "replay"
            results[step.key] = "REJECTED"
            continue
        if not step.signed:
            forged = {"from": robot_a, "to": robot_b, "action": step.action, "request": step.request}
            item = inbox.handle(forged, now=0.0)
            assert item.kind == "rejected"
            assert item.reason == "no signature"
            results[step.key] = "REJECTED"
            continue
        message = protocol.make_message(leader, robot_a, robot_b, step.request,
                                        action=step.action, reason=step.reason, kind="request")
        sent[step.key] = message
        assert inbox.handle(message, now=0.0) is None
        assert inbox.deciding
        assert inbox.due(0.5) == []
        (item,) = inbox.due(0.9)
        results[step.key] = item.decision.result
    assert results == {
        "inspect": "ALLOW",
        "partial": "PARTIAL",
        "attack": "REJECTED",
        "replay": "REJECTED",
    }


def test_a_signed_neighbor_can_look_but_cannot_unlock_even_at_full_trust():
    protocol, leader, robot_a, robot_b, inbox = _inbox(100)
    for action, expected in (("verify_package", "ALLOW"), ("verify_and_unlock", "PARTIAL"), ("unlock_door", "DENY")):
        message = protocol.make_message(leader, robot_a, robot_b, "please", action=action, reason="", kind="request")
        inbox.handle(message, now=0.0)
        (item,) = inbox.due(1.0)
        assert item.decision.result == expected


def test_signed_replies_are_final_only_when_done_or_denied():
    assert not is_final_reply({"phase": "ack", "result": "ALLOW"})
    assert is_final_reply({"phase": "done", "result": "ALLOW"})
    assert is_final_reply({"phase": "ack", "result": "DENY"})
    assert ack_text("DENY", "blocked by owner safety rule") == "No. blocked by owner safety rule."
    assert "will not unlock" in ack_text("PARTIAL", "")
    assert done_text("verify_package", "ALLOW", True) == "Package found at my front door."
    assert "stays locked" in done_text("verify_and_unlock", "PARTIAL", True)


class _FakeHouse:
    def __init__(self):
        self.t = 0.0
        self.layout = layout_from_manifest(MANIFEST, Path("/tmp"))
        self.robot = SimpleNamespace(
            kind="g1", arrived=True, walk_to=lambda spot: None,
            position=lambda _data: self.layout.door_spot.xy,
        )
        self.props = SimpleNamespace(parcel_position=lambda _data: (*self.layout.parcel_xy, 0.14))
        self.data = None
        self.package_found = False

    @property
    def now(self):
        return self.t


def _drive(house, generator, dt=0.02, limit=20.0):
    for _ in generator:
        house.t += dt
        if house.t > limit:
            raise AssertionError("task did not finish")


def test_standing_at_the_door_counts_as_finding_the_parcel():
    house = _FakeHouse()
    assert package_in_reach(house.layout.door_spot.xy, house.layout.parcel_xy)
    assert not package_in_reach(house.layout.spawn.xy, house.layout.parcel_xy)
    _drive(house, responder_task(house, "verify_and_unlock", "PARTIAL"))
    assert house.package_found
