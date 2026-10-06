import time

import numpy as np
import pytest

from unitree_demo.duo import COMMAND_LIMITS, LeaderMessage, follower_command, select_delayed_message, story_beat
from unitree_demo.go1_policy import go1_observation


def _leader_at(x, command=(0.0, 0.0, 0.0)):
    return LeaderMessage(1, time.time_ns(), np.array([x, 0.0], dtype=np.float32), 0.0,
                         np.asarray(command, dtype=np.float32))


def test_mqtt_leader_message_round_trip():
    original = LeaderMessage(42, time.time_ns(), np.array([1.0, -2.0], dtype=np.float32), 0.25,
                             np.array([0.3, -0.1, 0.2], dtype=np.float32))
    decoded = LeaderMessage.from_payload(original.to_payload())
    assert decoded.sequence == original.sequence
    assert decoded.sent_at_ns == original.sent_at_ns
    assert np.allclose(decoded.position_xy, original.position_xy)
    assert np.allclose(decoded.command, original.command)


def test_mqtt_payload_rejects_wrong_identity():
    with pytest.raises(ValueError, match="schema or robot_id"):
        LeaderMessage.from_payload(b'{"schema":"unitree.formation.v1","robot_id":"intruder"}')


def test_delayed_message_selects_an_older_packet_while_stream_is_fresh():
    first, newest = object(), object()
    selected, age = select_delayed_message([(first, 9.0), (newest, 9.8)], now=10.0, delay=0.5, max_age=0.5)
    assert selected is first
    assert age == pytest.approx(1.0)


def test_delayed_message_fails_closed_when_stream_is_stale():
    selected, age = select_delayed_message([(object(), 9.0)], now=10.0, delay=0.5, max_age=0.5)
    assert selected is None
    assert age == pytest.approx(float("inf"))


def test_story_progresses_from_discovery_to_cooperation_and_stop():
    assert "DISCOVER" in story_beat(0).name
    assert story_beat(2.5).action == "mirror_motion"
    assert "IMITATE" in story_beat(6).name
    assert story_beat(10).attack
    assert not np.allclose(story_beat(10).leader_command, 0)
    assert story_beat(12).action == "carry_together"
    assert np.allclose(story_beat(16).leader_command, 0)


def test_follower_holds_when_formation_is_settled():
    assert np.allclose(follower_command(_leader_at(1.2), np.array([0.0, 0.0]), 0.0, 1.2), 0.0)


def test_follower_moves_toward_formation_slot():
    assert follower_command(_leader_at(2.0), np.array([0.0, 0.0]), 0.0, 1.2)[0] > 0


def test_go1_follower_may_walk_faster_than_a_g1_to_catch_up():
    far = _leader_at(5.0)
    g1 = follower_command(far, np.array([0.0, 0.0]), 0.0, 1.2, COMMAND_LIMITS["g1"])
    go1 = follower_command(far, np.array([0.0, 0.0]), 0.0, 1.2, COMMAND_LIMITS["go1"])
    assert g1[0] == pytest.approx(COMMAND_LIMITS["g1"][0])
    assert go1[0] == pytest.approx(COMMAND_LIMITS["go1"][0])


def test_go1_observation_matches_the_published_policy_layout():
    observation = go1_observation(
        linvel=(0.1, 0.0, 0.0), gyro=(0.0, 0.0, 0.2), imu_xmat=np.eye(3),
        joint_position=np.zeros(12), joint_velocity=np.zeros(12), last_action=np.zeros(12),
        command=(0.7, 0.0, 0.0),
    )
    assert observation.shape == (48,)
    assert observation[45] == np.float32(0.7)
    assert observation[8] == np.float32(-1.0)
