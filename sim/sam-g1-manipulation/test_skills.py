"""Skill helpers, the agent loop helpers and the activity log wording."""

from types import SimpleNamespace

import numpy as np
import pytest

from agent import SYSTEM, TOOLS, brief, compact, first_call_only
from skills import describe_size, grasp_height, object_spheres
from trace_format import format_message


@pytest.mark.parametrize("placement_ok", [True, False, None])
def test_move_object_preserves_placement_verification(monkeypatch, placement_ok):
    import skills

    placed = {"summary": "Placement checked", "stage": "verify", "suggestion": "Inspect again"}
    if placement_ok is not None:
        placed["ok"] = placement_ok

    class FakeSkills:
        def __init__(self, console, epoch):
            self.robot = SimpleNamespace(contact_mode=False)

        def pick_up(self, *args):
            return {"ok": True, "side": "left", "summary": "Picked up"}

        def place(self, *args):
            return placed

    monkeypatch.setattr(skills, "Skills", FakeSkills)
    monkeypatch.setattr(skills, "forget_dropped", lambda console: None)
    result = skills.execute(None, "move_object", {
        "frame_id": "frame-1", "object_box": [0, 0, 10, 10], "relation": "next_to",
    }, 1)
    assert result["ok"] is (placement_ok is True)
    assert result["place"] == placed
    if placement_ok is not True:
        assert result["stage"] == "verify"
        assert result["suggestion"] == "Inspect again"


def test_grasp_height_keeps_pads_off_the_table_and_under_the_top():
    tall = {"top_z_m": 0.83, "bottom_z_m": 0.73, "bottom_certain": True}
    assert grasp_height(tall) == pytest.approx(0.78)
    low = {"top_z_m": 0.79, "bottom_z_m": 0.73, "bottom_certain": True}
    assert 0.73 + 0.016 - 1e-9 <= grasp_height(low) <= 0.79 - 0.010 + 1e-9
    flat = {"top_z_m": 0.75, "bottom_z_m": 0.73, "bottom_certain": True}
    assert grasp_height(flat) == pytest.approx(0.74)  # too flat for both limits: middle
    unsure = {"top_z_m": 0.83, "bottom_z_m": 0.73, "bottom_certain": False}
    assert grasp_height(unsure) == pytest.approx(0.81)


def test_object_spheres_cover_an_upright_object():
    half = np.array([0.03, 0.03, 0.05])
    spheres = object_spheres(np.array([0.0, 0.0, 0.0]), half)
    lowest = min(c[2] - r for c, r in spheres)
    highest = max(c[2] + r for c, r in spheres)
    assert lowest <= -half[2] + 1e-9
    assert highest >= half[2] - 1e-9
    assert all(r >= np.linalg.norm(half[:2]) for _, r in spheres)
    cylinder = object_spheres(np.zeros(3), half, round_shape=True)
    assert all(half[0] <= r < np.linalg.norm(half[:2]) for _, r in cylinder)


def test_size_is_described_in_centimetres():
    assert describe_size({"half_size_m": [0.03, 0.03, 0.05]}) == "6×6×10 cm"


def test_every_action_tells_the_person_why_and_release_is_always_allowed():
    names = {tool["name"] for tool in TOOLS}
    assert {"observe", "move_object", "pick_up", "place", "release", "grip", "finish"} <= names
    for tool in TOOLS:
        assert "say" in tool["parameters"]["required"]
    release = next(tool for tool in TOOLS if tool["name"] == "release")
    assert "Always allowed" in release["description"]
    assert "Never refuse to release" in SYSTEM


def test_model_turn_keeps_only_its_first_action():
    content = {"parts": [{"text": "plan"}, {"functionCall": {"name": "a"}}, {"functionCall": {"name": "b"}}]}
    assert first_call_only(content)["parts"] == content["parts"][:2]


def test_results_sent_to_the_model_are_small():
    result = {"ok": True, "geometry": {"x": 1}, "path": list(range(40)), "error": 0.123456789}
    assert brief(result) == {"ok": True, "path": "[40 values]", "error": 0.1235}


def test_only_recent_images_are_resent():
    image = {"inlineData": {"data": "x"}}
    contents = [{"role": "user", "parts": [image, image]} for _ in range(4)]
    trimmed = compact(contents, keep_images=1)
    kept = sum("inlineData" in part for message in trimmed for part in message["parts"])
    assert kept == 2


def test_activity_is_plain_language_and_debug_has_details():
    row = {
        "role": "result",
        "text": "Could not place: The arm cannot reach",
        "ok": False,
        "details": {"ok": False, "suggestion": "Choose a spot nearer the edge", "stage": "reach"},
    }
    activity = format_message(row)
    assert "✗ Could not place" in activity and "Suggestion: Choose a spot nearer the edge" in activity
    assert "{" not in activity
    assert '"stage": "reach"' in format_message(row, detailed=True)
