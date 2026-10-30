"""Episodic memory from camera measurements, never simulator object state.

Locations are immutable observations before a move, not a claim that an object
is still there. Reset clears the episode. Reusing a location requires fresh depth.
"""

import copy

import numpy as np


def appearance_distance(first, second):
    """Compare chromatic direction under changing neutral illumination.

    Grey/specular mixing changes saturation across cameras. Normalizing the
    distance from grey preserves hue; low-chroma regions remain conservative.
    """
    a, b = np.asarray(first) - 1 / 3, np.asarray(second) - 1 / 3
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if min(na, nb) < 0.08:
        return float(np.linalg.norm(a - b) * 2)
    return float(np.linalg.norm(a / na - b / nb))


def profile_distance(first, second):
    """Compare only appearance samples actually observed on each geometry."""
    return min(
        appearance_distance(a, b)
        for a in first.get("appearance_samples", [first["appearance_chroma"]])
        for b in second.get("appearance_samples", [second["appearance_chroma"]])
    )


class SceneMemory:
    def __init__(self):
        self.moves = []
        self.counter = 0

    def get(self, memory_id):
        for row in self.moves:
            if row["memory_id"] == memory_id:
                return copy.deepcopy(row)
        raise ValueError(
            "Unknown or expired camera memory; do not guess a former location"
        )

    def remember(self, geometry, description, frame_id, request):
        self.counter += 1
        row = {
            "memory_id": f"move-{self.counter}",
            "description": str(description)[:160],
            "request": str(request)[:500],
            "frame_id": str(frame_id),
            "before": copy.deepcopy(geometry),
            "status": "observed_before_attempt",
        }
        self.moves.append(row)
        self.moves = self.moves[-24:]
        return row["memory_id"]

    def mark(self, memory_id, status):
        for row in self.moves:
            if row["memory_id"] == memory_id:
                row["status"] = status

    def compatible_prior(self, geometry, memory_id=None):
        """Require a unique appearance and footprint match, not a colour name."""
        candidates = [self.get(memory_id)] if memory_id else list(reversed(self.moves))
        matches = []
        for row in candidates:
            old = row["before"]
            chroma_error = profile_distance(old, geometry)
            extent_error = np.max(
                np.abs(np.asarray(old["half_size_m"])[:2] - geometry["half_size_m"][:2])
            )
            if chroma_error < 0.15 and extent_error < 0.012:
                matches.append(row)
        if memory_id and not matches:
            raise ValueError(
                "Selected object does not match the remembered appearance/size"
            )
        if not matches:
            return None
        # Similar-looking instances with inconsistent heights cannot establish identity.
        heights = [row["before"]["half_size_m"][2] for row in matches]
        if max(heights) - min(heights) > 0.008:
            raise ValueError("Object memory is ambiguous; inspect another view")
        return copy.deepcopy(matches[0]["before"])

    def context(self):
        return [
            {
                "memory_id": r["memory_id"],
                "description": r["description"],
                "request": r["request"],
                "status": r["status"],
                "before_center_from_camera_m": r["before"]["center_world_m"],
                "measured_size_m": (
                    2 * np.asarray(r["before"]["half_size_m"])
                ).tolist(),
                "frame_id": r["frame_id"],
            }
            for r in self.moves[-12:]
            if r["status"] in ("lift_observed", "release_executed")
        ]


def memory_for(console):
    if not hasattr(console, "scene_memory"):
        console.scene_memory = SceneMemory()
    return console.scene_memory
