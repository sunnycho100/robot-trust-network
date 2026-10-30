"""Measure the space the legs and torso sweep while GR00T-WBC steps.

The posture table holds the body standing still. A stepping leg swings its hip
and knee several centimetres beyond that, so walking checks use this envelope:
the union of the body surface, relative to the planted feet, over forward,
backward, sideways and turning steps at the speeds the skills use.

Run from this directory: uv run --no-sync python build_gait_envelope.py
"""

import json
from pathlib import Path

import numpy as np

from body import Body, feet_pose
from build_posture_table import body_surface, voxel_downsample
from controller import ManipulationRuntime

OUTPUT = Path(__file__).with_name("gait_envelope.json")
COMMANDS = (
    (0.12, 0.0, 0.0),
    (0.25, 0.0, 0.0),
    (-0.12, 0.0, 0.0),
    (0.0, 0.15, 0.0),
    (0.0, -0.15, 0.0),
    (0.09, 0.12, 0.0),
    (-0.09, -0.12, 0.0),
    (0.0, 0.0, 0.4),
    (0.0, 0.0, -0.4),
    (0.15, 0.0, 0.3),
)


class Console:
    """Just enough of the app console for Body to drive the robot."""

    def __init__(self, robot):
        self.robot = robot
        self.epoch = 0

    def tick(self, epoch=None):
        self.robot.tick()


def main():
    robot = ManipulationRuntime(scene="arena", seed=7)
    body = Body(Console(robot), 0)
    body.wait(1.0)
    points = []
    try:
        for command in COMMANDS:
            robot.velocity(*command)
            for tick in range(round(3.0 / robot.dt)):
                body.step()
                if tick > 25 and tick % 3 == 0:
                    x, y, yaw = feet_pose(robot)
                    c, s = np.cos(yaw), np.sin(yaw)
                    surface = body_surface(robot) - [x, y, 0]
                    local = surface @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
                    points.append(local)
            body.come_to_rest()
    finally:
        robot.close()
    envelope = voxel_downsample(np.concatenate(points), 0.02)
    OUTPUT.write_text(json.dumps({"surface_feet_m": np.round(envelope, 4).tolist(), "commands": COMMANDS}))
    print(f"{len(envelope)} points; x {envelope[:, 0].min():.2f}..{envelope[:, 0].max():.2f}, "
          f"y {envelope[:, 1].min():.2f}..{envelope[:, 1].max():.2f}")


if __name__ == "__main__":
    main()
