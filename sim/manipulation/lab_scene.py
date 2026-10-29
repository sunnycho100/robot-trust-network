"""Small physical pinch test, defined through BiGym's external task API.

The cube is free, with gravity/contact dynamics. No welds or attachment tricks.
This is our calibration scene, not a modification of the upstream benchmark.
"""

import mujoco
import numpy as np
from bigym.envs.reach_target import ReachTargetSingleG1
from bigym.loco.tasks import TaskSpec


class PinchScene(ReachTargetSingleG1):
    variant = "calibration"

    def _initialize_env(self):
        super()._initialize_env()
        self.table = self.spec.worldbody.add_geom(
            name="lab_table",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[0.46, 0.20, 0.70],
            size=[0.10, 0.17, 0.03],
            rgba=[0.4, 0.3, 0.2, 1],
            friction=[1, 0.005, 0.0001],
        )
        body = self.spec.worldbody.add_body(name="lab_cube", pos=[0.42, 0.15, 0.755])
        self.cube_body = body
        self.cube_joint = body.add_freejoint(name="lab_cube_joint")
        self.cube_geom = body.add_geom(
            name="lab_cube_geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.025, 0.025, 0.025],
            mass=0.05,
            rgba=[0.1, 0.6, 0.9, 1],
            friction=[1, 0.005, 0.0001],
        )
        self.objects = {
            "blue_cube": {
                "joint": self.cube_joint,
                "geom": self.cube_geom,
                "spawn": [0.42, 0.15, 0.755],
                "half_size": [0.025] * 3,
                "color": "blue",
                "shape": "cube",
                "mass_kg": 0.05,
                "pickable": True,
            }
        }
        self.fixtures = {"left_table": self.table}
        self.zones = {"left_pad": [0.42, 0.28, 0.755]}
        self.support_body = self.spec.worldbody.add_body(name="lab_support", mocap=True)
        self.support_weld = self.spec.add_equality(
            name="lab_support_weld",
            type=mujoco.mjtEq.mjEQ_WELD,
            objtype=mujoco.mjtObj.mjOBJ_BODY,
            name1=self.robot.pelvis.name,
            name2="lab_support",
            active=False,
            data=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
        )
        if self.variant != "calibration":
            self._build_test_scene()

    def _box(self, name, position, half_size, color):
        geom = self.spec.worldbody.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=position,
            size=half_size,
            rgba=color,
            friction=[1, 0.005, 0.0001],
        )
        self.fixtures[name] = geom
        return geom

    def _object(
        self, name, position, half_size, color_name, rgba, shape="box", mass=0.1
    ):
        body = self.spec.worldbody.add_body(name=name, pos=position)
        joint = body.add_freejoint(name=name + "_joint")
        geom = body.add_geom(
            name=name + "_geom",
            type={
                "box": mujoco.mjtGeom.mjGEOM_BOX,
                "cube": mujoco.mjtGeom.mjGEOM_BOX,
                "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
                "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
            }[shape],
            size=half_size,
            mass=mass,
            rgba=rgba,
            friction=[1, 0.005, 0.0001],
        )
        self.objects[name] = {
            "joint": joint,
            "geom": geom,
            "spawn": position,
            "half_size": half_size,
            "color": color_name,
            "shape": shape,
            "mass_kg": mass,
            "pickable": shape == "cube" and half_size == [0.025] * 3,
        }

    def _build_test_scene(self):
        # Clear space around the calibrated grasp; varied distractors live on
        # separate supports. These are physical geometry, not screen overlays.
        self._box(
            "right_table",
            [0.65, -0.36, 0.70],
            [0.28, 0.27, 0.03],
            [0.30, 0.34, 0.38, 1],
        )
        self._box(
            "rear_platform",
            [1.22, 0.48, 0.39],
            [0.24, 0.24, 0.04],
            [0.47, 0.39, 0.29, 1],
        )
        self._object(
            "red_cube",
            [0.49, -0.22, 0.755],
            [0.025] * 3,
            "red",
            [0.9, 0.12, 0.1, 1],
            "cube",
            0.05,
        )
        self._object(
            "green_box",
            [0.76, -0.48, 0.78],
            [0.08, 0.045, 0.05],
            "green",
            [0.15, 0.7, 0.28, 1],
        )
        self._object(
            "yellow_cylinder",
            [0.76, -0.23, 0.78],
            [0.035, 0.05, 0],
            "yellow",
            [0.95, 0.75, 0.1, 1],
            "cylinder",
        )
        self._object(
            "purple_ball",
            [0.50, -0.49, 0.765],
            [0.035, 0, 0],
            "purple",
            [0.6, 0.2, 0.8, 1],
            "sphere",
            0.06,
        )
        self._object(
            "orange_box",
            [1.22, 0.48, 0.49],
            [0.085, 0.065, 0.06],
            "orange",
            [0.95, 0.4, 0.08, 1],
            mass=0.2,
        )
        self._object(
            "large_brown_box",
            [1.45, -0.95, 0.18],
            [0.22, 0.18, 0.18],
            "brown",
            [0.48, 0.28, 0.13, 1],
            mass=2,
        )
        self.zones.update(
            {
                "right_pad": [0.59, -0.36, 0.755],
                "rear_platform": [1.22, 0.48, 0.455],
                "floor_east": [1.0, 0, 0],
                "floor_north": [0, 1.0, 0],
            }
        )
        for name, xyz, rgba, size in (
            (
                "left_pad",
                [0.42, 0.28, 0.731],
                [0.1, 0.7, 0.7, 1],
                [0.045, 0.045, 0.0005],
            ),
            (
                "right_pad",
                [0.59, -0.36, 0.731],
                [0.9, 0.8, 0.65, 1],
                [0.05, 0.05, 0.0005],
            ),
            ("east_marker", [1, 0, 0.001], [0.1, 0.55, 0.6, 1], [0.18, 0.18, 0.0005]),
            ("north_marker", [0, 1, 0.001], [0.7, 0.3, 0.65, 1], [0.18, 0.18, 0.0005]),
        ):
            marker = self._box(name, xyz, size, rgba)
            marker.contype = marker.conaffinity = 0
        if self.variant == "clutter":
            self._box(
                "occluding_panel",
                [0.39, -0.36, 0.81],
                [0.025, 0.12, 0.08],
                [0.55, 0.57, 0.59, 1],
            )
            self._object(
                "white_box",
                [0.86, -0.36, 0.76],
                [0.035, 0.035, 0.03],
                "white",
                [0.9, 0.9, 0.85, 1],
            )
        if self.variant == "arena":
            # Move tabletop stations out of the walking lane at scene creation.
            for name, geom in self.fixtures.items():
                if name not in ("east_marker", "north_marker"):
                    geom.pos[0] += 1.5
            for obj in self.objects.values():
                obj["spawn"] = [obj["spawn"][0] + 1.5, *obj["spawn"][1:]]
            for name in ("left_pad", "right_pad", "rear_platform"):
                self.zones[name][0] += 1.5
            self._box(
                "west_obstacle",
                [-1.25, -0.85, 0.20],
                [0.20, 0.25, 0.20],
                [0.65, 0.45, 0.25, 1],
            )

    def _on_reset(self):
        super()._on_reset()
        for obj in self.objects.values():
            joint = self.data.bind(obj["joint"])
            x, y, z = obj["spawn"]
            jitter = 0.008 if obj["pickable"] else 0.015
            joint.qpos[:] = [
                x + np.random.uniform(-jitter, jitter),
                y + np.random.uniform(-jitter, jitter),
                z,
                1,
                0,
                0,
                0,
            ]
            joint.qvel[:] = 0
        # The inherited non-colliding reach marker is unnecessary in this scene.
        self.model.bind(self.targets[0].geom).rgba[3] = 0

    def _success(self):
        # The controller validates contact, lift, release, and placement itself.
        return False


# The lab subclasses BiGym's reach task, which disables GR00T-WBC's torso-pitch
# command. Leaning is how the robot reaches the middle of a low surface, so the
# lab enables it, as BiGym 2.0 does for 14 of its 20 official tasks.
TORSO_PITCH = {"controller": {"pitch_command": True}}

PINCH = TaskSpec(
    PinchScene,
    episode_length=100000,
    overrides=TORSO_PITCH,
)


class SortingScene(PinchScene):
    variant = "sorting"


class ClutterScene(PinchScene):
    variant = "clutter"


class ArenaScene(PinchScene):
    variant = "arena"


SORTING = TaskSpec(
    SortingScene,
    episode_length=100000,
    overrides=TORSO_PITCH,
)
CLUTTER = TaskSpec(
    ClutterScene,
    episode_length=100000,
    overrides=TORSO_PITCH,
)
ARENA = TaskSpec(
    ArenaScene,
    episode_length=100000,
    overrides=TORSO_PITCH,
)


def scene_state(robot):
    """Evaluator/assisted-mode truth. Never feed this to image-only trials."""
    inner = robot.tools.inner
    return {
        "preset": inner.variant,
        "seed": inner.seed,
        "zones": inner.zones,
        "surfaces": {
            name: {
                "description": {
                    "left_table": "small brown table in front of the left hand",
                    "right_table": "large grey table on the robot's right",
                    "rear_platform": "separate tan rear platform",
                }[name],
                "center": list(geom.pos),
                "half_size": list(geom.size),
                "top_z": float(geom.pos[2] + geom.size[2]),
                "bounds_xy": [
                    [
                        float(geom.pos[i] - geom.size[i]),
                        float(geom.pos[i] + geom.size[i]),
                    ]
                    for i in (0, 1)
                ],
            }
            for name, geom in inner.fixtures.items()
            if name in ("left_table", "right_table", "rear_platform")
        },
        "objects": {
            name: {
                "position": robot.tools.data.bind(obj["joint"]).qpos[:3].tolist(),
                "color": obj["color"],
                "shape": obj["shape"],
                "half_size": obj["half_size"],
                "mass_kg": obj["mass_kg"],
                "pickable_geometry": obj["pickable"],
            }
            for name, obj in inner.objects.items()
        },
    }
