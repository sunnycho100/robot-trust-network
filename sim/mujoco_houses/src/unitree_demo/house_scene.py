"""A RoboCasa kitchen with one walking robot, a front door and a parcel.

The kitchen MJCF is exported once by ``scripts/setup_robocasa_houses``. This module
adds the demo props and the robot, and holds the navigation helper the house
demo uses every control tick.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from .go1_policy import load_go1_spec, set_policy_gains


HOUSES_DIR = Path(__file__).resolve().parents[2] / "assets" / "houses"
ROBOT_PREFIX = "robot_"
ROBOT_TINT = {"house-1": np.array([0.95, 0.35, 0.22]), "house-2": np.array([0.20, 0.55, 1.00])}
MIN_TURN = 0.45
LOCKED_RGBA = [0.15, 0.85, 0.35, 1]
ALARM_RGBA = [1.0, 0.15, 0.1, 1]


@dataclass(frozen=True)
class Spot:
    x: float
    y: float
    yaw: float

    @property
    def xy(self) -> np.ndarray:
        return np.array([self.x, self.y])


@dataclass(frozen=True)
class HouseLayout:
    house: str
    xml_path: Path
    spawn: Spot
    door_spot: Spot
    parcel_xy: tuple[float, float]
    front_door_xy: tuple[float, float]
    room_center: tuple[float, float]


def _fixture(manifest: dict, prefix: str) -> dict:
    for fixture in manifest["fixtures"]:
        if fixture["name"].startswith(prefix):
            return fixture
    raise RuntimeError(f"{manifest['house']} has no fixture named {prefix}*")


def layout_from_manifest(manifest: dict, houses_dir: Path = HOUSES_DIR) -> HouseLayout:
    """Turn exported fixture poses into the spots each robot walks to."""
    floor = _fixture(manifest, "floor_room")
    fx, fy = floor["pos"][:2]
    front_y = fy - floor["size"][0]
    # The front door sits in the right-hand wall, which stays visible to the camera.
    side_wall_x = _fixture(manifest, "wall_right_room")["pos"][0]
    door_xy = (side_wall_x - 0.03, front_y + 0.85)
    return HouseLayout(
        house=manifest["house"],
        xml_path=houses_dir / manifest["xml"],
        spawn=Spot(fx + 0.05, fy - 0.35, math.pi / 2),
        door_spot=Spot(door_xy[0] - 0.65, door_xy[1], 0.0),
        parcel_xy=(door_xy[0] - 0.95, door_xy[1]),
        front_door_xy=door_xy,
        room_center=(fx, fy),
    )


def load_layout(house: str, houses_dir: Path = HOUSES_DIR) -> HouseLayout:
    path = houses_dir / f"{house}.json"
    if not path.is_file():
        raise RuntimeError(
            f"{path} is missing. Run ./scripts/setup_robocasa_houses once to export the RoboCasa kitchens."
        )
    return layout_from_manifest(json.loads(path.read_text()), houses_dir)


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def nav_command(position, yaw, target: Spot, max_speed: float, max_turn: float, near: bool = False):
    """Walk to a spot, then face its heading. Returns (vx, vy, wz) and whether it arrived.

    ``near`` widens the radius once the robot has reached the spot, so the small
    drift of turning in place does not send it walking again.
    """
    delta = target.xy - np.asarray(position, dtype=float)[:2]
    distance = float(np.hypot(*delta))
    if distance > (0.35 if near else 0.15):
        error = wrap_angle(math.atan2(delta[1], delta[0]) - yaw)
        turn = float(np.clip(1.8 * error, -max_turn, max_turn))
        alignment = max(0.0, 1.0 - abs(error) / 0.9)
        speed = max_speed * alignment * min(1.0, 0.25 + distance / 0.6)
        return np.array([speed, 0.0, turn], dtype=np.float32), False
    error = wrap_angle(target.yaw - yaw)
    if abs(error) < 0.15:
        return np.zeros(3, dtype=np.float32), True
    # The walk policies barely turn on very small yaw commands, so keep a floor.
    turn = math.copysign(min(max(1.6 * abs(error), MIN_TURN), max_turn), error)
    return np.array([0.0, 0.0, turn], dtype=np.float32), False


def _quat_from_yaw(yaw: float) -> list[float]:
    return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


def _add_props(spec, layout: HouseLayout) -> None:
    world = spec.worldbody
    dx, dy = layout.front_door_xy
    # Local +y faces into the room. Props are visual only.
    door = world.add_body(name="front_door", pos=[dx, dy, 0], quat=_quat_from_yaw(math.pi / 2))
    for name, kind, size, pos, rgba in (
        ("frame", mujoco.mjtGeom.mjGEOM_BOX, [0.52, 0.02, 1.08], [0, 0, 1.08], [0.92, 0.92, 0.9, 1]),
        ("panel", mujoco.mjtGeom.mjGEOM_BOX, [0.45, 0.03, 1.0], [0, 0.01, 1.0], [0.42, 0.27, 0.16, 1]),
        ("knob", mujoco.mjtGeom.mjGEOM_SPHERE, [0.035, 0, 0], [-0.33, 0.07, 1.0], [0.85, 0.75, 0.35, 1]),
        ("lock", mujoco.mjtGeom.mjGEOM_BOX, [0.06, 0.02, 0.035], [-0.33, 0.06, 1.22], LOCKED_RGBA),
    ):
        door.add_geom(name=f"front_door_{name}", type=kind, size=size, pos=pos, rgba=rgba,
                      contype=0, conaffinity=0, group=1)
    parcel = world.add_body(name="parcel", pos=[layout.parcel_xy[0], layout.parcel_xy[1], 0.12])
    parcel.add_geom(name="parcel_box", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.17, 0.13, 0.12],
                    rgba=[0.72, 0.55, 0.33, 1], contype=0, conaffinity=0, group=1)
    parcel.add_geom(name="parcel_tape", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.172, 0.03, 0.122],
                    rgba=[0.9, 0.85, 0.7, 1], contype=0, conaffinity=0, group=1)


def build_house_model(layout: HouseLayout, robot_kind: str, g1_xml: Path) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(layout.xml_path))
    _add_props(spec, layout)
    if robot_kind == "g1":
        robot = mujoco.MjSpec.from_file(str(g1_xml))
    elif robot_kind == "go1":
        robot = load_go1_spec()
    else:
        raise ValueError(f"Unsupported robot kind: {robot_kind}")
    frame = spec.worldbody.add_frame(pos=[layout.spawn.x, layout.spawn.y, 0.0], quat=_quat_from_yaw(layout.spawn.yaw))
    spec.attach(robot, prefix=ROBOT_PREFIX, frame=frame)
    model = spec.compile()
    tint = ROBOT_TINT.get(layout.house, np.array([0.8, 0.8, 0.8]))
    for geom_id, body_id in enumerate(model.geom_bodyid):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(body_id)) or ""
        if name.startswith(ROBOT_PREFIX):
            model.geom_rgba[geom_id, :3] = 0.3 * model.geom_rgba[geom_id, :3] + 0.7 * tint
        elif name.startswith("wall_front") or name.startswith("wall_left"):
            # Dollhouse view: the camera looks in through the missing walls.
            model.geom_rgba[geom_id, 3] = 0.0
        elif model.geom_group[geom_id] == 0:
            # RoboCasa's red collision hulls sit on the visual meshes and z-fight with them.
            # The robots follow fixed paths, so only the floor needs contacts.
            model.geom_rgba[geom_id, 3] = 0.0
            if not name.startswith("floor"):
                model.geom_contype[geom_id] = 0
                model.geom_conaffinity[geom_id] = 0
    if robot_kind == "go1":
        set_policy_gains(model, ROBOT_PREFIX)
    return model


class HouseProps:
    """The front-door lock that blinks when an attack is rejected, and the parcel."""

    def __init__(self, model) -> None:
        self.model = model
        self.lock_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "front_door_lock")
        self.parcel_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "parcel")

    def set_lock_alarm(self, alarm: bool) -> None:
        self.model.geom_rgba[self.lock_geom] = ALARM_RGBA if alarm else LOCKED_RGBA

    def parcel_position(self, data) -> np.ndarray:
        return data.xpos[self.parcel_body].copy()
