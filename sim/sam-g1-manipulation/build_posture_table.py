"""Measure GR00T-WBC postures once and store them for whole-body planning.

For each (pelvis height, torso pitch) command this records, relative to the
planted feet: the pelvis pose, the balance-owned joint angles, the head camera
pose, and the robot's own leg/torso collision surface. Planning then predicts a
posture's reach and clearance with one body model instead of guessing.

Run from this directory: uv run --no-sync python build_posture_table.py
"""

import json
from pathlib import Path

import mujoco
import numpy as np

from controller import ManipulationRuntime

HEIGHTS = (0.78, 0.74, 0.70, 0.66, 0.62, 0.58, 0.54, 0.50)
PITCHES = (0.0, 0.2, 0.4, 0.6, 0.8)
ARM_PARTS = ("shoulder", "elbow", "wrist", "dex1", "hand", "rubber")
OUTPUT = Path(__file__).with_name("posture_table.json")


def yaw_of(rotation):
    return float(np.arctan2(rotation[1, 0], rotation[0, 0]))


def feet_frame(robot):
    model, data = robot.tools.model, robot.tools.data
    namespace = robot.tools.robot.namespace
    feet = [model.body(namespace + f"{s}_ankle_roll_link").id for s in ("left", "right")]
    midpoint = np.mean(data.xpos[feet], axis=0)
    forward = np.mean(data.xmat[feet].reshape(2, 3, 3)[:, :, 0], axis=0)
    yaw = float(np.arctan2(forward[1], forward[0]))
    return midpoint, yaw


def heading(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def body_surface(robot):
    """Leg, pelvis, waist, torso and head collision surface, no arms or hands."""
    model, data = robot.tools.model, robot.tools.data
    namespace = robot.tools.robot.namespace
    points = []
    for body in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or ""
        if not name.startswith(namespace):
            continue
        if any(part in name.split("/")[-1] for part in ARM_PARTS):
            continue
        for geom in range(
            model.body_geomadr[body], model.body_geomadr[body] + model.body_geomnum[body]
        ):
            if not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
                continue
            rotation = data.geom_xmat[geom].reshape(3, 3)
            if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
                mesh = model.geom_dataid[geom]
                start = model.mesh_vertadr[mesh]
                local = model.mesh_vert[start : start + model.mesh_vertnum[mesh]]
            else:
                grid = np.linspace(-1, 1, 5)
                local = (
                    np.stack(np.meshgrid(grid, grid, grid), -1).reshape(-1, 3)
                    * model.geom_size[geom]
                )
            points.append(data.geom_xpos[geom] + local @ rotation.T)
    return np.concatenate(points)


def voxel_downsample(points, size=0.02):
    keys = np.floor(points / size).astype(int)
    _, index = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(index)]


def settle(robot, minimum=2.5, maximum=8.0):
    quiet = 0.0
    previous = np.asarray(robot.tools.pelvis_pose()[0])
    for step in range(round(maximum / robot.dt)):
        robot.tick()
        position = np.asarray(robot.tools.pelvis_pose()[0])
        speed = np.linalg.norm(position - previous) / robot.dt
        previous = position
        quiet = quiet + robot.dt if speed < 0.01 else 0.0
        if step * robot.dt >= minimum and quiet >= 0.5:
            return True
    return False


def measure(robot, height, pitch):
    feet, yaw = feet_frame(robot)
    frame = heading(yaw)
    position, quaternion = robot.tools.pelvis_pose()
    yaw_quaternion = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
    relative = np.empty(4)
    inverse = np.empty(4)
    mujoco.mju_negQuat(inverse, yaw_quaternion)
    mujoco.mju_mulQuat(relative, inverse, np.asarray(quaternion, dtype=float))
    camera = robot.tools._cam_id("head")
    data = robot.tools.data
    surface = voxel_downsample(body_surface(robot))
    local_surface = np.column_stack(
        ((surface[:, :2] - feet[:2]) @ frame[:2, :2], surface[:, 2])
    )
    return {
        "height_command": height,
        "pitch_command": pitch,
        "pelvis_offset_feet_m": (frame.T @ (np.asarray(position) - feet))[:2].tolist(),
        "pelvis_z_m": float(position[2]),
        "pelvis_quaternion_feet_wxyz": relative.tolist(),
        "balance_joints_rad": {
            name: float(data.qpos[joint["qpos"]])
            for name, joint in robot.joints.items()
            if joint["owner"] == "balance"
        },
        "head_camera_feet": {
            "position_m": np.r_[
                frame.T[:2, :2] @ (data.cam_xpos[camera][:2] - feet[:2]),
                data.cam_xpos[camera][2],
            ].tolist(),
            "forward": (frame.T @ -data.cam_xmat[camera].reshape(3, 3)[:, 2]).tolist(),
        },
        "body_surface_feet_m": np.round(local_surface, 4).tolist(),
    }


def main():
    robot = ManipulationRuntime(scene="arena", seed=7)
    for _ in range(100):
        robot.tick()
    entries = []
    for height in HEIGHTS:
        for pitch in PITCHES:
            robot.posture(height, pitch)
            try:
                settled = settle(robot)
            except RuntimeError as exc:
                print(f"h={height:.2f} p={pitch:.1f}: FELL ({exc})")
                robot.close()
                robot = ManipulationRuntime(scene="arena", seed=7)
                for _ in range(100):
                    robot.tick()
                continue
            entry = measure(robot, height, pitch)
            entry["settled"] = settled
            entries.append(entry)
            print(
                f"h={height:.2f} p={pitch:.1f}: pelvis z {entry['pelvis_z_m']:.3f} "
                f"offset {np.round(entry['pelvis_offset_feet_m'], 3)} "
                f"waist pitch {entry['balance_joints_rad']['waist_pitch_joint']:+.3f} "
                f"settled={settled} surface points {len(entry['body_surface_feet_m'])}"
            )
    robot.close()
    OUTPUT.write_text(
        json.dumps(
            {
                "description": __doc__.strip().splitlines()[0],
                "frame": "x forward, y left from the feet midpoint; z is world height",
                "entries": entries,
            }
        )
    )
    print(f"wrote {OUTPUT} with {len(entries)} postures")


if __name__ == "__main__":
    main()
