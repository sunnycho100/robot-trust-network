"""Two full-body Unitree G1 humanoids carrying one box in MuJoCo.

The published Unitree locomotion policy controls the twelve leg joints.  A
separate PD controller holds the waist and arms in a carrying posture.  The
two robots squeeze high-friction palm pads against opposite sides of a dynamic
box, so vertical support comes from contact friction.
"""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

import mujoco
import numpy as np

from .g1_policy import G1Config, G1Controller, G1UpperBodyController, UPSTREAM_ROOT, gait_phase


def demo_config() -> G1Config:
    """Unitree's published G1 deployment values without a PyYAML dependency."""
    return G1Config(
        policy_path=UPSTREAM_ROOT / "deploy" / "pre_train" / "g1" / "motion.pt",
        xml_path=UPSTREAM_ROOT / "resources" / "robots" / "g1_description" / "scene.xml",
        simulation_dt=0.002,
        control_decimation=10,
        kps=np.array([100, 100, 100, 150, 40, 40] * 2, dtype=np.float32),
        kds=np.array([2, 2, 2, 4, 2, 2] * 2, dtype=np.float32),
        default_angles=np.array([-0.1, 0, 0, 0.3, -0.2, 0] * 2, dtype=np.float32),
        ang_vel_scale=0.25,
        dof_pos_scale=1.0,
        dof_vel_scale=0.05,
        action_scale=0.25,
        cmd_scale=np.array([2.0, 2.0, 0.25], dtype=np.float32),
        num_actions=12,
        num_obs=47,
    )


def build_model(cfg: G1Config, assisted_grasp: bool = False) -> mujoco.MjModel:
    arena = mujoco.MjSpec.from_string("""
      <mujoco model="G1 cooperative box carry">
        <option timestep="0.002" integrator="implicitfast"/>
        <visual><headlight diffuse="0.7 0.7 0.7" ambient="0.25 0.25 0.25" specular="0.8 0.8 0.8"/>
          <rgba haze="0.15 0.20 0.25 1"/><global azimuth="145" elevation="-18" offwidth="960" offheight="720"/></visual>
        <asset>
          <texture type="skybox" builtin="gradient" rgb1="0.04 0.06 0.09" rgb2="0.18 0.22 0.28" width="512" height="3072"/>
          <texture type="2d" name="ground" builtin="checker" rgb1="0.18 0.20 0.22" rgb2="0.28 0.30 0.32" width="300" height="300"/>
          <material name="ground" texture="ground" texrepeat="10 10" reflectance="0.15"/>
        </asset>
        <worldbody>
          <light pos="0 0 4" dir="0 0 -1" directional="true"/>
          <geom name="floor" type="plane" size="0 0 0.05" material="ground"
                contype="9" conaffinity="9" friction="1.2 0.02 0.002"/>
          <body name="carry_box" pos="0 0 1.104">
            <freejoint name="carry_box_free"/>
            <inertial pos="0 0 0" mass="0.75" diaginertia="0.016 0.079 0.088"/>
            <geom name="box_geom" type="box" size="0.55 0.22 0.12" rgba="0.72 0.42 0.15 1"
                  contype="4" conaffinity="10" condim="4" friction="4.0 0.15 0.01"
                  solref="0.01 1" solimp="0.95 0.99 0.001"/>
            <site name="box_leader_left" pos="0.329 0.22 0" size="0.018" rgba="1 0.85 0.2 1"/>
            <site name="box_leader_right" pos="-0.329 0.22 0" size="0.018" rgba="1 0.85 0.2 1"/>
            <site name="box_follower_left" pos="-0.329 -0.22 0" size="0.018" rgba="1 0.85 0.2 1"/>
            <site name="box_follower_right" pos="0.329 -0.22 0" size="0.018" rgba="1 0.85 0.2 1"/>
          </body>
        </worldbody>
      </mujoco>
    """)
    # The robots stand at the two ends.  Each reaches around its end corner:
    # one palm contacts side A (+Y), the other contacts side B (-Y).
    leader_frame = arena.worldbody.add_frame(name="leader_spawn", pos=(-0.857, 0, 0), quat=(1, 0, 0, 0))
    follower_frame = arena.worldbody.add_frame(name="follower_spawn", pos=(0.857, 0, 0), quat=(0, 0, 0, 1))
    arena.attach(mujoco.MjSpec.from_file(str(cfg.robot_xml)), prefix="leader_", frame=leader_frame)
    arena.attach(mujoco.MjSpec.from_file(str(cfg.robot_xml)), prefix="follower_", frame=follower_frame)

    if assisted_grasp:
        pairs = (
            ("leader_left_hand_site", "box_leader_left"),
            ("leader_right_hand_site", "box_leader_right"),
            ("follower_left_hand_site", "box_follower_left"),
            ("follower_right_hand_site", "box_follower_right"),
        )
        for index, (hand, handle) in enumerate(pairs):
            arena.add_equality(
                name=f"soft_grasp_{index}", type=mujoco.mjtEq.mjEQ_CONNECT,
                objtype=mujoco.mjtObj.mjOBJ_SITE, name1=hand, name2=handle,
                solref=(0.03, 1.0), solimp=(0.90, 0.95, 0.002, 0.5, 2.0),
            )

    model = arena.compile()
    model.opt.timestep = cfg.simulation_dt
    for geom_id, body_id in enumerate(model.geom_bodyid):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(body_id)) or ""
        if body_name.startswith("leader_"):
            tint = np.array([0.90, 0.30, 0.18])
        elif body_name.startswith("follower_"):
            tint = np.array([0.16, 0.48, 0.95])
        else:
            continue
        model.geom_rgba[geom_id, :3] = 0.35 * model.geom_rgba[geom_id, :3] + 0.65 * tint
    return model


class RawVideoWriter:
    def __init__(self, path: Path, width: int, height: int, fps: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.process = subprocess.Popen([
            "ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
            "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p", str(path),
        ], stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        assert self.process.stdin is not None
        self.process.stdin.write(frame.tobytes())

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        if self.process.wait() != 0:
            raise RuntimeError("ffmpeg failed while writing the cooperative carry video")


def run(args) -> None:
    cfg = demo_config()
    model = build_model(cfg, assisted_grasp=args.assisted_grasp)
    data = mujoco.MjData(model)
    legs = [G1Controller(model, cfg, "leader_"), G1Controller(model, cfg, "follower_")]
    upper = [G1UpperBodyController(model, "leader_"), G1UpperBodyController(model, "follower_")]
    for controller in legs:
        controller.reset(data)
    for controller in upper:
        controller.set_carry_pose()
        data.qpos[controller.joint_qpos] = controller.target_q
    mujoco.mj_forward(model, data)

    fps = 30
    renderer = writer = None
    if args.video:
        renderer = mujoco.Renderer(model, height=720, width=960)
        writer = RawVideoWriter(args.video, 960, 720, fps)
    viewer = None
    if not args.headless:
        from mujoco import viewer as mj_viewer
        viewer = mj_viewer.launch_passive(model, data)
        viewer.cam.lookat[:] = (0, 0, 0.8)
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 3.8, 135, -18

    leader_command = np.array([args.squeeze_speed, 0.0, 0.0], dtype=np.float32)
    follower_command = np.array([args.squeeze_speed, 0.0, 0.0], dtype=np.float32)
    steps = int(args.duration / cfg.simulation_dt)
    render_every = max(1, round(1 / (fps * cfg.simulation_dt)))
    wall_start = time.monotonic()
    try:
        for step in range(steps):
            elapsed = step * cfg.simulation_dt
            approach = min(1.0, elapsed / max(args.approach_time, cfg.simulation_dt))
            inward_speed = args.squeeze_speed * (0.8 + 0.2 * approach)
            leader_command[0] = inward_speed
            follower_command[0] = inward_speed
            if elapsed >= args.settle_time:
                leader_command[1] = args.speed
                follower_command[1] = -args.speed
            for controller in legs:
                controller.apply(data)
            for controller in upper:
                controller.apply(data)
            mujoco.mj_step(model, data)
            if step % cfg.control_decimation == 0:
                phase = gait_phase(data)
                legs[0].update_policy(data, leader_command, phase)
                legs[1].update_policy(data, follower_command, phase)
            if renderer is not None and writer is not None and step % render_every == 0:
                camera = mujoco.MjvCamera()
                camera.lookat[:] = data.body("carry_box").xpos
                camera.lookat[2] = 0.8
                camera.distance, camera.azimuth, camera.elevation = 3.8, 135, -18
                renderer.update_scene(data, camera=camera)
                writer.write(renderer.render())
            if viewer is not None:
                if not viewer.is_running():
                    break
                viewer.sync()
                delay = data.time - (time.monotonic() - wall_start)
                if delay > 0:
                    time.sleep(delay)
    finally:
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()
        if viewer is not None:
            viewer.close()

    box = data.body("carry_box").xpos
    leader_pelvis = data.body("leader_pelvis").xpos
    follower_pelvis = data.body("follower_pelvis").xpos
    arm_error = max(float(np.max(np.abs(controller.target_q - data.qpos[controller.joint_qpos]))) for controller in upper)
    print(f"completed t={data.time:.2f}s box=({box[0]:.3f}, {box[1]:.3f}, {box[2]:.3f}) "
          f"pelvis_z=({leader_pelvis[2]:.3f}, {follower_pelvis[2]:.3f}) arm_error={arm_error:.3f}")
    if args.video:
        print(f"video={args.video}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--settle-time", type=float, default=1.5)
    parser.add_argument("--speed", type=float, default=0.12, help="Body-frame lateral walking speed")
    parser.add_argument("--squeeze-speed", type=float, default=0.12,
                        help="Small inward walking command that maintains side-grip normal force")
    parser.add_argument("--approach-time", type=float, default=1.2,
                        help="Seconds used to ramp the inward approach command")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--assisted-grasp", action="store_true",
                        help="Add compliant hand constraints for comparison; default uses friction only")
    args = parser.parse_args()
    if args.duration <= 0 or args.settle_time < 0:
        parser.error("duration must be positive and settle-time cannot be negative")
    run(args)


if __name__ == "__main__":
    main()
