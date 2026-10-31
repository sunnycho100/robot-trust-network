"""Runs CooHOI's trained policy in the Isaac Lab port and logs the success rate, box height and distance to the target.
Usage (Isaac Lab 3.0 venv on Isaac Sim 6.2, from this folder; see README.md):
  python play.py --mode single --episodes 100                # SingleAgent.pth, one humanoid and a cube
  python play.py --mode pair --episodes 100                  # TwoAgent.pth, two humanoids and one box
  python play.py --mode pair --episodes 1 --num_envs 1 --video ~/frames/pair   # viewport PNGs of env 0, then ffmpeg
  python play.py --mode pair --num_envs 1 --scene /robot/recon/da_world.usda --origin X Y Z   # inside a scan
An episode succeeds when it ends with the box centre within 0.2 m of the target (CooHOI's eval threshold,
humanoid_amp_carryobject.py). Video: the same viewport capture as demos/record_demo.py, one frame per policy step (30 fps).
"""
import argparse
import os

from isaaclab.app import add_launcher_args, launch_simulation

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--mode", choices=["pair", "single"], default="pair")
ap.add_argument("--checkpoint", help="default: TwoAgent.pth (pair) or SingleAgent.pth (single) from the CooHOI repo")
ap.add_argument("--episodes", type=int, default=100)
ap.add_argument("--num_envs", type=int, default=16)
ap.add_argument("--box_scale", type=float, default=1.0)
ap.add_argument("--plates", type=int, default=0, help="pair, looks only: draw the box as N strapped steel plates")
ap.add_argument("--ground_color", type=float, nargs=3, metavar=("R", "G", "B"), help="tint the ground plane, e.g. 0.35 0.36 0.37 for concrete")
ap.add_argument("--physics_dt", type=float, default=1 / 120, help="CooHOI: 1/60 with 2 substeps = 1/120")
ap.add_argument("--decimation", type=int, default=4, help="physics steps per policy step (30 Hz policy)")
ap.add_argument("--effort", choices=["gear", "unlimited", "usd"], default="gear", help="joint effort limit: MJCF gear, 1e9, or the USD's")
ap.add_argument("--scene", help="scene USD to load instead of the ground plane (e.g. our reconstruction)")
ap.add_argument("--origin", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("X", "Y", "Z"),
                help="where the task layout sits in the scene (floor height as Z)")
ap.add_argument("--video", help="folder for env 0's viewport frames (frame_00000.png ...)")
ap.add_argument("--seed", type=int, default=0)
add_launcher_args(ap)
args = ap.parse_args()
if args.video:
    # headless + enable_cameras loads isaaclab.python.headless.rendering.kit, which keeps a Kit viewport to capture
    args.enable_cameras = True

from coohoi_env import CoohoiEnv, make_cfg  # noqa: E402
from joint_map import coohoi_root  # noqa: E402
from policy import CoohoiActor  # noqa: E402

CAM_BACK, CAM_UP, LOOK_Z = 9.0, 3.5, 0.8  # as coohoi/record_coohoi.py: behind (-y) and above the box


def main():
    cfg = make_cfg(args.mode, args.num_envs, args.box_scale, args.physics_dt, args.decimation, args.effort,
                   args.scene, args.origin, args.seed, args.plates)
    if args.ground_color and cfg.scene.ground:
        cfg.scene.ground.spawn.color = tuple(args.ground_color)
    ckpt = args.checkpoint or os.path.join(coohoi_root(), "coohoi/data/models",
                                           "TwoAgent.pth" if args.mode == "pair" else "SingleAgent.pth")
    with launch_simulation(cfg, args):
        env = CoohoiEnv(cfg)
        pi = CoohoiActor.load(ckpt, env.device)
        obs, _ = env.reset()
        # quaternion-order check: after an identity-yaw reset the left thigh must sit on +y of the pelvis
        p = env.bodies(env.possible_agents[0])[0]
        dy = p[:, env.jmap.gym_bodies.index("left_thigh"), 1] - p[:, 0, 1]
        assert (dy > 0).all(), f"left thigh not on +y of the pelvis after reset ({dy.tolist()}): quaternion order?"
        if args.video:
            from isaaclab_physx.renderers.isaac_rtx_renderer_utils import ensure_isaac_rtx_render_update
            from isaaclab_physx.renderers.kit_viewport_utils import set_kit_renderer_camera_view
            from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
            os.makedirs(args.video, exist_ok=True)
        done, rows, frame = 0, [], 0
        print(f"[play] {args.mode}: {ckpt}, {args.num_envs} envs, dt {cfg.sim.dt:.5f} x {cfg.decimation}, effort {args.effort}")
        while done < args.episodes and env.sim.is_running():
            obs, _, term, trunc, _ = env.step({a: pi(o) for a, o in obs.items()})  # pi runs under no_grad
            a0 = env.possible_agents[0]
            ended = (term[a0] | trunc[a0]).nonzero().flatten().tolist()
            for i in ended:
                d, h = env.ep_dist[i].item(), env.ep_max_h[i].item()
                rows.append((d < 0.2, d, h, bool(term[a0][i])))
                print(f"[play] episode {len(rows)} env {i}: {'success' if d < 0.2 else 'fail'}, box to target "
                      f"{d:.2f} m, max box height {h:.2f} m{', fell or kicked' if term[a0][i] else ''}", flush=True)
            done += len(ended)
            if args.video:
                x, y, _ = (env.box.data.root_pos_w.torch[0]).tolist()
                set_kit_renderer_camera_view((x, y - CAM_BACK, CAM_UP), (x, y, LOOK_Z))  # follows the box
                ensure_isaac_rtx_render_update(force=True)
                capture_viewport_to_file(get_active_viewport(), os.path.join(args.video, f"frame_{frame:05d}.png"))
                frame += 1
        if args.video:
            import omni.kit.app

            for _ in range(10):  # captures are written asynchronously; let the last ones land
                omni.kit.app.get_app().update()
            print(f"[play] {frame} frames in {args.video}; mp4: ffmpeg -framerate 30 -i {args.video}/frame_%05d.png "
                  f"-c:v libx264 -pix_fmt yuv420p {args.video}.mp4")
        if rows:
            n = len(rows)
            print(f"[play] {n} episodes: success rate {sum(r[0] for r in rows) / n:.2f}, mean final distance "
                  f"{sum(r[1] for r in rows) / n:.2f} m, mean max box height {sum(r[2] for r in rows) / n:.2f} m, "
                  f"early terminations {sum(r[3] for r in rows)}")
        env.close()


if __name__ == "__main__":
    main()
