# Runs CooHOI's two-agent box carry with no display and saves an mp4 from an offscreen camera that follows the box.
# Usage (from the CooHOI repo root): python record_coohoi.py [--seconds 20] [--video out.mp4] <CooHOI run.py args>
#
# Viewing method: (a) Isaac Gym camera sensors, rendered offscreen and piped to ffmpeg. Why not (b) a virtual
# display with VNC: Isaac Gym's Vulkan renderer crashes in create_sim under Xvfb or X forwarding (NVIDIA forum
# thread 166222), so (b) would need a real headless Xorg on the NVIDIA driver, root config on the GPU that the
# Isaac Sim container also uses. Camera sensors need no window: forum replies (one from NVIDIA staff, threads
# 178901 and 183575) say the graphics.py and interop_torch.py examples run with --headless. The known failures
# have known fixes, applied below: graphics_device_id must not be -1 (CooHOI's BaseTask forces -1 when headless),
# and with the GPU pipeline fetch_results and step_graphics must run before render_all_camera_sensors or frames
# come out black (thread 178901). One camera only: several cameras per env segfault headless (thread 258924).
# unverified: no thread confirms a GPU with no monitor attached; one user reported it failing (thread 217244).
# unverified: the Preview 4 package's docs/ and python/examples were not available locally to check.
import argparse
import os
import subprocess
import sys

W, H = 1280, 720            # both divisible by 16 for libx264
FOV = 75.0                  # horizontal field of view, degrees
CAM_BACK, CAM_UP = 9.0, 3.5  # metres behind (-y) and above the box; at reset the humanoids stand ~5 m either side
LOOK_Z = 0.8                # aim between the box and the torsos


class Mp4:
    """Pipes raw RGBA frames into the system ffmpeg."""

    def __init__(self, path, fps):
        # ponytail: drop the conda LD_LIBRARY_PATH so apt's ffmpeg loads its own system libs
        env = {k: v for k, v in os.environ.items() if k != 'LD_LIBRARY_PATH'}
        self.proc = subprocess.Popen(
            ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgba', '-s', '%dx%d' % (W, H),
             '-r', str(fps), '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', path],
            stdin=subprocess.PIPE, env=env)

    def write(self, rgba):
        self.proc.stdin.write(rgba)

    def close(self):
        self.proc.stdin.close()
        return self.proc.wait()


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument('--seconds', type=float, default=20.0, help='simulated seconds to record')
    ap.add_argument('--video', default='coohoi_twoagent.mp4')
    ap.add_argument('--cam_env', type=int, default=1,
                    help='base env (odd id) of the pair to film; it holds the real box, its partner holds a ghost copy')
    rec, rest = ap.parse_known_args()
    if '--headless' not in rest:
        rest.append('--headless')  # no viewer: there is no display
    sys.argv = [sys.argv[0]] + rest  # CooHOI's own parser sees only its own args
    if not os.path.isdir('coohoi/env'):
        sys.exit('Run this from the CooHOI repo root (its configs are cwd-relative).')

    sys.path.insert(0, os.path.join(os.getcwd(), 'coohoi'))
    import run  # CooHOI's coohoi/run.py; it imports isaacgym before torch, as Isaac Gym requires
    import numpy as np
    from isaacgym import gymapi
    from env.tasks.base_task import BaseTask
    from env.tasks.share_humanoid_amp_carryobject import ShareHumanoidCarryObject as Task

    # 1. Keep a graphics device while headless (BaseTask sets graphics_device_id = -1 unless cameras are enabled).
    base_init = BaseTask.__init__

    def init_with_cameras(self, cfg, enable_camera_sensors=False):
        base_init(self, cfg, enable_camera_sensors=True)
    BaseTask.__init__ = init_with_cameras

    # 2. One camera sensor, created while the filmed env is built (the pattern Isaac Gym's examples use).
    build_env = Task._build_env

    def build_env_with_camera(self, env_id, env_ptr, humanoid_asset):
        build_env(self, env_id, env_ptr, humanoid_asset)
        if env_id == rec.cam_env:
            props = gymapi.CameraProperties()
            props.width, props.height, props.horizontal_fov = W, H, FOV
            self.rec_env, self.rec_cam = env_ptr, self.gym.create_camera_sensor(env_ptr, props)
            if self.rec_cam == -1:
                sys.exit('create_camera_sensor returned -1: no graphics device. See RUN.md, "If it fails".')
    Task._build_env = build_env_with_camera

    # 3. After every control step (30 Hz): move the camera to the box, render, pipe the frame to ffmpeg.
    mp4, left = None, 0

    def finish():
        if mp4 is not None:
            ok = mp4.close() == 0
            print('[record] %s %s' % ('saved' if ok else 'ffmpeg FAILED writing', os.path.abspath(rec.video)), flush=True)

    post = Task.post_physics_step

    def post_and_record(self):
        nonlocal mp4, left
        post(self)
        first = mp4 is None
        if first:
            fps = round(1.0 / self.dt)
            mp4, left = Mp4(rec.video, fps), int(rec.seconds * fps)
            print('[record] %d frames at %d fps -> %s' % (left, fps, rec.video), flush=True)
        x, y = self._box_states[rec.cam_env, 0:2].tolist()  # refreshed by post(); the base env's box is the real one
        # unverified: set_camera_location takes env-local coordinates, the same frame as the root state tensor
        # (CooHOI's ghost-box copy, shifted by 2 * envSpacing, only lines up if the tensor is env-local).
        self.gym.set_camera_location(self.rec_cam, self.rec_env, gymapi.Vec3(x, y - CAM_BACK, CAM_UP),
                                     gymapi.Vec3(x, y, LOOK_Z))
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        img = self.gym.get_camera_image(self.sim, self.rec_env, self.rec_cam, gymapi.IMAGE_COLOR)
        assert img.size == W * H * 4, 'unexpected camera image shape %s' % (img.shape,)
        if first:
            rgb_mean = np.asarray(img).reshape(-1, 4)[:, :3].mean()
            print('[record] first frame mean RGB %.1f (near 0 means the camera rendered nothing)' % rgb_mean, flush=True)
        mp4.write(img.tobytes())
        left -= 1
        if left == 0:
            finish()
            os._exit(0)  # ponytail: skip Isaac Gym's teardown, which often segfaults at exit; the mp4 is closed
    Task.post_physics_step = post_and_record

    try:
        run.main()  # rl_games' player loop; normally ends through os._exit above
    finally:
        finish()  # player stopped early or crashed: keep what was recorded


if __name__ == '__main__':
    main()
