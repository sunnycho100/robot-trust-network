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
#
# Both humanoids: an Isaac Gym camera renders only the actors of its own env, and CooHOI puts each humanoid of a
# pair in its own env (base env odd, partner env = base - 1). So a second camera sits in the partner env at the same
# world pose, and the two frames are merged per pixel by depth (composite() below). The partner env also holds a
# still "ghost" box; it gets segmentation id GHOST so the merge drops it. --one_cam goes back to a single camera.
# unverified: two cameras in two envs headless (the segfault in thread 258924 was several cameras in one env).
# unverified: the Preview 4 package's docs/ and python/examples were not available locally to check.
import argparse
import os
import subprocess
import sys

W, H = 1280, 720            # both divisible by 16 for libx264
FOV = 75.0                  # horizontal field of view, degrees
CAM_BACK, CAM_UP = 9.0, 3.5  # metres behind (-y) and above the box; at reset the humanoids stand ~5 m either side
LOOK_Z = 0.8                # aim between the box and the torsos
GHOST = 7                   # segmentation id given to the partner env's ghost box
DEPTH_EPS = 0.02            # metres: the partner pixel must be this much nearer to win (the ground plane ties)


def composite(rgba_base, depth_base, rgba_partner, depth_partner, seg_partner):
    """Per pixel, the nearer of two renders of the same world pose. Isaac Gym depth is the negative distance
    (-inf where nothing is hit), so nearer means larger. Ties (the shared ground) and ghost-box pixels keep the base."""
    take = (depth_partner > depth_base + DEPTH_EPS) & (seg_partner != GHOST)
    out = rgba_base.reshape(H, W, 4).copy()
    out[take] = rgba_partner.reshape(H, W, 4)[take]
    return out


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
    ap.add_argument('--one_cam', action='store_true', help='film only the base env (one humanoid shows)')
    ap.add_argument('--partner_shift', choices=['auto', 'env', 'none'], default='auto',
                    help='how to place the partner camera: env-local frames (shift by the env origins) or global')
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

    # 2. One camera sensor per filmed env, created while that env is built (the pattern Isaac Gym's examples use).
    build_env = Task._build_env
    partner = rec.cam_env - 1
    filmed = [rec.cam_env] if rec.one_cam else [rec.cam_env, partner]

    def build_env_with_camera(self, env_id, env_ptr, humanoid_asset):
        build_env(self, env_id, env_ptr, humanoid_asset)
        if env_id in filmed:
            props = gymapi.CameraProperties()
            props.width, props.height, props.horizontal_fov = W, H, FOV
            cam = self.gym.create_camera_sensor(env_ptr, props)
            if cam == -1:
                sys.exit('create_camera_sensor returned -1: no graphics device. See RUN.md, "If it fails".')
            if not hasattr(self, 'rec_cams'):
                self.rec_cams = {}
            self.rec_cams[env_id] = (env_ptr, cam)
            if env_id == partner:  # _build_box just created this env's box, the ghost copy
                self.gym.set_rigid_body_segmentation_id(env_ptr, self._box_handles[-1], 0, GHOST)
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
        pos, look = gymapi.Vec3(x, y - CAM_BACK, CAM_UP), gymapi.Vec3(x, y, LOOK_Z)
        base_env, base_cam = self.rec_cams[rec.cam_env]
        self.gym.set_camera_location(base_cam, base_env, pos, look)
        if not rec.one_cam:
            p_env, p_cam = self.rec_cams[partner]
            if first:
                # The base camera already frames the box from root-state coordinates, so the camera uses the same
                # frame as the root states. If those are env-local, both humanoids read near the same spot and the
                # partner camera must shift by the difference of the env origins; if global, they are ~10 m apart.
                ob, op = self.gym.get_env_origin(base_env), self.gym.get_env_origin(p_env)
                d_orig = ((ob.x - op.x) ** 2 + (ob.y - op.y) ** 2) ** 0.5
                hb, hp = self._humanoid_root_states[rec.cam_env, 0:2], self._humanoid_root_states[partner, 0:2]
                d_hum = float((hb - hp).norm())
                local = d_hum < d_orig / 2 if rec.partner_shift == 'auto' else rec.partner_shift == 'env'
                self.rec_shift = gymapi.Vec3(ob.x - op.x, ob.y - op.y, 0.0) if local else gymapi.Vec3(0, 0, 0)
                print('[record] env origins %.1f m apart, humanoid roots %.1f m apart -> %s frames, partner camera '
                      'shift (%.1f, %.1f)' % (d_orig, d_hum, 'env-local' if local else 'global',
                                              self.rec_shift.x, self.rec_shift.y), flush=True)
            sh = self.rec_shift
            self.gym.set_camera_location(p_cam, p_env, gymapi.Vec3(pos.x + sh.x, pos.y + sh.y, pos.z),
                                         gymapi.Vec3(look.x + sh.x, look.y + sh.y, look.z))
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        img = self.gym.get_camera_image(self.sim, base_env, base_cam, gymapi.IMAGE_COLOR)
        assert img.size == W * H * 4, 'unexpected camera image shape %s' % (img.shape,)
        if not rec.one_cam:
            grab = lambda env, cam, kind: np.asarray(self.gym.get_camera_image(self.sim, env, cam, kind)).reshape(H, W)
            p_img = self.gym.get_camera_image(self.sim, p_env, p_cam, gymapi.IMAGE_COLOR)
            img = composite(img, grab(base_env, base_cam, gymapi.IMAGE_DEPTH), p_img,
                            grab(p_env, p_cam, gymapi.IMAGE_DEPTH), grab(p_env, p_cam, gymapi.IMAGE_SEGMENTATION))
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
