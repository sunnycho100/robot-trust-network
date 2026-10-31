"""play.py with the viewer at a fixed eye (a real photo spot in a scan) instead of 9 m behind the box, which in a
scanned room sits inside walls or splat floaters. Same arguments as play.py plus --eye X Y Z [--look X Y Z];
without --look the camera stays at the eye and turns to follow the box.
  python play_scene.py --mode pair --episodes 4 --num_envs 1 --effort unlimited --scene ~/da/da_world.usda \
      --video ~/frames_da --eye 1.2 -6.3 1.6
"""
import argparse
import contextlib
import sys

ap = argparse.ArgumentParser(add_help=False)
ap.add_argument("--eye", type=float, nargs=3, required=True)
ap.add_argument("--look", type=float, nargs=3)
cam, rest = ap.parse_known_args()
sys.argv = [sys.argv[0], *rest]
import play  # noqa: E402  parses the remaining arguments

_launch = play.launch_simulation


@contextlib.contextmanager
def launch(cfg, args):
    with _launch(cfg, args) as app:
        import isaaclab_physx.renderers.kit_viewport_utils as kv  # needs Kit running

        set_view = kv.set_kit_renderer_camera_view
        kv.set_kit_renderer_camera_view = lambda eye, target, *a, **k: set_view(
            tuple(cam.eye), tuple(cam.look or target), *a, **k)  # play.py imports it inside main(), after this
        yield app


play.launch_simulation = launch
play.main()
