"""Run an Isaac Sim standalone example headless and save its viewport as PNG frames, then quit.
Usage (in the container): /isaac-sim/python.sh record_demo.py OUT_DIR SECONDS EXAMPLE.py [example args...]
Frames land in OUT_DIR/frame_00000.png ...; make the mp4 on the host with ffmpeg (see RUN.md).
"""
import os
import runpy
import sys

out, secs, script, *rest = sys.argv[1:]
os.makedirs(out, exist_ok=True)
FPS = 30
total = int(float(secs) * FPS)

from isaacsim import SimulationApp  # the example's own import gets this same class

_init = SimulationApp.__init__


def init(self, config=None, *a, **k):
    config = dict(config or {})
    config.update(headless=True, width=1280, height=720)
    _init(self, config, *a, **k)
    import omni.kit.app
    from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

    state = {"frame": 0, "tick": 0}

    def on_update(_):
        # ponytail: one capture per rendered frame after a 2 s warmup; skips none, so video speed = render speed
        state["tick"] += 1
        if state["tick"] < 60:
            return
        if state["frame"] >= total:
            omni.kit.app.get_app().post_quit()
            return
        capture_viewport_to_file(get_active_viewport(), os.path.join(out, f"frame_{state['frame']:05d}.png"))
        state["frame"] += 1

    self._recorder = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(on_update)


SimulationApp.__init__ = init
sys.argv = [script, *rest]
sys.path.insert(0, os.path.dirname(os.path.abspath(script)))  # examples import helpers from their own folder
runpy.run_path(script, run_name="__main__")
