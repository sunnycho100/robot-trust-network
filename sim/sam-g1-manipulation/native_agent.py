"""One native window for the MuJoCo viewport and English robot controls."""

import argparse
import threading
import time

import mujoco
import numpy as np

from agent_server import Console, plain
from sensor_renderer import STATE, SensorRenderer


class NativeConsole(Console):
    """Physics owns live data; the desktop receives copied integration states."""

    def __init__(self, model, port):
        super().__init__(model, port)
        self.view_lock = threading.Lock()
        self.view_model = None
        self.view_state = None
        self.closed = threading.Event()
        self.last_telemetry = 0
        self.wall_start = time.monotonic()
        self.view_frames = 0

    def make_renderer(self):
        return SensorRenderer(self.robot.tools.model, width=960, height=720)

    def reset(self):
        with self.view_lock:
            self.view_model = self.view_state = None
        self.wall_start = time.monotonic()
        self.last_telemetry = 0
        super().reset()

    def publish(self, force=False, segment=False, depth=False):
        model = self.robot.tools.model
        state = np.empty(mujoco.mj_stateSize(model, STATE))
        mujoco.mj_getState(model, self.robot.tools.data, state, STATE)
        with self.view_lock:
            self.view_model, self.view_state = model, state
        # Display rendering never encodes JPEGs. Only requested AI observations
        # go to the sensor renderer, whose camera is independent of the desktop.
        if segment or depth or (force and not self.frame):
            return super().publish(force=True, segment=segment, depth=depth)
        now = time.monotonic()
        if force or now - self.last_telemetry > 0.25:
            with self.lock:
                self.snapshot = plain(
                    {**self.robot.state(), "joints": self.robot.joint_state()}
                )
            self.last_telemetry = now

    def tick(self, epoch=None):
        if self.closed.is_set():
            raise RuntimeError("Desktop window was closed")
        super().tick(epoch)

    def keep_running(self):
        return not self.closed.is_set()

    def state(self):
        state = super().state()
        state["display"] = {
            "mode": "native",
            "layout": "combined",
            "agent_camera": self.agent_camera,
            "sensor_resolution": [960, 720],
            "viewer_camera": "user controlled",
            "rendered_frames": self.view_frames,
            "real_time_ratio": self.robot.ticks
            * self.robot.dt
            / max(0.001, time.monotonic() - self.wall_start)
            if self.robot
            else 0,
        }
        return state

    def shutdown(self):
        self.closed.set()
        self.epoch += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gemini-3.8-flash")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--scene",
        choices=("sorting", "clutter", "arena", "calibration"),
        default="sorting",
    )
    parser.add_argument("--control", choices=("normal", "rig"), default="normal")
    parser.add_argument(
        "--no-panel", action="store_true", help="Hide the control sidebar"
    )
    args = parser.parse_args()
    from desktop_window import run_desktop

    console = NativeConsole(args.model, args.port)
    console.settings.update(scene=args.scene, control=args.control)
    run_desktop(console, show_panel=not args.no_panel)


if __name__ == "__main__":
    main()
