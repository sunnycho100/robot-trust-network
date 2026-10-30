"""On-demand camera rendering in a separate macOS main-thread process.

Native MuJoCo owns the viewer's Cocoa/GLFW main thread. This worker renders
sensor images from a copied integration state, never steps or controls physics.
"""

import base64
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

STATE = mujoco.mjtState.mjSTATE_INTEGRATION


def render_rgb_exposure(renderer, exposure=1.0):
    """Scale rendering lights before RGB clipping, never depth or physics.

    Support both the subprocess adapter and a direct main-thread renderer.
    Always restore the original lighting even if image acquisition fails.
    """
    if not np.isfinite(exposure) or not 0.1 <= exposure <= 1.0:
        raise ValueError("Sensor light exposure must be in [0.1, 1]")
    if isinstance(renderer, SensorRenderer):
        return renderer.render(exposure=exposure)
    lights = renderer.scene.lights[: renderer.scene.nlight]
    saved = [
        (light.ambient.copy(), light.diffuse.copy(), light.specular.copy())
        for light in lights
    ]
    try:
        for light in lights:
            light.ambient[:] *= exposure
            light.diffuse[:] *= exposure
            light.specular[:] *= exposure
        return renderer.render()
    finally:
        for light, (ambient, diffuse, specular) in zip(lights, saved, strict=True):
            light.ambient[:] = ambient
            light.diffuse[:] = diffuse
            light.specular[:] = specular


class SensorRenderer:
    def __init__(self, model, width=960, height=720):
        self.model = model
        self.width, self.height = width, height
        self.directory = tempfile.TemporaryDirectory(prefix="robot-camera-")
        model_path = Path(self.directory.name) / "model.mjb"
        model.vis.global_.offwidth = max(width, model.vis.global_.offwidth)
        model.vis.global_.offheight = max(height, model.vis.global_.offheight)
        mujoco.mj_saveModel(model, str(model_path))
        self.process = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                str(model_path),
                str(width),
                str(height),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.scene = SimpleNamespace(flags=np.zeros(100, dtype=bool), camera=[])
        self.mode = "rgb"
        self.state = np.empty(mujoco.mj_stateSize(model, STATE))
        self.camera = None

    def update_scene(self, data, camera):
        mujoco.mj_getState(self.model, data, self.state, STATE)
        if isinstance(camera, (int, np.integer)):
            self.camera = {"fixed": int(camera)}
        else:
            self.camera = {
                "lookat": camera.lookat.tolist(),
                "distance": camera.distance,
                "azimuth": camera.azimuth,
                "elevation": camera.elevation,
            }

    def render(self, exposure=1.0):
        request = {
            "state": self.state.tolist(),
            "camera": self.camera,
            "mode": self.mode,
            "exposure": exposure,
        }
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("Sensor renderer stopped")
        reply = json.loads(line)
        if "error" in reply:
            raise RuntimeError("Sensor render failed: " + reply["error"])
        self.scene.camera = [SimpleNamespace(**row) for row in reply["cameras"]]
        return np.frombuffer(
            base64.b64decode(reply["data"]), dtype=reply["dtype"]
        ).reshape(reply["shape"])

    def enable_depth_rendering(self):
        self.mode = "depth"

    def enable_segmentation_rendering(self):
        self.mode = "segmentation"

    def disable_depth_rendering(self):
        self.mode = "rgb"

    def disable_segmentation_rendering(self):
        self.mode = "rgb"

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=3)
        self.directory.cleanup()


def worker(model_path, width, height):
    model = mujoco.MjModel.from_binary_path(model_path)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, width=width, height=height)
    try:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                mujoco.mj_setState(model, data, np.asarray(request["state"]), STATE)
                mujoco.mj_forward(model, data)
                config = request["camera"]
                if "fixed" in config:
                    camera = config["fixed"]
                else:
                    camera = mujoco.MjvCamera()
                    camera.lookat[:] = config["lookat"]
                    for key in ("distance", "azimuth", "elevation"):
                        setattr(camera, key, config[key])
                renderer.disable_depth_rendering()
                renderer.disable_segmentation_rendering()
                if request["mode"] == "depth":
                    renderer.enable_depth_rendering()
                elif request["mode"] == "segmentation":
                    renderer.enable_segmentation_rendering()
                renderer.update_scene(data, camera=camera)
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
                frame = (
                    render_rgb_exposure(renderer, request.get("exposure", 1.0))
                    if request["mode"] == "rgb"
                    else renderer.render()
                )
                cameras = [
                    {
                        key: np.asarray(getattr(c, key)).tolist()
                        for key in (
                            "pos",
                            "forward",
                            "up",
                            "frustum_top",
                            "frustum_bottom",
                            "frustum_near",
                            "frustum_far",
                            "frustum_center",
                        )
                    }
                    for c in renderer.scene.camera
                ]
                reply = {
                    "data": base64.b64encode(frame.tobytes()).decode(),
                    "dtype": str(frame.dtype),
                    "shape": frame.shape,
                    "cameras": cameras,
                }
            except (RuntimeError, ValueError, TypeError, KeyError) as exc:
                reply = {"error": str(exc)}
            print(json.dumps(reply), flush=True)
    finally:
        renderer.close()


if __name__ == "__main__":
    worker(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
