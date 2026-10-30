"""Local robot console: physics loop, camera frames, HTTP API and the agent."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

import agent
from controller import ManipulationRuntime
from google_brain import GoogleBrain
from scene_memory import SceneMemory
from skills import execute as run_skill
from skills import forget_dropped
from trace_format import exception_details

FRAME_CACHE = 16


def plain(value):
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


class Console:
    def __init__(self, model, port):
        self.model_name = model
        self._brain = None
        self.port = port
        self.lock = threading.RLock()
        self.commands = queue.Queue()
        self.epoch = 0
        self.busy = False
        self.status = "Starting simulation"
        self.messages = []
        self.snapshot = {}
        self.frame = b""
        self.last_render = 0
        self.realtime = True
        self.fault = None
        self.logging_error = None
        self.robot = None
        self.renderer = None
        self.settings = {
            "scene": "sorting",
            "seed": 7,
            "camera": "overview",
            "control": "normal",
            "perception": "actions",
        }
        self.motion_kind = None
        self.carrying_side = None
        self.visual_sensor = None
        self.sensor_id = 0
        self.agent_camera = "head"
        self.frames = {}
        self.held = {}
        self.scene_memory = SceneMemory()
        self.output = Path("runs/agent")
        self.output.mkdir(parents=True, exist_ok=True)
        self.implementation = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).parent.glob("*.py")
        }
        self.build_id = hashlib.sha256(
            json.dumps(self.implementation, sort_keys=True).encode()
        ).hexdigest()[:12]

    @property
    def brain(self):
        if self._brain is None:
            self._brain = GoogleBrain(self.model_name)
        return self._brain

    def message(self, role, text, **extra):
        with self.lock:
            row = {"time": time.time(), "role": role, "text": text, "epoch": self.epoch, **extra}
            self.messages.append(row)
            self.messages = self.messages[-500:]
            try:
                with (self.output / "events.jsonl").open("a") as f:
                    f.write(json.dumps(plain(row), default=str) + "\n")
                trial = getattr(self, "last_action_trial", None)
                if trial and role != "user" and getattr(self, "trace_epoch", None) == self.epoch:
                    with (trial / "trace.jsonl").open("a") as f:
                        f.write(json.dumps(plain(row), default=str) + "\n")
            except OSError as exc:
                self.log_storage_failure(exc)

    def log_storage_failure(self, exc):
        detail = f"Debug evidence could not be saved: {exc}. Some trace files are incomplete."
        with self.lock:
            if not self.logging_error:
                self.messages.append({"time": time.time(), "role": "error", "text": detail, "epoch": self.epoch})
            self.logging_error = detail

    def state(self):
        with self.lock:
            return {
                "model": self.model_name,
                "busy": self.busy,
                "status": self.status,
                "messages": list(self.messages),
                "robot": self.snapshot,
                "holding": {side: row["label"] for side, row in self.held.items()},
                "fault": self.fault,
                "logging_error": self.logging_error,
                "settings": dict(self.settings),
                "trace_directory": str(getattr(self, "last_action_trial", "")),
                "build_id": self.build_id,
            }

    def reset(self):
        self.held = {}
        self.frames = {}
        self.scene_memory = SceneMemory()
        if self.renderer:
            self.renderer.close()
        if self.robot:
            self.robot.close()
        self.robot = ManipulationRuntime(
            scene=self.settings["scene"],
            seed=self.settings["seed"],
            joint_lab=self.settings["control"] == "rig",
            episode_length=100000000,
        )
        self.renderer = self.make_renderer()
        self.fault = None
        self.last_render = 0
        for _ in range(50):
            self.robot.tick()
        self.publish(force=True)
        with self.lock:
            self.status = "Ready"

    def make_renderer(self):
        model = self.robot.tools.model
        model.vis.global_.offwidth = max(960, model.vis.global_.offwidth)
        model.vis.global_.offheight = max(720, model.vis.global_.offheight)
        return mujoco.Renderer(model, height=720, width=960)

    def keep_running(self):
        return True

    def camera_for(self, camera):
        if camera in ("head", "left_wrist", "right_wrist"):
            return self.robot.tools._cam_id(camera)
        view = mujoco.MjvCamera()
        pos = self.robot.state()["base_pos"]
        view.lookat[:] = [pos[0] + 0.6, pos[1], 0.65]
        view.distance = 3.1 if self.settings["scene"] != "arena" else 5.3
        view.azimuth, view.elevation = 145, -30
        if camera == "front":
            view.lookat[:] = [pos[0], pos[1], 0.75]
            view.distance, view.azimuth, view.elevation = 1.8, 140, -15
        elif camera == "top":
            view.azimuth, view.elevation = 90, -85
        return view

    def publish(self, force=False, segment=False, depth=False):
        now = time.monotonic()
        if not (force or now - self.last_render >= 0.2):
            return
        camera = self.agent_camera if depth else self.settings["camera"]
        self.renderer.update_scene(self.robot.tools.data, camera=self.camera_for(camera))
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
        from sensor_renderer import render_rgb_exposure

        frame = render_rgb_exposure(self.renderer, 1.0).copy()
        buffer = io.BytesIO()
        Image.fromarray(frame).save(buffer, format="JPEG", quality=85)
        snapshot = plain({**self.robot.state(), "camera": camera})
        if depth:
            gl = self.renderer.scene.camera
            calibration = {
                "position": np.mean([c.pos for c in gl], axis=0).tolist(),
                "forward": np.mean([c.forward for c in gl], axis=0).tolist(),
                "up": np.mean([c.up for c in gl], axis=0).tolist(),
                "top": float(gl[0].frustum_top),
                "bottom": float(gl[0].frustum_bottom),
                "near": float(gl[0].frustum_near),
                "far": float(gl[0].frustum_far),
                "center": float(np.mean([c.frustum_center for c in gl])),
            }
            self.renderer.enable_depth_rendering()
            try:
                depth_map = self.renderer.render().copy()
            finally:
                self.renderer.disable_depth_rendering()
            # The robot's own silhouette is a known sensor mask, never object labels.
            self.renderer.enable_segmentation_rendering()
            try:
                self_mask = self.renderer.render().copy()
            finally:
                self.renderer.disable_segmentation_rendering()
            model = self.robot.tools.model
            prefix = self.robot.tools.robot.namespace
            robot_geoms = [
                g
                for g in range(model.ngeom)
                if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) or "").startswith(prefix)
            ]
            self.sensor_id += 1
            self.visual_sensor = {
                "image": buffer.getvalue(),
                "rgb": frame,
                "depth": depth_map,
                "calibration": calibration,
                "robot_mask": np.isin(self_mask[:, :, 0], robot_geoms)
                & (self_mask[:, :, 1] == mujoco.mjtObj.mjOBJ_GEOM),
                "frame_id": str(self.sensor_id),
                "tick": self.robot.ticks,
                "camera": camera,
            }
            self.frames[str(self.sensor_id)] = self.visual_sensor
            while len(self.frames) > FRAME_CACHE:
                self.frames.pop(next(iter(self.frames)))
            self.archive_sensor()
        with self.lock:
            if not depth:
                self.frame = buffer.getvalue()
            self.snapshot = snapshot
        self.last_render = time.monotonic()

    def archive_sensor(self):
        """Keep the RGB-D frames a trial used, for reproducible debugging."""
        trial = getattr(self, "last_action_trial", None)
        if not trial or getattr(self, "trace_epoch", None) != self.epoch:
            return
        sensor = self.visual_sensor
        directory = Path(trial) / "sensors"
        path = directory / f"frame-{sensor['frame_id']}.npz"
        try:
            directory.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                path,
                rgb=sensor["rgb"],
                depth=sensor["depth"],
                robot_mask=sensor["robot_mask"],
                calibration=json.dumps(plain(sensor["calibration"])),
                camera=sensor["camera"],
                frame_id=sensor["frame_id"],
                tick=sensor.get("tick", -1),
            )
        except OSError as exc:
            path.unlink(missing_ok=True)
            self.log_storage_failure(exc)

    def robot_status(self):
        forget_dropped(self)
        pelvis = self.robot.tools.pelvis_pose()[0]
        return {
            "holding": {side: row["label"] for side, row in self.held.items()} or "nothing",
            "pelvis_height_m": round(float(pelvis[2]), 3),
            "gripper_closure": {k: round(v, 2) for k, v in self.robot.state()["gripper_closure"].items()},
        }

    def tick(self, epoch=None):
        if epoch is not None and epoch != self.epoch:
            raise RuntimeError("Cancelled by user; motion interrupted")
        start = time.monotonic()
        self.robot.tick()
        if self.status == "Decelerating":
            lin, ang, _ = self.robot.lower_proxy.get_base_obs()
            if np.linalg.norm(lin[:2]) < 0.04 and abs(ang[2]) < 0.08:
                self.status = "Ready"
        if self.motion_kind == "walk":
            self.check_walking_contacts()
        if len(self.robot.trace) > 500:
            del self.robot.trace[:-250]
        self.publish()
        if self.realtime:
            time.sleep(max(0, self.robot.dt - (time.monotonic() - start)))

    def check_walking_contacts(self):
        """Stop walking if any body part other than the feet bumps into something."""
        model, data = self.robot.tools.model, self.robot.tools.data
        namespace = self.robot.tools.robot.namespace
        holding = set(self.held)
        for index, contact in enumerate(data.contact):
            bodies = [
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) or ""
                for g in (contact.geom1, contact.geom2)
            ]
            own = [i for i, name in enumerate(bodies) if name.startswith(namespace)]
            if len(own) != 1:
                continue
            link = bodies[own[0]].split("/")[-1]
            if "ankle" in link or "foot" in link:
                continue
            if any(link.startswith(side + "_") and ("dex1" in link or "wrist" in link) for side in holding):
                continue  # the hand holding an object touches that object
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, index, force)
            if abs(force[0]) < 0.5:
                continue
            # Local debugging record only; the agent is told just which own link touched.
            self.last_contact_debug = {
                "robot_link": link,
                "other_body": bodies[1 - own[0]],
                "point_world_m": np.asarray(contact.pos).round(3).tolist(),
                "force_n": round(float(abs(force[0])), 2),
            }
            raise RuntimeError(
                f"My {link.replace('_', ' ')} bumped into something ({abs(force[0]):.1f} N); stopped walking"
            )

    def act(self, name, args, epoch):
        if epoch != self.epoch:
            raise RuntimeError("Cancelled request")
        if self.fault:
            raise RuntimeError("Simulation needs Reset: " + self.fault)
        result = run_skill(self, name, args, epoch)
        self.robot.motion_goal[:] = 0
        self.publish(force=True)
        return plain(result)

    def prompt(self, text):
        with self.lock:
            if self.status == "Resetting":
                raise ValueError("Wait for the reset to finish")
            if self.fault:
                raise ValueError("Press Reset before sending another command")
            if not self.frame:
                raise ValueError("Simulation is still starting")
            if self.busy:
                # A new instruction replaces the current one at once, so "drop it"
                # or "stop, put it back" never has to wait for a task to finish.
                self.epoch += 1
                self.commands.put(("stop", self.epoch, None))
                self.message("system", "New instruction: the current task was stopped")
            self.busy = True
            self.status = "Looking and deciding"
            epoch = self.epoch
            self.message("user", text)
        threading.Thread(target=self.think, args=(text, epoch), daemon=True).start()

    def think(self, text, epoch):
        try:
            agent.run(self, text, epoch)
        except Exception as exc:  # noqa: BLE001 - every failure must reach the user
            if epoch == self.epoch:
                self.message(
                    "error",
                    f"The request stopped: {exc}",
                    details=exception_details(exc),
                )
        finally:
            with self.lock:
                if epoch == self.epoch:
                    self.busy = False
                    self.status = "Reset required" if self.fault else "Ready"

    def interrupt(self, reset=False):
        with self.lock:
            self.epoch += 1
            self.busy = False
            self.status = "Resetting" if reset else "Stopping"
            self.commands.put(("reset" if reset else "stop", self.epoch, None))
            self.message("system", "Reset requested" if reset else "Stop requested; holding still")

    def configure(self, data):
        options = {
            "scene": ("calibration", "sorting", "clutter", "arena"),
            "camera": ("overview", "front", "top", "head", "left_wrist", "right_wrist"),
            "control": ("normal", "rig"),
            "perception": ("actions",),
        }
        if not isinstance(data, dict) or set(data) != set(self.settings):
            raise ValueError("Provide scene, seed, camera, control and perception")
        if type(data["seed"]) is not int or not 0 <= data["seed"] < 100000:
            raise ValueError("Seed must be an integer from 0 to 99999")
        if any(data[k] not in choices for k, choices in options.items()):
            raise ValueError("Unknown scene/camera/control/perception option")
        with self.lock:
            self.epoch += 1
            self.busy = False
            self.status = "Resetting"
            self.commands.put(("configure", self.epoch, dict(data)))
            self.message("system", "Loading scene: " + json.dumps(data))

    def handle(self, kind, epoch, payload):
        """Run one queued request on the physics thread."""
        if kind == "visual_capture":
            future, camera = payload
            try:
                if epoch != self.epoch:
                    raise RuntimeError("Cancelled capture")
                self.agent_camera = camera
                self.publish(force=True, depth=True)
                sensor = self.visual_sensor
                future.set_result(
                    {
                        "image": sensor["image"],
                        "frame_id": sensor["frame_id"],
                        "camera": sensor["camera"],
                        "robot_status": self.robot_status(),
                    }
                )
            except (ValueError, RuntimeError, OSError) as exc:
                future.set_exception(exc)
        elif kind == "action":
            name, args, future = payload
            try:
                outcome = self.act(name, args, epoch)
            except Exception as exc:  # noqa: BLE001 - always resolve the waiting agent
                self.robot.motion_goal[:] = 0
                if epoch != self.epoch:
                    # Replaced by a newer instruction or Stop: not a fault.
                    future.set_result({"ok": False, "summary": "Stopped for a newer instruction"})
                    return
                details = exception_details(exc)
                self.message("controller_error", str(exc), details={"tool": name, **details})
                if not isinstance(exc, (RuntimeError, ValueError, TypeError, KeyError, OSError)):
                    self.fault = f"Internal controller error: {type(exc).__name__}: {exc}"
                    self.status = "Reset required"
                outcome = {
                    "ok": False,
                    "summary": f"Could not {name.replace('_', ' ')}: {exc}",
                    "reason": str(exc),
                    "error_type": type(exc).__name__,
                }
            future.set_result(outcome)
        elif epoch == self.epoch:
            if kind in ("reset", "configure"):
                try:
                    if kind == "configure":
                        self.settings = payload
                    self.reset()
                except (RuntimeError, ValueError, OSError) as exc:
                    self.fault = str(exc)
                    self.status = "Reset failed"
            else:
                self.robot.hold()
                self.status = "Decelerating" if not self.fault else "Reset required"

    def run(self):
        server = ThreadingHTTPServer(("127.0.0.1", self.port), handler_for(self))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        print(f"Robot console: http://127.0.0.1:{self.port} | {self.model_name}", flush=True)
        try:
            try:
                self.reset()
            except (RuntimeError, ValueError, OSError) as exc:
                self.fault = f"Simulation startup failed: {exc}"
                self.status = "Reset failed"
                self.message("error", self.fault)
            while self.keep_running():
                try:
                    kind, epoch, payload = self.commands.get_nowait()
                except queue.Empty:
                    if self.fault:
                        time.sleep(0.02)
                    else:
                        try:
                            self.tick()
                        except (RuntimeError, ValueError, OSError) as exc:
                            self.fault = str(exc)
                            self.status = "Reset required"
                    continue
                self.handle(kind, epoch, payload)
        finally:
            server.shutdown()
            server.server_close()
            if self.renderer:
                self.renderer.close()
            if self.robot:
                self.robot.close()


def handler_for(console):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, code, body, mime="application/json"):
            if not isinstance(body, bytes):
                body = json.dumps(body, default=str).encode()
            self.send_response(code)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def trusted(self):
            allowed = {f"127.0.0.1:{console.port}", f"localhost:{console.port}"}
            return self.headers.get("Host") in allowed and self.headers.get("Origin") in (
                None,
                *["http://" + host for host in allowed],
            )

        def do_GET(self):
            if not self.trusted():
                return self.respond(403, {"error": "Local console only"})
            path = self.path.split("?")[0]
            if path == "/":
                self.respond(
                    200,
                    Path(__file__).with_name("dashboard.html").read_bytes(),
                    "text/html; charset=utf-8",
                )
            elif path == "/api/state":
                self.respond(200, console.state())
            elif path == "/frame.jpg":
                with console.lock:
                    frame = console.frame
                self.respond(200, frame, "image/jpeg")
            else:
                self.respond(404, {"error": "Not found"})

        def do_POST(self):
            if not self.trusted():
                return self.respond(403, {"error": "Local console only"})
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 8192:
                    raise ValueError("Invalid request size")
                data = json.loads(self.rfile.read(length))
                if self.path == "/api/prompt":
                    prompt = data.get("prompt")
                    if not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 2000:
                        raise ValueError("Enter a prompt of 1–2000 characters")
                    console.prompt(prompt.strip())
                elif self.path in ("/api/stop", "/api/reset"):
                    console.interrupt(reset=self.path.endswith("reset"))
                elif self.path == "/api/configure":
                    console.configure(data)
                else:
                    return self.respond(404, {"error": "Not found"})
                self.respond(202, {"accepted": True})
            except (ValueError, TypeError, AttributeError) as exc:
                self.respond(400, {"error": str(exc)})

    return Handler


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gemini-3.8-flash")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--scene", choices=("sorting", "calibration", "clutter", "arena"), default="sorting")
    parser.add_argument("--control", choices=("normal", "rig"), default="normal")
    args = parser.parse_args()
    console = Console(args.model, args.port)
    console.settings.update(scene=args.scene, control=args.control)
    console.run()
