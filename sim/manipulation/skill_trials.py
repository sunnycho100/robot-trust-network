"""Headless physical trials of the robot skills, without any model API calls.

The vision-language model's only job in a task is to draw a box around the
object the user means. Here an evaluator-only helper draws that box from the
camera's segmentation image instead. The skills still see only RGB-D, and
success is judged afterwards from simulator object positions (evaluation only).

Run: uv run --no-sync python skill_trials.py [task ...] [--seed N]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np

from agent_server import Console
from perception import capture
from skills import execute

OUTPUT = Path("runs/skill-trials")


class HeadlessConsole(Console):
    def __init__(self, seed=7, scene="sorting", record=None):
        super().__init__("offline", 0)
        self.realtime = False
        self.settings.update(seed=seed, scene=scene)
        self.output = OUTPUT
        self.output.mkdir(parents=True, exist_ok=True)
        self.record = record  # (camera, every_n_ticks) to keep frames for a GIF
        self.recorded = []

    def tick(self, epoch=None):
        super().tick(epoch)
        if self.record and self.robot.ticks % self.record[1] == 0:
            from PIL import Image

            self.renderer.update_scene(self.robot.tools.data, camera=self.camera_for(self.record[0]))
            self.recorded.append(Image.fromarray(self.renderer.render()).resize((480, 360)))

    def save_recording(self, path):
        if self.recorded:
            self.recorded[0].save(path, save_all=True, append_images=self.recorded[1:], duration=100, loop=0)
            print(f"     recording: {path} ({len(self.recorded)} frames)")

    def message(self, role, text, **extra):
        stamp = f"{self.robot.ticks * self.robot.dt:7.2f}s" if self.robot else ""
        print(f"  [{stamp}] {role:9s} {text}", flush=True)
        details = extra.get("details") or {}
        if isinstance(details, dict) and "route" in details:
            print("            route:", details["route"], flush=True)


def object_box(console, camera, name):
    """Evaluator stand-in for the model: a tight box around one named object."""
    sensor = capture(console, camera)
    renderer = console.renderer
    renderer.update_scene(console.robot.tools.data, camera=console.camera_for(camera))
    renderer.enable_segmentation_rendering()
    try:
        segmentation = renderer.render().copy()
    finally:
        renderer.disable_segmentation_rendering()
    geom = console.robot.tools.model.bind(console.robot.tools.inner.objects[name]["geom"]).id
    mask = (segmentation[:, :, 0] == geom) & (segmentation[:, :, 1] == mujoco.mjtObj.mjOBJ_GEOM)
    if mask.sum() < 150:
        return sensor, None
    v, u = np.nonzero(mask)
    h, w = mask.shape
    if v.min() == 0 or u.min() == 0 or v.max() == h - 1 or u.max() == w - 1:
        return sensor, None  # cut off at the image edge; a person would use another view
    return sensor, [v.min() / h * 1000, u.min() / w * 1000, (v.max() + 1) / h * 1000, (u.max() + 1) / w * 1000]


def find(console, name):
    for camera in ("head", "overview"):
        sensor, box = object_box(console, camera, name)
        if box is not None:
            return sensor["frame_id"], box
    raise RuntimeError(f"{name} is not visible from the head or overview camera")


def snapshot(console, label):
    """Save what the robot looked like at a failure (evaluation/debug only)."""
    from PIL import Image

    OUTPUT.mkdir(parents=True, exist_ok=True)
    for camera in ("overview", "front"):
        console.renderer.update_scene(console.robot.tools.data, camera=console.camera_for(camera))
        Image.fromarray(console.renderer.render()).save(OUTPUT / f"{label}-{camera}.png")
    print(f"     saved {OUTPUT}/{label}-overview.png and -front.png")


def positions(console):
    data = console.robot.tools.data
    return {
        name: np.asarray(data.bind(obj["joint"]).qpos[:3]).round(4).tolist()
        for name, obj in console.robot.tools.inner.objects.items()
    }


def run(console, name, args):
    print("     truth before (evaluation only):", {k: v for k, v in positions(console).items() if k in ("blue_cube", "red_cube", "yellow_cylinder", "orange_box", "green_box")})
    started = time.monotonic()
    result = execute(console, name, args, console.epoch)
    print(f"  -> {name}: {'OK' if result.get('ok') else 'FAILED'} {result.get('summary')}  ({time.monotonic() - started:.0f}s wall)")
    for key in ("contact", "placement_check"):
        if result.get(key) is not None:
            print(f"     {key}: {json.dumps(result[key], default=str)}")
    if "place" in result:
        print(f"     contact: {json.dumps(result['place'].get('contact'), default=str)}")
    if not result.get("ok"):
        print("     details:", json.dumps(result.get("details"), default=str)[:600])
        snapshot(console, f"{name}-fail")
        debug = getattr(console, "last_contact_debug", None)
        if debug:
            print("     contact debug (evaluation only):", debug)
            console.last_contact_debug = None
        print("     feet pose:", [round(v, 3) for v in __import__("body").feet_pose(console.robot)])
    return result


def on_top(console, source, reference):
    p = positions(console)
    objects = console.robot.tools.inner.objects
    src, ref = np.asarray(p[source]), np.asarray(p[reference])
    ref_half = np.asarray(objects[reference]["half_size"], dtype=float)
    src_half = np.asarray(objects[source]["half_size"], dtype=float)
    ref_top = ref[2] + (ref_half[2] if objects[reference]["shape"] != "cylinder" else ref_half[1])
    src_half_h = src_half[1] if objects[source]["shape"] == "cylinder" else (src_half[0] if objects[source]["shape"] == "sphere" else src_half[2])
    resting = abs(src[2] - src_half_h - ref_top) < 0.012
    offset = float(np.linalg.norm(src[:2] - ref[:2]))
    return {"ok": bool(resting and offset < 0.05), "xy_offset_m": round(offset, 4), "bottom_above_ref_top_m": round(float(src[2] - src_half_h - ref_top), 4)}


def next_to(console, source, reference):
    p = positions(console)
    objects = console.robot.tools.inner.objects
    src, ref = np.asarray(p[source]), np.asarray(p[reference])
    ref_support = ref[2] - (objects[reference]["half_size"][2] if objects[reference]["shape"] not in ("cylinder", "sphere") else (objects[reference]["half_size"][1] if objects[reference]["shape"] == "cylinder" else objects[reference]["half_size"][0]))
    src_half_h = objects[source]["half_size"][1] if objects[source]["shape"] == "cylinder" else (objects[source]["half_size"][0] if objects[source]["shape"] == "sphere" else objects[source]["half_size"][2])
    same_support = abs(src[2] - src_half_h - ref_support) < 0.015
    gap = float(np.linalg.norm(src[:2] - ref[:2]))
    return {"ok": bool(same_support and gap < 0.25), "center_distance_m": round(gap, 4), "same_support": bool(same_support)}


def task_blue_on_red(console):
    frame, box = find(console, "blue_cube")
    picked = run(console, "pick_up", {"frame_id": frame, "object_box": box, "object_name": "blue box"})
    if not picked.get("ok"):
        return {"ok": False, "stage": "pick"}
    frame, ref = find(console, "red_cube")
    run(console, "place", {"relation": "on_top", "frame_id": frame, "reference_box": ref})
    return on_top(console, "blue_cube", "red_cube")


def task_cylinder_on_orange_two_step(console):
    frame, box = find(console, "yellow_cylinder")
    picked = run(console, "pick_up", {"frame_id": frame, "object_box": box, "object_name": "yellow cylinder"})
    if not picked.get("ok"):
        return {"ok": False, "stage": "pick"}
    frame, ref = find(console, "orange_box")
    run(console, "place", {"relation": "on_top", "frame_id": frame, "reference_box": ref})
    return on_top(console, "yellow_cylinder", "orange_box")


def task_cylinder_next_to_green(console):
    frame, box = find(console, "yellow_cylinder")
    picked = run(console, "pick_up", {"frame_id": frame, "object_box": box, "object_name": "yellow cylinder"})
    if not picked.get("ok"):
        return {"ok": False, "stage": "pick"}
    frame, ref = find(console, "green_box")
    run(console, "place", {"relation": "next_to", "frame_id": frame, "reference_box": ref})
    return next_to(console, "yellow_cylinder", "green_box")


def task_drop(console):
    frame, box = find(console, "red_cube")
    picked = run(console, "pick_up", {"frame_id": frame, "object_box": box, "object_name": "red box"})
    if not picked.get("ok"):
        return {"ok": False, "stage": "pick"}
    before = positions(console)["red_cube"]
    run(console, "release", {})

    for _ in range(100):
        console.tick()
    after = positions(console)["red_cube"]
    return {"ok": after[2] < before[2] - 0.03 and not console.held, "fell_m": round(before[2] - after[2], 3)}


def task_red_back(console):
    start = positions(console)["red_cube"]
    frame, box = find(console, "red_cube")
    picked = run(console, "pick_up", {"frame_id": frame, "object_box": box, "object_name": "red box"})
    if not picked.get("ok"):
        return {"ok": False, "stage": "pick"}
    frame, ref = find(console, "blue_cube")
    run(console, "place", {"relation": "next_to", "frame_id": frame, "reference_box": ref})
    frame, box = find(console, "red_cube")
    run(console, "move_object", {"frame_id": frame, "object_box": box, "object_name": "red box", "relation": "back_where_it_was", "memory_id": picked["memory_id"]})
    end = positions(console)["red_cube"]
    error = float(np.linalg.norm(np.subtract(end[:2], start[:2])))
    return {"ok": error < 0.04 and abs(end[2] - start[2]) < 0.01, "return_error_m": round(error, 4)}


TASKS = {
    "blue_on_red": task_blue_on_red,
    "cylinder_on_orange": task_cylinder_on_orange_two_step,
    "cylinder_next_to_green": task_cylinder_next_to_green,
    "drop": task_drop,
    "red_back": task_red_back,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tasks", nargs="*", default=list(TASKS))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--record", default=None, help="camera to record a GIF from, e.g. overview")
    args = parser.parse_args()
    summary = {}
    for name in args.tasks:
        print(f"\n=== {name} (seed {args.seed}) ===", flush=True)
        console = HeadlessConsole(seed=args.seed, record=(args.record, 10) if args.record else None)
        console.reset()
        started = time.monotonic()
        try:
            outcome = TASKS[name](console)
        except Exception as exc:  # noqa: BLE001 - report every trial
            import traceback

            traceback.print_exc()
            outcome = {"ok": False, "exception": f"{type(exc).__name__}: {exc}"}
        outcome["sim_seconds"] = round(console.robot.ticks * console.robot.dt, 1)
        outcome["wall_seconds"] = round(time.monotonic() - started, 1)
        outcome["fell"] = bool(console.robot.state()["fell"])
        summary[name] = outcome
        print(f"=== {name}: {'PASS' if outcome.get('ok') else 'FAIL'} {json.dumps(outcome)}", flush=True)
        console.save_recording(OUTPUT / f"{name}-seed{args.seed}.gif")
        console.robot.close()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / f"summary-seed{args.seed}-{time.strftime('%H%M%S')}.json").write_text(json.dumps(summary, indent=2))
    passed = sum(1 for o in summary.values() if o.get("ok"))
    print(f"\n{passed}/{len(summary)} passed: " + ", ".join(f"{k}={'PASS' if v.get('ok') else 'FAIL'}" for k, v in summary.items()))


if __name__ == "__main__":
    main()
