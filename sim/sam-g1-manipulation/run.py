"""Run G1 manipulation bring-up: smoke test, Cartesian reach, or live console."""

from __future__ import annotations

import argparse
import json
import queue
import shlex
import sys
import threading
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from controller import ManipulationRuntime


def run_for(robot, seconds, viewer=None):
    for _ in range(round(seconds / robot.dt)):
        start = time.monotonic()
        state = robot.tick()
        if viewer is not None:
            if not viewer.is_running():
                raise KeyboardInterrupt
            viewer.sync()
            time.sleep(max(0, robot.dt - (time.monotonic() - start)))
    return state


def smoke(robot, viewer=None):
    """Test real finger travel and a state-dependent reach; no grasp claim."""
    robot.grip("left", 0)
    robot.grip("right", 0)
    opened = run_for(robot, 2, viewer)
    robot.grip("left", 1)
    robot.grip("right", 1)
    closed = run_for(robot, 2, viewer)
    finger_travel = {
        side: float(
            np.min(
                np.abs(
                    np.asarray(opened["finger_positions_m"][side])
                    - np.asarray(closed["finger_positions_m"][side])
                )
            )
        )
        for side in ("left", "right")
    }
    robot.grip("left", 0)
    # Relative to measured pose, not a replayed joint sequence.
    target = np.asarray(closed["left_hand_pos"]) + [0.08, 0.02, 0.06]
    robot.reach("left", target)
    final = run_for(robot, 6, viewer)
    error = final["target_errors_m"]["left"]
    v = np.asarray([row["command_velocity"] for row in robot.trace])
    a = np.asarray([row["command_acceleration"] for row in robot.trace])
    jerk = np.diff(a, axis=0) / robot.dt
    checks = {
        "no_fall": not final["fell"],
        "left_fingers_moved": finger_travel["left"] > 0.025,
        "right_fingers_moved": finger_travel["right"] > 0.025,
        "reach_within_2cm": error < 0.02,
        "arm_command_velocity_limited": float(np.abs(v[:, :14]).max()) <= 0.8001,
        "command_acceleration_limited": float(np.abs(a).max()) <= 2.0001,
        "command_jerk_limited": float(np.abs(jerk).max()) <= 10.001,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "reach_error_m": error,
        "finger_travel_m": finger_travel,
        "max_command_jerk": float(np.abs(jerk).max()),
        "final_state": final,
        "scope": "Stationary reach and empty-gripper physics test; not object pickup or a benchmark score",
    }


def console(robot, viewer):
    """Read input off-thread so waiting for a command does not freeze physics."""
    commands = queue.Queue()

    def read():
        try:
            while True:
                commands.put(input("g1> "))
        except EOFError:
            commands.put("quit")

    print(
        "Commands: state | reach left/right X Y Z | grip left/right 0..1 | height METRES | quit"
    )
    print(
        "World coordinates in metres. Free-space reaching only; no collision planner."
    )
    threading.Thread(target=read, daemon=True).start()
    while True:
        start = time.monotonic()
        if not commands.empty():
            try:
                words = shlex.split(commands.get())
                if words == ["quit"]:
                    return
                if words == ["state"]:
                    print(json.dumps(robot.state(), indent=2))
                elif len(words) == 5 and words[0] == "reach":
                    robot.reach(words[1], [float(x) for x in words[2:]])
                elif len(words) == 3 and words[0] == "grip":
                    robot.grip(words[1], float(words[2]))
                elif len(words) == 2 and words[0] == "height":
                    robot.posture(float(words[1]))
                else:
                    print("Expected state, reach SIDE X Y Z, grip SIDE VALUE, or quit")
            except ValueError as exc:
                print(f"Rejected: {exc}")
        robot.tick()
        if viewer is not None:
            if not viewer.is_running():
                return
            viewer.sync()
        time.sleep(max(0, robot.dt - (time.monotonic() - start)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "mode",
        choices=("smoke", "reach", "console", "pick"),
        nargs="?",
        default="smoke",
    )
    p.add_argument("--seed", type=int, default=7)
    p.add_argument(
        "--target",
        type=float,
        nargs=3,
        help="world x y z; default uses the scene target",
    )
    p.add_argument("--side", choices=("left", "right"), default="left")
    p.add_argument(
        "--viewer", action="store_true", help="on macOS launch with uv run mjpython"
    )
    p.add_argument("--output", type=Path, default=Path("runs/latest"))
    p.add_argument("--snapshot", action="store_true", help="save a rendered final PNG")
    p.add_argument(
        "--record",
        action="store_true",
        help="save demo.mp4; requires ffmpeg and graphics access",
    )
    args = p.parse_args()
    if sys.platform == "darwin" and args.viewer and (args.record or args.snapshot):
        p.error(
            "On macOS use run.py without --viewer for recording/snapshots; use view.py for the live window"
        )
    robot = ManipulationRuntime(
        args.seed, scene="pinch" if args.mode == "pick" else "reach"
    )
    args.output.mkdir(parents=True, exist_ok=True)
    result = {"passed": False}
    try:
        if args.record:
            robot.tools.start_recording(
                args.output / "demo.mp4", views=("front", "rec_side")
            )
        if args.viewer:
            import mujoco.viewer

            context = mujoco.viewer.launch_passive(robot.tools.model, robot.tools.data)
        else:
            context = nullcontext(None)
        with context as viewer:
            if viewer is not None:
                viewer.cam.lookat[:] = [0.1, 0, 0.8]
                viewer.cam.distance = 2.4
                viewer.cam.azimuth = 140
                viewer.cam.elevation = -15
            if args.mode == "smoke":
                result = smoke(robot, viewer)
            elif args.mode == "reach":
                target = (
                    args.target
                    if args.target is not None
                    else robot.state()["target_pos"]
                )
                robot.reach(args.side, target)
                final = run_for(robot, 8, viewer)
                result = {
                    "passed": final["target_errors_m"][args.side] < 0.02,
                    "final_state": final,
                }
            elif args.mode == "pick":
                from pick_place import execute

                result = execute(
                    robot, lambda seconds: run_for(robot, seconds, viewer), args.target
                )
            else:
                console(robot, viewer)
                result = {"passed": True, "final_state": robot.state()}
    except (RuntimeError, ValueError, KeyboardInterrupt) as exc:
        result = {
            "passed": False,
            "error": str(exc) or "Interrupted",
            "final_state": robot.state(),
        }
    finally:
        robot.tools.stop_recording()
        if hasattr(robot, "skill_events"):
            result["events"] = robot.skill_events
        try:
            if args.snapshot:
                from PIL import Image

                Image.fromarray(robot.tools.render("front")).save(
                    args.output / "final.png"
                )
        except (RuntimeError, ValueError, OSError) as exc:
            result["render_error"] = str(exc)
        finally:
            (args.output / "result.json").write_text(
                json.dumps(result, indent=2) + "\n"
            )
            with (args.output / "trace.jsonl").open("w") as f:
                for row in robot.trace:
                    f.write(json.dumps(row) + "\n")
            robot.close()
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] and "render_error" not in result else 1)


if __name__ == "__main__":
    main()
