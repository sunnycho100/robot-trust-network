"""Run plain-English requests through the real Gemini agent, headless.

The robot runs exactly as in the app (agent -> skills -> physics thread); only
the window is missing. Object positions are read afterwards for evaluation and
never reach the agent. Needs GEMINI_API_KEY in the environment.

Run: uv run --no-sync python agent_trials.py "put the blue box on the red box" [more requests] --seed 7
"""

from __future__ import annotations

import argparse
import threading
import time

import numpy as np

from agent_server import Console
from trace_format import format_message


class TrialConsole(Console):
    def __init__(self, seed, port):
        super().__init__("gemini-3.8-flash", port)
        self.settings.update(seed=seed, scene="sorting")

    def tick(self, epoch=None):
        # Skills run as fast as the computer allows; standing idle while the
        # model thinks runs in real time so the episode clock is not used up.
        self.realtime = epoch is None
        super().tick(epoch)


def positions(console):
    data = console.robot.tools.data
    return {
        name: np.asarray(data.bind(obj["joint"]).qpos[:3]).round(3).tolist()
        for name, obj in console.robot.tools.inner.objects.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("requests", nargs="+")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument(
        "--interrupt",
        default=None,
        help='an instruction to send mid-task as soon as the robot holds something, e.g. "drop it"',
    )
    args = parser.parse_args()
    console = TrialConsole(args.seed, args.port)
    threading.Thread(target=console.run, daemon=True).start()
    while not console.frame and not console.fault:
        time.sleep(0.2)
    shown = 0
    for request in args.requests:
        print(f"\n######## {request}", flush=True)
        before = positions(console)
        started = time.monotonic()
        console.prompt(request)
        interrupt = args.interrupt
        while time.monotonic() - started < args.timeout:
            state = console.state()
            for row in state["messages"][shown:]:
                print(format_message(row), flush=True)
            shown = len(state["messages"])
            if interrupt and state["busy"] and state["holding"]:
                print(f"######## (mid-task, while holding {state['holding']}) {interrupt}", flush=True)
                console.prompt(interrupt)
                interrupt = None
            elif not state["busy"]:
                break
            time.sleep(0.5)
        after = positions(console)
        moved = {
            name: {"from": before[name], "to": after[name]}
            for name in after
            if np.linalg.norm(np.subtract(after[name], before[name])) > 0.01
        }
        print(f"-------- took {time.monotonic() - started:.0f}s; moved (evaluation only): {moved}", flush=True)
        print(f"-------- holding: {console.state()['holding'] or 'nothing'}", flush=True)


if __name__ == "__main__":
    main()
