"""Viewer runtime, keyboard input and CLI options shared by the demos."""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


REEXEC_ENV = "UNITREE_DEMO_RUNNING_UNDER_MJPYTHON"


def ensure_macos_viewer_runtime(headless: bool, module: str) -> None:
    """The macOS MuJoCo viewer must run under mjpython; re-exec once if needed."""
    if headless or platform.system() != "Darwin" or os.environ.get(REEXEC_ENV) == "1":
        return
    mjpython = Path(sys.executable).with_name("mjpython")
    if not mjpython.is_file():
        raise RuntimeError("The macOS MuJoCo viewer needs mjpython in this Python environment.")
    env = os.environ.copy()
    env[REEXEC_ENV] = "1"
    print(f"macOS GUI detected; restarting with {mjpython.name} ...", flush=True)
    os.execve(str(mjpython), [str(mjpython), "-m", module, *sys.argv[1:]], env)


class HumanCommandInput:
    """Receives continuous input from a separate pygame process (see input_bridge)."""

    def __init__(self, title: str, duo: bool = False) -> None:
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(("127.0.0.1", 0))
        self.socket.setblocking(False)
        command = [str(Path(sys.executable).with_name("python")), "-m", "unitree_demo.input_bridge",
                   "--port", str(self.socket.getsockname()[1]), "--title", title]
        if duo:
            command.append("--duo")
        self.process = subprocess.Popen(command, env=os.environ.copy())
        self.command = np.zeros(3, dtype=np.float32)
        self.last_packet = time.monotonic()
        self.reply_address = None

    @property
    def running(self) -> bool:
        return self.process.poll() is None

    def poll(self) -> np.ndarray:
        while True:
            try:
                payload, address = self.socket.recvfrom(12)
            except BlockingIOError:
                break
            if len(payload) == 12:
                self.command[:] = struct.unpack("!fff", payload)
                self.last_packet = time.monotonic()
                self.reply_address = address
        # Fail closed if the input process stalls or disappears.
        if time.monotonic() - self.last_packet > 0.15:
            self.command[:] = 0.0
        return self.command.copy()

    def show_status(self, status: dict) -> None:
        """Send simulation/MQTT status back to the pygame command window."""
        if self.reply_address is None:
            return
        try:
            self.socket.sendto(json.dumps(status, separators=(",", ":")).encode(), self.reply_address)
        except (OSError, TypeError, ValueError):
            pass

    def close(self) -> None:
        self.socket.close()
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--duration", type=float, default=0.0,
                        help="Wall-clock seconds; 0 runs until a window is closed (default)")
    parser.add_argument("--headless", action="store_true", help="Run without human input or viewer")
    parser.add_argument("--command", type=float, nargs=3, metavar=("VX", "VY", "YAW"), default=(0.0, 0.0, 0.0),
                        help="Fixed leader command for --headless runs")


def check_run_arguments(args: argparse.Namespace) -> None:
    if args.duration < 0:
        raise SystemExit("--duration cannot be negative")
    if args.headless and args.duration == 0:
        raise SystemExit("--headless requires a positive --duration")
