"""Run the house story against a private broker; preserve logs, never touch the service broker."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from unitree_demo.trust_session import ensure_keys, load_trust


def records(path):
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    return [json.loads(line) for line in lines if line.endswith("\n")]


def wait_for(check, processes, timeout):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if check():
            return
        if any(process.poll() is not None for process in processes):
            raise RuntimeError("A child process exited early; inspect the saved logs")
        time.sleep(.1)
    raise TimeoutError("Timed out; inspect the saved logs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=("g1", "go1"), default="g1")
    parser.add_argument("--output", type=Path, required=True, help="New directory for the run logs")
    parser.add_argument("--timeout", type=float, default=180)
    scenario = parser.add_mutually_exclusive_group()
    scenario.add_argument("--missing-peer", action="store_true", help="Expect requester failure when no peer is started")
    scenario.add_argument("--responder-cutoff", action="store_true", help="Expect failure when duration interrupts a return request")
    parser.add_argument("--mosquitto", default=shutil.which("mosquitto") or r"C:\Program Files\mosquitto\mosquitto.exe")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    processes, files = [], []
    protocol, _ = load_trust()
    original_keys = protocol.KEYS_DIR
    try:
        with tempfile.TemporaryDirectory(prefix="test-keys-", dir=output) as keys_dir:
            assert Path(keys_dir).resolve().is_relative_to(output)
            protocol.KEYS_DIR = keys_dir
            ensure_keys(protocol)
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            config = output / "mosquitto.conf"
            config.write_text(f"listener {port} 127.0.0.1\nallow_anonymous true\npersistence false\n")

            def launch(command, name, env=None):
                log = (output / f"{name}.log").open("w", encoding="utf-8")
                files.append(log)
                process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                           env=env, creationflags=flags)
                processes.append(process)
                return process

            broker = launch([args.mosquitto, "-c", str(config)], "broker")

            def broker_ready():
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=.2):
                        return True
                except OSError:
                    return False

            wait_for(broker_ready, [broker], 10)
            env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONIOENCODING": "utf-8"}
            entry = ("import sys; from unitree_demo.trust_session import load_trust; "
                     "load_trust()[0].KEYS_DIR = sys.argv.pop(1); "
                     "from unitree_demo.house_demo import main; main()")

            def start_house(name, *extra):
                return launch([sys.executable, "-c", entry, keys_dir, "--house", name, "--robot", args.robot,
                               "--mqtt-port", str(port), "--headless", "--event-log", str(output / f"{name}.jsonl"), *extra],
                              name, env)

            if args.missing_peer:
                a = start_house("house-1")
                result = a.wait(timeout=args.timeout)
                events = records(output / "house-1.jsonl")
                assert result == 1, f"Missing-peer requester exited {result}, expected 1"
                assert any(e.get("stage") == "FAILED" for e in events)
                assert not any(e.get("stage") == "COMPLETE" for e in events)
                assert not any(e["event"] == "execution_started" for e in events)
                summary = {"robot": args.robot, "missing_peer": True, "requester_exit": result,
                           "false_completion": False}
                (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
                print(json.dumps(summary), flush=True)
                return

            responder_args = ("--trust", "10", "--duration", "5") if args.responder_cutoff else ()
            b = start_house("house-2", *responder_args)
            b_path = output / "house-2.jsonl"
            wait_for(lambda: any(e.get("stage") == "READY" for e in records(b_path)), [broker, b], 40)

            import paho.mqtt.publish as publish

            a_id, b_id = protocol.ROBOTS["a"], protocol.ROBOTS["b"]
            if args.responder_cutoff:
                # Low trust denies the inbound inspection and triggers B's return request.
                request = protocol.make_message(protocol.load_private_key(a_id), a_id, b_id, "please",
                                                action="verify_package", kind="request")
                publish.single(protocol.inbox(b_id), json.dumps(request), hostname="127.0.0.1", port=port, qos=1)
                result = b.wait(timeout=args.timeout)
                events = records(b_path)
                assert result == 1, f"Interrupted responder exited {result}, expected 1"
                assert any(e["event"] == "request_submitted" for e in events)
                assert any(e.get("stage") == "FAILED" and "duration" in e.get("text", "") for e in events)
                assert not any(e.get("stage") in ("RETURN CONFIRMED", "COMPLETE") for e in events)
                assert events[-1]["event"] == "run_ended" and events[-1]["exit_code"] == 1
                summary = {"robot": args.robot, "responder_cutoff": True, "responder_exit": result,
                           "false_completion": False}
                (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
                print(json.dumps(summary), flush=True)
                return
            unsupported = protocol.make_message(protocol.load_private_key(a_id), a_id, b_id, "please",
                                                action="open_door", kind="request")
            malformed = {**unsupported, "from": []}
            for message in (malformed, unsupported):
                publish.single(protocol.inbox(b_id), json.dumps(message), hostname="127.0.0.1", port=port, qos=1)
            wait_for(lambda: any(e["event"] == "execution_denied" and e.get("action") == "open_door"
                                 for e in records(b_path)), [broker, b], 10)
            a = start_house("house-1")
            try:
                result = a.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired as exc:
                raise TimeoutError("House-1 did not finish within the wall-time limit") from exc
            if result != 0:
                raise AssertionError(f"House-1 exited {result}")
            wait_for(lambda: any(e.get("stage") == "RETURN CONFIRMED" for e in records(b_path)), [broker, b], 10)
            a_events, b_events = records(output / "house-1.jsonl"), records(b_path)
            assert any(e.get("stage") == "COMPLETE" for e in a_events)
            assert not any(e.get("stage") in ("FALL", "TIMEOUT", "FAILED") for e in a_events + b_events)
            rejected = [e["reason"] for e in b_events if e["event"] == "authentication_rejected"]
            assert any("malformed" in reason for reason in rejected)
            assert "no signature" in rejected
            assert any(reason in ("replay", "too old or from the future") for reason in rejected)
            summary = {"robot": args.robot, "requester_exit": result, "return_confirmed": True,
                       "malformed_rejected": True, "unsupported_denied": True, "rejections": rejected,
                       "requester_end_sim_time": a_events[-1]["sim_time"]}
            (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(json.dumps(summary), flush=True)
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        for handle in files:
            handle.close()
        protocol.KEYS_DIR = original_keys


if __name__ == "__main__":
    main()
