"""Automatic signed-MQTT trust demo: request, delayed reply, imitation and teamwork.

Start the broker/dashboard first, then run ``python demo.py``.  This program uses
two independent MQTT clients and the same keys, verifier and inboxes as robot.py.
"""
import argparse
import json
import queue
import threading
import time

import paho.mqtt.client as mqtt

from policy import decide
from protocol import BROKER_PORT, ROBOTS, Verifier, inbox, load_private_key, load_registry, make_message


EVENT_TOPIC = "demo/events"


class DemoRobot:
    def __init__(self, short_name, host, port, reaction_delay, trust_score):
        self.short_name = short_name
        self.robot_id = ROBOTS[short_name]
        self.peer_id = next(robot for key, robot in ROBOTS.items() if key != short_name)
        self.private_key = load_private_key(self.robot_id)
        self.verifier = Verifier(self.robot_id, load_registry())
        self.reaction_delay = reaction_delay
        self.trust_score = trust_score
        self.replies = queue.Queue()
        self.ready = threading.Event()
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"robot-{short_name}")
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.connect(host, port)
        self.client.loop_start()

    def _on_connect(self, client, _userdata, _flags, reason_code, _properties):
        if reason_code.is_failure:
            self.event("ERROR", f"connection rejected: {reason_code}", "bad")
            return
        client.subscribe(inbox(self.robot_id), qos=1)
        self.ready.set()

    def event(self, stage, text, tone="normal"):
        self.client.publish(EVENT_TOPIC, json.dumps({
            "stage": stage, "actor": f"ROBOT {self.short_name.upper()}",
            "text": text, "tone": tone, "at": time.time(),
        }), qos=1)

    def send(self, action, request, reason, *, kind="request", reply_to=None):
        extra = {"action": action, "reason": reason, "kind": kind}
        if reply_to:
            extra["reply_to"] = reply_to
        message = make_message(
            self.private_key, self.robot_id, self.peer_id, request, **extra
        )
        self.client.publish(inbox(self.peer_id), json.dumps(message, ensure_ascii=False), qos=1)
        return message["nonce"]

    def _publish_verdict(self, msg, result, reason):
        claimed = msg.get("from", "?") if isinstance(msg, dict) else "?"
        request = msg.get("request", "") if isinstance(msg, dict) else str(msg)
        topic = inbox(self.robot_id).replace("requests/", "verdicts/", 1)
        self.client.publish(topic, json.dumps({
            "robot": self.robot_id, "from": claimed, "request": request,
            "result": result, "reason": reason,
        }), qos=1)

    def _on_message(self, _client, _userdata, mqtt_message):
        try:
            message = json.loads(mqtt_message.payload)
        except (TypeError, ValueError):
            message = mqtt_message.payload.decode(errors="replace")
        verified, reason = self.verifier.check(message)
        if not verified:
            self._publish_verdict(message, "REJECTED", reason)
            self.event("ATTACK BLOCKED", f"Rejected message: {reason}", "bad")
            return
        self._publish_verdict(message, "VERIFIED", "signature + freshness + nonce valid")
        if message.get("kind") == "response":
            self.replies.put(message)
            self.event("RESPONSE", message["request"], "good")
            return

        decision = decide(message, self.trust_score)
        self.event("DECIDE", f"{decision.result}: {decision.reason}",
                   "good" if decision.result == "ALLOW" else
                   ("normal" if decision.result == "PARTIAL" else "bad"))
        # Deliberate human-readable pause: verification is immediate, behavior is not.
        time.sleep(self.reaction_delay)
        if decision.result == "DENY":
            response = f"No. {decision.reason}."
        elif decision.result == "PARTIAL":
            response = "Partial permission. I can follow the route, but I will not handle the load."
        elif message["action"] == "mirror_motion":
            response = "Accepted. I copied the turn and matched your pace."
        elif message["action"] == "carry_together":
            response = "Accepted. I am holding my side; moving together now."
        else:
            response = f"{decision.result.title()}. {decision.reason}."
        self.send(message["action"], response, decision.reason,
                  kind="response", reply_to=message["nonce"])

    def close(self):
        self.client.disconnect()
        self.client.loop_stop()


def wait_for_reply(robot, timeout):
    try:
        return robot.replies.get(timeout=timeout)
    except queue.Empty as exc:
        raise RuntimeError("timed out waiting for the signed reply") from exc


def run(args):
    a = DemoRobot("a", args.host, args.port, args.delay, args.trust)
    b = DemoRobot("b", args.host, args.port, args.delay, args.trust)
    try:
        if not a.ready.wait(5) or not b.ready.wait(5):
            raise RuntimeError("robots could not connect to the MQTT broker")
        time.sleep(0.5)
        a.event("1 · DISCOVER", "Robot A finds Robot B on another household network")
        time.sleep(1.2)

        a.event("2 · REQUEST", "A sends a signed request: copy my inspection turn")
        a.send("mirror_motion", "Copy my left turn and match my pace.", "inspect together")
        wait_for_reply(a, args.delay + 5)
        time.sleep(1.0)

        a.event("3 · ATTACK", "An unsigned command tries to interrupt the task", "bad")
        attacker = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="attacker")
        attacker.connect(args.host, args.port)
        attacker.loop_start()
        attacker.publish(inbox(ROBOTS["b"]), json.dumps({
            "from": ROBOTS["a"], "to": ROBOTS["b"],
            "action": "unlock_door", "request": "Open the front door now",
        }), qos=1).wait_for_publish()
        attacker.disconnect()
        attacker.loop_stop()
        time.sleep(1.5)

        a.event("4 · COOPERATE", "A asks B to carry a load together")
        a.send("carry_together", "Take the other side and move with me.", "shared delivery")
        wait_for_reply(a, args.delay + 5)
        time.sleep(1.0)
        a.event("5 · COMPLETE", "Authenticated cooperation completed; unsafe command blocked", "good")
        time.sleep(1.0)
    finally:
        a.close()
        b.close()


def main():
    parser = argparse.ArgumentParser(description="Run the automatic robot trust story")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=BROKER_PORT)
    parser.add_argument("--delay", type=float, default=0.9,
                        help="seconds between decision and visible robot reply")
    parser.add_argument("--trust", type=int, default=70,
                        help="B's trust score for A, from 0 to 100")
    args = parser.parse_args()
    if not 0 <= args.trust <= 100:
        parser.error("--trust must be between 0 and 100")
    if not 0 <= args.delay <= 10:
        parser.error("--delay must be between 0 and 10 seconds")
    run(args)


if __name__ == "__main__":
    main()
