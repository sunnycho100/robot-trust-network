"""A receiving household: verify, decide, reply signed. Run: python house.py robot-b@house-2"""
import json, os, sys, threading, time
import policy, protocol
from common import QOS, STATE, connect, log, parse

GAIN, PENALTY = 5, 20  # trust points for a clean interaction, and for asking for a forbidden action


class House:
    def __init__(self, me):
        self.me = me
        self.key = protocol.load_private_key(me)
        self.registry = {}
        self.registry_mtime = None
        self.reload_registry()
        self.verifier = protocol.Verifier(me, self.registry)
        with open(os.path.join(STATE, "rules.json")) as f:
            self.rules = json.load(f)
        self.trust_path = os.path.join(STATE, "trust.json")
        if os.path.exists(self.trust_path):
            with open(self.trust_path) as f:
                self.trust = json.load(f)
        else:
            self.trust = dict(self.rules["trust"])  # the owner's starting scores
        self.last_gain = {}
        self.lock = threading.Lock()

    def trust_of(self, robot):
        return self.trust.get(robot, 0)

    def learn(self, robot, decision, action):
        """Trust moves only on verified messages, so a forger can never lower someone else's score."""
        change = 0
        if decision.result == "ALLOW":
            # One gain per interval, so a robot cannot farm trust with a burst of cheap requests.
            now = time.monotonic()
            if now - self.last_gain.get(robot, -1e9) >= self.rules.get("gain_interval_s", 3600):
                change, self.last_gain[robot] = GAIN, now
        elif action in policy.NEVER_ALLOWED:
            # Only the shared never list costs trust: a visitor cannot know one owner's private rules.
            change = -PENALTY
        if change:
            self.trust[robot] = max(0, min(100, self.trust_of(robot) + change))
            with open(self.trust_path + ".tmp", "w") as f:
                json.dump(self.trust, f, indent=2)
            os.replace(self.trust_path + ".tmp", self.trust_path)
        return change

    def reload_registry(self):
        """The owner edits registry.json by hand; pick changes up without a restart."""
        path = os.path.join(STATE, "registry.json")
        mtime = os.stat(path).st_mtime_ns
        if mtime != self.registry_mtime:
            self.registry.clear()
            self.registry.update(protocol.load_registry())  # same dict the Verifier holds
            self.registry_mtime = mtime

    def on_message(self, client, _userdata, mqtt_message):
        msg = parse(mqtt_message.payload)
        with self.lock:
            self.reload_registry()
            start = time.perf_counter()
            ok, reason = self.verifier.check(msg)
            verify_ms = (time.perf_counter() - start) * 1000
            fields = msg if isinstance(msg, dict) else {}
            event = {"kind": "verdict", "to": self.me, "from": fields.get("from"), "action": fields.get("action"),
                     "nonce": fields.get("nonce"), "ok": ok, "result": "REJECTED", "reason": reason,
                     "verify_ms": verify_ms}
            if ok and msg.get("kind") == "request":
                start = time.perf_counter()
                decision = policy.decide(msg, self.trust_of(msg["from"]), self.rules)
                decide_ms = (time.perf_counter() - start) * 1000
                event.update(result=decision.result, reason=decision.reason, trust=self.trust_of(msg["from"]),
                             decide_ms=decide_ms, trust_change=self.learn(msg["from"], decision, msg.get("action")))
            log(event)
        # Only verified requests get an answer; a forged message gets silence, not a signed echo.
        if ok and msg.get("kind") == "request":
            self.reply(client, msg, event["result"], event["reason"])

    def reply(self, client, msg, result, reason, **fields):
        signed = protocol.make_message(self.key, self.me, msg["from"], result, kind="response",
                                       reply_to=msg["nonce"], result=result, reason=reason, **fields)
        client.publish(protocol.inbox(msg["from"]), json.dumps(signed, ensure_ascii=False), qos=QOS)


def main(me):
    house = House(me)
    connect(me, [protocol.inbox(me)], house.on_message)
    log({"kind": "online", "robot": me})
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main(sys.argv[1])
