"""A receiving household: verify every message, then reply signed. Run: python house.py robot-b@house-2"""
import json, os, sys, threading, time
import protocol
from common import QOS, STATE, connect, log, parse


class House:
    def __init__(self, me):
        self.me = me
        self.key = protocol.load_private_key(me)
        self.registry = {}
        self.registry_mtime = None
        self.reload_registry()
        self.verifier = protocol.Verifier(me, self.registry)
        self.lock = threading.Lock()

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
        sender = msg.get("from") if isinstance(msg, dict) else None
        action = msg.get("action") if isinstance(msg, dict) else None
        log({"kind": "verdict", "to": self.me, "from": sender, "action": action,
             "ok": ok, "reason": reason, "verify_ms": verify_ms})
        if ok:
            self.reply(client, msg, "ACCEPTED", reason)

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
