"""House 1's side of each scenario. Prints one JSON line. Run inside the house-1 container."""
import argparse, json, socket, sys, threading, time
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import protocol
from common import QOS, connect, parse


def request(sender, receiver, action, text="", timeout=3.0, key=None, **fields):
    """Send one signed request and wait for the signed reply to it. Times out to a safe DENY."""
    registry = protocol.load_registry()
    verifier = protocol.Verifier(sender, registry)
    replies, got, sent = {}, threading.Event(), set()

    def on_message(_c, _u, m):
        msg = parse(m.payload)
        ok, reason = verifier.check(msg)
        if isinstance(msg, dict) and msg.get("reply_to") in sent:
            replies.update(msg=msg, ok=ok, reason=reason)
            got.set()

    client = connect(f"{sender}-{time.monotonic_ns()}", [protocol.inbox(sender)], on_message)
    time.sleep(0.2)  # let the subscription land before the reply can come back
    msg = protocol.make_message(key or protocol.load_private_key(sender), sender, receiver,
                                text or action, action=action, kind="request", **fields)
    sent.add(msg["nonce"])
    start = time.perf_counter()
    client.publish(protocol.inbox(receiver), json.dumps(msg, ensure_ascii=False), qos=QOS)
    arrived = got.wait(timeout)
    rtt_ms = (time.perf_counter() - start) * 1000
    client.loop_stop()
    client.disconnect()
    if not arrived:
        return {"result": "TIMEOUT", "reason": "no reply, treated as DENY", "rtt_ms": rtt_ms}
    if not replies["ok"]:
        return {"result": "BAD_REPLY", "reason": replies["reason"], "rtt_ms": rtt_ms}
    reply = replies["msg"]
    return {"result": reply["result"], "reason": reply["reason"], "rtt_ms": rtt_ms,
            "reply_matches": reply["reply_to"] == msg["nonce"] and reply["from"] == receiver}


def attack(kind, sender, receiver, other):
    """Publish one hostile message to receiver's inbox. Prints its nonce so the runner can find the verdict."""
    real = protocol.make_message(protocol.load_private_key(sender), sender, receiver, "status",
                                 action="status", kind="request")
    if kind == "unsigned":
        msg = {k: v for k, v in real.items() if k != "sig"}
    elif kind == "spoof":    # claims to be sender, signed with a key nobody paired
        msg = protocol.make_message(Ed25519PrivateKey.generate(), sender, receiver, "status",
                                    action="status", kind="request")
    elif kind == "tamper":   # a real signature, then the action changed on the way
        msg = dict(real, action="unlock_door")
    elif kind == "wrong_to":  # really signed by sender, but for another robot, bounced here
        msg = protocol.make_message(protocol.load_private_key(sender), sender, other, "status",
                                    action="status", kind="request")
    else:                    # replay: the genuine message, sent twice
        msg = real
    client = connect(f"stranger-{time.monotonic_ns()}")
    copies = 2 if kind == "replay" else 1
    for _ in range(copies):
        client.publish(protocol.inbox(receiver), json.dumps(msg), qos=QOS).wait_for_publish(2)
        time.sleep(0.3)
    client.loop_stop()
    client.disconnect()
    return {"nonce": msg["nonce"], "copies": copies}


def probe(host, port):
    """Can this house open a TCP connection to host? Refused still means the host is reachable."""
    try:
        socket.create_connection((host, port), timeout=2).close()
        return {"reachable": True, "error": None}
    except ConnectionRefusedError as e:
        return {"reachable": True, "error": repr(e)}
    except OSError as e:
        return {"reachable": False, "error": repr(e)}


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("request")
    r.add_argument("sender"); r.add_argument("receiver"); r.add_argument("action")
    r.add_argument("--timeout", type=float, default=3.0)
    at = sub.add_parser("attack")
    at.add_argument("kind", choices=["unsigned", "spoof", "tamper", "wrong_to", "replay"])
    at.add_argument("sender"); at.add_argument("receiver"); at.add_argument("other")
    pr = sub.add_parser("probe")
    pr.add_argument("host"); pr.add_argument("port", type=int)
    a = p.parse_args()
    if a.cmd == "request":
        out = request(a.sender, a.receiver, a.action, timeout=a.timeout)
    elif a.cmd == "attack":
        out = attack(a.kind, a.sender, a.receiver, a.other)
    else:
        out = probe(a.host, a.port)
    print(json.dumps(out))


if __name__ == "__main__":
    sys.exit(main())
