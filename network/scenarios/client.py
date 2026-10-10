"""House 1's side of each scenario. Prints one JSON line. Run inside the house-1 container."""
import argparse, json, socket, sys, threading, time
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
    pr = sub.add_parser("probe")
    pr.add_argument("host"); pr.add_argument("port", type=int)
    a = p.parse_args()
    if a.cmd == "request":
        out = request(a.sender, a.receiver, a.action, timeout=a.timeout)
    else:
        out = probe(a.host, a.port)
    print(json.dumps(out))


if __name__ == "__main__":
    sys.exit(main())
