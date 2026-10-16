"""House 1's side of each scenario. Prints one JSON line. Run inside the house-1 container."""
import argparse, json, os, socket, sys, threading, time
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import protocol
from common import QOS, STATE, connect, log, parse
from orders import check_update

LINK_LOST_S = 1.0  # no broker ack for this long means the link is gone; stop and treat as DENY


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

    try:
        client = connect(f"{sender}-{time.monotonic_ns()}", [protocol.inbox(sender)], on_message)
    except OSError as e:  # no route to the broker at all
        return {"result": "TIMEOUT", "reason": f"broker unreachable ({e.__class__.__name__}), treated as DENY", "rtt_ms": 0}
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
    out = {"result": reply["result"], "reason": reply["reason"], "rtt_ms": rtt_ms, "task": msg["nonce"],
           "reply_matches": reply["reply_to"] == msg["nonce"] and reply["from"] == receiver}
    out.update({k: reply[k] for k in ("base", "horizon", "order_id", "order_update_id") if k in reply})
    return out


def task(sender, receiver, action, seconds, rate=10.0, timeout=3.0):
    """Ask for a task, then keep it alive with signed steps. Stops by itself if the broker stops acking."""
    answer = request(sender, receiver, action, timeout=timeout)
    if answer["result"] not in ("ALLOW", "PARTIAL"):
        return dict(answer, steps=0, stopped="not granted")
    key = protocol.load_private_key(sender)
    client = connect(f"{sender}-steps-{time.monotonic_ns()}")
    pending, steps, stopped = [], 0, "finished"
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        step = protocol.make_message(key, sender, receiver, "step", kind="step", task=answer["task"])
        pending.append((time.monotonic(), client.publish(protocol.inbox(receiver), json.dumps(step), qos=QOS)))
        steps += 1
        pending = [(t, info) for t, info in pending if not info.is_published()]
        if pending and time.monotonic() - pending[0][0] > LINK_LOST_S:
            stopped = "link lost: broker stopped acknowledging"
            break
        time.sleep(1 / rate)
    if stopped != "finished":
        # No goodbye on a dead link: a clean disconnect would wait on a socket that goes nowhere.
        return dict(answer, steps=steps, stopped=stopped)
    done = protocol.make_message(key, sender, receiver, "step", kind="step", task=answer["task"], done=True)
    client.publish(protocol.inbox(receiver), json.dumps(done), qos=QOS).wait_for_publish(2)
    client.loop_stop()
    client.disconnect()
    return dict(answer, steps=steps, stopped=stopped)


STRANGER = "robot-x@house-9"  # a robot no house has ever paired with


ROUTE_TOPIC = "demo/route"  # what the map page draws; display only, nothing acts on it


def visit(sender, receiver, route, pace=1.5, seconds=120.0):
    """Ask to walk a route in another house. Walk only the released Base, wait at its end, and extend it
    only on an update that passes the signature checks and the order checks (orders.py)."""
    answer = request(sender, receiver, "visit", text="visit " + " ".join(route), route=route)
    state = {"robot": sender, "house": receiver, "route": route, "base": answer.get("base", []),
             "horizon": answer.get("horizon", route), "position": -1, "update_id": 0,
             "result": answer["result"], "status": "", "note": f"{answer['result']}: {answer['reason']}"}
    client = connect(f"{sender}-visit-{time.monotonic_ns()}", [protocol.inbox(sender)])
    lock = threading.Lock()

    def show(status, note=None):
        state["status"] = status
        if note:
            state["note"] = note
        client.publish(ROUTE_TOPIC, json.dumps(state), qos=QOS, retain=True)
        log(dict(state, kind="visit"))

    if answer["result"] not in ("ALLOW", "PARTIAL"):
        show("denied")
        return dict(answer, status="denied")
    order = {"id": answer["order_id"], "by": receiver, "update_id": 0, "base": list(answer["base"])}
    verifier = protocol.Verifier(sender, protocol.load_registry())
    received = []  # raw updates, kept so the replay attack has something real to resend

    def on_message(_c, _u, m):
        msg = parse(m.payload)
        if not isinstance(msg, dict) or msg.get("kind") != "order_update":
            return
        received.append(m.payload.decode())
        with open(os.path.join(STATE, "updates.jsonl"), "a") as f:
            f.write(m.payload.decode() + "\n")
        ok, reason = verifier.check(msg)
        if ok:
            ok, reason = check_update(order, msg)
        with lock:
            if not ok:
                show(state["status"], f"rejected an update: {reason}")
                return
            order["update_id"], order["base"] = msg["order_update_id"], order["base"] + msg["base"][1:]
            state.update(base=order["base"], horizon=msg["horizon"], update_id=msg["order_update_id"])
            show(state["status"], f"update {msg['order_update_id']} accepted: may go on to {msg['base'][-1]}")

    with open(os.path.join(STATE, "order.json"), "w") as f:
        json.dump(order, f)
    client.on_message = on_message
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        with lock:
            if state["position"] < len(order["base"]) - 1:
                state["position"] += 1
                show("walking", f"in {order['base'][state['position']]}")
            elif not state["horizon"]:
                if state["status"] != "arrived":  # stays on the job, still checking anything that comes in
                    show("arrived", f"arrived at {order['base'][-1]}")
            elif state["status"] != "waiting":
                show("waiting", f"waiting at the decision point {order['base'][-1]}: "
                                f"{state['horizon'][0]} is not released yet")
        time.sleep(pace)
    if state["status"] == "waiting":
        show("stopped", "gave up waiting for an update")
    client.loop_stop()
    client.disconnect()
    return {"result": answer["result"], "status": state["status"], "base": order["base"],
            "horizon": state["horizon"], "update_id": order["update_id"], "note": state["note"]}


def bad_route_update(kind, sender, receiver):
    """Attacks on a running visit: a forged update, or a real one sent again."""
    with open(os.path.join(STATE, "order.json")) as f:
        order = json.load(f)
    if kind == "fake_update":  # claims to be house 2, signed with a key nobody paired, opens everything
        payload = json.dumps(protocol.make_message(Ed25519PrivateKey.generate(), receiver, sender, "order update",
                                                   kind="order_update", order_id=order["id"], order_update_id=50,
                                                   base=[order["base"][-1], "safe_room"], horizon=[]))
    else:  # replay_update: a genuine update, captured off the broker (it can read everything, D12)
        with open(os.path.join(STATE, "updates.jsonl")) as f:
            seen = [line for line in f.read().splitlines() if '"' + order["id"] + '"' in line]
        payload = seen[-1] if seen else None
        if payload is None:
            return {"sent": None, "note": "nothing to replay yet: approve a room first so a real update exists"}
    client = connect(f"stranger-{time.monotonic_ns()}")
    client.publish(protocol.inbox(sender), payload, qos=QOS).wait_for_publish(2)
    client.loop_stop()
    client.disconnect()
    return {"sent": kind}


def attack(kind, sender, receiver, other, action="status", text=""):
    """Publish one hostile message to receiver's inbox. Prints its nonce so the runner can find the verdict."""
    text = text or action
    real = protocol.make_message(protocol.load_private_key(sender), sender, receiver, text,
                                 action=action, kind="request")
    if kind == "unsigned":
        msg = {k: v for k, v in real.items() if k != "sig"}
    elif kind == "spoof":    # claims to be sender, signed with a key nobody paired
        msg = protocol.make_message(Ed25519PrivateKey.generate(), sender, receiver, text,
                                    action=action, kind="request")
    elif kind == "stranger":  # honest about who it is, but nobody paired with it
        msg = protocol.make_message(Ed25519PrivateKey.generate(), STRANGER, receiver, text,
                                    action=action, kind="request")
    elif kind == "tamper":   # a real signature, then the action changed on the way
        msg = dict(real, action="unlock_door")
    elif kind == "wrong_to":  # really signed by sender, but for another robot, bounced here
        msg = protocol.make_message(protocol.load_private_key(sender), sender, other, text,
                                    action=action, kind="request")
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
    r.add_argument("--text", default="")
    t = sub.add_parser("task")
    t.add_argument("sender"); t.add_argument("receiver"); t.add_argument("action")
    t.add_argument("--seconds", type=float, default=5.0)
    at = sub.add_parser("attack")
    at.add_argument("kind", choices=["unsigned", "spoof", "stranger", "tamper", "wrong_to", "replay"])
    at.add_argument("sender"); at.add_argument("receiver"); at.add_argument("other")
    at.add_argument("--action", default="status")
    at.add_argument("--text", default="")
    v = sub.add_parser("visit")
    v.add_argument("sender"); v.add_argument("receiver"); v.add_argument("route", nargs="+")
    v.add_argument("--pace", type=float, default=1.5)
    v.add_argument("--seconds", type=float, default=120.0)
    bu = sub.add_parser("route_attack")
    bu.add_argument("kind", choices=["fake_update", "replay_update"])
    bu.add_argument("sender"); bu.add_argument("receiver")
    pr = sub.add_parser("probe")
    pr.add_argument("host"); pr.add_argument("port", type=int)
    a = p.parse_args()
    if a.cmd == "request":
        out = request(a.sender, a.receiver, a.action, text=a.text, timeout=a.timeout)
    elif a.cmd == "task":
        out = task(a.sender, a.receiver, a.action, a.seconds)
    elif a.cmd == "visit":
        out = visit(a.sender, a.receiver, a.route, a.pace, a.seconds)
    elif a.cmd == "route_attack":
        out = bad_route_update(a.kind, a.sender, a.receiver)
    elif a.cmd == "attack":
        out = attack(a.kind, a.sender, a.receiver, a.other, a.action, a.text)
    else:
        out = probe(a.host, a.port)
    print(json.dumps(out))


if __name__ == "__main__":
    sys.exit(main())
