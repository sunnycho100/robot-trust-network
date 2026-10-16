"""What a house owner does by hand. Run inside that house: python owner.py revoke robot-a@house-1"""
import json, os, sys
from common import STATE, log


def set_status(robot, status):
    path = os.path.join(STATE, "registry.json")
    with open(path) as f:
        registry = json.load(f)
    registry[robot]["status"] = status
    with open(path + ".tmp", "w") as f:
        json.dump(registry, f, indent=2)
    os.replace(path + ".tmp", path)  # the house never reads a half-written registry
    log({"kind": status, "robot": robot})


def approve(robot, room):
    """Let a visitor into one more room; the house sends it a signed order update."""
    path = os.path.join(STATE, "approved.json")
    approved = json.load(open(path)) if os.path.exists(path) else {}
    approved.setdefault(robot, [])
    if room not in approved[robot]:
        approved[robot].append(room)
    with open(path + ".tmp", "w") as f:
        json.dump(approved, f, indent=2)
    os.replace(path + ".tmp", path)
    log({"kind": "approved", "robot": robot, "room": room})


def bad_update(robot):
    """Stand-in for buggy house software: a properly signed update that skips the decision point."""
    import protocol
    from common import QOS, connect
    with open(os.path.join(STATE, "events.jsonl")) as f:
        order = [e for e in map(json.loads, f) if e.get("from") == robot and e.get("base")][-1]
    me = order["to"]
    update = protocol.make_message(protocol.load_private_key(me), me, robot, "order update", kind="order_update",
                                   order_id=order["nonce"], order_update_id=99,
                                   base=[order["route"][0], order["route"][-1]], horizon=[])
    client = connect(f"{me}-owner-{os.getpid()}")
    client.publish(protocol.inbox(robot), json.dumps(update), qos=QOS).wait_for_publish(2)
    client.disconnect()


if __name__ == "__main__":
    command, robot = sys.argv[1], sys.argv[2]
    if command == "approve":
        approve(robot, sys.argv[3])
    elif command == "bad_update":
        bad_update(robot)
    else:
        set_status(robot, {"revoke": "revoked", "activate": "active"}[command])
