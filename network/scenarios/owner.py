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


if __name__ == "__main__":
    command, robot = sys.argv[1], sys.argv[2]
    set_status(robot, {"revoke": "revoked", "activate": "active"}[command])
