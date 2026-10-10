"""Give each house its own private keys and a registry of only the robots it has paired with."""
import base64, json, os, sys
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

HOUSES = {  # house -> robots living there
    "house-1": ["robot-a@house-1", "robot-d@house-1"],
    "house-2": ["robot-b@house-2"],
    "house-3": ["robot-c@house-3"],
}
# House rules: starting trust the owner gives each paired robot, plus stricter thresholds.
RULES = {
    "house-1": {"trust": {"robot-b@house-2": 70, "robot-c@house-3": 70}},
    "house-2": {"trust": {"robot-a@house-1": 70, "robot-d@house-1": 0}},
    "house-3": {"trust": {"robot-a@house-1": 70, "robot-d@house-1": 0}, "minimum": {"join_route": 90}},
}


def main(out):
    public = {}
    for house, robots in HOUSES.items():
        os.makedirs(os.path.join(out, house), exist_ok=True)
        for robot in robots:
            key = Ed25519PrivateKey.generate()
            path = os.path.join(out, house, robot.split("@")[0] + ".key")
            with open(path, "wb") as f:
                f.write(key.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
                                          serialization.NoEncryption()))
            os.chmod(path, 0o600)
            raw = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
            public[robot] = base64.b64encode(raw).decode()
    for house, rules in RULES.items():
        registry = {robot: {"public_key": public[robot], "status": "active"} for robot in rules["trust"]}
        for name, data in (("registry.json", registry), ("rules.json", rules)):
            with open(os.path.join(out, house, name), "w") as f:
                json.dump(data, f, indent=2)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/out")
