"""Make a key pair for each robot and a registry of their public keys.

Private keys stay in keys/*.key (never commit them). keys/registry.json is what each
house would get from the owners pairing in person.
"""
import base64, json, os
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from protocol import KEYS_DIR, ROBOTS

os.makedirs(KEYS_DIR, exist_ok=True)
registry = {}
for robot_id in ROBOTS.values():
    key = Ed25519PrivateKey.generate()
    path = os.path.join(KEYS_DIR, robot_id.split("@")[0] + ".key")
    with open(path, "wb") as f:
        f.write(key.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
                                  serialization.NoEncryption()))
    os.chmod(path, 0o600)
    public = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    registry[robot_id] = {"public_key": base64.b64encode(public).decode(), "status": "active"}
    print(f"{robot_id}: private key -> {path}")

with open(os.path.join(KEYS_DIR, "registry.json"), "w") as f:
    json.dump(registry, f, indent=2)
print(f"registry -> {os.path.join(KEYS_DIR, 'registry.json')}")
