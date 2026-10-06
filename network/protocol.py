"""Signed request messages: build, sign and verify. Format: ../docs/message-format.md"""
import base64, json, os, time
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

BROKER_PORT = int(os.environ.get("MQTT_PORT", 1883))
WINDOW = 60  # seconds a message stays valid, and the allowed clock difference
KEYS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "keys")

ROBOTS = {"a": "robot-a@house-1", "b": "robot-b@house-2"}


def inbox(robot_id):
    """robot-b@house-2 -> requests/house-2/robot-b"""
    robot, house = robot_id.split("@")
    return f"requests/{house}/{robot}"


def canonical(msg):
    """Same message -> same bytes, in any language: sorted keys, no spaces, UTF-8, no sig."""
    body = {k: v for k, v in msg.items() if k != "sig"}
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def make_message(private_key, sender, receiver, request):
    msg = {"from": sender, "to": receiver, "request": request,
           "time": int(time.time()), "nonce": os.urandom(8).hex()}
    msg["sig"] = base64.b64encode(private_key.sign(canonical(msg))).decode()
    return msg


class Verifier:
    """One per robot. Keeps the key registry and the nonces it has already seen."""

    def __init__(self, me, registry):
        self.me = me
        self.registry = registry  # {robot_id: {"public_key": base64, "status": "active" | "revoked"}}
        self.seen = {}            # nonce -> message time

    def check(self, msg, now=None):
        """Returns (True, "ok") or (False, reason). Checks in the order of the message format doc."""
        now = time.time() if now is None else now
        if not isinstance(msg, dict):
            return False, "not a signed message"
        if "sig" not in msg:
            return False, "no signature"
        if not all(k in msg for k in ("from", "to", "time", "nonce")):
            return False, "missing fields"
        if msg["to"] != self.me:
            return False, "not addressed to me"
        entry = self.registry.get(msg["from"])
        if entry is None:
            return False, "unknown sender"
        if entry["status"] != "active":
            return False, "sender key revoked"
        try:
            key = Ed25519PublicKey.from_public_bytes(base64.b64decode(entry["public_key"]))
            key.verify(base64.b64decode(msg["sig"]), canonical(msg))
        except (InvalidSignature, ValueError, TypeError):
            return False, "bad signature"
        if abs(now - msg["time"]) > WINDOW:
            return False, "too old or from the future"
        # Forget a nonce only once its own message time is out of the window, not 60 s after it arrived
        self.seen = {n: t for n, t in self.seen.items() if t + WINDOW >= now}
        if msg["nonce"] in self.seen:
            return False, "replay"
        self.seen[msg["nonce"]] = msg["time"]
        return True, "ok"


def load_private_key(robot_id):
    with open(os.path.join(KEYS_DIR, robot_id.split("@")[0] + ".key"), "rb") as f:
        return Ed25519PrivateKey.from_private_bytes(f.read())


def load_registry():
    with open(os.path.join(KEYS_DIR, "registry.json")) as f:
        return json.load(f)
