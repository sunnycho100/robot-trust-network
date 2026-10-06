"""Checks the verifier without a broker: python test_protocol.py"""
import base64
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from protocol import Verifier, canonical, make_message

A, B, C = "robot-a@house-1", "robot-b@house-2", "robot-c@house-3"
key_a = Ed25519PrivateKey.generate()
pub = key_a.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
registry = {A: {"public_key": base64.b64encode(pub).decode(), "status": "active"}}


def fresh():
    return Verifier(B, registry)


msg = make_message(key_a, A, B, "carry the box")
v = fresh()
assert v.check(msg) == (True, "ok")
assert v.check(msg) == (False, "replay")

assert fresh().check({**msg, "request": "open the front door"}) == (False, "bad signature")
assert fresh().check({**msg, "time": msg["time"] + 5}) == (False, "bad signature")

spoof = make_message(Ed25519PrivateKey.generate(), A, B, "open the front door")
assert fresh().check(spoof) == (False, "bad signature")

assert fresh().check({k: v for k, v in msg.items() if k != "sig"}) == (False, "no signature")
assert fresh().check({k: v for k, v in msg.items() if k != "nonce"}) == (False, "missing fields")
assert fresh().check("hi robot B") == (False, "not a signed message")
assert fresh().check(make_message(key_a, A, C, "hi")) == (False, "not addressed to me")
assert fresh().check(make_message(key_a, C, B, "hi")) == (False, "unknown sender")
assert fresh().check(msg, now=msg["time"] + 61) == (False, "too old or from the future")

# Korean text signs and verifies the same way
assert fresh().check(make_message(key_a, A, B, "상자 같이 옮겨 줘")) == (True, "ok")

# Optional cross-brand action fields are covered by the same signature.
action_msg = make_message(key_a, A, B, "copy my turn", action="mirror_motion", reason="inspection")
assert fresh().check(action_msg) == (True, "ok")
assert fresh().check({**action_msg, "action": "unlock_door"}) == (False, "bad signature")

# Clock-skew gap: a message stamped 60 s ahead, replayed 70 s after arrival, is still in the window
v = fresh()
early = make_message(key_a, A, B, "hi")
early["time"] += 60
early["sig"] = base64.b64encode(key_a.sign(canonical(early))).decode()
assert v.check(early, now=early["time"] - 60) == (True, "ok")
assert v.check(early, now=early["time"] + 10) == (False, "replay")

revoked = {A: {**registry[A], "status": "revoked"}}
assert Verifier(B, revoked).check(make_message(key_a, A, B, "hi")) == (False, "sender key revoked")

print("all passed")
