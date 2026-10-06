# 02. Signed messages: proving who sent a request

Walkthrough 01 showed that anyone can send "open the front door" as robot A, and robot B cannot tell. This one fixes it with a digital signature.

## The idea in one paragraph

Each robot has a key pair. The **private key** never leaves the robot. The **public key** is shared with other houses through a registry. Robot A signs every message with its private key; robot B checks the signature with robot A's public key. The signature covers every field, so changing any of them breaks it. This does not hide the message (anyone on the broker can still read it). It proves who wrote it and that nobody changed it.

Algorithm: **Ed25519** (RFC 8032), a public standard that uses SHA-512 internally. Keys are 32 bytes, signatures 64 bytes. Its security comes from the private key staying secret, not from the algorithm being secret.

## Step 1. Try it by hand

```bash
cd ~/Documents/projects/robot/network
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3
```

```python
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
private_key = Ed25519PrivateKey.generate()
public_key = private_key.public_key()
msg = b"hi robot B, this is robot A"
sig = private_key.sign(msg)
len(sig)                                                # 64
public_key.verify(sig, msg)                             # no output: valid
public_key.verify(sig, b"open the front door")          # InvalidSignature: tamper caught
attacker_key = Ed25519PrivateKey.generate()
public_key.verify(attacker_key.sign(b"open the front door"), b"open the front door")  # InvalidSignature: spoof caught
public_key.verify(sig, msg)                             # no output: a replay still passes!
```

The last line is the catch: a signature proves **who** and **what**, not **when**. A recorded real message would verify forever.

## Step 2. Time and nonce stop replays

Every message carries two more signed fields:

- `time`: when it was sent. Robot B rejects anything more than 60 s off its own clock.
- `nonce`: a random one-time number (8 random bytes). Robot B remembers the ones it has seen and rejects a repeat.

An attacker who copies a message cannot fix either one: keeping the nonce gets it rejected as a replay, and changing the nonce or time breaks the signature. A nonce is kept until its own `time` + 60 s has passed, not 60 s after it arrived, so a robot whose clock runs ahead cannot open a gap.

Messages are JSON, with the signature as one field. Full format: [`../docs/message-format.md`](../docs/message-format.md).

```json
{"from": "robot-a@house-1", "to": "robot-b@house-2", "request": "hello",
 "time": 1791237458, "nonce": "8f3c1a9e2b7d4f60", "sig": "<64 bytes, base64>"}
```

## Step 3. Run two signed robots

One-time setup, then check the verifier without a broker:

```bash
python keygen.py          # keys/robot-a.key, keys/robot-b.key, keys/registry.json
python test_protocol.py   # all passed
```

Stop anything from walkthrough 01 first. The old `mosquitto_sub -i robot-a` uses the same name as the new robot, and the broker keeps kicking one off for the other:

```bash
pkill -x mosquitto; pkill -x mosquitto_sub
```

| Tab | Command |
|---|---|
| 1 | `cd ~/Documents/projects/robot/network/dashboard && ./start.sh` (broker and link monitor) |
| 2 | `cd ~/Documents/projects/robot/network && source .venv/bin/activate && python robot.py a` |
| 3 | `cd ~/Documents/projects/robot/network && source .venv/bin/activate && python robot.py b` |
| 4 | attacks (same `cd` and `source` first) |
| 5 | optional raw view: `mosquitto_sub -h localhost -t "requests/#" -v` |

Type a line in robot A's tab and press Enter. Robot B prints `ACCEPTED (ok) from robot-a@house-1: ...`. Start both robots before sending: right now the broker does not keep messages for a robot that is offline.

## Step 4. Attack robot B

| Command | What the attacker does | Real-world version | Robot B says |
|---|---|---|---|
| `python attack.py unsigned` | "open the front door" with no signature | Shouting a command, or imitating a robot's voice | REJECTED (no signature) |
| `python attack.py spoof` | Signs it with its own key, claiming to be robot A | A forged ID | REJECTED (bad signature) |
| `python attack.py replay` | Copies a real message from A and resends it | Recording and playing back | REJECTED (replay) |
| `python attack.py tamper` | Copies a real message from A and changes the request | Editing a signed check | REJECTED (bad signature) |

For `replay` and `tamper`, start the attack first, then send something from robot A. The real message is ACCEPTED and the copy REJECTED.

In the raw view, the spoof looks exactly like a real message: same `from`, a signature of the same length, fresh `time` and `nonce`. Only the check can tell them apart.

## Stopping everything

```bash
pkill -x mosquitto          # the broker only (plain "pkill mosquitto" also kills mosquitto_sub)
pkill -f "robot.py"         # the Python robots
pgrep -fl "mosquitto|robot.py"   # no output means everything stopped
```

## Still open
- How keys get into each house's registry in real life (proposal: owners pair in person, for example with a QR code).
- A signature proves a request is genuine, not that it is safe. Deciding how far to follow a genuine "open the front door" is the trust layer's job.
