# network

Network team: verify that a request really came from a known robot, then decide how far to follow it.

Message format: [`../docs/message-format.md`](../docs/message-format.md)

## Walkthroughs
1. [MQTT basics: two houses talking through a broker](01-mqtt-basics.md)
2. [Signed messages: proving who sent a request](02-signed-messages.md)

## Signed robots (W1 and W2)

Each robot signs what it sends and checks everything it receives: sender in the registry, valid signature, addressed to it, recent `time`, unseen `nonce`.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python keygen.py          # key pair per robot + keys/registry.json (keys/ is never committed)
python test_protocol.py   # checks the verifier without a broker
```

With the broker running (`./dashboard/start.sh`), one tab per robot. Type a line and press Enter to send it, signed, to the other robot:

```bash
python robot.py a
python robot.py b
```

Attacks on robot B, each should be REJECTED:

```bash
python attack.py unsigned   # no signature
python attack.py spoof      # signed with the attacker's own key
python attack.py replay     # resend a real message from A (start it, then send from A)
python attack.py tamper     # change a real message from A (start it, then send from A)
```

## Link monitor
A browser page showing which robots are connected, the messages between them and the broker log.

```bash
brew install mosquitto
./dashboard/start.sh
```

This starts the broker (ports 1883 and 9001, this laptop only) and opens the page. Open the page before connecting the robots, since it only sees connections that happen while it is open.

## Automatic presentation demo

Instead of typing in two robot terminals, run the five-stage story after the
broker/dashboard is ready:

```bash
python keygen.py                 # first run only
python demo.py --delay 0.9
```

The monitor shows: discover → signed request → delayed verified response →
blocked unsigned attack → authenticated cooperation. `--delay` is an explicit
robot decision/reaction pause so the exchange is readable to an audience; it is
not presented as measured broker latency. Robot B applies both layers: Ed25519,
freshness and replay verification first, then the owner policy and trust score.
Try `--trust 45` to see a `carry_together` request reduced to partial permission,
or `--trust 10` to see it denied.

## Plan
- **W1** Message format and Ed25519 sign and verify, with a test that one changed character fails
- **W2** Key registry (`active` and `revoked`), replay protection with `time` and `nonce`, `to` check
- **W3** Two house containers and one Mosquitto broker on separate Docker networks, one full request and signed reply
- **W4** Spoof, tamper, replay and eavesdrop tests, plus delay and packet loss with `tc netem`

Stack: Python, `cryptography`, Mosquitto, Docker.
