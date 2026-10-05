# network

Network team: verify that a request really came from a known robot, then decide how far to follow it.

Message format: [`../docs/message-format.md`](../docs/message-format.md)

## Plan
- **W1** Message format and Ed25519 sign and verify, with a test that one changed character fails
- **W2** Key registry (`active` and `revoked`), replay protection with `time` and `nonce`, `to` check
- **W3** Two house containers and one Mosquitto broker on separate Docker networks, one full request and signed reply
- **W4** Spoof, tamper, replay and eavesdrop tests, plus delay and packet loss with `tc netem`

Stack: Python, `cryptography`, Mosquitto, Docker.
