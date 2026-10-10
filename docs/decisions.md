# Decision log

Every major architecture and communication choice, with the alternatives that existed. Newest at the bottom of each section. To overrule one, change its status to `overruled`, write the new choice as a new entry, and link back.

Status: `agreed` (the team or Sunghwan chose it), `needs review` (made by an agent or one person, nobody has checked it yet), `overruled`.

Entry format:

```
### Dn. Title
- Date, by:
- Decision:
- Alternatives:
- Why:
- Revisit if:
- Status:
```

## Decisions so far

### D1. MQTT through one broker as the link between households
- Date, by: 2026-10-05, Sunghwan with Claude
- Decision: households talk only through a Mosquitto MQTT broker.
- Alternatives:
  - ROS 2 with DDS across houses: it assumes one trusted network and leaks every topic;
  - Zenoh: newer, with fewer people who know it;
  - HTTP or A2A between robots: each house would need a reachable address;
  - WebRTC: peer to peer, but needs a signaling server anyway.
- Why: simple, standard for IoT, works behind home routers since both sides dial out, and the trust layer sits above it, so the transport can be swapped later.
- Revisit if: we need peer to peer with no broker at all, or need ROS 2 tooling across houses.
- Status: agreed

### D2. The broker is untrusted, trust lives in the message
- Date, by: 2026-10-05, Sunghwan with Claude
- Decision: the broker only relays. Every check happens in the receiving robot, on the message itself.
- Alternatives: trust the broker with TLS plus per-client logins and topic rules; mutual TLS between robots.
- Why: the broker is run by someone else in real life, and a message-level signature survives any transport.
- Revisit if: privacy becomes a requirement. Signing is not encryption, so the broker can still read everything (see D12).
- Status: agreed

### D3. Ed25519 signatures
- Date, by: 2026-10-05, team plan (Korean draft) and Claude
- Decision: each robot has an Ed25519 key pair and signs every message.
- Alternatives:
  - ECDSA P-256: wider hardware support, but needs a random number on every signature;
  - HMAC with shared secrets: every pair of houses would share a key, and anyone holding the secret can forge;
  - RSA: slow, large keys.
- Why: small keys (32 bytes), deterministic, fast enough for 50 messages a second, in the `cryptography` package.
- Revisit if: real robots need a hardware secure element that only supports P-256.
- Status: agreed

### D4. Canonical JSON as the signed bytes
- Date, by: 2026-10-05, Claude, matching the team plan
- Decision: sign every field except `sig`, with keys sorted, no spaces, UTF-8 and `ensure_ascii=False`.
- Alternatives:
  - JWS or JWT: standard, but heavier and easy to misuse;
  - COSE with CBOR: compact, less readable;
  - protobuf: needs a schema step;
  - signing the raw received bytes: breaks as soon as anything re-encodes the JSON.
- Why: readable on the link monitor, the same bytes in any language, and Korean text signs the same everywhere.
- Revisit if: messages get large, or another brand needs a formal standard (JWS is the closest swap).
- Status: agreed

### D5. Replay protection with time and nonce
- Date, by: 2026-10-05, Sunghwan with Claude
- Decision: `time` within ±60 s, plus an 8-byte random `nonce` that is remembered until its own `time` + 60 s.
- Alternatives:
  - a per-sender sequence counter: needs state that survives restarts;
  - challenge and response: an extra round trip for every request;
  - time only: allows replays inside the window.
- Why: no extra round trip and no stored counters. Keying the memory to the message's own time closes the gap when clocks differ.
- Revisit if: clocks on real robots drift past 60 s.
- Status: agreed

### D6. The `to` field is checked
- Date, by: 2026-10-05, team plan
- Decision: a message names its receiver, and any other robot rejects it.
- Alternatives: rely on per-robot inbox topics only.
- Why: anyone can republish a captured message to another robot's topic; the signed `to` stops that.
- Status: agreed

### D7. One key registry file per house, `active` or `revoked`
- Date, by: 2026-10-05, team plan
- Decision: `keys/registry.json` maps robot id to public key and status. Keys are filled in by hand for now.
- Alternatives:
  - a certificate authority (PKI);
  - decentralized ids (DIDs);
  - one shared central registry: it becomes the thing everyone has to trust.
- Why: simplest thing that shows registry and revocation working. Pairing by QR code is the proposed real-world path.
- Revisit if: more than a handful of houses, or vouching between houses (C5 later stage).
- Status: agreed

### D8. Authorization is a separate step from authentication
- Date, by: 2026-10-06, Seohee (PR #1), reviewed by Sunghwan
- Decision: `policy.py` runs only after `Verifier.check` passes. It gives ALLOW, PARTIAL or DENY from a per-action minimum trust, with a never-allowed list.
- Alternatives: one combined check; a rules engine (OPA style); a language model reading the request.
- Why: "is it really them" and "should I do it" fail for different reasons and need different logs.
- Revisit if: owners need rules beyond a trust threshold per action.
- Status: agreed

### D9. QoS 0 in the terminal robots, QoS 1 in the simulation
- Date, by: 2026-10-06, by accident (two authors)
- Decision: `robot.py` publishes at QoS 0 and the MuJoCo demos at QoS 1.
- Alternatives: QoS 1 everywhere, with redelivered duplicates rejected as `replay` (harmless but noisy); QoS 0 everywhere.
- Why: not chosen on purpose. The network goal run should pick one and log it.
- Status: needs review

### D10. Sign every position message in the follow stream
- Date, by: 2026-10-10, Claude, asked by Sunghwan
- Decision: each 50 Hz leader position is a full signed envelope. The follower checks it with its own `Verifier` (separate nonce memory from the request inbox).
- Alternatives:
  - sign once, then use a cheaper shared session key (HMAC) for the stream: faster, more code;
  - sign every Nth message: leaves gaps an attacker can use;
  - rely on TLS to the broker: the broker is untrusted (D2).
- Why: reuses the exact checks we already trust, and Ed25519 is fast enough at 50 Hz.
- Revisit if: streams get faster (camera, joints) and signing shows up in timing.
- Status: needs review

### D11. Private keys as raw files, chmod 600, never committed
- Date, by: 2026-10-05, Claude
- Decision: `network/keys/<robot>.key`, gitignored.
- Alternatives: OS keychain; hardware secure element or TPM on the robot.
- Why: enough for simulation.
- Revisit if: anything runs on a real robot.
- Status: agreed

### D12. No encryption of message contents yet
- Date, by: 2026-10-05, Sunghwan with Claude
- Decision: messages are signed, not encrypted. The broker and anyone on the topic can read them.
- Alternatives: encrypt to the receiver's key (X25519 with an AEAD cipher); TLS to the broker, which still leaves the broker reading everything.
- Why: the project's first question is who sent it, not who can read it.
- Revisit if: requests carry private data (camera images, home layout).
- Status: agreed

## Decisions made during the autonomous run

(The agent appends here, each marked `needs review`.)
