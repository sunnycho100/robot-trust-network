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

### D13. QoS 1 for the container households (settles D9 for this harness)
- Date, by: 2026-10-10, Claude (autonomous run, M1)
- Decision: every publish in `network/scenarios/` uses QoS 1, so the broker acknowledges it.
- Alternatives: QoS 0 (fire and forget, losses are silent); QoS 2 (exactly once, an extra round trip per message).
- Why: the acknowledgement is how house 1 notices a dead link quickly (M5). A copy the broker redelivers is rejected by the nonce check as `replay`, so "at least once" never turns into "acted twice".
- Revisit if: the replay rejections from redelivery clutter the owner log. `robot.py` is still QoS 0.
- Status: needs review

### D14. One Docker network per house, broker attached to all of them
- Date, by: 2026-10-10, Claude (autonomous run, M1)
- Decision:
  - `house1`, `house2` and `house3` are separate Docker networks marked `internal` (no internet).
  - Each house container joins only its own network, and only the broker joins all three.
  - A house can therefore reach the broker but not another house, and S11 checks this both by IP and by name.
- Alternatives:
  - one shared network with firewall rules: isolation would depend on rules being right;
  - a gateway container per house in front of a robot container: closer to a real home router, but double the containers for no new test;
  - separate VMs: heavy;
  - ROS 2 `ROS_DOMAIN_ID` per house: separates topics, not the network.
- Why: real network separation with the fewest moving parts.
- Revisit if: a house needs its own robots talking locally (then add a per-house local network or bridge).
- Status: needs review

### D15. The runner lives on the host and drives house 1 with `docker compose exec`
- Date, by: 2026-10-10, Claude (autonomous run, M1)
- Decision:
  - `network/scenarios/run.py` (standard library only) builds the image, makes keys, starts the houses and runs each scenario.
  - House 1's side runs as `client.py` commands inside the house-1 container.
  - The receiving houses append to `events.jsonl` in their state folder, which the runner reads.
- Alternatives:
  - a runner inside house 1: it could not cut house 1's own link or play house 2's owner;
  - a separate test container on the broker networks: it would break the isolation the test checks;
  - pytest: an extra dependency on the host.
- Why: the host is the only place that can act as every owner and as the network itself, without punching holes in the isolation.
- Status: needs review

### D16. Fresh keys per run, each house gets only its own
- Date, by: 2026-10-10, Claude (autonomous run, M1)
- Decision:
  - `keygen.py` runs in a container with no network and writes `.run/<house>/`, which is gitignored.
  - Each folder holds only that house's private keys, a registry of the robots it has paired with, and its owner rules.
  - Each house mounts only its own folder.
- Alternatives: reuse `network/keys/` (shared by every robot, so no isolation of secrets); bake the keys into the image (leaks into image layers).
- Why: matches D7 (one registry per house) and keeps one house's secrets out of the others.
- Status: needs review

### D17. Three houses and two robots in house 1
- Date, by: 2026-10-10, Claude (autonomous run, M1)
- Decision:
  - House 1 has robot A (a trusted neighbor, starting trust 70) and robot D (newly paired, starting trust 0).
  - House 2 has robot B with default rules.
  - House 3 has robot C with stricter rules (`join_route` needs 90).
- Alternatives: one receiving house whose rules are swapped between runs, which would not show two owners disagreeing at the same time; an unregistered robot for S5, which the fixed receive checks reject outright (`unknown sender`), so "information only" needs a paired robot with zero trust.
- Why: S2 needs two owners with different rules, and S5 and S6 need a robot that starts with no trust.
- Status: needs review

### D18. Owner rules can only make a house stricter
- Date, by: 2026-10-10, Claude (autonomous run, M2)
- Decision: `decide(message, trust, rules)` takes one owner's `never` list and `minimum` thresholds. An owner can forbid more actions or raise a threshold, but cannot lower one below the default or unforbid `unlock_door`.
- Alternatives: rules fully replace the defaults (an owner could set `unlock_door` to trust 0); one global rule set (S2 cannot exist).
- Why: the defaults act as a safety floor every brand ships with, and owners tune upward.
- Revisit if: owners need to open up an action for one specific trusted robot (then add per-robot grants, not lower floors).
- Status: needs review

### D19. A rejected message gets no reply
- Date, by: 2026-10-10, Claude (autonomous run, M2)
- Decision: only messages that pass every receive check get a signed response. Everything else is logged in the house and answered with silence.
- Alternatives: reply "rejected, reason X" to the claimed sender.
- Why: a reply to a forged message would be a signed echo that an attacker can trigger at will, and it tells the attacker which check stopped them. The sender of a real request that times out treats it as DENY.
- Revisit if: honest robots with a broken clock need to learn why they are being ignored (an owner notification fits better than a network reply).
- Status: needs review

### D20. Responses carry `reply_to`, `result` and `reason` as signed fields
- Date, by: 2026-10-10, Claude (autonomous run, M2)
- Decision: a response is a normal signed envelope with `kind: response` and `reply_to` set to the request's nonce. The requester accepts it only if it verifies and `reply_to` matches a request it actually sent. Added to `message-format.md`.
- Alternatives: correlate by topic only (any signed reply could be matched to any open request); unsigned replies (anyone could answer for house 2).
- Why: the same checks in both directions, and a reply cannot be replayed onto a different request.
- Status: needs review

### D21. Trust update rules: +5 per clean interaction, -20 for asking for a forbidden action
- Date, by: 2026-10-10, Claude (autonomous run, M3)
- Decision:
  - Each house keeps `trust.json` (robot to score, 0 to 100), seeded from the owner's starting scores.
  - An ALLOWed request adds 5, at most once per `gain_interval_s`.
  - A verified request for an action on the shared never-allowed list (`unlock_door` and the rest) takes away 20.
  - A DENY for low trust changes nothing: asking is not wrong.
- Alternatives:
  - a success ratio (Beta reputation: good and bad counts): more principled, harder to explain on a slide;
  - time decay toward the starting score;
  - owner approval for every raise;
  - scores weighted by how risky the action was.
- Why: the simplest rule that makes S6 (earning access) and a penalty visible, and easy to change in one place (`GAIN`, `PENALTY` in `house.py`).
- Revisit if: the team wants failures (a task the sender aborted) or owner feedback to count, or decay over time. The 5 and 20 are guesses.
- Status: needs review

### D22. At most one trust gain per interval per robot
- Date, by: 2026-10-10, Claude (autonomous run, M3)
- Decision: a robot earns trust at most once per `gain_interval_s` (1 s in the test houses so the run takes seconds; the default when unset is 3600 s).
- Alternatives: no limit, where a robot could reach help-level by sending 12 `status` requests in a row; only count gains from actions above information level, which makes a new robot at 0 unable ever to start climbing.
- Why: closes the "farm trust with cheap requests" hole with one number.
- Revisit if: the interval feels wrong for real homes, or cheap requests should earn less than useful ones.
- Status: needs review

### D23. Trust only moves on messages that passed every check
- Date, by: 2026-10-10, Claude (autonomous run, M3)
- Decision: an unsigned, spoofed, tampered, replayed or misaddressed message never changes anyone's score. S7 checks this.
- Alternatives: lower the claimed sender's trust on every forgery.
- Why: a forgery cannot be blamed on the robot it names. Lowering the victim's trust would let any attacker lock a neighbor out by sending junk in its name.
- Status: needs review

### D24. Only the shared never-allowed list costs trust, not one owner's own forbidden list
- Date, by: 2026-10-10, Claude (autonomous run, M3, found by S3)
- Decision: asking house 3 for `join_route` (forbidden only by house 3's owner) is denied but not penalized. Asking anyone for `unlock_door` is denied and penalized.
- Alternatives: penalize both. The first run did this, and A lost 20 points at house 3 just for asking something it could not know was forbidden there.
- Why: penalize what every robot is expected to know, not one owner's private settings.
- Revisit if: owners publish their rules, so visitors could know them in advance.
- Status: needs review

### D25. A task stays alive only while signed steps keep arriving
- Date, by: 2026-10-10, Claude (autonomous run, M4)
- Decision:
  - An allowed `join_route`, `mirror_motion` or `carry_together` starts a task.
  - The requester then sends signed `kind: step` messages naming the task (10 a second in the test), and each step passes the full receive checks.
  - The receiver stops the task if no valid step arrives for 1 s (`STALE_S`), if a step says `done`, or if a new task replaces it.
  - The first step gets 3 s (`FIRST_STEP_S`, the same as the reply timeout), since it can only leave after the grant has travelled back. Two of six clean-clone runs failed when the first step had only 1 s.
- Alternatives:
  - grant once, then run until the requester says stop: a lost link leaves the robot moving;
  - a fixed duration in the request: no way to stop early;
  - a lease the requester renews every few seconds: fewer messages, slower to stop.
- Why: this is the same pattern as the signed pose stream in the duo demo (D10), and "no news means stop" is the safe default the architecture asks for in C2 stage 3.
- Revisit if: 10 Hz of signed steps is too chatty on real links (move to a lease), or a robot needs to finish its current motion safely before stopping instead of stopping at once.
- Status: needs review

### D26. Revocation is the owner editing the house registry, checked 20 times a second
- Date, by: 2026-10-10, Claude (autonomous run, M4)
- Decision:
  - `owner.py revoke <robot>` atomically rewrites that house's `registry.json`.
  - The house reloads the registry when the file changes, both on every message and in a 20 Hz watchdog.
  - The watchdog stops a running task whose sender is no longer `active`. S8 measured about 10 ms from revoke to stop.
- Alternatives:
  - a signed revocation message from the owner's own key over the network: needs an owner key and a way to reach the house;
  - a shared revocation list on the broker: the broker is untrusted (D2);
  - certificate expiry: revocation only takes effect when the certificate expires.
- Why: keeps D7 (each house owns its registry) and works with no network at all, which is exactly when an owner may want to cut a robot off.
- Revisit if: owners manage the house from a phone (then the signed owner message becomes the path), or more than one house must revoke the same robot together.
- Status: needs review

### D27. One running task per receiving robot
- Date, by: 2026-10-10, Claude (autonomous run, M4)
- Decision: a newly allowed task replaces the current one, and the old one is logged as stopped ("replaced by a new task").
- Alternatives: queue tasks; refuse new tasks while busy; run several at once.
- Why: a robot has one body. Replacing keeps the newest intent; refusing would let one robot hog another.
- Revisit if: a lower-trust robot should not be able to cancel a higher-trust robot's task. That is likely the right next rule.
- Status: needs review

### D28. Faults are injected from outside: `tc netem` for delay and loss, `docker network disconnect` for cuts
- Date, by: 2026-10-10, Claude (autonomous run, M5)
- Decision:
  - S9 adds `delay 300ms loss 10%` to the outgoing link of both house 1 and house 2, which covers both directions of the request path. Only these two get `NET_ADMIN`.
  - S10 pulls a house's only network link with `docker network disconnect`, then plugs it back in.
- Alternatives:
  - netem on the broker: the Mosquitto image has no `tc`;
  - Toxiproxy or Pumba: an extra tool to install;
  - killing the container: tests a crash, not a lost link.
- Why: real kernel-level delay and loss, and a real cut that the house cannot see coming, with no new dependencies.
- Revisit if: we need packets reordered or duplicated (netem can do both), or one-way faults.
- Status: needs review

### D29. A sender notices a dead link when the broker stops acknowledging for 1 s
- Date, by: 2026-10-10, Claude (autonomous run, M5)
- Decision:
  - While streaming steps, house 1 tracks its QoS 1 publishes. If any stays unacknowledged for 1 s (`LINK_LOST_S`), it stops the task and reports "link lost".
  - It then exits without a clean MQTT disconnect, because a clean disconnect on a dead socket hung for about 8 s (found by S10).
- Alternatives: MQTT keepalive alone (detects in 1.5 times the keepalive, several seconds); TCP errors (can take minutes); an application-level ping.
- Why: uses the acknowledgement QoS 1 already gives (D13), and makes both sides give up on the same 1 s scale: the receiver stops after 1 s without steps (D25), and the sender after 1 s without acks.
- Revisit if: real links have ack delays over 1 s on a good day, which would cause false "link lost" stops.
- Status: needs review

### D30. No answer means DENY
- Date, by: 2026-10-10, Claude (autonomous run, M5)
- Decision: a request waits 3 s for a signed reply. A timeout, or a broker that cannot be reached at all, is reported as `TIMEOUT` and treated as DENY. Nothing is retried automatically.
- Alternatives: retry with backoff; wait forever; treat a timeout as "maybe" and keep trying in the background.
- Why: the safe default never acts on a request nobody approved. A retry would have to be a new request with a new nonce anyway (D5).
- Revisit if: under 300 ms and 10% loss, 1 in 8 requests time out at 3 s (S9). A longer timeout or one retry would raise completion, at the cost of slower refusals.
- Status: needs review

### D31. Fault scenarios report on their own, not in the allow and block counts
- Date, by: 2026-10-10, Claude (autonomous run, M5)
- Decision: "unsafe requests blocked" and "safe requests allowed" count only decision scenarios. S9 and S10 print their own `fault:` lines.
- Alternatives: count a request lost to the network as "safe request not allowed". The first M5 run did this and showed 22 of 24, which blamed the decision layer for packet loss.
- Why: keeps the two questions apart: did the house decide right, and what happened when the network broke.
- Status: needs review

### D32. Houses run as the host user, `tc` runs as root
- Date, by: 2026-10-10, Claude (autonomous run, M6)
- Decision: containers run as the uid and gid of whoever ran `run.sh`, so `.run/` never fills with root-owned files on Linux or WSL. Only the `tc` commands exec as root, and only house 1 and house 2 have `NET_ADMIN` (D28).
- Alternatives: run everything as root (the next run fails to delete `.run/` on Linux); keep state in Docker volumes (the runner could no longer read the event logs directly).
- Why: Seohee runs scripts under WSL, where root-owned files would break the second run.
- Status: needs review
