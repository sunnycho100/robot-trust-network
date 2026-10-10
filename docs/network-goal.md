# Network team goal (autonomous run)

This file is the brief for an agent working on its own. It says what "done" means, how each step is checked, and what the agent may and may not decide. Every architecture or communication choice the agent makes goes into [`decisions.md`](decisions.md) so the team can review and change it later.

## The goal

Two households, each in its own isolated container network, can work together only through an MQTT broker that neither of them trusts. Every message is signed and checked, every request is decided by the receiving household's rules and its trust in the sender, trust changes with what happens, and the whole thing stays safe when the network is slow, lossy or cut. One command runs every scenario and prints the results table.

## Done means

From a clean checkout, with Docker running:

```bash
./network/scenarios/run.sh
```

1. builds the containers, runs every scenario in the table below and prints one row per scenario (name, expected, got, pass);
2. prints the numbers for the slides: unsafe requests blocked, safe requests allowed, time added by verify and decide (median and 95th percentile), and what happened under each network fault;
3. exits 0 only if every row passes.

And these still pass: `python network/test_protocol.py`, `python network/test_policy.py`, and `python -m pytest` in `sim/mujoco_houses`.

## Scenarios (the shared definition of done)

| # | Scenario | Expected |
|---|---|---|
| S1 | Known neighbor asks for something within its trust | ALLOW, signed reply arrives, reply matches the request nonce |
| S2 | Same request, receiving house has stricter rules | Different answer than S1 (PARTIAL or DENY) |
| S3 | Forbidden action (`unlock_door`) at maximum trust | DENY |
| S4 | Unsigned, spoofed, tampered, replayed, wrong `to` | All REJECTED with the right reason |
| S5 | Unknown robot asks for information | Only information-level actions allowed |
| S6 | A robot does K clean interactions | Trust rises from information-only to help-level access |
| S7 | Someone forges messages claiming to be robot A | Rejected, and robot A's trust does not drop |
| S8 | Robot A's key is revoked mid-task | Robot B stops acting on A within the stated time |
| S9 | 300 ms delay and 10% loss on the broker link | Requests still complete or time out cleanly, no hang |
| S10 | Broker link cut mid-request | Both robots end in a safe, known state (stop, then deny) |
| S11 | House 1 container tries to reach house 2 directly | Fails; only the broker path works |

## Milestones

Each milestone ends with its check passing and one commit. Order matters; do not start the next before the check passes.

| M | Build | Check |
|---|---|---|
| M1 | Docker Compose: broker, house-1, house-2 on separate networks | S11 passes; a signed message still goes house-1 to house-2 through the broker |
| M2 | Request and signed reply between containers, using `protocol.py` and `policy.py` | S1 to S5 pass |
| M3 | Trust store per house that changes with outcomes | S6 and S7 pass |
| M4 | Revocation that takes effect while a task is running | S8 passes |
| M5 | Fault injection with `tc netem` and link cuts, plus timeouts and safe defaults | S9 and S10 pass |
| M6 | `run.sh` that runs everything and prints the table and numbers | Done criteria above |

## Is it achievable

Yes, everything can be tested on this laptop. What makes it testable:

- **Isolation (M1):** Docker networks are real network separation, and S11 checks it directly.
- **Faults (M5):** `tc netem` works inside Docker Desktop's Linux VM when the container has `NET_ADMIN`.
- **Already in the repo:** the signing, verifying and deciding are done and tested (`protocol.py`, `policy.py`), so M2 is mostly wiring.

Risks:

- **Docker Desktop has to be running.** It is installed, but the daemon is off. The agent cannot start it, so the user starts it before the run.
- **Trust rules (M3) are a design choice, not a fact.** The agent picks simple rules and records them as "needs review".
- **ROS 2 is not part of this run.** Each house talks MQTT directly. Bridging a house's ROS 2 graph to MQTT is a later step, recorded in the log.

## Rules for the agent

- **Fixed, do not change:**
  - Ed25519 signing;
  - the canonical JSON rule;
  - the receive checks in [`message-format.md`](message-format.md);
  - the broker being untrusted.

  To add a message field, append it to the format doc and log it.
- **Free to choose, but log it:** container layout, topic names, trust update rules, timeouts, the revocation method, the scenario runner's shape.
- **Every major choice gets a `decisions.md` entry** with the alternatives that existed and why this one won, marked `needs review`. "Major" means a reviewer would want a say: anything about structure, protocol, security or behavior under failure. Not variable names.
- **Stay small:** Python standard library plus `cryptography` and `paho-mqtt`, and the `eclipse-mosquitto` and `python:3.12-slim` images. Ask before adding anything else.
- **Git:** work on branch `network-goal`, one commit per milestone, short subject, details in the body, no Co-Authored-By line, no push to `main`.
- **Writing:** no em dashes and no slash constructions in docs, comments or commit messages.
- **If blocked:** for example by Docker not running, or a check that cannot pass without breaking a fixed rule. Stop, write what was tried and what is needed in the log, and do not work around a fixed rule.
- **Do not touch** `sim/` beyond keeping its tests passing. The simulation team owns it.
