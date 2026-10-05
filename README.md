# robot-trust-network

A trust layer for robots from different owners to work together safely.

Each household's robot sits on its own isolated network. A request from another household's robot passes through a gatekeeper with two steps: **verify** (is it really that robot?) and **decide** (how far should we follow it?). Only then does the robot act, and the owner sees a plain-language log line.

## Layout

| Folder | Team | What lives there |
|---|---|---|
| `network/` | Network team | Message signing and verification, key registry, replay protection, MQTT transport, trust and decision rules |
| `sim/` | Simulation team | MuJoCo (later Isaac Sim) robots driven by a VLA model |
| `docs/` | Everyone | Shared message format, architecture, slides |

Start with [`docs/message-format.md`](docs/message-format.md). Both teams build against it, so change it only by agreement.

## Roadmap
1. Two robots talk over a shared ROS 2 network
2. Two isolated containers talk only through an MQTT broker, with signed messages
3. Robots in separate containers talk inside the simulator
4. Two real robots in the lab, not on the same server
