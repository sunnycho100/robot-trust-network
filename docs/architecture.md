# Architecture: build components

Each component is one thing that can be built and tested on its own. C1 and C2 are the ground the rest runs on. C3 to C5 are the actual contribution. Every component has stages so there is something working early.

---

## C1. Simulation environment
The world the robots live in. Starts with no physics at all.

- **Stage 1: terminal only.** Two Python programs, one per household, printing what they do. No simulator. This is enough to test the whole message and trust flow.
- **Stage 2: simple physics.** Two robots in Gazebo or MuJoCo. A request now causes an actual movement (drive to the porch). MuJoCo is lighter and faster to iterate on; Gazebo is closer to ROS 2 out of the box.
- **Stage 3: Isaac Sim.** A realistic house scene with two humanoids, for the demo video. Needs an NVIDIA GPU, so keep it last and optional.

Done when: a request from house 2 makes house 1's robot move in the simulator.

## C2. Transport layer (how the message gets there)
How a message travels from one household to another.

- **Stage 1: same ROS 2 network.** Both robots in one ROS 2 graph, messages over a topic. Easiest, and wrong on purpose (they can see everything about each other).
- **Stage 2: separated households.** Each household on its own `ROS_DOMAIN_ID`, so neither can see the other's topics. The only path between them is an MQTT server that both connect to. This is the realistic setup.
- **Stage 3: unreliable and mixed networks.** The two homes are on different internet providers, so messages are delayed, arrive out of order or get dropped, and one household can go offline. Test that a robot never gets stuck waiting and falls back to a safe default.

Done when: cutting the connection mid-request leaves both robots in a safe, known state.

## C3. Identity and message format
Proving who is asking, and saying it in a form any brand can read.

- A fixed request format: who is asking, which household, what action, what for, and a time stamp.
- Each robot signs its messages with its own key, and a registry maps keys to robots, so a robot cannot pretend to be the neighbor's.
- Replay protection, so an old captured "open the door" message cannot be resent later.

Done when: a forged or replayed message is rejected, and a valid one is accepted.

## C4. Gatekeeper (the decision layer)
One program per household. This is the core of the project.

- Takes a verified request and checks it against the owner's rules and the trust score.
- Outputs one of three things: do it, do part of it, or refuse, with a reason.
- Forbidden actions stay forbidden no matter how high the trust (door, camera, anything that risks the owner).
- Passes allowed actions to the robot's normal ROS 2 commands.

Done when: the same request gets different answers from two households with different rules.

## C5. Trust network
How much each household trusts each outside robot, and how that changes.

- A starting score set by the owner, raised by interactions that finish cleanly, lowered by failures or refused requests.
- Thresholds map a score to what that robot may ask for.
- Later: trust passed along from a neighbor ("house 3 vouches for this robot"), which is where it becomes a network instead of a list.

Done when: a robot earns its way from information-only up to help-level access through repeated good interactions.

## C6. Owner view
What the human sees.

- Every request and decision written as one plain sentence.
- A simple web page showing the feed and the current trust list, with a way to change the rules.

Done when: someone who knows nothing about ROS 2 can read the feed and understand what happened.

## C7. Scenarios and evaluation
How the project is judged.

- Fixed test scenarios: unknown robot asking for information, neighbor asking for help, a forbidden request, a forged sender, a network drop mid-task.
- Numbers to report: unsafe requests blocked, safe requests allowed, decision time added by the gatekeeper, behavior under a dropped connection.

Done when: the whole set runs automatically and produces the table for the paper or the slides.

---

**Build order:** C1 stage 1 + C2 stage 1 + a rough C4 gives a working demo in the terminal first. Then C3 hardens it, C2 stage 2 separates the households, C5 makes trust dynamic, C1 stage 2 adds real movement, and C6 and C7 make it presentable.

---

# Splitting the work in parallel

## The one thing that must come first
Before anyone splits off, everyone agrees on two things, written down on day one:

1. **The request format.** A JSON object: who is asking, which household, what action, why, time stamp, signature. Frozen early, changed only by agreement.
2. **Two function boundaries.**
   - `send(request) -> response` for the transport side
   - `decide(request) -> allow | partial | deny` for the gatekeeper side
   - `execute(action) -> result` for the robot side

Once those exist, every track can build against fake versions of the others. The transport team tests with a gatekeeper that always says yes. The gatekeeper team tests with messages typed by hand. Nobody waits.

## Option A: four people, four tracks

| Track | Owns | Works against | First deliverable |
|---|---|---|---|
| **1. Transport** | C2 | a fake gatekeeper that always allows | request travels between two separated ROS 2 households through MQTT, survives a dropped connection |
| **2. Gatekeeper and trust** | C4, C5 | hand-written request files, no network | given a request and a trust score, returns allow, partial or deny with a reason |
| **3. Simulation and execution** | C1 | a fake request source | robot performs an approved action in the simulator and reports back success or failure |
| **4. Identity, owner view, evaluation** | C3, C6, C7 | sample messages | signed and verified messages, plain-language feed page, scenario runner |

Track 2 is the research contribution, so put the strongest person there. Track 4 is the widest but each piece is small.

## Option B: two teams

- **Team Plumbing (C1, C2, C3):** the channel between households, the separated networks, the signing and the robot actually moving. Their output is "a verified request arrives, and an approved action happens."
- **Team Brains (C4, C5, C6, C7):** the decision rules, the trust scores, the owner's view and the test scenarios. Their output is "given a request, here is the decision and why."

They meet at the `decide()` boundary. Plumbing calls it, Brains implements it.

## How to keep them from blocking each other
- Each track keeps its own fake version of what it doesn't own. No shared database until integration.
- Fix a weekly integration point where the fakes are swapped for the real thing and the scenarios in C7 are run end to end.
- The scenario list from C7 is written first, by everyone, because it is the shared definition of done.
