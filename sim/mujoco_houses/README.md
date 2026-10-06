# Unitree G1 / Go1 in MuJoCo, linked by signed MQTT

Three demos, all on published walk policies (nothing is trained here):

| demo | what you see | script |
| --- | --- | --- |
| basic | one G1, you drive it with WASD | `./scripts/run_basic` |
| duo | a G1 leader and a G1 or Go1 follower; the follower follows only after a signed request is allowed | `./scripts/run_duo` |
| houses | two RoboCasa kitchens, one window each; the robots ask each other for help over MQTT, and forged or replayed commands are rejected | `./scripts/run_houses` |

Run every command from `sim/mujoco_houses/`. Signing, verification and the trust
policy are imported from the repo's top-level `network/` folder.

## Setup

```bash
# Python env with the demo dependencies (any env name)
python -m pip install -r requirements.txt
brew install mosquitto

# Robot models and weights, only the files the demos load (~35 MB) -> third_party/
./scripts/fetch_assets
```

`third_party/` holds four pinned git submodules. Do not run a plain
`git submodule update --init` (about 1 GB); the scripts check out only what is used:

| Submodule | Used for | Checked out by |
|---|---|---|
| [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym) | G1 12-DoF model, `g1.yaml`, `motion.pt` | `fetch_assets` (sparse, ~24 MB) |
| [mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie) | Go1 model | `fetch_assets` (sparse, ~12 MB) |
| [robosuite](https://github.com/ARISE-Initiative/robosuite), [robocasa](https://github.com/robocasa/robocasa) | kitchen export | `setup_robocasa_houses` (shallow) |

`fetch_assets` also downloads `go1_policy.onnx` from
[mujoco_playground](https://github.com/google-deepmind/mujoco_playground) and checks its SHA-256.

The scripts use the active `python`. Set `UNITREE_DEMO_PYTHON` to use another one.
On macOS the viewer restarts itself under `mjpython`.

## 1. Basic

```bash
./scripts/run_basic
```

Keep the small input window focused. Hold `W/S` to move, `A/D` to turn and
`Q/E` to step sideways. Releasing every key sends `[0, 0, 0]`. `Esc` closes
the demo. A connected gamepad is used automatically.

## 2. Duo

```bash
./scripts/run_mqtt_broker            # terminal 1

./scripts/run_duo                    # terminal 2: G1 follows G1, WASD drives the leader
./scripts/run_duo --follower go1     # Go1 follows G1
./scripts/run_duo --demo-mode story  # scripted: request, imitate, forged command, carry
./scripts/run_duo --demo-mode story --trust 45   # the carry is only PARTIAL
```

The leader (orange) publishes pose and velocity to the broker. The follower
(blue) starts on its own route. When the leader moves, it sends a signed
`mirror_motion` request. The follower verifies the signature, recipient, time
and nonce, decides with the trust policy, and only then walks into formation.
In story mode an unsigned `unlock_door` is injected by a separate client. The
follower rejects it and keeps following. If the pose stream goes stale for 0.5 s,
the follower stops.

`--trust 10` denies following. `--message-delay` adds a visible lag to the pose
stream, and `--trust-delay` sets the pause between the decision and the motion.
For brokers on another network, see `--mqtt-tls`, `--mqtt-ca` and the
per-robot `--mqtt-*-cert/key` options.

## 3. Houses

```bash
./scripts/setup_robocasa_houses      # once: RoboCasa in its own env, kitchen export

./scripts/run_mqtt_broker            # terminal 1
./scripts/run_houses                 # terminal 2: G1 in both houses
./scripts/run_houses --house1 go1    # Go1 in house-1, G1 in house-2
```

Each house is its own process and window, so the robots meet only over MQTT.
robot-a in house-1 asks; robot-b in house-2 verifies every message, decides with
the trust policy, then acts in its own kitchen:

1. signed `status`: house-2 answers that it is online
2. signed `verify_package`: ALLOW. robot-b walks to its front door and reports the parcel
3. signed `verify_and_unlock`: PARTIAL at every trust score. robot-b checks the door and leaves it locked
4. unsigned `unlock_door`: rejected, the door lock blinks red, the robot does not move
5. the step 2 message published again: rejected as a replay
6. house-2 asks house-1 for the same check; robot-a verifies it and walks to its own door

Default trust is 55. `--trust 10` refuses the inspection as well. Unlocking is
never allowed, even at trust 100.

RoboCasa needs MuJoCo < 3.10, so `setup_robocasa_houses` installs it from the submodules
into a separate conda env with only the textures and furniture (~1.9 GB), and
exports the two kitchens to `assets/houses/` once. The demo itself loads that
MJCF with the regular env.

## Layout

```text
src/unitree_demo/
  basic.py        demo 1
  duo.py          demo 2: MQTT pose stream, leader/follower, trust gate
  house_demo.py   demo 3: one process per house, the story
  house_link.py   one signed MQTT endpoint per house
  house_scene.py  kitchen + door + parcel, navigation
  g1_policy.py    G1 config and motion.pt controller
  go1_policy.py   Go1 model and go1_policy.onnx controller
  trust_session.py  bridge to ../../network (keys, verifier, policy)
  common.py       viewer runtime, keyboard input, shared CLI options
  input_bridge.py pygame input window
scripts/          run_basic, run_duo, run_houses, run_mqtt_broker, fetch_assets, setup_robocasa_houses
```

## Tests

```bash
python -m pytest
```

The tests need no display, broker or downloaded models.
