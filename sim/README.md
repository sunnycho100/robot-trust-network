# VLA sim

A MuJoCo sim where a vision-language model controls a robot through skill calls.
You type a command, the model sees a camera image plus the robot state, calls skills
(`pick`, `place_on`, `walk_to`, ...), sees the new image, and repeats until it calls `done`.

## Setup
```
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git menagerie
git -C menagerie sparse-checkout set franka_emika_panda unitree_go2
cp .env.example .env      # then fill in the key and model id
.venv/bin/mjpython vla_sim.py --robot panda
.venv/bin/mjpython vla_sim.py --robot go2
```
On macOS the viewer needs `mjpython`, plain `python` will not open the window.

## Robots
- `panda`: Franka arm with red, green, blue blocks. Real physics, the grasp can actually fail.
- `go2`: Unitree dog with three coloured markers. Kinematic only: the body glides and the legs
  play a trot animation. Real walking needs a trained locomotion policy.

## Test (no API key needed)
```
.venv/bin/python test_sim.py
```

## mujoco_houses
Unitree G1/Go1 walking on published policies, talking through MQTT with signed
`network/` requests: a basic WASD G1, a leader/follower duo (G1 or Go1 follower),
and two RoboCasa houses in separate windows. See [`mujoco_houses/README.md`](mujoco_houses/README.md).

## G1 manipulation

The separate [`manipulation/`](manipulation/README.md) runtime adds an articulated
G1 with two-finger grippers, GR00T balance, Mink arm IK, and Ruckig smoothing.
It includes camera-guided generic manipulation, stance/posture planning, and a
single native window for simulation, commands and traces. Physical task success
remains unreliable, and it is not connected to the two-household MQTT demo.
See the [October 29 implementation review](../docs/manipulation-status-2026-10-29.md)
for measured results and the integration plan.
