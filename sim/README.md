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
