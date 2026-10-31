# Unitree G1 / Go1 in MuJoCo

The current 20-robot factory scene, cooperative carrying, two-arm panel handling,
Spot walking and recording commands are documented in
[Factory setup](factory/SETUP.md) and [Factory scene](factory/DENSE_LAYOUT.md).

Run from `sim/mujoco_houses/`. Signing and the trust policy come from `network/`.

| demo | script |
| --- | --- |
| one G1, WASD | `./scripts/run_basic` |
| G1 leader, G1 or Go1 follower | `./scripts/run_duo` |
| two full-body G1s carrying a box | `./scripts/run_coop_carry` |
| two kitchens over MQTT | `./scripts/run_houses` |

## Environment

Demos use `python` on `PATH`, or `UNITREE_DEMO_PYTHON` if that is set. That interpreter needs `requirements.txt` (`mujoco`, `torch`, `pygame`, `onnxruntime`).

```bash
conda activate lerobot
# name not found: conda activate /path/to/envs/lerobot
# or: export UNITREE_DEMO_PYTHON=/path/to/envs/lerobot/bin/python
python -m pip install -r requirements.txt
brew install mosquitto
```

On macOS the viewer restarts under `mjpython`.

The house export uses a second conda env, `robocasa` (`ROBOCASA_ENV` to rename it), because RoboCasa needs MuJoCo 3.3.1. `./scripts/setup_robocasa_houses` creates it. The demos still run in the env above and only load the exported MJCF.

## Submodules

`third_party/` has four pinned submodules. Do not run `git submodule update --init`; that downloads about 1 GB. From `sim/mujoco_houses/`:

```bash
./scripts/fetch_assets                 # sparse unitree_rl_gym + mujoco_menagerie, ~35 MB
./scripts/setup_robocasa_houses        # shallow robosuite + robocasa, then the kitchen export
```

| Submodule | Used for | Script |
| --- | --- | --- |
| [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym) | G1 model, `g1.yaml`, `motion.pt` | `fetch_assets` |
| [mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie) | Go1 model | `fetch_assets` |
| [robosuite](https://github.com/ARISE-Initiative/robosuite), [robocasa](https://github.com/robocasa/robocasa) | kitchen export | `setup_robocasa_houses` |

`fetch_assets` also downloads `go1_policy.onnx` and checks its SHA-256.

## Windows

`scripts/` are zsh, so run them in WSL and follow the sections above. Install Mosquitto with `sudo apt install mosquitto` instead of `brew`. The viewer uses `python`; `mjpython` is macOS only.

Without WSL, asset scripts still need zsh. The demos themselves can start from PowerShell in `sim/mujoco_houses/`:

```powershell
$env:PYTHONPATH = "src"
python -m unitree_demo.basic
python -m unitree_demo.duo
python -m unitree_demo.house_demo --house house-2 --robot g1
python -m unitree_demo.house_demo --house house-1 --robot g1
```

Install Mosquitto from [mosquitto.org/download](https://mosquitto.org/download/), then `mosquitto -c config\mosquitto\dev.conf -v`. If `conda activate lerobot` cannot see the env, set `$env:UNITREE_DEMO_PYTHON` to that env's `python.exe`.

## Basic

```bash
./scripts/run_basic
```

Focus the input window. `W/S` move, `A/D` turn, `Q/E` strafe, `Esc` quits.

## Duo

```bash
./scripts/run_mqtt_broker
./scripts/run_duo
./scripts/run_duo --follower go1
./scripts/run_duo --demo-mode story
```

The follower moves only after a signed request is allowed. `--trust 10` denies it. A pose stream silent for 0.5 s stops the follower.

## Cooperative lift

### macOS setup and video recording

Run these commands from the repository root. Python 3.11 or newer is recommended.
The environment below lives inside this project; the older environment at
`sim/coohoi_mujoco/.venv` is not required.

```bash
cd sim/mujoco_houses
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 'mujoco>=3.3' numpy torch
brew install ffmpeg
./scripts/fetch_assets
```

`fetch_assets` downloads the Unitree G1 meshes and published walking policy
(`third_party/unitree_rl_gym/deploy/pre_train/g1/motion.pt`). No locally trained
checkpoint is required. It also downloads the Go1 assets used by the other demos.
MQTT, RoboCasa and Isaac are not required for this cooperative demo.

Record the eight-second side-A/side-B carry video (from `sim/mujoco_houses`):

```bash
PYTHONPATH=src .venv/bin/mjpython -m unitree_demo.coop_lift \
  --headless --duration 8 \
  --video outputs/g1_wrap_side_A_B.mp4
```

On macOS use `mjpython` for rendering, including headless video recording.
`ffmpeg` must be on `PATH`. The output directory is created automatically.
To open the interactive viewer, omit `--headless` and `--video`.
If reusing `sim/coohoi_mujoco/.venv`, replace `.venv/bin/mjpython` above with
`../coohoi_mujoco/.venv/bin/mjpython` after installing the same dependencies there.

This demo keeps Unitree's published 12-joint walking policy for the legs and
adds PD control for the waist and both five-joint arms. One G1 stands at each
end of a long box; each robot reaches around its end so one palm contacts
side A and the other contacts side B while they walk sideways together. `--assisted-grasp` enables the older compliant hand
constraints for comparison; the default uses contact and friction only.

```bash
PYTHONPATH=src python -m unitree_demo.coop_lift
PYTHONPATH=src python -m unitree_demo.coop_lift --headless --duration 8
PYTHONPATH=src python -m unitree_demo.coop_lift --headless --video outputs/g1_cooperative_lift.mp4
```

## Houses

```bash
./scripts/setup_robocasa_houses
./scripts/run_mqtt_broker
./scripts/run_houses
./scripts/run_houses --house1 go1
```

The export is written to `assets/houses/`.

Add `--event-log path/to/house.jsonl` to either house process to append structured local
events. Each record has a run ID and separate simulation, Unix and monotonic timestamps.
Policy decisions, execution results and authenticated peer reports are separate events.
The requester reports FAILED and exits nonzero when discovery, cooperation or the return
inspection fails. COMPLETE covers the cooperation tasks; it does not claim that the requester
independently observed the attack rejections in the other house.

The house executor supports `status` with ALLOW, `verify_package` with ALLOW, and
`verify_and_unlock` with PARTIAL. Other policy-approved actions receive a signed DENY because
the house simulator has no executor for them. This does not change the shared network policy.
In the scripted role, house-1 accepts only one return inspection from house-2; unrelated or
late requests are refused before acknowledgement. Unexecuted queued work receives a final
DENY when the story ends. Either house exits nonzero if its recorded request flow failed or
it stops with accepted work pending, including the decision delay and return walk.

## Tests

```bash
python -m pytest
```

If Windows denies access to pytest's system temporary directory, use a dedicated scratch
directory: `python -m pytest --basetemp .pytest_cache/pytest-run` (pytest replaces that directory).

For a real broker and two headless MuJoCo processes, with temporary keys and saved event logs:

```bash
python tools/check_house_flow.py --robot g1 --output .pytest_cache/check-g1
python tools/check_house_flow.py --robot go1 --output .pytest_cache/check-go1
python tools/check_house_flow.py --robot g1 --missing-peer --output .pytest_cache/check-no-peer
python tools/check_house_flow.py --robot g1 --responder-cutoff --output .pytest_cache/check-cutoff
```

Choose a new output directory each run. The tool starts its own loopback Mosquitto on a free
port and terminates only its own processes. Pass `--mosquitto /path/to/mosquitto` if needed.
The positive checks inject malformed and unsupported requests before the story, then require
both cooperation directions to finish without fall/timeout logs. The negative check requires
requester failure and no false COMPLETE. The cutoff check requires a responder interrupted
while awaiting its return reply to fail with exit code 1. These are smoke checks, not the full
C7 fault/physics matrix.
