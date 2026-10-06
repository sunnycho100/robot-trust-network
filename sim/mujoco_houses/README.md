# Unitree G1 / Go1 in MuJoCo

Run from `sim/mujoco_houses/`. Signing and the trust policy come from `network/`.

| demo | script |
| --- | --- |
| one G1, WASD | `./scripts/run_basic` |
| G1 leader, G1 or Go1 follower | `./scripts/run_duo` |
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

## Houses

```bash
./scripts/setup_robocasa_houses
./scripts/run_mqtt_broker
./scripts/run_houses
./scripts/run_houses --house1 go1
```

The export is written to `assets/houses/`.

## Tests

```bash
python -m pytest
```
