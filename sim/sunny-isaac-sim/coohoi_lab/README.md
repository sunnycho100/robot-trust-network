<!-- CooHOI's trained box carry in Isaac Lab 3.0 on Isaac Sim 6.2, no retraining. Usage: follow "Run order on Brev" top to bottom. -->
# CooHOI in Isaac Lab 3.0

CooHOI's `TwoAgent.pth` and `SingleAgent.pth` running in Isaac Lab 3.0 (Isaac Sim 6.2 PhysX), with no retraining. On a Brev L40S: **single carry 81% success, pair 88%** (32 episodes each, `--effort unlimited`, dt 1/120 with decimation 4), against 0% before the two USD fixes below. Not run yet: 100 episodes per mode, Isaac Gym's own rate on the same seeds as the baseline, and the 1/60 timing. CooHOI has no license file, so keep this internal.

| File | What |
|---|---|
| `policy.py` | The rl_games actor in plain torch: obs normalisation, MLP 1024-512 ReLU, mu, clamp to ±1 |
| `obs.py` | CooHOI's 299-dim observation (223 humanoid + 76 task), pair and single agent |
| `joint_map.py` | MJCF order vs Isaac Lab order by name, PD action map, MJCF gains. `python joint_map.py` prints the table |
| `humanoid28_pelvis_root.usda` | Override layer on Isaac Lab's Humanoid28: pelvis as the articulation root, Gym's 15 body masses, working knee and elbow drives |
| `coohoi_env.py` | `DirectMARLEnv`: two `HUMANOID_28_CFG` and one box (`pair`), or one and a cube (`single`) |
| `play.py` | Runs N episodes, logs success rate, box height, distance; viewport video; optional scene USD |
| `record_gym_reference.py` | Isaac Gym side: first N obs, actions, states, names and gains to an `.npz` |
| `compare_reference.py` | Isaac Lab side: per-term differences against that `.npz` |
| `test_*.py` | CPU checks: policy (both checkpoints), obs (against CooHOI's own code), joint map, env logic (fake Isaac Lab) |

## Three things the source says that the plan did not

- **Timing.** CooHOI runs sim dt 1/60 with `substeps: 2`, so physics is 120 Hz and the policy 30 Hz. Isaac Lab has no substeps; its migration guide maps that to dt 1/120 and twice the decimation. Default here: `--physics_dt 1/120 --decimation 4`. The plan's 1/60 and 2 is `--physics_dt 0.0166667 --decimation 2`.
- **Quaternions.** Isaac Lab 3.0 switched to xyzw (`docs/source/isaaclab/migrating.rst`), the same as Isaac Gym, so nothing is reordered. `play.py` asserts after reset that the left thigh sits on +y of the pelvis, which fails if the order is mixed up.
- **One physical box.** In CooHOI both humanoids of a pair share a collision group with the base env's box, and the partner env's box is a ghost in its own group. So in training both humanoids most likely pushed on one box, as in this env. unverified: that Isaac Gym collision groups span the two envs.

## Why the policy failed at first (fixed in `humanoid28_pelvis_root.usda`)

- **Knees and elbows had no drive.** They are the only one-axis joints in the USD, and their drive is authored as `PhysicsDriveAPI:X`, which a revolute joint ignores (it reads `angular`). Isaac Lab still reported the right gains, so every gain check passed while those four joints got no torque.
- **No masses in the USD.** PhysX used density 1000: 35.7 kg in total against Gym's 48.9 kg. The layer now sets Gym's runtime masses.
- **Root link.** The USD puts the articulation root on the top Xform, so PhysX rooted the tree at the torso. The layer moves it to the pelvis, CooHOI's root.

With these, the free rollout from Gym's recorded state drifts 1e-2 m in body positions after 9 policy steps (it was 0.31 m).

## Run order on Brev

0. Upload this folder to `~/robot/isaac-sim/coohoi_lab` (Jupyter file browser). CooHOI is expected at `~/CooHOI` (`setup_coohoi.sh`); set `COOHOI` otherwise.

1. **Isaac Lab 3.0 with Isaac Sim 6.2** (pip Isaac Sim through uv, a large download; it can run while step 2 records). The first Kit start shows the NVIDIA Omniverse EULA: read it and accept it yourself, then set `OMNI_KIT_ACCEPT_EULA=YES`.
   ```bash
   git clone --depth 1 --branch v3.0.0 https://github.com/isaac-sim/IsaacLab ~/IsaacLab
   cd ~/IsaacLab && UV_SYSTEM_PYTHON=false GIT_LFS_SKIP_SMUDGE=1 uv sync --extra isaacsim --extra skrl
   # Brev's 565 driver cannot run the cu130 torch that uv installs: swap in cu128, then never use `uv run` (it re-syncs)
   .venv/bin/python -m pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
   # local copy of the Humanoid28 asset for the override layer
   A=https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.2/Isaac/IsaacLab/Robots/Classic/Humanoid28
   mkdir -p ~/h28/Props && curl -sSo ~/h28/humanoid_28.usd $A/humanoid_28.usd && curl -sSo ~/h28/Props/instanceable_meshes.usd $A/Props/instanceable_meshes.usd
   ```
   Needs uv (`curl -LsSf https://astral.sh/uv/install.sh | sh`). Isaac Lab 3.0 runs headless by default and has no `--headless` flag.

2. **CPU tests and the Isaac Gym reference** (CooHOI conda env):
   ```bash
   eval "$(~/miniforge3/bin/conda shell.bash hook)" && conda activate coohoi   # or your conda
   cd ~/robot/isaac-sim/coohoi_lab
   python test_policy.py && python test_joint_map.py && python test_env_logic.py
   ISAACGYM_TORCH_UTILS=~/isaacgym/python/isaacgym/torch_utils.py python test_obs.py   # real Isaac Gym math; poselib's FBX warning is harmless
   cd ~/CooHOI && unset DISPLAY && L=~/robot/isaac-sim/coohoi_lab
   A="--test --num_envs 16 --headless --seed 0 --motion_file coohoi/data/motions/coohoi_data/coohoi_data.yaml"
   python $L/record_gym_reference.py --out ~/ref_pair.npz $A --task ShareHumanoidCarryObject \
     --cfg_env coohoi/data/cfg/share_humanoid_carrybox.yaml --cfg_train coohoi/data/cfg/train/share_humanoid_task_coohoi.yaml \
     --checkpoint coohoi/data/models/TwoAgent.pth
   python $L/record_gym_reference.py --out ~/ref_single.npz $A --task HumanoidAMPCarryObject \
     --cfg_env coohoi/data/cfg/humanoid_carrybox.yaml --cfg_train coohoi/data/cfg/train/amp_humanoid_task.yaml \
     --checkpoint coohoi/data/models/SingleAgent.pth
   cd $L && python compare_reference.py ~/ref_single.npz --policy_only && python compare_reference.py ~/ref_pair.npz --policy_only
   ```
   Check the `[ref]` lines: env 0's humanoid should sit 3 to 7 m on -x of the box it observes, env 1's 3 to 7 m on +x (the layout `coohoi_env.py` assumes). `--policy_only` should show `policy.py ... max |diff|` near 1e-6 and tells you Gym's dof effort (pick `--effort` from it).

3. **Compare** (Isaac Lab). If Kit fails to start, stop the Isaac Sim container first (`bash ~/robot/isaac-sim/isaac_up.sh down`).
   ```bash
   cd ~/IsaacLab && L=~/robot/isaac-sim/coohoi_lab && R=~/IsaacLab/.venv/bin/python
   export OMNI_KIT_ACCEPT_EULA=YES COOHOI_HUMANOID_USD=$L/humanoid28_pelvis_root.usda
   $R $L/compare_reference.py ~/ref_single.npz --effort unlimited && $R $L/compare_reference.py ~/ref_single.npz --effort unlimited --rollout
   $R $L/compare_reference.py ~/ref_pair.npz --effort unlimited && $R $L/compare_reference.py ~/ref_pair.npz --effort unlimited --rollout
   ```

4. **Play single**: `$R $L/play.py --mode single --episodes 100 --effort unlimited`

5. **Play the pair**: `$R $L/play.py --mode pair --episodes 100 --effort unlimited`, then a video:
   ```bash
   $R $L/play.py --mode pair --episodes 1 --num_envs 1 --effort unlimited --video ~/workspace/coohoi_lab_pair
   ffmpeg -framerate 30 -i ~/workspace/coohoi_lab_pair/frame_%05d.png -c:v libx264 -pix_fmt yuv420p ~/workspace/coohoi_lab_pair.mp4
   ```
   In our scan later: add `--scene ~/robot/recon/da_world.usda --origin X Y Z` (the floor point to stand on).
   Steel look: `--plates 8 --ground_color 0.26 0.27 0.28` draws the box as eight strapped steel plates on a concrete-gray floor. It changes the looks only, since the policy only knows CooHOI's 2 x 0.5 x 0.5 m box. A real thin plate failed every time: 0% at 8, 15 and 25 cm thick, and also when resting on trestles at the box's height.

## Reading the comparison

- `fk_body_pos` above 1e-3 at step 0: the USD's kinematics differ from the MJCF (joint order, axes or root frame). Stop there; no knob fixes it.
- Position terms fine but `root_rot_world` or `body_rot` off: a quaternion order or joint-convention problem.
- Teacher-forced velocity terms off while positions match: PhysX may not fill body velocities from a state write. Trust the rollout's step 1 for velocities.
- Rollout step 1 already far off: gains or limits. Try `--effort unlimited` or `--effort usd`, then `--physics_dt 0.0166667 --decimation 2`. The `[cmp] Lab joint ...` lines show Lab's gains next to Gym's.
- Box mass differs: CooHOI's `set_actor_scale` may not scale mass; only matters away from `box_scale 1`.

Known simplifications: falls are detected by body height only (CooHOI also needs a contact force), one box size per run instead of per env, no target marker in the video.
