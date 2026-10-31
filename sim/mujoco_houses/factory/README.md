# Factory workflow demo

For the latest populated factory scene (rather than the older repeated-cell
workflow demo), see [Dense factory layout](DENSE_LAYOUT.md). It reuses the
existing world and carrying behaviors, adds stocked industrial cells and uses
one continuous camera. This is the current scene-composition preview.

From `sim/mujoco_houses`, using the existing MuJoCo environment:

```bash
PYTHONPATH=src ../coohoi_mujoco/.venv/bin/python -m unitree_demo.factory_executor \
  --scenario metal --cells 1 --duration 100 --require-complete --report outputs/factory_check.json
```

Mac video recording (run from the user's graphical terminal):

```bash
PYTHONPATH=src ../coohoi_mujoco/.venv/bin/mjpython -m unitree_demo.factory_executor \
  --scenario metal --cells 3 --duration 100 --require-complete \
  --video outputs/factory_cooperation.mp4 --report outputs/factory_cooperation.json
```

Use `--viewer` for an interactive view. Default camera hides the warehouse shell
and overhead structures for an unobstructed cutaway overview.

`factory_scene.py` reads approved assets from `sim/roqsim_factory_upstream`.
No upstream Python installation or ROS is needed. Dependencies: MuJoCo, numpy,
and ffmpeg for MP4 recording. Existing G1 demos additionally use torch.
Install `tqdm` in the same environment (`python -m pip install tqdm`). Recording
shows frame progress, elapsed wall time, ETA and encoded video seconds. After
rendering it shows `Finalizing MP4`, then `Video saved` when ffmpeg finishes.

`factory_planner.py` maps tasks to model names independently of physics; it can
output a JSON plan with `--scenario metal|electronics|warehouse --cells 1|2|3`.
These are demo task mappings, not payload/reach-certified engineering proposals.
Metal uses AMRs, UR5e, Panda and UR10e. Electronics substitutes Panda stations.
Warehouse uses four AMRs and a pallet arm per cell. Cells=3 produces 15 robots.

Metal's coordinator tracks one physical part through machining, mobile transport,
assembly, a friction-driven roller conveyor, packaging and shipping transport.
Other scenarios currently exercise independent station controllers; do not
describe them as validated end-to-end workflows.

Verified headless run: `outputs/factory_mass_verified.json`, 100 simulated
seconds, 15 robots, three staggered production lines, 15 interstage handoffs,
all three lines `SHIPPED`, `completion_pass: true`. Each line tracks a single
0.15 kg workpiece; this does not validate heavy sheet handling or production
throughput. CNC is a simplified unload/transfer station; machining and
physical assembly/vision inspection are not simulated. Shipping completion
means arrival with retained cargo, not unloading onto a shipping pallet.

All robot motion uses velocity/position actuators and state feedback. Workpieces
have free joints and are never teleported during execution. Grasps and cargo
retention use proximity/arrival-gated weld constraints, with payload/arm collision
exclusions; this is an assisted tooling demo, not friction-only grasp validation.
The roller conveyor uses actuated joints and contact friction.

Read `production_state`, `handoffs`, arm state/error, AMR arrival/distance and the
timestamped `events` in the report. `SHIPPED` is the end-to-end completion state;
elapsed duration alone is not a pass. A blocked station waits rather than
automatically advancing after a fixed time.

See [asset policy](ASSET_POLICY.md) for attribution to include with published video.
