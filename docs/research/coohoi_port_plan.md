# CooHOI port to Isaac Lab 3.0: plan (2026-10-31)

Goal: two humanoids carrying one box inside Isaac Sim 6.2, so CooHOI can share a stage with our reconstructions and the stock robots. Today it only runs in Isaac Gym Preview 4.

## What the source says (read tonight, not run)

- **Same body in both stacks.** CooHOI uses ASE's `amp_humanoid.xml` (28 actuated DoF, 15 bodies). Isaac Lab 3.0 ships the same humanoid as `HUMANOID_28_CFG` (`Robots/Classic/Humanoid28/humanoid_28.usd`) for its skrl AMP tasks (`IsaacContrib-Humanoid-AMP-Walk-Direct` and others). Both simulate at 60 Hz with decimation 2, so the policy runs at 30 Hz in both.
- **One shared policy for both humanoids.** CooHOI's two-agent stage (`share_humanoid_amp_carryobject.py`, `share_humanoid_task_coohoi.yaml`) runs one network for every humanoid. Each one sees only itself and the box. Coordination comes through the box's motion, not through messages. That matches what Sunghwan asked for: one policy driving several robots, not a joint multi-agent model.
- **Policy input:** 223 body-state values (`1 + 15 * (3 + 6 + 3 + 3) - 3`, local root frame: root height, then each body's position, rotation as tangent and normal, velocity and angular velocity) plus 76 task values (box pose, its bounding points, standing and held points, target pose, box density). That is 299 in total (counted from the code, unverified against the checkpoint). Output: 28 PD targets, mapped as `offset + scale * a` from the joint limits (`humanoid.py:316`).
- **A detail that matters for physics:** each humanoid of a pair sits in its own Isaac Gym env, and the box state is copied from one env to its partner every step (`share_humanoid_amp_carryobject.py:433`). So in training the two humanoids never pushed on one physical box in one scene. In Isaac Lab we would put both humanoids and one box in a single env (`DirectMARLEnv`, two agents), which is more correct physics but not what the policy saw.
- **Isaac Lab 3.0 has the multi-agent pieces:** `DirectMARLEnv`, skrl IPPO and MAPPO, and a working example (`Isaac-Shadow-Handover-Direct`, two hands passing a ball). The AMP humanoid tasks list no Newton preset, so the humanoid side needs the Isaac Sim PhysX backend (`--extra isaacsim`).

## Plan, cheapest first

1. **Inference port, no training (about 1 day of code, no GPU training).** A `DirectMARLEnv` with two Humanoid28s and one box, observations rebuilt exactly as CooHOI computes them, and CooHOI's rl_games actor (MLP 1024 then 512, ReLU, running mean and std on the input) loaded into plain torch. Two traps to check first:
   - Joint and body order. Isaac Gym orders the MJCF depth first, while the USD articulation in Isaac Lab may order it breadth first. Build the permutation by joint name, and assert it against a recorded Isaac Gym rollout (same initial state, compare the first 10 observations).
   - PD gains and joint limits. Read them from the MJCF and check the USD matches; the action mapping depends on the limits.

   Test in order: `SingleAgent.pth` on one humanoid (success rate over 100 resets), then `TwoAgent.pth` on a pair. Runs on Colab with the pip Isaac Sim (proved tonight) or on Brev.
2. **If transfer degrades (likely somewhat: PhysX 4 to PhysX 5, and the shared box): fine-tune.** Same env, skrl MAPPO with the shared actor initialized from `TwoAgent.pth` and an AMP style reward from CooHOI's motion clips (`coohoi/data/motions`). Budget 2 to 4 L40S hours (estimate).
3. **Full retrain** only if both fail: about 15 h on a 3090 Ti per the paper, roughly 10 to 15 L40S hours.

## Caveats

- CooHOI has no license file. Internal testing only until the authors answer.
- Everything above comes from reading the code; nothing has been ported yet. The MAPPO smoke on Colab (below) checks only that Isaac Lab 3.0's multi-agent training runs on our setup.

## Smoke test tonight

See the morning report: skrl MAPPO on `Isaac-Shadow-Handover-Direct`, 4096 envs, Newton physics, on a Colab L4 (`overnight/colab/lab_mappo.sh`).
