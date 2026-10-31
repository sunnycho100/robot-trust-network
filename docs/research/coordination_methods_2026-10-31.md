# Multi-robot coordination and VLM-over-RL options for Isaac Sim 6.2

Checked 2026-10-31 against GitHub READMEs, Hugging Face model pages, arXiv and NVIDIA docs. "UNVERIFIED" means I could not confirm it on the source page.

## Key facts that drive everything below

- **Isaac Sim 6.2 pairs with Isaac Lab 3.0.** Most research repos pin Isaac Lab 2.1 to 2.3 (Isaac Sim 4.5 to 5.1), so they need a port, or you run their exported ONNX or TorchScript policy inside your own 6.2 loop.
- **"One shared policy for N robots" is the default in Isaac.** A locomotion policy is one network run on a batch of observations, one row per robot. Each robot gets its own command (velocity, goal). The Isaac Sim 6.2 H1 example already takes `--num-robots`, but (checked in the v6.2.0 source, `h1_standalone.py` and `runtime.py`) it builds one `RobotPolicyRunner` per robot: the same checkpoint and weights, one inference per robot, not one batched forward pass. Same behaviour, so the claim "one policy drives every H1" holds; true batching (one GPU call for N robots) needs Isaac Lab's `num_envs` or a custom runner over a multi-prim `Articulation`. What does not exist off the shelf is one policy across *different* robot types.
- **No released work combines all three:** an LLM or VLM over several robots, RL low-level policies, and Isaac Lab. CoLF and "Cognition to Control" come closest and have no code. We would be building this layer ourselves, which is fine since it is mostly glue.
- **No one has ported CooHOI to Isaac Lab.** All 12 forks were checked. CooHOI has no license file, so reuse beyond internal testing is legally unclear.

## A. Low-level skill layer (locomotion and whole-body control)

| Candidate | What it does | Weights | Sim | GPU (inference, training) | License | 6.2 on one L40S tomorrow |
|---|---|---|---|---|---|---|
| **Isaac Sim policy examples** ([docs](https://docs.isaacsim.omniverse.nvidia.com/latest/robot_simulation/ext_isaacsim_robot_policy_example.html)) | Flat-terrain walking for H1, Spot, Go2, plus Franka drawer. Velocity command (vx, vy, yaw rate) | Yes. `physx_policy.pt` and `physx_env.yaml` under `.../Assets/Isaac/6.2/Isaac/Samples/Policies/` (h1, Spot_Policies, go2), PhysX and Newton variants | Isaac Sim 6.2 native | Trivial, ships trained | NVIDIA asset terms (UNVERIFIED) | **Easy.** Native, `h1_standalone.py --num-robots N` |
| **Isaac Lab 3.0 velocity tasks** ([docs](https://isaac-sim.github.io/IsaacLab/main/source/isaaclab/environments.html)) | `Isaac-Velocity-Flat-G1`, `-Rough-G1`, `-Rough-H1`, `-Rough-UnitreeGo2`, `-Rough-AnymalD` | Published checkpoints via `--checkpoint pretrained`. Which tasks have one is UNVERIFIED, run `train_and_publish_checkpoints.py --list` | Isaac Lab 3.0 | Training a flat velocity policy is roughly 1 to 2 h on one GPU (UNVERIFIED) | BSD-3 | **Easy to medium.** Native; retrain if no checkpoint |
| **GEAR-SONIC** ([paper](https://arxiv.org/abs/2511.07820), [code](https://github.com/NVlabs/GR00T-WholeBodyControl)) | G1 motion-tracking foundation model with a kinematic planner that turns velocity or keyboard input into motion; also Decoupled WBC for GR00T N1.5 and N1.6 | Yes. `nvidia/GEAR-SONIC` (`model_encoder.onnx`, `model_decoder.onnx`, `planner_sonic.onnx`, `sonic_v1_1/`, `low_latency/`) | Trained in Isaac Lab 2.3, deployed via MuJoCo and C++ TensorRT | Inference fits easily; training was about 21k GPU hours, fine-tune wants 64+ GPUs | Code Apache-2.0, weights NVIDIA Open Model License (commercial OK) | **Medium.** ONNX runs anywhere, but you rebuild the observation pipeline in 6.2 |
| **ProtoMotions v3, MaskedMimic, Metatron** ([code](https://github.com/NVlabs/ProtoMotions)) | G1 and SMPL motion trackers; Metatron walks, interacts with scene boxes and recovers from falls, driven by masked pose, style and velocity goals | Yes. `nvidia/Metatron-v1` (`demo/model.onnx`, `racer/network.onnx` batched for 6 G1s). Tracker `g1-bones-deploy` in quickstart | Isaac Lab 3.0, Newton 1.0, Isaac Gym, MuJoCo | Inference light; AMASS training about 12 h on 4x A100 | Code Apache-2.0, weights NVIDIA Open Model License | **Easy to medium.** Same generation as 6.2 |
| **unitree_rl_gym** ([code](https://github.com/unitreerobotics/unitree_rl_gym)) | Velocity locomotion for G1, H1, H1_2 | Yes. `deploy/pre_train/{g1,h1,h1_2}/motion.pt` (no Go2 file) | Isaac Gym, MuJoCo sim2sim | Trivial inference | BSD-3 | **Medium.** Port observation and joint order |
| **unitree_rl_lab** ([code](https://github.com/unitreerobotics/unitree_rl_lab)) | Go2, H1, G1 velocity and mimic tasks | No policy files found (UNVERIFIED) | Isaac Lab 2.3, Isaac Sim 5.1 | Retrain needed | Apache-2.0 | **Medium.** Port to 3.0 and train |
| **ASAP** ([paper](https://arxiv.org/abs/2502.01143), [code](https://github.com/LeCAR-Lab/ASAP)) | Agile G1 motion tracking with delta-action sim2real | Locomotion ONNX `sim2real/models/dec_loco/.../model_6600.onnx`; tracking checkpoints UNVERIFIED | Isaac Gym, Isaac Sim 4.2, Genesis | Light | MIT | **Medium** |
| **AMO** ([paper](https://arxiv.org/abs/2505.03738), [code](https://github.com/OpenTeleVision/AMO)) | G1 whole body: vx, yaw, torso height, arm targets | Yes. `amo_jit.pt`, `adapter_jit.pt` | MuJoCo | Light | MIT | **Medium.** MuJoCo loop to port |
| **TWIST** ([paper](https://arxiv.org/abs/2505.02833), [code](https://github.com/YanjieZe/TWIST)) | G1 general motion tracker for teleop | Yes. `assets/twist_general_motion_tracker.pt` | Isaac Gym, MuJoCo | Train 1 to 2 days on one 4090 | MIT | **Medium** |
| **HOVER** ([paper](https://arxiv.org/abs/2410.21229), [code](https://github.com/NVlabs/HOVER)) | Multi-mode H1 whole-body controller | **No** | Isaac Lab 2.0 | Teacher about 23 h on a 4090, student about 0.3 h | Apache-2.0 | **Hard** |
| **OmniH2O** ([code](https://github.com/LeCAR-Lab/human2humanoid)) | H1 teleop from head and hand keypoints | **No** | Isaac Gym | Not stated | CC BY-NC 4.0 | **Hard** |
| **BeyondMimic** ([paper](https://arxiv.org/abs/2508.08241), [code](https://github.com/HybridRobotics/whole_body_tracking)) | Per-motion G1 tracking | **No** (users' own WandB only); repo archived, moved to mjlab | Isaac Lab 2.1 | Not stated | MIT | **Hard** |
| HumanoidVerse, HugWBC, FALCON, HOMIE, ULC, LeVERB | Various G1 and H1 controllers | **No public weights** (LeVERB ships only a dataset; HOMIE is CC BY-NC-SA) | Mostly Isaac Gym | n/a | Mixed | **Hard** |

Cross-embodiment note: the only verified multi-robot-type policy is URMA ([code](https://github.com/nico-bohlinger/one_policy_to_run_them_all), MuJoCo, JAX, MIT code). Pretrained path UNVERIFIED. Hard in Isaac.

## B. VLA and VLM layer

| Candidate | What it does | Weights (HF) | Official sim eval | GPU (inference, fine-tune) | License | 6.2 tomorrow |
|---|---|---|---|---|---|---|
| **GR00T N1.6** ([code](https://github.com/NVIDIA/Isaac-GR00T), branch `n1d6`, [N1 paper](https://arxiv.org/abs/2503.14734)) | Humanoid VLA. G1 action keys include `navigate_command` and `base_height_command`, which a WBC turns into joints | `nvidia/GR00T-N1.6-3B`, `nvidia/GR00T-N1.6-G1-PnPAppleToPlate`, and **`nvidia/GN1x-Tuned-Arena-G1-Loco-Manipulation`** (Isaac Lab-Arena G1 walk, pick, place) | Isaac Lab-Arena, LIBERO, SimplerEnv, RoboCasa | 16 GB+; fine-tune 40 GB+ (fits L40S) | Code Apache-2.0; N1.5 and N1.6 weights **non-commercial** | **Medium.** Arena main targets 6.2, but Arena is alpha |
| **GR00T N1.7** (main branch) | N1.6 with a new VLM backbone; `UNITREE_G1_SONIC` tag drives GEAR-SONIC via latent action tokens | `nvidia/GR00T-N1.7-3B`, `-LIBERO`, `-DROID`, `-SimplerEnv-Bridge`, `-SimplerEnv-Fractal` | LIBERO, SimplerEnv | Same as N1.6 | Weights NVIDIA Open Model License (**commercial OK**) | **Medium to hard.** No Arena G1 checkpoint yet (UNVERIFIED) |
| **openpi pi0, pi0-FAST, pi0.5** ([code](https://github.com/Physical-Intelligence/openpi), [pi0.5](https://arxiv.org/abs/2504.16054)) | PaliGemma-based flow and FAST-token VLAs | Official on GCS (`gs://openpi-assets/checkpoints/pi05_base` etc.); LeRobot ports `lerobot/pi0_base`, `lerobot/pi05_base` | Isaac Lab-Arena, LIBERO, ALOHA sim | Above 8 GB; LoRA above 22.5 GB (borderline L4); full above 70 GB | Code Apache-2.0, weights Gemma terms | **Medium.** Arms only |
| **SmolVLA** ([paper](https://arxiv.org/abs/2506.01844), [code](https://github.com/huggingface/lerobot)) | 450M VLA, cheap fine-tune | `lerobot/smolvla_base`, `nvidia/smolvla-arena-gr1-microwave` | Arena (blog used Isaac Sim 5.1), LIBERO | About 4 h for 20k steps on one A100; L4 fine | Apache-2.0 | **Easy to medium.** Arms only |
| **AgiBot GO-1** ([paper](https://arxiv.org/abs/2503.06669), [code](https://github.com/OpenDriveLab/AgiBot-World)) | VLA with latent planner on InternVL2.5 | `agibot-world/GO-1`, `agibot-world/GO-1-Air` | LIBERO, RoboTwin; Genie Sim needs Isaac Sim 5.1 | About 7 GB; fine-tune 24 to 70 GB | CC BY-NC-SA 4.0 | **Hard** |
| **OpenVLA, OpenVLA-OFT** ([paper](https://arxiv.org/abs/2406.09246), [code](https://github.com/openvla/openvla), [OFT](https://github.com/moojink/openvla-oft)) | 7B single-arm VLA | `openvla/openvla-7b`, `moojink/openvla-7b-oft-finetuned-libero-spatial` | LIBERO | OFT about 16 GB; LoRA about 27 GB | MIT code, Llama 2 terms | **Medium to hard.** Own wrapper |
| **RDT-1B, RDT2** ([RDT](https://github.com/thu-ml/RoboticsDiffusionTransformer), [RDT2](https://github.com/thu-ml/RDT2)) | Bimanual diffusion policies | `robotics-diffusion-transformer/rdt-1b`, `.../RDT2-FM`, `.../RDT2-VQ` | ManiSkill; RDT2 none | RDT2 about 16 GB | MIT; RDT2 Apache-2.0 | **Hard** |
| **Octo** ([code](https://github.com/octo-models/octo)) | Small generalist arm policy | `rail-berkeley/octo-base-1.5` | Gym wrappers only | Small | MIT | **Medium to hard.** Old JAX pin |
| **X-VLA** ([code](https://github.com/2toinf/X-VLA)) | 0.9B soft-prompt VLA, in LeRobot | `2toINF/X-VLA-Pt` and fine-tunes | LIBERO | Small | Apache-2.0 | **Medium** |
| **Cosmos-Reason2** (planner, not a VLA) | Physical-reasoning VLM on Qwen3-VL; NVIDIA lists robot planning as a use | `nvidia/Cosmos-Reason2-2B` (24 GB min), `nvidia/Cosmos-Reason2-8B` (32 GB min) | None | As stated | NVIDIA Open Model License (commercial OK) | **Easy** as a separate server |
| **Qwen3-VL-8B-Instruct** (planner) | General VLM with tool calling | `Qwen/Qwen3-VL-8B-Instruct` | None | About 18 GB in bf16 (estimate) | Apache-2.0 | **Easy** |

Others seen but not worth tomorrow: Cosmos Policy (training needs 8x80 GB), UnifoLM-VLA (`unitreerobotics/Unifolm-VLA-Base`, CC BY-NC-SA, LIBERO only), InternVLA-M1, WALL-OSS, GraspVLA, Dexbotic. GR00T N2 has no public weights (UNVERIFIED). Gemini Robotics On-Device is trusted-tester only; Gemini Robotics-ER 1.5 is a cloud API planner.

Only GR00T (N1.6 and N1.7 on G1) natively emits base commands for a locomotion layer. RDT-1B has a wheeled-base slot in its action space. Everything else is arm only.

## C. Multi-robot coordination methods

| Candidate | What it does | Weights | Sim | GPU | License | 6.2 tomorrow |
|---|---|---|---|---|---|---|
| **CooHOI** (baseline, [paper](https://arxiv.org/abs/2406.14558), [code](https://github.com/Winston-Gu/CooHOI)) | Two humanoids carry a box; AMP single-agent skill then MAPPO two-agent stage | Yes. `SingleAgent.pth`, `TwoAgent.pth` (about 42 MB each, in our local clone at repo root) | Isaac Gym Preview 4 | About 15 h on one 3090Ti, 4096 envs | **None** | **Hard** to port: Isaac Lab 3.0 has skrl AMP and `DirectMARLEnv` but no AMP plus MAPPO glue |
| **COLA** ([paper](https://arxiv.org/abs/2510.14293), [code](https://github.com/Yushi-Du/COLA_Code)) | One G1 policy for leader and follower co-carrying with a human | **No** | Isaac Lab on Isaac Sim 5.1 | Docs use 8 GPUs x 2048 envs, hours not stated | MIT | **Medium.** Version bump plus retrain |
| **Multi-Humanoid Pickup and Transport** ([paper](https://arxiv.org/abs/2609.17824), Sept 2026) | Decentralized object-centric control, teams of humanoids (Digit) pinching a shared object | **No code** | Isaac Lab per project site (UNVERIFIED on arXiv) | Not stated | n/a | **Hard** |
| **Multi-Quadruped Pinch-Lift-Move** ([paper](https://arxiv.org/abs/2509.14342)) | 2 to 10 legged manipulators move a shared object | No code found (UNVERIFIED) | UNVERIFIED | n/a | n/a | **Hard** |
| **MAPush** ([paper](https://arxiv.org/abs/2411.07104), [code](https://github.com/collaborative-mapush/MAPush)) | Hierarchical MARL for multi-Go1 box pushing | Yes, in repo (`resources/goals_net/example.pt`, `resources/command_nets/1.2x1.2.pt`) | Isaac Gym (MQE) | Not stated | None | **Hard** in 6.2, easy in Isaac Gym |
| **CoLF** ([paper](https://arxiv.org/abs/2602.07776)) | Two Go1s, leader and follower transport with CLIPSeg grounding | **No** | Isaac Lab | n/a | n/a | **Hard** |
| **Isaac Lab 3.0 MARL** ([code](https://github.com/isaac-sim/IsaacLab)) | `DirectMARLEnv` with `Isaac-Shadow-Handover-Direct` and `Isaac-Pendulum-MARL-Direct`, skrl IPPO and MAPPO configs | Not needed | Isaac Lab 3.0, Isaac Sim 6.2 | Handover trains in about 30 min on a DGX Spark (UNVERIFIED for L40S) | BSD-3 | **Easy** |
| **Isaac Sim multi-robot Nav2** ([docs](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/ros2_tutorials/robot_control/tutorial_ros2_multi_navigation.html)) | Three Nova Carters, namespaced Nav2 | Not needed | Isaac Sim 6.2 | Light | NVIDIA | **Easy** (install ROS 2 and Nav2) |
| HARL, Bi-DexHands, MARLlib, BenchMARL | MARL libraries | n/a | None in Isaac Lab (Bi-DexHands is Isaac Gym) | n/a | Mixed | **Hard** or not applicable |
| Mega Omniverse Blueprint | Factory fleet digital twin | No public repo | n/a | n/a | Enterprise | **Not available** |

### LLM and VLM task allocation (no planner weights needed, they call an LLM)

| Method | Idea | Original sim | Code, license | Reuse in 6.2 |
|---|---|---|---|---|
| **SMART-LLM** ([paper](https://arxiv.org/abs/2309.10062)) | Decompose task, form coalitions, allocate to robots | AI2-THOR | [repo](https://github.com/SMARTlab-Purdue/SMART-LLM), no license | **Easy** to copy the prompt pattern onto our skill API |
| **LaMMA-P** ([paper](https://arxiv.org/abs/2409.20560)) | LLM writes PDDL, Fast Downward plans | AI2-THOR | [repo](https://github.com/tasl-lab/LaMMA-P), no license | **Easy** for the planner part |
| **RoCo** ([paper](https://arxiv.org/abs/2307.04738)) | Arms negotiate plans in dialog | MuJoCo | [repo](https://github.com/MandiZhao/robot-collab), MIT | Easy for prompts, hard for envs |
| **COHERENT** ([paper](https://arxiv.org/abs/2409.15146)) | Propose, execute, feedback, adjust loop over heterogeneous robots | Isaac Sim 2022.2 plus OmniGibson | [repo](https://github.com/MrKeee/COHERENT), no license | **Medium to hard.** Old sim pin |
| **DART-LLM** ([paper](https://arxiv.org/abs/2411.09022)) | Dependency graph decomposition | Unity plus ROS | HF `YongdongWang/llama-3.2-1b-lora-qlora-dart-llm` | Medium |
| **RoboOS** ([repo](https://github.com/FlagOpen/RoboOS)) | Brain and cerebellum multi-robot OS with shared memory | No sim | Apache-2.0, planner `BAAI/RoboBrain2.0-7B` | **Medium.** Write Isaac skill adapters |
| EMOS, CoELA, MALMM, CLiMRS | Group discussion and communicating agents | Habitat, TDW, RLBench | Mixed | Medium; MALMM is single-arm |

## D. No weights: what training would cost

L4 throughput for Isaac Lab is roughly 2 to 3 times slower than a 4090 (my estimate from FP32 specs, UNVERIFIED for Isaac). Colab sessions cap around 24 h, so anything longer needs checkpoint resume. Running Isaac Lab on Colab at all is UNVERIFIED.

| Method | Paper figure | L40S estimate | Colab L4 feasible? |
|---|---|---|---|
| CooHOI retrain in Isaac Lab | About 15 h, one 3090Ti | About 10 to 15 h | Borderline, 30 to 45 h with resume |
| HOVER | About 23 h teacher on a 4090 | About 20 h | Borderline, 50 to 70 h |
| COLA | 8 GPUs, hours not stated | Unknown, cut envs | Unlikely |
| G1 or Go2 velocity policy | Usually 1 to 2 h | Under 2 h | Yes |
| SONIC from scratch | About 21k GPU hours | No | No |

## E. Top 3 to try tomorrow

1. **Isaac Sim 6.2 policy examples plus an LLM allocator in our reconstructed factory.** Load H1 (`--num-robots N`), Go2 or Spot, each type on one shared velocity policy (same checkpoint per robot, run once per robot), and put a SMART-LLM style planner on top. Reason: it is the only path with zero porting, it already demonstrates "one shared policy, N robots", and it gives the customer a fleet demo the same day.
2. **Isaac Lab-Arena (main, Isaac Sim 6.2) with `nvidia/GN1x-Tuned-Arena-G1-Loco-Manipulation`.** Reason: it is NVIDIA's own VLA over WBC stack (GR00T N1.6 sends `navigate_command` and arm targets to a G1 whole-body controller) and it is the most direct proof that a VLA can sit on a trained low-level RL policy in 6.2. Caveats: Arena is alpha and the weights are non-commercial, so use it for internal evaluation, and move to N1.7 (commercial) later.
3. **Isaac Lab 3.0 `Isaac-Shadow-Handover-Direct` with skrl MAPPO, as the scaffold for a CooHOI port.** Reason: it is the native multi-agent path in our stack, and CooHOI's second stage is MAPPO. Run it to learn the `DirectMARLEnv` API, then budget about 10 to 15 L40S hours to retrain CooHOI's carry in Isaac Lab, using ProtoMotions v3 or Isaac Lab AMP for the single-agent skill. Keep the original CooHOI in Isaac Gym for the cooperative-carry video meanwhile.

Runner-up: ProtoMotions v3 with `nvidia/Metatron-v1`, if we want G1s that walk around factory boxes and recover from pushes, already batched for six robots.

## F. Architecture sketch: VLM planner over a shared RL policy for N robots

```
 Camera renders + world state JSON           (0.2 to 1 Hz)
            |
   [Planner VLM] Cosmos-Reason2-8B or Qwen3-VL-8B, served by vLLM
            |   tool calls: assign(robot_id, skill, args)
   [Skill executor + traffic manager]  (Python, 10 to 50 Hz)
            |   per robot: goal -> path on factory occupancy map -> (vx, vy, yaw rate)
   [One shared locomotion policy]  physx_policy.pt, batch [N, obs] -> [N, joints]  (50 Hz)
            |
   Isaac Sim 6.2 articulations x N
```

- **Planner.** `nvidia/Cosmos-Reason2-8B` (commercial license, 32 GB) as a separate vLLM process. Isaac Sim 6.2 wants 16 GB or more, so 8B plus Isaac on one 48 GB L40S is tight. Default: `nvidia/Cosmos-Reason2-2B` on the same GPU, or run the 8B on the Colab L4 or a second machine. It only runs every few seconds, so network latency does not matter.
- **Planner input.** A top-down render plus a JSON state: robot ids, type, pose, battery or status, current skill and its progress; object and station poses from the reconstruction; the job list.
- **Planner output.** A fixed tool schema, for example `{"robot": "h1_2", "skill": "goto", "args": {"station": "press_3"}}`, plus `wait_for`, `follow`, `stop`. Validate against the schema and the map before executing. Re-plan only on events (skill done, failed, blocked, new job), not every tick.
- **Skill executor.** Turns `goto` into a path (A* on the occupancy grid from our reconstruction), then a simple pursuit controller into a velocity command, clipped to the range the policy was trained on. A reservation table on grid cells keeps robots from colliding at crossings. Reports `running`, `done`, `failed` back to the planner.
- **Low-level policy.** One set of weights per robot type (H1 `physx_policy.pt`, Go2 policy), each evaluated once per step on a batch of all robots of that type. Adding a robot adds a row, not a model.
- **Upgrade path.** For manipulation, swap a robot's skill for GR00T N1.6 or N1.7 on G1: the planner still emits `pick(object)`, and GR00T emits `navigate_command`, `base_height_command` and arm targets for the WBC. For cooperative carry, add a `carry_with(partner, object)` skill backed by the ported CooHOI policy.
