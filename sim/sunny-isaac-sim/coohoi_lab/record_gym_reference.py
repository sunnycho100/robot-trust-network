"""Records the original CooHOI rollout in Isaac Gym to an .npz for compare_reference.py: the first N observations, the
policy's actions, the full state each observation was computed from, and the joint and body names, limits and gains.
Usage (Brev VM, `conda activate coohoi`, from the CooHOI repo root; the same args as coohoi/run_coohoi.sh plus --seed):
  python ~/robot/isaac-sim/coohoi_lab/record_gym_reference.py --out ~/ref_pair.npz --steps 10 \\
    --test --task ShareHumanoidCarryObject --num_envs 16 --headless --seed 0 \\
    --cfg_env coohoi/data/cfg/share_humanoid_carrybox.yaml \\
    --cfg_train coohoi/data/cfg/train/share_humanoid_task_coohoi.yaml \\
    --motion_file coohoi/data/motions/coohoi_data/coohoi_data.yaml --checkpoint coohoi/data/models/TwoAgent.pth
Single agent: --task HumanoidAMPCarryObject --cfg_env coohoi/data/cfg/humanoid_carrybox.yaml
  --cfg_train coohoi/data/cfg/train/amp_humanoid_task.yaml --checkpoint coohoi/data/models/SingleAgent.pth
Keep --num_envs >= 4: CooHOI lays envs out int(sqrt(n)) per row and assumes env 2p+1 sits 10 m along +x of env 2p.
It builds the env and the rl_games player exactly as run.py --test does, then steps it by hand.
unverified: written from the CooHOI and rl_games 1.1.4 source and Isaac Gym Preview 4 API names, not run yet.
"""
import argparse
import os
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=10)
    rec, rest = ap.parse_known_args()
    sys.argv = [sys.argv[0]] + rest  # CooHOI's own parser (gymutil) sees only its own args
    if not os.path.isdir("coohoi/env"):
        sys.exit("Run this from the CooHOI repo root (its configs are cwd-relative).")
    sys.path.insert(0, os.path.join(os.getcwd(), "coohoi"))
    import run  # CooHOI's coohoi/run.py; imports isaacgym before torch, as Isaac Gym requires
    import torch
    from utils.config import get_args, load_cfg, set_np_formatting, set_seed

    # run.main() up to runner.run(): its create_rlgpu_env reads these module globals
    set_np_formatting()
    run.args = args = get_args()
    run.cfg, run.cfg_train, _ = load_cfg(args)
    cfg, cfg_train = run.cfg, run.cfg_train
    cfg_train["params"]["seed"] = set_seed(cfg_train["params"].get("seed", -1), cfg_train["params"].get("torch_deterministic", False))
    cfg["env"]["motion_file"] = args.motion_file
    cfg["env"]["eval_mode"], cfg["env"]["save_motions"] = False, False
    cfg_train["params"]["config"]["train_dir"] = args.output_path
    runner = run.build_alg_runner(run.RLGPUAlgoObserver())
    runner.load(cfg_train)
    player = runner.create_player()  # creates the env through run.create_rlgpu_env
    player.restore(args.checkpoint)
    env, task = player.env, player.env.task
    gym, e0, h0 = task.gym, task.envs[0], task.humanoid_handles[0]

    t_ = lambda x: x.detach().cpu().numpy().copy()  # noqa: E731
    keys = ["obs", "actions", "pd_targets", "root_states", "dof_pos", "dof_vel", "body_pos", "body_rot", "body_vel",
            "body_ang_vel", "box_states", "tar_pos", "tar_rot", "progress", "done"]
    log = {k: [] for k in keys}
    obs_dict, done_ids = player.env_reset(), []
    player.get_batch_size(obs_dict["obs"], 1)  # as common_player.run() does: marks obs as batched, else get_action flattens all envs
    for _ in range(rec.steps):
        obs_dict = player.env_reset(done_ids)  # the player's own loop: reset finished envs, read obs_buf
        # the state the observation was computed from (the offset envs' box and target already hold the copies)
        for k, v in (("obs", obs_dict["obs"]), ("root_states", task._humanoid_root_states), ("dof_pos", task._dof_pos),
                     ("dof_vel", task._dof_vel), ("body_pos", task._rigid_body_pos), ("body_rot", task._rigid_body_rot),
                     ("body_vel", task._rigid_body_vel), ("body_ang_vel", task._rigid_body_ang_vel),
                     ("box_states", task._box_states), ("tar_pos", task._target_pos), ("tar_rot", task._target_rot),
                     ("progress", task.progress_buf)):
            log[k].append(t_(v))
        act = player.get_action(obs_dict, True)  # deterministic mu, clamped to +-1, as in --test
        log["actions"].append(t_(act))
        log["pd_targets"].append(t_(task._action_to_pd_targets(act.clamp(-1.0, 1.0))))
        _, _, done, _ = player.env_step(env, act)
        log["done"].append(t_(done))
        done_ids = done.nonzero(as_tuple=False)[:, 0]

    out = {k: np.stack(v) for k, v in log.items()}  # (steps, num_envs, ...)
    # unverified: these Isaac Gym getters and property field names (no Preview 4 docs here); fix names if one raises
    out["dof_names"] = np.array(gym.get_actor_dof_names(e0, h0))
    out["body_names"] = np.array(gym.get_actor_rigid_body_names(e0, h0))
    props = gym.get_actor_dof_properties(e0, h0)
    for f in props.dtype.names:  # lower, upper, stiffness, damping, effort, velocity, armature, friction, ...
        out["dof_" + f] = np.asarray(props[f]).astype(np.float64)
    out["pd_offset"], out["pd_scale"] = t_(task._pd_action_offset), t_(task._pd_action_scale)
    out["body_mass"] = np.array([p.mass for p in gym.get_actor_rigid_body_properties(e0, h0)])
    hs = gym.get_actor_rigid_shape_properties(e0, h0)
    out["humanoid_shape_friction"] = np.array([s.friction for s in hs])
    b1 = task.envs[1]
    out["box_mass"] = np.array([gym.get_actor_rigid_body_properties(b1, task._box_handles[1])[0].mass])
    bs = gym.get_actor_rigid_shape_properties(b1, task._box_handles[1])[0]
    for f in ("friction", "rolling_friction", "torsion_friction", "restitution", "contact_offset", "rest_offset"):
        out["box_shape_" + f] = np.array([getattr(bs, f)])
    out["box_width"], out["box_length"] = t_(task._width_box_size), t_(task._length_box_size)
    if hasattr(task, "_height_box_size"):
        out["box_height"] = t_(task._height_box_size)
    out["asset_density"] = t_(task.asset_density)
    out["env_origins"] = np.array([[o.x, o.y, o.z] for o in (gym.get_env_origin(e) for e in task.envs)])
    sp = gym.get_sim_params(task.sim)
    out["sim_dt"], out["substeps"], out["control_freq_inv"] = np.array(sp.dt), np.array(sp.substeps), np.array(task.control_freq_inv)
    out["task"], out["checkpoint"], out["seed"] = np.array(args.task), np.array(args.checkpoint), np.array(cfg_train["params"]["seed"])
    if hasattr(task, "base_env_ids"):  # pair: offset env 2p (left), base env 2p+1 (right)
        out["base_env_ids"], out["offset_env_ids"] = t_(task.base_env_ids), t_(task.offset_env_ids)
        out["spacing"], out["offset"] = np.array(task.spacing), np.array(task.offset)
    np.savez(rec.out, **out)

    # what to eyeball in the log: is each humanoid where the layout in coohoi_env.py assumes?
    for i in range(min(2, task.num_envs)):
        d = out["root_states"][0, i, :2] - out["box_states"][0, i, :2]
        print(f"[ref] env {i}: origin {out['env_origins'][i].round(2)}, root {out['root_states'][0, i, :3].round(2)}, "
              f"root - observed box ({d[0]:.2f}, {d[1]:.2f}) m")
    print(f"[ref] dof effort {out['dof_effort'].round(1).tolist()}")
    print(f"[ref] box mass {out['box_mass'][0]:.2f} kg, friction {out['box_shape_friction'][0]}, "
          f"sim dt {float(sp.dt):.5f} x {int(sp.substeps)} substeps, decimation {task.control_freq_inv}")
    print(f"[ref] saved {rec.steps} steps x {task.num_envs} envs -> {rec.out}", flush=True)
    os._exit(0)  # ponytail: skip Isaac Gym's teardown, which often segfaults at exit (as record_coohoi.py)


if __name__ == "__main__":
    main()
