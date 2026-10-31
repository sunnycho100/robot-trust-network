"""Compares the Isaac Lab port with a recorded Isaac Gym rollout (record_gym_reference.py): joint and body names,
limits, gains, the policy's actions, and the observation per term. This is where ordering and gain mistakes show.
Usage (Isaac Lab 3.0 venv on Isaac Sim 6.2, from this folder):
  python compare_reference.py ~/ref_pair.npz                  # teacher forced: Lab set to Gym's state at every step
  python compare_reference.py ~/ref_pair.npz --rollout        # Gym's state at step 0, then Gym's actions open loop
  python compare_reference.py ~/ref_pair.npz --policy_only    # no Isaac Lab (CPU is fine): names, gains, actions
Teacher forced isolates the observation code, joint order and USD kinematics (expect ~1e-5 on positions; velocity
terms depend on PhysX filling body velocities from a state write, unverified). Rollout adds physics: small at step 1,
growing after. Uses the first pair (Gym envs 0 and 1) or env 0 for the single agent; one Isaac Lab env.
"""
import argparse
import os
import sys

import numpy as np
import torch

import joint_map as jm
import obs as ob
from policy import CoohoiActor

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("ref", help=".npz from record_gym_reference.py")
ap.add_argument("--rollout", action="store_true")
ap.add_argument("--policy_only", action="store_true")
ap.add_argument("--physics_dt", type=float, default=1 / 120)
ap.add_argument("--decimation", type=int, default=4)
ap.add_argument("--effort", choices=["gear", "unlimited", "usd"], default="gear")
if "--policy_only" not in sys.argv:
    from isaaclab.app import add_launcher_args, launch_simulation

    add_launcher_args(ap)


def agents_of(ref):
    """{agent: Gym env id}: the first pair (offset env = left, base env = right) or env 0."""
    if "base_env_ids" in ref:
        return {"left": int(ref["offset_env_ids"][0]), "right": int(ref["base_env_ids"][0])}
    return {"single": 0}


def lab_state(ref, t, agents):
    """Gym state at step t in one Isaac Lab env's task frame, as write_state() takes it. Each Gym env is shifted in
    xy so that its observed box (pair) or its humanoid (single) sat at the origin at step 0. The offset env observes
    the base env's box moved +10 m, so its humanoid lands 10 m to the left of the shared box, and every relative
    position each agent saw is kept."""
    pair = "base_env_ids" in ref
    box_env = agents["right"] if pair else agents["single"]
    anchor = {e: (ref["box_states"] if pair else ref["root_states"])[0, e, :2] for e in agents.values()}
    f = lambda x: torch.tensor(np.asarray(x, dtype=np.float32)).unsqueeze(0)  # noqa: E731

    def moved(x, e):
        x = np.array(x, dtype=np.float32)
        x[:2] -= anchor[e]
        return x

    roots = {a: f(moved(ref["root_states"][t, e], e)) for a, e in agents.items()}
    qpos = {a: f(ref["dof_pos"][t, e]) for a, e in agents.items()}
    qvel = {a: f(ref["dof_vel"][t, e]) for a, e in agents.items()}
    return (roots, qpos, qvel, f(moved(ref["box_states"][t, box_env], box_env)),
            f(moved(ref["tar_pos"][t, box_env], box_env)), f(ref["tar_rot"][t, box_env]))


def static_report(ref):
    names_ok = ref["dof_names"].tolist() == jm.JOINT_NAMES and ref["body_names"].tolist() == jm.BODIES
    print(f"[cmp] Gym dof and body order equals the MJCF parse in joint_map.py: {names_ok}")
    if not names_ok:
        print(f"      gym dofs {ref['dof_names'].tolist()}\n      gym bodies {ref['body_names'].tolist()}")
    off, sc = jm.pd_action_offset_scale()
    print(f"[cmp] pd offset/scale vs joint_map: max |diff| {np.abs(ref['pd_offset'] - off).max():.1e} / "
          f"{np.abs(ref['pd_scale'] - sc).max():.1e}")
    mj = {"lower": [j["lower"] for j in jm.JOINTS], "upper": [j["upper"] for j in jm.JOINTS],
          "stiffness": list(jm.gains("stiffness").values()), "damping": list(jm.gains("damping").values()),
          "armature": list(jm.gains("armature").values()), "effort": list(jm.gains("gear").values())}
    for k, v in mj.items():
        g = ref["dof_" + k]
        bad = [f"{n} gym {a:.4g} mjcf {b:.4g}" for n, a, b in zip(jm.JOINT_NAMES, g, v) if abs(a - b) > 1e-4 * max(1, abs(b))]
        print(f"[cmp] gym dof {k:9s} vs MJCF{' (gear)' if k == 'effort' else ''}: "
              + ("match" if not bad else f"{len(bad)} differ, e.g. {bad[:3]}"))
    print(f"[cmp] gym dof velocity limit {sorted(set(ref['dof_velocity'].round(3).tolist()))}, box mass "
          f"{ref['box_mass'][0]:.3f} kg, box friction {ref['box_shape_friction'][0]}, sim dt {float(ref['sim_dt']):.5f} "
          f"x {int(ref['substeps'])} substeps, control_freq_inv {int(ref['control_freq_inv'])}")


def policy_report(ref):
    ckpt = str(ref["checkpoint"])
    ckpt = ckpt if os.path.isabs(ckpt) else os.path.join(jm.coohoi_root(), ckpt)
    pi = CoohoiActor.load(ckpt)
    a = pi(torch.tensor(ref["obs"]).reshape(-1, 299)).numpy()
    print(f"[cmp] policy.py on Gym obs vs Gym's rl_games actions ({os.path.basename(ckpt)}): max |diff| "
          f"{np.abs(a - ref['actions'].reshape(-1, 28)).max():.1e}")
    return pi


def term_row(lab, gym):
    """Max |diff| per obs term; for body terms also the body with the largest diff."""
    out = []
    for name, s, e in ob.OBS_TERMS:
        d = (lab[s:e] - gym[s:e]).abs()
        cell = f"{d.max().item():.1e}"
        if name in ("body_pos", "body_rot", "body_vel", "body_ang_vel") and d.max() > 1e-3:
            names = jm.BODIES[1:] if name in ("body_pos", "body_rot") else jm.BODIES
            cell += " " + names[d.reshape(len(names), -1).max(-1).values.argmax().item()]
        out.append(cell)
    return out


def main():
    args = ap.parse_args()
    ref = dict(np.load(args.ref))
    agents = agents_of(ref)
    print(f"[cmp] {args.ref}: task {ref['task']}, {ref['obs'].shape[0]} steps, agents {agents}")
    static_report(ref)
    pi = policy_report(ref)
    if args.policy_only:
        return
    from coohoi_env import CoohoiEnv, make_cfg

    pair = "base_env_ids" in ref
    scale = float(ref["box_width"][agents["right" if pair else "single"]]) / 0.5
    cfg = make_cfg("pair" if pair else "single", 1, scale, args.physics_dt, args.decimation, args.effort)
    with launch_simulation(cfg, args):
        env = CoohoiEnv(cfg)
        env.reset()
        dev, ids = env.device, torch.tensor([0], device=env.device)
        r, g = env.robots[env.possible_agents[0]], jm.JointMap(env.robots[env.possible_agents[0]].joint_names)
        lab = lambda x: x.torch[0][g.joints_gym_from_lab].cpu().numpy()  # noqa: E731
        for k, attr in (("stiffness", "joint_stiffness"), ("damping", "joint_damping"), ("armature", "joint_armature"),
                        ("effort", "joint_effort_limits"), ("velocity", "joint_vel_limits")):
            v = lab(getattr(r.data, attr))
            print(f"[cmp] Lab joint {k:9s} vs Gym: max |diff| {np.abs(v - ref['dof_' + k]).max():.3g}  (lab {v[:4].round(3).tolist()}...)")
        lim = r.data.joint_pos_limits.torch[0][g.joints_gym_from_lab].cpu().numpy()
        print(f"[cmp] Lab joint limits vs Gym: max |diff| lower {np.abs(lim[:, 0] - ref['dof_lower']).max():.1e}, "
              f"upper {np.abs(lim[:, 1] - ref['dof_upper']).max():.1e}")
        print(f"[cmp] Lab box mass {env.box.data.body_mass.torch.flatten()[0].item():.3f} kg, Gym {ref['box_mass'][0]:.3f} kg")
        rows = {a: [] for a in agents}
        for t in range(ref["obs"].shape[0]):
            if t == 0 or not args.rollout:
                env.write_state(ids, *[{k: v.to(dev) for k, v in x.items()} if isinstance(x, dict) else x.to(dev)
                                       for x in lab_state(ref, t, agents)])
                o = env._get_observations()
            else:
                o, *_ = env.step({a: torch.tensor(ref["actions"][t - 1, e], device=dev).unsqueeze(0) for a, e in agents.items()})
            for a, e in agents.items():
                gym_obs = torch.tensor(ref["obs"][t, e], device=dev)
                p, q, _, _ = env.bodies(a)  # forward kinematics check, independent of the obs code
                fk = (p[0] - p[0, :1]) - torch.tensor(ref["body_pos"][t, e] - ref["body_pos"][t, e, :1], device=dev)
                act = (pi.to(dev)(o[a])[0] - torch.tensor(ref["actions"][t, e], device=dev)).abs().max().item()
                rows[a].append(term_row(o[a][0], gym_obs) + [f"{fk.abs().max().item():.1e}", f"{act:.1e}"])
        names = [n for n, _, _ in ob.OBS_TERMS] + ["fk_body_pos", "action"]
        for a, rr in rows.items():
            print(f"\n[cmp] {a} ({'rollout' if args.rollout else 'teacher forced'}), max |Lab - Gym| per term, columns = step")
            for i, n in enumerate(names):
                print(f"  {n:15s} " + "  ".join(f"{row[i]:>8s}" for row in rr))
        env.close()


if __name__ == "__main__":
    main()
