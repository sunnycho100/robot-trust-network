"""python test_obs.py: checks obs.py on CPU, against hand-computed cases and against CooHOI's own observation code.
The CooHOI comparison imports the original task classes with isaacgym stubbed out. It needs Isaac Gym's real
torch_utils.py (not in the CooHOI repo): ISAACGYM_TORCH_UTILS=~/isaacgym/python/isaacgym/torch_utils.py python test_obs.py
Without it only the hand-computed cases run. CooHOI is found via $COOHOI (see joint_map.coohoi_root).
"""
import importlib.util
import math
import os
import sys
import types

import torch
import yaml

import obs
from joint_map import coohoi_root

torch.manual_seed(0)
TOL = 2e-5


def yaw(theta):
    return obs.yaw_quat(torch.as_tensor(theta, dtype=torch.float32))


def close(a, b, tol=TOL):
    return (a - b).abs().max().item() < tol


def rand_quat(*shape):
    return obs.normalize(torch.randn(*shape, 4))


# ---- hand-computed cases ----
x, y = torch.tensor([1.0, 0, 0]), torch.tensor([0, 1.0, 0])
assert close(obs.quat_rotate(yaw(math.pi / 2), x), y), "yaw +90 deg takes x to y"
assert close(obs.quat_mul(yaw(0.3), yaw(0.5)), yaw(0.8)), "yaws compose"
q = obs.quat_mul(yaw(1.0), obs.quat_from_angle_axis(torch.tensor(0.4), y))  # yaw 1 rad, then a pitch
assert close(obs.calc_heading(q), torch.tensor(1.0)) and close(obs.calc_heading_quat_inv(q), yaw(-1.0))
assert close(obs.quat_to_tan_norm(torch.tensor([0, 0, 0, 1.0])), torch.tensor([1.0, 0, 0, 0, 0, 1]))
assert obs.to_xyzw(torch.tensor([1.0, 0, 0, 0]), "wxyz").tolist() == [0, 0, 0, 1.0]
assert obs.from_xyzw(obs.to_xyzw(q, "wxyz"), "wxyz").equal(q) and obs.from_xyzw(q).equal(q)
assert obs.OBS_TERMS[0][1] == 0 and obs.OBS_TERMS[-1][2] == 299
assert all(a[2] == b[1] for a, b in zip(obs.OBS_TERMS, obs.OBS_TERMS[1:])), "terms tile 0..299"

# Mirror case: the right agent is the left one turned 180 deg about the box centre, so both see the same task obs and
# the same humanoid obs except the root's world rotation (the ASE quirk in humanoid_obs).
n = 4
w, l = torch.full((n,), 0.5), torch.full((n,), 2.0)
geo = {s: obs.pair_geometry(w, l, s) for s in ("left", "right")}
assert close(geo["left"]["stand"][0], torch.tensor([-1.2, 0, 0])) and close(geo["right"]["held"][0], torch.tensor([0.75, 0, 0]))
pos = torch.randn(n, 15, 3) * 0.3 + torch.tensor([-1.2, 0, 0.9])
rot = obs.quat_mul(yaw(torch.rand(n, 1) - 0.5).expand(n, 15, 4), rand_quat(n, 15) * 0.2 + torch.tensor([0, 0, 0, 1.0]))
rot = obs.normalize(rot)
vel, avel = torch.randn(n, 15, 3), torch.randn(n, 15, 3)
flip = yaw(math.pi).expand(n, 15, 4)
mirror = [obs.quat_rotate(flip, pos), obs.quat_mul(flip, rot), obs.quat_rotate(flip, vel), obs.quat_rotate(flip, avel)]
box = torch.zeros(n, 13)
box[:, 2], box[:, 6] = 0.25, 1.0
box[:, 7:] = torch.randn(n, 6)
box_r = box.clone()
box_r[:, 7:10], box_r[:, 10:13] = obs.quat_rotate(yaw(math.pi).expand(n, 4), box[:, 7:10]), obs.quat_rotate(yaw(math.pi).expand(n, 4), box[:, 10:13])
tar_pos, tar_rot = torch.tensor([[0.0, 3.0, 0.25]]).expand(n, 3), yaw(0.2).expand(n, 4)
den = torch.full((n,), 100.0)
root = lambda p, r: torch.cat([p[:, 0], r[:, 0], torch.zeros(n, 6)], -1)  # noqa: E731
hl = obs.humanoid_obs(pos, rot, vel, avel)
hr = obs.humanoid_obs(*mirror)
assert hl.shape == (n, 223) and close(torch.cat([hl[:, :43], hl[:, 49:]], 1), torch.cat([hr[:, :43], hr[:, 49:]], 1))
tl = obs.pair_task_obs(root(pos, rot), box, tar_pos, tar_rot, geo["left"], den)
# mirrored world: positions and velocities turn by 180 deg; the box and target rotations (yaws of a box that looks
# the same turned 180 deg) stay as they are, which is the case CooHOI's extra pi rotation for the right agent covers
tr = obs.pair_task_obs(root(mirror[0], mirror[1]), box_r, obs.quat_rotate(yaw(math.pi).expand(n, 4), tar_pos),
                       tar_rot, geo["right"], den)
assert tl.shape == (n, 76) and close(tl, tr, 1e-4), (tl - tr).abs().max()
print("hand-computed cases ok (quaternions, heading, tan-norm, mirror symmetry of the pair obs)")

# ---- against CooHOI's own code ----
ref = os.environ.get("ISAACGYM_TORCH_UTILS")
if not ref:
    print("ISAACGYM_TORCH_UTILS not set: skipped the comparison with CooHOI's original functions")
    sys.exit(0)

# Stub isaacgym: real torch_utils (pure torch), dummy gymapi/gymtorch/gymutil. CooHOI's @torch.jit.script is made a
# no-op while importing (same math, no TorchScript compile on a recent torch).
ig = types.ModuleType("isaacgym")
spec = importlib.util.spec_from_file_location("isaacgym.torch_utils", os.path.expanduser(ref))
tu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tu)
gymutil = types.ModuleType("isaacgym.gymutil")
for name in ("get_property_setter_map", "get_property_getter_map", "get_default_setter_args", "apply_random_samples",
             "check_buckets", "generate_random_samples"):
    setattr(gymutil, name, None)
gymtorch = types.ModuleType("isaacgym.gymtorch")
gymtorch.wrap_tensor = lambda t: t
ig.torch_utils, ig.gymutil, ig.gymtorch, ig.gymapi = tu, gymutil, gymtorch, types.ModuleType("isaacgym.gymapi")
sys.modules.update({"isaacgym": ig, "isaacgym.torch_utils": tu, "isaacgym.gymutil": gymutil,
                    "isaacgym.gymtorch": gymtorch, "isaacgym.gymapi": ig.gymapi})
sys.path.insert(0, os.path.join(coohoi_root(), "coohoi"))
script, torch.jit.script = torch.jit.script, (lambda f: f)
import env.tasks.humanoid_amp_carryobject as hc  # noqa: E402
import env.tasks.share_humanoid_amp_carryobject as shc  # noqa: E402
from utils import torch_utils as cut  # noqa: E402

torch.jit.script = script

# primitives
a, b, v = rand_quat(64), rand_quat(64), torch.randn(64, 3)
assert close(obs.quat_mul(a, b), tu.quat_mul(a, b)) and close(obs.quat_rotate(a, v), tu.quat_rotate(a, v))
ang = torch.randn(64)
assert close(obs.quat_from_angle_axis(ang, v), tu.quat_from_angle_axis(ang, v))
assert close(obs.calc_heading_quat_inv(a), cut.calc_heading_quat_inv(a)) and close(obs.quat_to_tan_norm(a), cut.quat_to_tan_norm(a))


class FakeGym:
    def acquire_net_contact_force_tensor(self, sim):
        return torch.zeros(self.n * 16, 3)


def fake_task(cls, parent, cfg_file, n, sizes):
    """A CooHOI task object built by its real __init__, with the Isaac Gym parent constructors replaced."""
    cfg = yaml.safe_load(open(os.path.join(coohoi_root(), "coohoi/data/cfg", cfg_file)))
    cfg["env"].update(numEnvs=n, eval_mode=False, save_motions=False)

    def parent_init(self, cfg, **_):
        self.cfg, self.device, self.sim, self.num_bodies = cfg, "cpu", None, 15
        self._enable_task_obs, self._local_root_obs, self._root_height_obs = True, True, True
        self.gym, self.gym.n = FakeGym(), n
        self.envs, self.humanoid_handles = [None], [None]
        self._root_states = torch.zeros(n * 2, 13)  # actors per env: humanoid, box
        self._humanoid_root_states = self._root_states.view(n, 2, 13)[..., 0, :]
        self._rigid_body_state = torch.zeros(n * 16, 13)
        sizes(self)  # what _load_box_asset / _build_box would have set

    parent.__init__ = parent_init
    cls._build_lift_body_ids_tensor = lambda self, names: torch.tensor([5, 8])
    return cls(cfg=cfg, sim_params=None, physics_engine=None, device_type="cpu", device_id=0, headless=True)


def set_bodies(t, n):
    pos = torch.randn(n, 15, 3) * 0.4 + torch.tensor([0, 0, 0.9]) + torch.randn(n, 1, 3) * torch.tensor([3.0, 3, 0])
    rot = rand_quat(n, 15)
    t._rigid_body_pos, t._rigid_body_rot = pos, rot
    t._rigid_body_vel, t._rigid_body_ang_vel = torch.randn(n, 15, 3), torch.randn(n, 15, 3)
    t._humanoid_root_states[:, 0:3], t._humanoid_root_states[:, 3:7] = pos[:, 0], rot[:, 0]
    t._humanoid_root_states[:, 7:] = torch.randn(n, 6)


def rand_box_target(t, n):
    t._box_states[:, 0:3] = torch.randn(n, 3) * torch.tensor([3.0, 3, 0.2]) + torch.tensor([0, 0, 0.4])
    t._box_states[:, 3:7] = rand_quat(n)
    t._box_states[:, 7:] = torch.randn(n, 6)
    t._target_pos[:] = torch.randn(n, 3) * 3
    t._target_rot[:] = rand_quat(n)


# two-agent pairs (offset env 2p = left, base env 2p+1 = right)
n = 8
scale = torch.rand(n // 2).repeat_interleave(2) * 0.1 + 0.9


def pair_sizes(self):
    self._width_box_size[:], self._length_box_size[:] = 0.5 * scale, 2.0 * scale
    self.asset_density = torch.full((n,), 50.0)


t = fake_task(shc.ShareHumanoidCarryObject, shc.share_humanoid_amp_task.ShareHumanoidAMPTask, "share_humanoid_carrybox.yaml", n, pair_sizes)
set_bodies(t, n)
rand_box_target(t, n)
want = torch.cat([t._compute_humanoid_obs(None), t._compute_task_obs(None)], -1)  # also copies base box -> offset env
assert want.shape == (n, 299)
L, R = t.offset_env_ids, t.base_env_ids
shift = torch.tensor([t.offset * t.spacing * 2, 0.0, 0.0])  # offset-env coords -> base-env coords (CooHOI: x - 10)
box, tp, tr_ = t._box_states[R].clone(), t._target_pos[R], t._target_rot[R]
got = {}
for side, ids, d in (("left", L, shift), ("right", R, torch.zeros(3))):
    geo = obs.pair_geometry(t._width_box_size[ids], t._length_box_size[ids], side)
    rs = t._humanoid_root_states[ids].clone()
    rs[:, 0:3] += d
    got[side] = torch.cat([obs.humanoid_obs(t._rigid_body_pos[ids] + d, t._rigid_body_rot[ids], t._rigid_body_vel[ids],
                                            t._rigid_body_ang_vel[ids]),
                           obs.pair_task_obs(rs, box, tp, tr_, geo, 2 * t.asset_density[ids])], -1)
for side, ids in (("left", L), ("right", R)):
    for name, s, e in obs.OBS_TERMS:
        err = (got[side][:, s:e] - want[ids, s:e]).abs().max().item()
        assert err < 1e-4, f"pair {side} {name}: max |diff| {err}"
print("pair obs matches CooHOI ShareHumanoidCarryObject (both agents, all 16 terms, offset agent in the shared frame)")

# single agent
n = 6


def single_sizes(self):
    s = self.scaling_factor
    self._width_box_size[:], self._length_box_size[:], self._height_box_size[:] = 0.5 * s, 0.5 * s, 0.5 * s
    self.asset_density = self.scaling_factor_weight * 50.0 / s ** 3


t = fake_task(hc.HumanoidAMPCarryObject, hc.humanoid_amp_task.HumanoidAMPTask, "humanoid_carrybox.yaml", n, single_sizes)
set_bodies(t, n)
rand_box_target(t, n)
want = torch.cat([t._compute_humanoid_obs(None), t._compute_task_obs(None)], -1)
geo = obs.single_geometry(t._width_box_size, t._length_box_size, t._height_box_size)
got = torch.cat([obs.humanoid_obs(t._rigid_body_pos, t._rigid_body_rot, t._rigid_body_vel, t._rigid_body_ang_vel),
                 obs.single_task_obs(t._humanoid_root_states, t._box_states, t._target_pos, t._target_rot, geo,
                                     t.asset_density)], -1)
for name, s, e in obs.OBS_TERMS:
    err = (got[:, s:e] - want[:, s:e]).abs().max().item()
    assert err < 1e-4, f"single {name}: max |diff| {err}"
print("single-agent obs matches CooHOI HumanoidAMPCarryObject (all 16 terms)")
