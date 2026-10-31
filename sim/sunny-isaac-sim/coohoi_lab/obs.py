"""CooHOI's policy observation (223 humanoid + 76 task = 299) in plain torch, ported from the Isaac Gym code.
Usage: import obs; o = torch.cat([obs.humanoid_obs(...), obs.pair_task_obs(...)], -1). Checked by test_obs.py.

Quaternions are (x, y, z, w) everywhere here: Isaac Gym's tensor API uses xyzw, and so does Isaac Lab 3.0
(docs/source/isaaclab/migrating.rst at v3.0.0: "The quaternion format changed from WXYZ to XYZW", and
isaaclab.utils.math works in xyzw). So no reorder is needed between the two; LAB_QUAT_ORDER and to_xyzw() keep the
conversion explicit in case an API hands over wxyz (USD, Isaac Sim core, older Isaac Lab).

Sources (CooHOI 3183d99): share_humanoid.py compute_humanoid_observations_max (humanoid part),
share_humanoid_amp_carryobject.py __init__ / update_standing_and_held_points / compute_carrybox_observations (pair),
humanoid_amp_carryobject.py (single agent), utils/torch_utils.py (heading and tan-norm helpers).
"""
import math

import torch

LAB_QUAT_ORDER = "xyzw"  # Isaac Lab 3.0 data and math utils (verified in the v3.0.0 source)


def to_xyzw(q, order=LAB_QUAT_ORDER):
    return q if order == "xyzw" else q[..., [1, 2, 3, 0]]


def from_xyzw(q, order=LAB_QUAT_ORDER):
    return q if order == "xyzw" else q[..., [3, 0, 1, 2]]


# --- quaternion helpers, same math as isaacgym.torch_utils and CooHOI utils/torch_utils.py (xyzw) ---

def normalize(x, eps=1e-9):
    return x / x.norm(p=2, dim=-1).clamp(min=eps).unsqueeze(-1)


def quat_mul(a, b):
    x1, y1, z1, w1 = a.unbind(-1)
    x2, y2, z2, w2 = b.unbind(-1)
    return torch.stack([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2], dim=-1)


def quat_rotate(q, v):
    w, u = q[..., 3:4], q[..., :3]
    return v * (2.0 * w * w - 1.0) + torch.cross(u, v, dim=-1) * w * 2.0 + u * (u * v).sum(-1, keepdim=True) * 2.0


def quat_from_angle_axis(angle, axis):
    half = (angle / 2).unsqueeze(-1)
    return normalize(torch.cat([normalize(axis) * half.sin(), half.cos()], dim=-1))


def _z_axis(like):
    z = torch.zeros_like(like[..., 0:3])
    z[..., 2] = 1.0
    return z


def calc_heading(q):
    x = torch.zeros_like(q[..., 0:3])
    x[..., 0] = 1.0
    d = quat_rotate(q, x)
    return torch.atan2(d[..., 1], d[..., 0])


def calc_heading_quat(q):
    return quat_from_angle_axis(calc_heading(q), _z_axis(q))


def calc_heading_quat_inv(q):
    return quat_from_angle_axis(-calc_heading(q), _z_axis(q))


def quat_to_tan_norm(q):
    t = torch.zeros_like(q[..., 0:3])
    t[..., 0] = 1.0
    n = torch.zeros_like(q[..., 0:3])
    n[..., 2] = 1.0
    return torch.cat([quat_rotate(q, t), quat_rotate(q, n)], dim=-1)


def yaw_quat(theta):
    """Rotation by theta about +z, the way CooHOI builds every box and target rotation."""
    return quat_from_angle_axis(theta, _z_axis(torch.zeros(*theta.shape, 4, dtype=theta.dtype, device=theta.device)))


# --- humanoid part (223) ---

def humanoid_obs(body_pos, body_rot, body_vel, body_ang_vel):
    """compute_humanoid_observations_max with localRootObs=True and rootHeightObs=True. Inputs (N, 15, 3|4), bodies in
    MJCF order, body 0 = pelvis. Note CooHOI's (ASE's) quirk: with localRootObs=True the root slot of the rotation block
    holds the root's world rotation, not the heading-local one, so the policy sees its absolute yaw."""
    n, b = body_pos.shape[:2]
    root_pos, root_rot = body_pos[:, 0], body_rot[:, 0]
    heading = calc_heading_quat_inv(root_rot).unsqueeze(1).expand(n, b, 4)
    local_pos = quat_rotate(heading, body_pos - root_pos.unsqueeze(1)).reshape(n, -1)[:, 3:]
    rot_obs = quat_to_tan_norm(quat_mul(heading, body_rot)).reshape(n, -1).clone()
    rot_obs[:, 0:6] = quat_to_tan_norm(root_rot)
    vel = quat_rotate(heading, body_vel).reshape(n, -1)
    ang_vel = quat_rotate(heading, body_ang_vel).reshape(n, -1)
    return torch.cat([root_pos[:, 2:3], local_pos, rot_obs, vel, ang_vel], dim=-1)


# --- task part (76) ---

def _local_points(points, root_pos, heading, center_pos, center_rot):
    """convert_static_point_to_local_observation for (N, 8, 3) box-frame points: world, then heading-local."""
    world = center_pos.unsqueeze(1) + quat_rotate(center_rot.unsqueeze(1).expand(-1, 8, 4), points)
    return quat_rotate(heading.unsqueeze(1).expand(-1, 8, 4), world - root_pos.unsqueeze(1)).reshape(points.shape[0], 24)


def _task_obs(root_states, box_states, tar_pos, tar_rot, bps, box_point, standing_point, tar_point, box_rot_obs,
              tar_rot_obs, density):
    """The shared layout of both compute_carrybox_observations versions:
    [density, 8 target bps, target point, target rot (tan-norm), standing point, 8 box bps, box point, box rot,
     box vel, box ang vel], all in the root's heading frame."""
    root_pos, root_rot = root_states[:, 0:3], root_states[:, 3:7]
    heading = calc_heading_quat_inv(root_rot)
    box_pos, box_rot = box_states[:, 0:3], box_states[:, 3:7]
    stand = standing_point.clone()
    stand[:, 2] = 0.0
    loc = lambda v: quat_rotate(heading, v)  # noqa: E731
    return torch.cat([
        density.unsqueeze(-1),
        _local_points(bps, root_pos, heading, tar_pos, tar_rot),
        loc(tar_point - root_pos), quat_to_tan_norm(quat_mul(heading, tar_rot_obs)),
        loc(stand - root_pos),
        _local_points(bps, root_pos, heading, box_pos, box_rot),
        loc(box_point - root_pos), quat_to_tan_norm(quat_mul(heading, box_rot_obs)),
        loc(box_states[:, 7:10]), loc(box_states[:, 10:13]),
    ], dim=-1)


def pair_geometry(width, length, side):
    """Per-agent constants of the two-agent task (share_humanoid_amp_carryobject.py:64-136 and :217-264).
    side 'left' = CooHOI's offset env (even id), stands at the box's -x end; 'right' = base env (odd id), +x end.
    width, length: (N,) box sizes after scaling. Returns bps (N, 8, 3), standing and held offsets (N, 3) in the box
    frame, and the extra rotation CooHOI applies to the box and target rotations in this agent's obs."""
    w, h = width / 2.0, length / 2.0
    z = torch.zeros_like(width)
    if side == "left":
        xs, ys = (-h, width - h), (w, -w)
        stand, held, ang = (-h - 0.2, z, z), (-h + w, z, z), 0.0
    else:
        xs, ys = (h, -width + h), (-w, w)
        stand, held, ang = (h + 0.2, z, z), (h - w, z, z), math.pi
    bps = torch.stack([torch.stack([x, y, s * w], -1) for x in xs for y in ys for s in (1, -1)], 1)
    # CooHOI builds this from a float64 np.pi and stores it as float32 (cos(pi/2) = 6e-17, not float32's -4e-8)
    rot = yaw_quat(torch.tensor(ang, dtype=torch.float64)).to(width).expand(width.shape[0], 4)
    return dict(bps=bps, stand=torch.stack(stand, -1), held=torch.stack(held, -1), obs_rot=rot)


def pair_task_obs(root_states, box_states, tar_pos, tar_rot, geo, density):
    """One agent of the pair. root_states (N, 13) pelvis pos, quat, ...; box_states (N, 13); tar_pos (N, 3);
    tar_rot (N, 4); geo from pair_geometry; density (N,) as observed (CooHOI feeds 2 x 50 = 100)."""
    box_pos, box_rot = box_states[:, 0:3], box_states[:, 3:7]
    standing = box_pos + quat_rotate(box_rot, geo["stand"])
    held = box_pos + quat_rotate(box_rot, geo["held"])
    tar_held = tar_pos + quat_rotate(tar_rot, geo["held"])
    return _task_obs(root_states, box_states, tar_pos, tar_rot, geo["bps"], held, standing, tar_held,
                     quat_mul(box_rot, geo["obs_rot"]), quat_mul(tar_rot, geo["obs_rot"]), density)


def single_geometry(width, length, height):
    """Single-agent constants (humanoid_amp_carryobject.py:90-128): the 8 box corners and the left standing point."""
    w, l, hh = width / 2.0, length / 2.0, height / 2.0
    z = torch.zeros_like(width)
    bps = torch.stack([torch.stack([x, y, s], -1) for x in (-l, l) for y in (w, -w) for s in (hh, -hh)], 1)
    return dict(bps=bps, stand=torch.stack([-l - 0.2, z, z], -1))


def single_task_obs(root_states, box_states, tar_pos, tar_rot, geo, density):
    """The single-agent task obs: box and target centres, raw rotations. density (N,) = 0.8 * 50 / scale^3."""
    box_pos, box_rot = box_states[:, 0:3], box_states[:, 3:7]
    standing = box_pos + quat_rotate(box_rot, geo["stand"])
    return _task_obs(root_states, box_states, tar_pos, tar_rot, geo["bps"], box_pos, standing, tar_pos,
                     box_rot, tar_rot, density)


# Slices of the 299-dim observation, for per-term comparisons.
OBS_TERMS = [("root_h", 0, 1), ("body_pos", 1, 43), ("root_rot_world", 43, 49), ("body_rot", 49, 133),
             ("body_vel", 133, 178), ("body_ang_vel", 178, 223),
             ("density", 223, 224), ("tar_bps", 224, 248), ("tar_point", 248, 251), ("tar_rot", 251, 257),
             ("standing_point", 257, 260), ("box_bps", 260, 284), ("box_point", 284, 287), ("box_rot", 287, 293),
             ("box_vel", 293, 296), ("box_ang_vel", 296, 299)]
