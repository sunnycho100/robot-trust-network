"""Side-squeeze variants of Seohee's 2-G1 crate and 3-G1 fixture carries: scenarios 'crate_side', 'fixture_side'.

Added beside her code at runtime (install() patches her modules; her files stay unchanged). It follows her
sofa_side grip: each robot clamps the load between its own two palms, the palm pads are turned to face the
load, the arms get gravity compensation, and her measured place, release and return controller runs as is.
An IK reset puts every pad flat on a side face; the arm targets sit SQUEEZE inside that face, so the PD servos
press and friction carries the load.
  crate_side:   each G1 clamps the two long side faces of the crate near its own end.
  fixture_side: left and right G1s clamp the crossbar's front and back faces; the rear G1 clamps the stem sides.
  tote_side:    EXPERIMENTAL, not used by realistic_layout_v3.json and not passing yet. Her single-G1 sofa_side carry
                without the sofa back, the tote held level by flat palm pads on its side walls. Two palms on
                opposite walls act as a hinge: with hand-mesh contact the tote rolled 75 to 90 deg about them;
                with thick flat pads (squeeze .05, centre .90 m, reach .42 m) it stayed within 22 deg but only
                about half its weight reached the table and the release never finished.
Pressing on the end faces instead (one robot against the other) was tried first and drops the load in
0.4 s: the reaction pushes both robots backwards and the walking policy, a velocity tracker, cannot hold
a steady push, so the pads lose contact.
Usage (standalone check, her probe pattern):
  PYTHONPATH=<g1_cooperative_task>/sim/mujoco_houses/src python side_grip.py --scenario crate_side
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
from unitree_demo import g1_logistics, industrial_showcase
from unitree_demo.flexible_factory import carry_controller, plant_view

BASE = {'crate_side': 'bottom_crate', 'fixture_side': 'team_fixture', 'tote_side': 'sofa_side'}
SQUEEZE = .015  # arm target this far inside the face (m); .008 let the 3 kg T-frame slip, .015 to .025 pass
HEIGHT = {'crate_side': 1.10, 'fixture_side': 1.06, 'tote_side': .76}  # payload centre at reset; arms sag under load
REACH = {'tote_side': .40}  # payload x at reset when one robot carries it in front of itself
PAD = .010      # palm pad half thickness
PAD_TOTE = .025  # thicker pads for the tote
# (actor prefix, hand side) -> pad centre on the face (payload frame, x y) and the face's inward normal (x y).
TARGETS = {
    'crate_side': {('left_', 'left'): ((-.45, .32), (0, -1)), ('left_', 'right'): ((-.45, -.32), (0, 1)),
                   ('right_', 'left'): ((.45, -.32), (0, 1)), ('right_', 'right'): ((.45, .32), (0, -1))},
    # tote_side: her sofa_side seat only (the back is removed), held level by both palms on its side walls
    'tote_side': {('carrier_', 'left'): ((0., .215), (0, -1)), ('carrier_', 'right'): ((0., -.215), (0, 1))},
    'fixture_side': {('left_', 'left'): ((-.46, .33), (0, -1)), ('left_', 'right'): ((-.46, -.23), (0, 1)),
                     ('right_', 'left'): ((.46, -.23), (0, 1)), ('right_', 'right'): ((.46, .33), (0, -1)),
                     ('rear_', 'left'): ((-.26, -.58), (1, 0)), ('rear_', 'right'): ((.26, -.58), (-1, 0))},
}
_build_group = g1_logistics.build_group


def palm_normal(model, pad):
    """The hand mesh's thinnest axis (the palm normal) in its body frame."""
    b = model.geom_bodyid[pad]
    g = next(i for i in range(model.ngeom) if model.geom_bodyid[i] == b and model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH)
    r = np.zeros(9)
    mujoco.mju_quat2Mat(r, model.geom_quat[g])
    return r.reshape(3, 3)[:, int(np.argmin(model.geom_size[g]))]


def solve_arm(model, data, arm, pads, goals, iters=200, normals=None):
    """Damped least squares on one robot's ten arm joints so each pad centre reaches its goal, within joint limits.
    With normals, each hand's palm also turns flat to that world direction (sign free)."""
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    dofs = arm.joint_dof[1:]
    joints = [int(model.dof_jntid[i]) for i in dofs]
    lo, hi = model.jnt_range[joints, 0] + .02, model.jnt_range[joints, 1] - .02
    for _ in range(iters):
        mujoco.mj_forward(model, data)
        rows, errs = [], []
        for k, (g, goal) in enumerate(zip(pads, goals)):
            mujoco.mj_jacGeom(model, data, jacp, jacr, g)
            rows.append(jacp[:, dofs].copy())
            errs.append(goal - data.geom_xpos[g])
            if normals is not None:
                n = data.xmat[model.geom_bodyid[g]].reshape(3, 3) @ palm_normal(model, g)
                t = np.asarray(normals[k], float) * np.sign(n @ normals[k] or 1)
                rows.append(.3 * jacr[:, dofs].copy())
                errs.append(.3 * np.cross(n, t))
        J, e = np.vstack(rows), np.concatenate(errs)
        if np.abs(e).max() < 5e-4:
            break
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(len(e)), e)
        data.qpos[arm.joint_qpos[1:]] = np.clip(data.qpos[arm.joint_qpos[1:]] + np.clip(dq, -.05, .05), lo, hi)
    mujoco.mj_forward(model, data)
    return float(max(np.linalg.norm(goal - data.geom_xpos[g]) for g, goal in zip(pads, goals)))


def build_group(args, scene_hook=None):
    if args.scenario not in BASE:
        return _build_group(args, scene_hook)
    spec, model, data, legs, arms, actors, palms, cfg, half = _build_group(
        SimpleNamespace(scenario=BASE[args.scenario], mass=args.mass, carriers=getattr(args, 'carriers', 3)), scene_hook)
    p = model.jnt_qposadr[model.joint('payload_free').id]
    reach = REACH.get(args.scenario, 0.)
    data.qpos[p:p + 7] = (reach, 0, HEIGHT[args.scenario], 1, 0, 0, 0)
    if args.scenario == 'tote_side':  # a plain tote: the sofa back no longer exists for physics or mass
        back = model.geom('sofa_back').id
        model.geom_contype[back] = model.geom_conaffinity[back] = 0
        spec.geom('sofa_back').mass, spec.geom('sofa_back').group = 0, 3
        # Two point contacts on opposite walls act as a hinge and the tote rolled 90 deg about them. Flat palm pads
        # (face to face, several contact points) resist that roll, so only the pads touch the tote, and they are
        # thickened to stand proud of the hand mesh.
        for s_ in ('left', 'right'):
            model.geom_size[model.geom('carrier_' + s_ + '_palm_pad').id, 0] = PAD_TOTE
    data.qvel[:] = 0
    for g in ('load', 'fixture_stem'):
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, g)
        if i >= 0:
            model.geom_conaffinity[i] = 10 if args.scenario == 'tote_side' else 11  # 11: the whole robot collides
            # Her sofa_side contact settings: the load's friction wins, stiffer and with torsional friction,
            # so a sustained squeeze does not creep down through the soft contact.
            model.geom_priority[i], model.geom_solref[i], model.geom_condim[i] = 1, (.005, 1), 4
    loads = [i for i in range(model.ngeom) if model.geom_bodyid[i] == model.body('payload').id and model.geom_contype[i]]
    z, err = HEIGHT[args.scenario], 0.
    for (prefix, _, _), arm in zip(actors, arms):
        pads = [model.geom(prefix + s + '_palm_pad').id for s in ('left', 'right')]
        rows = [TARGETS[args.scenario][(prefix, s)] for s in ('left', 'right')]
        hands = [[i for i in range(model.ngeom) if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH
                  and model.body(int(model.geom_bodyid[i])).name == prefix + s + '_wrist_roll_rubber_hand'] + [pad]
                 for s, pad in zip(('left', 'right'), pads)]
        pad = PAD_TOTE if args.scenario == 'tote_side' else PAD
        gap = [pad + .001] * 2
        for _ in range(6):  # back each palm off until the hand mesh, not only the pad, clears the face
            goals = [np.r_[np.array(xy) + (reach, 0) - np.array(n) * g, z] for (xy, n), g in zip(rows, gap)]
            flat = [np.r_[n, 0.] for _, n in rows] if args.scenario == 'tote_side' else None
            err = max(err, solve_arm(model, data, arm, pads, goals, normals=flat))
            for g, (_, n) in zip(pads, rows):  # pad faces the load: thin local x along the normal, long side vertical
                n = np.r_[n, 0.]
                desired = np.column_stack([n, (0, 0, 1), np.cross(n, (0, 0, 1))])
                quat = np.zeros(4)
                mujoco.mju_mat2Quat(quat, (data.xmat[model.geom_bodyid[g]].reshape(3, 3).T @ desired).ravel())
                model.geom_quat[g] = quat
            mujoco.mj_forward(model, data)
            clear = [min(mujoco.mj_geomDistance(model, data, h, g, .3, None) for h in hs for g in loads) for hs in hands]
            if min(clear) > 0:
                break
            gap = [g + max(0., .001 - c) for g, c in zip(gap, clear)]
        touch = data.qpos[arm.joint_qpos].copy()
        err = max(err, solve_arm(model, data, arm, pads, [g + np.r_[n, 0.] * SQUEEZE for g, (_, n) in zip(goals, rows)], normals=flat))
        arm.target_q = data.qpos[arm.joint_qpos].astype(np.float32)
        data.qpos[arm.joint_qpos] = touch
    mujoco.mj_forward(model, data)
    build_group.ik_error = err
    for geom in spec.geoms:  # carry the pad turns and collision masks into the attached world, as her code does
        if geom.name:
            i = model.geom(geom.name).id
            geom.quat, geom.conaffinity = model.geom_quat[i].copy(), int(model.geom_conaffinity[i])
            geom.size = model.geom_size[i].copy()
            geom.priority, geom.solref, geom.condim = int(model.geom_priority[i]), model.geom_solref[i].copy(), int(model.geom_condim[i])
            geom.contype = int(model.geom_contype[i])
    if args.scenario == 'tote_side':  # recompile so the standalone model also drops the back's mass
        qpos = data.qpos.copy()
        model = spec.compile()
        data = mujoco.MjData(model)
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
    return spec, model, data, legs, arms, actors, palms, cfg, half


class SideCarry(carry_controller.CarryController):
    """Her controller, plus her sofa_side arm gravity compensation for the squeeze scenarios. The level tote
    also uses her sofa_side release (open the shoulder roll joints) instead of dropping the arms."""
    def after_step(self, data, step):
        if self.scenario == 'tote_side' and self.state == 'RELEASE':
            self.scenario = 'sofa_side'
            try:
                return super().after_step(data, step)
            finally:
                self.scenario = 'tote_side'
        return super().after_step(data, step)

    def apply(self, data):
        super().apply(data)
        if self.scenario in BASE:
            for i, arm in enumerate(self.arms):
                data.ctrl[arm.actuators] += data.qfrc_bias[self.dofs[i]]


def install():
    for name, base in BASE.items():
        carry_controller.PROFILES.setdefault(name, dict(carry_controller.PROFILES[base]))
    g1_logistics.build_group = industrial_showcase.build_group = build_group
    plant_view.CarryController = SideCarry


def probe(scenario, duration, profile):
    """Her carry_probe loop with the supported-fraction check from g1_logistics.run added."""
    install()
    seed = build_group(SimpleNamespace(scenario=scenario, mass=carry_controller.PROFILES[scenario]['mass'], carriers=3),
                       lambda spec: carry_controller.add_delivery_stage(spec, scenario, profile=profile))
    _, model, data, legs, arms, actors, _, cfg, _ = seed
    group = dict(scenario=scenario, legs=legs, arms=arms, actors=actors, cfg=cfg, body=model.body('payload').id)
    c = SideCarry(model, data, group, profile=profile)
    weight = c.profile['mass'] * 9.81
    supported = transit = 0
    low, deep, tilt = 9., {}, 0.
    for step in range(round(duration / model.opt.timestep)):
        c.apply(data)
        mujoco.mj_step(model, data)
        c.after_step(data, step)
        if c.state == 'TRANSIT':
            transit += 1
            supported += c.hand_force > .5 * weight
            low = min(low, float(data.xpos[c.load, 2]))
            tilt = max(tilt, float(np.degrees(np.arccos(np.clip(data.xmat[c.load][8], -1, 1)))))
        for con in data.contact:
            if con.dist < -.003 and c.load in (model.geom_bodyid[con.geom1], model.geom_bodyid[con.geom2]):
                o = con.geom2 if model.geom_bodyid[con.geom1] == c.load else con.geom1
                k = model.body(int(model.geom_bodyid[o])).name
                deep[k] = round(max(deep.get(k, 0), -float(con.dist)), 4)
    r = c.report(data)
    r.update(ik_error_m=getattr(build_group, 'ik_error', 0.), transit_supported_fraction=supported / max(1, transit),
             lowest_payload_in_transit_m=low, max_tilt_in_transit_deg=tilt, deep_contacts=deep,
             final_tilt_deg=float(np.degrees(np.arccos(np.clip(data.xmat[c.load][8], -1, 1)))))
    return r


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', choices=list(BASE) + ['sofa_side', 'bottom_crate', 'team_fixture'], default='crate_side')
    ap.add_argument('--duration', type=float, default=30)
    ap.add_argument('--layout', type=Path, default=Path(__file__).with_name('realistic_layout_v3.json'))
    ap.add_argument('--squeeze', type=float, default=SQUEEZE)
    ap.add_argument('--height', type=float, help='payload centre at reset')
    ap.add_argument('--top', type=float, help='placement surface height')
    ap.add_argument('--reach', type=float, help='payload x at reset (tote_side)')
    ap.add_argument('--table', nargs=4, type=float, help='placement surface centre x y and half sizes x y')
    a = ap.parse_args()
    SQUEEZE = a.squeeze
    if a.height:
        HEIGHT[a.scenario] = a.height
    if a.reach:
        REACH[a.scenario] = a.reach
    install()
    prof = json.loads(a.layout.read_text())['carry_profiles'].get(a.scenario) or dict(carry_controller.PROFILES[a.scenario])
    if a.table:
        prof.update(center=a.table[:2], size=a.table[2:])
    if a.top:
        prof['top'] = a.top
    r = probe(a.scenario, a.duration, prof)
    print(json.dumps({k: v for k, v in r.items() if k != 'events'}, indent=1))
    print([(round(e['t'], 2), e['event']) for e in r['events']])
