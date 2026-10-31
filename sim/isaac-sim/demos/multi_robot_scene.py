# Several robots, each with its stock walking policy, in one scene. Usage:
#   /isaac-sim/python.sh multi_robot_scene.py [--scene SCENE.usd] [--robot KIND X Y YAW_DEG VX VY WZ]... [--goal X Y]... [--stream]
# KIND is h1, spot, go2 or anymal. Without --robot: two H1s, a Spot, a Go2 and an ANYmal walking circles in the full
# warehouse. Without --scene: the stock full warehouse from the asset root. Record headless with record_demo.py.
# With --goal (one per robot or more): fleet.py assigns the goals and drives each robot there instead of its fixed
# command; the longest trip has right of way. Prints when each robot arrives and the closest approach between two robots.
# With --stop ROBOT X Y DWELL (job mode, instead of --goal): robot ROBOT (index) visits its stops in order and waits
# DWELL s at each; a station holds one robot at a time (fleet.Jobs), so a robot waits until the one before it has left.
# With --plan (goal mode): navmap.py builds an occupancy grid from the scene's meshes and each robot follows an A* path
# around obstacles to its goal instead of a straight line.
"""Isaac Sim 6.2. Built from the stock policy examples (isaacsim.robot.policy.examples, v6.2.0): one
RobotPolicyRunner per robot, stepped from one physics callback.

The robots were trained at different physics rates, but one stage has one physics dt. The scene runs at the
greatest common divisor of their dts and steps each runner every robot_dt / scene_dt physics ticks, so each policy still sees its
own training clock (it counts its own ticks for decimation). The PD gains then act at the finer scene dt, which is
not how they were trained. unverified: whether that changes gaits visibly; with only H1, Go2 and ANYmal (all 200 Hz
in Isaac Lab) the ratio is 1 and nothing changes. Verified on Colab: Spot trains at 2 ms, H1 at 5 ms.
"""

import argparse
import math
import os
import sys

from isaacsim import SimulationApp

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--scene", help="local .usd or .usdz; default: the stock full warehouse")
parser.add_argument("--robot", nargs=7, action="append", metavar=("KIND", "X", "Y", "YAW", "VX", "VY", "WZ"))
parser.add_argument("--goal", nargs=2, type=float, action="append", metavar=("X", "Y"))
parser.add_argument("--stop", nargs=4, type=float, action="append", metavar=("ROBOT", "X", "Y", "DWELL"))
parser.add_argument("--plan", action="store_true", help="with --goal: A* paths around the scene's obstacles")
parser.add_argument("--eye", nargs=3, type=float, metavar=("X", "Y", "Z"), help="camera position; default southwest of the robots")
parser.add_argument("--target", nargs=3, type=float, metavar=("X", "Y", "Z"), help="camera look-at; default the robots' centre at 0.3 m")
parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda")
parser.add_argument("--stream", action="store_true", help="stream over WebRTC to the web viewer (inside the container)")
args, sys.argv[1:] = parser.parse_known_args()

# circles of radius v / w, centres far enough apart that they never cross
DEFAULT = [
    ["h1", 0, 0, 0, 0.5, 0, 0.4],
    ["h1", 0, 3.5, 0, 0.5, 0, -0.4],
    ["spot", 4, -1, 90, 1.0, 0, 0.5],
    ["go2", 4, 4, 0, 0.6, 0, 0.6],
    ["anymal", -4, 2, 0, 0.6, 0, -0.5],
]
robots_cfg = [[r[0]] + [float(v) for v in r[1:]] for r in (args.robot or DEFAULT)]

stream = []
if args.stream:  # same flags as h1_in_scene.py
    ls = "--/exts/omni.kit.livestream.app/primaryStream/"
    stream = ["--enable", "omni.kit.livestream.app", f"{ls}publicIp={os.environ.get('ISAACSIM_HOST', '127.0.0.1')}",
              f"{ls}signalPort={os.environ.get('ISAACSIM_SIGNAL_PORT', '49100')}",
              f"{ls}streamPort={os.environ.get('ISAACSIM_STREAM_PORT', '47998')}", f"{ls}streamType=webrtc",
              "--/app/window/drawMouse=true"]
simulation_app = SimulationApp({"headless": args.stream, "hide_ui": False if args.stream else None, "extra_args": stream})

import carb
import isaacsim.core.experimental.utils.stage as stage_utils
import numpy as np
import omni.timeline
from isaacsim.core.rendering_manager import RenderingManager, ViewportManager
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.simulation_manager.impl.isaac_events import IsaacEvents
from isaacsim.robot.policy.examples import (PolicyEnvConfig, RobotPolicyRunner, get_anymal_spec, get_go2_spec,
                                            get_h1_spec, get_spot_spec)
from isaacsim.storage.native import get_assets_root_path

SPECS = {"h1": get_h1_spec, "spot": get_spot_spec, "go2": get_go2_spec, "anymal": get_anymal_spec}
# ponytail: speeds inside the stock command ranges as far as known (unverified per robot); tune if gaits stumble
VMAX = {"h1": 0.6, "spot": 1.0, "go2": 0.8, "anymal": 0.8}
SPAWN_Z = {"h1": 1.05, "go2": 0.5, "anymal": 0.7}  # from the stock examples; Spot uses its env config's pose

stage = stage_utils.get_current_stage(backend="usd")
scene = args.scene and os.path.abspath(args.scene) or get_assets_root_path() + "/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"
stage_utils.add_reference_to_stage(scene, "/World/Scene")
if args.scene:  # a splat is self-lit; without a light the robots render black
    from pxr import UsdLux
    UsdLux.DomeLight.Define(stage, "/World/DomeLight").CreateIntensityAttr(1000.0)
# unverified for local scenes: they need collision at the floor; h1_in_scene.py shows how to add a ground plane
stage_utils.define_prim("/World/PhysicsScene", "PhysicsScene")
SimulationManager.set_physics_sim_device(args.device)

runners, commands = [], []
for i, (kind, x, y, yaw_deg, vx, vy, wz) in enumerate(robots_cfg):
    spec = SPECS[kind]()
    z = SPAWN_Z.get(kind)
    if z is None:
        z = (PolicyEnvConfig.from_file(spec.engines["physx"].env_config_path).initial_root_pose[0] or [0, 0, 0.7])[2]
    yaw = math.radians(yaw_deg)
    r = RobotPolicyRunner(spec, prim_path=f"/World/{kind}_{i}", position=[x, y, z],
                          orientation=[math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
    r.spawn()
    runners.append(r)
    commands.append(np.asarray([vx, vy, wz], dtype=np.float32))

dts = [r.physics_dt for r in runners]
scene_dt = math.gcd(*[round(dt * 1e6) for dt in dts]) / 1e6  # Spot 2 ms and H1 5 ms -> 1 ms
every = [round(dt / scene_dt) for dt in dts]
for r, dt, k in zip(robots_cfg, dts, every):
    assert abs(k * scene_dt - dt) < 1e-9, f"{r[0]} physics dt {dt} is not a multiple of the scene dt {scene_dt}"
print("[multi] scene dt %.4f s; robots: %s" % (scene_dt, ", ".join(f"{c[0]} every {k}" for c, k in zip(robots_cfg, every))))
SimulationManager.set_physics_dt(scene_dt)
RenderingManager.set_dt(0.04)  # 25 fps render, as the stock H1 example

marks = [g[:2] for g in args.goal or []] + [s[1:3] for s in args.stop or []]
xs, ys = [c[1] for c in robots_cfg] + [g[0] for g in marks], [c[2] for c in robots_cfg] + [g[1] for g in marks]
cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
span = max(max(xs) - min(xs), max(ys) - min(ys), 4.0)
# eye at 0.5 span stays below the warehouse lights, which hid robots at 0.9
ViewportManager.set_camera_view(ViewportManager.get_camera(), eye=args.eye or [cx - 1.2 * span, cy - 1.2 * span, 0.5 * span],
                                target=args.target or [cx, cy, 0.3])

tick, first_step, reset_needed, policy_failed = 0, True, False, False
FLEET = bool(args.goal or args.stop)
if FLEET:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from fleet import Jobs, assign, check_goals, command, priority, smooth, yaw_from_wxyz

    if args.stop:  # job mode: each route starts where the robot stands; robot index is its priority
        routes = [[(c[1], c[2], 0.0)] for c in robots_cfg]
        for r_i, x, y, dwell in args.stop:
            routes[int(r_i)].append((x, y, dwell))
        jobs, rank, leg = Jobs(routes), list(range(len(robots_cfg))), [0] * len(robots_cfg)
        print("[fleet] routes: " + "; ".join(f"{c[0]}_{i} {r[1:]}" for i, (c, r) in enumerate(zip(robots_cfg, routes))))
    else:
        assert len(args.goal) >= len(robots_cfg), "give at least one --goal per robot"
        check_goals(args.goal)
        goal_of = assign([c[1:3] for c in robots_cfg], args.goal)
        rank = priority([c[1:3] for c in robots_cfg], [args.goal[g] for g in goal_of])
        print("[fleet] goals: " + ", ".join(f"{c[0]}_{i} -> {args.goal[g]}" for i, (c, g) in enumerate(zip(robots_cfg, goal_of))))
        paths = [[args.goal[g]] for g in goal_of]
        if args.plan:
            from navmap import NavMap, stage_points

            for _ in range(30):  # let the scene's references load before reading its meshes
                simulation_app.update()
            # stock scenes have the floor at z = 0; reconstructions get it estimated
            # one map per robot type, inflated by its body radius plus 0.2 m because following cuts corners (test_navmap.py)
            # ponytail: radii are half the body length from spec sheets, approximate; centre distance is still used between robots
            pts = stage_points(stage.GetPrimAtPath("/World/Scene"))
            BODY = {"h1": 0.3, "go2": 0.35, "anymal": 0.5, "spot": 0.55}
            nav = {k: NavMap(pts, radius=BODY[k] + 0.2, floor_z=None if args.scene else 0.0) for k in {c[0] for c in robots_cfg}}
            n0 = nav[robots_cfg[0][0]]
            np.savez("navmap.npz", blocked=n0.blocked, lo=n0.lo, res=n0.res, points=pts[::max(1, len(pts) // 200000)])
            print(f"[fleet] navmap {n0.shape} cells at {n0.res} m, floor z {n0.floor_z:.2f}, blocked: "
                  + ", ".join(f"{k} {m.blocked.mean():.0%}" for k, m in nav.items()))
            for i, (c, g) in enumerate(zip(robots_cfg, goal_of)):
                p = nav[c[0]].path(c[1:3], args.goal[g])
                print(f"[fleet] {c[0]}_{i} path " + (f"{[(round(x, 1), round(y, 1)) for x, y in p]}" if p else "NONE, goes straight"))
                paths[i] = p[1:] if p else paths[i]
        wp = [0] * len(robots_cfg)  # waypoint index per robot
    arrived, min_gap = [None] * len(runners), float("inf")
    fleet_every = max(1, round(0.02 / scene_dt))  # replan at 50 Hz


def fleet_update() -> None:
    """New velocity commands from the robots' root poses (written in place, so runner.step sees them)."""
    global min_gap
    poses, zs = [], []
    for r in runners:
        p, q = r.articulation.get_world_poses()
        p, q = (np.asarray(v.numpy() if hasattr(v, "numpy") else v).reshape(-1) for v in (p, q))
        poses.append((float(p[0]), float(p[1]), yaw_from_wxyz(q[:4])))
        zs.append(round(float(p[2]), 2))  # base height, to spot falls
    if not args.stop:  # next waypoint once within 0.5 m of the current one
        for i, q in enumerate(poses):
            if wp[i] < len(paths[i]) - 1 and math.dist(q[:2], paths[i][wp[i]]) < 0.5:
                wp[i] += 1
    goals = [jobs.goal(i, q, tick * scene_dt) if args.stop else paths[i][wp[i]] for i, q in enumerate(poses)]
    for i, (c, pose, goal) in enumerate(zip(robots_cfg, poses, goals)):
        if args.stop and jobs.idx[i] != leg[i]:
            leg[i] = jobs.idx[i]
            print(f"[fleet] {c[0]}_{i} leaves for stop {leg[i]} {tuple(goal)} at {tick * scene_dt:.1f} s")
        commands[i][:] = smooth(commands[i], command(pose, goal, [q[:2] for j, q in enumerate(poses) if rank[j] < rank[i]],
                                                     VMAX[c[0]], [q[:2] for j, q in enumerate(poses) if rank[j] > rank[i]],
                                                     holonomic=c[0] != "h1"))
        if arrived[i] is None and (jobs.done(i) if args.stop else wp[i] == len(paths[i]) - 1 and math.dist(pose[:2], goal) < 0.35):
            arrived[i] = tick * scene_dt
            print(f"[fleet] {c[0]}_{i} reached {goal} at {arrived[i]:.1f} s")
        for j in range(i):
            min_gap = min(min_gap, math.dist(pose[:2], poses[j][:2]))
    if tick % round(5 / scene_dt) == 0:  # recordings kill the app, so report as it goes
        left = [round(math.dist(q[:2], g), 2) for q, g in zip(poses, goals)]
        print(f"[fleet] t {tick * scene_dt:.0f} s, arrived {sum(x is not None for x in arrived)} of {len(arrived)}, "
              f"distance to goal now {left}, base z {zs}, "
              f"closest approach {min_gap:.2f} m (centre to centre)")


def on_physics_step(step_size: float, context: object) -> None:
    """Restart every runner on the first step, then step each on its own clock (as the stock examples do)."""
    global tick, first_step, reset_needed, policy_failed
    if policy_failed:
        return
    try:
        if first_step:
            for r, c in zip(runners, commands):
                r.restart_from_default_state(c)
            first_step, tick = False, 0
        elif reset_needed:
            reset_needed, first_step = False, True
        else:
            if FLEET and tick % fleet_every == 0:
                fleet_update()
            for r, c, k in zip(runners, commands, every):
                if tick % k == 0:
                    r.step(r.physics_dt, c)
            tick += 1
    except Exception as error:  # noqa: BLE001 - a physics callback must not raise
        policy_failed = True
        carb.log_error(f"multi_robot_scene: policy deployment failed, stopping: {error}")


_cb = SimulationManager.register_callback(on_physics_step, IsaacEvents.POST_PHYSICS_STEP)
timeline = omni.timeline.get_timeline_interface()
timeline.play()
simulation_app.update()
while simulation_app.is_running():
    simulation_app.update()
    if not SimulationManager.is_simulating():
        reset_needed = True
timeline.stop()
if FLEET:
    print(f"[fleet] arrived {sum(a is not None for a in arrived)} of {len(arrived)}, closest approach {min_gap:.2f} m")
SimulationManager.deregister_callback(_cb)
for r in runners:
    r.close()
simulation_app.close()
