# Several robots, each with its stock walking policy, in one scene. Usage:
#   /isaac-sim/python.sh multi_robot_scene.py [--scene SCENE.usd] [--robot KIND X Y YAW_DEG VX VY WZ]... [--stream]
# KIND is h1, spot, go2 or anymal. Without --robot: two H1s, a Spot, a Go2 and an ANYmal walking circles in the full
# warehouse. Without --scene: the stock full warehouse from the asset root. Record headless with record_demo.py.
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
SPAWN_Z = {"h1": 1.05, "go2": 0.5, "anymal": 0.7}  # from the stock examples; Spot uses its env config's pose

stage = stage_utils.get_current_stage(backend="usd")
scene = args.scene and os.path.abspath(args.scene) or get_assets_root_path() + "/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"
stage_utils.add_reference_to_stage(scene, "/World/Scene")
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

xs, ys = [c[1] for c in robots_cfg], [c[2] for c in robots_cfg]
cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
span = max(max(xs) - min(xs), max(ys) - min(ys), 4.0)
ViewportManager.set_camera_view(ViewportManager.get_camera(), eye=[cx - 1.2 * span, cy - 1.2 * span, 0.9 * span],
                                target=[cx, cy, 0.3])

tick, first_step, reset_needed, policy_failed = 0, True, False, False


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
SimulationManager.deregister_callback(_cb)
for r in runners:
    r.close()
simulation_app.close()
