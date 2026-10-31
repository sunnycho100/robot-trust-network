# One H1 walking in a local scene file with the bundled policy. Usage: /isaac-sim/python.sh h1_in_scene.py SCENE.usd|.usdz [--pos X Y Z] [--yaw DEG] [--command VX VY WZ] [--stream]
"""Isaac Sim 6.2. Based on standalone_examples/api/isaacsim.robot.policy.examples/h1_standalone.py (v6.2.0).

Differences from the stock example:
- SCENE is a local file path. The stock --env-url is prefixed with the asset root, so it can't load a local file.
- The scene is referenced into the new stage, never opened as the root stage: in 6.2, opening a .usdz as the root
  stage makes references added afterwards (the robot) load empty.
- An invisible collision plane at z = 0 is added unless the scene already has collision prims.
- One robot, one fixed command (no scripted course, no keyboard). Ctrl+C to quit.
"""

import argparse
import math
import os
import sys

from isaacsim import SimulationApp

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("scene", help="local .usd or .usdz file")
parser.add_argument("--pos", type=float, nargs=3, default=[0.0, 0.0, 1.05], help="H1 spawn position in m (stock: z 1.05)")
parser.add_argument("--yaw", type=float, default=0.0, help="H1 spawn heading in degrees")
parser.add_argument("--command", type=float, nargs=3, default=[0.5, 0.0, 0.0], help="vx m/s, vy m/s, yaw rate rad/s")
parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda", help="Simulation device")
parser.add_argument("--stream", action="store_true", help="stream over WebRTC to the web viewer (inside the container)")
args, sys.argv[1:] = parser.parse_known_args()  # SimulationApp passes argv on to Kit: only what's left over

stream = []
if args.stream:  # the address and ports the container's runheadless.sh passes, from the compose environment
    ls = "--/exts/omni.kit.livestream.app/primaryStream/"
    stream = [
        "--enable",
        "omni.kit.livestream.app",
        f"{ls}publicIp={os.environ.get('ISAACSIM_HOST', '127.0.0.1')}",
        f"{ls}signalPort={os.environ.get('ISAACSIM_SIGNAL_PORT', '49100')}",
        f"{ls}streamPort={os.environ.get('ISAACSIM_STREAM_PORT', '47998')}",
        f"{ls}streamType=webrtc",
        "--/app/window/drawMouse=true",
    ]
# unverified: a standalone app with omni.kit.livestream.app is reachable from the compose web viewer the same way
# as the stock streaming app (6.2's livestream.py streams this way, but to the native client in its docs).
simulation_app = SimulationApp({"headless": args.stream, "hide_ui": False if args.stream else None, "extra_args": stream})

import carb
import isaacsim.core.experimental.utils.stage as stage_utils
import numpy as np
import omni.timeline
from isaacsim.core.rendering_manager import RenderingManager, ViewportManager
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.simulation_manager.impl.isaac_events import IsaacEvents
from isaacsim.robot.policy.examples import RobotPolicyRunner, get_h1_spec
from pxr import PhysicsSchemaTools, Usd, UsdGeom, UsdLux, UsdPhysics

first_step = True
reset_needed = False
policy_failed = False


def on_physics_step(step_size: float, context: object) -> None:
    """Initialize, reset, or advance the H1 policy runner after a physics step (as in the stock example)."""
    global first_step, reset_needed, policy_failed
    if policy_failed:
        return
    try:
        if first_step:
            h1.restart_from_default_state(command)
            first_step = False
        elif reset_needed:
            reset_needed = False
            first_step = True
        else:
            h1.step(step_size, command)
    except Exception as error:  # noqa: BLE001 - a physics callback must not raise
        policy_failed = True
        carb.log_error(f"h1_in_scene: policy deployment failed, stopping: {error}")


# spawn scene: a reference on the new stage that SimulationApp created
stage = stage_utils.get_current_stage(backend="usd")
scene = stage_utils.add_reference_to_stage(os.path.abspath(args.scene), "/World/Scene")
# unverified: whether the NuRec cafe .usdz has collision prims, and that its floor is at z = 0 with open floor at the origin
if any(p.HasAPI(UsdPhysics.CollisionAPI) for p in Usd.PrimRange(scene, Usd.TraverseInstanceProxies())):
    print("Scene has its own collision, no ground plane added")
else:
    # same call GroundPlane uses: an Xform with a collision Plane and a visual Mesh; hide the mesh
    PhysicsSchemaTools.addGroundPlane(stage, "/World/GroundCollision", "Z", 50.0, (0.0, 0.0, 0.0), (0.5, 0.5, 0.5))
    for child in stage.GetPrimAtPath("/World/GroundCollision").GetChildren():
        if child.IsA(UsdGeom.Mesh):
            UsdGeom.Imageable(child).MakeInvisible()
    print("Scene has no collision prims: added an invisible ground plane at z = 0")
UsdLux.DomeLight.Define(stage, "/World/DomeLight").CreateIntensityAttr(1000.0)  # the robot needs light in a splat scene

# spawn physics scene (as in the stock example)
stage_utils.define_prim("/World/PhysicsScene", "PhysicsScene")
RenderingManager.set_dt(8.0 / 200.0)
SimulationManager.set_physics_sim_device(args.device)
SimulationManager.set_physics_dt(1.0 / 200.0)

command = np.asarray(args.command, dtype=np.float32)
yaw = math.radians(args.yaw)
h1 = RobotPolicyRunner(
    get_h1_spec(), prim_path="/World/H1", position=args.pos, orientation=[math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
)
h1.spawn()
x, y, z = args.pos  # camera 2.5 m behind the robot, looking at it
ViewportManager.set_camera_view(
    ViewportManager.get_camera(), eye=[x - 2.5 * math.cos(yaw), y - 2.5 * math.sin(yaw), z + 0.6], target=[x, y, z - 0.3]
)

_physics_callback_id = SimulationManager.register_callback(on_physics_step, IsaacEvents.POST_PHYSICS_STEP)
timeline = omni.timeline.get_timeline_interface()
timeline.play()
simulation_app.update()

while simulation_app.is_running():
    simulation_app.update()
    if not SimulationManager.is_simulating():
        reset_needed = True
timeline.stop()
SimulationManager.deregister_callback(_physics_callback_id)
h1.close()
simulation_app.close()
