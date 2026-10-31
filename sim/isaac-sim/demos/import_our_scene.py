# Our reconstructed scene (export_scene.py output) to one USD for h1_in_scene.py. Usage: /isaac-sim/python.sh import_our_scene.py EXPORT_DIR
"""Isaac Sim 6.2, headless. Writes EXPORT_DIR/our_scene.usd:

    /World/Scene   reference to scene.urdf converted by the 6.2 URDF importer, base fixed to the world
                   (textured scan as the visual; floor, wall boxes and convex hulls as collisions)
    /World/Ground  ground.obj as a static triangle-mesh collider, when the export has one (outdoor scenes)

The importer writes its own package to EXPORT_DIR/<robot name>/, so EXPORT_DIR must be writable
(isaac_up.sh gives the container user write access to ~/robot).
"""

import argparse
import os
import sys

from isaacsim import SimulationApp

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("export_dir", help="folder with scene.urdf, its meshes, and ground.obj for outdoor scenes")
args, sys.argv[1:] = parser.parse_known_args()  # SimulationApp passes argv on to Kit: only what's left over
export_dir = os.path.abspath(args.export_dir)

simulation_app = SimulationApp({"headless": True})

import omni.kit.app

# enabled first, as in standalone_examples/api/isaacsim.asset.importer.urdf/urdf_import.py (v6.2.0)
for ext in ("omni.scene.optimizer.core", "isaacsim.robot.schema"):
    omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate(ext, True)

import isaacsim.core.experimental.utils.physics as physics_utils
import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.asset.importer.urdf.impl import URDFImporter, URDFImporterConfig
from pxr import UsdGeom


def read_obj(path):
    """Points, face sizes and 0-based indices from an OBJ's v and f lines (export_scene.py writes nothing else)."""
    points, counts, indices = [], [], []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                points.append(tuple(float(x) for x in line.split()[1:4]))
            elif line.startswith("f "):
                face = [int(t.split("/")[0]) - 1 for t in line.split()[1:]]
                counts.append(len(face))
                indices.extend(face)
    return points, counts, indices


# The importer converts to a file and leaves that file open as the current stage; it returns the .usda path.
# unverified: how the bundled urdf-usd-converter turns the visual's .mtl texture and the hull meshes into USD
# (its source is not in the IsaacSim repo); the hulls are already convex, so a convex-hull approximation is exact.
# unverified: whether the single-link, joint-free URDF comes out as a rigid body that fix_base anchors, or as plain
# static colliders (apply_fix_base then only logs a warning); either way the scene stays put.
config = URDFImporterConfig(urdf_path=os.path.join(export_dir, "scene.urdf"), usd_path=export_dir, fix_base=True)
scene_usd = URDFImporter(config).import_urdf()
print(f"URDF imported: {scene_usd}")

stage_utils.create_new_stage()
stage = stage_utils.get_current_stage(backend="usd")
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
stage.SetDefaultPrim(stage_utils.define_prim("/World"))
stage_utils.add_reference_to_stage(scene_usd, "/World/Scene")

ground_obj = os.path.join(export_dir, "ground.obj")
if os.path.exists(ground_obj):
    points, counts, indices = read_obj(ground_obj)
    mesh = UsdGeom.Mesh.Define(stage, "/World/Ground")
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    physics_utils.apply_collision(mesh.GetPrim(), approximation="none")  # "none" = triangle mesh, static only
    print(f"ground.obj: {len(points)} vertices, {len(counts)} faces, triangle-mesh collision")
else:
    print("no ground.obj (indoor export): the floor is a box in scene.urdf")

out = os.path.join(export_dir, "our_scene.usd")
if not stage_utils.save_stage(out):
    raise SystemExit(f"could not save {out}")
print(f"wrote {out}")
simulation_app.close()
