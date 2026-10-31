# Print the up axis and world bounding box of a USD scene. Usage (in the container): /isaac-sim/python.sh scene_bbox.py SCENE.usd
import sys
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
from pxr import Usd, UsdGeom
stage = Usd.Stage.Open(sys.argv[1])
print("UPAXIS", UsdGeom.GetStageUpAxis(stage), "MPU", UsdGeom.GetStageMetersPerUnit(stage))
cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy"])
for p in ["/World", "/World/Scene"]:
    prim = stage.GetPrimAtPath(p)
    if prim:
        r = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        print("BBOX", p, [round(v, 2) for v in r.GetMin()], [round(v, 2) for v in r.GetMax()])
for prim in stage.Traverse():
    if prim.GetName() in ("floor", "Floor") or "floor" in prim.GetPath().pathString.lower():
        r = cache.ComputeWorldBound(prim).ComputeAlignedRange(); print("FLOOR", prim.GetPath(), [round(v, 2) for v in r.GetMin()], [round(v, 2) for v in r.GetMax()]); break
app.close()
