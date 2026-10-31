# Lists the stock assets the CarDA line needs and prints the UR10e joint tree and a character's skeleton.
# Usage: /content/isim/bin/python probe_assets.py
from isaacsim import SimulationApp

app = SimulationApp({"headless": True})
import omni.client
import omni.usd
from isaacsim.storage.native import get_assets_root_path
from pxr import Usd, UsdPhysics, UsdSkel, UsdGeom

root = get_assets_root_path()
print("ROOT", root)


FOUND = []


def ls(rel, depth=1, pre=""):
    r, entries = omni.client.list(root + rel)
    if r != omni.client.Result.OK:
        print("LS FAIL", rel, r)
        return
    for e in entries:
        print("LS", pre + rel + "/" + e.relative_path)
        FOUND.append(rel + "/" + e.relative_path)
        if depth > 1 and not e.relative_path.endswith((".usd", ".usda", ".usdz", ".png", ".jpg", ".mdl")):
            ls(rel + "/" + e.relative_path, depth - 1, pre)


for d, n in [("/Isaac/Robots/UniversalRobots", 2), ("/Isaac/Robots/Idealworks", 2), ("/Isaac/Robots/NVIDIA", 3),
             ("/Isaac/People/Characters", 2), ("/Isaac/People", 1), ("/Isaac/Environments/Simple_Warehouse", 1),
             ("/Isaac/Robots/Doosan", 2), ("/Isaac/Environments", 1), ("/Isaac/Props", 1)]:
    ls(d, n)

ctx = omni.usd.get_context()
stage = ctx.get_stage()


def ref(path, url):
    p = stage.DefinePrim(path, "Xform")
    p.GetReferences().AddReference(url)
    return p


for _ in range(5):
    app.update()
for url in [root + "/Isaac/Robots/UniversalRobots/ur10e/ur10e.usd"]:
    p = ref("/World/ur", url)
    for _ in range(20):
        app.update()
    for q in Usd.PrimRange(p):
        depth = str(q.GetPath()).count("/")
        if depth <= 5:
            print("PRIM", q.GetPath(), q.GetTypeName(), q.GetAppliedSchemas()[:4], "inst" if q.IsInstance() else "")
        if q.IsA(UsdPhysics.Joint):
            j = UsdPhysics.Joint(q)
            ax = q.GetAttribute("physics:axis").Get()
            print("JOINT", q.GetPath(), q.GetTypeName(), j.GetBody0Rel().GetTargets(), j.GetBody1Rel().GetTargets(),
                  j.GetLocalPos0Attr().Get(), j.GetLocalRot0Attr().Get(), j.GetLocalPos1Attr().Get(), j.GetLocalRot1Attr().Get(),
                  ax, q.GetAttribute("physics:lowerLimit").Get(), q.GetAttribute("physics:upperLimit").Get())
chars = [f for f in FOUND if "/Characters/" in f and f.endswith(".usd")]
chars.sort(key=lambda f: ("construction" not in f.lower(), f))
print("CHARS", chars[:12])
if chars:
    c = ref("/World/person", root + chars[0])
    for _ in range(20):
        app.update()
    for q in Usd.PrimRange(c):
        if q.IsA(UsdSkel.Skeleton):
            sk = UsdSkel.Skeleton(q)
            print("SKEL", q.GetPath(), list(sk.GetJointsAttr().Get())[:80])
            print("REST0", list(sk.GetRestTransformsAttr().Get())[:3])
            print("ANIMSRC", UsdSkel.BindingAPI(q).GetAnimationSourceRel().GetTargets())
        if q.GetTypeName() in ("SkelRoot", "SkelAnimation") or "AnimationGraph" in str(q.GetAppliedSchemas()):
            print("SKPRIM", q.GetPath(), q.GetTypeName(), q.GetAppliedSchemas())
    bb = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeWorldBound(c).ComputeAlignedRange()
    print("CHAR BBOX", bb)
w = ref("/World/wh", root + "/Isaac/Environments/Simple_Warehouse/warehouse.usd")
for _ in range(20):
    app.update()
print("WH BBOX", UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeWorldBound(w).ComputeAlignedRange())
app.close()
