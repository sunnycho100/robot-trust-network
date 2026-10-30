"""Contact sheet of a built scene: top view, oblique view, the collision proxies, and the CooHOI layout.

    MUJOCO_GL=egl python render_preview.py SCENE_DIR OUT.jpg
"""
import json, sys
from pathlib import Path
import mujoco
import numpy as np
from PIL import Image


def shots(xml, views, show_proxies=False, size=(480, 640)):
    m = mujoco.MjModel.from_xml_path(str(xml))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, *size)
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = show_proxies
    out = []
    for lookat, az, el, dist in views:
        cam = mujoco.MjvCamera()
        cam.lookat[:], cam.azimuth, cam.elevation, cam.distance = lookat, az, el, dist
        r.update_scene(d, cam, opt)
        out.append(r.render())
    return out


def main():
    scene, out = Path(sys.argv[1]), sys.argv[2]
    info = json.loads((scene / "scene_info.json").read_text())
    sx, sy = info["size_m"][:2]
    spawn = [*info["spawn_xy"], info.get("spawn_z", 0.0) + 0.8]
    wide = max(sx, sy) * 0.9
    images = shots(scene / "scene.xml", [(spawn, 90, -89, wide), (spawn, 135, -30, 8)])
    images += shots(scene / "scene.xml", [(spawn, 135, -30, 8)], show_proxies=True)
    task = scene / "export" / "carry_task.json"
    if task.exists():
        box = json.loads(task.read_text())["box"]["pos"]
        images += shots(scene / "export" / "preview.xml", [([box[0], box[1], box[2] + 0.5], 45, -25, 6)])
    while len(images) % 2:
        images.append(np.zeros_like(images[0]))
    Image.fromarray(np.vstack([np.hstack(images[i:i + 2]) for i in range(0, len(images), 2)])).save(out, quality=85)
    print("wrote", out)


if __name__ == "__main__":
    main()
