"""Run Seohee's factory (plant_view) unchanged, headless, and record every body pose at 30 fps.
Usage: PYTHONPATH=<g1_cooperative_task>/sim/mujoco_houses/src python record_mujoco.py OUT_DIR [--duration 40]
       [--layout realistic_layout_v3.json]   (a variant layout; also enables the side_grip scenarios)
Writes OUT_DIR/plant.mjb (compiled model, meshes and textures inside), plant.xml (spec.to_xml),
poses.npz (xpos, xquat, geom rgba per frame) and report.json (their own physics report).
"""
import argparse
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
from unitree_demo.flexible_factory import plant_scene, plant_view

p = argparse.ArgumentParser()
p.add_argument('out', type=Path)
p.add_argument('--duration', type=float, default=40.)
p.add_argument('--layout', type=Path, help='variant of factory/realistic_layout.json')
a = p.parse_args()
if a.layout:
    import side_grip
    side_grip.install()
    plant_scene.LAYOUT = a.layout.resolve()
a.out.mkdir(parents=True, exist_ok=True)
rec = dict(xpos=[], xquat=[], rgba=[], t=[])
spec_box = {}

# Keep the composed spec so it can be written out as one MJCF.
_build_world = plant_scene.build_world
def build_world(hook=None, layout=None):
    def keep(spec):
        hook(spec)
        spec_box['spec'] = spec
    return _build_world(keep, layout=layout)
plant_scene.build_world = build_world


class PoseRecorder:
    """Stands in for mujoco.Renderer: plant_view calls update_scene once per 30 fps frame."""
    def __init__(self, model, height=0, width=0):
        self.model = model
        mujoco.mj_saveModel(model, str(a.out / 'plant.mjb'), None)
        self.t0 = time.time()

    def update_scene(self, data, camera=None):
        rec['xpos'].append(data.xpos.astype(np.float32))
        rec['xquat'].append(data.xquat.astype(np.float32))
        rec['rgba'].append((self.model.geom_rgba * 255).round().astype(np.uint8))
        rec['t'].append(data.time)
        if len(rec['t']) % 150 == 0:
            print(f'sim {data.time:.1f}s, wall {time.time() - self.t0:.0f}s', flush=True)

    def render(self): return np.zeros((1, 1, 3), np.uint8)
    def close(self): pass


class NoWriter:
    def __init__(self, *_, **__): pass
    def write(self, _): pass
    def close(self): pass


mujoco.Renderer = PoseRecorder
plant_view.RawVideoWriter = NoWriter
plant_view.draw_factory_hud = lambda *_, **__: None
args = argparse.Namespace(duration=a.duration, view='overview', labels=False, diagnostics=False,
                          video=a.out / 'unused.mp4', report=a.out / 'report.json')
plant_view.run(args)

model = mujoco.MjModel.from_binary_path(str(a.out / 'plant.mjb'))
names = [model.body(i).name for i in range(model.nbody)]
np.savez_compressed(a.out / 'poses.npz', xpos=np.stack(rec['xpos']), xquat=np.stack(rec['xquat']),
                    rgba=np.stack(rec['rgba']), t=np.array(rec['t']), body_names=np.array(names))
try:
    (a.out / 'plant.xml').write_text(spec_box['spec'].to_xml())
except Exception as e:  # ponytail: the mjb is the source of truth; the XML is a convenience
    print('to_xml failed:', e, file=sys.stderr)
print(f'saved {len(rec["t"])} frames, {model.nbody} bodies, {model.ngeom} geoms to {a.out}')
