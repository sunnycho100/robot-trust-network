"""Export RoboCasa kitchens as plain MJCF for the two-house demo.

Run with the ``robocasa`` env (MuJoCo 3.3.x). The output loads in the demo env
with ``mujoco.MjSpec.from_file`` and has no robot in it. Mesh and texture paths
are absolute and point into that env's robocasa package.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


HOUSES = {"house-1": 1, "house-2": 2}


def _fixture_info(name, fixture):
    info = {"name": name, "class": type(fixture).__name__}
    for attribute in ("pos", "size"):
        value = getattr(fixture, attribute, None)
        if value is not None:
            info[attribute] = [round(float(v), 4) for v in np.asarray(value).ravel()]
    return info


def _without_sampled_props(node):
    """Countertop props are placed by a sampler in the full Kitchen env and their
    meshes come from the object packs this demo does not download."""
    if isinstance(node, dict):
        return {key: _without_sampled_props(value) for key, value in node.items()}
    if isinstance(node, list):
        return [
            _without_sampled_props(item) for item in node
            if not (isinstance(item, dict) and "placement" in item)
        ]
    return node


def export_house(house: str, layout_id: int, style_id: int, out_dir: Path) -> dict:
    import yaml
    from robocasa.models.scenes import KitchenArena
    from robocasa.models.scenes.scene_registry import get_layout_path
    from robosuite.models.tasks import ManipulationTask

    layout = yaml.safe_load(Path(get_layout_path(layout_id=layout_id)).read_text())
    arena = KitchenArena(layout_id=_without_sampled_props(layout), style_id=style_id, clutter_mode=0)
    fixtures = dict(arena.fixtures)
    task = ManipulationTask(mujoco_arena=arena, mujoco_robots=[], mujoco_objects=list(fixtures.values()))
    xml_path = out_dir / f"{house}.xml"
    task.save_model(str(xml_path))
    manifest = {
        "house": house,
        "layout_id": layout_id,
        "style_id": style_id,
        "xml": xml_path.name,
        "fixtures": [_fixture_info(name, fixture) for name, fixture in fixtures.items()],
    }
    (out_dir / f"{house}.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--layout", type=int, default=1)
    parser.add_argument("--styles", type=int, nargs=2, default=list(HOUSES.values()),
                        metavar=("HOUSE1", "HOUSE2"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for house, style_id in zip(HOUSES, args.styles):
        manifest = export_house(house, args.layout, style_id, args.out)
        print(f"{house}: layout {manifest['layout_id']} style {manifest['style_id']}", flush=True)


if __name__ == "__main__":
    main()
