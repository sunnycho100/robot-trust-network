import mujoco
from unitree_demo.flexible_factory.config import load_config
from unitree_demo.flexible_factory.scene import add_labels

def test_eight_section_labels_without_a_graphics_context():
    model=mujoco.MjModel.from_xml_string('<mujoco><worldbody/></mujoco>')
    scene=mujoco.MjvScene(model,maxgeom=32)
    add_labels(scene,load_config())
    assert scene.ngeom==8
    assert any('MACHINING' in scene.geoms[i].label for i in range(scene.ngeom))
