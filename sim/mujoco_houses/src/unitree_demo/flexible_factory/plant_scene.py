"""Industrial-scale equipment around the existing models and carry controllers."""
import json
from pathlib import Path
import mujoco
from ..industrial_showcase import box,build_world
from .carry_controller import add_delivery_stage
from .plant_props import populate
from ..factory_scene import UPSTREAM

FIXED_CELLS=[
    ('cnc_arm','ur5e',(-5.5,3.45,.65),0),
    ('deburr_arm','ur5e',(-5.0,-4.1,.60),1.57079633),
    ('assembly_arm','panda',(1.5,-5.15,.65),1.57079633),
    ('packaging_arm','ur10e',(9.,3.55,.65),1.57079633),
]

def add_fixed_cells(spec):
    import math
    for name,model,pos,yaw in FIXED_CELLS:
        robot=mujoco.MjSpec.from_file(str(UPSTREAM/'roqsim_manipulation_assets/src/roqsim_manipulation_assets/models'/model/(model+'.xml')))
        for key in list(robot.keys): robot.delete(key)
        if model.startswith('ur'):
            grip=mujoco.MjSpec.from_file(str(UPSTREAM/'roqsim_manipulation_assets/src/roqsim_manipulation_assets/models/robotiq_2f85/robotiq_2f85.xml'))
            for key in list(grip.keys): grip.delete(key)
            robot.attach(grip,prefix='grip_',site=robot.site('tool_site') or robot.site('attachment_site'))
        spec.attach(robot,prefix=name+'_',frame=spec.worldbody.add_frame(pos=pos,quat=(math.cos(yaw/2),0,0,math.sin(yaw/2))))
        box(spec,name+'_pedestal',(pos[0],pos[1],pos[2]/2),(.16,.16,pos[2]/2),(.18,.25,.29,1))

LAYOUT=Path(__file__).resolve().parents[3]/'factory/realistic_layout.json'

def load_layout(): return json.loads(LAYOUT.read_text())

def visual_box(spec,name,pos,size,color):
    geom=box(spec,name,pos,size,color); geom.contype=geom.conaffinity=0; return geom

def worktable(spec,name,center,width=1.2,depth=.8,height=.85,color=(.35,.4,.43,1)):
    x,y=center
    box(spec,name+'_top',(x,y,height-.04),(width/2,depth/2,.04),color)
    for a in (-1,1):
        for b in (-1,1):
            box(spec,f'{name}_leg{a}_{b}',(x+a*(width/2-.07),y+b*(depth/2-.07),(height-.08)/2),(.04,.04,(height-.08)/2),(.18,.23,.27,1))

def outline(spec,name,center,size,color):
    x,y=center; w,h=size
    for sign in (-1,1):
        visual_box(spec,f'{name}_x{sign}',(x+sign*w/2,y,.004),(.018,h/2,.004),color)
        visual_box(spec,f'{name}_y{sign}',(x,y+sign*h/2,.004),(w/2,.018,.004),color)

def decorate_plant(spec,layout):
    # Existing staged-demonstration boundaries and temporary racks are replaced
    # by functional cells; the original build_world default remains unchanged.
    for geom in list(spec.geoms):
        # Depot spells its combined column mesh PILLERS, not PILLARS.
        # Hide the oversized legacy visual assemblies only in this layout.
        if geom.name.startswith('scene_Depot_') and any(
            token in geom.name.lower() for token in ('pillers','boxset','crates','window')
        ):
            geom.rgba=(1,1,1,0)
        if '_lane' in geom.name or geom.name.startswith(('rack','bin','post')):
            spec.delete(geom)
    for name,scenario,_,offset in layout['cases']:
        add_delivery_stage(spec,scenario,offset,name+'_delivery_stand',profile=layout.get('carry_profiles',{}).get(scenario))
    yellow=(.96,.73,.08,1); white=(.83,.85,.86,1); blue=(.1,.42,.8,1)
    # 2.2 m main aisle, visibly separate from machines and WIP staging.
    for y in (.4,2.6): visual_box(spec,'main_aisle'+str(y),(0,y,.004),(12.6,.025,.004),yellow)
    for x in (-12.6,12.6): visual_box(spec,'aisle_end'+str(x),(x,1.5,.004),(.025,1.1,.004),yellow)
    for x in (-8.,0.,7.5):
        for y in (1.0,1.5,2.0):
            visual_box(spec,f'crosswalk{x}_{y}',(x,y,.006),(.4,.10,.006),white)
    # Dashed access spurs enter actual workcells instead of fencing each robot
    # class into a separate decorative demonstration lane.
    for x,ys in [(-2.4,(2.8,3.2,3.6,4.0)),
                  (1.4,(-3.4,-3.,-2.6,-2.2,-1.8,-1.4,-1.,-.6,-.2)),
                  (7.,(-2.,-1.6,-1.2,-.8,-.4,0.))]:
        for y in ys:
            for side in (-1,1):
                visual_box(spec,f'cell_access{x}_{y}_{side}',(x+side*.48,y,.006),(.015,.11,.004),blue)
    for name,center,size in [
        ('receiving',(-10.5,4.5),(3.6,3.2)),('cnc',(-5.5,4.6),(3.2,3.2)),
        ('deburr',(-.5,4.5),(3.0,3.0)),('qc',(4,4.2),(3.4,3.0)),
        ('packaging',(9,4.3),(3.6,3.4)),('sheet',(-9,-3),(3.3,3.8)),
        ('flex',(-4.2,-3),(3.0,3.5)),('assembly',(4,-3.3),(4.4,4.4)),('shipping',(11,-3.5),(3.8,4.0))]:
        outline(spec,name+'_cell',center,size,white)
    for name,center in [('raw',(-10.5,3.4)),('cnc_in',(-6.45,3.1)),('cnc_out',(-4.55,3.1)),
                        ('asm_in',(1.8,-2.6)),('qc_wip',(5.3,3.3)),('rework',(3.0,4.9)),('fg',(10.5,-2.6))]:
        outline(spec,name+'_buffer',center,(.9,.75),blue)
    # Full-size CNC enclosure. Its front service point is within a UR5e workspace;
    # large enclosure dimensions do not imply an unreachable rear fixture.
    box(spec,'cnc_foundation',(-5.5,4.9,.35),(1.15,.9,.35),(.23,.32,.37,1))
    for x in (-6.65,-4.35): box(spec,'cnc_wall'+str(x),(x,4.9,1.45),(.055,.9,.75),(.20,.39,.46,1))
    box(spec,'cnc_back',(-5.5,5.8,1.45),(1.2,.055,.75),(.20,.39,.46,1))
    box(spec,'cnc_top',(-5.5,4.9,2.22),(1.2,.95,.04),(.20,.39,.46,1))
    box(spec,'cnc_fixture',(-5.5,4.18,.92),(.22,.15,.04),(.55,.58,.61,1))
    box(spec,'cnc_control_panel',(-4.15,4.0,1.2),(.15,.10,.25),(.12,.18,.23,1))
    visual_box(spec,'cnc_screen',(-4.15,3.895,1.27),(.11,.005,.12),(.1,.55,.65,1))
    worktable(spec,'cnc_input',(-6.45,3.5),width=.8,depth=.65)
    worktable(spec,'cnc_output',(-4.4,3.5),width=.8,depth=.65)
    # Sheet rack and press remain outside the two carriers' measured footpaths.
    for z in (.28,.48,.68): box(spec,'sheet_rack'+str(z),(-11.4,-3.2,z),(.5,.8,.025),(.26,.35,.40,1))
    for z in (.32,.52,.72): box(spec,'sheet_stock'+str(z),(-11.4,-3.2,z),(.45,.72,.012),(.62,.66,.7,1))
    for x in (-10.2,-7.8): box(spec,'press_column'+str(x),(x,-.5,1.3),(.12,.3,1.3),(.32,.42,.48,1))
    box(spec,'press_beam',(-9,-.5,2.6),(1.32,.3,.15),(.32,.42,.48,1))
    # Distinct cleaning and assembly fixtures rather than repeated carry lanes.
    worktable(spec,'deburr',(-.5,4.7),width=1.6,depth=.9)
    box(spec,'deburr_backguard',(-.5,5.16,1.1),(.8,.035,.3),(.18,.4,.4,.7))
    spec.worldbody.add_geom(name='cleaning_wheel',type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=(-.9,4.7,1.02),quat=(.70710678,.70710678,0,0),size=(.13,.05,.001),rgba=(.45,.5,.52,1))
    worktable(spec,'precision_assembly',(1.5,-4.4),width=1.6,depth=1.0)
    for dx in (-.45,0,.45): box(spec,'component_tray'+str(dx),(1.5+dx,-4.4,.89),(.17,.22,.04),(.15,.38,.55,1))
    worktable(spec,'rework',(2.3,5.2),width=1.0,depth=.7,color=(.4,.3,.45,1))
    # One fixed QC-to-packaging connection; other lanes remain mobile logistics.
    box(spec,'qc_pack_conveyor_frame',(6.9,3.05,.65),(1.35,.23,.08),(.18,.25,.3,1))
    for i in range(28):
        spec.worldbody.add_geom(name=f'qc_pack_roller{i}',type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            pos=(5.58+i*.10,3.05,.76),quat=(.70710678,.70710678,0,0),size=(.04,.22,.001),rgba=(.55,.6,.65,1))
    worktable(spec,'packaging',(9.3,4.5),width=1.8,depth=.9)
    for x in (-3.0,-2.0):
        outline(spec,'parking'+str(x),(x,-5.6),(.8,1.0),blue)
        box(spec,'charging_dock'+str(x),(x,-6.2,.25),(.3,.12,.25),(.15,.24,.3,1))
    populate(spec)
    # Move the cleaning cell intact beside sheet handling; the single G1 now
    # works beside CNC/QC. This breaks the old north-arm / south-humanoid split.
    import numpy as np
    for geom in spec.geoms:
        if geom.name.startswith(('deburr','cleaning_wheel')):
            geom.pos=np.asarray(geom.pos)+np.array([-4.5,-8.0,0.])
        elif geom.name.startswith('flex_cell'):
            geom.pos=np.asarray(geom.pos)+np.array([3.7,7.0,0.])
    # Loaded tote decks are rigid mounted cargo for these commissioning routes.
    # Only the QC workpiece is currently unloaded by an arm.
    for index in range(2,len(layout['courier_starts'])):
        base=spec.body(f'tb{index}_base_link')
        z=.325 if layout['courier_models'][index]=='husky_a200' else .245
        box(spec,f'tb{index}_cargo_tote',(0,0,z),(.10,.075,.028),(.16,.43,.55,1),body=base,mass=.25)
    add_fixed_cells(spec)
    from .cell_work import add_workpieces
    add_workpieces(spec)
    from .quadrupeds import add_quadrupeds
    add_quadrupeds(spec,layout.get('quadrupeds',[]))
    from .plant_finish import apply_finish
    apply_finish(spec,layout)
    from .dual_arm import add_cooperative_panel
    add_cooperative_panel(spec)

def build_plant():
    layout=load_layout()
    model,data,groups,arm=build_world(lambda spec:decorate_plant(spec,layout),layout=layout)
    # Reset only: thereafter the models' native position servos hold these poses.
    for name,kind,_,_ in FIXED_CELLS:
        pose=[0,-.5,0,-2,0,1.5,.8] if kind=='panda' else [0,-1.4,1.5,-1.7,-1.57,0]
        acts=[i for i in range(model.nu) if model.actuator(i).name.startswith(name+'_')]
        for act,target in zip(acts,pose):
            joint=int(model.actuator_trnid[act,0])
            data.qpos[model.jnt_qposadr[joint]]=target
            data.ctrl[act]=target
    mujoco.mj_forward(model,data)
    return model,data,groups,arm,layout
