"""Direct integration of approved roqsim assets, without ROS."""
from pathlib import Path
import json
import mujoco

ROOT=Path(__file__).resolve().parents[2]
UPSTREAM=ROOT.parent/'roqsim_factory_upstream'
CONFIG=ROOT/'factory/metal_parts.json'

def load_config(path=CONFIG):
    return json.loads(Path(path).read_text())

def build_model(config=None):
    config=config or load_config()
    spec=mujoco.MjSpec.from_file(str(UPSTREAM/'roqsim_scenes/src/roqsim_scenes/worlds/depot/depot.xml'))
    def box(name,pos,size,color,body=None,mass=0):
        return (body or spec.worldbody).add_geom(name=name,type=mujoco.mjtGeom.mjGEOM_BOX,pos=pos,size=size,rgba=color,mass=mass)
    for row in config['robots']:
        name,model,pos=(row[k] for k in ('name','model','position'))
        package='roqsim_mobile' if model=='husky_a200' else 'roqsim_manipulation_assets'
        robot=mujoco.MjSpec.from_file(str(UPSTREAM/package/'src'/package/'models'/model/(model+'.xml')))
        for pair in list(robot.pairs):
            robot.delete(pair)
        if model.startswith('ur'):
            grip=mujoco.MjSpec.from_file(str(UPSTREAM/'roqsim_manipulation_assets/src/roqsim_manipulation_assets/models/robotiq_2f85/robotiq_2f85.xml'))
            robot.attach(grip,prefix='grip_',site=robot.site('tool_site') or robot.site('attachment_site'))
        spec.attach(robot,prefix=name+'_',frame=spec.worldbody.add_frame(name=name+'_spawn',pos=pos))
        if package=='roqsim_manipulation_assets':
            x,y,z=pos
            box(name+'_pedestal',(x,y,.4),(.12,.12,.4),(.18,.24,.3,1))
            box(name+'_table',(x,y+.60,.78),(.65,.5,.04),(.35,.39,.43,1))
            for dx in (-.55,.55):
                box(name+f'_leg{dx}',(x+dx,y+.6,.38),(.045,.4,.38),(.15,.18,.22,1))
            part=spec.worldbody.add_body(name=name+'_part',pos=(x-.25,y+.55,.86))
            part.add_freejoint(name=name+'_part_free')
            geom=box(name+'_part_geom',(0,0,0),(.035,.035,.035),(.95,.52,.08,1),body=part,mass=.15)
            geom.conaffinity=3
            spec.add_equality(name=name+'_grasp',type=mujoco.mjtEq.mjEQ_WELD,objtype=mujoco.mjtObj.mjOBJ_BODY,
                name1=name+('_hand' if model=='panda' else '_wrist_3_link'),name2=name+'_part',active=False,solref=(.02,1))
            # Assisted-grasp mode: prevent the hand's collision meshes fighting its
            # payload weld. Payload still collides with tables, floor and other parts.
            spec.worldbody.add_site(name=name+'_output',pos=(x+.25,y+.55,.86),size=(.05,.05,.005),rgba=(.1,.8,.5,.6))
    armrows=[r for r in config['robots'] if r['model']!='husky_a200']
    for payload in armrows:
        for body in spec.bodies:
            if any(body.name.startswith(r['name']+'_') for r in armrows) and not body.name.endswith('_part'):
                spec.add_exclude(bodyname1=body.name,bodyname2=payload['name']+'_part')
    for row in config['robots']:
        if row['model']=='husky_a200' and armrows:
            spec.add_equality(name=row['name']+'_cargo',type=mujoco.mjtEq.mjEQ_WELD,
                objtype=mujoco.mjtObj.mjOBJ_BODY,name1=row['name']+'_base_link',
                name2=armrows[0]['name']+'_part',active=False,solref=(.02,1))
    for x in (-3.65,-2.35):
        box('cnc_wall'+str(x),(x,1.15,1.05),(.04,.25,.25),(.22,.36,.43,1))
    box('cnc_roof',(-3,1.15,1.34),(.69,.25,.04),(.22,.36,.43,1))
    box('cnc_back',(-3,1.4,1.05),(.65,.04,.25),(.22,.36,.43,1))
    for row in config['robots']:
        if not row['name'].endswith('assembly'): continue
        prefix=row['name'].removesuffix('assembly'); y=row['position'][1]
        box(prefix+'conveyor',(1.5,y+.55,.76),(1.25,.16,.04),(.10,.14,.19,1))
        for i in range(35):
            roller=spec.worldbody.add_body(name=prefix+f'roller{i}',pos=(.30+i*.07,y+.55,.83))
            roller.add_joint(name=prefix+f'roller_joint{i}',type=mujoco.mjtJoint.mjJNT_HINGE,axis=(0,1,0),damping=.01)
            roller.add_geom(name=prefix+f'roller_geom{i}',type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                quat=(.7071,.7071,0,0),size=(.05,.16,.001),mass=.1,contype=2,conaffinity=0,
                friction=(1,.005,.0001),rgba=(.6,.65,.7,1))
            spec.add_actuator(name=prefix+f'roller_motor{i}',target=prefix+f'roller_joint{i}',trntype=mujoco.mjtTrn.mjTRN_JOINT,
                gainprm=(.1,0,0,0,0,0,0,0,0,0),biastype=mujoco.mjtBias.mjBIAS_AFFINE,
                biasprm=(0,0,-.1,0,0,0,0,0,0,0))
    box('inspection_camera',(0,1.05,1.5),(.06,.05,.04),(.05,.08,.12,1))
    box('shipping_pallet',(4.5,1,.08),(.6,.4,.08),(.55,.32,.14,1))
    spec.option.timestep=.002
    return spec.compile()
