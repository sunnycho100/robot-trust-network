"""Nine-robot mixed logistics warehouse, with selectable showreel cameras.

G1 payloads use physical contact only. The Panda's small-part grasp is explicitly
assisted by a proximity-gated weld; all base and arm motion uses feedback.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import mujoco
import numpy as np

from .g1_logistics import build_group
from .g1_policy import G1Controller, G1UpperBodyController, gait_phase, yaw_from_quaternion
from .factory_scene import UPSTREAM
from .factory_executor import Arm
from .coop_lift import RawVideoWriter

CASES=[('sofa','sofa_side',2.,(-4.,-2.,0)),
       ('crate','bottom_crate',5.,(0.,-2.,0)),
       ('frame','team_fixture',3.,(3.7,-1.,0))]
SHOTS=['overview','sofa','crate','frame','supply']


def box(spec,name,pos,size,color,body=None,mass=0):
    return (body or spec.worldbody).add_geom(name=name,type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=pos,size=size,rgba=color,mass=mass)


def build_world(scene_hook=None,layout=None):
    layout=layout or {}
    cases=layout.get('cases',CASES)
    courier_starts=layout.get('courier_starts',((-.95,3.05),(-4.,4.5)))
    courier_models=layout.get('courier_models',['turtlebot3_waffle']*len(courier_starts))
    initial_parts=layout.get('initial_part_names',['supply_part','return_part'])
    panda_position=layout.get('panda_position',(0,3.5,.5))
    table_position=layout.get('table_position',(.35,3.8,.64))
    table_size=layout.get('table_size',(.25,.25,.04))
    world=mujoco.MjSpec.from_file(str(UPSTREAM/'roqsim_scenes/src/roqsim_scenes/worlds/depot/depot.xml'))
    world.option.integrator=mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    world.option.cone=mujoco.mjtCone.mjCONE_ELLIPTIC
    floor=world.geom('floor'); floor.contype=9; floor.conaffinity=9; floor.friction=(1.2,.02,.002)
    for geom in world.geoms:
        name=geom.name.lower()
        if any(s in name for s in ('roof','walls','pillars','fans','fan1','fan2','fanback','window','pipes','stairs')):
            geom.rgba=(1,1,1,0)
        if any(s in name for s in ('boxset','crates')):
            geom.pos=np.array(geom.pos)+np.array([8,0,0])
    seeds=[]
    for name,scenario,mass,offset in cases:
        seed=build_group(SimpleNamespace(scenario=scenario,mass=mass,carriers=3))
        spec=seed[0]
        spec.delete(spec.geom('floor'))
        for light in list(spec.lights): spec.delete(light)
        world.attach(spec,prefix=name+'_',frame=world.worldbody.add_frame(name=name+'_zone',pos=offset))
        seeds.append(seed)
    # Clear lanes and compact fixtures, outside the validated carry footprints.
    colors=[(.15,.65,.7,1),(.95,.6,.12,1),(.42,.5,.85,1)]
    for (name,_,_,offset),color in zip(cases,colors):
        x,y,_=offset
        for dx in (-1.5,1.5): box(world,name+f'_lane{dx}',(x+dx,y+.4,.003),(.018,1.9,.003),color).contype=0
    for n,y in enumerate((-4.,0.,4.)):
        for x in (-6.5,6.5):
            for z in (.4,1.1,1.8):
                box(world,f'rack{n}_{x}_{z}',(x,y,z),(.35,.6,.04),(.16,.27,.35,1))
                for dy in (-.30,.30):
                    box(world,f'bin{n}_{x}_{z}_{dy}',(x,y+dy,z+.16),(.22,.20,.12),(.62,.4,.18,1))
            for dy in (-.55,.55):
                box(world,f'post{n}_{x}_{dy}',(x,y+dy,1),(.04,.04,1),(.16,.27,.35,1))
    for index,(x,y) in enumerate(courier_starts):
        model_name=courier_models[index]
        robot=mujoco.MjSpec.from_file(str(UPSTREAM/'roqsim_mobile/src/roqsim_mobile/models'/model_name/(model_name+'.xml')))
        for pair in list(robot.pairs): robot.delete(pair)
        for key in list(robot.keys): robot.delete(key)
        tray_z=.28 if model_name=='husky_a200' else .20
        box(robot,'tray',(0,0,tray_z),(.10,.09,.012),(.28,.38,.45,1),body=robot.body('base_link'),mass=.05)
        world.attach(robot,prefix=f'tb{index}_',frame=world.worldbody.add_frame(pos=(x,y,0)))
        if model_name in ('clearpath_jackal','husky_a200'):
            friction=.5 if model_name=='clearpath_jackal' else 1.
            for wheel in ('front_left','rear_left','front_right','rear_right'):
                world.add_pair(geomname1=f'tb{index}_{wheel}_wheel_geom',geomname2='floor',
                    condim=3,solref=(.02,1),friction=(friction,friction,.005,.0001,.0001))
        part_name=initial_parts[index] if index<len(initial_parts) else None
        if part_name:
            body=world.worldbody.add_body(name=part_name,pos=(x,y,.467 if model_name=='husky_a200' else .247))
            body.add_freejoint()
            box(world,f'part{index}',(0,0,0),(.025,.025,.025),(.95,.5,.1,1) if index==0 else (.2,.6,.9,1),body=body,mass=.05)
    panda=mujoco.MjSpec.from_file(str(UPSTREAM/'roqsim_manipulation_assets/src/roqsim_manipulation_assets/models/panda/panda.xml'))
    world.attach(panda,prefix='supply_',frame=world.worldbody.add_frame(pos=panda_position))
    world.add_equality(name='supply_grasp',type=mujoco.mjtEq.mjEQ_WELD,objtype=mujoco.mjtObj.mjOBJ_BODY,
        name1='supply_hand',name2='supply_part',active=False,solref=(.02,1))
    for body in world.bodies:
        if body.name.startswith('supply_') and body.name!='supply_part':
            world.add_exclude(bodyname1=body.name,bodyname2='supply_part')
    box(world,'panda_pedestal',(panda_position[0],panda_position[1],panda_position[2]/2),(.12,.12,panda_position[2]/2),(.18,.24,.3,1))
    box(world,'inspection_table',table_position,table_size,(.32,.38,.42,1))
    world.option.timestep=.002; world.option.integrator=mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    world.option.cone=mujoco.mjtCone.mjCONE_ELLIPTIC
    world.visual.global_.offwidth=1280; world.visual.global_.offheight=720
    world.visual.headlight.ambient=(.4,.4,.4)
    world.visual.headlight.diffuse=(.7,.7,.7)
    if scene_hook is not None:
        scene_hook(world)
    model=world.compile(); data=mujoco.MjData(model)
    groups=[]
    for (name,scenario,mass,offset),seed in zip(cases,seeds):
        _,old,initial,_,old_arms,actors,_,cfg,_=seed
        # Transfer reset state by named joint. Offsets apply only to free roots;
        # this happens once at initialization, never while the scene is running.
        for j in range(old.njnt):
            new=model.joint(name+'_'+old.joint(j).name).id
            a,b=old.jnt_qposadr[j],model.jnt_qposadr[new]
            count=7 if old.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE else 1
            data.qpos[b:b+count]=initial.qpos[a:a+count]
            if count==7: data.qpos[b:b+3]+=offset
        legs=[]; arms=[]
        for (prefix,_,_),old_arm in zip(actors,old_arms):
            leg=G1Controller(model,cfg,name+'_'+prefix); legs.append(leg)
            arm=G1UpperBodyController(model,name+'_'+prefix); arm.target_q=old_arm.target_q.copy(); arms.append(arm)
        groups.append(dict(name=name,scenario=scenario,mass=mass,offset=np.array(offset),actors=actors,
            cfg=cfg,legs=legs,arms=arms,body=model.body(name+'_payload').id))
    arm=Arm(model,data,dict(name='supply',model='panda',position=panda_position,minimum_lift_z=layout.get('minimum_lift_z',.8)))
    arm.pipeline=True; arm.state='WAIT_INPUT'; arm.dest=np.array(layout.get('drop_target',(.35,3.8,.71)))
    mujoco.mj_forward(model,data)
    for group in groups:
        group.update(start=data.xpos[group['body']].copy(),minimum_clearance=100.,supported=0,force_sum=0.,max_penetration=0.)
    return model,data,groups,arm


class Courier:
    def __init__(self,model,index,goal):
        self.name=f'tb{index}'; self.body=model.body(self.name+'_base_link').id
        self.acts=[model.actuator(self.name+'_'+s+'_wheel_motor').id for s in ('left','right')]
        self.goal=np.array(goal); self.arrived=False

    def update(self,data,events):
        delta=self.goal-data.xpos[self.body,:2]
        if np.linalg.norm(delta)<.045:
            if not self.arrived: events.append(dict(t=data.time,event='COURIER_ARRIVED',robot=self.name))
            self.arrived=True; data.ctrl[self.acts]=0; return
        yaw=yaw_from_quaternion(data.xquat[self.body])
        error=(np.arctan2(delta[1],delta[0])-yaw+np.pi)%(2*np.pi)-np.pi
        v=min(.22,1.3*np.linalg.norm(delta))*max(0.,np.cos(error)); w=np.clip(2*error,-1,1)
        data.ctrl[self.acts]=[(v-.144*w)/.033,(v+.144*w)/.033]


def camera_for(shot,data,groups):
    camera=mujoco.MjvCamera(); camera.azimuth=105; camera.elevation=-42
    if shot=='overview': camera.lookat[:]=(0,1,.7); camera.distance=14.8; camera.elevation=-55
    elif shot=='supply': camera.lookat[:]=(-.3,3.25,.6); camera.distance=3.5; camera.azimuth=135
    else:
        group=next(g for g in groups if g['name']==shot)
        camera.lookat[:]=data.xpos[group['body']]; camera.distance=3.3 if shot=='sofa' else 4.7
        camera.azimuth=135; camera.elevation=-24
    return camera


def run(args,scene_hook=None,scene_labels=None,overlay=None):
    print('Building mixed warehouse and loading six G1 policies...',flush=True)
    model,data,groups,arm=build_world(scene_hook)
    couriers=[Courier(model,0,(-.35,3.05)),Courier(model,1,(-2.8,4.5))]
    events=[]; dispatched=False; handoff=False
    renderer=writer=None; frames=0; steps=round(args.duration/.002)
    if args.video:
        renderer=mujoco.Renderer(model,height=720,width=1280)
        writer=RawVideoWriter(args.video,1280,720,30,total_frames=int(np.ceil(args.duration*30)))
    force=np.zeros(6)
    try:
        for step in range(steps):
            for g in groups:
                for leg in g['legs']: leg.apply(data)
                for upper in g['arms']: upper.apply(data)
                if g['scenario']=='sofa_side':
                    upper=g['arms'][0]; dofs=model.jnt_dofadr[model.actuator_trnid[upper.actuators,0]]
                    data.ctrl[upper.actuators]+=data.qfrc_bias[dofs]
            if step%5==0:
                for courier in couriers: courier.update(data,events)
                if couriers[0].arrived and not dispatched:
                    arm.source=data.xpos[arm.part].copy(); arm.transition(data,events,'APPROACH'); dispatched=True
                    events.append(dict(t=data.time,event='PART_PRESENTED_TO_PANDA',part_id=int(arm.part)))
                arm.update(model,data,events)
                if data.eq_active[arm.eq] and not handoff:
                    handoff=True
                    events.append(dict(t=data.time,event='TURTLEBOT_TO_PANDA_HANDOFF',part_id=int(arm.part)))
            mujoco.mj_step(model,data)
            if not np.isfinite(data.qpos).all(): raise RuntimeError('Nonfinite shared-world state')
            for g in groups:
                if step%g['cfg'].control_decimation==0:
                    speed=.08 if data.time>1.5 else 0.
                    for leg,(prefix,_,facing) in zip(g['legs'],g['actors']):
                        cmd=np.array([speed if g['name']=='sofa' or facing==0 else .04,0 if g['name']=='sofa' or facing==0 else speed*facing,0],dtype=np.float32)
                        if g['name']=='sofa':
                            body=data.body('sofa_'+prefix+'pelvis'); local=body.xpos-g['offset']
                            target_x=.08*max(0.,data.time-1.5)
                            vel=np.array([speed,0.])+.8*np.array([target_x-local[0],-local[1]])
                            cmd[:2]=np.clip(body.xmat.reshape(3,3)[:2,:2].T@vel,-.25,.25)
                            cmd[2]=np.clip(-yaw_from_quaternion(body.xquat),-.4,.4)
                        leg.update_policy(data,cmd,gait_phase(data))
            supports={g['body']:0. for g in groups}
            for j,c in enumerate(data.contact):
                b1,b2=int(model.geom_bodyid[c.geom1]),int(model.geom_bodyid[c.geom2])
                for g in groups:
                    if g['body'] not in (b1,b2): continue
                    g['max_penetration']=max(g['max_penetration'],max(0.,-float(c.dist)))
                    other=b2 if b1==g['body'] else b1
                    if model.body(other).name.endswith('wrist_roll_rubber_hand'):
                        mujoco.mj_contactForce(model,data,j,force)
                        supports[g['body']]+=abs(float((c.frame.reshape(3,3).T@force[:3])[2]))
            for g in groups:
                support=supports[g['body']]; g['supported']+=support>.5*g['mass']*9.81; g['force_sum']+=support
                for geo in range(model.ngeom):
                    if model.geom_bodyid[geo]==g['body']:
                        z=data.geom_xpos[geo,2]-np.abs(data.geom_xmat[geo].reshape(3,3)[2])@model.geom_size[geo]
                        g['minimum_clearance']=min(g['minimum_clearance'],float(z))
            if writer and step*.002>=frames/30:
                shot=SHOTS[min(4,int(data.time/args.duration*5))] if args.shot=='auto' else args.shot
                renderer.update_scene(data,camera=camera_for(shot,data,groups))
                if scene_labels is not None: scene_labels(renderer.scene)
                pixels=renderer.render()
                from PIL import Image,ImageDraw,ImageFont
                try:
                    font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf',22)
                    small=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',15)
                except OSError:
                    font=small=ImageFont.load_default()
                screen=Image.fromarray(pixels); draw=ImageDraw.Draw(screen)
                draw.rectangle((0,0,1280,65),fill=(15,22,31))
                titles=dict(overview='MIXED LOGISTICS | 9 ROBOTS | ONE PHYSICAL WORLD',sofa='SOFA MODULE | 1 G1 | 2 kg | OPPOSITE SIDE CONTACT',
                    crate='EQUIPMENT CRATE | 2 G1 | 5 kg | BOTTOM SUPPORT',frame='ECCENTRIC T-FRAME | 3 G1 | 3 kg | THREE SUPPORT POINTS',supply='SMALL-PART SUPPLY | TURTLEBOT + PANDA | FEEDBACK HANDOFF')
                draw.text((18,8),titles[shot],font=font,fill=(245,245,245))
                draw.text((18,39),f't={data.time:.1f}s | Contact-only G1 carries | Panda assisted grasp: {arm.state}',font=small,fill=(100,230,190))
                draw.rectangle((0,696,1280,720),fill=(15,22,31))
                draw.text((12,702),'Depot: Rafi Abdullah & Louise Poubel / Open Robotics, CC-BY-4.0; MuJoCo: roqsim; cutaway and workcells added',fill=(190,195,205))
                if overlay is not None: overlay(screen,draw,data,groups,arm)
                writer.write(np.asarray(screen)); frames+=1
    finally:
        if writer: writer.close()
        if renderer: renderer.close()
    results={}
    for g in groups:
        pelvis=[float(data.body(g['name']+'_'+p+'pelvis').xpos[2]) for p,_,_ in g['actors']]
        result=dict(mass_kg=g['mass'],robots=len(g['actors']),displacement_m=float(np.linalg.norm(data.xpos[g['body'],:2]-g['start'][:2])),
            support_fraction=g['supported']/steps,mean_hand_force_N=g['force_sum']/steps,min_floor_clearance_m=g['minimum_clearance'],max_penetration_m=g['max_penetration'],pelvis_z=pelvis)
        result['pass']=bool(result['displacement_m']>.15 and result['support_fraction']>.9 and result['min_floor_clearance_m']>.2 and max(result['pelvis_z'])<1.2 and min(pelvis)>.6 and result['max_penetration_m']<.005)
        results[g['name']]=result
    report=dict(duration_s=data.time,robot_count=9,cases=results,courier_arrived=[c.arrived for c in couriers],
        panda_state=arm.state,panda_cycles=arm.cycles,panda_part=data.xpos[arm.part].tolist(),handoff=handoff,
        panda_error_m=arm.error,panda_tcp=data.site_xpos[arm.site].tolist(),
        panda_contacts=[(model.body(int(model.geom_bodyid[c.geom1])).name,model.body(int(model.geom_bodyid[c.geom2])).name,c.dist) for c in data.contact if model.body(int(model.geom_bodyid[c.geom1])).name.startswith('supply_') or model.body(int(model.geom_bodyid[c.geom2])).name.startswith('supply_')],events=events)
    report['all_carries_pass']=all(r['pass'] for r in results.values())
    report['completion_pass']=bool(report['all_carries_pass'] and handoff and arm.state=='DONE'
        and all(c.arrived for c in couriers) and np.linalg.norm(data.xpos[arm.part]-arm.dest)<.06)
    args.report.parent.mkdir(parents=True,exist_ok=True); args.report.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    if args.require_complete and not report['completion_pass']:
        raise RuntimeError('Shared-world commissioning incomplete; inspect report')


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--duration',type=float,default=8.)
    parser.add_argument('--shot',choices=SHOTS+['auto'],default='auto')
    parser.add_argument('--video',type=Path)
    parser.add_argument('--report',type=Path,default=Path('outputs/industrial_showcase.json'))
    parser.add_argument('--require-complete',action='store_true')
    args=parser.parse_args()
    if not 0<args.duration<=8: parser.error('Validated carry windows are up to 8 seconds; use separate shots for a longer edit')
    run(args)
