"""Feedback-driven factory commissioning demo, with explicit assisted grasps."""
import argparse
import json
import time
from pathlib import Path
import mujoco
import numpy as np
from .factory_scene import build_model, load_config, CONFIG
from .coop_lift import RawVideoWriter


class AMR:
    def __init__(self, model, name, origin):
        self.name=name
        self.body=model.body(name+'_base_link').id
        self.act=[model.actuator(name+'_'+p+'_wheel_motor').id for p in ('front_left','rear_left','front_right','rear_right')]
        x,y,_=origin
        self.points=np.array([(x+2,y),(x+2,y+1.2),(x,y+1.2),(x,y)])
        self.index=0
        self.arrivals=0
        self.distance=0.
        self.last=None
        self.finished=False; self.stop_at_end=False

    def route(self,points):
        self.points=np.array(points); self.index=0; self.finished=False; self.stop_at_end=True

    def update(self, model, data, events):
        if self.finished:
            data.ctrl[self.act]=0
            return
        pos=data.xpos[self.body,:2]
        if self.last is not None:
            self.distance+=float(np.linalg.norm(pos-self.last))
        self.last=pos.copy()
        delta=self.points[self.index]-pos
        if np.linalg.norm(delta)<.16:
            self.arrivals+=1
            events.append(dict(t=round(data.time,3),robot=self.name,event='WAYPOINT_ARRIVED',waypoint=self.index))
            if self.stop_at_end and self.index==len(self.points)-1:
                self.finished=True; data.ctrl[self.act]=0
                return
            self.index=(self.index+1)%len(self.points)
            delta=self.points[self.index]-pos
        rot=data.xmat[self.body].reshape(3,3)
        yaw=np.arctan2(rot[1,0],rot[0,0])
        error=(np.arctan2(delta[1],delta[0])-yaw+np.pi)%(2*np.pi)-np.pi
        v=min(.45,float(np.linalg.norm(delta)))*max(0.,np.cos(error))
        w=float(np.clip(2*error,-.9,.9))
        data.ctrl[self.act]=[(v-.2854*w)/.1651]*2+[(v+.2854*w)/.1651]*2


def weld_payload(model,data,eq,body,part):
    model.eq_obj2id[eq]=part
    model.eq_data[eq,:3]=0
    model.eq_data[eq,3:6]=data.xmat[body].reshape(3,3).T@(data.xpos[part]-data.xpos[body])
    inv=data.xquat[body].copy(); inv[1:]*=-1
    quat=np.zeros(4); mujoco.mju_mulQuat(quat,inv,data.xquat[part])
    model.eq_data[eq,6:10]=quat
    data.eq_active[eq]=1


class Arm:
    def __init__(self, model, data, row):
        self.name=row['name']; self.model=row['model']
        self.minimum_lift_z=row.get('minimum_lift_z',1.1)
        self.lift_clearance=row.get('lift_clearance',.25)
        self.transfer_clearance=row.get('transfer_clearance',.12 if self.model=='panda' else .25)
        self.tolerance=row.get('tolerance',.035)
        p=self.name+'_'
        panda=self.model=='panda'
        self.site=model.site(p+('pinch' if panda else 'grip_pinch')).id
        joints=[p+f'joint{i}' for i in range(1,8)] if panda else [p+s+'_joint' for s in ('shoulder_pan','shoulder_lift','elbow','wrist_1','wrist_2','wrist_3')]
        j=np.array([model.joint(n).id for n in joints])
        self.q=model.jnt_qposadr[j]; self.d=model.jnt_dofadr[j]
        self.limits=model.jnt_range[j].copy()
        self.acts=np.array([model.actuator(p+f'actuator{i}').id for i in range(1,8)]) if panda else np.array([model.actuator(p+s).id for s in ('shoulder_pan','shoulder_lift','elbow','wrist_1','wrist_2','wrist_3')])
        self.grip=model.actuator(p+('actuator8' if panda else 'grip_fingers_actuator')).id
        self.part=model.body(row.get('part_name',p+'part')).id
        self.eq=model.equality(row.get('grasp_name',p+'grasp')).id
        self.base=model.eq_obj1id[self.eq]
        home=[0,-.5,0,-2,0,1.5,.8] if panda else [0,-1.4,1.5,-1.7,-1.57,0]
        data.qpos[self.q]=home
        data.ctrl[self.acts]=home
        self.state='APPROACH'; self.enter=0.; self.cycles=0; self.error=0.
        self.pipeline=False
        self.auto_transitions=True
        self.grasp_offset=np.array(row.get('grasp_offset',[0.,0.,0.]))
        self.max_joint_step=row.get('max_joint_step',.15)
        self.desired_rotation=np.diag([1.,-1.,-1.])
        self.x,self.y,_=row['position']
        self.source=np.array([self.x-.25,self.y+.55,.86]); self.dest=np.array([self.x+.25,self.y+.55,.86])
        self.jp=np.zeros((3,model.nv)); self.jr=self.jp.copy()
        self.ik=mujoco.MjData(model)

    def transition(self,data,events,state):
        events.append(dict(t=round(data.time,3),robot=self.name,event=state,part=data.xpos[self.part].tolist()))
        self.state=state; self.enter=data.time

    def attach(self,model,data):
        # Set the actual relative pose before enabling the weld: no object teleport.
        weld_payload(model,data,self.eq,self.base,self.part)

    def update(self,model,data,events):
        if self.state in ('WAIT_INPUT','DONE'):
            data.qfrc_applied[self.d]=data.qfrc_bias[self.d]
            return
        part=data.xpos[self.part]+data.xmat[self.part].reshape(3,3)@self.grasp_offset
        if self.state=='APPROACH': target=part+np.array([0,0,.20])
        elif self.state=='GRASP': target=part.copy()
        elif self.state=='LIFT':
            target=self.source.copy(); target[2]=max(self.minimum_lift_z,self.source[2]+self.lift_clearance)
        elif self.state=='TRANSFER': target=self.dest+np.array([0,0,self.transfer_clearance])
        elif self.state=='PLACE': target=self.dest.copy()
        else: target=self.dest+np.array([0,0,self.transfer_clearance])
        delta=target-data.site_xpos[self.site]
        self.error=float(np.linalg.norm(delta))
        desired=self.desired_rotation
        self.ik.qpos[:]=data.qpos
        for _ in range(10):
            mujoco.mj_kinematics(model,self.ik)
            mujoco.mj_comPos(model,self.ik)
            mujoco.mj_jacSite(model,self.ik,self.jp,self.jr,self.site)
            current=self.ik.site_xmat[self.site].reshape(3,3)
            orientation=.5*sum((np.cross(current[:,i],desired[:,i]) for i in range(3)))
            weight=.08 if self.model=='panda' else .25
            jac=np.vstack([self.jp[:,self.d],weight*self.jr[:,self.d]])
            err=np.r_[target-self.ik.site_xpos[self.site],weight*orientation]
            dq=jac.T@np.linalg.solve(jac@jac.T+.001*np.eye(6),err)
            self.ik.qpos[self.q]+=np.clip(dq,-.1,.1)
            self.ik.qpos[self.q]=np.clip(self.ik.qpos[self.q],self.limits[:,0]+.005,self.limits[:,1]-.005)
        qtarget=data.qpos[self.q]+np.clip(self.ik.qpos[self.q]-data.qpos[self.q],-self.max_joint_step,self.max_joint_step)
        data.qfrc_applied[self.d]=data.qfrc_bias[self.d]
        data.ctrl[self.acts]=np.clip(qtarget,model.actuator_ctrlrange[self.acts,0],model.actuator_ctrlrange[self.acts,1])
        data.ctrl[self.grip]=255 if self.state in ('LIFT','TRANSFER','PLACE') else 0
        if not self.auto_transitions: return
        age=data.time-self.enter
        if self.error<self.tolerance:
            if self.state=='APPROACH': self.transition(data,events,'GRASP')
            elif self.state=='GRASP' and np.linalg.norm(part-data.site_xpos[self.site])<.045:
                self.attach(model,data); self.transition(data,events,'LIFT')
            elif self.state=='LIFT': self.transition(data,events,'TRANSFER')
            elif self.state=='TRANSFER': self.transition(data,events,'PLACE')
            elif self.state=='PLACE':
                if np.linalg.norm(part-self.dest)<.07:
                    data.eq_active[self.eq]=0; self.transition(data,events,'RETRACT')
            elif self.state=='RETRACT' and age>.5:
                self.cycles+=1
                if not self.pipeline:
                    self.source,self.dest=self.dest.copy(),self.source.copy()
                self.transition(data,events,'DONE' if self.pipeline else 'APPROACH')


class ProductionLine:
    """One physical workpiece: CNC -> carrier -> Panda -> conveyor -> UR10e -> shipping carrier."""
    def __init__(self,model,data,arms,amrs,prefix='',y=0):
        self.prefix=prefix; self.y=y
        self.arms={a.name.removeprefix(prefix):a for a in arms if a.name.startswith(prefix)}
        self.amrs={a.name.removeprefix(prefix):a for a in amrs if a.name.startswith(prefix)}
        self.cnc=self.arms['cnc']; self.assembly=self.arms['assembly']; self.pack=self.arms['packaging']
        self.first=self.amrs['amr1']; self.second=self.amrs['amr2']
        self.part=self.cnc.part
        self.state='DISPATCH_TO_CNC'
        self.first.route([(-2.55,y-.45)])
        self.second.route([(3.45,y-.45)])
        for a in self.arms.values():
            a.pipeline=True; a.state='WAIT_INPUT'
            a.part=self.part; model.eq_obj2id[a.eq]=self.part
        self.cnc.dest=np.array([-2.55,y-.45,.45])
        self.assembly.dest=np.array([.43,y+.55,.93])
        self.pack.dest=np.array([3.45,y-.45,.45])
        self.eq1=model.equality(prefix+'amr1_cargo').id; self.eq2=model.equality(prefix+'amr2_cargo').id
        self.rollers=np.array([model.actuator(prefix+f'roller_motor{i}').id for i in range(35)])
        self.handoffs=0
        self.dispatch_at=6*int(prefix[4:-1]) if prefix.startswith('cell') else 0
        if self.dispatch_at:
            self.first.finished=self.second.finished=True
            self.state='QUEUED'

    def event(self,data,events,state):
        events.append(dict(t=round(data.time,3),line=self.prefix or 'main',event=state,part_id=int(self.part),part=data.xpos[self.part].tolist()))
        self.state=state

    def update(self,model,data,events):
        state=self.state
        if state=='QUEUED' and data.time>=self.dispatch_at:
            self.first.route([(-2.55,self.y-.45)])
            self.second.route([(3.45,self.y-.45)])
            self.event(data,events,'DISPATCH_TO_CNC')
        elif state=='DISPATCH_TO_CNC' and self.first.finished:
            self.cnc.transition(data,events,'APPROACH'); self.event(data,events,'CNC_WORK')
        elif state=='CNC_WORK' and self.cnc.state=='DONE':
            if np.linalg.norm(data.xpos[self.part]-self.cnc.dest)<.13:
                weld_payload(model,data,self.eq1,self.first.body,self.part)
                self.first.route([(-.45,self.y-.45)])
                self.handoffs+=1; self.event(data,events,'AMR_TO_ASSEMBLY')
        elif state=='AMR_TO_ASSEMBLY' and self.first.finished:
            data.eq_active[self.eq1]=0
            self.assembly.source=data.xpos[self.part].copy()
            self.assembly.transition(data,events,'APPROACH')
            self.handoffs+=1; self.event(data,events,'ASSEMBLY_WORK')
        elif state=='ASSEMBLY_WORK' and self.assembly.state=='DONE':
            self.handoffs+=1; self.event(data,events,'CONVEYOR_RUNNING')
        elif state=='CONVEYOR_RUNNING' and data.xpos[self.part,0]>2.55 and self.second.finished:
            self.pack.source=data.xpos[self.part].copy()
            self.pack.transition(data,events,'APPROACH')
            self.handoffs+=1; self.event(data,events,'PACKAGING_WORK')
        elif state=='PACKAGING_WORK' and self.pack.state=='DONE':
            if np.linalg.norm(data.xpos[self.part]-self.pack.dest)<.13:
                weld_payload(model,data,self.eq2,self.second.body,self.part)
                self.second.route([(4.5,self.y-2.)])
                self.handoffs+=1; self.event(data,events,'SHIPPING_TRANSPORT')
        elif state=='SHIPPING_TRANSPORT' and self.second.finished:
            self.event(data,events,'SHIPPED')


def run(args):
    config=load_config(args.config)
    if args.scenario or args.cells>1:
        from .factory_planner import plan
        config=plan(args.scenario or 'metal',args.cells)
    model=build_model(config); data=mujoco.MjData(model)
    arms=[Arm(model,data,r) for r in config['robots'] if r['model']!='husky_a200']
    amrs=[AMR(model,r['name'],r['position']) for r in config['robots'] if r['model']=='husky_a200']
    mujoco.mj_forward(model,data)
    events=[]; renderer=writer=viewer=None
    pipeline=config['type'] in ('metal','metal_parts')
    lines=[]
    if pipeline:
        for row in config['robots']:
            if row['name'].endswith('assembly'):
                lines.append(ProductionLine(model,data,arms,amrs,row['name'].removesuffix('assembly'),row['position'][1]))
    handoff=False
    camera=mujoco.MjvCamera(); camera.lookat[:]=(0,0,.8)
    camera.distance=18 if args.cells>1 else 10; camera.azimuth=90; camera.elevation=-55
    # Open the warehouse roof for an oblique overview; retain source assets unchanged.
    for i in range(model.ngeom):
        if model.geom_type[i]==mujoco.mjtGeom.mjGEOM_MESH:
            mid=model.geom_dataid[i]
            name=(model.geom(i).name or '').lower()
            if mid>=0 and name.startswith('scene_') and any(s in name for s in ('roof','walls','pillars','fans','fan1','fan2','fanback','window','pipes','stairs')):
                model.geom_rgba[i,3]=0
            if mid>=0 and name.startswith('scene_') and any(s in name for s in ('boxset','crates')):
                model.geom_pos[i,0]+=8
    if args.video:
        model.vis.global_.offwidth=1280; model.vis.global_.offheight=720
        renderer=mujoco.Renderer(model,height=720,width=1280)
        writer=RawVideoWriter(args.video,1280,720,30,total_frames=(round(args.duration/.002)+16)//17)
    if args.viewer:
        from mujoco import viewer as mj_viewer
        viewer=mj_viewer.launch_passive(model,data)
        viewer.cam.lookat[:]=camera.lookat; viewer.cam.distance=camera.distance
        viewer.cam.azimuth=camera.azimuth; viewer.cam.elevation=camera.elevation
    start=time.monotonic()
    try:
        for step in range(round(args.duration/.002)):
            if step%5==0:
                for line in lines:
                    line.update(model,data,events)
                    data.ctrl[line.rollers]=4.0 if line.state=='CONVEYOR_RUNNING' else 0.
                handoff=bool(lines) and all(line.handoffs>=4 for line in lines)
                for ctrl in arms+amrs: ctrl.update(model,data,events)
            mujoco.mj_step(model,data)
            if not np.isfinite(data.qpos).all(): raise RuntimeError('Nonfinite physics state')
            if renderer is not None and step%17==0:
                renderer.update_scene(data,camera=camera)
                frame=renderer.render()
                try:
                    from PIL import Image,ImageDraw
                    screen=Image.fromarray(frame); draw=ImageDraw.Draw(screen)
                    draw.rectangle((0,0,1280,76),fill=(15,22,31))
                    draw.text((18,12),f"FACTORY WORKFLOW | {config['type'].upper()} | {len(config['robots'])} ROBOTS | t={data.time:.1f}s",fill=(245,245,245))
                    status=' | '.join(f'{l.prefix or "LINE"}: {l.state}' for l in lines) or 'Independent controller commissioning'
                    draw.text((18,36),status,fill=(100,230,190))
                    draw.text((18,56),'Feedback control | assisted grasp constraints | same physical workpiece per production line',fill=(190,195,205))
                    draw.rectangle((0,695,1280,720),fill=(15,22,31))
                    draw.text((12,701),'Depot: Rafi Abdullah & Louise Poubel / Open Robotics, CC-BY-4.0; MuJoCo adaptation: roqsim; cutaway + equipment added',fill=(190,195,205))
                    frame=np.asarray(screen)
                except ImportError:
                    pass
                writer.write(frame)
            if viewer is not None and step%17==0:
                if not viewer.is_running(): break
                viewer.sync()
                time.sleep(max(0,data.time-(time.monotonic()-start)))
    finally:
        if writer: writer.close()
        if renderer: renderer.close()
        if viewer: viewer.close()
    report=dict(duration=data.time,grasp_mode=config['grasp_mode'],conveyor_handoff=handoff,
        production_state={line.prefix or 'main':line.state for line in lines},handoffs=sum(line.handoffs for line in lines),
        arms={a.name:dict(state=a.state,cycles=a.cycles,error_m=round(a.error,4),part=data.xpos[a.part].tolist(),
            ee=data.site_xpos[a.site].tolist(),joints=data.qpos[a.q].tolist(),command=data.ctrl[a.acts].tolist()) for a in arms},
        contacts=[(model.geom(c.geom1).name,model.geom(c.geom2).name) for c in data.contact],
        amrs={a.name:dict(arrivals=a.arrivals,distance_m=round(a.distance,3)) for a in amrs},events=events)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2))
    report['completion_pass']=bool(lines) and all(line.state=='SHIPPED' for line in lines)
    args.report.write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('events','contacts','arms')},indent=2))
    if args.require_complete and not report['completion_pass']:
        raise RuntimeError('Factory workflow incomplete; inspect the report for the blocked station')


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--config',type=Path,default=CONFIG)
    parser.add_argument('--duration',type=float,default=30)
    parser.add_argument('--scenario',choices=['metal','electronics','warehouse'])
    parser.add_argument('--cells',type=int,default=1)
    parser.add_argument('--video',type=Path)
    parser.add_argument('--viewer',action='store_true')
    parser.add_argument('--require-complete',action='store_true')
    parser.add_argument('--report',type=Path,default=Path('outputs/factory_report.json'))
    run(parser.parse_args())
