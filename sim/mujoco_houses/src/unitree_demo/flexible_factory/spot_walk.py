"""Contact-force crawl controller for the existing free-base Spot model.

Only joint actuators are commanded: no forces on the base, welds, or runtime
root/joint position writes. Three stance feet support each swing phase.
"""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from .quadrupeds import SpotStand,add_quadrupeds

def skew(v):
    x,y,z=v
    return np.array([[0,-z,y],[z,0,-x],[-y,x,0.]])

def support_target(point,feet):
    """Closest point in a slightly inset support triangle."""
    center=feet.mean(axis=0); vertices=center+.88*(feet-center)
    signs=[]; candidates=[]
    for i in range(3):
        a=vertices[i]; b=vertices[(i+1)%3]; edge=b-a
        signs.append(edge[0]*(point-a)[1]-edge[1]*(point-a)[0])
        t=np.clip(np.dot(point-a,edge)/max(np.dot(edge,edge),1e-9),0,1)
        candidates.append(a+t*edge)
    if min(signs)>=0 or max(signs)<=0: return point
    return min(candidates,key=lambda p:np.linalg.norm(point-p))

class SpotWalk(SpotStand):
    def __init__(self,model,data,name,distance=1.,pace='brisk'):
        super().__init__(model,data,name)
        self.foot_ids=[model.geom(name+'_'+n).id for n in ('FL','FR','HL','HR')]
        mujoco.mj_forward(model,data)
        self.mass=model.body_subtreemass[self.body]
        self.initial=data.xpos[self.body].copy()
        self.rotation=data.xmat[self.body].reshape(3,3).copy()
        self.forward=self.rotation[:,0].copy(); self.forward[2]=0
        self.distance=distance; self.mode='SETTLE'; self.entered=0.
        self.anchors=data.geom_xpos[self.foot_ids].copy()
        self.order=[0,3,1,2]; self.step_index=0; self.swing=0
        self.steps=0; self.airborne_samples=0; self.max_foot_lift=0.
        self.jac=np.zeros((3,model.nv)); self.jacr=self.jac.copy()
        self.com_jac=self.jac.copy(); self.target_com=None
        self.swing_start=None; self.swing_goal=None; self.command=self.pose.copy()
        self.last_update=-1.; self.fault=False
        self.pace=pace
        self.stride=.20 if pace=='brisk' else .10
        self.shift_time=.30 if pace=='brisk' else 1.2
        self.swing_time=.40 if pace=='brisk' else .8
        self.land_time=.12 if pace=='brisk' else .35

    def change(self,data,mode):
        self.mode=mode; self.entered=float(data.time)

    def apply(self,data):
        if data.time<1.5:
            super().apply(data); return
        if data.time-self.last_update<.009:
            data.ctrl[self.acts]=self.command; return
        self.last_update=float(data.time)
        m=self.model
        feet=data.geom_xpos[self.foot_ids].copy()
        com=data.subtree_com[self.body].copy()
        if self.mode=='SETTLE':
            self.anchors=feet.copy(); self.height=float(com[2]); self.change(data,'SHIFT')
        progress=float((data.xpos[self.body]-self.initial)@self.forward)
        if self.mode=='SHIFT' and self.target_com is None:
            self.swing=self.order[self.step_index%4]
            support=[i for i in range(4) if i!=self.swing]
            self.target_com=self.anchors[support].mean(axis=0)
            if self.pace=='brisk':
                desired=self.anchors.mean(axis=0)+self.forward*.07
                self.target_com[:2]=support_target(desired[:2],self.anchors[support,:2])
            self.target_com[2]=self.height
        age=data.time-self.entered
        if self.mode=='SHIFT' and age>self.shift_time and np.linalg.norm(com[:2]-self.target_com[:2])<(.045 if self.pace=='brisk' else .035):
            self.swing_start=feet[self.swing].copy()
            self.swing_goal=self.anchors[self.swing]+self.forward*self.stride
            self.swing_goal[2]=.036
            self.change(data,'SWING')
        if self.mode=='SWING' and data.time-self.entered>self.swing_time:
            self.anchors[self.swing]=self.swing_goal
            self.change(data,'LAND')
        if self.mode=='LAND' and data.time-self.entered>self.land_time:
            self.steps+=1; self.step_index+=1; self.target_com=None
            self.change(data,'PARK' if progress>=self.distance else 'SHIFT')
        stance=[i for i in range(4) if self.mode!='SWING' or i!=self.swing]
        target=self.target_com if self.target_com is not None else self.anchors.mean(axis=0)
        target=target.copy(); target[2]=self.height
        mujoco.mj_jacSubtreeCom(m,data,self.com_jac,self.body)
        velocity=self.com_jac@data.qvel
        acc=np.array([35.,35.,90.] if self.pace=='brisk' else [12.,12.,90.])*(target-com)-np.array([10.,10.,18.] if self.pace=='brisk' else [7.,7.,18.])*velocity
        force=self.mass*(acc+np.array([0,0,9.81]))
        force[:2]=np.clip(force[:2],-.4*self.mass*9.81,.4*self.mass*9.81)
        force[2]=np.clip(force[2],.2*self.mass*9.81,2*self.mass*9.81)
        rot=data.xmat[self.body].reshape(3,3)
        error=.5*sum((np.cross(rot[:,i],self.rotation[:,i]) for i in range(3)))
        angular=np.zeros(6)
        mujoco.mj_objectVelocity(m,data,mujoco.mjtObj.mjOBJ_BODY,self.body,angular,0)
        torque=90*error-12*angular[:3]
        matrix=np.concatenate([np.vstack([np.eye(3),skew(feet[i]-com)]) for i in stance],axis=1)
        wrench=np.r_[force,torque]
        forces=(matrix.T@np.linalg.solve(matrix@matrix.T+.001*np.eye(6),wrench)).reshape(-1,3)
        tau=np.zeros(12)
        for i,f in zip(stance,forces):
            f[2]=np.clip(f[2],0,250)
            lateral=np.linalg.norm(f[:2]); limit=1.2*f[2]
            if lateral>limit: f[:2]*=limit/max(lateral,1e-9)
            mujoco.mj_jacGeom(m,data,self.jac,self.jacr,self.foot_ids[i])
            j=self.jac[:,self.dofs]
            correction=60*(self.anchors[i]-feet[i])-8*(self.jac@data.qvel)
            tau+=j.T@(-f+correction)
        if self.mode=='SWING':
            phase=np.clip((data.time-self.entered)/self.swing_time,0,1)
            smooth=phase*phase*(3-2*phase)
            goal=self.swing_start+(self.swing_goal-self.swing_start)*smooth
            goal[2]+=.075*np.sin(np.pi*phase)
            mujoco.mj_jacGeom(m,data,self.jac,self.jacr,self.foot_ids[self.swing])
            tau+=self.jac[:,self.dofs].T@(600*(goal-feet[self.swing])-25*(self.jac@data.qvel))
            lift=float(feet[self.swing,2]-.036)
            self.max_foot_lift=max(self.max_foot_lift,lift)
            self.airborne_samples+=lift>.025
        q=data.qpos[m.jnt_qposadr[m.actuator_trnid[self.acts,0]]]
        data.qfrc_applied[self.dofs]=data.qfrc_bias[self.dofs]
        self.command=q+(tau+12*data.qvel[self.dofs])/180.
        self.command=np.clip(self.command,m.actuator_ctrlrange[self.acts,0],m.actuator_ctrlrange[self.acts,1])
        data.ctrl[self.acts]=self.command

    def report(self,data):
        report=super().report(data)
        progress=float((data.xpos[self.body]-self.initial)@self.forward)
        report.update(mode=self.mode,pace=self.pace,steps=self.steps,progress_m=progress,
            max_foot_lift_m=self.max_foot_lift,airborne_samples=self.airborne_samples,
            pass_walking=bool(self.minimum_height>.3 and self.minimum_up>.85
                and self.base_ground_contacts==0 and progress>.15 and self.steps>=4
                and self.airborne_samples>10))
        return report

def probe(args):
    spec=mujoco.MjSpec()
    spec.option.timestep=.002
    spec.option.integrator=mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.worldbody.add_geom(name='floor',type=mujoco.mjtGeom.mjGEOM_PLANE,size=(10,10,.05),friction=(1.2,.02,.002))
    add_quadrupeds(spec,[dict(name='dog',position=[0,0])])
    model=spec.compile(); data=mujoco.MjData(model)
    dog=SpotWalk(model,data,'dog',args.distance,pace=args.pace)
    for step in range(round(args.duration/.002)):
        dog.apply(data); mujoco.mj_step(model,data); dog.observe(data)
    report=dog.report(data)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--duration',type=float,default=20.)
    parser.add_argument('--distance',type=float,default=.5)
    parser.add_argument('--pace',choices=['brisk','crawl'],default='brisk')
    parser.add_argument('--report',type=Path,default=Path('outputs/spot_walk_probe.json'))
    probe(parser.parse_args())
