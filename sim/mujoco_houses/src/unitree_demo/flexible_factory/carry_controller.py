"""Contact-only measured transport/place/release/return controller.

Shares the existing G1 leg policies and upper-body motors. Initialization is a
held-load warmstart: this controller does not claim source rendezvous/pickup.
"""
import mujoco
import numpy as np
from ..g1_policy import gait_phase,yaw_from_quaternion

PROFILES={
    'team_fixture':dict(mass=3.,goal=(0.,.8),top=.94,center=(0.,.50),size=(.12,.22)),
    'bottom_crate':dict(mass=5.,goal=(.30,.30),top=.84,center=(.30,.30),size=(.12,.12)),
    'sofa_side':dict(mass=2.,goal=(.90,0.),top=.40,center=(.98,.06),size=(.16,.12)),
}

def add_delivery_stage(spec,scenario,offset=(0.,0.,0.),name='delivery_stand',profile=None):
    p={**PROFILES[scenario],**(profile or {})}
    return spec.worldbody.add_geom(name=name,type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=(p['center'][0]+offset[0],p['center'][1]+offset[1],p['top']/2),
        size=(*p['size'],p['top']/2),rgba=(.2,.4,.5,1),contype=8,conaffinity=7,friction=(2,.02,.002))

class CarryController:
    def __init__(self,model,data,group,stand_name='delivery_stand',active=True,profile=None):
        self.model=model; self.group=group; self.scenario=group['scenario']; self.profile={**PROFILES[self.scenario],**(profile or {})}
        self.legs=group['legs']; self.arms=group['arms']; self.actors=group['actors']; self.cfg=group['cfg']
        self.load=group['body']; self.stand=model.geom(stand_name).id
        self.goal=np.array(self.profile['goal'])+np.array(group.get('offset',(0,0,0)))[:2]
        self.original=[a.target_q.copy() for a in self.arms]
        self.dofs=[model.jnt_dofadr[model.actuator_trnid[a.actuators,0]] for a in self.arms]
        self.targets=[data.xpos[l.body_id,:2].copy() for l in self.legs]
        self.start_position=data.xpos[self.load].copy()
        self.initial_roots=[data.xpos[l.body_id,:2].copy() for l in self.legs]
        self.state='WARMSTART_GRASP' if active else 'WAIT_TASK'; self.started=data.time; self.entered=data.time
        self.events=[]; self.release_poses=None; self.max_penetration=0.; self.min_pelvis=10.
        self.hand_force=0.; self.stand_force=0.; self.floor_contacts=0; self.stable_placement=0
        self.force=np.zeros(6); self.minimum_floor_clearance=100.
        self.payload_geoms=[i for i in range(model.ngeom) if model.geom_bodyid[i]==self.load]

    def start(self,data):
        if self.state!='WAIT_TASK': return False
        self.started=data.time; self.transition(data,'WARMSTART_GRASP'); return True

    def transition(self,data,state):
        self.events.append(dict(t=data.time,event=state,payload=data.xpos[self.load].tolist()))
        self.state=state; self.entered=data.time

    def apply(self,data):
        for leg in self.legs: leg.apply(data)
        for i,arm in enumerate(self.arms):
            arm.apply(data)
            if self.scenario=='sofa_side': data.ctrl[arm.actuators]+=data.qfrc_bias[self.dofs[i]]

    def after_step(self,data,step):
        m=self.model; self.hand_force=self.stand_force=0.
        for i,c in enumerate(data.contact):
            b1,b2=m.geom_bodyid[c.geom1],m.geom_bodyid[c.geom2]
            if self.load not in (b1,b2): continue
            self.max_penetration=max(self.max_penetration,max(0.,-float(c.dist)))
            mujoco.mj_contactForce(m,data,i,self.force)
            vertical=abs(float((c.frame.reshape(3,3).T@self.force[:3])[2]))
            if self.stand in (c.geom1,c.geom2): self.stand_force+=vertical
            if m.geom(c.geom1).name=='floor' or m.geom(c.geom2).name=='floor': self.floor_contacts+=self.force[0]>0
            other=b2 if b1==self.load else b1
            if m.body(int(other)).name.endswith('wrist_roll_rubber_hand'): self.hand_force+=vertical
        for geom in self.payload_geoms:
            low=data.geom_xpos[geom,2]-np.abs(data.geom_xmat[geom].reshape(3,3)[2])@m.geom_size[geom]
            self.minimum_floor_clearance=min(self.minimum_floor_clearance,float(low))
        pelvis=min(float(data.xpos[l.body_id,2]) for l in self.legs); self.min_pelvis=min(self.min_pelvis,pelvis)
        if pelvis<.6 or self.minimum_floor_clearance<.1:
            if self.state!='FAULT': self.transition(data,'FAULT')
        if step%self.cfg.control_decimation: return
        weight=self.profile['mass']*9.81
        if self.stand_force>.8*weight and self.hand_force<.1*weight: self.stable_placement+=1
        else: self.stable_placement=0
        if self.state=='WARMSTART_GRASP' and data.time-self.started>1.5 and self.hand_force>.7*weight:
            self.transition(data,'TRANSIT')
        if self.state=='TRANSIT' and np.linalg.norm(data.xpos[self.load,:2]-self.goal)<.12:
            self.targets=[data.xpos[l.body_id,:2].copy() for l in self.legs]; self.transition(data,'PLACE')
        if self.state=='PLACE':
            for arm,pose in zip(self.arms,self.original):
                arm.target_q=pose.copy(); arm.target_q[[1,6]]+=min(.35,self.profile.get('lowering_rate',.025)*(data.time-self.entered))
            if self.stand_force>.7*weight:
                self.release_poses=[a.target_q.copy() for a in self.arms]; self.transition(data,'RELEASE')
        if self.state=='RELEASE':
            for arm,pose,released in zip(self.arms,self.original,self.release_poses):
                if self.scenario=='sofa_side':
                    arm.target_q=released.copy(); opening=min(.30,.12*(data.time-self.entered))
                    arm.target_q[2]+=opening; arm.target_q[7]-=opening
                else:
                    arm.target_q=pose.copy(); arm.target_q[[1,6]]+=.6
            if self.hand_force<.1*weight and data.time-self.entered>.5:
                self.targets=[data.xpos[l.body_id,:2]-.25*data.xmat[l.body_id].reshape(3,3)[:2,0] for l in self.legs]
                self.transition(data,'RETURN')
        if self.state=='RETURN' and self.stable_placement>=5 and all(np.linalg.norm(data.xpos[l.body_id,:2]-t)<.1 for l,t in zip(self.legs,self.targets)):
            self.transition(data,'AVAILABLE')
        for leg,(_,_,facing),target,initial_root in zip(self.legs,self.actors,self.targets,self.initial_roots):
            rot=data.xmat[leg.body_id].reshape(3,3)
            if self.state=='TRANSIT':
                speed=self.profile.get('speed',.08)
                velocity=rot[:2,:2].T@np.clip(.3*(self.goal-data.xpos[self.load,:2]),-speed,speed)
                if self.scenario!='sofa_side' and facing!=0: velocity[0]+=.04
                if self.profile.get('formation_tracking'):
                    travel=self.goal-self.start_position[:2]; length=np.linalg.norm(travel)
                    direction=travel/max(length,1e-9)
                    progress=min(length,speed*max(0.,data.time-self.started-1.5))
                    reference=initial_root+direction*progress
                    world=direction*(speed if progress<length else 0.)+.8*(reference-data.xpos[leg.body_id,:2])
                    velocity=rot[:2,:2].T@np.clip(world,-.20,.20)
                    if self.scenario=='bottom_crate': velocity[0]+=.03
            else:
                velocity=rot[:2,:2].T@np.clip(.8*(target-data.xpos[leg.body_id,:2]),-.15,.15)
            yaw0=0 if self.scenario=='sofa_side' or facing==1 else (np.pi if facing==-1 else np.pi/2)
            error=(yaw0-yaw_from_quaternion(data.xquat[leg.body_id])+np.pi)%(2*np.pi)-np.pi
            cmd=np.r_[velocity,np.clip(error,-.3,.3)].astype(np.float32)
            if self.state=='FAULT': cmd[:]=0
            leg.update_policy(data,cmd,gait_phase(data))

    def report(self,data):
        return dict(scenario=self.scenario,state=self.state,events=self.events,payload=data.xpos[self.load].tolist(),
            displacement_m=float(np.linalg.norm(data.xpos[self.load,:2]-self.start_position[:2])),minimum_pelvis_z=self.min_pelvis,
            max_penetration_m=self.max_penetration,warmstart=True,constraints=int(self.model.neq),
            floor_contacts=int(self.floor_contacts),final_stand_force_N=self.stand_force,minimum_floor_clearance_m=self.minimum_floor_clearance,
            pass_probe=bool(self.state=='AVAILABLE' and self.min_pelvis>.6 and self.max_penetration<.005 and self.floor_contacts==0 and self.stand_force>.8*self.profile['mass']*9.81))
