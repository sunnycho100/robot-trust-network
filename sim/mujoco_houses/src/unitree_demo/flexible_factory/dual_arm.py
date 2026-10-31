"""Temporary two-arm team sharing one physical panel after packing work.

Reuses measured IK and native servos. Both proximity-gated soft welds are
explicit assisted grasps; their constraint loads are measured independently.
"""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from ..factory_scene import UPSTREAM
from ..factory_executor import Arm
from ..industrial_showcase import box

PART='cooperative_panel'
BUDDY='panel_partner'
BUDDY_POS=(10.4,3.75,.65)
SOURCE=(9.55,4.65,.89)
DESTINATION=(9.55,4.20,.885)

def add_arm(spec,name,pos):
    folder=UPSTREAM/'roqsim_manipulation_assets/src/roqsim_manipulation_assets/models'
    robot=mujoco.MjSpec.from_file(str(folder/'ur10e/ur10e.xml'))
    grip=mujoco.MjSpec.from_file(str(folder/'robotiq_2f85/robotiq_2f85.xml'))
    for key in list(robot.keys): robot.delete(key)
    for key in list(grip.keys): grip.delete(key)
    robot.attach(grip,prefix='grip_',site=robot.site('tool_site') or robot.site('attachment_site'))
    spec.attach(robot,prefix=name+'_',frame=spec.worldbody.add_frame(pos=pos,quat=(.70710678,0,0,.70710678)))
    box(spec,name+'_pedestal',(pos[0],pos[1],pos[2]/2),(.16,.16,pos[2]/2),(.18,.25,.29,1))

def add_cooperative_panel(spec,include_leader=False):
    if include_leader: add_arm(spec,'packaging_arm',(9.,3.55,.65))
    add_arm(spec,BUDDY,BUDDY_POS)
    panel=spec.worldbody.add_body(name=PART,pos=SOURCE)
    panel.add_freejoint(name=PART+'_free')
    box(spec,PART+'_geom',(0,0,0),(.43,.12,.03),(.45,.54,.59,1),body=panel,mass=1.5)
    for side,x in enumerate((-.30,.30)):
        g=box(spec,PART+'_grasp_mark'+str(side),(x,0,.031),(.025,.10,.001),(.92,.68,.13,1),body=panel)
        g.contype=g.conaffinity=0
    for name,eq in [('packaging_arm','coop_lead_grasp'),(BUDDY,'coop_partner_grasp')]:
        spec.add_equality(name=eq,type=mujoco.mjtEq.mjEQ_WELD,objtype=mujoco.mjtObj.mjOBJ_BODY,
            name1=name+'_grip_base_mount',name2=PART,active=False,solref=(.04,1))
        for body in list(spec.bodies):
            if body.name.startswith(name+'_'): spec.add_exclude(bodyname1=body.name,bodyname2=PART)

class DualArmWork:
    def __init__(self,model,data):
        self.model=model; self.part=model.body(PART).id
        self.arms=[]
        for name,pos,offset,eq in [('packaging_arm',(9.,3.55,.65),(-.30,0,0),'coop_lead_grasp'),
                                   (BUDDY,BUDDY_POS,(.30,0,0),'coop_partner_grasp')]:
            arm=Arm(model,data,dict(name=name,model='ur10e',position=pos,part_name=PART,
                grasp_name=eq,grasp_offset=offset,max_joint_step=.20,minimum_lift_z=1.13,
                lift_clearance=.24,transfer_clearance=.25,tolerance=.03))
            arm.pipeline=True; arm.auto_transitions=False; arm.state='WAIT_INPUT'
            self.arms.append(arm)
        self.state='WAIT_FOR_PACKING'; self.events=[]; self.entered=0.; self.force=np.zeros(6)
        self.force_sums=np.zeros(2); self.force_samples=0; self.peak_forces=np.zeros(2)
        self.max_arm_penetration=0.; self.min_panel_z=10.
        self.support_force=0.; self.contacts=set()
        self.offset_world=[a.grasp_offset.copy() for a in self.arms]

    def transition(self,data,state,arm_state):
        self.state=state; self.entered=float(data.time)
        self.events.append(dict(t=float(data.time),event=state,position=data.xpos[self.part].tolist()))
        for arm in self.arms: arm.transition(data,self.events,arm_state)

    def update(self,model,data,ready):
        if self.state=='WAIT_FOR_PACKING':
            self.arms[1].update(model,data,self.events)
            if not ready: return
            for arm in self.arms:
                arm.source=data.xpos[self.part]+arm.grasp_offset
                arm.dest=np.array(DESTINATION)+arm.grasp_offset
            self.transition(data,'RENDEZVOUS','APPROACH')
        if self.state=='PLACE':
            for arm,offset in zip(self.arms,self.offset_world):
                arm.dest[2]=DESTINATION[2]+offset[2]-min(.02,.004*(data.time-self.entered))
        for arm in self.arms: arm.update(model,data,self.events)
        close=all(a.error<.035 for a in self.arms)
        if self.state=='RENDEZVOUS' and close:
            self.transition(data,'SYNCHRONIZED_GRASP','GRASP')
        elif self.state=='SYNCHRONIZED_GRASP' and close:
            # Freeze the measured grasp frames; both rigid grasps must describe
            # the same panel pose rather than fighting mismatched IK residuals.
            rotation=data.xmat[self.part].reshape(3,3)
            self.offset_world=[]
            for arm in self.arms:
                offset=data.site_xpos[arm.site]-data.xpos[self.part]
                arm.grasp_offset=rotation.T@offset
                arm.source=data.site_xpos[arm.site].copy()
                arm.dest=np.array(DESTINATION)+offset
                arm.desired_rotation=data.site_xmat[arm.site].reshape(3,3).copy()
                self.offset_world.append(offset.copy())
                arm.attach(model,data)
            self.transition(data,'LIFT','LIFT')
        elif self.state=='LIFT' and close:
            self.transition(data,'TRANSFER','TRANSFER')
        elif self.state=='TRANSFER' and close:
            self.transition(data,'PLACE','PLACE')
        elif self.state=='PLACE' and close and self.support_force>1.5*9.81*.7:
            for arm in self.arms: data.eq_active[arm.eq]=0
            self.transition(data,'RETRACT','RETRACT')
        elif self.state=='RETRACT' and close and data.time-self.entered>.4:
            self.transition(data,'COMPLETE','WAIT_INPUT')

    def observe(self,model,data):
        self.min_panel_z=min(self.min_panel_z,float(data.xpos[self.part,2]-.03))
        self.support_force=0.
        for i,c in enumerate(data.contact):
            b1,b2=int(model.geom_bodyid[c.geom1]),int(model.geom_bodyid[c.geom2])
            if self.part in (b1,b2):
                other=b2 if b1==self.part else b1
                if other==0:
                    mujoco.mj_contactForce(model,data,i,self.force)
                    self.support_force+=abs(float((c.frame.reshape(3,3).T@self.force[:3])[2]))
            names=[model.body(b).name for b in (b1,b2)]
            if any(n.startswith('packaging_arm_') for n in names) and any(n.startswith(BUDDY+'_') for n in names):
                self.max_arm_penetration=max(self.max_arm_penetration,max(0.,-float(c.dist)))
        if self.state in ('LIFT','TRANSFER') and all(data.eq_active[a.eq] for a in self.arms):
            measured=[]
            for arm in self.arms:
                rows=np.flatnonzero((data.efc_type==mujoco.mjtConstraint.mjCNSTR_EQUALITY)&(data.efc_id==arm.eq))
                measured.append(abs(float(data.efc_force[rows[2]])) if len(rows)>=3 else 0.)
            self.force_sums+=measured; self.force_samples+=1
            self.peak_forces=np.maximum(self.peak_forces,measured)

    def report(self,data):
        means=self.force_sums/max(1,self.force_samples)
        error=float(np.linalg.norm(data.xpos[self.part]-DESTINATION))
        return dict(state=self.state,events=self.events,part=data.xpos[self.part].tolist(),
            destination_error_m=error,minimum_panel_clearance_m=self.min_panel_z,
            mean_arm_vertical_load_N=means.tolist(),peak_arm_vertical_load_N=self.peak_forces.tolist(),
            max_arm_penetration_m=self.max_arm_penetration,assisted_grasps=True,
            pass_cooperation=bool(self.state=='COMPLETE' and error<.06 and min(means)>1.
                and self.min_panel_z>.8 and self.max_arm_penetration<.005))

def probe(args):
    spec=mujoco.MjSpec(); spec.option.timestep=.002
    spec.option.integrator=mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.worldbody.add_geom(name='floor',type=mujoco.mjtGeom.mjGEOM_PLANE,size=(20,20,.05))
    box(spec,'packaging_top',(9.3,4.5,.81),(.9,.45,.04),(.3,.4,.4,1))
    add_cooperative_panel(spec,include_leader=True)
    model=spec.compile(); data=mujoco.MjData(model)
    team=DualArmWork(model,data); mujoco.mj_forward(model,data)
    for step in range(round(args.duration/.002)):
        if step%5==0: team.update(model,data,True)
        mujoco.mj_step(model,data); team.observe(model,data)
    report=team.report(data)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--duration',type=float,default=25.)
    parser.add_argument('--report',type=Path,default=Path('outputs/dual_arm_probe.json'))
    probe(parser.parse_args())
