"""Finite physical work orders for the existing plant's four fixed cells.

Reuses Arm's measured IK/servo transitions and proximity-gated assisted grasp.
Parts exist at initialization and are never respawned or teleported. This is
cell commissioning, not yet factory-wide WIP dispatch.
"""
import numpy as np
import mujoco
from ..factory_executor import Arm
from ..industrial_showcase import box

ORDERS={
    'cnc_arm':dict(kind='ur5e',source=(-5.9,3.8,.89),
        steps=[('LOAD_MACHINE',(-5.5,4.08,1.0),1.5),('UNLOAD_MACHINE',(-5.9,3.9,.89),0.)]),
    'deburr_arm':dict(kind='ur5e',source=(-5.4,-3.6,.89),
        steps=[('LOAD_CLEANING_FIXTURE',(-5.,-3.45,.89),.7),('CLEANED_OUTPUT',(-4.6,-3.6,.89),0.)]),
    'assembly_arm':dict(kind='panda',source=(1.05,-4.4,.97),
        steps=[('INSERT_COMPONENT_A',(1.5,-4.4,.97),.4),
               ('INSERT_COMPONENT_B',(1.5,-4.4,1.04),0.)],
        second_source=(1.95,-4.4,.97)),
    'packaging_arm':dict(kind='ur10e',source=(9.5,4.35,.90),
        steps=[('PALLET_POSITION_1',(8.7,4.35,.90),.2),
               ('PALLET_POSITION_2',(8.7,4.55,.90),0.)],
        second_source=(9.85,4.35,.90)),
}

def add_workpieces(spec):
    box(spec,'cnc_service_table',(-5.5,3.8,.81),(.66,.18,.04),(.34,.39,.42,1))
    for name,order in ORDERS.items():
        positions=[order['source']]
        if 'second_source' in order: positions.append(order['second_source'])
        for index,pos in enumerate(positions):
            partname=name+'_part'+('' if index==0 else str(index))
            part=spec.worldbody.add_body(name=partname,pos=pos)
            part.add_freejoint(name=partname+'_free')
            size=(.045,.04,.04) if name=='packaging_arm' else (.03,.03,.035)
            box(spec,partname+'_geom',(0,0,0),size,
                (.69,.49,.29,1) if name=='packaging_arm' else (.65,.7,.73,1),body=part,mass=.12)
            for body in list(spec.bodies):
                if body.name.startswith(name+'_') and '_part' not in body.name:
                    spec.add_exclude(bodyname1=body.name,bodyname2=partname)
        spec.add_equality(name=name+'_grasp',type=mujoco.mjtEq.mjEQ_WELD,
            objtype=mujoco.mjtObj.mjOBJ_BODY,name1=name+('_hand' if order['kind']=='panda' else '_grip_base_mount'),
            name2=name+'_part',active=False,solref=(.02,1))

class CellWork:
    def __init__(self,model,data,name,position):
        self.name=name; self.order=ORDERS[name]; self.index=0; self.completed=[]
        self.arm=Arm(model,data,dict(name=name,model=self.order['kind'],position=position,
            max_joint_step=.15 if name=='deburr_arm' else .22,minimum_lift_z=1.05 if name=='cnc_arm' else 1.15,
            lift_clearance=.10 if name=='cnc_arm' else .18,
            transfer_clearance=.08 if name=='cnc_arm' else .18))
        self.arm.pipeline=True
        self.part_ids=[model.body(name+'_part').id]
        if 'second_source' in self.order: self.part_ids.append(model.body(name+'_part1').id)
        self.wait_until=None; self.state='STARTING'; self.start(data,[])

    def start(self,data,events):
        a=self.arm; step=self.order['steps'][self.index]
        self.state=step[0]
        a.part=self.part_ids[min(self.index,len(self.part_ids)-1)]
        a.source=data.xpos[a.part].copy(); a.dest=np.array(step[1])
        a.transition(data,events,'APPROACH')

    def update(self,model,data,events):
        a=self.arm
        a.update(model,data,events)
        if self.state=='COMPLETE': return
        if a.state=='DONE':
            if self.wait_until is None:
                self.wait_until=data.time+self.order['steps'][self.index][2]
                self.completed.append(dict(operation=self.state,t=float(data.time),part=int(a.part)))
            if data.time>=self.wait_until:
                self.index+=1; self.wait_until=None
                if self.index==len(self.order['steps']): self.state='COMPLETE'
                else: self.start(data,events)

    def report(self,data):
        final_targets={self.part_ids[min(i,len(self.part_ids)-1)]:np.array(step[1])
                       for i,step in enumerate(self.order['steps'])}
        placement_errors={str(p):float(np.linalg.norm(data.xpos[p]-target))
                          for p,target in final_targets.items()}
        return dict(operation=self.state,motion=self.arm.state,error_m=self.arm.error,
            completed=self.completed,parts={str(p):data.xpos[p].tolist() for p in self.part_ids},
            placement_errors_m=placement_errors,
            pass_work_order=bool(self.state=='COMPLETE' and all(e<.07 for e in placement_errors.values())),
            assisted_grasp=True)
