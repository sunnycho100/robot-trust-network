"""Free-base Spot instances with a quiet joint-servo standing controller.

Uses the existing BSD-3 Menagerie model packaged by roqsim. No external policy
weights are downloaded. SpotWalk adds contact-force crawl control separately.
"""
import math
import mujoco
import numpy as np
from ..factory_scene import UPSTREAM

MODEL=UPSTREAM/'roqsim_quadruped/src/roqsim_quadruped/models/spot.xml'

def add_quadrupeds(spec,placements):
    for row in placements:
        robot=mujoco.MjSpec.from_file(str(MODEL))
        for key in list(robot.keys): robot.delete(key)
        for act in robot.actuators:
            act.gainprm[0]=180.
            act.biasprm[1]=-180.; act.biasprm[2]=-12.
        yaw=row.get('yaw',0.)
        spec.attach(robot,prefix=row['name']+'_',frame=spec.worldbody.add_frame(
            pos=(*row['position'],0),quat=(math.cos(yaw/2),0,0,math.sin(yaw/2))))

class SpotStand:
    def __init__(self,model,data,name):
        self.model=model; self.name=name; self.body=model.body(name+'_base_link').id
        names=[leg+'_'+joint for leg in ('fl','fr','hl','hr') for joint in ('hx','hy','kn')]
        self.acts=np.array([model.actuator(name+'_'+n).id for n in names])
        joints=np.array([model.joint(name+'_'+n).id for n in names])
        self.dofs=model.jnt_dofadr[joints]
        self.pose=np.array([v for leg in ('fl','fr','hl','hr') for v in
                            ((.1 if leg.endswith('l') else -.1),.8,-1.6)])
        data.qpos[model.jnt_qposadr[joints]]=self.pose
        data.ctrl[self.acts]=self.pose
        self.minimum_height=float('inf'); self.minimum_up=1.
        self.feet={model.geom(name+'_'+foot).id for foot in ('FL','FR','HL','HR')}
        self.supporting_feet=set(); self.base_ground_contacts=0

    def apply(self,data):
        data.ctrl[self.acts]=self.pose
        data.qfrc_applied[self.dofs]=data.qfrc_bias[self.dofs]

    def observe(self,data):
        self.minimum_height=min(self.minimum_height,float(data.xpos[self.body,2]))
        self.minimum_up=min(self.minimum_up,float(data.xmat[self.body].reshape(3,3)[2,2]))
        self.supporting_feet=set()
        for c in data.contact:
            if c.dist>0: continue
            for geom,other in ((c.geom1,c.geom2),(c.geom2,c.geom1)):
                if 'floor' not in self.model.geom(other).name.lower(): continue
                if geom in self.feet: self.supporting_feet.add(int(geom))
                elif self.model.geom_bodyid[geom]==self.body: self.base_ground_contacts+=1

    def report(self,data):
        return dict(model='boston_dynamics_spot',mode='STANDING',position=data.xpos[self.body].tolist(),
            minimum_base_height_m=self.minimum_height,minimum_up=self.minimum_up,
            supporting_feet=len(self.supporting_feet),base_ground_contacts=self.base_ground_contacts,
            pass_standing=bool(self.minimum_height>.30 and self.minimum_up>.9
                               and len(self.supporting_feet)>=3 and self.base_ground_contacts==0))
