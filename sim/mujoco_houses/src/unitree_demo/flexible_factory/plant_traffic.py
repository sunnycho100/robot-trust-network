"""Measured wheel control for distinct, finite cross-section delivery missions.

Routes are commissioning waypoints, not a global path planner. Local yielding
uses measured mobile and humanoid positions. No robot pose replay is used.
"""
import mujoco
import numpy as np

MISSIONS=[
    ('QC_PRESENTATION',[(3.65,3.05)]),
    ('RECEIVING_TO_CNC',[(-11.,1.0),(-6.3,1.0),(-6.3,2.95)]),
    ('ASSEMBLY_REPLENISHMENT',[(0.,1.9),(-1.9,1.9),(-1.9,4.5),(-1.9,1.0),(1.4,1.0),(1.4,-3.6)]),
    ('QC_TO_REWORK',[(2.9,1.0),(-3.,1.0),(-3.,-1.7),(-3.8,-1.7),(-3.8,-2.4)]),
    ('BULK_RAW_SUPPLY',[(-7.8,1.9),(-7.8,3.25)]),
    ('SHIPPING_TO_PACKAGING',[(10.,-2.6),(9.,-2.6),(7.,-1.4),(7.,1.9),(11.,1.9),(11.,3.1)]),
]

class Delivery:
    def __init__(self,model,index,kind):
        self.index=index; self.name=f'tb{index}'; self.kind=kind
        self.mission,points=MISSIONS[index]; self.points=np.array(points); self.waypoint=0
        self.body=model.body(self.name+'_base_link').id
        husky=kind=='husky_a200'
        jackal=kind=='clearpath_jackal'
        wheels=('front_left','rear_left','front_right','rear_right') if husky or jackal else ('left','right')
        self.acts=[model.actuator(self.name+'_'+w+'_wheel_motor').id for w in wheels]
        self.radius=.1651 if husky else (.098 if jackal else .033)
        self.half_track=.2854 if husky else (.187795 if jackal else .144)
        self.width=.36 if husky else (.23 if jackal else .19)
        self.length=.55 if husky else (.30 if jackal else .22)
        self.speed=.55 if husky else (.60 if jackal else .25)
        self.slip_factor=1.7 if jackal else (3. if husky else 1.)
        self.arrived=False; self.yielding=False; self.distance=0.; self.last=None
        self.started=None; self.completed_at=None; self.hold_count=0

    def update(self,model,data,others,humanoids,events):
        pos=data.xpos[self.body,:2]; rot=data.xmat[self.body].reshape(3,3)
        if self.last is not None: self.distance+=float(np.linalg.norm(pos-self.last))
        self.last=pos.copy()
        if self.arrived:
            data.ctrl[self.acts]=0; return
        if self.started is None: self.started=float(data.time)
        delta=self.points[self.waypoint]-pos
        if np.linalg.norm(delta)<(.13 if self.kind=='husky_a200' else .045):
            events.append(dict(t=float(data.time),robot=self.name,mission=self.mission,event='DELIVERY_WAYPOINT',waypoint=self.waypoint))
            self.waypoint+=1
            if self.waypoint==len(self.points):
                self.arrived=True; self.completed_at=float(data.time); data.ctrl[self.acts]=0; return
            delta=self.points[self.waypoint]-pos
        yaw=np.arctan2(rot[1,0],rot[0,0])
        error=(np.arctan2(delta[1],delta[0])-yaw+np.pi)%(2*np.pi)-np.pi
        v=min(self.speed,1.3*float(np.linalg.norm(delta)))*max(0.,np.cos(error))
        # Turn before translating at cell entrances; skid-steer corner cutting
        # otherwise pushes the Husky into the edge of adjacent staging bays.
        if abs(error)>.20: v=0.
        w=float(np.clip(2*error,-1.5,1.5))
        self.yielding=False
        # An occupied path ahead inhibits translation, including a parked peer.
        obstacles=[(data.xpos[o.body,:2],o.length,o.width) for o in others if o is not self]
        obstacles.extend((data.xpos[b,:2],.25,.25) for b in humanoids)
        for position,length,width in obstacles:
            local=rot[:2,:2].T@(position-pos)
            if 0<local[0]<self.length+length+.22 and abs(local[1])<self.width+width+.10:
                v=0.; self.yielding=True; self.hold_count+=1; break
        left=(v-self.half_track*w*self.slip_factor)/self.radius
        right=(v+self.half_track*w*self.slip_factor)/self.radius
        commands=np.array([left,left,right,right] if len(self.acts)==4 else [left,right])
        limits=np.max(np.abs(model.actuator_ctrlrange[self.acts]),axis=1)
        commands/=max(1.,float(np.max(np.abs(commands)/limits)))
        data.ctrl[self.acts]=commands

    def report(self,data):
        return dict(model=self.kind,mission=self.mission,arrived=self.arrived,
            waypoint=self.waypoint,total_waypoints=len(self.points),distance_m=self.distance,
            yielding=self.yielding,position=data.xpos[self.body].tolist(),completed_at=self.completed_at)

class PlantTraffic:
    def __init__(self,model,groups,layout):
        self.robots=[Delivery(model,i,kind) for i,kind in enumerate(layout['courier_models'])]
        self.humanoids=[leg.body_id for group in groups for leg in group['legs']]
        # Map every robot link to its resource, to measure inter-robot contacts.
        self.owner={}
        prefixes=[r.name+'_' for r in self.robots]
        prefixes.extend(g['name']+'_'+p for g in groups for p,_,_ in g['actors'])
        for b in range(1,model.nbody):
            for prefix in prefixes:
                if model.body(b).name.startswith(prefix): self.owner[b]=prefix; break
        self.contact_pairs=set(); self.max_inter_robot_penetration=0.
        self.obstacle_contacts=set()

    def update(self,model,data,events):
        for robot in self.robots: robot.update(model,data,self.robots,self.humanoids,events)

    def observe(self,model,data):
        for c in data.contact:
            a=self.owner.get(int(model.geom_bodyid[c.geom1])); b=self.owner.get(int(model.geom_bodyid[c.geom2]))
            if a and b and a!=b and (a.startswith('tb') or b.startswith('tb')) and c.dist<0:
                self.contact_pairs.add(tuple(sorted((a,b))))
                self.max_inter_robot_penetration=max(self.max_inter_robot_penetration,-float(c.dist))
            for owner,other,geom in ((a,b,c.geom2),(b,a,c.geom1)):
                if owner and owner.startswith('tb') and other is None and c.dist<-.001:
                    name=model.geom(geom).name
                    cargo={'tb0_':'part0','tb1_':'part1'}.get(owner)
                    if 'floor' not in name.lower() and name!=cargo:
                        self.obstacle_contacts.add((owner,name))
