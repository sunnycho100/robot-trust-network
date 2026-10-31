"""Contact-only G1 logistics prototypes: bottom-supported crate or solo sofa module."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from .coop_lift import demo_config, RawVideoWriter
from .g1_policy import G1Controller, G1UpperBodyController, gait_phase, yaw_from_quaternion


def build_group(args,scene_hook=None):
    """Prepare a reproducible contact scene for standalone or shared-world use."""
    cfg=demo_config()
    spec=mujoco.MjSpec.from_string('''<mujoco><option timestep="0.002" integrator="implicitfast" cone="elliptic"/>
      <visual><global offwidth="1280" offheight="720"/><headlight ambient="0.4 0.4 0.4"/></visual>
      <worldbody><light pos="0 -2 4"/><geom name="floor" type="plane" size="0 0 .05"
      rgba=".22 .25 .28 1" contype="9" conaffinity="9" friction="1.2 .02 .002"/></worldbody></mujoco>''')
    sofa=args.scenario in ('solo_sofa','sofa_underneath','sofa_side')
    solo=sofa
    side_grip=args.scenario in ('solo_sofa','sofa_side')
    fixture=args.scenario=='team_fixture'
    actors=[('carrier_',0.,1)] if solo else [('left_',-.857,1),('right_',.857,-1)]
    if fixture and args.carriers==3: actors.append(('rear_',0.,0))
    robot_xml=getattr(cfg,'robot_xml_23dof',cfg.robot_xml)
    for prefix,x,facing in actors:
        spec.attach(mujoco.MjSpec.from_file(str(robot_xml)),prefix=prefix,
            frame=spec.worldbody.add_frame(pos=(x,-.9 if facing==0 else 0,0),
                quat=(1,0,0,0) if facing==1 else ((0,0,0,1) if facing==-1 else (.70710678,0,0,.70710678))))
    payload=spec.worldbody.add_body(name='payload',pos=(0,0,1.25))
    payload.add_freejoint(name='payload_free')
    seat=payload.add_geom(name='load',type=mujoco.mjtGeom.mjGEOM_BOX,size=(.55,.32,.12),
        mass=args.mass,rgba=(.88,.48,.08,1),contype=4,conaffinity=10,friction=(3,.05,.005))
    half_height=.12
    if fixture:
        half_height=.08
        seat.size=(.56,.28,.08); seat.pos=(0,.05,0); seat.mass=args.mass*.25
        seat.rgba=(.35,.42,.52,1)
        seat.conaffinity=11
        payload.add_geom(name='fixture_stem',type=mujoco.mjtGeom.mjGEOM_BOX,pos=(0,-.46,0),
            size=(.26,.23,.08),mass=args.mass*.75,rgba=(.85,.52,.12,1),contype=4,conaffinity=11,
            friction=(3,.05,.005))
    if sofa:
        half_height=.09
        width=.215 if args.scenario=='sofa_side' else .38
        seat.size=(.14,width,.09); seat.mass=args.mass*.9; seat.rgba=(.14,.55,.62,1)
        # The back starts at the seat top: no overlapping component volumes.
        payload.add_geom(name='sofa_back',type=mujoco.mjtGeom.mjGEOM_BOX,pos=(-.10,0,.22),
            size=(.04,width,.13),mass=args.mass*.1,rgba=(.12,.42,.49,1),contype=4,conaffinity=11)
        seat.conaffinity=11
        if args.scenario=='sofa_side':
            seat.priority=1; seat.solref=(.005,1); seat.condim=4
            back=spec.geom('sofa_back'); back.priority=1; back.solref=(.005,1); back.condim=4
    if scene_hook is not None: scene_hook(spec)
    model=spec.compile(); data=mujoco.MjData(model)
    legs=[]; arms=[]
    for prefix,_,_ in actors:
        leg=G1Controller(model,cfg,prefix); leg.reset(data); legs.append(leg)
        arm=G1UpperBodyController(model,prefix); arm.set_carry_pose()
        if side_grip: arm.target_q[[2,7]]=[.65,-.65]
        if args.scenario=='sofa_underneath':
            arm.target_q[[1,6]]=[-.8,-.8]
            arm.target_q[[4,9]]=[1.8,1.8]
        if args.scenario=='sofa_side':
            arm.target_q[[1,6]]=[-1.2,-1.2]
            arm.target_q[[4,9]]=[1.8,1.8]
            arm.target_q[[2,7]]=[.30,-.30]
            arm.target_q[[5,10]]=[1.57,-1.57]
        data.qpos[arm.joint_qpos]=arm.target_q; arms.append(arm)
    mujoco.mj_forward(model,data)
    palms=[model.geom(prefix+side+'_palm_pad').id for prefix,_,_ in actors for side in ('left','right')]
    centers=np.array([data.geom_xpos[i] for i in palms])
    p=model.jnt_qposadr[model.joint('payload_free').id]
    if side_grip:
        # Match the seat to the fixed hand separation, not an invisible grasp anchor.
        if args.scenario!='sofa_side':
            model.geom_size[model.geom('load').id,1]=np.mean(np.abs(centers[:,1]))-.010
        model.geom_size[model.geom('sofa_back').id,1]=model.geom_size[model.geom('load').id,1]
        data.qpos[p:p+3]=centers.mean(axis=0)
        if args.scenario=='sofa_side': data.qpos[p]+=.03
        desired=np.array([[0,0,1],[1,0,0],[0,1,0]],dtype=float)
        for i in palms:
            body=model.geom_bodyid[i]
            quat=np.zeros(4)
            mujoco.mju_mat2Quat(quat,(data.xmat[body].reshape(3,3).T@desired).ravel())
            model.geom_quat[i]=quat
        if args.scenario!='sofa_side': arms[0].target_q[[2,7]]=[.50,-.50]
    else:
        # Rotate each thin palm pad so its contact normal points upward at reset.
        desired=np.array([[0,0,-1],[0,1,0],[1,0,0]],dtype=float)
        for i in palms:
            body=model.geom_bodyid[i]
            local=data.xmat[body].reshape(3,3).T@desired
            quat=np.zeros(4); mujoco.mju_mat2Quat(quat,local.ravel())
            model.geom_quat[i]=quat
        center=centers.mean(axis=0) if solo else np.array([0.,0.,centers[:,2].mean()])
        data.qpos[p:p+3]=center+np.array([0,0,.012+half_height])
        if fixture:
            # Start above the visible hand meshes, not intersecting their convex hulls.
            top=centers[:,2].max()+.012
            for i in range(model.ngeom):
                if model.body(int(model.geom_bodyid[i])).name.endswith('wrist_roll_rubber_hand') and model.geom_type[i]==mujoco.mjtGeom.mjGEOM_MESH:
                    mid=model.geom_dataid[i]; first=model.mesh_vertadr[mid]; count=model.mesh_vertnum[mid]
                    vertices=model.mesh_vert[first:first+count]@data.geom_xmat[i].reshape(3,3).T+data.geom_xpos[i]
                    top=max(top,float(vertices[:,2].max())+.003)
            data.qpos[p+2]=top+half_height
    mujoco.mj_forward(model,data)
    if args.scenario=='sofa_side':
        hand_meshes=[i for i in range(model.ngeom) if model.geom_type[i]==mujoco.mjtGeom.mjGEOM_MESH and model.body(int(model.geom_bodyid[i])).name.endswith('wrist_roll_rubber_hand')]
        load_geoms=[model.geom('load').id,model.geom('sofa_back').id]
        lo,hi=.30,.85
        for _ in range(20):
            roll=(lo+hi)/2
            arms[0].target_q[[2,7]]=[roll,-roll]
            data.qpos[arms[0].joint_qpos]=arms[0].target_q
            mujoco.mj_forward(model,data)
            distance=min(mujoco.mj_geomDistance(model,data,h,g,.2,None) for h in hand_meshes for g in load_geoms)
            if distance<.0005: lo=roll
            else: hi=roll
        arms[0].target_q[[2,7]]=[roll-.08,-roll+.08]
    # Runtime palm alignment must also travel with an attached scene spec.
    for geom in spec.geoms:
        index=model.geom(geom.name).id if geom.name else -1
        if index>=0:
            geom.quat=model.geom_quat[index].copy()
            geom.size=model.geom_size[index].copy()
    return spec,model,data,legs,arms,actors,palms,cfg,half_height


def run(args):
    _,model,data,legs,arms,actors,palms,cfg,half_height=build_group(args)
    solo=len(actors)==1
    start=data.body('payload').xpos.copy(); min_z=float(start[2])
    upper_dofs=model.jnt_dofadr[model.actuator_trnid[arms[0].actuators,0]]
    jacp=np.zeros((3,model.nv)); jacr=jacp.copy()
    if args.scenario=='sofa_underneath' and args.furniture_feedback:
        arms[0].kp[1:]=280
    renderer=writer=None
    steps=round(args.duration/cfg.simulation_dt)
    if args.video:
        renderer=mujoco.Renderer(model,height=720,width=1280)
        writer=RawVideoWriter(args.video,1280,720,30,total_frames=(steps+16)//17)
    force=np.zeros(6); supported_steps=0; vertical_force=0.; bottom_contacts=0; hand_contacts=0
    contact_bodies={}; worst_penetration=0.
    side_faces={'left_A':0,'right_B':0,'other':0}
    load_id=model.body('payload').id
    payload_geoms=[i for i in range(model.ngeom) if model.geom_bodyid[i]==load_id]
    minimum_floor_clearance=float('inf')
    try:
        for step in range(steps):
            if args.scenario=='sofa_side' and args.furniture_feedback and step%5==0:
                rows=[]; errors=[]
                for side,i in zip((1,-1),palms):
                    offset=np.array([-.03,side*(model.geom_size[model.geom('load').id,1]+.006),0.])
                    target=data.body('payload').xpos+data.body('payload').xmat.reshape(3,3)@offset
                    mujoco.mj_jacGeom(model,data,jacp,jacr,i)
                    rows.append(jacp[:,upper_dofs].copy()); errors.append(target-data.geom_xpos[i])
                jac=np.vstack(rows); jac[:,[0,5,10]]=0
                delta=jac.T@np.linalg.solve(jac@jac.T+.001*np.eye(6),np.concatenate(errors))
                arms[0].target_q=data.qpos[arms[0].joint_qpos]+np.clip(delta,-.08,.08)
                arms[0].target_q[[5,10]]=[1.57,-1.57]
                data.qfrc_applied[upper_dofs]=data.qfrc_bias[upper_dofs]
            if args.scenario=='sofa_underneath' and args.furniture_feedback and step%5==0:
                rows=[]; errors=[]
                for side,i in zip((1,-1),palms):
                    # Track the actual load horizontally; keep a supported carrying
                    # height above the measured pelvis, using joint torques only.
                    target=np.array([data.body('payload').xpos[0],
                        data.body('payload').xpos[1]+side*.214,
                        data.body('carrier_pelvis').xpos[2]+.32])
                    mujoco.mj_jacGeom(model,data,jacp,jacr,i)
                    normal=data.geom_xmat[i].reshape(3,3)[:,0]
                    rows.extend([jacp[:,upper_dofs].copy(),.15*jacr[:,upper_dofs].copy()])
                    errors.extend([target-data.geom_xpos[i],.15*np.cross(normal,np.array([0,0,1]))])
                jac=np.vstack(rows); jac[:,0]=0
                err=np.concatenate(errors)
                delta=jac.T@np.linalg.solve(jac@jac.T+.002*np.eye(12),err)
                arms[0].target_q=data.qpos[arms[0].joint_qpos]+np.clip(delta,-.08,.08)
                data.qfrc_applied[upper_dofs]=data.qfrc_bias[upper_dofs]
            for leg in legs: leg.apply(data)
            for arm in arms: arm.apply(data)
            if args.scenario=='sofa_side' and not args.furniture_feedback:
                # Compensate the arm links' own gravity through the existing
                # torque-limited motors; the separate free payload is not in bias.
                data.ctrl[arms[0].actuators]+=data.qfrc_bias[upper_dofs]
            mujoco.mj_step(model,data)
            if step%cfg.control_decimation==0:
                speed=args.speed if data.time>1.5 else 0.
                for leg,(prefix,_,facing) in zip(legs,actors):
                    command=np.array([speed if solo or facing==0 else .04,0 if solo or facing==0 else speed*facing,0],dtype=np.float32)
                    if args.scenario=='sofa_side':
                        body=data.body(prefix+'pelvis')
                        yaw=yaw_from_quaternion(body.xquat)
                        desired_x=args.speed*max(0.,data.time-1.5)
                        error=np.array([desired_x-body.xpos[0],-body.xpos[1]])
                        world=np.array([speed,0.])+.8*error
                        command[:2]=np.clip(body.xmat.reshape(3,3)[:2,:2].T@world,-.25,.25)
                        command[2]=np.clip(-yaw,-.4,.4)
                    leg.update_policy(data,command,gait_phase(data))
            support=0.
            for j,c in enumerate(data.contact):
                load_id=model.body('payload').id
                load_contact=load_id in (model.geom_bodyid[c.geom1],model.geom_bodyid[c.geom2])
                if load_contact:
                    other=c.geom2 if model.geom_bodyid[c.geom1]==load_id else c.geom1
                    name=model.body(int(model.geom_bodyid[other])).name
                    mujoco.mj_contactForce(model,data,j,force)
                    if force[0]>0:
                        contact_bodies[name]=contact_bodies.get(name,0)+1
                    worst_penetration=max(worst_penetration,max(0.,-float(c.dist)))
                if load_contact and model.body(int(model.geom_bodyid[other])).name.endswith('wrist_roll_rubber_hand'):
                    mujoco.mj_contactForce(model,data,j,force)
                    support+=abs(float((c.frame.reshape(3,3).T@force[:3])[2]))
                    local=data.body('payload').xmat.reshape(3,3).T@(c.pos-data.body('payload').xpos)
                    hand_contacts+=1
                    if abs(local[2]+half_height)<.04: bottom_contacts+=1
                    on_side=abs(abs(local[1])-model.geom_size[model.geom('load').id,1])<.02
                    hand_name=model.body(int(model.geom_bodyid[other])).name
                    if on_side and 'left_wrist' in hand_name and local[1]>0: side_faces['left_A']+=1
                    elif on_side and 'right_wrist' in hand_name and local[1]<0: side_faces['right_B']+=1
                    else: side_faces['other']+=1
            if support>args.mass*9.81*.5: supported_steps+=1
            vertical_force+=support
            min_z=min(min_z,float(data.body('payload').xpos[2]))
            minimum_floor_clearance=min(minimum_floor_clearance,min(float(data.geom_xpos[g,2]-np.abs(data.geom_xmat[g].reshape(3,3)[2])@model.geom_size[g]) for g in payload_geoms))
            if writer and step%17==0:
                camera=mujoco.MjvCamera(); camera.lookat[:]=data.body('payload').xpos
                camera.distance=4; camera.azimuth=135; camera.elevation=-22
                renderer.update_scene(data,camera=camera); writer.write(renderer.render())
    finally:
        if writer: writer.close()
        if renderer: renderer.close()
    end=data.body('payload').xpos.copy()
    report=dict(scenario=args.scenario,mass_kg=args.mass,robots=len(actors),constraints=int(model.neq),
        start=start.tolist(),end=end.tolist(),minimum_payload_z=min_z,
        displacement_m=float(np.linalg.norm(end[:2]-start[:2])),
        supported_fraction=supported_steps/max(1,steps),mean_hand_support_N=vertical_force/max(1,steps),
        bottom_contact_fraction=bottom_contacts/max(1,hand_contacts),
        payload_size_m=(2*model.geom_size[model.geom('load').id]).tolist(),
        contact_bodies=contact_bodies,maximum_penetration_m=worst_penetration,
        side_face_contacts=side_faces,minimum_floor_clearance_m=minimum_floor_clearance,
        pelvis_z=[float(data.body(prefix+'pelvis').xpos[2]) for prefix,_,_ in actors])
    # Use the object's actual lowest point, not a fixed COM-height threshold:
    # furniture and plates have different heights and can be carried low.
    report['carry_pass']=bool(minimum_floor_clearance>.2 and report['supported_fraction']>.9 and all(z>.6 for z in report['pelvis_z']) and report['displacement_m']>.15 and worst_penetration<.005)
    args.report.parent.mkdir(parents=True,exist_ok=True); args.report.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--scenario',choices=['bottom_crate','solo_sofa','sofa_underneath','sofa_side','team_fixture'],default='bottom_crate')
    parser.add_argument('--carriers',type=int,choices=[2,3],default=3,help='Team size for the T-shaped fixture comparison')
    parser.add_argument('--furniture-feedback',action='store_true',help='Experimental furniture tracking controller')
    parser.add_argument('--mass',type=float,default=5.)
    parser.add_argument('--duration',type=float,default=8.)
    parser.add_argument('--speed',type=float,default=.08)
    parser.add_argument('--video',type=Path)
    parser.add_argument('--report',type=Path,default=Path('outputs/g1_logistics.json'))
    args=parser.parse_args()
    if args.mass<=0 or args.duration<=0: parser.error('mass and duration must be positive')
    run(args)
