"""One continuous view of the populated plant; no montage or production claims.

Without --video, run physics checks without opening a graphical context.
The preserved humanoid carries start with held loads and place them on stands.
Fixed cells execute finite work orders; factory-wide scheduling is not enabled.
"""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from .plant_scene import build_plant,FIXED_CELLS
from .carry_controller import CarryController
from .scene import add_labels
from .cell_work import CellWork
from ..coop_lift import RawVideoWriter
from .plant_traffic import PlantTraffic
from .quadrupeds import SpotStand
from .spot_walk import SpotWalk
from .plant_finish import update_indicators
from .factory_graphics import draw_factory_hud
from .dual_arm import DualArmWork

VIEWS={
    'overview':((0,.1,.6),23.5,90.,-55.),
    'hero':((0,.5,.8),25.5,110.,-42.),
    'receiving':((-9.,4.4,.9),10.,115.,-42.),
    'machining':((-4.,4.2,1.),10.,125.,-40.),
    'assembly':((2.1,-3.9,.8),10.,110.,-46.),
    'shipping':((9.5,-1.3,.7),12.,130.,-48.),
    'quadrupeds':((-.5,-.9,.4),4.8,120.,-28.),
    'cooperative_arms':((9.55,4.2,1.),4.8,135.,-28.),
}

def run(args):
    model,data,groups,arm,layout=build_plant()
    controllers=[CarryController(model,data,g,g['name']+'_delivery_stand',profile=layout.get('carry_profiles',{}).get(g['scenario'])) for g in groups]
    cells=[CellWork(model,data,name,pos) for name,_,pos,_ in FIXED_CELLS]
    dual=DualArmWork(model,data)
    dogs=[SpotWalk(model,data,row['name'],distance=row['walk_distance'])
          if row.get('walk_distance',0)>0 else SpotStand(model,data,row['name'])
          for row in layout.get('quadrupeds',[])]
    mujoco.mj_forward(model,data)
    # QC begins only after its physical courier reaches the presentation point.
    events=[]
    traffic=PlantTraffic(model,groups,layout)
    traffic.humanoids.extend(dog.body for dog in dogs)
    for dog in dogs:
        for b in range(model.nbody):
            if model.body(b).name.startswith(dog.name+'_'): traffic.owner[b]=dog.name+'_'
    couriers=traffic.robots
    robot_count=sum(len(g['actors']) for g in groups)+2+len(cells)+len(couriers)+len(dogs)
    qc_started=False
    camera=mujoco.MjvCamera()
    camera.lookat[:],camera.distance,camera.azimuth,camera.elevation=VIEWS[args.view]
    renderer=writer=None
    frames=0; steps=round(args.duration/model.opt.timestep)
    try:
        if args.video:
            args.video.parent.mkdir(parents=True,exist_ok=True)
            renderer=mujoco.Renderer(model,height=720,width=1280)
            writer=RawVideoWriter(args.video,1280,720,30,total_frames=int(np.ceil(args.duration*30)))
        for step in range(steps):
            for control in controllers: control.apply(data)
            for dog in dogs: dog.apply(data)
            if step%5==0:
                traffic.update(model,data,events)
                if couriers[0].arrived and not qc_started:
                    arm.source=data.xpos[arm.part].copy()
                    arm.transition(data,events,'APPROACH'); qc_started=True
                arm.update(model,data,events)
                for cell in cells: cell.update(model,data,events)
                dual.update(model,data,cells[-1].state=='COMPLETE')
                update_indicators(model,cells)
            mujoco.mj_step(model,data)
            traffic.observe(model,data)
            dual.observe(model,data)
            for dog in dogs: dog.observe(data)
            for control in controllers: control.after_step(data,step)
            if not np.isfinite(data.qpos).all(): raise RuntimeError('Nonfinite plant state')
            if writer and step*model.opt.timestep>=frames/30:
                renderer.update_scene(data,camera=camera)
                if args.labels: add_labels(renderer.scene,layout)
                pixels=renderer.render()
                from PIL import Image
                screen=Image.fromarray(pixels)
                draw_factory_hud(screen,data,cells,robot_count,len(couriers),args.diagnostics,dual=dual)
                writer.write(np.asarray(screen)); frames+=1
    finally:
        if writer: writer.close()
        if renderer: renderer.close()
    report=dict(duration_s=data.time,robot_count=robot_count,geoms=model.ngeom,
                view=args.view,production_scheduler_enabled=False,
                carries={g['name']:c.report(data) for g,c in zip(groups,controllers)},
                cells={c.name:c.report(data) for c in cells},
                quadrupeds={dog.name:dog.report(data) for dog in dogs},
                dual_arm=dual.report(data),
                qc=dict(state=arm.state,cycles=arm.cycles,error_m=arm.error,part=data.xpos[arm.part].tolist()),
                couriers=[c.report(data) for c in couriers],
                traffic_contacts=sorted(traffic.contact_pairs),
                traffic_max_penetration_m=traffic.max_inter_robot_penetration,
                traffic_obstacle_contacts=sorted(traffic.obstacle_contacts),
                finite_state=bool(np.isfinite(data.qpos).all()),
                simulation_warnings={str(i):int(w.number) for i,w in enumerate(data.warning) if w.number})
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--duration',type=float,default=15.)
    parser.add_argument('--view',choices=list(VIEWS),default='overview')
    parser.add_argument('--labels',action='store_true',help='Enable 3D section labels')
    parser.add_argument('--diagnostics',action='store_true',help='Show controller states over the clean factory presentation')
    parser.add_argument('--video',type=Path)
    parser.add_argument('--report',type=Path,default=Path('outputs/factory_dense_layout.json'))
    args=parser.parse_args()
    if args.duration<=0: parser.error('--duration must be positive')
    run(args)
