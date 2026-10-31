"""Commission the shared measured placement controller on an existing G1 scene."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import mujoco
from ..g1_logistics import build_group
from .carry_controller import CarryController,PROFILES,add_delivery_stage

def run(args):
    from .plant_scene import load_layout
    profile=load_layout().get('carry_profiles',{}).get(args.scenario) if args.factory_route else None
    seed=build_group(SimpleNamespace(scenario=args.scenario,mass=PROFILES[args.scenario]['mass'],carriers=3),
        lambda spec:add_delivery_stage(spec,args.scenario,profile=profile))
    _,model,data,legs,arms,actors,_,cfg,_=seed
    group=dict(scenario=args.scenario,legs=legs,arms=arms,actors=actors,cfg=cfg,body=model.body('payload').id)
    controller=CarryController(model,data,group,profile=profile)
    for step in range(round(args.duration/model.opt.timestep)):
        controller.apply(data); mujoco.mj_step(model,data); controller.after_step(data,step)
    report=controller.report(data)
    args.report.parent.mkdir(parents=True,exist_ok=True); args.report.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--scenario',choices=list(PROFILES),default='team_fixture')
    parser.add_argument('--duration',type=float,default=30)
    parser.add_argument('--factory-route',action='store_true')
    parser.add_argument('--report',type=Path,default=Path('outputs/fms_carry_probe.json'))
    run(parser.parse_args())
