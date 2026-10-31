"""Simulation-independent task-to-fleet rules and replicated demo cells.

These are demonstration task mappings, not certified industrial sizing recommendations.
Payload, reach, tooling and cycle-time feasibility must be checked before deployment.
"""
import copy
import json
import argparse
from .factory_scene import load_config

TASK_MODELS={'material_transport':'husky_a200','shipping_transport':'husky_a200',
    'machine_tending':'ur5e','assembly_inspection':'panda','palletizing':'ur10e',
    'precision_assembly':'panda','visual_inspection':'panda'}

def plan(scenario='metal',cells=1):
    if cells not in (1,2,3): raise ValueError('Use 1–3 cells within the commissioned warehouse footprint')
    config=load_config()
    config['robots'][0]['position'][1]=-2
    config['robots'][1]['position'][1]=-2
    if scenario=='electronics':
        config['robots'][2]['task']='precision_assembly'
        config['robots'][4]['task']='visual_inspection'
    elif scenario=='warehouse':
        # Four mobile bases and one pallet arm per cell.
        for i in (2,3):
            config['robots'][i]['task']='material_transport'
            config['robots'][i]['position']=[-4+(i-2)*4,-1,0]
    rows=[]
    offsets={1:[0],2:[-2,2],3:[-4,0,4]}[cells]
    for cell,y in enumerate(offsets):
        for original in config['robots']:
            row=copy.deepcopy(original)
            row['model']=TASK_MODELS[row['task']]
            row['name']=original['name'] if cells==1 else f'cell{cell}_'+original['name']
            row['position'][1]+=y
            rows.append(row)
    config.update(type=scenario,robots=rows,cells=cells,
        recommendation_status='demo task mapping; not payload/reach validated',
        workflow=['storage','cnc','assembly','inspection','packaging','shipping'])
    return config

if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--scenario',choices=['metal','electronics','warehouse'],default='metal')
    parser.add_argument('--cells',type=int,default=1)
    args=parser.parse_args()
    print(json.dumps(plan(args.scenario,args.cells),indent=2))
