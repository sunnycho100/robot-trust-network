"""Conservative demonstrated envelopes, not manufacturer load ratings."""
from .resource_selector import Resource

def current_resources():
    resources=[
        Resource('AMR-1',frozenset(['tote_transport','replenishment','WIP_delivery','finished_goods']),(-.95,3.05),.05),
        Resource('AMR-2',frozenset(['tote_transport','replenishment','WIP_delivery','finished_goods']),(-4,4.5),.05),
        Resource('FLEX_WORKER',frozenset(['tote_transport','urgent_delivery','QC_support','rework_transfer','exception_handling']),(-4,-2),2.,frozenset(['solo_side']),max_dimensions=(.43,.43,.5)),
        Resource('PANDA',frozenset(['deburring','assembly','rework','inspection_support']),(0,3.5),.15,fixed=True),
    ]
    for name,pos,profile,limit in [
        ('CARRY-L',(-.857,-2),'two_bottom',2.5),('CARRY-R',(.857,-2),'two_bottom',2.5),
        ('MODULE-L',(2.843,-1),'three_frame',1.),('MODULE-R',(4.557,-1),'three_frame',1.),('MODULE-S',(3.7,-1.9),'three_frame',1.)]:
        resources.append(Resource(name,frozenset(['team_transport']),pos,limit,frozenset([profile])))
    return resources

def station_resources():
    # Already-used project models, enabled only when actually attached to the scene.
    return [Resource('UR5E',frozenset(['machine_tending','gauging']),(-3,1),.15,fixed=True),
            Resource('UR10E',frozenset(['palletizing','heavy_pick_place']),(3.5,3.7),.15,fixed=True),
            Resource('QC_CONVEYOR',frozenset(['conveyor_transport']),(1,3.1),.15,fixed=True)]
