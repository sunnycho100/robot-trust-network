"""Identity and custody of material; no motion or simulation imports."""
from dataclasses import dataclass,field
from enum import IntEnum

class Priority(IntEnum):
    RECOVERY=0
    BLOCKED=1
    STARVED=2
    REWORK=3
    URGENT=4
    WIP=5
    FINISHED=6
    EMPTY=7

@dataclass
class Job:
    id:str
    product:str
    route:tuple[str,...]
    location:str
    mass_kg:float=.05
    dimensions:tuple[float,float,float]=(.05,.05,.05)
    stage:int=0
    qc_attempts:int=0
    reworks:int=0
    body_name:str|None=None
    components:list[str]=field(default_factory=list)
    required_grasp_points:int=1
    history:list[dict]=field(default_factory=list)

    @property
    def next_station(self):
        return self.route[self.stage] if self.stage<len(self.route) else None

@dataclass
class Task:
    id:str
    kind:str
    job_id:str
    source:str
    destination:str
    priority:Priority
    capabilities:frozenset[str]
    required_agents:int=1
    profile:str|None=None
    status:str='QUEUED'
    resources:tuple[str,...]=()
    created:float=0.
    picked:bool=False
    metadata:dict=field(default_factory=dict)
