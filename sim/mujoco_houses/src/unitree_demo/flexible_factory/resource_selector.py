"""Capability compatibility precedes priority, proximity and affinity."""
from dataclasses import dataclass,field
from itertools import combinations
from math import dist

@dataclass
class Resource:
    id:str
    capabilities:frozenset[str]
    position:tuple[float,float]=(0.,0.)
    max_payload_kg:float=1.
    profiles:frozenset[str]=frozenset()
    state:str='AVAILABLE'
    task_id:str|None=None
    mission:str='AVAILABLE'
    fixed:bool=False
    max_dimensions:tuple[float,float,float]=(float('inf'),float('inf'),float('inf'))

    def compatible(self,task):
        return self.state=='AVAILABLE' and task.capabilities<=self.capabilities and (not task.profile or task.profile in self.profiles)

def select(task,job,resources,origin=(0.,0.)):
    eligible=[r for r in resources if r.compatible(task)]
    teams=[]
    for team in combinations(eligible,task.required_agents):
        if sum(r.max_payload_kg for r in team)<job.mass_kg: continue
        if len(team)==1 and any(size>limit for size,limit in zip(job.dimensions,team[0].max_dimensions)): continue
        # Reserve flex capacity for exceptions unless no ordinary carrier is free.
        flex_cost=sum('exception_handling' in r.capabilities for r in team) if int(task.priority)>=4 else 0
        teams.append(((flex_cost,sum(dist(r.position,origin) for r in team),tuple(r.id for r in team)),team))
    return min(teams,key=lambda row:row[0])[1] if teams else ()
