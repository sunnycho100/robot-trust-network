"""Capacity and atomic inbound/outbound reservations, independent of physics."""
from dataclasses import dataclass,field

@dataclass
class Buffer:
    name:str
    capacity:int
    items:list[str]=field(default_factory=list)
    inbound:dict[str,str]=field(default_factory=dict)
    outbound:dict[str,str]=field(default_factory=dict)

    def __post_init__(self):
        if self.capacity<=0: raise ValueError('Buffer capacity must be positive')

    @property
    def free(self): return self.capacity-len(self.items)-len(self.inbound)

    @property
    def full(self): return self.free<=0

    def reserve_out(self,job_id,task_id):
        if job_id not in self.items or job_id in self.outbound: return False
        self.outbound[job_id]=task_id; return True

    def reserve_in(self,job_id,task_id):
        if self.full or job_id in self.items or job_id in self.inbound.values(): return False
        self.inbound[task_id]=job_id; return True

    def take(self,job_id,task_id):
        if self.outbound.get(job_id)!=task_id: raise ValueError('Pickup without custody reservation')
        self.items.remove(job_id); del self.outbound[job_id]

    def put(self,job_id,task_id=None):
        if job_id in self.items: raise ValueError('Duplicate material identity in buffer')
        if task_id is not None:
            if self.inbound.get(task_id)!=job_id: raise ValueError('Delivery without destination reservation')
            del self.inbound[task_id]
        elif self.full:
            raise OverflowError(self.name+' is full')
        self.items.append(job_id)

    def cancel(self,task_id):
        self.inbound.pop(task_id,None)
        for job in [j for j,t in self.outbound.items() if t==task_id]: del self.outbound[job]
