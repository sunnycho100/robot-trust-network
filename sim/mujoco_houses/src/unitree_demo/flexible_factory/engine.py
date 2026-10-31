"""Production logic. Only explicit backend acknowledgements move job custody.

The engine has no synthetic 'completed after N seconds' motion path. Machine
timers belong to station backends after a measured fixture/door operation.
"""
import zlib
from .job import Job,Task,Priority
from .buffer import Buffer
from .event_bus import EventBus
from .dispatch import Dispatcher
from .station import Station,STATIONS

class Factory:
    def __init__(self,config,resources,backend):
        self.config=config; self.resources={r.id:r for r in resources}; self.backend=backend
        self.buffers={name:Buffer(name,row['capacity']) for name,row in config['buffers'].items()}
        self.buffers.setdefault('QC_OUTPUT',Buffer('QC_OUTPUT',4))
        self.buffers.setdefault('REWORK_OUTPUT',Buffer('REWORK_OUTPUT',2))
        self.buffers.setdefault('COMPONENTS',Buffer('COMPONENTS',4))
        self.stations={row[0]:Station(*row) for row in STATIONS}
        self.jobs={}; self.tasks={}; self.bus=EventBus(); self.dispatcher=Dispatcher()
        self.time=0.; self.serial=0; self.shipped=[]; self.rejects=0; self.pallets=0
        self.active_event='NORMAL PRODUCTION'; self.action='Dispatching compatible resources'

    def add_job(self,job):
        if job.id in self.jobs: raise ValueError('Duplicate job ID')
        if job.location not in self.buffers: raise ValueError('Initial material needs an explicit buffer')
        self.buffers[job.location].put(job.id)
        self.jobs[job.id]=job
        self.bus.emit(self.time,'MATERIAL_REGISTERED',job=job.id,product=job.product,location=job.location,body=job.body_name)

    def submit(self,kind,job_id,source,destination,priority,capability,agents=1,profile=None,**metadata):
        job=self.jobs[job_id]
        if any(t.job_id==job_id and t.status not in ('DONE','FAILED') for t in self.tasks.values()): return None
        src,dst=self.buffers[source],self.buffers[destination]
        self.serial+=1; token=f'TASK-{self.serial:05d}'
        if not src.reserve_out(job_id,token): return None
        if not dst.reserve_in(job_id,token): src.cancel(token); return None
        task=Task(token,kind,job_id,source,destination,Priority(priority),frozenset([capability]),
            required_agents=agents,profile=profile,created=self.time,metadata=metadata)
        self.tasks[token]=task
        self.bus.emit(self.time,'TASK_CREATED',task=token,job=job_id,kind=kind,priority=int(priority))
        return task

    def acknowledge_pickup(self,task_id):
        task=self.tasks[task_id]
        if task.status!='ASSIGNED' or task.picked: raise ValueError('Invalid or repeated pickup')
        self.buffers[task.source].take(task.job_id,task.id)
        task.picked=True; task.status='PICKED'
        self.jobs[task.job_id].location='ON:'+','.join(task.resources)
        self.bus.emit(self.time,'MEASURED_PICKUP',task=task.id,job=task.job_id)

    def acknowledge_delivery(self,task_id):
        task=self.tasks[task_id]
        if task.status!='PICKED': raise ValueError('No delivery before physical pickup')
        if task.metadata.get('station')=='ASSEMBLY' and not task.metadata.get('component_picked'):
            raise ValueError('Assembly cannot complete before component handling')
        job=self.jobs[task.job_id]
        self.buffers[task.destination].put(job.id,task.id)
        job.location=task.destination; task.status='DONE'
        if task.kind=='PROCESS':
            station=self.stations[task.metadata['station']]
            station.task_id=None
            if station.name=='QC':
                job.qc_attempts+=1
                # Deterministic ~10% first-pass rejects; repairs get a fresh QC pass.
                reject=(zlib.crc32(job.id.encode())%self.config['qc_failure_modulus']==7 and job.reworks==0)
                if reject:
                    self.rejects+=1; task.metadata['qc_result']='FAIL'
                    self.active_event=f'{job.id} QC REJECT'; self.action='Request high-priority rework transfer'
                    self.bus.emit(self.time,'QC_FAIL',job=job.id)
                else:
                    task.metadata['qc_result']='PASS'; job.stage+=1
                    self.bus.emit(self.time,'QC_PASS',job=job.id)
            elif station.name=='REWORK':
                job.reworks+=1
                self.bus.emit(self.time,'REWORK_COMPLETED',job=job.id)
            else:
                job.stage+=1
            station.set_state('IDLE',self.time,self.bus)
        if task.destination=='SHIPPING':
            self.shipped.append(job.id)
            self.bus.emit(self.time,'SHIPPED',job=job.id)
        if task.destination=='FINISHED' and len(self.buffers['FINISHED'].items)==self.config['pallet_capacity']:
            self.pallets+=1; self.bus.emit(self.time,'PALLET_COMPLETE',pallet=self.pallets)
        job.history.append(dict(t=self.time,source=task.source,destination=task.destination,task=task.id))
        self.bus.emit(self.time,'MEASURED_DELIVERY',task=task.id,job=job.id,destination=task.destination)
        self.dispatcher.release(task,self.resources,self.time,self.bus)

    def acknowledge_component(self,task_id):
        task=self.tasks[task_id]
        if task.status!='PICKED' or task.metadata.get('component_picked'): raise ValueError('Invalid component pickup')
        component=self.jobs[task.metadata['component_id']]
        self.buffers['COMPONENTS'].take(component.id,task.id)
        component.location='ASSEMBLED:'+task.job_id
        task.metadata['component_picked']=True
        self.jobs[task.job_id].components.append(component.id)
        self.bus.emit(self.time,'COMPONENT_JOINED',job=task.job_id,component=component.id)

    def _qc_failed(self,job):
        return any(t.job_id==job.id and t.status=='DONE' and t.metadata.get('qc_result')=='FAIL' for t in self.tasks.values()) and job.reworks==0

    def generate(self):
        # Transport requests arise from buffer contents and downstream demand.
        for job in self.jobs.values():
            source=job.location
            if source not in self.buffers or source=='SHIPPING': continue
            if any(t.job_id==job.id and t.status not in ('DONE','FAILED') for t in self.tasks.values()): continue
            if source=='QC_OUTPUT' and self._qc_failed(job):
                self.submit('REWORK_TRANSFER',job.id,source,'REWORK',Priority.REWORK,'rework_transfer'); continue
            if source=='REWORK_OUTPUT':
                self.submit('QC_RETURN',job.id,source,'QC_INPUT',Priority.REWORK,'rework_transfer'); continue
            if job.product=='COMPONENT_B':
                if source=='RECEIVING' and self.stations['ASSEMBLY'].state=='STARVED':
                    self.submit('ASSEMBLY_REPLENISHMENT',job.id,source,'COMPONENTS',Priority.STARVED,'tote_transport')
                continue
            if job.product in ('SHEET','D'):
                destination='MACHINE_STAGING' if job.product=='SHEET' else 'FINAL_STAGING'
                if source!=destination:
                    self.submit('COOPERATIVE_SHEET' if job.product=='SHEET' else 'COOPERATIVE_MODULE',job.id,source,destination,
                        Priority.WIP,'team_transport',agents=job.required_grasp_points,
                        profile='two_bottom' if job.product=='SHEET' else 'three_frame',
                        single_robot='UNSUITABLE',dimensions=job.dimensions,required_grasp_points=job.required_grasp_points)
                continue
            target=job.next_station
            if source=='REWORK': target='REWORK'
            if target=='SHIPPING' and source=='FINISHED':
                # Shipping is a pallet-completion task, not one arbitrary part at a time.
                if len(self.buffers['FINISHED'].items)>=self.config['pallet_capacity']:
                    for item in list(self.buffers['FINISHED'].items):
                        self.submit('FINISHED_GOODS',item,'FINISHED','SHIPPING',Priority.FINISHED,'finished_goods')
                continue
            if target not in self.stations: continue
            station=self.stations[target]
            if source==station.input_buffer: continue
            priority=Priority.BLOCKED if self.buffers[source].full else (Priority.STARVED if not self.buffers[station.input_buffer].items else Priority.WIP)
            capability='tote_transport'
            kind='CNC_OUTPUT_RECOVERY' if source=='CNC_OUTPUT' and self.buffers[source].full else ('REPLENISHMENT' if source=='RECEIVING' else 'WIP_DELIVERY')
            if source=='QC_OUTPUT' and target=='PACKAGING':
                capability='conveyor_transport'; kind='CONVEYOR_TRANSFER'; priority=Priority.WIP
            self.submit(kind,job.id,source,station.input_buffer,priority,capability)
        for station in self.stations.values():
            if station.task_id is not None:
                phase='FAULT' if self.tasks[station.task_id].status=='FAULT' else self.backend.phase(station.task_id)
                station.set_state(phase or 'RUNNING',self.time,self.bus); continue
            output=self.buffers[station.output_buffer]
            available=[j for j in self.buffers[station.input_buffer].items if j not in self.buffers[station.input_buffer].outbound]
            if output.full:
                station.set_state('BLOCKED',self.time,self.bus)
                self.active_event=station.name+' OUTPUT BLOCKED'; self.action='Prioritize downstream material removal'
            elif not available:
                station.set_state('STARVED',self.time,self.bus)
            else:
                if station.name=='ASSEMBLY':
                    components=[j for j in self.buffers['COMPONENTS'].items if j not in self.buffers['COMPONENTS'].outbound]
                    if not components:
                        station.set_state('STARVED',self.time,self.bus,missing='COMPONENT_B')
                        self.active_event='ASSEMBLY COMPONENT SHORTAGE'; self.action='Request compatible replenishment resource'
                        continue
                task=self.submit('PROCESS',available[0],station.input_buffer,station.output_buffer,
                    Priority.REWORK if station.name=='REWORK' else Priority.WIP,station.capability,station=station.name)
                if task:
                    if station.name=='ASSEMBLY':
                        task.metadata['component_id']=components[0]
                        if not self.buffers['COMPONENTS'].reserve_out(components[0],task.id): raise AssertionError('Component race')
                    station.task_id=task.id; station.set_state('WAIT_RESOURCE',self.time,self.bus)

    def tick(self,time):
        if time<self.time: raise ValueError('Factory clock cannot run backwards')
        self.time=float(time)
        # Signals originate in measured controllers. No synthetic motion completion.
        for token,event in self.backend.poll():
            if event=='PICKED': self.acknowledge_pickup(token)
            elif event=='COMPONENT_PICKED': self.acknowledge_component(token)
            elif event=='DELIVERED': self.acknowledge_delivery(token)
            elif event=='FAULT':
                task=self.tasks[token]; task.status='FAULT'
                for name in task.resources:
                    self.resources[name].state='FAULT_HOLD'; self.resources[name].mission='SAFETY HOLD: '+token
                self.active_event='MOTION FAULT'; self.action=f'Hold {token}; retain custody and reservations'
                self.bus.emit(self.time,'MOTION_FAULT',task=token)
        self.generate()
        positions={name:tuple(row['position'][:2]) for name,row in self.config['buffers'].items()}
        self.dispatcher.assign(self.tasks,self.jobs,self.resources,positions,self.backend,self.time,self.bus)
        self.audit()

    def audit(self):
        stored=[j for b in self.buffers.values() for j in b.items]
        if len(stored)!=len(set(stored)): raise AssertionError('One item exists in two buffers')
        for buffer in self.buffers.values():
            if len(buffer.items)+len(buffer.inbound)>buffer.capacity: raise AssertionError('Buffer overbooking')
        for job in self.jobs.values():
            if job.location in self.buffers and job.id not in self.buffers[job.location].items:
                raise AssertionError('Logical custody does not match buffer')
        busy=[name for t in self.tasks.values() if t.status in ('ASSIGNED','PICKED','FAULT') for name in t.resources]
        if len(busy)!=len(set(busy)): raise AssertionError('Robot assigned to two concurrent tasks')

    def metrics(self):
        active=sum(r.state!='AVAILABLE' for r in self.resources.values())
        return dict(time_s=self.time,throughput_parts_hr=len(self.shipped)*3600/max(1.,self.time),
            shipped=len(self.shipped),wip=sum(j.product!='COMPONENT_B' and j.location not in ('RECEIVING','SHIPPING') for j in self.jobs.values()),
            robots_active=active,robots_total=len(self.resources),qc_rejects=self.rejects,
            qc_queue=len(self.buffers['QC_INPUT'].items),pallet=len(self.buffers['FINISHED'].items),
            pallet_capacity=self.config['pallet_capacity'],stations={n:s.state for n,s in self.stations.items()},
            buffers={n:dict(count=len(b.items),capacity=b.capacity,inbound=len(b.inbound)) for n,b in self.buffers.items()},
            event=self.active_event,action=self.action,missions={n:r.mission for n,r in self.resources.items()})
