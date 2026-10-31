"""Non-preemptive priority dispatch; never releases a robot mid-carry."""
from .resource_selector import select

class Dispatcher:
    def assign(self,tasks,jobs,resources,positions,backend,time,bus):
        for task in sorted((t for t in tasks.values() if t.status=='QUEUED'),key=lambda t:(int(t.priority),t.created,t.id)):
            team=select(task,jobs[task.job_id],list(resources.values()),positions.get(task.source,(0.,0.)))
            if not team: continue
            names=tuple(r.id for r in team)
            if not backend.can_execute(task,names): continue
            # A rejected hardware command cannot silently consume material.
            if not backend.start(task,names):
                bus.emit(time,'MOTION_REJECTED',task=task.id,resources=names); continue
            task.resources=names; task.status='ASSIGNED'
            for resource in team:
                resource.state='RESERVED' if len(team)>1 else 'BUSY'
                resource.task_id=task.id; resource.mission=f'{task.kind}: {task.source} -> {task.destination}'
            bus.emit(time,'TEAM_FORMED' if len(team)>1 else 'TASK_ASSIGNED',task=task.id,job=task.job_id,resources=names,priority=int(task.priority))

    @staticmethod
    def release(task,resources,time,bus):
        for name in task.resources:
            resource=resources[name]
            resource.state='AVAILABLE'; resource.task_id=None; resource.mission='AVAILABLE'
        bus.emit(time,'TEAM_DISSOLVED' if len(task.resources)>1 else 'RESOURCE_AVAILABLE',task=task.id,resources=task.resources)
