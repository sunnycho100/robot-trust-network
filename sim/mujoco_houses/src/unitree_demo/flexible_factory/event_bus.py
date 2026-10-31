class EventBus:
    def __init__(self): self.events=[]

    def emit(self,time,event,**fields):
        row=dict(t=round(float(time),4),event=event,**fields)
        self.events.append(row)
        return row
