from dataclasses import dataclass

@dataclass
class Station:
    name:str
    input_buffer:str
    output_buffer:str
    capability:str
    state:str='STARVED'
    task_id:str|None=None

    def set_state(self,state,time,bus,**fields):
        if state!=self.state:
            self.state=state; bus.emit(time,self.name+'_'+state,station=self.name,**fields)

STATIONS=(
    ('CNC','CNC_INPUT','CNC_OUTPUT','machine_tending'),
    ('DEBURR','DEBURR_INPUT','DEBURR_OUTPUT','deburring'),
    ('ASSEMBLY','ASSEMBLY_INPUT','ASSEMBLY_OUTPUT','assembly'),
    ('QC','QC_INPUT','QC_OUTPUT','inspection_support'),
    ('REWORK','REWORK','REWORK_OUTPUT','rework'),
    ('PACKAGING','PACKAGING_INPUT','FINISHED','palletizing'),
)
