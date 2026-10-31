"""Original industrial set dressing, in metres; no marketplace dependencies.

Decorative details have no collision. Rack decks, stocked cartons and cabinets
remain solid, so navigable space is distinct from occupied staging space.
"""
import mujoco
from ..industrial_showcase import box

STEEL=(.17,.25,.30,1)
BLUE=(.13,.31,.43,1)
ORANGE=(.91,.43,.12,1)

def detail(spec,name,pos,size,color):
    g=box(spec,name,pos,size,color)
    g.contype=g.conaffinity=0
    return g

def carton(spec,name,xy,base=0,size=(.48,.36,.34),variant=0):
    """Closed carton with sealing tape, top seam and shipping-label graphics."""
    x,y=xy; w,d,h=size
    palette=[(.62,.44,.27,1),(.72,.53,.33,1),(.54,.38,.23,1)]
    box(spec,name,(x,y,base+h/2),(w/2,d/2,h/2),palette[variant%3])
    detail(spec,name+'_tape',(x,y,base+h+.001),(.035,d/2,.001),(.79,.66,.44,1))
    detail(spec,name+'_seam',(x,y,base+h+.003),(.001,d/2,.001),(.33,.25,.16,1))
    detail(spec,name+'_label',(x+w*.19,y-d/2-.001,base+h*.59),(w*.17,.001,h*.18),(.92,.91,.84,1))
    for i in range(7):
        detail(spec,f'{name}_barcode{i}',(x+w*(.08+i*.027),y-d/2-.003,base+h*.57),
               (.002 if i%2 else .004,.001,h*.09),(.12,.15,.17,1))
    for sign in (-1,1):
        detail(spec,name+f'_edge{sign}',(x+sign*w/2,y,base+h-.005),(.002,d/2,.003),(.45,.31,.19,1))

def pallet(spec,name,xy,base=0):
    x,y=xy
    wood=(.51,.36,.22,1)
    for dx in (-.43,0,.43):
        box(spec,f'{name}_runner{dx}',(x+dx,y,base+.035),(.055,.6,.035),wood)
        for dy in (-.48,0,.48):
            box(spec,f'{name}_block{dx}_{dy}',(x+dx,y+dy,base+.095),(.055,.07,.025),wood)
    for i in range(5):
        box(spec,f'{name}_board{i}',(x-.44+i*.22,y,base+.135),(.10,.6,.015),(.62,.46,.29,1))

def stocked_pallet(spec,name,xy,layers=2):
    pallet(spec,name,xy)
    for layer in range(layers):
        for i in range(2):
            for j in range(3):
                carton(spec,f'{name}_c{layer}_{i}_{j}',
                       (xy[0]+(i-.5)*.49,xy[1]+(j-1)*.38),
                       .15+layer*.30,(.46,.35,.30),layer+i+j)

def rack(spec,name,xy,bays=2):
    x,y=xy
    for i in range(bays+1):
        for sign in (-1,1):
            box(spec,f'{name}_upright{i}_{sign}',(x+i*1.6,y+sign*.46,1.35),(.04,.04,1.35),BLUE)
    for i in range(bays):
        for level,z in enumerate((.18,1.0,1.82)):
            box(spec,f'{name}_deck{i}_{level}',(x+.8+i*1.6,y,z),(.76,.49,.025),STEEL)
            for sign in (-1,1):
                detail(spec,f'{name}_beam{i}_{level}_{sign}',(x+.8+i*1.6,y+sign*.48,z-.04),(.78,.025,.055),ORANGE)
            for k in range(3):
                carton(spec,f'{name}_stock{i}_{level}_{k}',(x+.28+i*1.6+k*.51,y),z+.025,(.45,.70,.42),i+level+k)

def cabinet(spec,name,xy,width=.75):
    x,y=xy
    box(spec,name,(x,y,.47),(width/2,.29,.47),BLUE)
    detail(spec,name+'_top',(x,y,.96),(width/2+.025,.315,.02),(.49,.53,.54,1))
    for i in range(5):
        z=.13+i*.16
        detail(spec,f'{name}_drawer{i}',(x,y-.293,z),(width/2-.025,.006,.068),(.21,.37,.46,1))
        detail(spec,f'{name}_handle{i}',(x,y-.307,z+.025),(.11,.009,.008),(.70,.73,.73,1))

def cart(spec,name,xy):
    x,y=xy
    for z in (.20,.65):
        box(spec,f'{name}_shelf{z}',(x,y,z),(.38,.26,.025),STEEL)
    for dx in (-.34,.34):
        for dy in (-.22,.22):
            box(spec,f'{name}_post{dx}_{dy}',(x+dx,y+dy,.46),(.017,.017,.30),BLUE)
            spec.worldbody.add_geom(name=f'{name}_wheel{dx}_{dy}',type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                pos=(x+dx,y+dy,.085),quat=(.70710678,.70710678,0,0),size=(.075,.025,.001),rgba=(.08,.10,.12,1))
    carton(spec,name+'_load',(x,y),.675,(.54,.38,.25))

def populate(spec):
    # Back-of-house stock wall. Aisles at y=.4..2.6 remain unobstructed.
    for name,x,bays in [('receiving_stock',-12.5,2),('cnc_tool_stock',-8.4,1),
                        ('component_stock',-2.2,2),('packing_stock',7.0,3)]:
        rack(spec,name,(x,6.45),bays)
    # Satellite staging, not three copies of a carrying demonstration.
    for name,xy,layers in [
        ('incoming_a',(-12.0,4.65),2),('incoming_b',(-10.5,4.65),3),
        ('raw_overflow',(-8.1,4.6),2),('dispatch_a',(9.0,-5.5),2),
        ('dispatch_b',(10.45,-5.5),3),('dispatch_c',(11.9,-5.5),2),
        ('packaging_stock',(11.9,4.6),2),('shipping_ready',(11.9,-1.7),3)]:
        stocked_pallet(spec,name,xy,layers)
    for name,xy in [('machinist_tools',(-3.65,5.3)),('deburr_tools',(-2.0,4.8)),
                    ('qc_tools',(5.6,5.4)),('assembly_tools',(.1,-4.7)),
                    ('packaging_tools',(10.8,5.3)),('press_tools',(-12.1,-.7))]:
        cabinet(spec,name,xy)
    for name,xy in [('cnc_wip',(-3.2,3.5)),('deburr_wip',(1.0,3.6)),
                    ('qc_rejects',(2.5,4.0)),('assembly_kitting',(.1,-3.4)),
                    ('shipping_docs',(8.1,-4.3)),('press_offcuts',(-11.9,-4.8))]:
        cart(spec,name,xy)
    # Peripheral assembly stock leaves the validated six humanoid footprints free.
    rack(spec,'assembly_material',(1.8,-6.25),2)
    rack(spec,'maintenance_stock',(-6.8,-6.25),2)
    for i,x in enumerate((-10.5,-9.,-7.5)):
        pallet(spec,f'sheet_pallet{i}',(x,-5.8))
        for j in range(4):
            box(spec,f'metal_bundle{i}_{j}',(x,-5.8,.17+j*.035),(.46,.52,.017),(.46,.52,.57,1))
        for dx in (-.28,.28):
            detail(spec,f'metal_strap{i}_{dx}',(x+dx,-5.8,.31),(.018,.53,.005),(.13,.17,.20,1))
    for x in (-2.2,-1.4):
        box(spec,f'waste_bin{x}',(x,5.7,.35),(.24,.23,.35),(.24,.32,.31,1))
        detail(spec,f'waste_lid{x}',(x,5.7,.71),(.26,.25,.018),(.12,.18,.18,1))
    # Repeated bollards protect the long process-cell edge without a camera wall.
    for x in (-12.8,-8.9,-7.2,-3.1,1.7,5.8,8.5,12.8):
        spec.worldbody.add_geom(name=f'safety_bollard{x}',type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            pos=(x,2.95,.36),size=(.065,.36,.001),rgba=(.95,.68,.08,1))
        detail(spec,f'bollard_band{x}',(x,2.95,.47),(.067,.067,.045),(.13,.15,.16,1))
