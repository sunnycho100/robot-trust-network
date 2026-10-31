"""Add industrial structure to the current world; do not duplicate carry scenes."""
import mujoco
from ..industrial_showcase import box

def decorate(spec,config):
    for geom in list(spec.geoms):
        if '_lane' in geom.name: spec.delete(geom)
    # Yellow main traffic spine and blue material staging, away from carries.
    for y in (.75,2.25):
        box(spec,f'traffic_{y}',(0,y,.004),(5.8,.018,.004),(.95,.72,.08,1)).contype=0
    for name,buffer in config['buffers'].items():
        x,y,z=buffer['position']
        if name=='QC_INPUT': continue  # Existing TurtleBot/Panda dock is retained.
        for dx in (-.25,.25):
            g=box(spec,name+f'_stage{dx}',(x+dx,y,.006),(.012,.22,.006),(.1,.45,.9,1))
            g.contype=0; g.conaffinity=0
    # Enclosed CNC, fixture, sliding door, and distinct adjacent gauging bench.
    box(spec,'cnc_bed',(-3,1.9,.60),(.43,.28,.06),(.18,.25,.28,1))
    for x in (-3.45,-2.55): box(spec,'cnc_side'+str(x),(x,1.9,1.0),(.04,.32,.4),(.2,.36,.42,1))
    box(spec,'cnc_rear',(-3,2.2,1.0),(.45,.04,.4),(.2,.36,.42,1))
    box(spec,'cnc_roof',(-3,1.9,1.43),(.48,.34,.035),(.2,.36,.42,1))
    door=spec.worldbody.add_body(name='fms_cnc_door',pos=(-3,1.55,1.0))
    door.add_joint(name='fms_cnc_door_slide',type=mujoco.mjtJoint.mjJNT_SLIDE,axis=(1,0,0),range=(0,.8),limited=True,damping=8)
    box(spec,'fms_cnc_door_glass',(0,0,0),(.38,.018,.32),(.15,.45,.52,.45),body=door,mass=1)
    spec.add_actuator(name='fms_cnc_door_motor',target='fms_cnc_door_slide',trntype=mujoco.mjtTrn.mjTRN_JOINT,
        gainprm=(200,0,0,0,0,0,0,0,0,0),biastype=mujoco.mjtBias.mjBIAS_AFFINE,biasprm=(0,-200,-25,0,0,0,0,0,0,0))
    # Deburr booth and assembly fixtures have recognizably different tooling.
    box(spec,'deburr_bench',(-1.3,4.7,.60),(.55,.30,.05),(.25,.36,.35,1))
    box(spec,'deburr_guard',(-1.3,4.99,.9),(.55,.025,.3),(.18,.4,.4,.65))
    box(spec,'deburr_fixture',(-1.3,4.65,.70),(.12,.10,.05),(.55,.58,.62,1))
    box(spec,'assembly_fixture',(1.2,2.8,.65),(.35,.25,.05),(.35,.4,.45,1))
    box(spec,'rework_table',(.35,4.3,.48),(.25,.2,.04),(.45,.3,.48,1))
    # Charging and shipping are destinations, not decorative robot loops.
    for x in (-5.5,-4.7):
        box(spec,'charger'+str(x),(x,5.5,.2),(.2,.12,.2),(.12,.22,.32,1))
    for i in range(3):
        box(spec,f'shipping_pallet{i}',(4.7+i*.6,5.1,.08),(.25,.35,.08),(.52,.33,.18,1))

def add_labels(scene,config):
    import numpy as np
    for section in config['sections'].values():
        if scene.ngeom>=scene.maxgeom: break
        geom=scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_LABEL,np.zeros(3),
            np.array([*section['center'],2.2]),np.eye(3).ravel(),np.array([.95,.95,.95,1]))
        geom.label=section['label']; scene.ngeom+=1
