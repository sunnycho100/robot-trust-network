"""Architectural and equipment finish on the existing, commissioned layout.

All added architecture/details are visual-only; validated contact geometry,
actuators, part poses and logistics routes are preserved.
"""
import mujoco
from .plant_props import detail
from .factory_graphics import sign_asset

PAINT=(.72,.76,.76,1)
GRAPHITE=(.16,.22,.26,1)
TEAL=(.12,.40,.43,1)
WHITE=(.86,.89,.87,1)

def cylinder(spec,name,pos,radius,half,color):
    return spec.worldbody.add_geom(name=name,type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=pos,size=(radius,half,.001),rgba=color,contype=0,conaffinity=0)

def graphic(spec,code,title,subtitle,pos,width=2.1,height=.525,floor=False):
    suffix='_floor' if floor else '_sign'
    texture=spec.add_texture(name='finish_'+code+suffix,type=mujoco.mjtTexture.mjTEXTURE_2D,
                             file=str(sign_asset(code,title,subtitle,floor=floor)))
    material=spec.add_material(name='finish_mat_'+code+suffix,rgba=(1,1,1,1),
                               specular=.08,shininess=.15)
    material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB]=texture.name
    if floor:
        spec.worldbody.add_geom(name='finish_decal_'+code,type=mujoco.mjtGeom.mjGEOM_PLANE,
            pos=pos,size=(width/2,height/2,.001),material=material.name,contype=0,conaffinity=0)
    else:
        detail(spec,'finish_back_'+code,pos,(width/2,.025,height/2),GRAPHITE)
        for face,(dy,quat) in enumerate(((-.026,(.70710678,.70710678,0,0)),(.026,(0,0,.70710678,.70710678)))):
            spec.worldbody.add_geom(name=f'finish_graphic_{code}_{face}',type=mujoco.mjtGeom.mjGEOM_PLANE,
                pos=(pos[0],pos[1]+dy,pos[2]),size=(width/2,height/2,.001),quat=quat,
                material=material.name,contype=0,conaffinity=0)

def apply_finish(spec,layout):
    # Keep Depot's attributed floor; remove isolated shell fragments now replaced
    # by a coherent cutaway building. The original standalone scenes are untouched.
    for geom in list(spec.geoms):
        if geom.name.startswith('scene_Depot_') and 'FLOOR' not in geom.name:
            spec.delete(geom)
    # Visual epoxy finish, 2 mm above the unchanged physical contact plane.
    spec.worldbody.add_geom(name='finish_epoxy',type=mujoco.mjtGeom.mjGEOM_PLANE,
        pos=(0,0,.002),size=(13.6,7.3,.001),rgba=(.43,.47,.48,1),contype=0,conaffinity=0)
    detail(spec,'finish_main_aisle',(0,1.5,.0028),(12.6,1.08,.0003),(.24,.30,.33,1))
    zones=[
        ('A1','RECEIVING','RAW STOCK / INBOUND',(-10.5,4.5),(3.6,3.2),(.44,.45,.42,1)),
        ('B1','SHEET / PRESS','OVERSIZED MATERIAL',(-9,-3),(3.3,3.8),(.36,.41,.43,1)),
        ('C1','MACHINING','CNC / TOOL MANAGEMENT',(-5.5,4.6),(3.2,3.2),(.35,.43,.45,1)),
        ('D1','CLEANING','DEBURR / FINISH',(-5,-3.5),(3.,3.),(.35,.41,.41,1)),
        ('E1','ASSEMBLY','KITTING / MODULE BUILD',(3.6,-3.8),(4.4,4.4),(.36,.44,.41,1)),
        ('F1','QUALITY','MEASUREMENT / REWORK',(4,4.2),(3.4,3.),(.44,.49,.50,1)),
        ('G1','PACKING','PACK / PALLETIZE',(9,4.3),(3.6,3.4),(.48,.46,.41,1)),
        ('H1','SHIPPING','FINISHED GOODS',(11,-3.5),(3.8,4.),(.41,.45,.46,1)),
    ]
    for code,title,subtitle,(x,y),(w,h),color in zones:
        detail(spec,'finish_zone_'+code,(x,y,.0035),(w/2,h/2,.0003),color)
        graphic(spec,code,title,subtitle,(x,y-h/2+.30,.0085),width=min(2.5,w-.2),height=.48,floor=True)
    # Tall rear wall + low side walls: an architectural interior without a
    # foreground wall or roof hiding robot work from the overview camera.
    detail(spec,'finish_rear_wall',(0,7.35,1.85),(13.7,.10,1.85),PAINT)
    detail(spec,'finish_wall_dado',(0,7.23,.50),(13.7,.025,.50),GRAPHITE)
    detail(spec,'finish_wall_stripe',(0,7.20,1.09),(13.7,.018,.035),TEAL)
    for x in (-13.7,13.7):
        detail(spec,'finish_side_wall'+str(x),(x,0,.55),(.08,7.3,.55),PAINT)
        detail(spec,'finish_side_base'+str(x),(x,0,.12),(.09,7.3,.12),GRAPHITE)
    for x in (-11,-7,-3,1,5,9):
        detail(spec,'finish_window_frame'+str(x),(x,7.21,2.78),(1.55,.04,.49),GRAPHITE)
        detail(spec,'finish_window'+str(x),(x,7.165,2.78),(1.50,.009,.44),(.40,.57,.62,1))
        for dx in (-.5,.5):
            detail(spec,f'finish_window_mullion{x}_{dx}',(x+dx,7.148,2.78),(.015,.014,.44),PAINT)
    # Signs mounted to rear posts/racks, not floating UI labels.
    for code,title,subtitle,x in [
        ('A1','RECEIVING','RAW STOCK / INBOUND',-10.8),
        ('C1','MACHINING','CNC / TOOL MANAGEMENT',-5.6),
        ('F1','QUALITY','MEASUREMENT / REWORK',3.9),
        ('G1','PACKING','PACK / PALLETIZE',9.1)]:
        graphic(spec,code,title,subtitle,(x,6.97,3.20),width=2.5,height=.625)
        for dx in (-.9,.9):
            detail(spec,f'finish_signpost{code}_{dx}',(x+dx,7.,1.55),(.025,.025,1.55),GRAPHITE)
    # A rear service backbone and ceiling battens give scale without a roof.
    detail(spec,'finish_service_trunk',(0,7.02,3.61),(13.3,.12,.06),GRAPHITE)
    for x in (-10,-5,0,5,10):
        detail(spec,'finish_light_bar'+str(x),(x,6.6,3.67),(1.5,.055,.025),WHITE)
    # Dock door / bumpers remain at the perimeter, outside all tested routes.
    detail(spec,'finish_dock_recess',(13.56,-3.5,1.55),(.055,1.35,1.55),GRAPHITE)
    for i in range(20):
        detail(spec,f'finish_dock_slat{i}',(13.49,-3.5,.20+i*.14),(.025,1.22,.061),(.63,.69,.7,1))
    for y in (-4.9,-2.1):
        detail(spec,'finish_dock_bumper'+str(y),(13.42,y,.40),(.12,.13,.40),(.08,.11,.13,1))
    # CNC service panels, control bezel and access opening.
    for name in ('cnc_foundation','cnc_back','cnc_top'):
        spec.geom(name).rgba=PAINT
    for x in (-6.65,-4.35): spec.geom('cnc_wall'+str(x)).rgba=PAINT
    for x in (-6.18,-4.82):
        detail(spec,'finish_cnc_panel'+str(x),(x,3.97,1.44),(.27,.025,.57),PAINT)
        detail(spec,'finish_cnc_glazing'+str(x),(x,3.941,1.62),(.20,.008,.24),(.12,.27,.31,1))
        detail(spec,'finish_cnc_handle'+str(x),(x+.17,3.92,1.24),(.018,.015,.12),GRAPHITE)
    detail(spec,'finish_cnc_header',(-5.5,3.97,2.08),(.42,.027,.12),TEAL)
    detail(spec,'finish_cnc_plinth',(-5.5,3.985,.30),(1.15,.02,.30),GRAPHITE)
    for i in range(9):
        detail(spec,f'finish_cnc_vent{i}',(-6.18,3.935,.92+i*.035),(.19,.004,.006),GRAPHITE)
    # Tower indicator signals are driven by the actual cell controller below.
    cylinder(spec,'finish_cnc_beacon_stem',(-4.15,4.,1.57),.016,.11,GRAPHITE)
    for i,color in enumerate(((.8,.17,.12,1),(.95,.64,.12,1),(.12,.75,.45,1))):
        cylinder(spec,'finish_cnc_beacon'+str(i),(-4.15,4.,1.72+i*.065),.045,.028,color)
    # Hydraulic press: recognizable ram, platen, rear frame and operator panel.
    detail(spec,'finish_press_back',(-9,-.17,1.3),(1.08,.045,1.05),GRAPHITE)
    detail(spec,'finish_press_bed',(-9,-.60,.74),(.98,.45,.11),PAINT)
    detail(spec,'finish_press_platen',(-9,-.55,1.57),(.87,.31,.07),PAINT)
    cylinder(spec,'finish_press_ram',(-9,-.5,2.1),.13,.42,(.52,.59,.61,1))
    detail(spec,'finish_press_panel',(-10.43,-.72,1.05),(.13,.10,.27),PAINT)
    detail(spec,'finish_press_estop',(-10.43,-.827,1.08),(.034,.008,.034),(.85,.15,.10,1))
    # Rear extraction plenum above the cleaning bench.
    detail(spec,'finish_extraction_hood',(-5,-2.73,1.18),(.78,.11,.30),PAINT)
    for i in range(12):
        detail(spec,f'finish_extraction_grille{i}',(-5.66+i*.12,-2.85,1.18),(.012,.008,.23),GRAPHITE)
    cylinder(spec,'finish_extraction_duct',(-5,-2.7,1.9),.12,.42,GRAPHITE)
    # Actual inspection table geometry was a fixed slab; give it visible supports.
    for x in (4.4,5.4):
        for y in (3.82,4.38):
            detail(spec,f'finish_qc_leg{x}_{y}',(x,y,.335),(.035,.035,.335),GRAPHITE)
    for x in (4.5,5.3):
        detail(spec,'finish_qc_post'+str(x),(x,4.4,1.07),(.025,.025,.32),GRAPHITE)
    detail(spec,'finish_qc_light', (4.9,4.4,1.40),(.43,.055,.025),WHITE)
    detail(spec,'finish_qc_sensor',(4.9,4.25,1.32),(.06,.08,.055),GRAPHITE)
    cylinder(spec,'finish_qc_lens',(4.9,4.25,1.25),.027,.013,(.12,.30,.38,1))
    spec.geom('precision_assembly_top').rgba=(.16,.31,.30,1)
    # Conveyor legs and end rails provide a complete machine silhouette.
    for x in (5.7,6.9,8.1):
        for y in (2.85,3.25):
            detail(spec,f'finish_conveyor_leg{x}_{y}',(x,y,.32),(.025,.025,.32),GRAPHITE)
    for y in (2.79,3.31):
        detail(spec,'finish_conveyor_rail'+str(y),(6.9,y,.80),(1.42,.018,.05),PAINT)
    # Controlled lighting replaces the single broad default spot.
    for light in list(spec.lights): spec.delete(light)
    spec.worldbody.add_light(name='finish_key',pos=(-4,-3,9),dir=(.25,.25,-1),
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,diffuse=(.65,.65,.61),ambient=(.12,.13,.14),castshadow=True)
    spec.worldbody.add_light(name='finish_fill',pos=(8,4,7),dir=(-.3,-.2,-1),
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,diffuse=(.27,.32,.35),ambient=(.05,.06,.07),castshadow=False)
    spec.visual.headlight.diffuse=(.30,.30,.30)
    spec.visual.headlight.ambient=(.25,.25,.25)
    spec.visual.quality.shadowsize=4096
    spec.visual.quality.offsamples=4
    spec.add_texture(name='finish_background',type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,width=256,height=1536,
        rgb1=(.22,.29,.33),rgb2=(.10,.15,.19))

def update_indicators(model,cells):
    running=cells[0].state!='COMPLETE'
    for i in range(3):
        model.geom_rgba[model.geom('finish_cnc_beacon'+str(i)).id]=(
            (.12,.78,.43,1) if i==2 and running else (.19,.23,.24,1))
