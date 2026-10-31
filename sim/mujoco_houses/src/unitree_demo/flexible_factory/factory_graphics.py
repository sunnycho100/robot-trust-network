"""Original factory wayfinding textures and an understated production HUD."""
from pathlib import Path
from functools import lru_cache
from PIL import Image,ImageDraw,ImageFont

ASSETS=Path(__file__).resolve().parents[3]/'assets/factory_graphics'

@lru_cache(maxsize=24)
def font(size,bold=False):
    candidates=[f'/System/Library/Fonts/Supplemental/Arial{" Bold" if bold else ""}.ttf',
                f'/usr/share/fonts/truetype/dejavu/DejaVuSans{"-Bold" if bold else ""}.ttf']
    for path in candidates:
        try: return ImageFont.truetype(path,size)
        except OSError: pass
    try: return ImageFont.load_default(size=size)
    except TypeError: return ImageFont.load_default()

def sign_asset(code,title,subtitle,accent=(58,155,159),floor=False):
    ASSETS.mkdir(parents=True,exist_ok=True)
    path=ASSETS/(code.lower()+('_floor' if floor else '_sign')+'_v1.png')
    if path.exists(): return path
    image=Image.new('RGB',(1024,256),(27,40,49) if not floor else (69,80,84))
    draw=ImageDraw.Draw(image)
    draw.rectangle((0,0,17,256),fill=accent)
    draw.rounded_rectangle((38,42,212,214),radius=12,fill=accent)
    draw.text((62,85),code,font=font(69,True),fill=(244,247,243))
    title_font=font(51 if len(title)<19 else 41,True)
    draw.text((247,62),title,font=title_font,fill=(242,245,242))
    draw.line((250,137,973,137),fill=(99,119,126),width=2)
    draw.text((250,157),subtitle,font=font(26),fill=(181,199,200))
    image.save(path)
    return path

def draw_factory_hud(screen,data,cells,robot_count,mobiles,diagnostics=False,dual=None):
    draw=ImageDraw.Draw(screen)
    draw.rectangle((0,0,1280,62),fill=(19,30,38))
    draw.rectangle((0,0,7,62),fill=(53,174,169))
    draw.text((23,9),'FLEXIBLE METALWORKS',font=font(23,True),fill=(238,244,243))
    draw.text((24,37),'HIGH-MIX PRODUCTION  /  CONNECTED FACTORY FLOOR',font=font(12),fill=(154,181,185))
    draw.text((858,12),f'{robot_count:02} ROBOTS   /   {mobiles:02} MOBILE RESOURCES',font=font(16,True),fill=(227,236,236))
    draw.text((858,37),f'LIVE SIMULATION     {data.time:06.1f} s',font=font(12),fill=(112,201,187))
    draw.rectangle((0,650,1280,700),fill=(19,30,38))
    labels=['MACHINING','CLEANING','ASSEMBLY','PACKING']
    for i,(label,cell) in enumerate(zip(labels,cells)):
        x=22+i*307; done=cell.state=='COMPLETE'
        team_active=i==3 and dual is not None and dual.state not in ('WAIT_FOR_PACKING','COMPLETE')
        if team_active: done=False; label='TWO-ARM TEAM'
        color=(110,153,166) if done else (68,208,172)
        draw.ellipse((x,667,x+9,676),fill=color)
        draw.text((x+18,661),label,font=font(13,True),fill=(229,235,236))
        draw.text((x+18,679),'SHARED PANEL HANDLING' if team_active else ('WORK ORDER COMPLETE' if done else 'PROCESS IN PROGRESS'),font=font(10),fill=color)
    if diagnostics:
        for i,cell in enumerate(cells):
            draw.text((18,80+i*19),f'{cell.name}: {cell.state} / {cell.arm.state}',font=font(12),
                fill=(242,242,230),stroke_width=1,stroke_fill=(20,25,30))
    draw.rectangle((0,700,1280,720),fill=(12,21,28))
    draw.text((14,704),'Depot: Rafi Abdullah & Louise Poubel / Open Robotics, CC-BY-4.0 | Robots: roqsim / Menagerie | Original factory equipment & graphics',
        font=font(9),fill=(151,171,179))
