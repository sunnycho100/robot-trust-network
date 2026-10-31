"""Regression entry point for incremental factory commissioning."""
import argparse
from pathlib import Path
from .config import load_config
from .scene import decorate,add_labels
from ..industrial_showcase import run

def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--duration',type=float,default=8.)
    parser.add_argument('--shot',choices=['overview','sofa','crate','frame','supply'],default='overview')
    parser.add_argument('--video',type=Path)
    parser.add_argument('--report',type=Path,default=Path('outputs/fms_phase01.json'))
    parser.add_argument('--require-complete',action='store_true')
    args=parser.parse_args()
    if not 0<args.duration<=8: parser.error('Commissioning window: 0 < duration <= 8')
    config=load_config()
    def overlay(screen,draw,data,groups,arm):
        draw.rectangle((0,0,1280,65),fill=(12,23,32))
        draw.text((18,10),'FLEXIBLE FACTORY | LAYOUT COMMISSIONING | 8 SECTIONS',fill=(245,245,245))
        draw.text((18,34),f'Existing carries preserved | QC material handoff: {arm.state} | t={data.time:.1f}s',fill=(100,230,190))
    run(args,scene_hook=lambda spec:decorate(spec,config),scene_labels=lambda scene:add_labels(scene,config),overlay=overlay)

if __name__=='__main__': main()
