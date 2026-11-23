"""Encode the synchronized measured HUD and extract its actual keyframes."""
import json
from pathlib import Path
import subprocess
import argparse
import numpy as np
from PIL import Image,ImageDraw,ImageFont

def stamp(seconds):
    cs=round(seconds*100)
    return f'{cs//360000}:{cs//6000%60:02}:{cs//100%60:02}.{cs%100:02}'

def compose(out,package,left,right,shell,proxy):
    a=np.load(package);s=np.load(shell);p=np.load(proxy)
    linfo=json.loads(shell.with_suffix('.json').read_text())
    rinfo=json.loads(proxy.with_suffix('.json').read_text())
    length=len(a['time'])-1
    assert length==240
    header="""[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 720
ScaledBorderAndShadow: yes
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Main,Liberation Sans,29,&H00F4F7F7,&H00FFFFFF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1
Style: Note,Liberation Sans,18,&H00D1D9DC,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1
Style: HUD,Liberation Sans,24,&H00F4F7F7,&H00FFFFFF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events=[]
    def label(start,end,x,y,text,style):
        events.append(f'Dialogue: 0,{stamp(start)},{stamp(end)},{style},,0,0,0,,{{\\pos({x},{y})}}{text}')
    for x,title in [(25,'High-Fidelity MuJoCo flex / Thin-Shell'),(985,'Ours: Low-Order Passive Proxy (Visual Skin)')]:
        label(0,9.6,x,8,title,'Main')
    label(0,9.6,25,46,f"PE  /  Folded textile  /  65 g      •      1 env: {linfo['steps_s']:.0f} steps/s  ({1000*linfo.get('dt',.0002):g} ms)",'Note')
    label(0,9.6,985,46,f"1 env: {rinfo['steps_s']:.0f} steps/s (2 ms)   •   Empty-arm benchmark:  4,096 envs: 157k steps/s  |  30,000 envs: 636k steps/s",'Note')
    for i in range(length):
        t=i/25
        phase='Approach' if t<1.56 else 'Grasp' if t<3.15 else 'Lift' if t<5.15 else 'Hold'
        for x,key in [(25,'flex'),(985,'proxy')]:
            label(t,t+.04,x,660,f'{t:04.2f} s     {phase.upper()}','HUD')
            label(t,t+.04,x+310,660,f"Clearance   {max(0,1000*a[key+'_clearance'][i]):.0f} mm",'HUD')
            label(t,t+.04,x+600,660,f"COM speed   {1000*np.linalg.norm(a[key+'_v'][i]):.1f} mm/s",'HUD')
    ass=out/'metrics.ass';ass.write_text(header+'\n'.join(events)+'\n')
    movie=out/'compare_flex_vs_proxy_side_by_side.mp4'
    filters=(f'[0:v][1:v]hstack=inputs=2,drawbox=x=0:y=0:w=iw:h=77:c=0x17242B:t=fill,'
             f'drawbox=x=0:y=646:w=iw:h=74:c=0x17242B:t=fill,'
             f'drawbox=x=959:y=0:w=2:h=ih:c=0x42535B:t=fill,subtitles={ass}')
    subprocess.run(['ffmpeg','-v','error','-n','-i',str(left),'-i',str(right),'-filter_complex',filters,
        '-c:v','libx264','-crf','16','-preset','fast','-pix_fmt','yuv420p','-movflags','+faststart',str(movie)],check=True)
    grip=(s['grip'].min(1)>.1)&p['capture'].all(1)
    close=np.where(grip&(a['time']>=2.93)&(a['time']<3.15))[0][0]
    lift=np.where(grip&(a['time']>=3.15)&(a['flex_clearance']>.005)&(a['proxy_clearance']>.005))[0][0]
    images=[]
    for name,i in [('grasp',close),('liftoff',lift),('hold',length-2)]:
        target=out/f'{name}.png'
        subprocess.run(['ffmpeg','-v','error','-n','-i',str(movie),'-vf',f'select=eq(n\\,{i})','-frames:v','1',str(target)],check=True)
        images.append(Image.open(target).convert('RGB').resize((1280,480),Image.Resampling.LANCZOS))
    sheet=Image.new('RGB',(1280,1440))
    for i,im in enumerate(images):sheet.paste(im,(0,i*480))
    sheet.save(out/'keyframes.png')
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('output','package','left','right','shell','proxy'):parser.add_argument(name,type=Path)
    v=parser.parse_args();compose(v.output,v.package,v.left,v.right,v.shell,v.proxy)
