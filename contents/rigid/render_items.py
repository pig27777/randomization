"""Render the three presets: the same structure with different parameters."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import mujoco
import numpy as np
from pathlib import Path
import unified_content as uc

HERE = Path(__file__).resolve().parent
frames=[]
for name,x in uc.PRESETS.items():
    body,tendons,_=uc.item_xml(x,pos=(0,0,x.half[2]+.002))
    xml=uc.scene_header()+'<worldbody><light pos=".3 -.3 1" dir="-.3 .3 -1"/><geom type="plane" size="1 1 .01" rgba=".85 .87 .9 1"/>'+body+f'</worldbody><tendon>{tendons}</tendon></mujoco>'
    m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m)
    for _ in range(1000): mujoco.mj_step(m,d)
    r=mujoco.Renderer(m,500,640);c=mujoco.MjvCamera();c.lookat[:]=(0,0,x.half[2]);c.distance=.55;c.azimuth=135;c.elevation=-28
    r.update_scene(d,c);frames.append(r.render().copy());r.close()
img=np.concatenate(frames,axis=1)
(HERE / 'results').mkdir(exist_ok=True)
import zlib,struct
h,w,_=img.shape
raw=b''.join(b'\0'+img[i].tobytes() for i in range(h))
def chunk(t,b): return struct.pack('>I',len(b))+t+b+struct.pack('>I',zlib.crc32(t+b)&0xffffffff)
(HERE / 'results' / 'items_book_clothing_bottle.png').write_bytes(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',w,h,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(raw))+chunk(b'IEND',b''))
print('ok',w,h)
