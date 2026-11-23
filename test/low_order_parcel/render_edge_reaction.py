"""Draw the four-block PPO parcel directly from recorded state, at 25 fps."""
import argparse
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 1152, 620
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
FONTS = {size: ImageFont.truetype(FONT, size) for size in (16, 18, 21, 27, 33)}
SIGNS = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
FACES = ((0,1,3,2), (4,6,7,5), (0,4,5,1), (2,3,7,6), (0,2,6,4), (1,5,7,3))
INK, SOFT, ORANGE, TEAL = '#e8ecf1', '#a4b6c5', '#ffc168', '#72d8d4'


def render(path, output, probe):
    a = np.load(path)
    centers = a['centers']
    n = len(centers)
    dt = float(a['dt'])
    delta = centers[:,:2].mean(1) - centers[:,2:].mean(1)
    angle = np.degrees(np.arctan2(delta[:,2], np.linalg.norm(delta[:,:2], axis=1)))
    extent = a['half_extents']
    right = np.array([.35,.937,0.])
    toward = np.array([.937,-.35,.34])
    toward /= np.linalg.norm(toward)
    up = np.cross(toward,right)
    def project(point):
        return (round(285 + 1030 * np.dot(point,right)), round(511 - 1030 * np.dot(point,up)))
    def side(point):
        return (round(845 + 1030 * point[1]), round(511 - 1030 * point[2]))
    def frame(i):
        im = Image.new('RGB',(WIDTH,HEIGHT),'#15212c')
        d = ImageDraw.Draw(im)
        def label(x,y,text,size=18,color=INK):
            d.text((x,y),text,fill=color,font=FONTS[size],anchor='lt')
        label(26,19,'从边缘提起 PE 袋装折叠衣物',33)
        label(26,70,('受控指垫  ·  PPO 训练用包裹动力学' if probe else '训练检查点  ·  YAM 实际指尖轨迹')+'  |  C5  ·  500 g',18,SOFT)
        label(1025,32,f'{(i+1)*dt:05.2f} s',21)
        d.line((0,112,WIDTH,112),fill='#415261')
        d.line((565,112,565,535),fill='#415261')
        label(26,130,'四块软内容物与指垫',21)
        label(590,130,'正侧视  ·  两排子块中心连线',21)
        for j in range(-4,5):
            for u,v in (([j*.05,-.2,0],[j*.05,.2,0]),([-.2,j*.05,0],[.2,j*.05,0])):
                d.line((project(u),project(v)),fill='#2f404e')
        c=centers[i]; R=a['frame'][i]; h=a['half_height'][i]
        pad=a['pads'][i]; captured=a['capture'][i]>.5
        surfaces=[]
        for k in range(4):
            verts=c[k]+(SIGNS*np.r_[.5*extent[:2],h[k]])@R.T
            for f,indices in enumerate(FACES):
                xyz=verts[list(indices)]
                surfaces.append((np.dot(xyz.mean(0),toward),list(map(project,xyz)),f,k))
        for _,poly,f,k in sorted(surfaces,key=lambda p:p[0]):
            light=(.73,.87,.77,.82,.60,1.)[f]
            color=tuple(int(x*light) for x in ((81,173,173) if k%2 else (102,193,189)))
            d.polygon(poly,fill=color,outline='#304b56',width=2)
        ends=np.array([c[:2].mean(0),c[2:].mean(0)])
        if probe:
            w=a['weights']
            mounts=w@c + (w@h)[:,None]*R[:,2]
            for k in range(2):
                if captured[k]:
                    d.line([project(pad[k]),project(mounts[k])],fill=ORANGE,width=3)
                    d.line([side(pad[k]),side(mounts[k])],fill='#bb935e',width=2)
        for p in pad:
            for method in (project,side):
                x,y=method(p); d.rounded_rectangle((x-5,y-11,x+5,y+2),radius=2,fill=ORANGE)
        d.line((590,511,1120,511),fill=SOFT,width=2)
        label(1058,482,'台面',16,SOFT)
        x,y=side(ends[0])
        for j in range(0,135,10):
            d.line((x+j,y,x+j+5,y),fill=SOFT)
            d.line((x,y+j,x,y+j+5),fill=SOFT)
        d.line(list(map(side,ends)),fill=TEAL,width=5)
        for pt in ends:
            u,v=side(pt); d.ellipse((u-5,v-5,u+5,v+5),fill=TEAL)
        label(865,172,f'倾角  {angle[i]:.1f}°',27,TEAL)
        label(865,216,'水平 0°     竖直 90°',16,SOFT)
        d.rectangle((0,541,WIDTH,HEIGHT),fill='#202f3a')
        label(26,557,f'双指捕获   {captured.sum()} / 2',21,ORANGE)
        label(316,557,f'模型离台净空   {1000*a["clearance"][i]:.1f} mm',21)
        label(735,557,f'台面支撑   {a["support"][i]:.2f} N',21)
        label(26,594,'PE 袋膜以夹持和拉力等效表示。橙色为指垫，青色为四块内容物。',16,SOFT)
        return im
    selected=np.minimum((np.arange(int(np.ceil(n*dt*25)))/25/dt).round().astype(int),n-1)
    selected[-1]=n-1
    cmd=['ffmpeg','-v','error','-n','-f','rawvideo','-pix_fmt','rgb24','-s',f'{WIDTH}x{HEIGHT}',
         '-r','25','-i','pipe:','-c:v','libx264','-threads','3','-crf','19','-pix_fmt','yuv420p',
         '-movflags','+faststart',str(output)]
    with subprocess.Popen(cmd,stdin=subprocess.PIPE) as encoder:
        for i in selected:
            encoder.stdin.write(frame(i).tobytes())
        encoder.stdin.close()
        if encoder.wait():
            raise RuntimeError('video encoding failed')
    frame(n-1).save(output.with_suffix('.png'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trajectory',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--probe',action='store_true')
    args=parser.parse_args()
    render(args.trajectory,args.output,args.probe)
