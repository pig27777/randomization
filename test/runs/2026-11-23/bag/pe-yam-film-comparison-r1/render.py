"""Common studio, YAM CAD, PE texture and camera. All geometry is driven from recorded frames."""
from pathlib import Path
import argparse
import sys
import subprocess
from xml.etree import ElementTree as ET
ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT))
from experience.yam_platform.render_video import SoftwareContext
from experience.yam_platform.robot import ASSET
from mujoco.rendering.classic import gl_context
import mujoco as mj
import numpy as np
from PIL import Image,ImageDraw,ImageFilter,ImageChops
gl_context.GLContext=SoftwareContext
W,H=960,720
def string(values):
    return ' '.join(map(str,np.asarray(values).ravel()))
def make_model(v,f,material_xy=None):
    root=ET.parse(ASSET/'yam_linear_4310.xml').getroot()
    root.find('compiler').set('meshdir',str(ASSET/'assets'))
    visual=root.find('visual')
    ET.SubElement(visual,'global',offwidth=str(W),offheight=str(H),fovy='34')
    ET.SubElement(visual,'headlight',ambient='.38 .38 .40', diffuse='.23 .24 .24',specular='.05 .05 .05')
    ET.SubElement(visual,'quality',offsamples='4',shadowsize='4096')
    ET.SubElement(visual,'rgba',haze='.44 .49 .54 1')
    world=root.find('worldbody');world.find("body[@name='arm']").set('pos','-.3475 -.31 0')
    for name in ('left_pad','right_pad'):
        pad=world.find(f".//body[@name='{name}']")
        ET.SubElement(pad,'geom',name=name+'_rubber',type='box',pos='0 -.004 -.1244',
                      size='.010 .004 .020',rgba='.19 .20 .21 1',group='2',contype='0',conaffinity='0')
    asset=root.find('asset')
    ET.SubElement(asset,'texture',name='studio',type='skybox',builtin='gradient',rgb1='.35 .42 .49',rgb2='.73 .77 .79',height='256',width='256')
    ET.SubElement(asset,'texture',name='PE',type='2d',file=str(ROOT/'bag_sim/textures/pe.png'))
    ET.SubElement(asset,'material',name='mailfilm',texture='PE',rgba='.97 .99 1 .95',specular='.25',shininess='.25',texuniform='false')
    xy=v[:,:2] if material_xy is None else material_xy
    uv=(xy-xy.min(0))/np.ptp(xy,axis=0)
    attributes=dict(vertex=string(v-v.mean(0)),face=string(f),texcoord=string(uv),normal=string(normals(v,f)))
    ET.SubElement(asset,'mesh',name='dynamic',**attributes)
    ET.SubElement(world,'geom',name='surface',type='mesh',mesh='dynamic',material='mailfilm',mass='0',contype='0',conaffinity='0')
    ET.SubElement(world,'geom',name='desk',type='box',pos='0 0 -.005',size='.2975 .65 .005',rgba='.53 .56 .57 1')
    ET.SubElement(world,'geom',type='box',pos='-.352 -.31 -.026',size='.055 .072 .026',rgba='.25 .29 .32 1')
    ET.SubElement(world,'geom',type='plane',size='0 0 .04',pos='0 0 -.065',rgba='.40 .43 .46 1')
    ET.SubElement(world,'light',pos='.3 -.65 1.4',dir='-.20 .30 -1',directional='true',diffuse='.16 .16 .15',specular='.10 .10 .09')
    ET.SubElement(world,'light',pos='-.7 .45 1',dir='.5 -.4 -1',directional='true',castshadow='false',diffuse='.43 .44 .46',specular='.15 .16 .17')
    if material_xy is not None:
        # Low side-light reveals differences in slope across the actual folds.
        visual.find('headlight').attrib.update(ambient='.19 .20 .21',diffuse='.16 .16 .16')
        lamps=world.findall('light')
        lamps[-2].attrib.update(pos='.7 -.15 .65',dir='-.85 -.2 -.39',diffuse='.94 .91 .88',specular='.32 .32 .32')
        lamps[-1].attrib.update(diffuse='.24 .26 .29')
    spec=mj.MjSpec.from_string(ET.tostring(root,encoding='unicode'))
    if material_xy is not None:
        mesh=spec.mesh('dynamic')
        mesh.usernormal=crease_normals(v,f).ravel().tolist()
        mesh.userfacenormal=np.arange(f.size).tolist()
    return spec.compile()
def normals(v,f):
    n=np.zeros_like(v)
    a=np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]])
    for i in range(3):np.add.at(n,f[:,i],a)
    return n/np.maximum(1e-16,np.linalg.norm(n,axis=1,keepdims=True))
def crease_normals(v,f):
    # Average adjacent surface normals only across smooth bends. Heat seals
    # and sharp material folds retain distinct normals on each side.
    tri=v[f]
    area=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    unit=area/np.maximum(np.linalg.norm(area,axis=1)[:,None],1e-16)
    adjacent=[[] for _ in v]
    for i,face in enumerate(f):
        for j in face:adjacent[j].append(i)
    shading=np.empty((len(f),3,3))
    for i,face in enumerate(f):
        for k,j in enumerate(face):
            local=np.asarray(adjacent[j])
            shared=area[local[unit[local]@unit[i]>.94]].sum(0)
            shading[i,k]=shared/max(np.linalg.norm(shared),1e-16)
    return shading.reshape(-1,3)
def raw_cmd(out):
    return ['ffmpeg','-v','error','-n','-f','rawvideo','-pixel_format','rgb24','-video_size',f'{W}x{H}','-framerate','25','-i','-','-an','-c:v','libx264','-preset','veryfast','-crf','15','-pix_fmt','yuv420p','-movflags','+faststart',str(out)]
def studio_exposure(renderer):
    # Mix diffuse fill with the softened, actually cast OpenGL shadow on both cameras.
    flags=renderer.scene.flags
    flags[mj.mjtRndFlag.mjRND_SHADOW]=0
    fill=Image.fromarray(renderer.render().copy())
    flags[mj.mjtRndFlag.mjRND_SHADOW]=1
    key=Image.fromarray(renderer.render().copy())
    attenuation=ImageChops.subtract(fill,key).filter(ImageFilter.GaussianBlur(3.0)).point(lambda a:round(.36*a))
    return ImageChops.subtract(fill,attenuation)
def render(out,v,f,links,quats,names,pads,preview,frame_index=0,material_xy=None,hold_detail=False):
    m=make_model(v[0],f,material_xy);d=mj.MjData(m);mj.mj_forward(m,d)
    opt=mj.MjvOption();opt.geomgroup[3]=0;opt.sitegroup[:]=0
    cam=mj.MjvCamera();cam.azimuth=135;cam.elevation=-19;cam.distance=1.34;cam.lookat[:]=[-.18,-.25,.237]
    detail=mj.MjvCamera();detail.azimuth=135;detail.elevation=-21;detail.distance=.42
    if hold_detail:detail.distance=.79
    mid=m.mesh('dynamic').id; gid=m.geom('surface').id
    n=len(v[0]);vi=m.mesh_vertadr[mid];ni=m.mesh_normaladr[mid]
    assert n==m.mesh_vertnum[mid]
    normal_count=m.mesh_normalnum[mid]
    bid=[m.body(str(name)).id for name in names if name in {m.body(i).name for i in range(m.nbody)}]
    index=[list(names).index(m.body(b).name) for b in bid]
    # Restore CAD geom transforms relative to each link, including MuJoCo mesh centering.
    geoms=[g for g in range(m.ngeom) if m.geom_bodyid[g] in bid]
    parent=[index[bid.index(m.geom_bodyid[g])] for g in geoms]
    mat=np.empty(9)
    with mj.Renderer(m,H,W) as ren:
        print(ren._gl_context.renderer,flush=True)
        proc=None if preview else subprocess.Popen(raw_cmd(out),stdin=subprocess.PIPE)
        try:
            for i in ((frame_index,) if preview else range(len(v)-1)):
                rot=[]
                for quat in quats[i]:
                    mj.mju_quat2Mat(mat,quat);rot.append(mat.copy().reshape(3,3))
                rot=np.asarray(rot)
                for g,b in zip(geoms,parent):
                    mj.mju_quat2Mat(mat,m.geom_quat[g])
                    d.geom_xpos[g]=links[i,b]+rot[b]@m.geom_pos[g]
                    d.geom_xmat[g]=(rot[b]@mat.reshape(3,3)).ravel()
                center=v[i].mean(0)
                m.mesh_vert[vi:vi+n]=v[i]-center
                m.mesh_normal[ni:ni+normal_count]=normals(v[i],f) if material_xy is None else crease_normals(v[i],f)
                d.geom_xpos[gid]=center;d.geom_xmat[gid]=np.eye(3).ravel()
                mj.mjr_uploadMesh(m,ren._mjr_context,mid)
                ren.update_scene(d,camera=cam,scene_option=opt)
                frame=studio_exposure(ren)
                focus=pads[i].mean(0).copy()
                focus[2]=max(.028,focus[2]-.040) if i/25>=3.15 else .040
                if hold_detail:focus[2]=pads[i,:,2].mean()-.056
                detail.lookat[:]=focus
                ren.update_scene(d,camera=detail,scene_option=opt)
                inset=studio_exposure(ren).resize((264,198),Image.Resampling.LANCZOS)
                place=(18,H-284)
                ImageDraw.Draw(frame).rectangle((place[0]-2,place[1]-2,place[0]+265,place[1]+199),fill='#e9eded')
                frame.paste(inset,place)
                if preview:frame.save(out)
                else:proc.stdin.write(frame.tobytes())
        finally:
            if proc:
                proc.stdin.close()
                if proc.wait():raise RuntimeError('encoding error')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('data',type=Path);p.add_argument('output',type=Path);p.add_argument('--side',choices=['flex','proxy'],default='flex');p.add_argument('--preview',action='store_true')
    p.add_argument('--frame',type=int,default=0)
    a=p.parse_args();s=np.load(a.data)
    render(a.output,s['V_'+a.side],s['F'],s['links'],s['link_quat'],s['link_names'],s['pads'],a.preview,a.frame)
