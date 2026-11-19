"""CPU-only close-up replay of two measured whole-bag surface-rub cases."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'experience')]
from yam_platform.render_video import SoftwareContext
from mujoco.rendering.classic import gl_context
import mujoco
import numpy as np


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('left',type=Path)
    parser.add_argument('right',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--slow',type=float,default=5.)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if args.slow<=0:parser.error('slow factor must be positive')
    gl_context.GLContext=SoftwareContext
    scenes=[]
    for directory in (args.left,args.right):
        cfg=json.loads((directory/'config.json').read_text())
        states=np.load(directory/'trajectory.npz')
        m=mujoco.MjModel.from_binary_path(str(directory/'model.mjb'))
        m.vis.global_.offwidth=640;m.vis.global_.offheight=480
        if m.nmat:
            m.mat_texid[:]=-1
            m.mat_rgba[:]=[.66,.72,.78,1.]
        m.flex_rgba[:]=[.66,.72,.78,1.]
        d=mujoco.MjData(m)
        renderer=mujoco.Renderer(m,480,640)
        option=mujoco.MjvOption();option.geomgroup[3]=False
        option.flags[mujoco.mjtVisFlag.mjVIS_FLEXFACE]=True
        option.flags[mujoco.mjtVisFlag.mjVIS_FLEXEDGE]=True
        camera=mujoco.MjvCamera();camera.azimuth=-90;camera.elevation=-18;camera.distance=.24
        ids=np.unique(np.concatenate(cfg['rows']))
        camera.lookat[:]=states['initial_film'][ids].mean(0)+[0,0,.003]
        scenes.append((cfg,states,m,d,renderer,option,camera))
    end=min(float(s[1]['time'][-1]) for s in scenes)
    fps=25;frames=round(end*args.slow*fps)+1
    caption_dir=args.output.parent/(args.output.stem+'-captions')
    caption_dir.mkdir()
    force=scenes[0][0]['target_force_each_N']
    left,right=[scene[0] for scene in scenes]
    header=caption_dir/'header.txt'
    header.write_text(f'当前整袋表面搓动｜每指目标压力{force:g} N｜{args.slow:g}倍慢放\n'
        f'左：设定μ={left["friction"]:g}；右：设定μ={right["friction"]:g}；每指搓动{left["distance_per_pad_m"]*1000:g} mm\n'
        '过程：下压→横向搓动→离开；橙点为固定材料节点\n')
    filters=[f'pad=iw:ih+100:0:100:color=0x141920']
    font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    filters.append(f'drawtext=fontfile={font}:textfile={header}:fontcolor=white:fontsize=23:line_spacing=8:x=16:y=6')
    command=['/usr/bin/ffmpeg','-hide_banner','-loglevel','error','-f','rawvideo','-pixel_format','rgb24','-video_size','1280x480','-framerate',str(fps),'-i','pipe:0','-vf',','.join(filters),'-c:v','libx264','-preset','fast','-crf','22','-pix_fmt','yuv420p','-movflags','+faststart',str(args.output)]
    process=subprocess.Popen(command,stdin=subprocess.PIPE)
    for frame in range(frames):
        when=min(end,frame/(fps*args.slow))
        images=[]
        for cfg,states,m,d,renderer,option,camera in scenes:
            times=states['time'];index=max(0,min(len(times)-2,int(np.searchsorted(times,when)-1)))
            fraction=float(np.clip((when-times[index])/max(times[index+1]-times[index],1e-12),0,1))
            d.qpos[:]=(1-fraction)*states['qpos'][index]+fraction*states['qpos'][index+1]
            mujoco.mj_kinematics(m,d);mujoco.mj_flex(m,d);mujoco.mj_camlight(m,d)
            renderer.update_scene(d,camera=camera,scene_option=option)
            for ids in cfg['rows']:
                for node in ids:
                    geom=renderer.scene.geoms[renderer.scene.ngeom]
                    mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_SPHERE,np.array([.0007]*3),d.flexvert_xpos[node],np.eye(3).ravel(),np.array([1.,.35,.02,1.]))
                    renderer.scene.ngeom+=1
            images.append(renderer.render().copy())
        process.stdin.write(np.concatenate(images,axis=1).tobytes())
    process.stdin.close()
    if process.wait():raise RuntimeError('FFmpeg failed')
    metadata=dict(cpu_renderer=[s[4]._gl_context.renderer for s in scenes],sources=[str(args.left),str(args.right)],fps=fps,frames=frames,slow_factor=args.slow,simulation_end_s=end,physics_recomputed=False,interpolated=True,appearance_only='printing hidden; fixed material nodes marked orange; no geometric exaggeration')
    for scene in scenes:scene[4].close()
    args.output.with_suffix('.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps(metadata),flush=True)


if __name__=='__main__':main()
