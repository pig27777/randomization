"""Render saved trial states on CPU (Mesa llvmpipe), without using the GPU lock."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess

# Explicit Mesa surfaceless display avoids probing the busy NVIDIA device.
os.environ["MUJOCO_GL"]="disable"
os.environ["PYOPENGL_PLATFORM"]="egl"
os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"]="/usr/share/glvnd/egl_vendor.d/50_mesa.json"
os.environ["LIBGL_ALWAYS_SOFTWARE"]="1"
import mujoco
import numpy as np
from OpenGL import EGL, GL


class SoftwareContext:
    def __init__(self,width,height):
        address=EGL.eglGetProcAddress(b"eglGetPlatformDisplayEXT")
        get_display=ctypes.CFUNCTYPE(ctypes.c_void_p,ctypes.c_uint,ctypes.c_void_p,ctypes.POINTER(ctypes.c_int))(address)
        self.display=ctypes.cast(get_display(0x31DD,None,None),EGL.EGLDisplay)
        major,minor=ctypes.c_int(),ctypes.c_int()
        if not EGL.eglInitialize(self.display,major,minor):
            raise RuntimeError("Mesa surfaceless EGL initialization failed")
        EGL.eglBindAPI(EGL.EGL_OPENGL_API)
        attrs=(EGL.EGLint*13)(EGL.EGL_SURFACE_TYPE,EGL.EGL_PBUFFER_BIT,EGL.EGL_RENDERABLE_TYPE,EGL.EGL_OPENGL_BIT,
                              EGL.EGL_RED_SIZE,8,EGL.EGL_GREEN_SIZE,8,EGL.EGL_BLUE_SIZE,8,EGL.EGL_DEPTH_SIZE,24,EGL.EGL_NONE)
        configs=(EGL.EGLConfig*1)();count=EGL.EGLint()
        if not EGL.eglChooseConfig(self.display,attrs,configs,1,count) or count.value<1:
            raise RuntimeError("no Mesa OpenGL EGL config")
        pb=(EGL.EGLint*5)(EGL.EGL_WIDTH,width,EGL.EGL_HEIGHT,height,EGL.EGL_NONE)
        self.surface=EGL.eglCreatePbufferSurface(self.display,configs[0],pb)
        self.context=EGL.eglCreateContext(self.display,configs[0],EGL.EGL_NO_CONTEXT,None)
        self.make_current()
        self.renderer=GL.glGetString(GL.GL_RENDERER).decode()
        if "llvmpipe" not in self.renderer.lower() and "soft" not in self.renderer.lower():
            raise RuntimeError(f"rendering must stay on CPU: {self.renderer}")

    def make_current(self):
        if not EGL.eglMakeCurrent(self.display,self.surface,self.surface,self.context):
            raise RuntimeError("could not make Mesa context current")

    def free(self):
        if getattr(self,"context",None):
            EGL.eglMakeCurrent(self.display,EGL.EGL_NO_SURFACE,EGL.EGL_NO_SURFACE,EGL.EGL_NO_CONTEXT)
            EGL.eglDestroyContext(self.display,self.context)
            EGL.eglDestroySurface(self.display,self.surface)
            self.context=None


def render(run,output,tetra=False,focus=False):
    if output.exists():raise FileExistsError(output)
    from mujoco.rendering.classic import gl_context
    gl_context.GLContext=SoftwareContext
    result=json.loads((run/"result.json").read_text())
    trajectory=np.load(run/"trajectory.npz")
    states=trajectory["qpos"]
    times=trajectory["time"] if "time" in trajectory else np.arange(len(states))*.2+.001
    model=mujoco.MjModel.from_xml_path(str(run/"model.xml"))
    model.vis.global_.offwidth=640;model.vis.global_.offheight=480
    data=mujoco.MjData(model)
    mujoco.mj_kinematics(model,data)
    bodies=model.flex_vertbodyid
    rest_vertices=data.xpos[bodies]+np.einsum('nij,nj->ni',data.xmat[bodies].reshape(-1,3,3),model.flex_vert)
    renderer=mujoco.Renderer(model,480,640)
    renderer_name=renderer._gl_context.renderer
    option=mujoco.MjvOption()
    option.geomgroup[3]=False
    option.flags[mujoco.mjtVisFlag.mjVIS_TENDON]=False
    option.flags[mujoco.mjtVisFlag.mjVIS_FLEXFACE]=True
    detail_option=mujoco.MjvOption()
    detail_option.geomgroup[3]=False
    detail_option.flags[mujoco.mjtVisFlag.mjVIS_TENDON]=False
    detail_option.flags[mujoco.mjtVisFlag.mjVIS_FLEXFACE]=not tetra
    detail_option.flags[mujoco.mjtVisFlag.mjVIS_FLEXSKIN]=not tetra
    tets=model.flex_elem.reshape(-1,4) if tetra else None
    if tetra:model.flex_rgba[:,3]=.12
    full=mujoco.MjvCamera();full.lookat[:]=[0,-.02,.20];full.distance=1.35;full.azimuth=-45;full.elevation=-25
    detail=mujoco.MjvCamera();detail.distance=.67;detail.azimuth=-45;detail.elevation=-30
    if focus:
        option.geomgroup[:]=False;detail_option.geomgroup[:]=False
        full.distance=.46;full.azimuth=-65;full.elevation=-20
        detail.distance=.16;detail.azimuth=-65;detail.elevation=-15
        final=trajectory['final_qpos'];data.qpos[:]=final;mujoco.mj_kinematics(model,data)
        final_vertices=data.xpos[bodies]+np.einsum('nij,nj->ni',data.xmat[bodies].reshape(-1,3,3),model.flex_vert)
        def dets(p):
            t=p[tets];return np.linalg.det(np.stack([t[:,i]-t[:,0] for i in (1,2,3)],axis=2))
        min_tet=int(np.argmin(dets(final_vertices)/dets(rest_vertices)))
        focus_reference=rest_vertices[tets[min_tet]].mean(0)
    fps=20
    cfg=result["config"]
    model_label='Segmented v1.3' if result.get('backend')=='segmented' else '3D flex'
    title=f'{model_label} | {"TETRAHEDRA | " if tetra else ""}{cfg["category"]} | yaw {cfg["yaw"]:g} deg | pinch {cfg["pinch_coeff"]:g} | {result["outcome"]}'
    if focus:title=f'{cfg.get("shape","box").upper()} tetrahedra | 10x slow | RED = inverted | final frame held 3s'
    label=f"drawtext=text='{title}':fontcolor=white:fontsize=20:x=16:y=12:box=1:boxcolor=black@0.65"
    if focus:label+=",tpad=stop_mode=clone:stop_duration=3"
    process=subprocess.Popen(["ffmpeg","-hide_banner","-loglevel","error","-f","rawvideo","-pixel_format","rgb24",
                              "-video_size","1280x480","-framerate",str(fps),"-i","pipe:0","-vf",label,
                              "-c:v","libx264","-preset","fast","-crf","22","-pix_fmt","yuv420p","-movflags","+faststart",str(output)],stdin=subprocess.PIPE)
    begin=max(0.,result['sim_time_s']-.8) if focus else 0.
    slow=10 if focus else 1
    frames=max(1,round((result["sim_time_s"]-begin)*slow*fps)+1)
    for i in range(frames):
        t=result["sim_time_s"] if i==frames-1 else begin+i/fps/slow
        j=max(0,min(len(times)-2,int(np.searchsorted(times,t)-1)))
        a=float(np.clip((t-times[j])/max(times[j+1]-times[j],1e-12),0,1)) if len(times)>1 else 0.
        data.qpos[:]=(1-a)*states[j]+a*states[min(j+1,len(states)-1)]
        mujoco.mj_forward(model,data)
        if model.nflex:
            detail.lookat[:]=data.flexvert_xpos.mean(0)
        elif result.get('backend')=='segmented':
            content_geoms=[g for g in range(model.ngeom) if model.geom(g).name.startswith('s') and model.geom(g).name[1:].isdigit()]
            detail.lookat[:]=data.geom_xpos[content_geoms].mean(0)
        else:
            detail.lookat[:]=[0,0,.15]
        if focus:
            full.lookat[:]=data.flexvert_xpos.mean(0)
            detail.lookat[:]=data.flexvert_xpos[tets[min_tet]].mean(0)
        def edges(cutaway):
            if not tetra:return
            selected=tets[np.mean(rest_vertices[tets,1],axis=1)>=np.median(rest_vertices[:,1])] if cutaway else tets
            if focus and cutaway:
                selected=tets[np.linalg.norm(rest_vertices[tets].mean(1)-focus_reference,axis=1)<.035]
            unique={}
            for tet in selected:
                ref=rest_vertices[tet];now=data.flexvert_xpos[tet]
                refdet=np.linalg.det(np.stack([ref[k]-ref[0] for k in (1,2,3)]))
                det=np.linalg.det(np.stack([now[k]-now[0] for k in (1,2,3)]))/refdet
                for a,b in ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3)):
                    edge=tuple(sorted((int(tet[a]),int(tet[b]))))
                    unique[edge]=min(unique.get(edge,1.),det)
            for (a,b),det in unique.items():
                geom=renderer.scene.geoms[renderer.scene.ngeom]
                color=[1.,.05,.03,1.] if det<=0 else [.1,.7,.85,.55 if focus else .8]
                mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_LINE,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array(color))
                mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_LINE,5. if det<=0 else 1.2,data.flexvert_xpos[a],data.flexvert_xpos[b])
                renderer.scene.ngeom+=1
            if focus:
                red_vertices=set()
                for (a,b),det in unique.items():
                    if det<=0:red_vertices.update((a,b))
                for node in red_vertices:
                    geom=renderer.scene.geoms[renderer.scene.ngeom]
                    mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_SPHERE,np.array([.0012]*3),data.flexvert_xpos[node],np.eye(3).ravel(),np.array([1.,.05,.03,1.]))
                    renderer.scene.ngeom+=1
        renderer.update_scene(data,camera=full,scene_option=option);edges(False);left=renderer.render().copy()
        renderer.update_scene(data,camera=detail,scene_option=detail_option);edges(True);right=renderer.render().copy()
        frame=np.concatenate([left,right],axis=1)
        process.stdin.write(frame.tobytes())
        if i==0:
            (output.parent/(output.stem+"_first.ppm")).write_bytes(b"P6\n1280 480\n255\n"+frame.tobytes())
    process.stdin.close()
    if process.wait()!=0:raise RuntimeError("ffmpeg failed")
    renderer.close()
    (output.parent/(output.stem+"_render.json")).write_text(json.dumps(dict(renderer=renderer_name,fps=fps,frames=frames,
        source=str(run),state_interval_s=float(np.median(np.diff(times))) if len(times)>1 else None,
        interpolated=True,physics_recomputed=False,tetrahedra=tetra,focus=focus,slow_factor=slow),indent=2))
    print(output,renderer_name,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("run",type=Path);parser.add_argument("output",type=Path)
    parser.add_argument('--tetra',action='store_true')
    parser.add_argument('--focus',action='store_true')
    args=parser.parse_args();render(args.run,args.output,args.tetra,args.focus)
