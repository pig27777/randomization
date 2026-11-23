"""Render the actual contact attempt without waiting for an acceptance review."""
import argparse
from pathlib import Path
import subprocess
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parent/"multi-parcel-translucent-r1"),str(HERE.parent/"umi-skirt-c-r1")]
from studio import render
from native_contact import HALF_PAD, sheets, CENTER, CLOTH, lengths
from limiter import compile_limit
from project import project

COLOR=np.array([.90,.83,.69],dtype=np.float32)
COUNT=240


def fade(x):
    u=np.clip(x,0,1)
    return u*u*(3-2*u)


def box_surface(nx=7,ny=5):
    x,y=np.meshgrid(np.linspace(-1.,1.,nx),np.linspace(-1.,1.,ny),indexing="ij")
    xy=np.column_stack((x.ravel(),y.ravel()))
    n=len(xy)
    faces=[]
    for i in range(nx-1):
        for j in range(ny-1):
            a=i*ny+j;b=a+ny;c=b+1;d=a+1
            faces.extend(((a,b,c),(a,c,d),(n+a,n+c,n+b),(n+a,n+d,n+c)))
    rim=np.r_[np.arange(ny),np.arange(ny-1+ny,n,ny),np.arange(n-2,n-ny-1,-1),
              np.arange(n-2*ny,0,-ny)]
    for a,b in zip(rim,np.roll(rim,-1),strict=True):
        faces.extend(((a,n+a,b),(b,n+a,n+b)))
    return xy,np.asarray(faces)


def pad_ceiling(points, upper, boxes, actual_tops, rim=.014):
    """Smooth depression beneath the swept pad footprints, evaluated in XY."""
    out=upper.copy()
    for (center,half),top in zip(boxes,actual_tops,strict=True):
        dx=np.maximum(abs(points-center[:2])-half,0.)
        margin=np.linalg.norm(dx,axis=1)
        blend=1-fade(margin/rim)
        out-=np.maximum(0.,out-top)*blend
    return out


def proxy_geometry(path, reference):
    ref,initial,faces,edges=sheets()
    # The upper and lower sheets share the seal, so it belongs to the upper array.
    first_count=len(np.unique(ref[:,0]))*len(np.unique(ref[:,1]))
    top=np.arange(len(initial))<first_count
    xy=initial[:,:2]-CENTER
    normalized=np.abs((ref[:,:2]-CENTER)/[.145,.195])
    radius=normalized.max(1)
    seam=(top & (radius>.9999999))
    held_patch=top & (abs(xy[:,0])<.010) & (abs(xy[:,1])<.014)
    bxy,solid_faces=box_surface()
    orig=path["centers"][0] - path["centers"][0].mean(0)
    dd=(ref[:,None,:2]-CENTER)-orig[None,:,:2]
    weights=np.exp(-.5*np.sum(dd**2,axis=2)/.06**2)
    weights/=weights.sum(axis=1,keepdims=True)
    film_mesh,contents=[],[]
    scratch,limit=compile_limit()
    edges=np.ascontiguousarray(edges,dtype=np.int64)
    rest=np.ascontiguousarray(np.minimum(lengths(ref,edges),lengths(initial,edges)))
    for i in range(len(path["time"])):
        centers=path["centers"][i]
        frame=path["frame"][i]
        h=path["half"][i]
        half=np.tile(CLOTH/[4.,4.,2.],(4,1))
        half[:,2]=h
        apex=path["pads"][i].mean(0)
        mean=centers.mean(0)
        local=(apex-mean)@frame
        clearance=float(path["clearance"][i])
        captured=bool(path["capture"][i].all())
        supported=1-fade(clearance/.033)
        rest_shape=initial.copy()
        rest_shape[:,:2]-=CENTER
        rest_shape[:,2]-=.010
        rest_shape[top,2]-=(.010-h.mean())
        rest_shape[~top,2]+=(.010-h.mean())
        on_table=(rest_shape@frame.T)+mean
        boxes=[(x,np.array([.0048,.0105])) for x in reference["pad_box_centers"][i*20]]
        ceilings=path["pads"][i,:,2]-.0009
        p0=on_table[top,2].copy()
        dented=pad_ceiling(on_table[top,:2],p0,boxes,ceilings)
        on_table[top,2]+=supported*(dented-p0)
        # Use physical gathered slack and distance to the actual jaw opening to
        # draw the local fabric inward and raise the centre of the top sheet.
        gap=np.linalg.norm(path["pads"][i,1]-path["pads"][i,0])
        slack=float(path["slack"][i].sum())
        fold=fade((.033-gap)/.024)*fade(slack/.003)*supported
        u=on_table[top,:2]-apex[:2]
        footprint=np.exp(-.5*((u[:,0]/.022)**2+(u[:,1]/.023)**2))
        on_table[top,0]-=.19*fold*footprint*u[:,0]
        ridge=.005*fold*np.exp(-.5*((u[:,0]/max(.001,gap*.19))**2+(u[:,1]/.018)**2))
        on_table[top,2]+=ridge
        top_now=on_table[top,2].copy()
        on_table[top,2]=pad_ceiling(on_table[top,:2],top_now,boxes,ceilings,rim=.006)

        # Carry with the actual four centres and actual pad midpoint.
        hanging=rest_shape.copy()
        hanging[:,:2]*=[.983,.990]
        hanging[:,:2]+=local[:2]*(1-radius[:,None])
        seam_height=.0048
        hanging[top,2]=local[2]*(1-radius[top])+seam_height*radius[top]
        hanging[~top,2]=-.0112
        hanging[seam,2]=seam_height
        hanging[held_patch]=local+np.column_stack((ref[held_patch,:2]-CENTER,np.zeros(held_patch.sum())))
        hanging=hanging@frame.T+mean
        alpha=(1-supported) if captured else 0.
        film=(1-alpha)*on_table+alpha*hanging
        residual=centers-mean-orig@frame.T
        locked=held_patch & (alpha>.9999)
        film[~locked] += (weights@residual)[~locked]
        if alpha>.98:
            film=project(film,ref,faces,edges,locked,centers,frame,half,finite_contact=True,initial=initial)
        mobility=np.ascontiguousarray(~locked,dtype=np.float64)
        film=np.ascontiguousarray(film)
        limit(film,edges,rest,mobility,len(edges))
        blocks=[]
        for c,b in zip(centers,half,strict=True):
            xy0=bxy*b[:2]
            raw=np.column_stack((np.tile(xy0,(2,1)),np.repeat([b[2],-b[2]],len(xy0))))
            points=(raw@frame.T)+c
            count=len(xy0)
            z=points[:count,2]
            dented=pad_ceiling(points[:count,:2],z,boxes,ceilings-.0004)
            points[:count,2]=np.maximum(points[count:,2]+.0007,z+(dented-z)*supported)
            blocks.append(points)
        film_mesh.append(film)
        contents.append(blocks)
    return dict(film=np.array(film_mesh),faces=faces,contents=np.array(contents),
                content_faces=np.tile(solid_faces,(4,1,1)))


def record_side(side, output, reuse_proxy_skin):
    reference=np.load(HERE/"q_ref_top_center.npz")
    if side=="left":
        native=np.load(HERE/"native_preview.npz")
        v,F=native["film"],native["faces"]
        inside=native["contents"][:,None]
        inside_faces=native["content_faces"][None]
    else:
        cached=HERE/"proxy_skin_preview.npz"
        if reuse_proxy_skin:
            with np.load(cached) as stored:
                mesh=dict(stored)
        else:
            raw=np.load(HERE/"proxy_probe.npz")
            mesh=proxy_geometry(raw,reference)
            np.savez_compressed(cached,**mesh)
        v,F,inside,inside_faces=(mesh[k] for k in ("film","faces","contents","content_faces"))
    k=len(v)*20
    render(output/f"{side}_contact.mp4",v,F,reference["links"][:k:20],reference["quats"][:k:20],
           reference["names"],reference["pads"][:k:20],inside,inside_faces,COLOR,grasp_detail=True)


def compose(output):
    movie=output/"compare_real_grasp_skirt_c.mp4"
    font="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    filters=("[0:v][1:v]hstack,drawbox=x=0:y=0:w=iw:h=74:color=0x18252f:t=fill,"
             "drawbox=x=0:y=648:w=iw:h=72:color=0x18252f:t=fill,"
             "drawbox=x=959:y=0:w=2:h=ih:color=0x537887:t=fill,"
             f"drawtext=fontfile={font}:text='MuJoCo 4 | Real Contact Friction Grasp':fontsize=27:fontcolor=white:x=24:y=11,"
             f"drawtext=fontfile={font}:text='Frozen Passive Proxy + PBD Film':fontsize=27:fontcolor=white:x=984:y=11,"
             f"drawtext=fontfile={font}:text='250 g folded skirt | shared vertical centre approach':fontsize=17:fontcolor=0xd9e2e7:x=24:y=45,"
             f"drawtext=fontfile={font}:text='250 g folded skirt | model-based contact and capture':fontsize=17:fontcolor=0xd9e2e7:x=984:y=45,"
             f"drawtext=fontfile={font}:text='%{{pts\\:flt}} s   |   Direct export of the contact attempt':fontsize=23:fontcolor=white:x=24:y=667")
    subprocess.run(["ffmpeg","-v","error","-n","-i",str(output/"left_contact.mp4"),
      "-i",str(output/"right_contact.mp4"),"-filter_complex",filters,"-an","-c:v","libx264",
      "-preset","fast","-crf","17","-pix_fmt","yuv420p","-movflags","+faststart",str(movie)],check=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("stage",choices=["left","right","compose"])
    p.add_argument("--output-dir",type=Path,default=HERE)
    p.add_argument("--reuse-proxy-skin",action="store_true")
    a=p.parse_args()
    a.output_dir.mkdir(parents=True,exist_ok=True)
    if a.stage=="compose":
        compose(a.output_dir)
    else:
        record_side(a.stage,a.output_dir,a.reuse_proxy_skin)
