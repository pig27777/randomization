"""Material-space sealed envelope and the four garment-block collision surfaces."""
import numpy as np
from pathlib import Path
import sys
RUN=Path(__file__).resolve().parent
sys.path.insert(0,str(RUN.parent/'pe-yam-film-comparison-r1'))
from record import make_scene,pad_faces
import mujoco as mj

BAG=np.array([.29,.39])
CLOTH=np.array([.22,.31,.020])
MASS=.065
def robot_state():
    q=np.load(RUN/'q_ref.npz')['q_path'][4000]
    r=make_scene(); k=r.kin;m=r.sim.m
    k.qpos[np.r_[r.qadr,r.fingers]]=q
    mj.mj_kinematics(m,k)
    ids=np.arange(m.body('arm').id,m.nbody)
    return ([m.body(int(i)).name for i in ids],k.xpos[ids].copy(),k.xquat[ids].copy(),pad_faces(r,k))

def envelope(n=8,rings=5,mirror=False):
    # Each meridian is a continuous strip of PE from the crown to the edge seal.
    perimeter=[]
    for a,b in [((-1,-1),(1,-1)),((1,-1),(1,1)),((1,1),(-1,1)),((-1,1),(-1,-1))]:
        perimeter.extend(np.asarray(a)+(np.asarray(b)-a)*u/n for u in range(n))
    uv=np.array(perimeter); count=len(uv)
    base=[]; faces=[]; side=[]; ring=[]
    def point(v,which,rho):
        base.append(v);side.append(which);ring.append(rho);return len(base)-1
    def layer(which):
        rows=[];center=point([0,0],which,0)
        for rho in np.r_[.065,np.r_[np.linspace(.23,.65,rings-2),.83,1.]]:
            if which<0 and rho==1:
                rows.append(seam)
            else:
                half=BAG/2*rho
                if which>0:
                    patch=np.array([.0013,.011])
                    half=patch+(BAG/2-patch)*(rho-.065)/.935
                rows.append([point(v*half,which,rho) for v in uv])
        for j in range(count):
            k=(j+1)%count
            tri=[(center,rows[0][j],rows[0][k])]
            for a,b in zip(rows,rows[1:]):
                if mirror and (j//(n//2))%2:
                    tri.extend(((a[j],b[j],a[k]),(b[j],b[k],a[k])))
                else:
                    tri.extend(((a[j],b[j],b[k]),(a[j],b[k],a[k])))
            faces.extend(tri if which>0 else [t[::-1] for t in tri])
        return rows[-1]
    seam=layer(1);layer(-1)
    uv=np.asarray(base); tags=np.array(side);rho=np.array(ring)
    # Rest metric of each of the two flat rectangular pieces, including the seal.
    ref=np.column_stack([uv, np.zeros(len(uv))])
    faces=np.asarray(faces)
    edges=np.unique(np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1),axis=0)
    return ref,faces,edges,tags,rho,np.asarray(seam)

def seed(ref,side,rho,apex,plane_z,span=(.225,.342)):
    # Straight rulings run from a finite pinch patch to the hem. Small
    # alternating pleats store circumferential surplus; gravity/PBD can move the hem.
    pos=np.empty_like(ref)
    xy=ref[:,:2]*np.asarray(span)/BAG
    phase=np.arctan2(ref[:,1]/BAG[1],ref[:,0]/BAG[0])
    pleat=np.cos(12*phase+.35)*rho
    xy*=1+.024*pleat[:,None]
    pos[:,:2]=apex[:2]+xy
    # The seal lies near the garment top; the lower piece turns up outside
    # the clothing edges. Both panel interiors initially enclose all eight corners.
    hem=plane_z+.0032
    lower=side<0
    inner=rho[lower]
    bottom_span=CLOTH[:2]+.008
    pos[lower,:2]=apex[:2]+ref[lower,:2]*bottom_span/(.83*BAG)
    pos[lower,2]=plane_z-.0215
    pos[:,2]=np.where(side>0,apex[2]+rho*(hem-apex[2]),plane_z-.0215)
    pos[:,2]+=np.where(side>0,.0015,.00025)*pleat
    pins=(side>0)&(rho<=.066)
    # Pin a strip across the actual pad faces, with finite contact length in Y.
    pos[pins,0]=apex[0]+ref[pins,0]
    pos[pins,1]=apex[1]+ref[pins,1]
    pos[pins,2]=apex[2]
    free_top=(side>0)&~pins
    pos[free_top,2]+= .065/.935*(1-rho[free_top])*(apex[2]-hem)
    # Ruled radial pleats store the excess across the four tent faces.
    # Displacement grows linearly along a ruling, avoiding a concave funnel.
    waves=.5+.5*np.cos(8*phase+.65*np.sin(3*phase))
    pos[free_top,2]+=.0105*waves[free_top]*np.maximum(rho[free_top]-.065,0)/.935
    return pos,pins

def lengths(v,e):return np.linalg.norm(v[e[:,1]]-v[e[:,0]],axis=-1)

def block_top(center,rotation,half):
    return center+half[:,None]*rotation[:,2]

def blocks(proxy,frame):
    centers=proxy['centers'][frame];R=proxy['frame'][frame]
    halves=np.tile(CLOTH/np.array([4,4,2]),(4,1));halves[:,2]=proxy['half_height'][frame]
    return centers,R,halves
