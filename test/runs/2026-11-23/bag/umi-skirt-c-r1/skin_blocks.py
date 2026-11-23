"""Shallow pressed-garment surface, four-block LBS and finite-face PBD."""
from model_c import *
from project import project, audit

def build(episode,out):
    r=np.load(episode);i=200
    if not r['capture'][i].all() or abs(r['support'][i])>1e-7:raise ValueError('no independent stable carry')
    ref,F,E,tags,rho,seam,initial=template()
    centers=r['centers'][i];R=r['frame'][i]
    half=np.tile(CLOTH/[4,4,2],(4,1));half[:,2]=r['half_height'][i]
    plane=(centers+half[:,2,None]*R[:,2]).mean(0)
    pads=r['pads'][i];apex=pads.mean(0);a=(apex-plane)@R
    # Centre the fitted garment around its real blocks; the centre material
    # of the skirt and film enters the same jaw strip in this grasp mode.
    v=initial.copy();v[:,:2]-=CENTER
    v[:,:2]*=[.983,.990]
    u=np.abs((ref[:,:2]-CENTER)/(BAG/2))
    frac=np.max(u,axis=1)
    top=tags>0;pins=top&(rho<=.066)
    crest=frac
    # The skirt's shorter meridian meets the thick block at a shallower
    # drop; the long-axis hem can turn downward beyond the block end.
    along_x=u[:,0]/np.maximum(frac,1e-10)
    end_height=-.0055+.0076*along_x**4
    v[top,2]=a[2]*(1-crest[top])+end_height[top]*crest[top]
    angle=np.arctan2(ref[:,1]-CENTER[1],ref[:,0]-CENTER[0])
    v[top,2]+=.0023*np.maximum(0,1-frac[top])*frac[top]*np.cos(8*angle[top])
    # Perimeter has 2.2 cm of film to turn down around the packed skirt.
    v[~top,2]=initial[~top,2]-.0211
    v[seam,2]=end_height[seam]
    v[:,:2]+=a[:2]*(1-frac[:,None])
    ix=ref[pins,:2]-CENTER;v[pins]=a+np.column_stack((ix,np.zeros(len(ix))))
    v=v@R.T+plane
    b0=r['centers'][0];local=b0-b0.mean(0);dx=(ref[:,:2]-CENTER)[:,None]-local[None,:,:2]
    sigma=.45*np.linalg.norm(np.ptp(b0,axis=0));w=np.exp(-.5*(dx**2).sum(2)/sigma**2);w/=w.sum(1)[:,None]
    delta=centers-centers.mean(0)-local@R.T
    v[~pins]+=(w@delta)[~pins]
    v=project(v,ref,F,E,pins,centers,R,half,finite_contact=True,initial=initial)
    cross,outside=audit(v,F,centers,R,half)
    names,links,quats,check=robot_state();assert np.allclose(check,pads,atol=1e-8)
    np.savez_compressed(out,V=v,ref=ref,E=E,F=F,pins=pins,seam=seam,pads=pads,links=links,quats=quats,names=names,
       initial=initial,top=top,weights=w,centers=centers,frame=R,half=half,cross=cross,outside=outside)
    print('H L W maxL/L0, crossings:',(apex[2]-v[:,2].min())*100, np.ptp(v,0)[:2][::-1]*100,
          np.max(lengths(v,E)/lengths(ref,E)),cross,outside)
if __name__=='__main__':
    import sys;build(*sys.argv[1:])
