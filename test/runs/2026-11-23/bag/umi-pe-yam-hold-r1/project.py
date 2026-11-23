"""30 position-based sweeps with free seams, fixed pad strip and convex garment obstacles."""
import numpy as np
from geometry import *
from bag_sim.physical_parcel import triangle_box_intersections,winding_number
SIGNS=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])

def color_edges(edges):
    taken=[];groups=[]
    for a,b in edges:
        for used,g in zip(taken,groups):
            if a not in used and b not in used:break
        else:
            used=set();g=[];taken.append(used);groups.append(g)
        used.update((a,b));g.append((a,b))
    return [np.array(g) for g in groups]

def audit(v,F,centers,R,half):
    crossing=outside=0
    for c,h in zip(centers,half):
        crossing+=triangle_box_intersections(v,F,h,c,R).sum()
        outside+=np.sum(abs(winding_number(v,F,(SIGNS*h)@R.T+c))<.5)
    return int(crossing),int(outside)

def footprint_barycentrics(vertices,extent,axis):
    # Clip the triangle against the finite contact face, retaining the
    # barycentric coordinates needed to distribute contact at the actual points.
    poly=list(np.eye(3))
    for k in (0,1,2):
        if k==axis:continue
        for sign in (-1,1):
            if not poly:return []
            clipped=[]
            a=poly[-1];da=extent[k]-sign*(a@vertices[:,k])
            for b in poly:
                db=extent[k]-sign*(b@vertices[:,k])
                if (da<0)!=(db<0):clipped.append(a+(b-a)*(da/(da-db)))
                if db>=0:clipped.append(b)
                a,da=b,db
            poly=clipped
    return poly

def project(v,ref,F,E,pins,centers,R,half,its=30,finite_contact=False,initial=None):
    v=v.copy();groups=color_edges(E)
    constraints=[(g[:,0],g[:,1],np.minimum(lengths(ref,g),lengths(initial,g)) if initial is not None else lengths(ref,g)) for g in groups]
    anchor=v[pins].copy();free=(~pins).astype(float)
    for it in range(its):
        for a,b,l0 in constraints:
            delta=v[b]-v[a];dist=np.linalg.norm(delta,axis=1)
            # A membrane carries tension and can release compression into folds.
            excess=np.maximum(dist-1.025*l0,0)
            corr=delta*(excess/np.maximum(dist,1e-12)/np.maximum(free[a]+free[b],1))[:,None]
            v[a]+=free[a,None]*corr;v[b]-=free[b,None]*corr
        for c,h in zip(centers,half):
            # Separate whole triangles from each solid, including those whose vertices
            # are outside. A positive clearance avoids coplanar ambiguity in the audit.
            safe=h+.00065
            local=(v-c)@R
            ii=np.flatnonzero(triangle_box_intersections(local,F,safe,np.zeros(3)))
            for fid in ii:
                ids=F[fid]; vs=local[ids]
                depth=np.stack([safe-vs.min(0),safe+vs.max(0)])
                sign,ax=np.unravel_index(np.argmin(depth),depth.shape)
                direction=1-2*sign
                if finite_contact:
                    for weights in footprint_barycentrics(vs,safe,ax):
                        penetration=safe[ax]+1e-7-direction*(weights@local[ids,ax])
                        denominator=(weights**2)@free[ids]
                        if penetration>0 and denominator>1e-10:
                            local[ids,ax]+=direction*penetration*weights*free[ids]/denominator
                else:
                    shift=np.maximum(safe[ax]+1e-7-direction*local[ids,ax],0)*free[ids]
                    local[ids,ax]+=direction*shift
            v=local@R.T+c
        v[pins]=anchor
    return v

def preview(proxy,out):
    r=np.load(proxy);i=200
    if not r['capture'][i].all() or r['support'][i]>1e-5:raise ValueError('garment has no physical hold')
    names,links,quats,pads=robot_state();apex=pads.mean(0)
    ref,F,E,tags,rho,seam=envelope()
    centers,R,half=blocks(r,i)
    plane=(centers+half[:,2,None]*R[:,2]).mean(0)
    # Work in the actual four-block material frame so the whole garment tilt is retained.
    local_apex=(apex-plane)@R
    v,pins=seed(ref,tags,rho,local_apex,0,span=(.225,.34))
    # The long-axis ends hang below the top of the textile where the seal
    # clears its ends. Keep the crease generator straight right up to the hem.
    along=np.clip(abs(ref[:,1])/(.39/2)/np.maximum(rho,1e-8),0,1)
    ends=along**8*np.maximum(rho-.065,0)/.935
    v[tags>0,2]-=.0195*ends[tags>0]
    # At the seam the drape surrounds the garment, even when it swings under the fingers.
    blend=np.maximum(rho-.065,0)/.935
    v[:,:2]-=local_apex[:2]*blend[:,None]
    v=v@R.T+plane
    # Blend the measured residual displacement of each block after its common
    # frame rotation. This preserves differential bending and vertical squash.
    initial=r['centers'][0]
    offset=initial-initial.mean(0)
    dx=ref[:,None,:2]-offset[None,:,:2]
    sigma=.45*np.linalg.norm(np.ptp(initial,axis=0))
    weights=np.exp(-.5*(dx*dx).sum(2)/sigma**2)
    weights/=weights.sum(1)[:,None]
    residual=centers-centers.mean(0)-offset@R.T
    squash=r['half_height'][i]-r['half_height'][0]
    corr=weights@residual+(weights@squash)[:,None]*tags[:,None]*R[:,2]
    v[~pins]+=corr[~pins]
    pos=project(v,ref,F,E,pins,centers,R,half,finite_contact=True)
    cross,outside=audit(pos,F,centers,R,half)
    out_dims=np.ptp(pos,axis=0)
    print('height',apex[2]-plane[2], 'span',out_dims,'strain',np.max(lengths(pos,E)/lengths(ref,E)),'intersect',cross,outside,flush=True)
    np.savez_compressed(out,V=pos,F=F,E=E,ref=ref,pins=pins,seam=seam,top_z=plane[2],pads=pads,links=links,quats=quats,names=names,
                centers=centers,R=R,half=half,weights=weights,crossing=cross,outside=outside)
if __name__=='__main__':
    import sys
    preview(sys.argv[1],sys.argv[2])
