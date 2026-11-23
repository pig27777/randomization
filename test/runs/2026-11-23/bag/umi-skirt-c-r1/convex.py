"""Audit convex garment parts against complete triangles of the PE surface."""
import numpy as np
from itertools import combinations
from bag_sim.wrap import winding_number

def planes_from_mesh(v,f):
    tri=v[f];n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);n/=np.linalg.norm(n,axis=1)[:,None]
    n*=np.sign(np.einsum('ij,ij->i',tri[:,0]-v.mean(0),n))[:,None]
    a=np.column_stack([n,np.einsum('ij,ij->i',n,tri[:,0])])
    _,ix=np.unique(a.round(7),axis=0,return_index=True)
    return a[ix]

def vertices(planes):
    out=[]
    for tri in combinations(range(len(planes)),3):
        a=planes[list(tri)]
        if abs(np.linalg.det(a[:,:3]))<1e-8:continue
        x=np.linalg.solve(a[:,:3],a[:,3])
        if np.max(planes[:,:3]@x-planes[:,3])<1e-8:out.append(x)
    return np.unique(np.round(out,10),axis=0)

def clipped(dist):
    poly=list(np.eye(3))
    for d in dist.T:
        if not poly:return []
        out=[];a=poly[-1];da=a@d
        for b in poly:
            db=b@d
            if (da>0)!=(db>0):out.append(a+(b-a)*(da/(da-db)))
            if db<=0:out.append(b)
            a,da=b,db
        poly=out
    return poly

def candidates(v,F,p,margin):
    dist=v@p[:,:3].T-p[:,3]-margin
    return dist,np.flatnonzero((dist[F].min(1)<=0).all(1))

def audit(v,F,solids):
    collisions=outside=0
    for p in solids:
        dist,ii=candidates(v,F,p,-1e-7)
        collisions+=sum(bool(clipped(dist[F[j]])) for j in ii)
        outside+=int(np.sum(abs(winding_number(v,F,vertices(p)))<.5))
    return collisions,outside
