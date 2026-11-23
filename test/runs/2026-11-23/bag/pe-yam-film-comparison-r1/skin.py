"""Tessellate both surfaces identically and skin only the proxy recording."""
from pathlib import Path
import argparse
import numpy as np

def subdivision(n, faces):
    edges, inverse=np.unique(np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1),axis=0,return_inverse=True)
    a,b,c=(inverse.reshape(3,-1)+n)
    i,j,k=faces.T
    triangles=np.concatenate([np.stack(t,axis=1) for t in ((i,a,c),(a,j,b),(c,b,k),(a,b,c))])
    return edges,triangles

def refine(v,edges):
    return np.concatenate([v,(v[...,edges[:,0],:]+v[...,edges[:,1],:])*.5],axis=-2)

def skin_proxy(rest,r,names,lquats):
    import mujoco as mj
    centers=r['centers']; R=r['frame']
    delta=rest[:,None,:]-centers[0]
    distances=(delta**2).sum(2)
    sigma=.45*np.linalg.norm(np.ptp(centers[0],axis=0))
    w=np.exp(-.5*(distances-distances.min(1,keepdims=True))/sigma**2)
    w/=w.sum(1,keepdims=True)
    local=delta@R[0]
    anchors=r['anchors'][0];initial_ref=r['reference']
    neck_weight=np.exp(-np.sum((rest[:,None,:2]-anchors[:,:2])**2,axis=2)/(2*.026**2))
    neck_weight*=np.clip((rest[:,2:3]-rest[:,2].min())/np.ptp(rest[:,2]),0,1)**2
    neck_weight/=neck_weight.max(axis=0)
    pad_indices=[list(names).index(name) for name in ('left_pad','right_pad')]
    mesh=[];capture_blend=np.zeros(2)
    for i in range(len(centers)):
        # Each block has its measured center, the four-point material frame and its measured half-thickness.
        deformed=local.copy()
        deformed[:,:,2]*=r['half_height'][i]/r['half_height'][0]
        pos=np.einsum('vk,vkj->vj',w,deformed@R[i].T+centers[i])
        capture_blend+=np.clip(r['capture'][i].astype(float)-capture_blend,-.34,.34)
        # The physical contact plane begins at the distal edge and extends 40 mm up each pad.
        pair=r['pads'][i].copy()
        for j,b in enumerate(pad_indices):
            mat=np.empty(9)
            mj.mju_quat2Mat(mat,lquats[i,b])
            pair[j]+=.008*mat.reshape(3,3)[:,2]
        lateral=pair[1]-pair[0];lateral/=np.linalg.norm(lateral)
        neck=neck_weight*capture_blend
        if neck.any():
            difference=pos[:,None,:]-r['anchors'][i]
            along=difference@lateral
            tangential=difference-along[:,:,None]*lateral
            cones=pair[None]+.25*tangential+.08*along[:,:,None]*lateral
            mix=np.minimum(1.,neck.max(1)*1.02)
            target=np.einsum('vf,vfj->vj',neck,cones)/np.maximum(neck.sum(1)[:,None],1e-12)
            pos+=mix[:,None]*(target-pos)
            relative=rest[:,:2]-anchors[:,:2].mean(0)
            theta=np.arctan2(relative[:,1],relative[:,0])
            radius=np.linalg.norm(relative,axis=1)
            crease=.0015*np.sin(9*theta+.6*np.sin(3*theta)+55*radius)*4*mix*(1-mix)
            pos+=crease[:,None]*R[i,:,2]
        mesh.append(pos)
    mesh=np.array(mesh)
    assert np.max(abs(mesh[0]-rest))<1e-7 and np.isfinite(mesh).all()
    return mesh,w,initial_ref

def export(shell_file,proxy_file,out):
    l=np.load(shell_file);r=np.load(proxy_file)
    V=l['V'];F=l['F']
    assert np.array_equal(l['time'],r['time']) or np.allclose(l['time'],r['time'],rtol=0,atol=1e-12)
    for _ in range(2):
        edges,F=subdivision(V.shape[1],F);V=refine(V,edges)
    rest=V[0]
    mesh,w,initial_ref=skin_proxy(rest,r,l['body_names'],l['link_quat'])
    centers=r['centers'];R=r['frame']
    np.savez_compressed(out,time=l['time'],F=F.astype(np.int32),V_flex=V.astype('f4'),V_proxy=mesh.astype('f4'),
        V_0=rest.astype('f4'),weights=w.astype('f4'),links=l['links'],link_quat=l['link_quat'],
        link_names=l['body_names'],proxy_centers=centers,proxy_frame=R,proxy_anchors=r['anchors'],
        proxy_half_height=r['half_height'],proxy_capture=r['capture'],proxy_contact_reference=initial_ref,pads=r['pads'],
        flex_com=l['com'],proxy_com=r['com'],flex_v=l['velocity'],proxy_v=r['velocity'],
        flex_clearance=np.minimum(l['clearance'],l['diagnostics'][:,-1]),proxy_clearance=r['clearance'])
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('shell',type=Path);a.add_argument('proxy',type=Path);a.add_argument('output',type=Path)
    v=a.parse_args();export(v.shell,v.proxy,v.output)
