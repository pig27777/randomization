"""Read-only kraft assets, uniform size changes, and topology-preserving coarsening."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
INPUT_ROOT=Path(os.environ.get('RANDOM_L5_INPUT_ROOT',ROOT))
BAG=INPUT_ROOT/'bag_sim'


def freeze_inputs(out):
    """Snapshot only the read-only project inputs actually used by this experiment."""
    import shutil
    out.mkdir(parents=True,exist_ok=False)
    files=['bag_sim/bag_model.py','bag_sim/wrap.py','bag_sim/legacy/contents.py',
           'bag_sim/legacy/initialization.py','bag_sim/reference/runtime.py',
           'bag_sim/profiles/kraft.json','bag_sim/geometry/kraft.npz',
           'isaaclab/results/descriptions/compressible-describe-1.json']
    manifest={}
    for name in files:
        source=ROOT/name;target=out/name
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        manifest[name]=hashlib.sha256(target.read_bytes()).hexdigest()
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def original_mesh():
    with np.load(BAG/'geometry/kraft.npz',allow_pickle=False) as g:
        return {k:g[k].copy() for k in g.files}


def area(vertices,faces):
    a,b,c=vertices[faces].transpose(1,0,2)
    return np.linalg.norm(np.cross(b-a,c-a),axis=1)/2


def topology(faces):
    edges,counts=np.unique(np.sort(np.vstack((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]])),axis=1),axis=0,return_counts=True)
    return dict(vertices=len(np.unique(faces)),faces=len(faces),edges=len(edges),
                boundary_edges=int(sum(counts==1)),nonmanifold_edges=int(sum(counts>2)),
                euler=len(np.unique(faces))-len(edges)+len(faces))


def coarsen(mesh,target=150):
    """Collapse short edges only within the same panel/boundary class, preserving link topology."""
    v=mesh['vertices'].copy();f=mesh['faces'].copy();pid=mesh['panel_ids'].copy()
    uv=mesh['uv'].copy()
    while len(np.unique(f))>target:
        edges,counts=np.unique(np.sort(np.vstack((f[:,[0,1]],f[:,[1,2]],f[:,[2,0]])),axis=1),axis=0,return_counts=True)
        boundary=set(edges[counts==1].ravel())
        panels={int(i):set(pid[np.any(f==i,axis=1)]) for i in np.unique(f)}
        neighbors={int(i):set(edges[np.any(edges==i,axis=1)].ravel())-{i} for i in np.unique(f)}
        accepted=False
        for index in np.argsort(np.linalg.norm(v[edges[:,0]]-v[edges[:,1]],axis=1),kind='stable'):
            a,b=map(int,edges[index])
            if panels[a]!=panels[b] or (a in boundary)!=(b in boundary):
                continue
            shared=np.any(f==a,axis=1)&np.any(f==b,axis=1)
            opposite=set(f[shared].ravel())-{a,b}
            if neighbors[a]&neighbors[b]!=opposite:
                continue
            position=v[a] if a in boundary else (v[a]+v[b])/2
            affected=np.any((f==a)|(f==b),axis=1)&~shared
            old=f[affected];changed=old.copy();changed[changed==b]=a
            old_norm=np.cross(v[old[:,1]]-v[old[:,0]],v[old[:,2]]-v[old[:,0]])
            trial=v.copy();trial[a]=position
            new_norm=np.cross(trial[changed[:,1]]-trial[changed[:,0]],trial[changed[:,2]]-trial[changed[:,0]])
            if np.any(np.sum(old_norm*new_norm,axis=1)<=1e-18):
                continue
            v[a]=position;uv[a]=(uv[a]+uv[b])/2
            f=f[~shared];pid=pid[~shared];f[f==b]=a
            accepted=True
            break
        if not accepted:
            raise RuntimeError(f'no safe panel-preserving collapse remains at {len(np.unique(f))} vertices')
    used=np.unique(f);remap=np.full(len(v),-1);remap[used]=np.arange(len(used));f=remap[f]
    masses=np.zeros(len(used));np.add.at(masses,f.ravel(),np.repeat(area(v[used],f)/3,3))
    masses*=mesh['masses_kg'].sum()/masses.sum()
    result=dict(vertices=v[used],faces=f,panel_ids=pid,uv=uv[used],masses_kg=masses)
    assert topology(f)['euler']==topology(mesh['faces'])['euler']
    assert not topology(f)['nonmanifold_edges']
    return result


def body_area(mesh):
    front=np.unique(mesh['faces'][mesh['panel_ids']==0])
    lo,hi=mesh['vertices'][front,:2].min(0),mesh['vertices'][front,:2].max(0)
    return float(np.prod(hi-lo)),lo,hi


def boundary_edges(faces):
    directed=np.vstack((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]))
    _,inverse,counts=np.unique(np.sort(directed,axis=1),axis=0,return_inverse=True,return_counts=True)
    return directed[counts[inverse]==1]


def seam_pairs(mesh):
    v,f,pid=mesh['vertices'],mesh['faces'],mesh['panel_ids']
    mouth=np.intersect1d(np.unique(boundary_edges(f)),np.unique(f[pid==0]))
    back=np.unique(f[np.isin(pid,[3,7])])
    return np.array([(int(a),int(back[np.argmin(np.linalg.norm(v[back]-v[a],axis=1))])) for a in mouth],np.int32)


def closed_surface(vertices,faces):
    """Metric-only mouth cap; the physical mesh is sealed separately by seam springs."""
    edge=boundary_edges(faces)
    if not len(edge):return vertices,faces
    center=vertices[np.unique(edge)].mean(0)
    return np.vstack((vertices,center)),np.vstack((faces,np.c_[edge[:,1],edge[:,0],np.full(len(edge),len(vertices))]))


def close_mouth(rest,q,mesh,item,radius,iterations=800):
    """Close the adhesive seam during loading, without preloading a widely open mouth."""
    from initialization import edge_colors
    from contents import signed_distance
    from wrap import open_pairs,panel_pairs
    q=q.copy();faces=mesh['faces'];seams=seam_pairs(mesh);gap=2*radius+.0002
    edges=np.unique(np.sort(np.vstack((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]])),axis=1),axis=0)
    groups=edge_colors(edges)
    targets=[np.linalg.norm(rest[g[:,1]]-rest[g[:,0]],axis=1) for g in groups]
    pairs=panel_pairs(rest,faces,mesh['panel_ids'],[0],[3,7]);floor=q[:,2].min()
    for _ in range(iterations):
        for g,target in zip(groups,targets):
            a,b=g.T;delta=q[b]-q[a];length=np.linalg.norm(delta,axis=1)
            correction=.45*((length-target)/np.maximum(length,1e-12))[:,None]*delta
            q[a]+=correction;q[b]-=correction
        distance,normal=signed_distance(q,item)
        q+=np.maximum(radius+.0003-distance,0)[:,None]*normal
        open_pairs(q,pairs,gap);q[:,2]=np.maximum(q[:,2],floor)
        a,b=seams.T;middle=(q[a]+q[b])/2;middle[:,2]=np.maximum(middle[:,2],floor+gap/2)
        q[a]=middle+[0,0,gap/2];q[b]=middle-[0,0,gap/2]
    lengths=np.linalg.norm(rest[edges[:,1]]-rest[edges[:,0]],axis=1)
    return q,dict(maximum_edge_strain=float(np.max(np.abs(np.linalg.norm(q[edges[:,1]]-q[edges[:,0]],axis=1)/lengths-1))),
        maximum_seam_error_m=float(np.max(np.abs(np.linalg.norm(q[seams[:,1]]-q[seams[:,0]],axis=1)-gap))),
        minimum_wall_gap_m=float(np.min(q[pairs[:,0],2]-q[pairs[:,1],2])),seam_gap_m=gap,
        method='generic wrap + sealed-mouth edge/content/wall projections')


def wrapped(mesh,size,scale,iterations=1600):
    # These project modules are called read-only and do not import a simulator.
    for p in (BAG/'legacy',BAG):
        if str(p) not in sys.path:sys.path.insert(0,str(p))
    from contents import normalize_contents
    from wrap import wrap_generic
    rest=mesh['vertices']*scale
    # Kraft's y length becomes task x length, retaining its aspect ratio and flap.
    rest=np.c_[-rest[:,1],rest[:,0],rest[:,2]]
    # Nominal 10 mm lateral clearance is loading preparation, not a physical constraint.
    padded=np.asarray(size)+[.020,.020,0.]
    item=normalize_contents([dict(shape='box',size_m=padded.tolist(),mass_kg=1.)])[0]
    q,report=wrap_generic(rest,rest.copy(),mesh['faces'],mesh['panel_ids'],[0],[3,7],item,.00035,iterations=iterations)
    q,report=close_mouth(rest,q,mesh,item,.00035)
    report['loading_envelope_size_m']=padded.tolist()
    q[:,:2]-=item['position'][:2];rest[:,:2]-=item['position'][:2]
    return rest,q,np.array([0.,0.,item['position'][2]]),report


def inspect_scales(size):
    mesh=original_mesh();front_area,lo,hi=body_area(mesh)
    loose=np.sqrt(3*np.prod(size[:2])/front_area)
    # wrap_generic reserves 8 mm per edge for the seam in addition to the packing clearance.
    minimum=max((size[0]+.0361)/(hi[1]-lo[1]),(size[1]+.0361)/(hi[0]-lo[0]))
    scales=np.linspace(minimum,loose,6)
    return [dict(scale=float(s),body_area_ratio=float(front_area*s*s/np.prod(size[:2])),
                 **wrapped(mesh,size,float(s))[3]) for s in scales]


def prepare_assets(out):
    out.mkdir(parents=True,exist_ok=False)
    mesh=original_mesh();coarse=coarsen(mesh)
    with (out/'kraft_150.npz').open('xb') as f:np.savez(f,**coarse)
    scale_source=ROOT/'isaaclab/results/loose_package/assets_formal/manifest.json'
    scale_cases=json.loads(scale_source.read_text())['cases']
    manifest=dict(area_definition='front body unfolded area / content xy projection, no flap',
        fill_scale_source=str(scale_source),fill_scale_source_sha256=hashlib.sha256(scale_source.read_bytes()).hexdigest(),
        edge_strain_is_diagnostic=True,preparation_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256((BAG/'geometry/kraft.npz').read_bytes()).hexdigest(),
        profile_sha256=hashlib.sha256((BAG/'profiles/kraft.json').read_bytes()).hexdigest(),
        fine_topology=topology(mesh['faces']),coarse_topology=topology(coarse['faces']),cases=[])
    for category,size in [('hard',np.array([.08,.045,.02])),('compressible',np.array([.35,.25,.12]))]:
        scans=inspect_scales(size)
        # Keep the already documented size grid; user approved reporting strain instead of gating on 5%.
        for name in ('tight','medium','loose'):
            original=next(c for c in scale_cases if c['category']==category and c['fill']==name)
            if original['size_m']!=size.tolist():raise ValueError('frozen fill grid has a different content size')
            scale=original['scale']
            case=dict(category=category,size_m=size.tolist(),fill=name,scale=scale,
                body_area_ratio=body_area(mesh)[0]*scale*scale/np.prod(size[:2]),initialization={})
            for label,g in [('fine',mesh),('coarse',coarse)]:
                rest,q,center,report=wrapped(g,size,scale)
                with (out/f'{category}_{name}_{label}.npz').open('xb') as f:
                    np.savez(f,rest=rest,initial=q,center=center,**g)
                case['initialization'][label]=report
                case['initialization'][label]['edge_strain_above_5_percent']=report['maximum_edge_strain']>.05
            case['scale_scan']=scans
            manifest['cases'].append(case)
    with (out/'manifest.json').open('x') as f:json.dump(manifest,f,indent=2,allow_nan=False)
    return manifest


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--prepare',action='store_true')
    a=ap.parse_args()
    if a.prepare:
        print(json.dumps(prepare_assets(a.output),indent=2));raise SystemExit(0)
    mesh=original_mesh();coarse=coarsen(mesh)
    result=dict(source_sha256=hashlib.sha256((BAG/'geometry/kraft.npz').read_bytes()).hexdigest(),
                original=topology(mesh['faces']),coarse=topology(coarse['faces']),
                area_definition='kraft front-panel unfolded body area / content xy projected area, flap excluded',
                hard=inspect_scales(np.array([.08,.045,.02])),
                compressible=inspect_scales(np.array([.35,.25,.12])))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    with (a.output.parent/'kraft_coarse_150.npz').open('xb') as f:np.savez(f,**coarse)
    print(json.dumps(result,indent=2))
