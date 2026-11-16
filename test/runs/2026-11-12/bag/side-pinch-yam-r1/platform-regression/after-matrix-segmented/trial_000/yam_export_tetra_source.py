"""Export recorded native full-flex states to an offline tetrahedron playback."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np


def positions(m,d):
    mujoco.mj_kinematics(m,d)
    b=m.flex_vertbodyid
    return d.xpos[b]+np.einsum('nij,nj->ni',d.xmat[b].reshape(-1,3,3),m.flex_vert)


def export(run,output):
    if output.exists():raise FileExistsError(output)
    m=mujoco.MjModel.from_xml_path(str(run/'model.xml'));d=mujoco.MjData(m)
    if m.nflex!=1 or m.flex_dim[0]!=3:raise ValueError('one full dim=3 flex required')
    a=np.load(run/'trajectory.npz');result=json.loads((run/'result.json').read_text());trace=json.loads((run/'trace.json').read_text())
    rest=positions(m,d);tets=m.flex_elem.reshape(-1,4)
    def det(v):
        p=v[tets];return np.linalg.det(np.stack([p[:,j]-p[:,0] for j in (1,2,3)],axis=2))
    d.qpos[:]=a['final_qpos'];selected=int(np.argmin(det(positions(m,d))/det(rest)))
    frames=[]
    stride=max(1,round(.2/np.median(np.diff(a['time']))))
    indices=sorted(set(range(0,len(a['qpos']),stride))|set(range(max(0,len(a['qpos'])-5),len(a['qpos']))))
    for i in indices:
        d.qpos[:]=a['qpos'][i];v=positions(m,d);jaws=[]
        for side in ('left','right'):
            g=m.geom(side+'_pad').id;s=m.geom_size[g]
            local=np.array([[x,y,z] for x in (-s[0],s[0]) for y in (-s[1],s[1]) for z in (-s[2],s[2])])
            jaws.append((local@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]).round(6).tolist())
        time=float(a['time'][i]);frames.append(dict(t=time,v=v.round(6).tolist(),jaws=jaws,phase=min(trace,key=lambda x:abs(x['t']-time))['phase']))
    data=dict(source=str(run),selected_tet=selected,rest=rest.round(6).tolist(),tets=tets.tolist(),frames=frames,
              note=f'{result["config"].get("shape","box")} · {result["outcome"]} · 红色为翻转单元；保存状态之间仅作显示插值。')
    template=Path(__file__).with_name('tetra_view.html').read_text()
    fragment=template.replace('__MESH_DATA__',json.dumps(data,separators=(',',':'),ensure_ascii=False))
    if len(fragment.encode())>=1_000_000:raise ValueError('playback exceeds inline size budget')
    output.write_text(fragment)
    print(output,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();export(args.run,args.output)
