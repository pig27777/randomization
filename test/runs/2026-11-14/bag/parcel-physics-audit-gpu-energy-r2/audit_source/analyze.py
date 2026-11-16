"""Figures and machine-readable digest of completed frozen-model experiments."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(root,output):
    output.mkdir(parents=True,exist_ok=False)
    def load(name):return json.loads((root/name/'summary.json').read_text())
    payload=load('parcel-physics-audit-payload-r3')
    kernels=load('parcel-physics-audit-kernels-r2')
    batch=load('parcel-physics-audit-batch-r1')
    consistency=load('parcel-physics-audit-consistency-r3')
    suspension=load('parcel-physics-audit-suspension-r3')
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    for g,color in [(0.,'C0'),(9.81,'C1')]:
        rows=[x for x in payload['bending'] if x['gravity']==g]
        axes[0,0].scatter([x['target']*1000 for x in rows],[x['measured']*1000 for x in rows],
                          color=color,label='tip load only' if g==0 else 'tip load + self weight')
    axes[0,0].plot([0,65],[0,65],color='gray',ls='--')
    axes[0,0].set(xlabel='target sag (mm)',ylabel='simulated tip sag (mm)');axes[0,0].legend()
    for proto in (2,4):
        direct=np.load(root/f'parcel-physics-audit-payload-r3/compression-p{proto}-mass0.1.npz')['trace']
        platen=np.load(root/f'parcel-physics-audit-payload-r3/platen-p{proto}-ramp1.0.npz')['trace']
        mask=(direct[:,0]<1.)
        axes[0,1].plot(direct[mask,1]*1000,direct[mask,2],label=f'P{proto} direct cap')
        mask=(platen[:,0]<2.4)
        axes[0,1].plot(platen[mask,1]*1000,platen[mask,2],ls='--',label=f'P{proto} actual platen')
    axes[0,1].set(xlabel='compression (mm)',ylabel='applied force (N)');axes[0,1].legend()
    for dt in (.00025,.0005,.001):
        data=np.load(root/f'parcel-physics-audit-payload-r3/energy-dt{dt}-c0.0.npz')['trace']
        initial=next(x['initial_energy'] for x in payload['energy'] if x['dt']==dt and x['damping']==0)
        axes[0,2].plot(data[:,0],100*data[:,-1]/initial,label=f'dt={dt*1000:g} ms')
    axes[0,2].set(xlabel='time (s)',ylabel='spring-only energy error (%)');axes[0,2].legend()
    ss=np.linspace(0,.008,200)
    residual=38-1200*ss-30*np.clip((.008-ss)/.005,0,1)-2000*ss
    axes[1,0].plot(ss*1000,residual)
    axes[1,0].axhline(0,color='gray',lw=1)
    axes[1,0].scatter([2.5,3.5714286],[0,0],label='interior roots')
    axes[1,0].scatter([8],[12.4],marker='x',s=60,label='GPU detachment')
    axes[1,0].set(xlabel='slip (mm)',ylabel='implicit residual (N)');axes[1,0].legend()
    a=np.load(root/'parcel-physics-audit-batch-r1/coverage-segmented.npz')
    p=a['params'];ratio=(10/p['compression'])/(.2*p['size'][:,2])
    axes[1,1].hist(ratio,bins=24)
    axes[1,1].axvline(1,color='C3',ls='--',label='core-contact limit')
    axes[1,1].set(xlabel='20 N target / core collision depth',ylabel='randomized parcels');axes[1,1].legend()
    ring=[x for x in payload['compression'] if x.get('prototype')==2 and 'cap_mass_fraction' in x]
    axes[1,2].plot([x['cap_mass_fraction']*100 for x in ring],[x['measured_frequency_hz'] for x in ring],marker='o')
    axes[1,2].set(xlabel='cap mass fraction (%)',ylabel='measured small-signal frequency (Hz)')
    for ax in axes.flat:ax.grid(alpha=.2)
    fig.savefig(output/'physics-audit.png',dpi=160)
    fig.savefig(output/'physics-audit.svg')
    np.savetxt(output/'bending.csv',np.array([[x[k] for k in ('length','mass','target','gravity','measured','reference','relative_target_error')] for x in payload['bending']]),
               delimiter=',',header='length,mass,target,gravity,measured,reference,relative_target_error',comments='')
    # Check provenance rather than silently attributing old results to a new model.
    frozen=root/'parcel-physics-audit-kernels-r2/source'
    live=Path('contents/parcel')
    fingerprint={p.name:dict(frozen=hashlib.sha256(p.read_bytes()).hexdigest(),
        current=hashlib.sha256((live/p.name).read_bytes()).hexdigest() if (live/p.name).exists() else None)
        for p in frozen.iterdir() if p.is_file() and p.name!='yam_adapter.py'}
    def functions(path):
        tree=ast.parse(path.read_text())
        result={}
        for node in tree.body:
            if isinstance(node,ast.ClassDef):
                for f in node.body:
                    if isinstance(f,ast.FunctionDef):result[node.name+'.'+f.name]=ast.dump(f,include_attributes=False)
        return result
    old=functions(frozen/'yam_adapter.py');new=functions(Path('test/content_model/parcel/yam_adapter.py'))
    contract={key:old.get(key)==new.get(key) for key in ['ParcelSim.apply','MaterialSpring.force','Attachment.try_bind']}
    reaction=[]
    for paths in ('1','0-2','0-1-2'):
        x=np.load(root/f'parcel-physics-audit-suspension-r3/paths-{paths}.npz')
        servo=1500*(.12-x['xpos'][-50:,:,3,2])-60*x['qvel'][-50:,:,8]
        reaction.append(dict(paths=paths,mean_palm_load_after_gripper_weight=float(servo.mean()-.24*9.81)))
    digest=dict(kernels=kernels,consistency=consistency,batch=batch,payload=payload,suspension=suspension,
                frozen_vs_current=fingerprint,yam_force_ast_unchanged=contract,native_palm_reaction=reaction)
    (output/'audit_summary.json').write_text(json.dumps(digest,indent=2,ensure_ascii=False))
    print(output/'physics-audit.png')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('test/runs/2026-11-14/bag'))
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();main(args.root,args.output)
