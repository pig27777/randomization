"""Export measured material response curves from saved runs."""
import argparse
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(fixtures,material,output,slip=None):
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for proto in (2,3,4,5):
        x=np.load(fixtures/f'p{proto}-compression.npz')
        # Compression coordinates depend on topology, copied from the fixed template.
        axes[0,0].plot(x['time'],-x['qpos'][:,:, [7,11,15]].mean(axis=(1,2))*1000,label=f'prototype {proto}')
    axes[0,0].set(xlabel='time (s)',ylabel='compression (mm)');axes[0,0].legend()
    for factor in (.5,1.,2.):
        x=np.load(material/f'sag-{factor}.npz')
        axes[0,1].plot(x['time'],x['sag'].mean(1)*1000,label=f'stiffness x{factor}')
    axes[0,1].set(xlabel='time (s)',ylabel='cantilever droop (mm)');axes[0,1].legend()
    for proto in (0,2,4):
        x=np.load(fixtures/f'p{proto}-bag.npz')
        axes[1,0].plot(x['time'],x['state']['tension'].sum(2).mean(1),label=f'prototype {proto}')
    axes[1,0].set(xlabel='time (s)',ylabel='ideal bag support tension (N)');axes[1,0].legend()
    if slip is not None:
        for friction in (.05,.8):
            x=np.load(slip/f'slip-{friction}.npz')
            axes[1,1].plot(x['time'],x['state']['slip'].mean(1)*1000,label=f'friction {friction}')
    axes[1,1].set(xlabel='time after engagement (s)',ylabel='accumulated film slip (mm)');axes[1,1].legend()
    for ax in axes.flat:ax.grid(alpha=.2)
    fig.savefig(output,dpi=160)
    fig.savefig(output.with_suffix('.svg'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fixtures',type=Path,required=True)
    p.add_argument('--material',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--slip',type=Path)
    a=p.parse_args();plot(a.fixtures,a.material,a.output,a.slip)
