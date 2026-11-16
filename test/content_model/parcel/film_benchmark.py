"""Full six-face membrane CUDA Graph benchmark, including custom material state."""
from pathlib import Path
import argparse,gc,json,subprocess,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import numpy as np
import warp as wp
from contents.parcel import Batch
from film_checks import manifest


def main(output,sizes,steps):
    output.mkdir(parents=True,exist_ok=False)
    manifest(output,sizes=sizes,steps=steps,warmup=400,dt=.0000625)
    rows=[]
    for rigid in (True,False):
        for n in sizes:
            b=Batch(n,rigid=rigid,envelope=True,dt=.0000625)
            graph=b.graph();b.reset()
            for _ in range(400):wp.capture_launch(graph)
            wp.synchronize()
            start=time.perf_counter()
            for _ in range(steps):wp.capture_launch(graph)
            wp.synchronize();elapsed=time.perf_counter()-start
            film=b.data.flexvert_xpos.numpy();edges=b.surface.topo.edges
            ratio=np.linalg.norm(film[:,edges[:,1]]-film[:,edges[:,0]],axis=2)/b.surface.lengths.numpy()
            row=dict(rigid=rigid,n=n,steps=steps,elapsed=elapsed,physics_steps_s=steps/elapsed,
                     env_steps_s=n*steps/elapsed,dt=b.dt,cuda_graph=True,
                     warp_pool_used_bytes=wp.get_mempool_used_mem_current(),
                     warp_pool_peak_bytes=wp.get_mempool_used_mem_high(),
                     device_memory_used_mib=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip(),
                     finite=bool(np.isfinite(b.data.qpos.numpy()).all()),overflow=int(b.data.overflow.numpy().max()),
                     max_edge_ratio=float(ratio.max()),
                     broken_edges=int(b.surface.damage.numpy().sum()))
            rows.append(row);print(json.dumps(row),flush=True)
            (output/'summary.json').write_text(json.dumps(rows,indent=2))
            assert row['finite'] and row['overflow']==0 and row['broken_edges']==0 and row['max_edge_ratio']<2, row
            del b,graph;gc.collect()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--sizes',default='1,64,512,2048,4096');p.add_argument('--steps',type=int,default=200)
    a=p.parse_args();main(a.output,[int(x) for x in a.sizes.split(',')],a.steps)
