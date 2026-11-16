"""One lock covers the serial and 28-process, same-machine CPU benchmark."""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def main(default_backend="yam_platform.rigid_control"):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--backend',default=default_backend)
    args=parser.parse_args()
    # Verify the lock is already held before creating any throughput results.
    probe=subprocess.run(['flock','-n','-E','73','/tmp/claude-gpu.lock','true'])
    if probe.returncode!=73:raise RuntimeError('launch this benchmark under flock /tmp/claude-gpu.lock')
    args.output.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy()
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):env[key]='1'
    env['PYTHONPATH']=os.pathsep.join([*sys.path, env.get('PYTHONPATH', '')])
    records={}
    for workers,trials in ((1,1),(28,28)):
        output=args.output/f'cpu{workers}'
        command=[sys.executable,'-B',str(Path(__file__).with_name('run_suite.py')),'speed','--locked',
                 '--backend',args.backend,'--workers',str(workers),'--trials',str(trials),'--output',str(output)]
        with (args.output/f'cpu{workers}.log').open('w') as log:
            subprocess.run(command,check=True,env=env,stdout=log,stderr=subprocess.STDOUT)
        records[f'cpu{workers}']=json.loads((output/'summary.json').read_text())
    (args.output/'summary.json').write_text(json.dumps(dict(hostname=platform.node(),
        lock='/tmp/claude-gpu.lock',threads_per_process=1,backend=args.backend,measurements=records),indent=2))
    print(args.output/'summary.json',flush=True)


if __name__=='__main__':main()
