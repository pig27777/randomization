"""Run a model adapter through the common YAM grasp protocol."""
import argparse
from dataclasses import asdict
import importlib
import json
from pathlib import Path
import platform
import traceback
from grasp_platform import run


def execute(backend_name, options, output):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    backend=importlib.import_module(backend_name)
    cfg=backend.Config(**options)
    try:
        return run(backend,cfg,output)
    except Exception as error:
        output.mkdir(parents=True,exist_ok=True)
        result=dict(config=asdict(cfg),backend=backend_name,hostname=platform.node(),
                    outcome='setup_fail' if 'initial pose unreachable' in str(error) else 'execution_error',
                    reason=str(error),error=type(error).__name__,traceback=traceback.format_exc(),timing_is_throughput=False)
        (output/'result.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result),flush=True)
        return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',default='native_flex')
    parser.add_argument('--model-options',default='{}',help='Model-specific JSON options')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--category',choices=('bendable','compressible'),default='bendable')
    for name,default in (('yaw',0.),('tilt',0.),('dt',.001),('pinch-coeff',1.),('contact-time',.005),('wall-limit',1800.),('smoke-seconds',0.),('arm-height',.27)):
        parser.add_argument(f'--{name}',type=float,default=default)
    args=vars(parser.parse_args());backend=args.pop('backend');output=args.pop('output');extra=json.loads(args.pop('model_options'))
    if set(args)&set(extra):parser.error('model-options cannot replace shared protocol options')
    result=execute(backend,{**args,**extra},output)
    return int(result['outcome'] in ('setup_fail','execution_error'))


if __name__=='__main__':raise SystemExit(main())
