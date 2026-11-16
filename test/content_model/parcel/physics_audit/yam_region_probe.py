"""Reproduce the regional-label attack with the frozen model and shared YAM.

The model and adapter are loaded from the audit snapshot. The shared platform
and robot assets come from the checkout; their hashes are recorded by its runner.
"""
import argparse
import importlib.util
from pathlib import Path
import sys
from common import SNAPSHOT,ROOT
from experience.yam_platform.angles import execute

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--region',type=int,choices=(0,1),required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--smoke-seconds',type=float,default=0.)
    args=parser.parse_args()
    name='parcel_audit_frozen_yam'
    spec=importlib.util.spec_from_file_location(name,SNAPSHOT/'yam_adapter.py')
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    execute(name,dict(prototype=0,region=args.region,seed=7,dt=.0005,smoke_seconds=args.smoke_seconds),args.output)
