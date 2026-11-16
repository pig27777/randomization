"""Run the existing regression suite against the recorded audit implementation."""
import argparse
import importlib.util
import sys
import unittest
from common import ROOT,setup,write

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    out=setup(parser.parse_args().output,'existing behavioral regressions on frozen source')
    name='parcel_audit_original_tests'
    spec=importlib.util.spec_from_file_location(name,ROOT/'test/content_model/parcel/test_physics.py')
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    write(out/'summary.json',dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),success=result.wasSuccessful()))
    sys.exit(0 if result.wasSuccessful() else 1)
