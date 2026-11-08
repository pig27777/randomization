"""Portable factories for the four classified deformable bags.

Use one live owner per process. Advance owner.step() and always call close().
All four defaults are empty bags; legacy payloads are an explicit option.
"""
from pathlib import Path
import importlib.util
import sys
import weakref

ROOT = Path(__file__).resolve().parent
FOLDERS = {'pe':'01_pe_film', 'kraft':'02_kraft_paper',
           'paperboard':'03_paperboard_envelope', 'legacy':'04_generic_experimental'}
_ACTIVE = None

def _module(name, path):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]

def load_asset(name='pe', **kwargs):
    global _ACTIVE
    if name not in FOLDERS:
        raise ValueError('Choose pe, kraft, paperboard, or legacy')
    previous = _ACTIVE() if _ACTIVE else None
    if previous is not None and not (getattr(previous,'closed',False) or getattr(previous,'_closed',False)):
        raise RuntimeError('Close the previous material owner before loading another asset')
    folder = ROOT/'assets'/FOLDERS[name]
    if name=='legacy':
        module = _module('classified_legacy_loader', folder/'load_asset.py')
        scene = module.load_bag(**kwargs)
    else:
        module = _module('paper_shell_runtime', ROOT/'assets/runtime.py')
        scene = module.load_paper(folder, **kwargs)
    _ACTIVE = weakref.ref(scene)
    return scene
