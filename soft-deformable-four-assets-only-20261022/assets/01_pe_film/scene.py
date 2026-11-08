"""Factory entrypoint: keep the Python material callback for deformable behavior."""
from pathlib import Path
import importlib.util
import sys


def create_scene():
    root = Path(__file__).resolve().parent
    # Share one callback dispatcher with other assets in this runtime package.
    name = 'paper_shell_runtime'
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, root.parent/'runtime.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module.load_paper(root)
