"""Load the shared package runtime; opening XML alone omits crease history."""
from pathlib import Path
import importlib.util

def create_scene():
    root = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("paper_shell_runtime", root.parent/"runtime.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_paper(root)
