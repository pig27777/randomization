"""CPU PE edge projection. Native function only mutates its input vertex buffer."""
import ctypes,subprocess,tempfile
from pathlib import Path
import numpy as np
def compile_limit():
    source=Path(__file__).with_name('edge_limit.c')
    scratch=tempfile.TemporaryDirectory()
    lib=Path(scratch.name)/'edge_limit.so'
    subprocess.run(['cc','-O3','-shared','-fPIC',str(source),'-lm','-o',str(lib)],check=True)
    native=ctypes.CDLL(str(lib));fn=native.project_edges
    f64=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    i64=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')
    fn.argtypes=[f64,i64,f64,f64,ctypes.c_size_t];fn.restype=ctypes.c_double
    return scratch,fn
