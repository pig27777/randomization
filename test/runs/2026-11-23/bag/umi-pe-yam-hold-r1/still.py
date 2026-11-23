"""Render a single solver snapshot in the shared YAM studio, including its detail view."""
from geometry import *
from render import render

def still(source,out):
    a=np.load(source)
    render(out,a['V'][None],a['F'],a['links'][None],a['quats'][None],a['names'],a['pads'][None],True,
           material_xy=a['ref'][:,:2],hold_detail=True)

if __name__=='__main__':
    import sys
    still(sys.argv[1],sys.argv[2])
