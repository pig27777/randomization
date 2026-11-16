"""Shared YAM 20 x 40 mm rounded fingertip geometry."""
import numpy as np

def _v(x):
    return " ".join(f"{float(a):.8g}" for a in x)


def fingertip_xml(name, normal, pos, width=0.020, height=0.040, thick=0.008, fillet=0.002):
    """Kinematic fingertip (test/policy/README.md v2: YAM, 20 mm wide x 40 mm high, 2 mm edge radius), contact face
    normal to world axis `normal` (0 or 1), height along z: a box core and four capsules on the face edges, all
    colliding with the bag only (bit 4). Same body / joint naming as mj_adapter.pad_xml, so Sim drives it."""
    u = 1 - normal                                    # in-plane horizontal axis
    half = np.zeros(3)
    half[u], half[normal], half[2] = width / 2 - fillet, thick / 2, height / 2 - fillet
    c = 'contype="4" conaffinity="0"'
    g = [f'<body name="{name}" pos="{_v(pos)}" gravcomp="1">', '<inertial pos="0 0 0" mass="100" diaginertia="1 1 1"/>',
         *[f'<joint name="{name}_{a}" type="slide" axis="{_v(np.eye(3)[i])}"/>' for i, a in enumerate("xyz")],
         f'<geom name="{name}_core" type="box" size="{_v(half)}" rgba=".2 .6 .9 .6" {c}/>']
    k = 0
    for side in (-1, 1):                              # both faces (the gripper is symmetric)
        off = np.zeros(3)
        off[normal] = side * (thick / 2 - fillet)
        for s in (-1, 1):
            a, b = off.copy(), off.copy()             # vertical edges
            a[u] = b[u] = s * half[u]
            a[2], b[2] = -half[2], half[2]
            c1, c2 = off.copy(), off.copy()           # horizontal edges (top / bottom)
            c1[2] = c2[2] = s * half[2]
            c1[u], c2[u] = -half[u], half[u]
            for p0, p1 in ((a, b), (c1, c2)):
                g.append(f'<geom name="{name}_e{k}" type="capsule" fromto="{_v(p0)} {_v(p1)}" size="{fillet}" '
                         f'rgba=".2 .6 .9 .6" {c}/>')
                k += 1
    g.append("</body>")
    return "\n".join(g)
