"""Independent scalar references for auditing; no production imports."""
import numpy as np


def return_roots(extension, stiffness, captured, old, dynamic_capacity, zeta, dt):
    """All roots of the continuous piecewise-linear implicit residual."""
    extension=np.asarray(extension,dtype=float)
    k=np.broadcast_to(stiffness,extension.shape)
    def residual(s):
        tension=k*np.maximum(extension-s,0)
        phi=np.clip((captured-s)/.005,0,1)
        return float(tension.sum()-dynamic_capacity*phi-zeta*(s-old)/dt)
    knots=np.unique(np.clip(np.r_[old,captured,captured-.005,extension],old,captured))
    roots=[]
    for lo,hi in zip(knots[:-1],knots[1:]):
        fl,fh=residual(lo),residual(hi)
        if hi<=lo:continue
        slope=(fh-fl)/(hi-lo)
        if abs(fl)<1e-10:roots.append((float(lo),float(slope)))
        if fl*fh<0:roots.append((float(lo-fl/slope),float(slope)))
    if abs(residual(captured))<1e-10:roots.append((float(captured),None))
    return roots,residual


def cantilever_reference(a, k, end_force, masses, gravity, height):
    """Two planar hinges, first body clamped, tip point load and distributed weight.

    Hinge locations 0 and a. Tip 2a. Segment 1/2 COMs a/2 and a+a/2.
    The 10% caps contribute initial COM offset .05H above the core frame.
    """
    m1,m2=masses
    A=m1*a/2+m2*a
    B=m2*a/2
    z=.05*height
    q=np.zeros(2)
    for _ in range(60):
        u,v=q[0],q.sum()
        cross=end_force*a*np.sin(v)+gravity*B*np.sin(v)-gravity*z*m2*np.cos(v)
        grad=np.array([k*q[0]-end_force*a*(np.cos(u)+np.cos(v))-gravity*(A*np.cos(u)+B*np.cos(v))-gravity*z*(m1*np.sin(u)+m2*np.sin(v)),
                       k*q[1]-end_force*a*np.cos(v)-gravity*B*np.cos(v)-gravity*z*m2*np.sin(v)])
        h=np.array([[k+end_force*a*np.sin(u)+gravity*A*np.sin(u)-gravity*z*m1*np.cos(u)+cross,cross],
                    [cross,k+cross]])
        dq=np.linalg.solve(h,grad)
        q-=dq
        if np.max(np.abs(dq))<1e-12:break
    return q,float(a*(np.sin(q[0])+np.sin(q.sum())))
