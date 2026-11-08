"""Recoverable nonlinear SLS hinge: fixed reference + a transient Maxwell branch.

E = U_bilinear(theta) + .5 * K_memory * wrap(theta-z)^2
z_dot = wrap(theta-z)/tau; at fixed theta the branch dissipates energy.
z is a viscous internal angle, never a permanent manufacturing rest angle.
"""
import numpy as np

def wrap(x):
    return np.arctan2(np.sin(x),np.cos(x))

def moment(delta, memory, knee, post, memory_ratio):
    a=np.abs(delta)
    equilibrium=np.sign(delta)*np.where(a<=knee,a,knee+post*(a-knee))
    return equilibrium+memory_ratio*wrap(delta-memory)

def potential(delta,memory,knee,post,memory_ratio):
    a=np.abs(delta);u=np.minimum(a,knee);v=np.maximum(a-knee,0.)
    return .5*u*u+knee*v+.5*post*v*v+.5*memory_ratio*wrap(delta-memory)**2

def advance(memory,delta,dt,tau):
    # Exact integration with theta held over this accepted time step.
    return wrap(memory+(-np.expm1(-dt/tau))*wrap(delta-memory))

def plastic_return(delta,plastic,accumulated,knee,post):
    """1D associative return map with isotropic hardening, unit elastic K.

    H=post/(1-post) matches the reversible model's first-loading bilinear
    curve. Plastic flow dissipates K*knee*dgamma and cannot heal at rest.
    """
    if post>=1:return plastic.copy(),accumulated.copy(),np.zeros_like(delta)
    hardening=post/(1-post)
    trial=wrap(delta-plastic)
    increment=np.maximum(abs(trial)-knee-hardening*accumulated,0)/(1+hardening)
    return wrap(plastic+increment*np.sign(trial)),accumulated+increment,increment

def crease_history(peak,delta,onset,width,maximum):
    """Irreversible, bounded local bending softening with a fixed rest angle.

    A phenomenological history law, distinct from permanent plastic rotation.
    It is frozen in force evaluations and advanced after accepted steps.
    """
    peak=np.maximum(peak,abs(delta))
    damage=maximum*(-np.expm1(-np.maximum(peak-onset,0.)/width))
    return peak,damage
