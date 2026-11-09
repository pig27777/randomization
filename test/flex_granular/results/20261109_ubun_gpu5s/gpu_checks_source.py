"""Small per-step diagnostics included in the granular benchmark timer."""
import warp as wp


@wp.kernel
def state_check(qpos:wp.array2d(dtype=float),qvel:wp.array2d(dtype=float),qacc:wp.array2d(dtype=float),
                flags:wp.array(dtype=int)):
    w,i=wp.tid()
    bad=int(0)
    if i<qpos.shape[1]:
        if not wp.isfinite(qpos[w,i]): bad=1
    if i<qvel.shape[1]:
        if not wp.isfinite(qvel[w,i]) or not wp.isfinite(qacc[w,i]): bad=1
    if bad!=0: wp.atomic_or(flags,w,bad)


@wp.kernel
def counters(overflow:wp.array(dtype=int),nefc:wp.array(dtype=int),iterations:wp.array(dtype=int),
             nacon:wp.array(dtype=int),flags:wp.array(dtype=int),peak:wp.array(dtype=int)):
    w=wp.tid()
    if overflow[w]!=0: wp.atomic_or(flags,w,2)
    wp.atomic_max(peak,0,nacon[0])
    wp.atomic_max(peak,1,nefc[w])
    wp.atomic_max(peak,2,iterations[w])
    wp.atomic_or(peak,3,overflow[w])


@wp.kernel
def contact_check(dist:wp.array(dtype=float),nacon:wp.array(dtype=int),peak:wp.array(dtype=float)):
    i=wp.tid()
    if i<nacon[0]: wp.atomic_max(peak,0,wp.max(-dist[i],0.))
