"""GPU version of LooseSim's hard-box containment and penetration diagnostics."""
import warp as wp


@wp.func
def edge_distance(p: wp.vec3, a: wp.vec3, b: wp.vec3):
    delta=b-a
    t=wp.clamp(wp.dot(p-a,delta)/wp.max(wp.dot(delta,delta),1e-20),0.,1.)
    return wp.length(p-a-t*delta)


@wp.func
def triangle_distance(p: wp.vec3,a: wp.vec3,b: wp.vec3,c: wp.vec3):
    v0=b-a;v1=c-a;v2=p-a
    aa=wp.dot(v0,v0);ab=wp.dot(v0,v1);bb=wp.dot(v1,v1)
    av=wp.dot(v2,v0);bv=wp.dot(v2,v1)
    det=aa*bb-ab*ab
    result=wp.min(edge_distance(p,a,b),wp.min(edge_distance(p,b,c),edge_distance(p,c,a)))
    if wp.abs(det)>1e-20:
        s=(bb*av-ab*bv)/det;t=(aa*bv-ab*av)/det
        if s>=0. and t>=0. and s+t<=1.:
            result=wp.abs(wp.dot(v2,wp.normalize(wp.cross(v0,v1))))
    return result


@wp.kernel
def penetration_vertices(vertices:wp.array2d(dtype=wp.vec3),xpos:wp.array2d(dtype=wp.vec3),
                         xmat:wp.array2d(dtype=wp.mat33),obj:int,half:wp.vec3,
                         state:wp.array2d(dtype=float),outcome:wp.array(dtype=int),
                         peak:wp.array(dtype=float)):
    w,i=wp.tid()
    if int(state[w,11])%12==0 and outcome[w]==0:
        p=wp.transpose(xmat[w,obj])*(vertices[w,i]-xpos[w,obj])
        inside=wp.min(half[0]-wp.abs(p[0]),wp.min(half[1]-wp.abs(p[1]),half[2]-wp.abs(p[2])))
        wp.atomic_max(peak,w,wp.max(inside,0.))


@wp.kernel
def penetration_corners(vertices:wp.array2d(dtype=wp.vec3),faces:wp.array(dtype=wp.vec3i),
                        xpos:wp.array2d(dtype=wp.vec3),xmat:wp.array2d(dtype=wp.mat33),obj:int,half:wp.vec3,
                        state:wp.array2d(dtype=float),outcome:wp.array(dtype=int),
                        peak:wp.array(dtype=float),escaped:wp.array(dtype=int)):
    w,i=wp.tid()
    if int(state[w,11])%12==0 and outcome[w]==0:
        local=wp.vec3(0.,0.,0.)
        if i<8:
            local=wp.vec3((2.*float(i//4)-1.)*half[0],(2.*float((i//2)%2)-1.)*half[1],
                          (2.*float(i%2)-1.)*half[2])
        p=xpos[w,obj]+xmat[w,obj]*local
        angle=float(0.);distance=float(100.)
        for j in range(faces.shape[0]):
            f=faces[j]
            a=vertices[w,f[0]];b=vertices[w,f[1]];c=vertices[w,f[2]]
            aa=a-p;bb=b-p;cc=c-p
            na=wp.length(aa);nb=wp.length(bb);nc=wp.length(cc)
            numerator=wp.dot(aa,wp.cross(bb,cc))
            denominator=na*nb*nc+wp.dot(aa,bb)*nc+wp.dot(bb,cc)*na+wp.dot(cc,aa)*nb
            angle=angle+2.*wp.atan2(numerator,denominator)
            if i<8: distance=wp.min(distance,triangle_distance(p,a,b,c))
        if wp.abs(angle)<2.*wp.pi:
            if i==8: escaped[w]=1
            else: wp.atomic_max(peak,w,distance)
