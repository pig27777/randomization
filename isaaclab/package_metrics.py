"""Containment and material-contact sliding measurements for L5, matching bag-line conventions."""
from __future__ import annotations

import sys
import numpy as np

from package_assets import BAG, closed_surface

sys.path.insert(0,str(BAG))
from wrap import winding_number, point_triangle_distance


def rotation(q):
    x,y,z,w=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def geometry(scene,bodies,cloth,penetration=False):
    if not scene.cloth:return dict(centres_outside=0.,corners_outside=0.,penetration_mm=0.)
    v,f=closed_surface(cloth,scene.faces)
    pts=[];true_corners=[];depth=0.
    signs=np.array([[a,b,c] for a in (-1,1) for b in (-1,1) for c in (-1,1)])
    for b,half in zip(scene.content_bodies,scene.content_half):
        R=rotation(bodies[b,3:]);p=bodies[b,:3]
        inner=np.maximum(half-.003,.25*half)
        pts.extend(np.vstack((p,p+(signs*inner)@R.T)))
        if penetration:
            true_corners.extend(p+(signs*half)@R.T)
            local=(cloth-p)@R
            depth=max(depth,float(np.max(np.min(half-np.abs(local),axis=1))))
    pts=np.asarray(pts);inside=np.abs(winding_number(v,f,pts))>=.5
    if penetration:
        corners=np.asarray(true_corners)
        outside=np.abs(winding_number(v,f,corners))<.5
        if np.any(outside):depth=max(depth,float(np.max(point_triangle_distance(corners[outside],v,f))))
    flags=np.zeros(len(pts),bool);flags[::9]=True
    return dict(centres_outside=float(1-inside[flags].mean()),
                corners_outside=float(1-inside[~flags].mean()),penetration_mm=max(depth,0.)*1000)


class ContactSliding:
    def __init__(self):
        self.references={};self.max_mm=0.;self.evidence=None;self.measured=0

    def update(self,scene,bodies,cloth,t):
        if not scene.cloth:return
        contacts=scene.solver.get_proxy_contacts('rigid','cloth')
        count=int(contacts.soft_contact_count.numpy()[0])
        ids=contacts.soft_contact_indices.numpy()[:count]
        bary=contacts.soft_contact_barycentric.numpy()[:count]
        shapes=contacts.soft_contact_shape.numpy()[:count]
        shape_body=scene.model.shape_body.numpy()
        half_by_body=dict(zip(scene.content_bodies,scene.content_half))
        for feature,w,shape in zip(ids,bary,shapes):
            b=int(shape_body[shape])
            if b not in half_by_body:continue
            valid=feature>=0;feature=feature[valid];w=w[valid]
            key=(b,tuple(map(int,feature)))
            R=rotation(bodies[b,3:])
            if key not in self.references:
                local=(w@cloth[feature]-bodies[b,:3])@R
                axis=int(np.argmin(np.abs(np.abs(local)-half_by_body[b])))
                self.references[key]=(w.copy(),local,axis,t)
                continue
            original,initial,axis,first_t=self.references[key]
            local=(original@cloth[feature]-bodies[b,:3])@R
            half=half_by_body[b]
            if abs(abs(local[axis])-half[axis])>2*scene.params['collision_radius_m']:continue
            tangent=local-initial;tangent[axis]=0
            movement=float(np.linalg.norm(tangent))*1000
            self.measured+=1
            if movement>self.max_mm:
                self.max_mm=movement
                self.evidence=dict(body=b,feature=list(key[1]),weights=original.tolist(),first_t=first_t,t=t,
                    initial_body_local=initial.tolist(),current_body_local=local.tolist(),normal_axis=axis,
                    tangential_motion_mm=movement)
