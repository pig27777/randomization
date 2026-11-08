"""Round 3 B1-B4: native MuJoCo bag carrier and slack spatial tendons.

The outline body represents the bag's observed envelope, not a rigid physical film.
It collides with the table/fingertips; contents see the original content contacts.
Tendon friction represents bag/content sliding friction, and six opposed length
limits per segment bound the contents' motion in the carrier frame.
"""
from dataclasses import dataclass
import math

import mujoco
import numpy as np
from xml.etree import ElementTree as ET

from mj_loose import LooseBag, LooseBuilt, LooseSim, _v
from mj_adapter import scene, BUSHING_DT
from pinch import fragments, PinchAttachment, DEPTH_MIN


@dataclass
class EffectBag(LooseBag):
    kind: str = 'effect'
    mu_bag_content: float = .45
    bag_pinch_coeff: float = 1.
    slack: tuple = (.082, .082, .030)

    def resolved(self, content_size):
        result=super().resolved(content_size)
        if self.mass is None:
            result.mass *= result.size[0]*result.size[1]/(.35*.25)
        return result


def effect_scene(desc, bag, pads=(), dt=None, table_xmin=-.6, fingertips=()):
    bag=bag.resolved(desc.kappa.size)
    if any(not math.isfinite(v) or v < 0 for v in [bag.mu_bag_content,bag.bag_pinch_coeff,*bag.slack]):
        raise ValueError('finite nonnegative bag friction, pinch coefficient and slack required')
    dt=BUSHING_DT if dt is None else dt
    size=np.asarray(desc.kappa.size)
    origin=np.array([0.,0.,size[2]/2+.0001])
    half=np.array([*np.asarray(bag.size)/2,size[2]/2])
    # Native free carrier; it does not collide with the content geometry (bit 2).
    affinity=5 if fingertips or any(p[1] == 2 for p in pads) else 1
    body=[f'<body name="effect_bag" pos="{_v(origin)}"><freejoint name="effect_bag_free"/>',
          f'<geom name="bag_envelope" type="box" size="{_v(half)}" mass="{bag.mass}" '
          f'contype="8" conaffinity="{affinity}" friction="{bag.friction} .005 .0001" '
          'rgba=".72 .56 .36 .4"/>']
    ropes,content_sites=[],{}
    slack=np.minimum(np.asarray(bag.slack),np.r_[(np.asarray(bag.size)-size[:2])/2,bag.slack[2]])
    for i,segment in enumerate(desc.segments):
        name='obj' if desc.rigid else f's{i}'
        local=segment.pos if desc.rigid else np.zeros(3)
        content_sites.setdefault(name,[]).append(f'<site name="bc_{i}" pos="{_v(local)}" size=".001"/>')
        for axis in range(3):
            radius=float(max(bag.size[0],bag.size[1],size[2]))
            for sign in (-1,1):
                anchor=segment.pos.copy();anchor[axis]+=sign*radius
                label=f'rope_{i}_{axis}_{sign}'
                body.append(f'<site name="{label}" pos="{_v(anchor)}" size=".001"/>')
                ropes.append(f'<spatial name="{label}" limited="true" range="0 {radius+slack[axis]:.9g}" '
                             f'solreflimit=".005 1" frictionloss="{bag.mu_bag_content*segment.mass*9.81/6:.9g}">'
                             f'<site site="{label}"/><site site="bc_{i}"/></spatial>')
    body.append('</body>')
    frame,eq=fragments(desc)
    eq+='\n<connect name="bag_pinch" body1="pinch_frame" body2="effect_bag" anchor="0 0 0" '
    eq+='active="false" solref=".005 1" solimp=".99 .99 .001"/>'
    body.append(frame)
    if fingertips:
        from top_grasp import fingertip_xml
        body.extend(fingertip_xml(f[0],f[2]).replace('contype="0" conaffinity="0"','contype="4" conaffinity="0"')
                    for f in fingertips)
    built=scene(desc,origin,pads,dt=dt,table_xmin=table_xmin,worldbody='\n'.join(body),
                equality=eq,tendon='\n'.join(ropes))
    tree=ET.fromstring(built.xml)
    if any(p[1] == 2 for p in pads):
        for p,_,_ in pads:
            for geom in tree.findall(f'.//body[@name="{p}"]/geom'):
                geom.set('contype','4')
    for name,sites in content_sites.items():
        # Insert sites as direct children, keeping all content joints and coefficients unchanged.
        target=tree.find(f'.//body[@name="{name}"]')
        if target is None: raise RuntimeError(f'frozen content body not found: {name}')
        for text in sites: target.append(ET.fromstring(text))
    built.xml=ET.tostring(tree,encoding='unicode')
    if fingertips:
        from mj_adapter import pad_pair
        tree=ET.fromstring(built.xml)
        contact=tree.find('contact')
        for p,_,_ in fingertips:
            geoms=tree.findall(f'.//body[@name="{p}"]/geom')
            for geom in geoms:
                for i,s in enumerate(desc.segments):
                    text=pad_pair(geom.get('name'),f's{i}',desc.kappa.mass if desc.rigid else s.mass,dt,
                                  min(bag.friction,bag.mu_bag_content),desc.kappa.skin_compliance)
                    contact.append(ET.fromstring(text))
        built.xml=ET.tostring(tree,encoding='unicode')
    return LooseBuilt(built,bag,dict(bodies=['effect_bag'],half=half,slack=slack,integrator='implicitfast'))


class EffectSim(LooseSim):
    def __init__(self,built,pad_names=()):
        super().__init__(built,pad_names)
        self.m.pair_friction[:,:2]=min(self.bag.friction,self.bag.mu_bag_content)

    def bag_points(self):
        return self.bag_surface()[0]

    def bag_surface(self):
        half=self.meta['half']
        local=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])*half
        b=self.bag_bodies[0]
        v=self.d.xpos[b]+local@self.d.xmat[b].reshape(3,3).T
        faces=np.array([[0,4,6],[0,6,2],[1,3,7],[1,7,5],[0,1,5],[0,5,4],
                        [2,6,7],[2,7,3],[0,2,3],[0,3,1],[4,5,7],[4,7,6]])
        return v,faces

    def penetration(self):
        return 0.  # No membrane exists in the effect model; report native tether violations separately.

    def outside_fraction(self):
        b=self.bag_bodies[0]
        points=self.content_world_points()
        local=(points-self.d.xpos[b])@self.d.xmat[b].reshape(3,3)
        # A suspended content may hang below the carrier; the projected bag outline is bounded.
        outside=np.any(np.abs(local[:,:2])>np.asarray(self.bag.size)/2,axis=1)
        return float(outside[::9].mean()),float(outside[np.arange(len(points))%9!=0].mean())

    def footprint(self):
        v,_=self.bag_surface()
        lo,hi=v[:,:2].min(0),v[:,:2].max(0)
        return (lo+hi)/2,hi-lo,float(v[:,2].max())

    def rope_violation(self):
        ids=[self.m.tendon(i).id for i in range(self.m.ntendon) if self.m.tendon(i).name.startswith('rope_')]
        return float(np.maximum(self.d.ten_length[ids]-self.m.tendon_range[ids,1],0).max())


class BagPinch(PinchAttachment):
    def __init__(self,sim,coefficient):
        self.sim,self.coefficient=sim,float(coefficient)
        self.mocap=sim.m.body_mocapid[sim.m.body('pinch_frame').id]
        self.active=self.segment=None
        self.segments,self.events=[],[]
        self.broken=False
        self.load=self.peak_force=0.
        self.contact_height=None
        self.body=sim.bag_bodies[0]
        self.equations={0:sim.m.equality('bag_pinch').id}

    def top_contacts(self):
        found={p:set() for p in self.sim.pad_names}
        bag_geom=self.sim.m.geom('bag_envelope').id
        for c in self.sim.d.contact:
            if c.efc_address<0 or self.sim.d.efc_force[c.efc_address] <= .05: continue
            for p,geoms in self.sim.pad_geoms.items():
                if (c.geom1 in geoms and c.geom2==bag_geom) or (c.geom2 in geoms and c.geom1==bag_geom): found[p].add(0)
        return tuple(found[p] for p in self.sim.pad_names)

    def candidate(self,tips):
        if self.active is not None or self.broken or self.contact_height is None: return None
        if np.any(self.contact_height-tips[:,2]<DEPTH_MIN-1e-12): return None
        return [0] if all(self.top_contacts()) else None

    def bind_segments(self,segments):
        m,d=self.sim.m,self.sim.d
        eq=self.equations[0]
        m.eq_data[eq,:3]=0.
        m.eq_data[eq,3:6]=d.xmat[self.body].reshape(3,3).T@(d.mocap_pos[self.mocap]-d.xpos[self.body])
        d.eq_active[eq]=True
        self.active,self.segment,self.segments=[eq],0,[0]
        self.events.append(dict(t=float(d.time),event='bind',target='bag'))
