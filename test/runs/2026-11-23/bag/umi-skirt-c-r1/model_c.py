"""C specimen. Native skirt gores with rotational compliance, PE envelope and measured YAM FK."""
from pathlib import Path
import sys, numpy as np
from xml.etree import ElementTree as ET
import mujoco as mj
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'umi-pe-yam-hold-r1'))
from geometry import envelope, lengths, BAG, robot_state
from project import color_edges
from record import make_scene,pad_faces,vec
MASS=.50
CLOTH=np.array([.245,.345,.020])
CENTER=np.array([-.03,-.225])
def pad_path():
    q=np.load(HERE/'q_ref.npz')['q_path'];r=make_scene();m=r.sim.m;d=r.kin
    p=[]
    for angles in q:
        d.qpos[np.r_[r.qadr,r.fingers]]=angles;mj.mj_kinematics(m,d);p.append(pad_faces(r,d))
    return np.array(p)

def template():
    ref,F,E,side,rho,seam=envelope(n=6,rings=4,mirror=True)
    # Bend the outer 22 mm of each flat sheet over the garment hem. Each
    # generator turns through a constant angle without in-plane extension.
    # Extra rings sample this turnover on both panels; their seals coincide.
    uv=ref[:,:2].copy()
    for i in np.flatnonzero((rho>.82)&(rho<.99)):
        nominal=(np.array([.0013,.011])+(BAG/2-np.array([.0013,.011]))*(rho[i]-.065)/.935
                 if side[i]>0 else BAG/2*rho[i])
        uv[i]=uv[i]/nominal*(BAG/2-.022)
    ref[:,:2]=uv+CENTER
    fold=np.maximum(abs(uv)-(BAG/2-.022),0)
    sn=.010175/.022;cs=np.sqrt(1-sn*sn)
    p=ref.copy();p[:,:2]-=np.sign(uv)*(1-cs)*fold
    p[:,2]=np.where(side>0,.02070-sn*fold.max(1),.00035+sn*fold.max(1))
    return ref,F,E,side,rho,seam,p

def scene(dt,skirt_rigidity=(1.25,.53)):
    ref,F,E,side,rho,seam,initial=template();root=ET.Element('mujoco')
    ET.SubElement(root,'compiler',angle='radian',autolimits='true')
    ET.SubElement(root,'option',timestep=str(dt),integrator='discrete',iterations='100',jacobian='sparse',solver='Newton',ccd_tolerance='1e-10',gravity='0 0 -9.81',tolerance='1e-8')
    default=ET.SubElement(root,'default')
    ET.SubElement(default,'geom',friction='.60 .005 .0001',solref=f'{max(.0025,6*dt)} 1',solimp='.999 .9999 .0002')
    asset=ET.SubElement(root,'asset');world=ET.SubElement(root,'worldbody');eq=ET.SubElement(root,'equality')
    ET.SubElement(world,'geom',name='table',type='plane',size='1 1 .01',contype='1',conaffinity='10')
    steel=ET.SubElement(world,'body',name='jaw',mocap='true',pos=vec([*CENTER,.020]))
    hub=ET.SubElement(world,'body',name='skirt_center',pos=vec([*CENTER,.020]))
    ET.SubElement(hub,'freejoint',name='skirt_root')
    film_mass=2*np.prod(BAG)*.0552
    cloth_mass=MASS-film_mass
    ET.SubElement(hub,'inertial',pos='0 0 -.006',mass=str(.03*cloth_mass),diaginertia='0.000001 0.000001 0.000001')
    ET.SubElement(hub,'geom',name='cloth_tab',type='box',size='.0045 .011 .003',pos='0 0 -.003',contype='8',conaffinity='3',mass='0')
    # Four independently loaded quadrants bend away from a finite central patch.
    # Use the skirt's flexural rigidity calibrated to the measured 40 mm
    # incremental drop across the 20 mm-thick resting specimen.
    slots=[(0,-1),(0,1),(1,-1),(1,1)]
    half=CLOTH[:2]/2
    for j,(axis,sign) in enumerate(slots):
        perp=1-axis
        dist=half[axis];wide=half[perp]
        area_mass=.97*cloth_mass/4
        along=np.zeros(3);along[axis]=sign
        ortho=np.cross([0,0,1.],along)
        base=along*.004
        # Truncated triangular cloth ply, tapered near the clamped centre.
        xy=np.array([base+.003*ortho,base-.003*ortho,dist*along-wide*.982*ortho,dist*along+wide*.982*ortho])
        rim=xy[2:].copy()
        shoulder=rim*.78
        outline=np.concatenate([xy[:2],shoulder,rim])
        top=outline.copy();bottom=outline.copy()
        # Full 20 mm loft through the body of the skirt, with folded hems that
        # fit the measured mailer as its two sheets turn into the edge seal.
        top[4:,2]-=.0093
        bottom[:,2]-=[.003,.003,.0195,.0195,.0160,.0160]
        pts=np.concatenate((top,bottom))
        ET.SubElement(asset,'mesh',name=f'panel{j}',vertex=vec(pts))
        b=ET.SubElement(hub,'body',name=f'skirt{j}')
        angle=np.arcsin((.036 if axis==0 else .040)/dist)
        lever=.67*dist
        k=np.broadcast_to(skirt_rigidity,(2,))[axis]*area_mass*9.81*lever*np.cos(angle)/angle
        ET.SubElement(b,'joint',name=f'fold{j}',type='hinge',axis=vec(ortho),pos=vec(base),range='-.2 .65',
                      stiffness=str(k),damping=str(.65*2*(k*area_mass*lever**2)**.5))
        ET.SubElement(b,'geom',name=f'garment{j}',type='mesh',mesh=f'panel{j}',mass=str(area_mass),contype='8',conaffinity='3',margin='.00035')
    # A held cloth centre carries the weight. Film alone cannot activate this fixture.
    for j,(x,y) in enumerate(((-.0012,-.008),(.0012,-.008),(0.,.008))):
        ET.SubElement(hub,'site',name=f'clothgrip{j}',pos=vec([x,y,0]),size='.0001')
        ET.SubElement(steel,'site',name=f'grip{j}',pos=vec([x,y,0]),size='.0001')
        ET.SubElement(eq,'connect',name=f'grip{j}',site1=f'clothgrip{j}',site2=f'grip{j}',active='false',solref=f'{2*dt} 1',solimp='.9999 .99995 .0001')
    # Reference coordinates are the flat PE material metric; actual loaded coordinates
    # are independent. The edge tendons enforce one-sided near-inextensibility.
    coords=initial.copy()
    flex=ET.SubElement(world,'flexcomp',name='pe',type='direct',dim='2',point=vec(coords),element=vec(F),mass=str(film_mass),radius='.0003')
    ET.SubElement(flex,'edge',equality='false')
    ET.SubElement(flex,'contact',contype='2',conaffinity='11',solref=f'{max(.0025,6*dt)} 1',solimp='.999 .9999 .0003',selfcollide='auto')
    ET.SubElement(flex,'elasticity',young='20000000',poisson='.3',thickness='.00006',elastic2d='both',damping='.00003')
    proto=mj.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.xml') as tmp:
        mj.mj_saveLastXML(tmp.name,proto);tree=ET.parse(tmp.name).getroot()
    eq=tree.find('equality');ten=ET.SubElement(tree,'tendon')
    pins=(side>0)&(rho<=.066)
    for i in range(len(ref)):
        b=tree.find(f".//body[@name='pe_{i}']");ET.SubElement(b,'site',name=f'v{i}',size='.0001')
        if pins[i]:
            # A held-mode clamp couples both the garment and the PE strip to the jaw.
            xy=ref[i,:2]-CENTER
            target=tree.find(".//body[@name='jaw']")
            ET.SubElement(target,'site',name=f'p{i}',pos=vec([*xy,0]),size='.0001')
            ET.SubElement(eq,'connect',name=f'pegrip{i}',site1=f'v{i}',site2=f'p{i}',active='false',solref=f'{2*dt} 1',solimp='.9999 .99995 .0001')
    for (a,b),ll in zip(E,lengths(ref,E)):
        t=ET.SubElement(ten,'spatial',limited='true',range=f'0 {ll:.10g}',solreflimit=f'{2*dt} 1',solimplimit='.999 .9999 .0002')
        ET.SubElement(t,'site',site=f'v{a}');ET.SubElement(t,'site',site=f'v{b}')
    m=mj.MjModel.from_xml_string(ET.tostring(tree,encoding='unicode'));d=mj.MjData(m)
    ids=m.flex_vertbodyid
    qa=m.jnt_qposadr[m.body_jntadr[ids]][:,None]+np.arange(3)
    va=m.jnt_dofadr[m.body_jntadr[ids]][:,None]+np.arange(3)
    d.qpos[qa]=initial-coords;mj.mj_forward(m,d)
    return m,d,ref,F,E,pins,seam,qa,va

def film_pressure(vertices,faces,volume0=None,bulk=600.,limit=220.):
    # Pressure responds to the signed volume of the sealed mailer. Applying
    # area forces to every triangle conserves total force on the closed film.
    tri=vertices[faces]-vertices.mean(0)
    area=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])*.5
    volume=np.einsum('ij,ij->',tri[:,0],area)/3
    if volume0 is None:return volume
    pressure=np.clip(bulk*(1-volume/volume0),-limit,limit)
    forces=np.zeros_like(vertices)
    for k in range(3):np.add.at(forces,faces[:,k],area*pressure/3)
    return forces

def remove_outward_velocity(d,va,dp):
    # Remove only the part of nodal velocity that would undo a tensile
    # correction. A geometric projection must not kick a resting parcel.
    u=dp/np.maximum(np.linalg.norm(dp,axis=1,keepdims=True),1e-15)
    v=d.qvel[va]
    d.qvel[va]=v-u*np.minimum(np.sum(v*u,axis=1,keepdims=True),0.)
