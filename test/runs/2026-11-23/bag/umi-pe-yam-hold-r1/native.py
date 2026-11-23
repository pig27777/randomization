"""Native MuJoCo flex equilibrium of a loaded, pad-held sealed mailer."""
import argparse
from xml.etree import ElementTree as ET
import numpy as np
import mujoco as mj
from geometry import BAG,CLOTH,MASS,RUN,robot_state,envelope,seed,lengths
from record import vec
from bag_sim.physical_parcel import Config,payload_xml

def settle(out,seconds,dt,patch_half,bend_factor,initial_gap):
    ref,F,E,side,rho,seam=envelope(n=8,rings=5,mirror=True)
    names,links,quats,pads=robot_state()
    apex=pads.mean(0)
    plane=apex[2]-initial_gap
    held,pins=seed(ref,side,rho,apex,plane)
    # The native fixture holds a folded strip across the 22 mm pad length.
    # Its two nearly vertical flaps conserve length between the 3 mm-open pads.
    scaled=np.divide(ref[:,0],np.where(rho==0,1,.0013+(.145-.0013)*(rho-.065)/.935))
    top=side>0
    ref[top,0]=scaled[top]*(patch_half+(.145-patch_half)*(rho[top]-.065)/.935)
    ref[rho==0,0]=0
    local=ref[pins,0]
    lateral=pads[1]-pads[0];lateral/=np.linalg.norm(lateral)
    tangent=np.cross([0.,0.,1.],lateral);tangent/=np.linalg.norm(tangent)
    into_pad=np.cross(lateral,tangent)
    # Fold both flaps up the real tilted pad planes in the locked FK pose.
    held[pins]=apex+local[:,None]*(.0013/patch_half)*lateral+ref[pins,1,None]*tangent
    held[pins]+=(patch_half-abs(local))[:,None]*np.sqrt(1-(.0013/patch_half)**2)*into_pad
    from project import project,audit
    object_center=np.r_[apex[:2],plane-.010][None]
    held=project(held,ref,F,E,pins,object_center,np.eye(3),CLOTH[None]/2)
    print('seed enclosure',audit(held,F,object_center,np.eye(3),CLOTH[None]/2),flush=True)
    ref[:,:2]+=apex[:2]
    # Separate flat sheets by imperceptible epsilon to select their orientation.
    ref[:,2]=apex[2]+side*1e-6
    film=2*np.prod(BAG)*.0552
    cfg=Config(mode='bendable',payload_size=(CLOTH[1],CLOTH[0],CLOTH[2]),payload_mass=MASS-film)
    asset,payload=payload_xml(cfg,np.r_[apex[:2],plane-.010])
    root=ET.fromstring('<mujoco><compiler angle="radian" autolimits="true"/><option/><asset>'+asset+'</asset><worldbody>'+payload+'</worldbody><equality/></mujoco>')
    opt=root.find('option'); opt.attrib.update(timestep=str(dt),gravity='0 0 -9.81',ccd_tolerance='1e-10',iterations='120',solver='Newton',integrator='Euler',jacobian='sparse',tolerance='1e-9')
    world=root.find('worldbody'); eq=root.find('equality')
    cloth=world.find("body[@name='payload_0']")
    # payload_xml places the leftmost of the three segments at body_0.
    offset=np.fromstring(cloth.get('pos'),sep=' ')-np.r_[apex[:2],plane-.010]
    cloth.set('pos',vec(np.r_[apex[:2],plane-.010]+[-offset[1],offset[0],offset[2]]))
    cloth.set('quat',vec([np.sqrt(.5),0,0,np.sqrt(.5)]))
    for g in world.findall('.//geom'):
        g.set('solref',str(2.5*dt)+' 1');g.set('solimp','.999 .9999 .0002')
        g.set('friction','.55 .005 .0001');g.set('margin','.00045')
    # Rounded garment hems supply continuous contacts at the corners of each
    # native convex segment. All capsule surfaces lie inside its measured bounds.
    from itertools import product
    half=np.array([CLOTH[1]/6,CLOTH[0]/2,CLOTH[2]/2])
    radius=.0017
    for b in world.findall(".//body"):
        if not b.get('name','').startswith('payload_'):continue
        for axis in (0,1,2):
            axes=[k for k in (0,1,2) if k!=axis]
            for signs in product((-1,1),repeat=2):
                pt=np.zeros(3);pt[axes]=np.array(signs)*(half[axes]-radius)
                ends=np.stack([pt,pt]);ends[:,axis]=[-half[axis]+radius,half[axis]-radius]
                ET.SubElement(b,'geom',type='capsule',fromto=vec(ends),size=str(radius),mass='0',
                              contype='8',conaffinity='2',solref=str(2.5*dt)+' 1',solimp='.999 .9999 .0002',friction='.55 .005 .0001')
    flex=ET.SubElement(world,'flexcomp',name='bag',type='direct',dim='2',point=vec(ref),element=vec(F.astype(int)),mass=str(film),radius='.00025')
    ET.SubElement(flex,'elasticity',young=str(2e7*bend_factor),poisson='.3',thickness='.00006',elastic2d='bend',damping='.00005')
    ET.SubElement(flex,'edge',equality='false',solref=str(3*dt)+' 1',solimp='.997 .999 .0005')
    ET.SubElement(flex,'contact',contype='2',conaffinity='10',friction='.38 .005 .0001',selfcollide='auto',margin='.0003',solref=str(3*dt)+' 1',solimp='.995 .999 .0005')
    for i in np.flatnonzero(pins):
        ET.SubElement(world,'body',name='pad_mount_'+str(i),mocap='true',pos=vec(ref[i]))
        ET.SubElement(eq,'connect',body1='bag_'+str(i),body2='pad_mount_'+str(i),anchor='0 0 0',
                      solref=str(2*dt)+' 1',solimp='.9998 .99995 .0001')
    xml=ET.tostring(root,encoding='unicode');proto=mj.MjModel.from_xml_string(xml)
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.xml') as tmp:
        mj.mj_saveLastXML(tmp.name,proto)
        expanded=ET.parse(tmp.name).getroot()
    tendons=ET.SubElement(expanded,'tendon')
    for i in range(len(ref)):
        b=expanded.find(f".//body[@name='bag_{i}']")
        ET.SubElement(b,'site',name='p'+str(i),size='.0001')
    for (a,b),l in zip(E,lengths(ref,E)):
        tendon=ET.SubElement(tendons,'spatial',limited='true',range=f'0 {l:.10g}',width='.0001',
                             solreflimit=str(3*dt)+' 1',solimplimit='.999 .9999 .0002')
        for k in (a,b):ET.SubElement(tendon,'site',site='p'+str(k))
    m=mj.MjModel.from_xml_string(ET.tostring(expanded,encoding='unicode'))
    d=mj.MjData(m);nodes=m.flex_vertbodyid
    qa=m.jnt_qposadr[m.body_jntadr[nodes]][:,None]+np.arange(3)
    d.qpos[qa]=held-ref
    for i in np.flatnonzero(pins):d.mocap_pos[m.body_mocapid[m.body('pad_mount_'+str(i)).id]]=held[i]
    film_dofs=m.jnt_dofadr[m.body_jntadr[nodes]][:,None]+np.arange(3)
    # Garment compression springs are the same segmented model as the workcell.
    compression=[(m.joint(j).qposadr[0],m.joint(j).dofadr[0]) for j in range(m.njnt) if m.joint(j).name.startswith('compression_')]
    from bag_sim.physical_parcel import compression_force
    mj.mj_forward(m,d)
    vprev=None; trace=[]
    rest_lengths=lengths(ref,E)
    from project import color_edges
    batches=color_edges(E)
    # These are corrections to native DOFs before the next contact solve.
    # They retain the hard material limit even during the first load transient.
    edges_by_color=[(g[:,0],g[:,1],lengths(ref,g)) for g in batches]
    inv_mass=1/m.body_mass[nodes];inv_mass[pins]=0
    offsets=d.flexvert_xpos.copy()-d.qpos[qa]
    peak_trial=0.; projected_steps=0
    maximum_ratio=0.
    peak_time=0.; peak_edge=-1
    maximum_after_settling=0.
    for j in range(round(seconds/dt)):
        # Progressively transfer the load as the prepared equilibrium fixture
        # is released. The last seven seconds use full terrestrial gravity.
        u=min(1.,j*dt/.20)
        m.opt.gravity[2]=-9.81*u*u*(3-2*u)
        trial=d.qpos[qa]+offsets
        pre=float(np.max(lengths(trial,E)/rest_lengths)); peak_trial=max(pre,peak_trial)
        if pre>1.043:
            corrected=trial.copy()
            for _ in range(24):
                for a,b,l0 in edges_by_color:
                    delta=corrected[b]-corrected[a];length=np.linalg.norm(delta,axis=1)
                    factor=np.maximum(length-1.04*l0,0)/np.maximum((inv_mass[a]+inv_mass[b])*length,1e-20)
                    corrected[a]+=delta*(inv_mass[a]*factor)[:,None]
                    corrected[b]-=delta*(inv_mass[b]*factor)[:,None]
                if np.max(lengths(corrected,E)/rest_lengths)<=1.045:break
            correction=corrected-trial
            d.qpos[qa]+=correction
            d.qvel[film_dofs]+=correction/dt
            projected_steps+=1
        mj.mj_step1(m,d)
        ratios=lengths(d.flexvert_xpos,E)/rest_lengths
        ratio=float(ratios.max())
        if ratio>maximum_ratio:
            maximum_ratio=ratio;peak_time=d.time;peak_edge=int(np.argmax(ratios))
        if d.time>=1.:maximum_after_settling=max(maximum_after_settling,ratio)
        d.qfrc_passive[film_dofs] -= .003*d.qvel[film_dofs]
        for a,b in compression:d.qfrc_passive[b]-=compression_force(d.qpos[a],d.qvel[b],cfg)
        mj.mj_step2(m,d)
        if j%round(.05/dt)==0:
            v=d.flexvert_xpos.copy()
            trace.append([d.time,*np.ptp(v,axis=0),np.max(lengths(v,E)/lengths(ref,E)),0 if vprev is None else np.max(np.linalg.norm(v-vprev,axis=1))/.05])
            vprev=v
            if j%round(.25/dt)==0:print(trace[-1],flush=True)
            if d.warning.number.any():raise RuntimeError(d.warning.number)
    mj.mj_forward(m,d)
    corners=[]; box_centers=[]; box_rotations=[]; box_halves=[]; surfaces=[]
    for g in range(m.ngeom):
        if m.geom(g).name.startswith(('plate_geom_','payload_geom_')):
            mi=m.geom_dataid[g];vv=m.mesh_vert[m.mesh_vertadr[mi]:m.mesh_vertadr[mi]+m.mesh_vertnum[mi]]
            rr=d.geom_xmat[g].reshape(3,3).copy(); half=np.ptp(vv,axis=0)/2
            c=(vv.max(0)+vv.min(0))/2@rr.T+d.geom_xpos[g]
            box_centers.append(c);box_rotations.append(rr);box_halves.append(half)
            xyz=vv@rr.T+d.geom_xpos[g];corners.extend(xyz)
            axis=int(np.argmin(half));up=rr[:,axis]*np.sign(rr[2,axis])
            surfaces.append((c+half[axis]*up,up))
    corners=np.asarray(corners)
    cp,nh=surfaces[len(surfaces)//2]
    plane_height=cp[2]-np.dot(nh[:2],apex[:2]-cp[:2])/nh[2]
    np.savez_compressed(out,proposed_peak_before_projection=peak_trial,projected_steps=projected_steps,peak_all_substeps=maximum_ratio,peak_time=peak_time,peak_edge=peak_edge,peak_after_settling=maximum_after_settling,box_centers=box_centers,box_rotations=box_rotations,box_halves=box_halves,V=d.flexvert_xpos,F=F,E=E,ref=ref,seam=seam,corners=corners,top_z=plane_height,
       pads=pads,links=links,quats=quats,names=names,pins=pins,trace=trace,hold_seed=held,time=d.time)
    print('finished',d.time,pads.mean(0)[2]-plane_height,flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('output');a.add_argument('--seconds',type=float,default=.8);a.add_argument('--dt',type=float,default=.00025)
    a.add_argument("--patch-half",type=float,default=.0045)
    a.add_argument("--bend-factor",type=float,default=1.)
    a.add_argument("--initial-gap",type=float,default=.0935)
    v=a.parse_args();settle(v.output,v.seconds,v.dt,v.patch_half,v.bend_factor,v.initial_gap)
