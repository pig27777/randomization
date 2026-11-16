"""Independent CPU reference fixtures built from actual GPU model parameters.

No robot actions are implemented. Removing the fixture and clamping the first
body are explicit boundary conditions of material tests. Baseline forces remain
MuJoCo native springs plus the production quadratic compression expression.
"""
import argparse
import xml.etree.ElementTree as ET
from common import ROOT,setup,write
import numpy as np
import mujoco
import mujoco_warp as mjw
import warp as wp
from contents.parcel import Batch
from contents.parcel import kernels as K
from contents.parcel.model import xml,initialize_fixture
from reference import cantilever_reference


def set_params(b,p):
    b.params.assign(p)
    wp.launch(K.configure,b.nworld,[int(b.rigid),b.cpu.geom('table').id,b.ids,b.params,b.model.body_pos,
        b.model.body_ipos,b.model.body_mass,b.model.body_inertia,b.model.geom_size,b.model.geom_friction,
        b.model.jnt_pos,b.model.jnt_stiffness,b.model.dof_damping,b.model.qpos0,b.data.qpos])
    wp.launch(initialize_fixture,b.nworld,[b.params,b.addresses,b.model.qpos0,b.data.qpos])
    mjw.set_const(b.model,b.data);mjw.forward(b.model,b.data)


def clone(b,clamp=True,fixture=False,gravity=9.81,dt=.0005):
    tree=ET.fromstring(xml(b.rigid,dt))
    for body in tree.iter('body'):body.set('simple','false')
    if clamp:
        root=tree.find(".//body[@name='segment0']")
        root.remove(root.find('freejoint'))
        for i in range(1 if b.rigid else 3):
            ET.SubElement(tree.find('contact'),'exclude',body1=f'segment{i}',body2=f'cap{i}')
    if not fixture:
        world=tree.find('worldbody');world.remove(world.find("body[@name='cart']"))
    m=mujoco.MjModel.from_xml_string(ET.tostring(tree,encoding='unicode'))
    bm=b.model;source=b.cpu
    for field in ('body_pos','body_ipos','body_mass','body_inertia','body_quat','body_iquat'):
        arr=getattr(bm,field).numpy()[0]
        for i in range(1,m.nbody):getattr(m,field)[i]=arr[source.body(m.body(i).name).id]
    for field in ('geom_size','geom_friction'):
        arr=getattr(bm,field).numpy()[0]
        for i in range(m.ngeom):getattr(m,field)[i]=arr[source.geom(m.geom(i).name).id]
    stiffness=bm.jnt_stiffness.numpy()[0];damping=bm.dof_damping.numpy()[0]
    q0=bm.qpos0.numpy()[0];pos=bm.jnt_pos.numpy()[0]
    for j in range(m.njnt):
        old=source.joint(m.joint(j).name).id
        m.jnt_pos[j]=pos[old];m.jnt_stiffness[j]=stiffness[old]
        size=6 if m.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE else 1
        qa=m.jnt_qposadr[j];va=m.jnt_dofadr[j]
        qo=source.jnt_qposadr[old];vo=source.jnt_dofadr[old]
        nq=7 if size==6 else 1
        m.qpos0[qa:qa+nq]=q0[qo:qo+nq]
        m.dof_damping[va:va+size]=damping[vo:vo+size]
    if clamp:m.body_pos[m.body('segment0').id]=q0[:3]+[0,0,.5]
    m.opt.gravity[:]=[0,0,-gravity]
    d=mujoco.MjData(m);mujoco.mj_setConst(m,d);mujoco.mj_resetData(m,d)
    if fixture:
        p=b.params.numpy()[0]
        d.qpos[m.joint('cart_x').qposadr[0]]=m.body_pos[m.body('segment0').id,0]
        d.qpos[m.joint('cart_z').qposadr[0]]=m.body_pos[m.body('segment0').id,2]+.6*p['size'][2]+.025
        for side in ('left','right'):d.qpos[m.joint(side).qposadr[0]]=.025
    mujoco.mj_forward(m,d)
    return m,d


def compression_force(m,d,p):
    for i in range(3):
        joint=m.joint(f'compress{i}');depth=max(-d.qpos[joint.qposadr[0]],0)
        d.qfrc_applied[joint.dofadr[0]]+=float(p['quadratic'])*depth*depth


def energy(m,d,p):
    mujoco.mj_energyPos(m,d);mujoco.mj_energyVel(m,d)
    compression=0.;linear=0.
    for i in range(3):
        j=m.joint(f'compress{i}');q=d.qpos[j.qposadr[0]]
        compression+=float(p['quadratic'])*max(-q,0)**3/3
        linear+=.5*float(p['compression'])*q*q
    # Native position energy includes gravity and all native linear springs.
    grav=float(np.sum(m.body_mass*d.xipos[:,2])*abs(m.opt.gravity[2]))
    native=float(d.energy[0])
    return np.array([float(d.energy[1]),grav,native-grav-linear,linear,compression])


def bending(out,b):
    rows=[]
    for length,mass,target,gravity in [(L,M,S,G) for L,M in ((.20,.25),(.28,.5),(.32,.9)) for S in (.005,.02,.04,.06) for G in (0.,9.81)]:
        p=b.params.numpy();p['size'][0,0]=length;p['mass']=mass;p['weights']=1/3;p['com']=0
        a=length/3;k=mass*9.81*5*a*a/target;p['bend']=k;p['twist']=k*.5
        set_params(b,p);m,d=clone(b,gravity=gravity)
        end=m.body('segment2').id
        def tip():return d.xpos[end]+d.xmat[end].reshape(3,3)@np.array([a/2,0,0])
        initial=tip().copy();force=np.array([0,0,-mass*9.81]);trace=[]
        dofs=[m.joint(f'rot{i}_1').dofadr[0] for i in (1,2)]
        for step in range(6000):
            mujoco.mj_step1(m,d);d.qfrc_applied[:]=0
            compression_force(m,d,p[0]);mujoco.mj_applyFT(m,d,force,np.zeros(3),tip(),end,d.qfrc_applied)
            mujoco.mj_step2(m,d)
            if step%40==0:trace.append([d.time,initial[2]-tip()[2],*d.qpos[[m.joint(f'rot{i}_1').qposadr[0] for i in (1,2)]]])
        measured=initial[2]-tip()[2]
        angles=d.qpos[[m.joint(f'rot{i}_1').qposadr[0] for i in (1,2)]]
        refq,refsag=cantilever_reference(a,k,mass*9.81,[mass/3,mass/3],gravity,float(p['size'][0,2]))
        row=dict(length=length,mass=mass,target=target,gravity=gravity,k=k,measured=measured,
                 relative_target_error=(measured-target)/target,reference=refsag,reference_angles=refq,
                 joint_angles=angles,restoring_torque=-k*angles,max_qvel=float(abs(d.qvel).max()))
        rows.append(row)
        np.savez_compressed(out/f'bend-L{length}-M{mass}-S{target}-g{gravity}.npz',trace=trace,params=p)
    return rows


def compression(out,b):
    rows=[]
    # Direct first-cap force versus an actual native-contact platen, same κ.
    for prototype in (2,4):
        base=Batch(1,prototype=prototype)
        p=base.params.numpy();set_params(base,p)
        for fraction in (.01,.1,.3):
            m,d=clone(base,gravity=0.)
            cap=m.body('cap0').id;core=m.body('segment0').id
            total=m.body_mass[cap]+m.body_mass[core]
            # Controlled mass-only ablation, with exact matching inertia scaling.
            for body,mass in ((cap,total*fraction),(core,total*(1-fraction))):
                m.body_inertia[body]*=mass/m.body_mass[body];m.body_mass[body]=mass
            mujoco.mj_setConst(m,d)
            j=m.joint('compress0');qa,va=j.qposadr[0],j.dofadr[0]
            traj=[];work=0.;diss=0.;previous=0.
            for step in range(8000):
                t=step*m.opt.timestep
                load=20*min(t/.8,1) if t<2 else 0.
                mujoco.mj_step1(m,d);d.qfrc_applied[:]=0;compression_force(m,d,p[0]);d.qfrc_applied[va]-=load
                old=d.qpos[qa];velocity=d.qvel[va];mujoco.mj_step2(m,d)
                work-=load*(d.qpos[qa]-old);diss+=float(np.sum(m.dof_damping*d.qvel**2))*m.opt.timestep
                if step%20==0:
                    E=energy(m,d,p[0]);traj.append([d.time,-d.qpos[qa],load,velocity,*E,work,diss,E.sum()+diss-work])
            target=10/float(p['compression'][0])
            data=np.asarray(traj);steady=data[(data[:,0]>1.7)&(data[:,0]<1.95),1].mean()
            freq=np.sqrt(float(p['compression'][0])/(total*fraction+float(m.dof_armature[va])))/(2*np.pi)
            m.geom_contype[:]=0;m.geom_conaffinity[:]=0;m.jnt_limited[:]=0;m.dof_damping[:]=0
            m.opt.timestep=.00001;mujoco.mj_resetData(m,d);d.qpos[qa]=-1e-6
            ring=[]
            for sample in range(20000):
                d.qfrc_applied[:]=0;compression_force(m,d,p[0]);mujoco.mj_step(m,d)
                ring.append(d.qpos[qa])
            crossings=np.where((np.array(ring[:-1])<0)&(np.array(ring[1:])>=0))[0]*m.opt.timestep
            measured_frequency=float(1/np.diff(crossings).mean()) if len(crossings)>2 else None
            np.savez_compressed(out/f'ring-p{prototype}-mass{fraction}.npz',time=np.arange(len(ring))*m.opt.timestep,q=ring)
            rows.append(dict(prototype=prototype,cap_mass_fraction=fraction,target=target,measured=steady,
                relative_error=(steady-target)/target,linear_frequency_hz=freq,measured_frequency_hz=measured_frequency,final_energy_residual=float(data[-1,-1])))
            np.savez_compressed(out/f'compression-p{prototype}-mass{fraction}.npz',trace=data,params=p,
                columns=['time','depth','load','compression_velocity','KE','GPE','joint_U','compression_linear_U','compression_quadratic_U','work','dissipation','energy_residual'])
        for ramp in (.2,1.0):
            m,d=clone(base,fixture=True,gravity=0.)
            for name,target in [('cart_x',m.body_pos[m.body('segment0').id,0]),('cart_y',0.),('left',.025),('right',.025)]:
                j=m.joint(name);m.jnt_stiffness[j.id]=100000.;m.dof_damping[j.dofadr[0]]=100.
                m.qpos_spring[j.qposadr[0]]=target
            m.dof_damping[m.joint('cart_z').dofadr[0]]=10.
            saved_qpos=d.qpos.copy()
            mujoco.mj_setConst(m,d)
            d.qpos[:]=saved_qpos
            mujoco.mj_forward(m,d)
            qa=m.joint('compress0').qposadr[0];va=m.joint('cart_z').dofadr[0]
            trace=[];pad_ids={m.geom('left').id,m.geom('right').id};coreload=capload=0.
            buf=np.zeros(6)
            for step in range(8000):
                t=step*m.opt.timestep;load=20*min(t/ramp,1) if t<2.5 else 0.
                mujoco.mj_step1(m,d);d.qfrc_applied[:]=0;compression_force(m,d,p[0]);d.qfrc_applied[va]-=load
                mujoco.mj_step2(m,d)
                if step%20==0:
                    coreload=capload=0.
                    for i,c in enumerate(d.contact):
                        if not any(g in pad_ids for g in c.geom):continue
                        mujoco.mj_contactForce(m,d,i,buf)
                        if m.geom('core0').id in c.geom:coreload+=max(buf[0],0)
                        if m.geom('capgeom0').id in c.geom:capload+=max(buf[0],0)
                    trace.append([d.time,-d.qpos[qa],load,coreload,capload,float(np.min(d.contact.dist,initial=0))])
            data=np.array(trace);mask=(data[:,0]>2)&(data[:,0]<2.4)
            rows.append(dict(prototype=prototype,test='native_platen',ramp_seconds=ramp,
                target=10/float(p['compression'][0]),depth=float(data[mask,1].mean()),
                core_load=float(data[mask,3].mean()),cap_load=float(data[mask,4].mean())))
            np.savez_compressed(out/f'platen-p{prototype}-ramp{ramp}.npz',trace=data,params=p)
    return rows


def energy_tests(out,b):
    rows=[]
    for dt in (.00025,.0005,.001):
        for damp in (0.,.1):
            m,d=clone(b,gravity=0.,dt=dt)
            m.dof_damping[:]=damp
            # Controlled spring-only ledger excludes all contact and stop losses.
            m.geom_contype[:]=0;m.geom_conaffinity[:]=0;m.jnt_limited[:]=0
            d.qpos[m.joint('rot1_1').qposadr[0]]=.1
            d.qpos[m.joint('rot2_1').qposadr[0]]=-.05
            d.qpos[m.joint('compress0').qposadr[0]]=-.005
            mujoco.mj_forward(m,d);initial=energy(m,d,b.params.numpy()[0]).sum()
            trace=[];diss=0.
            for i in range(round(.1/dt)):
                mujoco.mj_step1(m,d);d.qfrc_applied[:]=0;compression_force(m,d,b.params.numpy()[0]);mujoco.mj_step2(m,d)
                mujoco.mj_forward(m,d)
                E=energy(m,d,b.params.numpy()[0]);diss+=float(np.sum(m.dof_damping*d.qvel**2))*dt
                trace.append([d.time,*E,diss,0.,E.sum()+diss-initial])
            data=np.array(trace)
            rows.append(dict(dt=dt,damping=damp,initial_energy=initial,final_energy=float(data[-1,1:6].sum()),
                relative_residual=float(data[-1,-1]/initial),max_positive_residual=float(max(data[:,-1].max(),0.)/initial)))
            np.savez_compressed(out/f'energy-dt{dt}-c{damp}.npz',trace=data,
                columns=['time','KE','GPE','joint_U','compression_linear_U','compression_quadratic_U','dissipation','external_work','residual'])
    return rows


def main(output):
    out=setup(output,'payload references and energy')
    b=Batch(1,prototype=3)
    write(out/'original_kappa.json',b.params.numpy())
    results={'bending':bending(out,b)}
    b.reset()
    results['compression']=compression(out,b)
    b.reset()
    results['energy']=energy_tests(out,b)
    write(out/'summary.json',results)
    print(out/'summary.json',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    main(parser.parse_args().output)
