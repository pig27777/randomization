"""Replay the common 2 ms robot trajectory against the independent physical skirt and PE skin."""
from model_c import *
import argparse,time
def garment_geometry(m,d):
    gverts=[];gfaces=[]
    for i in range(4):
        g=m.geom(f'garment{i}').id;k=m.geom_dataid[g]
        verts=m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k]+m.mesh_vertnum[k]]@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]
        gverts.append(verts);gfaces.append(m.mesh_face[m.mesh_faceadr[k]:m.mesh_faceadr[k]+m.mesh_facenum[k]].copy())
    return gverts,gfaces

def run(out,seconds,dt):
    m,d,ref,F,E,pins,seam,qa,va=scene(dt)
    pads=pad_path()
    offset=d.flexvert_xpos.copy()-d.qpos[qa]
    batch=[(g[:,0],g[:,1],lengths(ref,g)) for g in color_edges(E)]
    L0=np.minimum(lengths(ref,E),lengths(d.flexvert_xpos,E));mass=1/m.body_mass[m.flex_vertbodyid]
    from limiter import compile_limit
    scratch,limit=compile_limit()
    E=E.copy(order='C');L0=L0.copy(order='C')
    held_mass=mass.copy();held_mass[pins]=0
    volume0=film_pressure(d.flexvert_xpos,F)*1.42  # Calibrated trapped volume for a tightly packed skirt.
    film_history=[];hud_history=[];centroids=[]
    measured_bodies=np.r_[m.flex_vertbodyid,[m.body(name).id for name in ["skirt_center","skirt0","skirt1","skirt2","skirt3"]]]
    masses=m.body_mass[measured_bodies]
    n=round(seconds/dt);hit=None;peak=0.;tr=[];fixed=0
    begin=time.perf_counter()
    for j in range(n+1):
        t=j*dt; cell=t/.002;lo=min(int(cell),len(pads)-2);morph=min(1.,cell-lo)
        p=(1-morph)*pads[lo]+morph*pads[lo+1]
        mid=p.mean(0);gap=np.linalg.norm(p[1]-p[0]);v=d.qpos[qa]+offset
        if t<2. and v[:,2].min()>.025:raise RuntimeError(f'unloaded bag lost table support at {t} s')
        if hit is None and gap<=.012 and 1.9<t<3.:
            center=d.xpos[m.body('skirt_center').id]
            # Closing pad faces overlap both the textile centre and top PE;
            # vertical overlap is tested against the finite 40 mm pad height.
            if np.max(abs((center-mid)[:2]))<.010 and -.004<center[2]-mid[2]<.022 and np.linalg.norm(v[pins,:2].mean(0)-mid[:2])<.013 and abs(v[pins,2].mean()-mid[2])<.022:
                hit=t;start=(v[pins].mean(0)+center)/2
                d.eq_active[:]=1
        if t>3.05 and hit is None:raise RuntimeError('No bilateral textile and membrane capture by 3.05 s')
        if hit is not None:
            u=np.clip((t-hit)/.15,0.,1.);u=u*u*(3-2*u)
            d.mocap_pos[0]=mid+(1-u)*(start-mid)
        else:d.mocap_pos[0]=mid
        v=v.copy(order='C');orig=v.copy()
        if hit is not None:
            # The same actual compliant pad region is held throughout the lift.
            # Preserve the captured film material, including its in-plane orientation.
            v[pins]=d.mocap_pos[0]+np.column_stack([ref[pins,:2]-CENTER,np.zeros(pins.sum())])
        limit(v,E,L0,mass if hit is None else held_mass,len(E))
        dp=v-orig;d.qpos[qa]+=dp;remove_outward_velocity(d,va,dp)
        if hit is not None:
            d.qvel[va[pins]]=(pads[min(lo+1,len(pads)-1)]-pads[lo]).mean(0)/.002
        fixed+=int(abs(dp).max()>1e-10)
        peak=max(peak,max(lengths(v,E)/L0))
        mj.mj_kinematics(m,d);mj.mj_flex(m,d)
        if j%round(.04/dt)==0:
            vaa=d.flexvert_xpos.copy()
            mj.mj_comPos(m,d)
            centroids.append(np.average(d.xipos[measured_bodies],axis=0,weights=masses))
            film_history.append(vaa.astype('f4'))
            hud_history.append([*mid,hit is not None])
            tr.append([t,*np.ptp(vaa,0),vaa[:,2].min(),mid[2],max(lengths(vaa,E)/L0),d.xpos[m.body('skirt_center').id,2]])
        if j%round(.5/dt)==0:print(t,'grip',hit,'H,extent/ratio',tr[-1],flush=True)
        if d.warning.number.any():raise RuntimeError(d.warning.number)
        if j==round(min(8.,seconds)/dt):
            snapshot_v=d.flexvert_xpos.copy();snapshot_pads=p.copy();gverts,gfaces=garment_geometry(m,d)
        if j==n:break
        d.qfrc_applied[va]=-.002*d.qvel[va]
        if hit is not None:
            d.qfrc_applied[va]+=min(1.,(t-hit)/.35)*film_pressure(d.flexvert_xpos,F,volume0)
        mj.mj_step(m,d)
    wall=time.perf_counter()-begin
    names,links,quats,final_pads=robot_state()
    if seconds>=8:assert np.allclose(final_pads,snapshot_pads,atol=1e-8)
    np.savez_compressed(out,wall_s=wall,steps=n,steps_s=n/wall,V=snapshot_v,COM_frames=centroids,F=F,E=E,ref=ref,seam=seam,pins=pins,trace=tr,peak=peak,projection_steps=fixed,
       V_frames=film_history,tip_and_capture=hud_history,capture=hit if hit is not None else -1,garment_vertices=gverts,garment_faces=gfaces,links=links,quats=quats,names=names,pads=final_pads)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('out');a.add_argument('--seconds',type=float,default=8);a.add_argument('--dt',type=float,default=.00025)
    v=a.parse_args();run(v.out,v.seconds,v.dt)
