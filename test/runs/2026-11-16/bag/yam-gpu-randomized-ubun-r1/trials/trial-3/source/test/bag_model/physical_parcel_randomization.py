"""Seeded, saved randomization and actual membrane-surface grasp sampling."""
from dataclasses import asdict
import numpy as np
from physical_parcel_yam import Config, parcel_world
from bag_sim.parcel_contents import mass_properties

RANGES = dict(
    distributions='uniform except density and payload joint stiffness (log uniform); shape/mode/profile stratified',
    bag_size_m=[[.30,.40],[.22,.29]],
    materials=dict(kraft=dict(young_pa=[2e9,5e9],thickness_m=[.00010,.00018],density_kg_m3=[700,1000]),
                   pe=dict(young_pa=[1e7,3e7],thickness_m=[.00004,.00009],density_kg_m3=[900,960]),
                   bubble=dict(young_pa=[1e5,3e5],thickness_m=[.0015,.0025],density_kg_m3=[45,80])),
    poisson=[.25,.38], bag_friction=[.3,.7], payload_friction=[.25,.8],
    box_size_m=[[.075,.14],[.045,.095],[.014,.036]], sphere_diameter_m=[.028,.06],
    axial_length_m=[.065,.125],axial_diameter_m=[.022,.045],
    density_kg_m3=[120,1800], density_gradient_component=[-.18,.18],
    payload_euler_rad=[[-.12,.12],[-.12,.12],[-3.141592653589793,3.141592653589793]],
    payload_offset_m=[[-.015,.015],[-.012,.012],[-.001,.001]],
    joint_stiffness_Nm_rad=[.02,.2], damping_ratio=[.2,.8],
    compression_modulus_pa=[20000,80000],compression_densification=[10,30],
    cushion_core_young_pa=[6000,14000],cushion_damping_ratio=[.8,1.2],
    rayleigh_s=[.0003,.0008],bend_damping_ratio=[.4,.8],
    grasp_twist_deg=[-35,35],grasp_normal_tilt_deg=[0,15],
    per_world_gpu_reset=dict(distributions='independent uniform, stiffness/damping log uniform; stored episode and seed',
         payload_mass_scale=[.8,1.2],payload_friction=[.25,.65],table_friction=[.3,.7],
         joint_stiffness_scale=[.25,4.],joint_damping_scale=[.5,2.],compression_stiffness_scale=[.5,2.],
         compression_damping_rule='sqrt(compression_stiffness_scale)'),
    group_policy='geometry, pose, bag material are sampled per compile group; mass, payload/table friction, compliance independently per GPU environment')

ARCHETYPES=[('box','rigid'),('sphere','rigid'),('cylinder','rigid'),('capsule','rigid'),('box','bendable'),('box','compressible')]


def sample_config(rng,profile,shape,mode):
    uniform=lambda pair:float(rng.uniform(*pair))
    material={key:uniform(bounds) for key,bounds in RANGES['materials'][profile].items()}
    material['poisson']=uniform(RANGES['poisson'])
    if shape=='box':
        size=np.array([uniform(b) for b in RANGES['box_size_m']])
    elif shape=='sphere':
        size=np.full(3,uniform(RANGES['sphere_diameter_m']))
    else:
        diameter=uniform(RANGES['axial_diameter_m'])
        size=np.array([uniform(RANGES['axial_length_m']),diameter,diameter])
    density=float(np.exp(rng.uniform(*np.log(RANGES['density_kg_m3']))))
    volume=mass_properties(shape,size,1.)['volume_m3']
    gradient=rng.uniform(-.18,.18,3) if mode=='rigid' else np.zeros(3)
    # Segmented bending stiffness and damping scale with the same mass/inertia.
    k=float(np.exp(rng.uniform(*np.log(RANGES['joint_stiffness_Nm_rad']))))
    inertia=mass_properties(shape,size,density*volume)['inertia']
    damping=2*uniform(RANGES['damping_ratio'])*np.sqrt(k*inertia[1,1]/3)
    area=size[0]*size[1]/3
    modulus=uniform(RANGES['compression_modulus_pa'])
    k1=modulus*area/size[2]
    k2=k1*uniform(RANGES['compression_densification'])/size[2]
    cfg=Config(geometry='geometry/kraft_edge_sealed_35.npz',bag_profile=profile,material=material,
        bag_size=tuple(uniform(b) for b in RANGES['bag_size_m']),payload_size=tuple(size),
        payload_mass=density*volume,payload_shape=shape,mode=mode,density_gradient=tuple(gradient),
        payload_euler=tuple(uniform(b) for b in RANGES['payload_euler_rad']),
        payload_offset=tuple(uniform(b) for b in RANGES['payload_offset_m']),
        friction=uniform(RANGES['bag_friction']),payload_friction=uniform(RANGES['payload_friction']),
        joint_stiffness=k,joint_damping=float(damping),compression_k1=k1,compression_k2=k2,
        compression_damping=float(2*uniform(RANGES['damping_ratio'])*np.sqrt(k1*density*volume/30)),
        dt=.0001,rayleigh=uniform(RANGES['rayleigh_s']),bend_damping_ratio=uniform(RANGES['bend_damping_ratio']),
        bag_radius={'kraft':.00035,'pe':.00025,'bubble':.00175}[profile],
        cushion_core_young_pa=uniform(RANGES['cushion_core_young_pa']) if profile=='bubble' else None,
        cushion_damping_ratio=uniform(RANGES['cushion_damping_ratio']) if profile=='bubble' else None,
        support_height=.15,gripper_overtravel=.005,complete_motion=True,simulation_seconds=18.,wall_limit=1800.)
    return cfg


def sample_surface(parcel,cfg,rng,region):
    points=parcel.envelope['current']+[0,0,cfg.support_height]
    faces=parcel.envelope['faces']
    tri=points[faces]
    normals=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    areas=np.linalg.norm(normals,axis=1)
    normals/=areas[:,None]
    ref=parcel.envelope['reference'][faces].mean(axis=1)
    radial=np.abs((ref[:,:2]-parcel.cfg.center)/(np.asarray(cfg.bag_size)/2))
    if region=='top': mask=(normals[:,2]>.5)&(radial.max(1)<.6)
    elif region=='edge': mask=(normals[:,2]>.1)&(radial.max(1)>.65)&(radial.min(1)<.6)
    elif region=='corner': mask=(normals[:,2]>.1)&(radial.min(1)>.55)
    elif region=='side': mask=(normals[:,2]>0)&(radial.max(1)>.7)
    else: raise ValueError('unknown surface region')
    ids=np.flatnonzero(mask)
    if not len(ids):raise ValueError('surface region has no triangles')
    index=int(rng.choice(ids,p=areas[ids]/areas[ids].sum()))
    bary=rng.dirichlet([1.,1.,1.])
    point=bary@tri[index]
    normal=normals[index].copy()
    if region=='side':
        # Near-horizontal, slightly downward approach to the actual side seam.
        normal=np.r_[point[:2]-parcel.cfg.center,.01]
        normal/=np.linalg.norm(normal)
    tangent=np.cross(normal,[1.,0.,0.])
    if np.linalg.norm(tangent)<.1:tangent=np.cross(normal,[0.,1.,0.])
    tangent/=np.linalg.norm(tangent)
    twist=np.deg2rad(rng.uniform(-35,35))
    closing=np.cos(twist)*tangent+np.sin(twist)*np.cross(normal,tangent)
    tilt=np.deg2rad(rng.uniform(0,15))
    normal=np.cos(tilt)*normal+np.sin(tilt)*np.cross(closing,normal)
    return dict(surface_point=tuple(point),surface_normal=tuple(normal),surface_closing=tuple(closing),
                surface_region=region),dict(triangle=index,barycentric=bary.tolist(),point_m=point.tolist(),
                normal=normal.tolist(),closing=closing.tolist(),twist_deg=float(np.rad2deg(twist)),
                normal_tilt_deg=float(np.rad2deg(tilt)))
