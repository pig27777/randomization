"""Export actual archived model definitions and compiled counts without stepping dynamics."""
import json
import platform
from xml.etree import ElementTree as ET

import mujoco
import numpy as np

from probe_feedback import digest,save
from video_parcel_heap import BASE,ROOT,archived_trial


def inspect(name):
    heap,row,path=archived_trial(name)
    robot=heap.scene(row['case'],row['layout'],row['action'],row['jitter'])
    sim=robot.sim;m=sim.m;desc=sim.desc
    segmented=sim.info['has_segment_content']
    from model import load_table,table_entry,TABLE_PATH
    table=load_table();entry=table_entry(desc.kappa,table)
    selected=next((key for key,value in table.items() if value is entry),None)
    flexes=[]
    for i,node in enumerate(ET.fromstring(sim.built.xml).iter('flexcomp')):
        elastic={k:float(v) if k!='elastic2d' else v for k,v in node.find('elasticity').attrib.items()}
        thickness=elastic['thickness'];young=elastic['young'];poisson=elastic['poisson']
        flexes.append(dict(name=node.get('name'),vertices=int(m.flex_vertnum[i]),
                           triangles=int(m.flex_elemnum[i]),compiled_edges=int(m.flex_edgenum[i]),
                           mass_kg=float(node.get('mass')),contact_thickness_mm=float(m.flex_radius[i]*2000),
                           elasticity=elastic,membrane_Et_N_per_m=young*thickness,
                           bending_Et3_over_12_1nu2_Nm=young*thickness**3/(12*(1-poisson**2)),
                           contact=node.find('contact').attrib))
    axes=[]
    for axis in (range(3) if segmented else ()):
        joints=[j for j in desc.joints if j.axis==axis]
        axes.append(dict(axis='xyz'[axis],connections=len(joints),fixed=sum(j.fixed for j in joints),
                         k_comp_N_per_m=sorted({float(j.k_comp) for j in joints}),
                         k_tens_N_per_m=sorted({float(j.k_tens) for j in joints}),
                         k_shear_N_per_m=[list(v) for v in sorted({tuple(j.k_shear) for j in joints})],
                         k_bend_Nm_per_rad=[list(v) for v in sorted({tuple(j.k_bend) for j in joints})],
                         k_twist_Nm_per_rad=sorted({float(j.k_twist) for j in joints}),
                         lock_compression=sorted({float(j.lock_comp) for j in joints}),
                         lock_bending_rad=sorted({float(j.lock_bend) for j in joints})))
    return dict(case=row['case'],original=str(path.relative_to(ROOT)),original_sha256=digest(path),
                driver_sha256=digest(heap.__file__),kappa=desc.kappa.to_dict() if segmented else None,
                internal_mapping=desc.internal if segmented else None,
                mapping_entry=selected,mapping_table_sha256=digest(TABLE_PATH),
                segment_elastic_coefficients_active=segmented and not desc.rigid,
                content=dict(kind='segment_rigid' if segmented else 'paper_flex',
                             descriptor_cells=len(desc.segments) if segmented else None,
                             cell_size_mm=(desc.cell*1000).tolist() if segmented else None,
                             descriptor_connections=len(desc.joints) if segmented else None,
                             compiled_bodies=len(sim.content_bodies),compiled_geoms=len(sim.obj_geoms),
                             free_body_velocity_coordinates=int(sum(m.body_dofnum[b] for b in sim.content_bodies)),
                             elastic_bushings=0 if sim.bushings is None else sim.bushings.n,
                             nominal_mass_kg=desc.kappa.mass if segmented else sum(f['mass_kg'] for f in flexes if f['name']!='bag'),
                             compiled_mass_kg=float(m.body_mass[sim.content_bodies].sum()),
                             geom_friction=np.unique(m.geom_friction[list(sim.obj_geoms)],axis=0).tolist(),
                             mobile_enabled=desc.mobile is not None,joints_by_axis=axes),
                shell_initial_vertex_span_mm=(np.ptp(sim.info['shell_initial'],axis=0)*1000).tolist(),
                shell_sha256=sim.info['shell_sha256'],flexes=flexes,settings=row['settings'],
                compiled_model=dict(nv=m.nv,njnt=m.njnt,neq=m.neq,initial_active_equalities=int(sim.d.eq_active.sum())),
                inactive_parameters=dict(pinch_coeff='No pinch attachment is installed by this heap driver.',
                     surface_friction='Kappa field does not install pad-content pairs in the heap fixture; flex uses mu=0.6.',
                     skin_compliance='Heap fixture does not install the equivalent skin pad-content pair.',
                     mobile_defaults='mobile_frac/range/freq are inactive when mobile=False.'))


def main():
    rows=[inspect(n) for n in ('book','clothes','paper_envelope','card_envelope_side')]
    a,b=rows[:2]
    same={k:a['kappa'][k] for k in a['kappa'] if a['kappa'][k]==b['kappa'][k]}
    differences={k:dict(book=a['kappa'][k],clothes=b['kappa'][k]) for k in a['kappa'] if k not in same}
    assert a['shell_sha256']==b['shell_sha256'] and a['flexes']==b['flexes']
    assert a['content']['compiled_bodies']==1 and b['content']['compiled_bodies']==36
    assert a['content']['elastic_bushings']==0 and b['content']['elastic_bushings']==72
    assert a['kappa']['size']==b['kappa']['size'] and a['kappa']['mass']==b['kappa']['mass']
    save(BASE/'model_specification.json',dict(host=platform.node(),mujoco=mujoco.__version__,
         inspector_sha256=digest(__file__),models=rows,same_parameters=same,different_parameters=differences,
         audit_passed=True,dynamics_steps=0,
         note='Historical policy heap models. Effective coefficients are not measured book/fabric material properties.'))
    print(json.dumps(dict(audit_passed=True,dynamics_steps=0,differences=differences,
                          flexes=[r['flexes'] for r in (a,rows[2])]),ensure_ascii=False))


if __name__=='__main__':main()
