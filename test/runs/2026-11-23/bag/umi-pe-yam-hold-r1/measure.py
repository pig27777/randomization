"""Independent whole-mesh stretch, projected spans, and garment enclosure audit."""
from project import *
import json
def measure(path):
    a=np.load(path)
    v=a['V'];mesh=a['F'];ref=a['ref'];e=a['E'];pads=a['pads'];apex=pads.mean(0)
    if 'box_centers' in a.files:
        c=a['box_centers'];Rs=a['box_rotations'];hs=a['box_halves'];long=c[-1]-c[0]
        plane=a['top_z']
    else:
        c=a['centers'];Rs=np.repeat(a['R'][None],len(c),axis=0);hs=a['half'];long=a['R'][:,1]
        top=(c+hs[:,2,None]*a['R'][:,2]).mean(0)
        normal=a['R'][:,2];plane=top[2]-np.dot(normal[:2],apex[:2]-top[:2])/normal[2]
    along=long[:2]/np.linalg.norm(long[:2]); across=np.array([along[1],-along[0]])
    hits=np.sum([audit(v,mesh,[center],rot,[half]) for center,rot,half in zip(c,Rs,hs)],axis=0)
    strain=lengths(v,e)/lengths(ref,e)
    axis=pads[1]-pads[0];gap=np.linalg.norm(axis);axis/=gap
    patch_gap_error=max(0.,float(np.max(abs((v[a['pins']]-apex)@axis))-gap/2))
    rel=ref[:,:2]-ref[0,:2]
    seam=a['seam'];rulings=[]
    for axis in (0,1):
        for sign in (-1,1):
            j=seam[np.argmin(abs(rel[seam,1-axis])+np.abs(rel[seam,axis]-sign*BAG[axis]/2))]
            delta=apex-v[j]
            rulings.append(np.degrees(np.arctan2(delta[2],np.linalg.norm(delta[:2]))))
    return dict(material_sheet_cm=[39,29],package_nominal_thickness_cm=2, garment_cm=list(CLOTH[[1,0,2]]*100),
       height_cm=float(100*(apex[2]-plane)),short_cm=float(100*np.ptp(v[:,:2]@across)),long_cm=float(100*np.ptp(v[:,:2]@along)),
       world_AABB_xy_cm=(100*np.ptp(v[:,:2],axis=0)).tolist(),yaw_deg=float(np.degrees(np.arctan2(along[0],along[1]))),
       max_patch_outside_jaw_gap_mm=1000*patch_gap_error,max_edge_ratio=float(strain.max()),edges=len(e),
       accepted_native_substep_max=float(a['peak_all_substeps']) if 'peak_all_substeps' in a else None,
       native_substeps_projected=int(a['projected_steps']) if 'projected_steps' in a else None,
       native_trial_before_projection=float(a['proposed_peak_before_projection']) if 'proposed_peak_before_projection' in a else None,intersecting_triangle_box_pairs=int(hits[0]),outside_corners=int(hits[1]),
       meridian_slope_deg=list(rulings),max_snapshot_motion_mm_s=float(a['trace'][-1,-1]*1000) if 'trace' in a else None)
if __name__=='__main__':
    import sys
    for path in sys.argv[1:]:print(path,json.dumps(measure(path),indent=1))
