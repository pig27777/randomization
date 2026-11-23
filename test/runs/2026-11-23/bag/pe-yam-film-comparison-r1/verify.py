"""Gate publication on the shared boundary, bilateral support and continuous physical hold."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np

def verify(directory,ref_path,shell_path,proxy_path):
    s=np.load(shell_path);p=np.load(proxy_path);q=np.load(ref_path)
    linfo=json.loads(shell_path.with_suffix('.json').read_text())
    rinfo=json.loads(proxy_path.with_suffix('.json').read_text())
    # The last visible frame is at 9.56 s. Use that exact three-second window.
    start,end=6.56,9.56
    def during(t):
        return (t>=start-1e-9)&(t<=end+1e-9)
    w=during(s['time'])
    h=s['hold_audit'];h=h[during(h[:,0])]
    # audit_2ms timestamps label the start of the just-integrated proxy step.
    a=p['audit_2ms'];a=a[during(a[:,0]+.002)]
    shell_bottom=np.minimum(s['clearance'][w],s['diagnostics'][w,-1])
    shell_min=min(h[:,1].min(),shell_bottom.min())
    nominal=q['q_path']
    calibration={'ref_sha256':hashlib.sha256(ref_path.read_bytes()).hexdigest(),
                 'maximum_joint_or_rail_error':float(abs(s['q_path']-nominal).max()),
                 'maximum_pad_fk_error_m':float(abs(s['pad_path'][::20]-p['pads']).max())}
    diag=s['diagnostics'];enclosure=s['enclosure']
    facts=dict(hold_s=[start,end],timebase='25 fps; inclusive endpoints, 240 display frames',trajectory=calibration,
        shell=dict(wall_s=linfo['wall_s'],steps_s=linfo['steps_s'],dt=linfo.get('dt',.0002),steps=linfo['steps'],
                   min_clearance_mm=1000*shell_min,max_com_speed_mm_s=1000*h[:,2].max(),
                   min_grip_N=float(s['grip'][w].min()),max_support_N=float(abs(s['support'][w]).max()),
                   native_peak_overlap_mm=diag[:,1].max(),hold_peak_overlap_mm=diag[w,1].max(),
                   geometry_peak_box_intrusion_mm=1000*enclosure[:,2].max(),max_exposed_box_corners=int(enclosure[:,1].max()),
                   closest_film_to_table_mm=float(1000*s['clearance'].min()),
                   self_overlap_mm=1000*diag[:,4].max(),arm_table_overlap_mm=1000*diag[:,3].max(),
                   nonfinger_film_overlap_mm=diag[:,2].max(),mass_kg=linfo['mass_kg'],hold_samples=len(h)),
        proxy=dict(wall_s=rinfo['wall_s'],steps_s=rinfo['steps_s'],dt=rinfo['dt'],steps=rinfo['steps'],
                   min_clearance_mm=1000*a[:,1].min(),max_com_speed_mm_s=1000*a[:,3].max(),
                   min_captured_fingers=int(2*a[:,4].min()),max_support_N=abs(a[:,2]).max(),
                   mass_kg=rinfo['mass'],hold_samples=len(a)))
    passed=(len(h)==round(3/linfo.get('dt',.0002))+1 and len(a)==1501 and len(s['V'])==len(p['com'])==241
        and abs(linfo['mass_kg']-rinfo['mass'])<1e-8
        and np.allclose(s['time'],p['time'],rtol=0,atol=1e-10)
        and np.isfinite(h).all() and np.isfinite(a).all()
        and calibration['maximum_joint_or_rail_error']<1e-9 and calibration['maximum_pad_fk_error_m']<1e-7
        and shell_min>.15 and a[:,1].min()>.15 and h[:,2].max()<.020 and a[:,3].max()<.020
        and s['grip'][w].min()>.05 and np.all(a[:,4]==1)
        and abs(s['support'][w]).max()<1e-5 and abs(a[:,2]).max()<1e-5
        and np.max(abs(diag[:,2:5]))<1e-8)
    facts['capture_and_hold_gate']=bool(passed)
    facts['shell']['min_bilateral_substep_contact']=int(h[:,3:5].min()) if h.shape[1]>=5 else None
    # A lifted body is insufficient when material escapes the enclosing film.
    numerical=(diag[:,1].max()<=1. and enclosure[:,1].max()==0 and enclosure[:,2].max()<=.001)
    facts['shell_geometry_gate']=bool(numerical)
    if h.shape[1]>=5:passed=passed and np.all(h[:,3:5]>0)
    passed=passed and numerical
    facts['mission_gate']=bool(passed)
    (directory/'physics-check.json').write_text(json.dumps(facts,indent=2)+'\n')
    print(json.dumps(facts,indent=2))
    if not passed:raise SystemExit('hold or common-boundary check failed')
if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for arg in ('out','reference','shell','proxy'):ap.add_argument(arg,type=Path)
    v=ap.parse_args();verify(v.out,v.reference,v.shell,v.proxy)
