"""Check 8 s geometry, common pad coordinates and the completed physics interval."""
from pathlib import Path
import sys,json,numpy as np
from compare_c import snapshot
from PIL import Image
def check(left,skin,episode,preview):
    left,skin,episode,preview=map(Path,(left,skin,episode,preview))
    a=np.load(left);r=np.load(episode);p=np.load(skin)
    if not np.allclose(a['pads'],p['pads'],rtol=0,atol=1e-8):raise ValueError('pad FK mismatch')
    metrics={key:snapshot(s,flag) for key,s,flag in [('native',left,True),('proxy',skin,False)]}
    with Image.open(preview) as im:
        assert im.size==(1920,824)
        im.verify()
    for name,side in metrics.items():
        assert max(side['flat_ratio'],side['packed_ratio'])<=1.05,(name,side)
        assert side['crossing']==side['outside']==0,(name,side)
    assert a['capture']>=0 and a['peak']<=1.05
    assert np.allclose(a['V_frames'][200],a['V'],atol=1e-7)
    tr=a['trace'];full=tr[-1,0]>=9.56;t0=6.56 if full else 5.
    span=(tr[:,0]>=t0-1e-7)&(tr[:,0]<=t0+3.+1e-7)
    assert np.all(a['tip_and_capture'][span,3]) and tr[span,4].min()>.15
    q=r['audit_2ms'];h=(q[:,0]>=t0-1e-7)&(q[:,0]<=t0+3.+1e-7)
    assert np.all(q[h,4]) and q[h,2].max()==0 and q[h,1].min()>.15
    stats=dict(native=metrics['native'],proxy=metrics['proxy'],hold_start_s=t0,hold_end_s=t0+3.,
               native_min_clearance_cm=tr[span,4].min()*100,proxy_min_block_clearance_cm=q[h,1].min()*100,
               proxy_max_com_speed_mm_s=q[h,3].max()*1000,native_substep_peak_ratio=float(a['peak']),
               native_capture_s=float(a['capture']),native_simulated_s=tr[-1,0],
               proxy_simulated_s=9.6,pad_error_mm=float(np.linalg.norm(a['pads']-p['pads'],axis=1).max()*1000))
    if 'COM_frames' in a:
        vel=np.diff(a['COM_frames'][span],axis=0)/.04
        stats['native_peak_com_speed_mm_s']=float(np.linalg.norm(vel,axis=1).max()*1000)
    if 'wall_s' in a:stats.update(native_wall_s=float(a['wall_s']),native_steps_s=float(a['steps_s']))
    preview.with_name('hold_checks.json').write_text(json.dumps(stats,indent=2)+'\n')
    print(json.dumps(stats,indent=2))
if __name__=='__main__':check(*sys.argv[1:])
