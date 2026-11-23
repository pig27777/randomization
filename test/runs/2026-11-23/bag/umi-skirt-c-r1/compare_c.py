"""Render and measure the case-C hold pose from two independent snapshots."""
from model_c import *
from convex import audit,planes_from_mesh
from still import still
from PIL import Image,ImageDraw,ImageFont
import json,argparse

def snapshot(path, native):
    a=np.load(path);v=a['V'];p=a['pads'].mean(0);f=a['F'];e=a['E']
    ref=a['ref'];initial=template()[-1] if native else a['initial']
    if native:
        solids=[planes_from_mesh(x,y) for x,y in zip(a['garment_vertices'],a['garment_faces'])]
    elif 'half' in a:
        from project import audit as box_audit
        crossing,outside=box_audit(v,f,a['centers'],a['frame'],a['half'])
        solids=None
    else:
        solids=[a[k] for k in a.files if k.startswith('planes_')]
    if solids is not None:crossing,outside=audit(v,f,solids)
    rho=template()[4];side=template()[3]
    ids=np.flatnonzero((rho==.83)&(side>0)&(np.abs(ref[:,:2]-CENTER).min(1)<1e-6))
    tilts=np.rad2deg(np.arctan2(p[2]-v[ids,2],np.linalg.norm(v[ids,:2]-p[:2],axis=1)))
    return dict(height_cm=float((p[2]-v[:,2].min())*100),length_cm=float(np.ptp(v[:,1])*100),
        width_cm=float(np.ptp(v[:,0])*100),flat_ratio=float(max(lengths(v,e)/lengths(ref,e))),
        packed_ratio=float(max(lengths(v,e)/lengths(initial,e))),minimum_clearance_cm=float(v[:,2].min()*100),
        slope_deg=tilts.tolist(),crossing=int(crossing),outside=int(outside))

def compare(left,right,output):
    native=np.load(left);is_replay='tip_and_capture' in native
    if is_replay:
        if native['capture']<0 or native['V'][:,2].min()<=.15:raise ValueError('replay has no suspended grasp')
        mode='8.00 s  |  captured during trajectory replay'
    else:mode='Hold pose  |  suspended fixture at the 8.00 s jaw pose'
    data={};canvas=Image.new('RGB',(1920,824),'#19252e');d=ImageDraw.Draw(canvas)
    fonts='/usr/share/fonts/truetype/dejavu/DejaVuSans'
    head=ImageFont.truetype(fonts+'-Bold.ttf',25);font=ImageFont.truetype(fonts+'.ttf',19)
    for i,(label,src,heading,desc) in enumerate([('native',left,'MuJoCo flex  /  PE shell',mode),('proxy',right,'Low-order proxy  /  PBD visual skin','8.00 s  |  recorded bilateral grasp')]):
        m=data[label]=snapshot(src,i==0);dest=output.parent/(label+'_c_detail.png');still(src,dest)
        x=960*i;canvas.paste(Image.open(dest).convert('RGB'),(x,60))
        d.text((x+18,4),heading,font=head,fill='white');d.text((x+18,34),desc,font=font,fill='#c8d7e1')
        d.text((x+18,790),f"Grip to base  {m['height_cm']:.2f} cm     |     L × W  {m['length_cm']:.2f} × {m['width_cm']:.2f} cm     |     Max L/L0  {m['flat_ratio']:.4f}",font=font,fill='white')
    d.line((959,0,959,824),fill='#77919e',width=2);canvas.save(output)
    data['native_method']=mode
    output.with_suffix('.json').write_text(json.dumps(data,indent=2)+'\n')
if __name__=='__main__':compare(Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]))
