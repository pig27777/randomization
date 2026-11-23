"""One 8 s calibration still with the two independent measurements."""
from PIL import Image,ImageDraw,ImageFont
from measure import *
from still import still
from pathlib import Path
import json
def compare(native,proxy,out):
    base=Path(out).parent
    canvas=Image.new('RGB',(1920,828),'#19252e');draw=ImageDraw.Draw(canvas)
    fam='/usr/share/fonts/truetype/dejavu/'
    title=ImageFont.truetype(fam+'DejaVuSans-Bold.ttf',25)
    note=ImageFont.truetype(fam+'DejaVuSans.ttf',20)
    metrics={}
    for i,(side,source,heading,detail) in enumerate([
        ('native',native,'MuJoCo flex  /  PE thin shell','8.00 s  |  suspended equilibrium, pad patch held'),
        ('proxy',proxy,'Low-order proxy  /  PBD visual skin','8.00 s  |  recorded bilateral grasp')]):
        a=measure(source);metrics[side]=a
        target=base/(side+'_steady.png')
        still(source,target)
        x=960*i
        canvas.paste(Image.open(target).convert('RGB'),(x,61))
        draw.text((x+20,6),heading,font=title,fill='#ffffff');draw.text((x+20,35),detail,font=note,fill='#bbcbd3')
        draw.text((x+20,792),f"Tip to textile  {a['height_cm']:.2f} cm     |     Footprint  {a['short_cm']:.2f} × {a['long_cm']:.2f} cm     |     Max L/L0  {a['max_edge_ratio']:.5f}",font=note,fill='#edf4f5')
    draw.line((959,0,959,828),fill='#647780',width=2)
    canvas.save(out)
    (base/'hold_measurements.json').write_text(json.dumps(metrics,indent=2)+'\n')
if __name__=='__main__':
    import sys
    compare(*sys.argv[1:])
