"""Render selected actual YAM trials with bag-category and outcome labels."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[2]
TITLES={'kraft':'纸袋','pe':'塑料薄膜袋','bubble':'气泡信封袋'}
OUTCOMES={'grasp_fail':'抓取失败','drop':'滑脱','sim_fail':'仿真失败','success':'成功','incomplete':'未完成','over_squeeze':'挤压力超限','compute_limit':'计算预算耗尽'}


def label(raw,output,title,outcome,detail=""):
    info=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,r_frame_rate','-of','json',str(raw)]))['streams'][0]
    width,height=info['width'],info['height'];fps=info['r_frame_rate']
    fontpath=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    if not fontpath.is_file():raise FileNotFoundError('Chinese category captions require NotoSansCJK-Regular.ttc')
    font=ImageFont.truetype(str(fontpath),28)
    decoder=subprocess.Popen(['ffmpeg','-v','error','-i',str(raw),'-f','rawvideo','-pix_fmt','rgb24','-'],stdout=subprocess.PIPE)
    encoder=subprocess.Popen(['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height+56}',
                              '-r',fps,'-i','-','-an','-c:v','libx264','-crf','21','-pix_fmt','yuv420p',str(output)],stdin=subprocess.PIPE)
    frames=0
    while True:
        data=decoder.stdout.read(width*height*3)
        if not data:break
        if len(data)!=width*height*3:raise RuntimeError('truncated input video frame')
        canvas=Image.new('RGB',(width,height+56),(20,25,32))
        canvas.paste(Image.frombytes('RGB',(width,height),data),(0,56))
        draw=ImageDraw.Draw(canvas)
        draw.text((20,9),f'{title} · 实际结果：{OUTCOMES.get(outcome,outcome)}{detail}',font=font,
                  fill=(90,225,160) if outcome=='success' else (255,180,110))
        encoder.stdin.write(canvas.tobytes());frames+=1
    encoder.stdin.close();decoder.stdout.close()
    if decoder.wait() or encoder.wait():raise RuntimeError('video decode/encode failed')
    subprocess.run(['ffmpeg','-v','error','-i',str(output),'-f','null','-'],check=True)
    return dict(path=str(output.relative_to(ROOT)),frames=frames,width=width,height=height+56,fps=fps,full_decode=True)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run',type=Path,required=True);args=p.parse_args()
    run=args.run.resolve();(run/'video').mkdir(exist_ok=True)
    trials=json.loads((run/'trial_summary.json').read_text())
    capacity=json.loads((run/'capacity_summary.json').read_text()) if (run/'capacity_summary.json').exists() else {}
    capacity_trial=capacity.get('selected_trial',{}).get('index')
    selected={}
    for trial in trials:
        path=ROOT/trial['path'];result=json.loads((path/'summary.json').read_text())
        profile=result['config']['bag_profile']
        score=(int(not result['stages']['close'].get('missing',False)),int(trial['index']==capacity_trial),int('gpu_timing' in result),result['sim_time_s'])
        if profile not in selected or score>selected[profile][2]:selected[profile]=(path,result,score)
    summary=[]
    for profile,(path,result,_) in selected.items():
        raw=run/'video'/f'{profile}-raw.mp4';final=run/'video'/f'{profile}.mp4'
        if not raw.exists():
            subprocess.run([sys.executable,'-B',str(ROOT/'test/bag_model/render_physical_parcel_video.py'),
                           str(path/'trial'),str(raw),'--ffmpeg','/usr/bin/ffmpeg'],cwd=ROOT,check=True)
        if final.exists():raise FileExistsError(final)
        row=label(raw,final,TITLES[profile],result['outcome'],
                  '（已记录抓取失败）' if result.get('grasp_failure') else '')
        row.update(profile=profile,outcome=result['outcome'],source_trial=str(path.relative_to(ROOT)))
        summary.append(row);(run/'video/verification.json').write_text(json.dumps(summary,indent=2))


if __name__=='__main__':main()
