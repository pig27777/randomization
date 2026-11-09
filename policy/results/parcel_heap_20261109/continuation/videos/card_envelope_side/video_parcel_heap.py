"""Record unchanged archived trials densely, verify them, then export labelled videos."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

os.environ.setdefault('MUJOCO_GL','egl')
import mujoco
import numpy as np

from probe_feedback import digest,save

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'policy/results/parcel_heap_20261109/continuation'
OUT=BASE/'videos'
FPS=30
FONT='/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'
TRIALS={
    'book':('overhead','grasp_book_top_center_0.0_replay1','book',
            '书本软包 | 低堆顶 / 上方中心夹取 / dt=0.2 ms',
            '单次成功：抬升 102.8 mm，搬运 99.0 mm；步长稳定性未通过'),
    'clothes':('overhead','grasp_clothes_top_center_0.0','clothes',
               '衣物软包 | 低堆顶 / 上方中心夹取 / dt=0.2 ms',
               '未达标：抬升 54.7 mm，搬运 99.6 mm；要求抬升至少 100 mm'),
    'book_small_step':('overhead_quarter','grasp_book_top_center_0.0','book',
                       '书本软包 | 低堆顶 / 上方中心夹取 / dt=0.1 ms',
                       '物理超限停止：穿透 1.132 mm；不作为真实抓取失败'),
    'paper_envelope':('overhead','grasp_paper_envelope_top_center_0.0','paper_envelope',
                      '装纸薄信封 | 低堆顶 / 上方中心夹取 / dt=0.05 ms',
                      '物理超限停止：穿透 0.151 mm；接触膜厚 0.15 mm'),
    'card_envelope_side':('side_gap','grasp_gap_side_0.0','card_envelope',
                          '装薄硬片信封 | 高堆空隙 / 露边侧夹 / dt=0.05 ms',
                          '未建立稳定双指夹持：合拢 3 s 超时，未进入抬升'),
}
PHASES=dict(settle='静置',approach='接近',close='合拢',lift='抬升',carry='搬运',hold='保持')


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def archived_trial(name):
    folder,stage,case,*_=TRIALS[name]
    run=BASE/folder
    path=run/(stage+'.jsonl')
    row=next(json.loads(line) for line in path.read_text().splitlines() if json.loads(line)['case']==case)
    metadata=json.loads((run/(stage+'_metadata.json')).read_text())
    snapshot=run/'driver_source'/stage
    for file,expected in metadata['files'].items():
        if digest(snapshot/Path(file).name)!=expected:
            raise RuntimeError(f'archived source mismatch: {file}')
    if digest(run/'settings.json')!=metadata['settings_sha256']:
        raise RuntimeError('archived settings changed')
    heap=load_module('video_archived_heap',snapshot/'parcel_heap.py')
    # Historical driver resolves ROOT from __file__; use its original repository root.
    heap.ROOT=ROOT
    heap.install_backend(run)
    yam=load_module('yam_probe_robot',snapshot/'yam_probe_robot.py')
    if yam.asset_fingerprint()!=json.loads((run/'yam_asset_sha256.json').read_text()):
        raise RuntimeError('YAM assets changed')
    return heap,row,path


def record(name):
    out=OUT/name;out.mkdir(parents=True,exist_ok=True)
    if (out/'states.npz').exists():
        raise FileExistsError(out/'states.npz')
    heap,original,path=archived_trial(name)
    shutil.copy2(__file__,out/'video_parcel_heap.py')
    step=heap.HeapSim.step
    times,positions=[],[]
    next_sample=0.
    last_sim=None

    def capture(sim):
        times.append(float(sim.d.time));positions.append(sim.d.qpos.copy())

    def recorded_step(sim):
        nonlocal next_sample,last_sim
        last_sim=sim
        if not times:
            capture(sim);next_sample=1/FPS
        result=step(sim)
        if sim.d.time+1e-12>=next_sample:
            capture(sim);next_sample+=1/FPS
        return result

    heap.HeapSim.step=recorded_step
    job={k:original[k] for k in ('case','layout','action','jitter','preflight')}
    replay=heap.evaluate(job)
    if times[-1]!=float(last_sim.d.time):
        capture(last_sim)
    save(out/'replay.json',replay)
    np.savez_compressed(out/'states.npz',times=times,qpos=positions)
    phase_equal=[r['phase'] for r in replay['keyframes']]==[r['phase'] for r in original['keyframes']]
    qpos_error=(float(np.max(np.abs(np.asarray([r['qpos'] for r in replay['keyframes']])-
                                   np.asarray([r['qpos'] for r in original['keyframes']]))))
                if phase_equal else None)
    displacement_error=float(np.max(np.abs(np.asarray(replay['displacement_mm'])-original['displacement_mm'])))
    peaks_error=max(abs(replay['peaks'][k]-original['peaks'][k]) for k in original['peaks'])
    phase_time_error=(max(abs(a['t']-b['t']) for a,b in zip(replay['keyframes'],original['keyframes']))
                      if phase_equal else None)
    passed=(replay['outcome']==original['outcome'] and phase_equal and qpos_error<=1e-10 and
            phase_time_error<=1e-10 and displacement_error<=1e-6 and peaks_error<=1e-6)
    verification=dict(passed=passed,outcome=replay['outcome'],original=str(path.relative_to(ROOT)),
                      original_sha256=digest(path),recorder_sha256=digest(__file__),
                      driver_sha256=digest(heap.__file__),host=platform.node(),fps=FPS,
                      frames=len(times),qpos_error=qpos_error,phase_time_error_s=phase_time_error,
                      displacement_error_mm=displacement_error,peaks_error_mm=peaks_error,
                      states_sha256=digest(out/'states.npz'),
                      note='Unchanged archived dynamics replay with post-step qpos copies. No keyframe interpolation.')
    save(out/'verification.json',verification)
    print(json.dumps(verification),flush=True)
    if not passed:
        raise RuntimeError('replay differs from original; video export refused')


def ass_time(seconds):
    cs=round(seconds*100)
    return f'{cs//360000}:{cs//6000%60:02}:{cs//100%60:02}.{cs%100:02}'


def subtitles(name,row,times):
    header='''[Script Info]
ScriptType: v4.00+
PlayResX: 960
PlayResY: 1080
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Noto Sans CJK SC,22,&H00FFFFFF,&H00FFFFFF,&H00000000,&H90000000,0,0,0,0,100,100,0,0,1,2,0,7,18,18,12,1
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
'''
    duration=len(times)*2/FPS+2
    title=TRIALS[name][3]
    status=TRIALS[name][4]
    events=[f'Dialogue: 0,0:00:00.00,{ass_time(duration)},Default,,0,0,0,,{{\\pos(18,10)}}{title}\\N{status}',
            f'Dialogue: 0,0:00:00.00,{ass_time(duration)},Default,,0,0,0,,{{\\pos(18,1025)}}YAM 整机仿真重放 | 0.5 倍速 | 未实物标定\\N上：整机视角；下：抓取近景']
    for i,t in enumerate(times):
        phase=next((p['phase'] for p in row['phase_times'] if p['start_s']-1e-9<=t<p['end_s']-1e-9),None)
        label=PHASES.get(phase,'试次结束')
        end=(i+1)*2/FPS if i<len(times)-1 else duration
        if i==len(times)-1:
            label+=' / 末帧暂停'
        events.append(f'Dialogue: 1,{ass_time(i*2/FPS)},{ass_time(end)},Default,,0,0,0,,{{\\pos(18,555)}}仿真 t={t:.3f} s | {label}')
    (OUT/name/'captions.ass').write_text(header+'\n'.join(events)+'\n')
    return duration


def render(name):
    out=OUT/name
    check=json.loads((out/'verification.json').read_text())
    if not check['passed'] or check['states_sha256']!=digest(out/'states.npz'):
        raise RuntimeError('unverified or modified states')
    heap,row,_=archived_trial(name)
    state=np.load(out/'states.npz');times=state['times'];qpos=state['qpos']
    robot=heap.scene(row['case'],row['layout'],row['action'],row['jitter'])
    m=robot.sim.m;data=mujoco.MjData(m)
    # Pure position rendering avoids constraint solving and cannot feed into dynamics.
    options=mujoco.MjvOption()
    options.flags[mujoco.mjtVisFlag.mjVIS_TENDON]=False
    options.sitegroup[:]=0;options.geomgroup[3]=0
    raised=row['settings']['stack_lift']
    overview=mujoco.MjvCamera();overview.lookat[:]=[-.10,.06,.30+raised*.3]
    overview.distance=1.16;overview.azimuth=55;overview.elevation=-30
    close=mujoco.MjvCamera();close.lookat[:]=[-.19,.04,.16+raised]
    close.distance=.64;close.azimuth=55;close.elevation=-40
    subtitles(name,row,times)
    dest=OUT/(name+'.mp4')
    if dest.exists():
        raise FileExistsError(dest)
    command=['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','960x1080','-r',str(FPS),
             '-i','pipe:0','-an','-vf',f'ass={out/"captions.ass"}',
             '-c:v','libx264','-preset','fast','-crf','21','-threads','2','-pix_fmt','yuv420p',
             '-movflags','+faststart',str(dest)]
    encoder=subprocess.Popen(command,stdin=subprocess.PIPE)
    try:
        with mujoco.Renderer(m,540,960,max_geom=20000) as renderer:
            for index,position in enumerate(qpos):
                data.qpos[:]=position;mujoco.mj_fwdPosition(m,data)
                views=[]
                for camera in (overview,close):
                    renderer.update_scene(data,camera=camera,scene_option=options)
                    views.append(renderer.render())
                pixels=np.concatenate(views,axis=0)
                if index in (0,len(qpos)-1):
                    from render_probe import write_png
                    write_png(out/('first.png' if index==0 else 'last.png'),pixels)
                for _ in range(2):
                    encoder.stdin.write(pixels.tobytes())
            for _ in range(FPS*2):
                encoder.stdin.write(pixels.tobytes())
    finally:
        encoder.stdin.close()
        code=encoder.wait()
    if code:
        raise RuntimeError(f'ffmpeg exited {code}')
    save(out/'video.json',dict(file=dest.name,sha256=digest(dest),source_states_sha256=digest(out/'states.npz'),
                             video_frames=2*len(times)+2*FPS,fps=FPS,speed=.5,
                             terminal_hold_s=2,no_state_interpolation=True))
    print(dest,flush=True)


def compose():
    groups=[('book_clothes_comparison','book','clothes'),
            ('book_timestep_comparison','book','book_small_step'),
            ('thin_envelope_comparison','paper_envelope','card_envelope_side')]
    for title,a,b in groups:
        dest=OUT/(title+'.mp4')
        if dest.exists():raise FileExistsError(dest)
        subprocess.run(['ffmpeg','-v','error','-i',str(OUT/(a+'.mp4')),'-i',str(OUT/(b+'.mp4')),
                        '-filter_complex','[0:v]tpad=stop_mode=clone:stop_duration=30[a];'
                                          '[1:v]tpad=stop_mode=clone:stop_duration=30[b];'
                                          '[a][b]hstack=inputs=2[v]',
                        '-map','[v]','-t',str(max(json.loads((OUT/n/'video.json').read_text())['video_frames']/FPS for n in (a,b))),
                        '-an','-c:v','libx264','-preset','fast','-crf','21','-threads','2',
                        '-pix_fmt','yuv420p','-movflags','+faststart',str(dest)],check=True)
        print(dest,flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=('record','render','compose'))
    parser.add_argument('--trial',choices=TRIALS)
    args=parser.parse_args();OUT.mkdir(exist_ok=True)
    if args.stage=='compose':compose()
    elif not args.trial:parser.error('--trial is required')
    elif args.stage=='record':record(args.trial)
    else:render(args.trial)


if __name__=='__main__':main()
