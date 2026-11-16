"""Collect saved sampling, trial, resident GPU and capacity evidence."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import platform
import numpy as np

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run',type=Path,required=True);args=p.parse_args()
    run=args.run.resolve();groups=json.loads((run/'preparation/groups.json').read_text())
    sampling=defaultdict(Counter);rejections=Counter()
    for g in groups:
        for s in g['surface_samples']:
            sampling[s['region']]['sampled']+=1;sampling[s['region']]['accepted']+=int(s['accepted'])
            if not s['accepted']:rejections[s['reason'].split(':')[0]]+=1
    trials=[];engines=defaultdict(list)
    for entry in json.loads((run/'trial_summary.json').read_text()):
        path=ROOT/entry['path'];r=json.loads((path/'summary.json').read_text())
        trace=json.loads((path/'trial/trace.json').read_text())
        lift=max(row['com'][2]-r['initial_com'][2] for row in trace)
        suspended=[row for row in trace if row['com'][2]-r['initial_com'][2]>=.05 and row['support_N']<.05]
        stages=sorted(set(row['phase'] for row in trace))
        selection=json.loads((path/'selection.json').read_text())
        point=np.asarray(selection['sample']['sample']['point_m'])
        approach=[x for x in trace if x['phase']=='approach']
        contact=[x for x in approach if max(x['contact_magnitude_each_N'])>.05]
        actual=dict(sampled_point_m=point.tolist(),first_contact_tip_mid_m=contact[0]['tip_mid'] if contact else None,
                    first_contact_time_s=contact[0]['t'] if contact else None,
                    closest_tip_distance_to_initial_sample_m=min((float(np.linalg.norm(np.asarray(x['tip_mid'])-point)) for x in approach),default=None))
        row=dict(**entry,actual_execution=actual,profile=r['config']['bag_profile'],shape=r['config']['payload_shape'],mode=r['config']['mode'],
                 reason=r['reason'],motion_completed=r['motion_completed'],stages=stages,
                 grasp_reached='close' in stages,
                 completed_protocol_attempt=('approach' in stages and r['outcome'] not in ('compute_limit','incomplete','smoke_only')),
                 success=r['outcome']=='success',peak_lift_m=lift,
                 suspended_samples=len(suspended),peak_penetration_mm=r['peaks']['penetration_mm'],
                 exposed_samples=sum(x['payload_geometry']['outside_corners']>0 for x in trace),
                 peak_geometry_depth_m=max(x['payload_geometry']['geometric_depth_upper_m'] for x in trace),
                 timings=r.get('benchmark_timings'),gpu_timing=r.get('gpu_timing'))
        trials.append(row);engines[entry['engine']].append(row)
    rates={}
    for engine,rows in engines.items():
        wall=sum(row['wall_s'] for row in rows)
        rates[engine]=dict(trials=len(rows),wall_s=wall,trials_per_s=len(rows)/wall,
              grasp_attempts=sum(row['grasp_reached'] for row in rows),
              grasp_attempts_per_s=sum(row['grasp_reached'] for row in rows)/wall,
              completed_protocol_attempts=sum(row['completed_protocol_attempt'] for row in rows),
              completed_protocol_attempts_per_s=sum(row['completed_protocol_attempt'] for row in rows)/wall,
              successful_transfers=sum(row['success'] for row in rows),
              success_rate=sum(row['success'] for row in rows)/len(rows),
              outcomes=dict(Counter(row['outcome'] for row in rows)))
    gpu=[];loaded=[]
    for path in sorted((run/'gpu').glob('*')) if (run/'gpu').exists() else []:
        if (path/'loaded.json').exists():loaded.append(dict(path=str(path.relative_to(ROOT)),**json.loads((path/'loaded.json').read_text())))
        if not (path/'result.json').exists():continue
        r=json.loads((path/'result.json').read_text())
        if r.get('status')!='completed':continue
        samples=r['samples']
        gpu.append(dict(path=str(path.relative_to(ROOT)),worlds=r['worlds'],phase=r['phase'],profile=r['profile'],
             shape=r['model']['payload_shape'],mode=r['model']['mode'],loaded=r['source_stage']['loaded'],
             load_bearing=r['source_stage'].get('load_bearing',False),
             contact_load_preserved_at_window_end=all(s.get('loaded_window_eligible',False) for s in samples),
             final_finger_load_fraction_min=min(s.get('finger_load_fraction_final',0) for s in samples),
             eps=r['median_environment_steps_per_s'],
             batch_step_ms=float(np.median([s['batch_step_ms'] for s in samples])),
             per_environment_realtime=float(np.median([s['per_environment_realtime'] for s in samples])),
             device_memory_MiB=max(s['device_memory_MiB'] for s in samples),
             mempool_peak_MiB=max(s['mempool_allocated_peak_MiB'] for s in samples),
             peak_contacts=max(s['total_contacts'] for s in samples),
             peak_iterations=max(s['peak_solver_iterations'] for s in samples),
             overflow_worlds=max(s['overflow_worlds'] for s in samples),nonfinite_worlds=max(s['nonfinite_worlds'] for s in samples),
             finite_zero_overflow=all(s.get('finite_zero_overflow',s['numerically_valid']) for s in samples),
             numerically_valid=r['numerically_valid'],physically_valid=r['physically_valid'],
             upload_reset_s=r['upload_reset_s'],capture_compile_s=r['warmup_capture_compile_s'],
             reset_s=[s['reset_s'] for s in samples],simulated_seconds=samples[0]['actual_gpu_time_range_s'],
             repeats=len(samples)))
    mixed=[];mixed_loaded=[]
    for path in sorted((run/'mixed').glob('*')) if (run/'mixed').exists() else []:
        if (path/'loaded.json').exists():mixed_loaded.append(json.loads((path/'loaded.json').read_text()))
        if not (path/'result.json').exists():continue
        r=json.loads((path/'result.json').read_text());samples=r['samples']
        mixed.append(dict(path=str(path.relative_to(ROOT)),groups=r['groups'],worlds=r['worlds'],
             worlds_per_group=r['worlds_per_group'],eps=r['median_environment_steps_per_s'],
             geometry_checked_worlds_per_group=r.get('geometry_checked_worlds_per_group',r['worlds_per_group']),
             geometry_all_worlds_checked=r.get('geometry_all_worlds_checked',True),
             batch_step_ms=float(np.median([v['batch_step_ms'] for v in samples])),
             per_environment_realtime=float(np.median([v['per_environment_realtime'] for v in samples])),
             device_memory_MiB=max(v['device_memory_MiB'] for v in samples),
             mempool_peak_MiB=max(v['mempool_peak_MiB'] for v in samples),
             peak_contacts=max(sum(g['contacts'] for g in v['health']) for v in samples),
             peak_iterations=max(g['peak_iterations'] for v in samples for g in v['health']),
             numerically_valid=r['numerically_valid'],physically_valid=r['physically_valid']))
    maximum=lambda rows,key:max((r[key] for r in rows),default=0)
    summary=dict(machine=platform.node(),compile_geometry_material_groups=len(groups),
       compile_cpu_initialization_s=sum(g['compile_initialization_s'] for g in groups),
       preparation=dict(passed=sum(g.get('preparation',{}).get('passed',False) for g in groups),
                        total=len(groups),
                        scene_and_initial_ik_s=sum(s.get('scene_ik_s',0) for g in groups for s in g['surface_samples'])),
       sampling={k:dict(v) for k,v in sampling.items()},rejections=dict(rejections),
       trials=trials,trial_rates=rates,gpu=gpu,mixed_gpu=mixed,
       trial_batches=[dict(batch=i//3,indices=[r['index'] for r in trials if i<=r['index']<i+3],
                      wall_s=sum(r['wall_s'] for r in trials if i<=r['index']<i+3),
                      successes=sum(r['success'] for r in trials if i<=r['index']<i+3)) for i in (0,3,6)],
       capacity=dict(max_loaded_worlds=maximum(loaded,'worlds'),
          max_finite_zero_overflow_worlds=maximum([r for r in gpu if r.get('finite_zero_overflow')],'worlds'),
          max_numerically_valid_worlds=maximum([r for r in gpu if r['numerically_valid']],'worlds'),
          max_physically_valid_worlds=maximum([r for r in gpu if r['physically_valid']],'worlds'),
          contact_loaded_max_physically_valid_worlds=maximum([r for r in gpu if r['physically_valid'] and r['contact_load_preserved_at_window_end']],'worlds'),
          loaded_grasp_max_physically_valid_worlds=maximum([r for r in gpu if r['physically_valid'] and r['load_bearing']],'worlds') if any(r['load_bearing'] for r in gpu) else None,
          bearing_hold_measured=any(r['phase']=='lift_hold' and r['load_bearing'] for r in gpu),
          mixed=dict(max_complete_loaded_worlds=maximum([r for r in mixed_loaded if r['loaded_groups']==len(trials)],'worlds'),
                     max_numerically_valid_worlds=maximum([r for r in mixed if r['numerically_valid']],'worlds'),
                     max_physically_valid_worlds=maximum([r for r in mixed if r['physically_valid']],'worlds'))))
    summary['capacity']['measured']=bool(gpu or mixed)
    if (run/'load-capacity-summary.json').exists():
        loading=json.loads((run/'load-capacity-summary.json').read_text())
        summary['load_only_capacity']=loading
        if 'single' in loading:summary['capacity']['max_loaded_worlds']=loading['single']['max_loaded_worlds']
        if 'mixed' in loading:summary['capacity']['mixed']['max_complete_loaded_worlds']=loading['mixed']['max_loaded_worlds']
    summary['status']='halted_before_capacity' if (run/'halt.json').exists() else 'recorded'
    summary['distinct_executed_parameter_groups']=len({row['group'] for row in trials})
    summary['cpu_trial_timings_s']={key:sum((row['timings'] or {}).get(key,0) for row in trials)
                                    for key in ['cpu_command_s','cpu_ik_s','cpu_geometry_check_s','cpu_feedback_s']}
    (run/'summary.json').write_text(json.dumps(summary,indent=2))
    lines=['# 完整YAM随机袋 GPU 运行汇总','',f'机器：{platform.node()}。各GPU档3次同步计时，100步≈10ms。', '',
           '| 袋型 / 内容物 | 阶段 | 夹持接触/承重 | 环境数 | 环境步/s | 整批步/ms | 单环境实时率 | 显存/MiB | 数值/物理有效 |',
           '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for r in gpu:
        lines.append(f"| {r['profile']} / {r['mode']}-{r['shape']} | {r['phase']} | {r['loaded']}/{r['load_bearing']} | {r['worlds']} | {r['eps']:.1f} | {r['batch_step_ms']:.3f} | {r['per_environment_realtime']:.4f} | {r['device_memory_MiB']} | {r['numerically_valid']}/{r['physically_valid']} |")
    lines+=['','| 混合组数 | 环境数 | 环境步/s | 整批步/ms | 单环境实时率 | 显存/MiB | 数值/物理有效 |',
            '| ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    for r in mixed:
        lines.append(f"| {r['groups']} | {r['worlds']} | {r['eps']:.1f} | {r['batch_step_ms']:.3f} | {r['per_environment_realtime']:.4f} | {r['device_memory_MiB']} | {r['numerically_valid']}/{r['physically_valid']} |")
    lines+=['','数值有效要求有限状态、零overflow及求解未触上限；物理有效另要求几何深度≤1mm、无表面采样点外露、原生穿透≤1mm、求解未到上限。阶段测速固定真实采样状态和驱动指令，不是完整闭环任务吞吐。','',
            '| 试验 | 袋型 / 内容物 | 引擎 | 结果 | 仿真/s | 墙钟/s | 抬升/mm |', '| --- | --- | --- | --- | ---: | ---: | ---: |']
    for r in trials:
        lines.append(f"| {r['index']} | {r['profile']} / {r['mode']}-{r['shape']} | {r['engine']} | {r['outcome']} | {r['sim_s']:.3f} | {r['wall_s']:.2f} | {r['peak_lift_m']*1000:.3f} |")
    (run/'summary.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(capacity=summary['capacity'],trial_rates=rates,gpu_measurements=len(gpu)),indent=2))


if __name__=='__main__':main()
