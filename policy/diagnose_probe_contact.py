"""Replay only test episodes with unexpected non-finger contact; preserve the test set."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import json

from probe_feedback import DEFAULT_RUN,load_backend,read_rows,save


def diagnose(item):
    spec,reference=item
    from yam_probe_robot import Yam
    from probe_env import evaluate
    original=Yam.observe
    events=[]
    def observed(robot):
        result=original(robot)
        if robot.steps % 25==0:
            for c in robot.sim.d.contact:
                g1,g2=c.geom
                if c.dist < 0 and c.efc_address >= 0 and (
                    (robot.nonfinger_mask[g1] and (g2 in robot.sim.obj_geoms or g2==robot.sim.table_geom)) or
                    (robot.nonfinger_mask[g2] and (g1 in robot.sim.obj_geoms or g1==robot.sim.table_geom))):
                    events.append(dict(t=float(robot.sim.d.time),
                        geoms=[robot.sim.m.geom(int(g)).name for g in (g1,g2)],penetration_mm=-float(c.dist)*1000))
        return result
    Yam.observe=observed
    result=evaluate(spec['kappa'],probe=reference['arm']!='direct',
                    selector=lambda geometry,feedback:reference['selected_force_N'],seed=reference['seed'])
    Yam.observe=original
    phases=result['phases']
    grouped={}
    for event in events:
        phase=next((p['phase'] for p in reversed(phases) if p['t']<=event['t']),'settle')
        key=tuple(event['geoms'])+(phase,)
        row=grouped.setdefault(key,dict(geoms=event['geoms'],phase=phase,samples=0,max_penetration_mm=0.,first_t=event['t']))
        row['samples']+=1
        row['max_penetration_mm']=max(row['max_penetration_mm'],event['penetration_mm'])
    fields=('outcome','reason','selected_force_N','max_grip_force_N','sim_s','probe_digest','robot')
    return dict(object_id=reference['object_id'],seed=reference['seed'],arm=reference['arm'],
                replay_equal={name:result[name]==reference[name] for name in fields},
                contacts=sorted(grouped.values(),key=lambda r:-r['max_penetration_mm']))


if __name__=='__main__':
    run=DEFAULT_RUN
    load_backend(run)
    rows=read_rows(run/'test.jsonl')
    objects={o['id']:o for o in json.loads((run/'test_objects.json').read_text())}
    selected=[r for r in rows if r['robot']['nonfinger_contact_samples'] and r['arm']!='probe_feedback']
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('fork')) as pool:
        results=list(pool.map(diagnose,[(objects[r['object_id']],r) for r in selected]))
    save(run/'nonfinger_contact_diagnosis.json',results)
    print(json.dumps(results,indent=2))
