"""Read-only analysis of diagnose_r3_v13 contact/closing traces; no physics runs."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path


def read(path):
    def reject(value):raise ValueError(value)
    return json.loads(path.read_text(),parse_constant=reject)


def force(row):
    return sum(row['legacy_normal_y_N'].values())/2


def contact_key(contact):
    return tuple(contact['geom'])+tuple(contact['pos_m'])


def pad_contacts(row):
    return [c for c in row['contacts'] if any(n and n.startswith(('pA_','pB_')) for n in c['names'])
            and c['efc_address']>=0]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('directory',type=Path);a=ap.parse_args();root=a.directory
    runs={n:read(root/f'{n}.json') for n in ('cpu','gpu')};summary=read(root/'summary.json')
    comparisons=[];decoder_error=0.;matched=0
    for row in runs['gpu']['rows']:
        native={contact_key(c):c for c in row['native_contacts']}
        if len(native)!=len(row['native_contacts']):raise ValueError('ambiguous native contact keys')
        for c in row['contacts']:
            n=native[contact_key(c)]
            decoder_error=max(decoder_error,max(abs(x-y) for x,y in zip(c['contact_force_local_N'],n['native_force_local_N'])))
            matched+=1
    for cpu,gpu in zip(runs['cpu']['rows'],runs['gpu']['rows']):
        if abs(cpu['t']-gpu['t'])>1e-9:raise ValueError('diagnostic sample times do not align')
        comparisons.append(dict(t=cpu['t'],cpu_force_N=force(cpu),gpu_force_N=force(gpu),
            cpu_pad_contacts=len(pad_contacts(cpu)),gpu_pad_contacts=len(pad_contacts(gpu)),
            cpu_gap_um=cpu['pad_gap_y_m']*1e6,gpu_gap_um=gpu['pad_gap_y_m']*1e6,
            cpu_frozen=cpu['closer']['frozen'],gpu_frozen=gpu['closer']['frozen'],
            cpu_squeeze_um=cpu['closer']['squeeze_m']*1e6,gpu_squeeze_um=gpu['closer']['squeeze_m']*1e6,
            cpu_filtered_min_N=min(cpu['closer']['filtered_force_N'].values()),
            gpu_filtered_min_N=min(gpu['closer']['filtered_force_N'].values())))
    first=next(r for r in comparisons if r['cpu_pad_contacts']!=r['gpu_pad_contacts'])
    selected={}
    for name,run in runs.items():
        r=min(run['rows'],key=lambda x:abs(x['t']-1.8));pads=pad_contacts(r)
        squeeze=run['events'][0]['squeeze_m'];gain=min(.001,max(squeeze,1e-8)/(.05*20.))
        selected[name]=dict(t=r['t'],pad_positions_m=r['pad_positions_m'],force_N=force(r),
            pad_gap_m=r['pad_gap_y_m'],pad_indentation_m=r['pad_indentation_m'],
            object_compression_m=r['object_compression_m'],contact_count=len(pads),
            cone=r['cone_cpu_name'],per_contact_normal_range_N=[min(c['contact_force_local_N'][0] for c in pads),
                max(c['contact_force_local_N'][0] for c in pads)],
            pairs=[dict(geoms=list(pair),count=count) for pair,count in sorted(Counter(tuple(c['names']) for c in pads).items())],
            hold_admittance_m_per_N_s=gain)
    analysis=dict(exploration=True,physics_rerun=False,formal_acceptance=False,
        method='Match exported and native GPU contact forces by exact geometry pair and position; align CPU/GPU trace times.',
        matched_gpu_contacts=matched,max_export_vs_native_force_torque_difference=decoder_error,
        first_sample_with_pad_contact_count_difference=first,
        closing=[r for r in comparisons if .3792-1e-10<=r['t']<=.3824+1e-10],
        selected=selected,protocol_result={n:r['result'] for n,r in runs.items()},
        pre_means={n:summary['summary'][n]['pre_means'] for n in runs},
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),)},
        input_sha256={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in ('cpu.json','gpu.json','summary.json')})
    with (root/'analysis.json').open('x') as f:json.dump(analysis,f,indent=1,allow_nan=False)
    with (root/'comparison.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(comparisons[0]));writer.writeheader();writer.writerows(comparisons)
    print(json.dumps({k:analysis[k] for k in ('matched_gpu_contacts','max_export_vs_native_force_torque_difference',
        'first_sample_with_pad_contact_count_difference')},indent=2))


if __name__=='__main__':main()
