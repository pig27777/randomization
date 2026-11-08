"""Enforce the F6/F7 deadline on this run's existing process trees, including queued work."""
import json
import os
import signal
import time
from datetime import datetime
from pathlib import Path

OUT=Path(__file__).resolve().parent/'outputs/round3_flex'
manifest=OUT/'run_manifest.json'


def process(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().rsplit(') ',1)[1].split()
        return int(fields[1]),fields[19]
    except FileNotFoundError:
        return None


def main():
    data=json.loads(manifest.read_text())
    roots={data[k]:process(data[k]) for k in ['f7_queue_pid','final_behavior_queue_pid','gpu_queue_pid']}
    deadline=datetime.fromisoformat(data['deadline'])
    while True:
        if json.loads(manifest.read_text()).get('f6_f7_complete_at'): return
        remaining=(deadline-datetime.now(deadline.tzinfo)).total_seconds()
        if remaining<=0: break
        time.sleep(min(30.,remaining))
    active={pid for pid,identity in roots.items() if identity is not None and process(pid)==identity}
    tree={int(p.name):process(int(p.name)) for p in Path('/proc').iterdir() if p.name.isdigit()}
    descendants=set(active)
    while True:
        extended=descendants|{pid for pid,info in tree.items() if info is not None and info[0] in descendants}
        if extended==descendants: break
        descendants=extended
    stopped=[]
    for pid in list(active)+sorted(descendants-active):
        if process(pid)==tree.get(pid):
            try:
                os.kill(pid,signal.SIGTERM)
                os.kill(pid,signal.SIGCONT)  # Also terminate a worker paused for timing.
                stopped.append(pid)
            except ProcessLookupError:
                pass
    (OUT/'deadline_stop.json').write_text(json.dumps(dict(deadline=deadline.isoformat(),stopped_pids=stopped))+'\n')


if __name__=='__main__': main()
