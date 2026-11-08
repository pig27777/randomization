"""Run a job file (one loose_protocols.py argument line each) on N parallel processes.
Usage: run_jobs.py [N] [jobs.txt]"""
import shlex, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
HERE = Path(__file__).resolve().parent
PY = "/home/ubuntu/Dev/Envs/mjwarp/bin/python"
jobs = [l for l in (HERE / (sys.argv[2] if len(sys.argv) > 2 else "jobs.txt")).read_text().splitlines() if l.strip()]
log = open(HERE / "logs" / "batch.log", "a")

def one(line):
    r = subprocess.run([PY, "-B", "loose_protocols.py", *shlex.split(line)], cwd=HERE.parents[1],
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    log.write(f"{r.returncode} {line}\n" + (r.stderr[-2000:] if r.returncode else ""))
    log.flush()

with ThreadPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 28) as ex:
    list(ex.map(one, jobs))
