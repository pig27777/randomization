"""F12：空袋端（u = 0.4）顶面抓，F7 同一网格与流程，捏取规则换成 FilmGatherPinch（聚拢条件 + 只计拉离/水平承载）。

网格：w 40/60/90 mm × d 6/12 mm × F 10/20 N × bag_pinch_coeff 0.5/1/2 × 3 种子 = 108 次；
另补 F7 同样的低承载对照（系数 0.1、F 10 N、w60/d12，3 种子）。硬盒，0.8 ms，第 3 轮基准袋，缓存静置状态。
--u 0 可跑袋中部作对照。正式运行须排队：flock /tmp/claude-gpu.lock <命令>。
"""
import argparse
import itertools
import json
import math
import socket
import time
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
import sys

BAG = Path(__file__).resolve().parents[2] / "bag_sim"
sys.path.insert(0, str(BAG))

from round3_flex import BASE, content_kappa
from film_grasp import evaluate_film
from film_gather_pinch import FilmGatherPinch, GATHER_MIN

OUT = None


def one(job):
    params, coefficient, seed = job
    t0 = time.perf_counter()
    r = evaluate_film(content_kappa('box_loose'), BASE, params, coefficient, seed, cache_dir=OUT / 'cache',
                      binding_cls=FilmGatherPinch)
    return dict(r, total_wall_s=time.perf_counter() - t0, gather_min_mm=1e3 * GATHER_MIN)


def main():
    global OUT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--u', type=float, default=.4)
    ap.add_argument('--procs', type=int, default=28)
    ap.add_argument('--limit', type=int, default=0, help='只跑前 N 个（入口检查）')
    a = ap.parse_args()
    OUT = a.output
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f'{a.name}.jsonl'
    if path.exists(): raise FileExistsError(path)
    jobs = [([a.u, 0., math.pi / 2, w, d, F, .1, 2.], c, s)
            for w, d, F, c, s in itertools.product([.040, .060, .090], [.006, .012], [10., 20.], [.5, 1., 2.], range(3))]
    jobs += [([a.u, 0., math.pi / 2, .060, .012, 10., .1, 2.], .1, s) for s in range(3)]
    if a.limit: jobs = jobs[:a.limit]
    with Pool(a.procs, initializer=_init, initargs=(OUT,)) as pool:
        for r in pool.imap_unordered(one, jobs, chunksize=1):
            r.update(hostname=socket.gethostname(), finished_at=datetime.now().astimezone().isoformat())
            with path.open('a') as f:
                f.write(json.dumps(r, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x)) + '\n')
            print(json.dumps({k: r.get(k) for k in ('params', 'bag_pinch_coeff', 'seed', 'outcome', 'reason',
                                                    'gather_mm', 'bag_lift_mm', 'content_in_bag', 'penetration_mm')}),
                  flush=True)


def _init(out):
    global OUT
    OUT = out


if __name__ == '__main__':
    main()
