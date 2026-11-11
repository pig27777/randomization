"""F13：flex 薄膜袋 + v1.3 分段软内容物合在一起的 CPU 单步成本（单线程、单进程）。

配置（--case）：
  hard08 / hard04       松装牛皮纸 flex 袋（第 3 轮基准）+ 硬盒，0.8 / 0.4 ms
  bendable04            同一只袋 + v1.3 易弯锚点（240×160×50 mm，6×3×2 段），0.4 ms
  compressible04        v1.3 易压锚点（350×250×120 mm）装不进同一只袋，袋按 fill 0.75 放大，0.4 ms
  bendable_only04 / compressible_only04   不装袋，内容物单独在桌上静置，作为分项对照
袋内配置跑一次 F3 中心侧夹（evaluate，与 F3 输入 centre 种子 0 相同），记录单步耗时、bushing 力耗时、
约束行数、接触数、迭代数。内容物代码全部从 v1.3 冻结归档导入，只读；bushing 计时靠包一层计时器，不改其计算。
正式运行须排队：flock /tmp/claude-gpu.lock <命令>；OMP_NUM_THREADS=1。
"""
import argparse
import json
import socket
import time
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
import sys

BAG = Path(__file__).resolve().parents[2] / "bag_sim"
sys.path.insert(0, str(BAG))

import mujoco
import numpy as np

from loose_protocols import THRESH, content_kappa, evaluate
from round3_flex import BASE, f3_inputs
import bushing
from mj_adapter import OMEGA_MAX, Sim, scene
from model import ANCHORS, build, load_table

TIMER = dict(bushing_s=0., calls=0)
_apply = bushing.Bushings.apply


def timed_apply(self, m, d):
    t0 = time.perf_counter()
    _apply(self, m, d)
    TIMER['bushing_s'] += time.perf_counter() - t0
    TIMER['calls'] += 1


bushing.Bushings.apply = timed_apply

CASES = {
    'hard08': ('box_loose', BASE, .0008), 'hard04': ('box_loose', BASE, .0004),
    'bendable04': ('bendable', BASE, .0004),
    'compressible04': ('compressible', replace(BASE, size=None, fill=.75), .0004),
}


def in_bag(case):
    content, bag, dt = CASES[case]
    name, params, seed = next(iter(f3_inputs()))          # centre, 种子 0
    cls = 'hard' if content.startswith('box') else content
    spec = dict(kappa=content_kappa(content).to_dict(), cls=cls, bag=asdict(bag), dt=dt, cache_dir=None)
    t0 = time.perf_counter()
    r = evaluate(spec, params, seed)
    r.update(case=case, input=f'{name}:{seed}', dt=dt, bag=asdict(bag), total_wall_s=time.perf_counter() - t0)
    return r


def alone(case, seconds):
    content = case.split('_')[0]
    desc = build(ANCHORS[content], omega_max=OMEGA_MAX, table=load_table())
    built = scene(desc, (0., 0., ANCHORS[content].size[2] / 2 + .001), (), dt=.0004)
    sim = Sim(built)
    n = round(seconds / sim.m.opt.timestep)
    nefc = ncon = iters = 0
    t0 = time.perf_counter()
    for _ in range(n):
        sim.step()
        nefc = max(nefc, sim.d.nefc); ncon = max(ncon, sim.d.ncon); iters += int(sim.d.solver_niter.max())
    wall = time.perf_counter() - t0
    return dict(case=case, dt=float(sim.m.opt.timestep), steps=n, nv=int(sim.m.nv), step_ms=1e3 * wall / n,
                max_nefc=int(nefc), max_ncon=int(ncon), mean_iterations=iters / n, sim_t=n * sim.m.opt.timestep)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--case', required=True, choices=[*CASES, 'bendable_only04', 'compressible_only04'])
    ap.add_argument('--seconds', type=float, default=2., help='单独内容物静置的仿真时长')
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    path = a.output / f'{a.case}.json'
    if path.exists(): raise FileExistsError(path)
    r = in_bag(a.case) if a.case in CASES else alone(a.case, a.seconds)
    steps = max(1, TIMER['calls'])
    r.update(bushing_calls=TIMER['calls'], bushing_ms_per_step=1e3 * TIMER['bushing_s'] / steps if TIMER['calls'] else 0.,
             hostname=socket.gethostname(), finished_at=datetime.now().astimezone().isoformat())
    path.write_text(json.dumps(r, indent=1, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x)))
    print(json.dumps({k: r.get(k) for k in ('case', 'dt', 'outcome', 'fail', 'step_ms', 'bushing_ms_per_step',
                                            'max_nefc', 'max_ncon', 'mean_iterations', 'sim_t', 'nv',
                                            'penetration_mm', 'total_wall_s')}))


if __name__ == '__main__':
    main()
