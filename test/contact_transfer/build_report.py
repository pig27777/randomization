"""Build the benchmark report and figure from saved data, without running physics."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shlex
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_suite import command

HERE = Path(__file__).resolve().parent


def main():
    results = {p.stem: json.loads(p.read_text()) for p in sorted((HERE / 'results').glob('*.json'))
               if not p.stem.startswith('replay_')}
    # v2 tightens the settled check to include tool velocity; raw traces are unchanged.
    for name, r in results.items():
        s = r['summary']
        r['checks']['settled'] = abs(s['sink_drift_last_half_second_mm']) <= .05 and max(s['max_speed_mm_s'], s.get('tool_speed_mm_s', 0.)) <= 1.
        r['checks_version'] = 2
        r['passed'] = all(r['checks'].values())
        (HERE / 'results' / f'{name}.json').write_text(json.dumps(r, indent=2, allow_nan=False) + '\n')

    def s(name, key): return results[name]['summary'][key]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    panels = [
        ('Rigid plate: support / applied load', ['guided_base_plate', 'guided_iter40_plate', 'guided_dt0p5_plate', 'guided_dt0p25_plate']),
        ('Off-grid finger: support / applied load', ['guided_base_finger_off', 'guided_iter40_finger_off', 'fine_area_scaled_finger_off', 'vertex_only_finger_off']),
        ('Membrane: repeatability and refinement checks', ['wave_soft30_membrane', 'wave_halfstep_membrane', 'wave_iter20_membrane', 'wave_repeat1_membrane', 'wave_repeat2_membrane']),
        ('Direct pressure: mean top displacement (mm)', ['baseline_direct_plate', 'fine_grid_direct_plate', 'mj0p1d20_direct_plate']),
    ]
    for ax, (title, names) in zip(axes.ravel(), panels):
        displacement = 'displacement' in title
        for name in names:
            if name not in results: continue
            r = results[name]
            ax.plot([v['t'] for v in r['trace']],
                    [v['sink_mm'] if displacement else v['support_N'] / r['load_N'] for v in r['trace']], label=name)
            if r['failure'] and r['trace']:
                last = r['trace'][-1]
                ax.plot(last['t'], last['sink_mm'] if displacement else last['support_N']/r['load_N'], 'kx', ms=8)
        if not displacement: ax.axhline(1, color='black', ls='--', lw=1, alpha=.6)
        ax.set(title=title, xlabel='Simulation time (s)', ylabel='mm' if displacement else 'Force ratio')
        ax.grid(alpha=.2); ax.legend(fontsize=7, loc='best')
    fig.savefig(HERE / 'comparison.png', dpi=170)
    plt.close(fig)

    lines = [
        '# 第一项：接触传力基准报告', '',
        '会话日期：2026-11-02（Asia/Taipei）。所有代码与产物均在 `test/contact_transfer/`。业务模型、原始资产和交接文档未修改。', '',
        '## 决策', '',
        '**当前 Newton 1.6.1 配置不通过完整场景的平台验收，暂不进入内容物本构扩展和装袋整合。** 这是本机版本、指定模型与下列测试范围内的结论，不是对整个 Newton 项目的能力否定。', '',
        '四面体直接受压能正确承载；刚体全表面接触下，平均反力有时正确，但局部残差和持续振动不通过。仅顶点接触会漏掉网格间隙内的小工具。膜—四面体在固定配置下3次通过，但步长减半、增加迭代后的复核失稳，因此尚未通过数值收敛验收。', '',
        'MuJoCo 3.17 备选的同网格、同 Neo-Hookean 体积力实现，压板和偏置小手指可以通过；节点正上方手指出现倒置，膜片测试也未完成，所以也没有获得“完整场景已验证”的结论。MuJoCo 的 GPU 批量路径本轮没有重新验证。', '',
        '## 方法与验收', '',
        '软块 300×250×50 mm、0.3 kg，Lamé μ=1500 Pa、λ=1000 Pa；基准网格 12×10×4 单元，715 个节点、2400 个四面体。细网格 24×20×8 单元，4725 个节点、19200 个四面体。底层固定，关闭重力，0.2 秒平滑施加已知载荷；压板 19.62 N，小手指 2 N。正式刚体对照使用无驱动、无轴向阻尼的竖直移动副，避免自由工具翻倒混淆传力。', '',
        '独立 float64 能量导数计算材料节点力和底部支撑反力；刚体接触力用当前接触记录和求解器本构重算；几何穿透通过 deformed top triangles 的竖直射线检查。直接压力用三角形形函数积分，保持合力和力矩，**不假定均布压力与刚性压头产生相同位移**。', '',
        '最后 0.5 s：支撑反力误差≤5%，压头反力误差≤5%，四面体活动节点残差范数之和≤载荷5%；下陷漂移≤0.05 mm，软体和工具速度≤1 mm/s；采样中无倒置、界面几何穿透≤0.05 mm。每50 ms采样；步长、迭代和网格对照另行判断。提前失败的运行没有稳态，其末段均值不应当作静态误差。', '',
        '## 核心数据', '',
        '| 运行 | 支撑反力误差 | 材料残差/载荷 | 最大节点速度均值 mm/s | 结论 |',
        '|---|---:|---:|---:|---|',
    ]
    primary = ['baseline_direct_plate', 'guided_base_plate', 'guided_iter40_plate', 'guided_dt0p5_plate', 'guided_dt0p25_plate',
               'guided_base_finger', 'guided_base_finger_off', 'fine_area_scaled_plate', 'fine_area_scaled_finger_off',
               'wave_soft30_membrane', 'wave_halfstep_membrane', 'wave_iter20_membrane', 'wave_repeat1_membrane', 'wave_repeat2_membrane',
               'long5s_plate', 'long5s_finger_off', 'mj0p1d20_plate', 'mj0p1d20_finger', 'mj0p1d20_finger_off', 'mj0p1d20_membrane']
    for name in primary:
        if name not in results: continue
        r = results[name]; v = r['summary']
        status = '单次通过' if r['passed'] else ('提前失败' if not r['checks']['finished'] else '未通过')
        lines.append(f"| [{name}](results/{name}.json) | {v['support_error_pct']:.3f}% | {v['material_residual_pct']:.3f}% | {v['max_speed_mm_s']:.3f} | {status} |")
    lines += ['', '![接触力与位移对照](comparison.png)', '', '## 已缩小的问题范围', '',
        f"1. 直接受压：Newton 粗网格下陷 {s('baseline_direct_plate','sink_mm'):.5f} mm，MuJoCo 同网格为 {s('mj0p1d20_direct_plate','sink_mm'):.5f} mm。两者静态形变接近，说明当前问题不能简单归结为四面体完全不承力。Newton 细网格直接受压为 {s('fine_grid_direct_plate','sink_mm'):.5f} mm，较粗网格变化 {abs(s('fine_grid_direct_plate','sink_mm')/s('baseline_direct_plate','sink_mm')-1)*100:.2f}%，尚不能声称达到5%的网格收敛。", '',
        '2. 刚体接触：独立增加迭代（1 ms，10→40）和固定40次迭代下减小子步（1→0.5→0.25 ms），没有让全部场景通过。增加全局阻尼、增大接触阻尼、关闭 tile、SDF 体素1.5→0.5 mm、网格加密也没有给出可接受的通用配置。细网格还做了接触 ke/kd 按名义顶面节点面积缩放至1/4的对照，仍未通过。载荷传至材料的平均值、瞬时接触力和局部平衡必须同时检查。', '',
        '3. 网格间隙：关闭全表面接触后，偏置20 mm手指穿过25 mm节点间隙，不能靠关闭该功能解决振动。将接触刚度降到30 N/m也会让小工具穿过；这不是可接受的修复。', '',
        '4. 膜片：接触刚度3000→30 N/m、子步0.1 ms、10次迭代时，固定配置3/3次通过，反力误差约1%，下陷4.417–4.426 mm；0.05 ms以及20次迭代复核失稳。这组配置值得保留，但还不能据此宣称随数值精度收敛。', '',
        '5. k_start：本机 SolverVBD 默认 beta=0，源码把 rigid_contact_k_start_value 设为 -1，实际直接使用材料刚度；改1e12不能解释或修复此问题。GPU轨迹有差异，不能把两次轨迹差异当作该无效参数的因果效应。', '',
        '以上只把故障缩小到当前接触路径及其数值稳定/收敛问题；还没有定位到可确认的单一求解器缺陷，也没有用调软实体材料来掩盖它。', '',
        '## 下一步边界', '',
        '继续第一项问题修复时，优先保留“偏置小手指”这个最小复现，检查全表面接触点/法线/重心坐标随子步的变化以及耦合迭代残差；先把该场景和压板同时做稳。若转MuJoCo，应先解决本报告中节点上手指的倒置和膜片稳定性，并重新验收。当前证据不足以批准整套模型迁移或进入第二项。', '',
        '## 复现与原始记录', '',
        '测量工具单元校验：`test_metrics.py`，包括能量差分梯度、压力合力/力矩、变形表面射线。', '',
        '快速复现（约数分钟，预期返回非零，因为包含应当检出的失败）：', '',
        '```bash', 'python3 /home/ubuntu/Dev/Projects/random/test/contact_transfer/run_suite.py --group quick', '```', '',
        '完整复现：将 `quick` 改为 `all`。脚本使用每条JSON保存的解释器和配置，结果加 `replay_` 前缀，原始数据保留。所有具体命令在 `MANIFEST.json`。运行不自动安装依赖、不修改生产模型。', '',
        'Newton环境为本机1.6.1 / Warp1.18；MuJoCo环境为3.17。Newton JSON记录了求解器源码SHA-256。耗时含建模、编译缓存、graph捕获及CPU测量；有CPU/GPU对照任务交错执行，不能把这些 wall_s 当作严格吞吐基准。', '',
        'MuJoCo备选使用相同NH体积弹性力，但采用20/s节点阻尼、显式体积力、原生接触，没有实现Newton的四面体粘性项；膜片使用MuJoCo原生StVK拉伸，Newton膜片是NH。因此只用其检验接口传力，不能直接比较膜片位移或宣称两后端完全等价。', '',
        '几何检查限定当前平压场景；50 ms采样不能证明两个采样点之间绝无瞬时倒置或穿透。物性均为设定值，未做实物标定。本轮未测试完整袋子、夹爪提起或GPU批量环境。', '',
        '早期 `baseline_*` 和 `iter40_*` 含自由刚体的探索运行，正式结论使用 `guided_*` 及之后运行；`baseline_direct_plate` 没有工具，不受这一区别影响。v2修正“静止”判定，加入工具速度，原始trace未改动。', '',
        '## 全部运行', '', '| 记录 | dt ms | 迭代 | ke N/m | 完成 | 通过 |', '|---|---:|---:|---:|---|---|']
    manifest = dict(generated_at=datetime.now(ZoneInfo('Asia/Taipei')).isoformat(), runs=[], sha256={})
    for name,r in results.items():
        c=r['config']
        lines.append(f"| [{name}](results/{name}.json) | {r['actual_dt_s']*1000:.4g} | {c.get('iterations','—')} | {c.get('ke','原生约束')} | {r['checks']['finished']} | {r['passed']} |")
        cmd=command(r)
        manifest['runs'].append(dict(name=name, passed=r['passed'], failure=r['failure'], replay_command=shlex.join(cmd), config=c))
    (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (HERE/'SUMMARY.json').write_text(json.dumps({name:dict(passed=r['passed'],failure=r['failure'],summary=r['summary'],checks=r['checks']) for name,r in results.items()},indent=2)+'\n')
    for p in sorted(HERE.rglob('*')):
        if p.is_file() and p.name!='MANIFEST.json' and '__pycache__' not in p.parts:
            manifest['sha256'][str(p.relative_to(HERE))]=hashlib.sha256(p.read_bytes()).hexdigest()
    (HERE/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(runs=len(results),single_run_passes=sum(r['passed'] for r in results.values()),report=str(HERE/'REPORT.md'))))


if __name__=='__main__': main()
