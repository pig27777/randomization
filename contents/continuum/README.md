# 三维弹塑性连续体：A = g_B(X)

本目录是用户指定的新内容物候选。MuJoCo 4原生full 3D StVK弹性与J₂塑性、MJWarp CUDA时间推进、Newton约束求解。没有CG或CPU时间推进回退。CPU负责模型编译、机械臂初始化/IK与数据导出；原共享`.venv`仍为3.17。

## 固定模板与变化参数

`Template`定义固定B：5×4×3节点、六四面体分格规则、full自由度、原生材料规律与数值配置。默认60顶点/144四面体、180内容物自由度、864塑性状态量。`Parameters`定义X：

| 参数 | 含义 |
| --- | --- |
| size | 真实外包络长宽高，m |
| roundness / bulge / taper | 圆角轮廓、中央厚边缘薄、两端宽度渐变 |
| density | 平均体积密度，kg/m³；总质量由实际四面体体积导出 |
| young / poisson | 弹性模量Pa与泊松比；压缩和弯曲响应相互耦合 |
| damping | 原生Rayleigh阻尼，s |
| yield_strain | 正的J₂屈服应变阈值；很大阈值作弹性对照 |
| friction | 表面滑动摩擦系数 |

四种预设形状`box/rounded/pillow/tapered`改变整个实体参考几何，所有实例保持同一连接拓扑。质量采用实际网格体积×平均密度；顶点集中质量分配沿用原生flexcomp，不宣称完成均匀密度有限元质量矩阵标定。硬/易弯/易压预设尺寸密度相同，材料参数不同；名称是探索目标，尚未标定或验收相应响应范围。采样器产生代数合法参数，动态可行域另验。

不增加压密本构。J₂塑性描述永久形状变化，不代表衣物层间滑动、蠕变或泡沫压实。初版支持单个连通实心体；袋膜耦合、任意拓扑与目标响应κ到材料参数的换算表不在当前实现中。

当前关闭实体自碰撞，不能据此保证大幅折叠时表面不穿越。每步保持1e-5有效质量显式残差门槛；必要时最多两次GPU同方程精化，记录精化前后指标，约束求解仍为Newton。

## 文件

- `model.py`：参数、固定生成器、形状、原生MJCF和模型编译。
- `gpu.py`：CUDA推进、原始overflow/有限性/J/独立残差检查、完整integration state保存恢复。有效质量诊断复用`bag_sim/flex_residual.py`，不启用旧膜曲率/metric补丁。
- `grasp_adapter.py`：接入现有完整YAM平台；只使用原生接触，pinch_coeff=0；塑性历史随轨迹保存。

不同X以各自编译模型分组运行；一个`Simulation`内是同X的独立world，不能把它称为异材料全GPU随机化。当前抓取适配每步导出GPU数据用于既有控制器，优先核查行为，不作为优化吞吐实现。

## 环境与入口

```sh
uv venv --python /usr/bin/python3 .venv-continuum
uv pip install --python .venv-continuum/bin/python -r envs/continuum-requirements.txt

# 材料与状态检查：输出目录必须不存在，全部GPU调用持锁。
flock /tmp/claude-gpu.lock .venv-continuum/bin/python -B test/content_model/validate_continuum.py \
  --output test/runs/YYYY-MM-DD/contents/NEW_RUN/checks

# 使用原公共机械臂、动作、力限和判定；显式关闭效果捏取。
PYTHONPATH=contents flock /tmp/claude-gpu.lock .venv-continuum/bin/python -B test/flex3d/angles.py \
  --backend continuum.grasp_adapter --dt .0005 --pinch-coeff 0 --yaw 0 \
  --model-options '{"material":"bendable","shape":"pillow"}' \
  --output test/runs/YYYY-MM-DD/contents/NEW_RUN/grasp
```

状态保存必须包含`mjSTATE_PLASTIC`。公共平台产生的`trajectory.npz`增加`integration_state`、`integration_state_spec`与`final_integration_state`；qpos-only回放仅能展示几何，不能用于重启塑性动力学。

本轮计划和结果见[连续体报告](../../test/content_model/CONTINUUM_REPORT.md)，任务检查见[检查报告](../../reports/content-continuum-test.md)。当前是否通过，以报告最新结果为准。

## 调参后运行机械臂

```sh
.venv-continuum/bin/python -B test/content_model/grasp_ui.py --port 8766
```

打开`http://127.0.0.1:8766`，调整材料、圆润/鼓包/渐变形状、实际尺寸、屈服应变、摩擦和抓取角度；先预览或直接运行。UI每次最多10秒仿真时间，抓取失败或数值失效提前停止；它不等于10秒墙钟。左图为完整YAM、右图为夹爪局部，来自真实仿真样本，结束后提供轨迹录像和结果JSON。当前软形状存在塑性非有限失败，不能将界面可运行视为物理验收。

界面复用原平台的执行器和判定，GPU动力学持共享锁；CPU Mesa只画图。每步导出与CPU控制是现有UI适配的开销。纯GPU常驻测速独立于UI，状态与控制留在GPU；每100步计时结束后只读取诊断值，异κ分别上传：

```sh
flock /tmp/claude-gpu.lock .venv-continuum/bin/python -B test/content_model/benchmark_continuum.py \
  --matrix --worlds 1 32 --steps 2000 --repeats 3 \
  --output test/runs/YYYY-MM-DD/contents/NEW_RUN/gpu-matrix
```

公共平台的旧袋子依赖会导入一个同名`contents.py`，因此统一使用`PYTHONPATH=contents`和`continuum.grasp_adapter`，不修改历史包的导入规则。

`--model-options`还可直接指定`roundness/bulge/taper`覆盖命名形状的参数，连续改变实体参考几何；`size/density/young/poisson/damping/yield_strain/friction`也可指定。
