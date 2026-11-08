# 四个可变形软包资产

本包仅包含四个分类软包及其运行支持文件，物理建模与调参报告单独提供。

| 加载键 | 分类 | 目录 |
| --- | --- | --- |
| pe | PE 薄膜快递袋 | assets/01_pe_film |
| kraft | 牛皮纸邮寄袋 | assets/02_kraft_paper |
| paperboard | 纸板文件封 | assets/03_paperboard_envelope |
| legacy | 通用实验软袋 材质未确认 | assets/04_generic_experimental |

Python 3.12 及以上，MuJoCo 3.14.0。在解压根目录执行：

```bash
python -m pip install -r requirements.txt
python -B verify_package.py
python -B inspect_asset.py --asset pe
python -B inspect_asset.py --asset kraft
python -B inspect_asset.py --asset paperboard
python -B inspect_asset.py --asset legacy
python -B inspect_asset.py --asset pe --viewer
```

界面开始暂停，空格切换物理运行。统一入口默认加载空袋。

```python
from load_assets import load_asset
bag = load_asset('pe')
try:
    bag.step()
finally:
    bag.close()
```

必须使用对象的 step() 与 close()。同一进程切换袋子前关闭上一个对象。
前三类使用共享 assets/runtime.py；材料配置在各自 material.json。
旧袋材料配置在 config.json 的 bag_params，也可调用时传 params 覆盖。

PE 保持 plasticity=false、precreased_panel_boundaries=false，不采用纸张式永久折痕或预折线。其他袋配置沿用分类交付基线，具体修正见 CHANGES.json。四个资产均未完成实物物性标定。

CATALOG.json 为四个核心袋资产索引；旧袋 XML 模板及可选内装物示例为运行支持文件，不计为额外袋资产。SOURCE_MAP.json 与 provenance/ 保留来源信息，原第三方声明继续适用。修改配置后哈希检查显示该文件改变是预期结果。
