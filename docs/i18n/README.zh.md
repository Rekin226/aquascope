# AquaScope 快速开始

[English](../index.md)

AquaScope 把 **37 个全球水文数据源**统一到同一套 Python 数据结构之下，并在其上叠加完整的科学计算栈——从**洪水频率方法**到 **FAO-56 作物需水量**；一个 AI 引擎会针对你的数据集为 **27 种研究方法**打分，并自动执行 **26 条分析流水线**。核心的[浏览器地图](../explorer.md)工作流无需安装，也不需要 API key；采集结果汇总为开放的[数据档案库](../archive.md)。

## ⚡ 安装

```bash
pip install aquascope              # 核心 — 采集器 + 水文分析
pip install "aquascope[all]"       # 全部 — 机器学习、可视化、空间分析、仪表盘
uv tool install "aquascope[all]"   # 或作为独立环境中的命令行工具安装
```

以后升级时，`aquascope update` 会找到最新版本，并按当初的安装方式升级（uv、pipx 或当前环境的 pip）；`aquascope update --check` 只检查而不升级，`aquascope --version` 显示你当前的版本。

按功能分组的 extras：

```bash
pip install "aquascope[ml]"           # sklearn, xgboost, statsmodels
pip install "aquascope[viz]"          # matplotlib, seaborn, folium
pip install "aquascope[scientific]"   # xarray, netcdf4, h5py
pip install "aquascope[interop]"      # xarray + geopandas（采集为 as_xarray / as_geodataframe）
pip install "aquascope[spatial]"      # rasterio, geopandas, shapely
pip install "aquascope[dashboard]"    # streamlit
pip install "aquascope[forecast]"     # prophet, torch（用于 LSTM）
pip install "aquascope[studio]"       # matplotlib, openpyxl, python-docx（Studio 的 Word、Excel、图表）
pip install "aquascope[basins]"       # pyogrio, geopandas（BasinATLAS 集水区、相似流域）
```

开发安装：

```bash
git clone https://github.com/Rekin226/aquascope.git
cd aquascope
pip install -e ".[all,dev]"
```

---

## 🚀 示例

### 1. 洪水频率分析（Bulletin 17C）

```python
from aquascope.api import flood_analysis

result = flood_analysis(daily_discharge, method="gev", return_periods=[10, 50, 100])
print(result.return_periods)
# {10: 1840.2, 50: 2530.7, 100: 2870.4}
print(result.confidence_intervals)
# {10: (1690.4, 2010.6), 50: (2280.1, 2820.9), 100: (2540.6, 3260.5)}
```

把 `method` 换成 `"lp3"`、`"gumbel"`、`"gev_lmoments"` 或 `"gpd"`。非平稳 GEV（`fit_nonstationary_gev`）与用于删失记录的 Bulletin 17C EMA（`expected_moments_algorithm`）可在 `aquascope.hydrology.flood_frequency` 中找到。

### 2. 基流分割 + 水文特征

```python
from aquascope.api import baseflow_analysis, compute_all_signatures

bf  = baseflow_analysis(daily_discharge, method="eckhardt")   # 或 "lyne_hollick"
sig = compute_all_signatures(daily_discharge)

print(bf.bfi)                  # 基流指数，例如 0.42
print(sig.q5, sig.q95)         # 高流量 / 低流量超越值
print(sig.flashiness_index)    # Richards-Baker 闪发性指数
```

22 个水文特征，覆盖量级、变异性、时间性、退水与闪发性——见 [docs/features.md](../features.md#hydrological-analysis)。

---

## 💻 CLI

AquaScope 提供 55 条命令的 CLI（`agri`、`basins`、`caravan`、`eval`、`evidence`、`groundwater`、`gym`、`layers`、`playbooks`、`river` 和 `route` 还带有子命令），覆盖最常见的工作流：

```bash
# 查找水文站，然后采集数据
aquascope stations --bbox -0.5,51.3,0.3,51.7 --variable discharge --format geojson

aquascope collect --source usgs --days 365

# 水文分析
aquascope hydro --analysis flood-freq --file discharge.csv
aquascope hydro --analysis baseflow --file discharge.csv --method eckhardt -o baseflow.json

# 农业规划
aquascope agri plan --crop maize --planting-date 2026-04-01 --lat 30.0 --lon 31.25 -o plan.csv

# AI 推荐 + 自然语言问题求解
aquascope recommend --parameters DO,BOD5,COD --goal "pollution trend detection" -o recommendations.json
aquascope solve "Design flow for a road crossing, 100-year return period" --lat 51.415 --lon -0.308

# 交互式 Streamlit 仪表盘 — 多页面工作区，包含 37 个实时数据源、
# 智能自动洞察，以及完全交互式的 Plotly 图表
aquascope dashboard
```

运行 `aquascope --help` 查看完整命令列表。
