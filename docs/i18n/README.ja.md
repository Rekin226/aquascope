# AquaScope クイックスタート

[English](../index.md)

## ⚡ インストール

```bash
pip install aquascope              # core — collectors + hydrology
pip install "aquascope[all]"       # everything — ML, viz, spatial, dashboard
uv tool install "aquascope[all]"   # or as a command-line tool in its own environment
```

`aquascope update` は最新版を確認し、インストール時と同じ方法（uv、pipx、または現在の環境の pip）で更新します。
`aquascope update --check` で更新の有無を確認でき、`aquascope --version` でインストール済みのバージョンを表示できます。

必要な機能に応じて、追加の依存パッケージをインストールできます。

```bash
pip install "aquascope[ml]"           # sklearn, xgboost, statsmodels
pip install "aquascope[viz]"          # matplotlib, seaborn, folium
pip install "aquascope[scientific]"   # xarray, netcdf4, h5py
pip install "aquascope[interop]"      # xarray + geopandas (collect as_xarray / as_geodataframe)
pip install "aquascope[spatial]"      # rasterio, geopandas, shapely
pip install "aquascope[dashboard]"    # streamlit
pip install "aquascope[forecast]"     # prophet, torch (for LSTM)
pip install "aquascope[studio]"       # matplotlib, openpyxl, python-docx (the Studio's Word, Excel, figures)
pip install "aquascope[basins]"       # pyogrio, geopandas (BasinATLAS catchments, similar basins)
```

開発環境をセットアップするには、次のコマンドを実行します。

```bash
git clone https://github.com/Rekin226/aquascope.git
cd aquascope
pip install -e ".[all,dev]"
```

---

## 🚀 使用例

### 1. 洪水頻度解析（Bulletin 17C）

```python
from aquascope.api import flood_analysis

result = flood_analysis(daily_discharge, method="gev", return_periods=[10, 50, 100])
print(result.return_periods)
# {10: 1840.2, 50: 2530.7, 100: 2870.4}
print(result.confidence_intervals)
# {10: (1690.4, 2010.6), 50: (2280.1, 2820.9), 100: (2540.6, 3260.5)}
```

`method` には `"lp3"`、`"gumbel"`、`"gev_lmoments"`、`"gpd"` も指定できます。
`aquascope.hydrology.flood_frequency` では、非定常 GEV の当てはめ（`fit_nonstationary_gev`）と、
打ち切りデータを含む観測記録に対する Bulletin 17C の Expected Moments Algorithm（EMA、`expected_moments_algorithm`）も利用できます。

### 2. 基底流分離と水文シグネチャ

```python
from aquascope.api import baseflow_analysis, compute_all_signatures

bf  = baseflow_analysis(daily_discharge, method="eckhardt")   # or "lyne_hollick"
sig = compute_all_signatures(daily_discharge)

print(bf.bfi)                  # baseflow index, e.g. 0.42
print(sig.q5, sig.q95)         # high-flow / low-flow exceedances
print(sig.flashiness_index)    # Richards-Baker flashiness index
```

22 種類の水文シグネチャで、流量の大きさと変動性、流況の時期特性、減水特性、流量の急変性を把握できます。
詳しくは[水文解析の機能一覧](../features.md#hydrological-analysis)を参照してください。

---

## 💻 CLI

AquaScope の CLI には 55 個のコマンドがあります。
`agri`、`basins`、`caravan`、`eval`、`evidence`、`groundwater`、`gym`、`layers`、`playbooks`、`river`、`route` にはサブコマンドがあります。
よく使う操作の例を以下に示します。

```bash
# Find stations, then collect data
aquascope stations --bbox -0.5,51.3,0.3,51.7 --variable discharge --format geojson
aquascope harvest stations --out archive          # the open gauge catalog (GeoParquet)
aquascope basins at 48.85 2.35                    # the catchment of any point: area, climate, land cover, soils, dams (BasinATLAS)
aquascope basins similar 25.04 121.56             # gauged basins whose catchments look most like this point's (ungauged-site donors)
aquascope basins regionalize 52.29 -3.51          # estimated flow regime of an ungauged point from those donors, with the leave-one-out skill
aquascope river snap 46.948 7.452                 # the river reach at a point (GEOGLOWS v2), or "no stream within 1 km"
aquascope river record --at 46.948 7.452          # that reach's simulated daily flow since 1940: return periods, FDC (modelled)
aquascope river trace --at 46.948 7.452           # follow it to the sea: length, path, the gauges, dams and countries it passes
aquascope river dams --at 46.948 7.452            # the dams upstream of that reach and the degree of regulation
aquascope now --station usgs/USGS-01350000        # today against normal, and the 15-day forecast corrected to the gauge
aquascope evidence verify --out verify --days 30  # how close the issued forecasts came to the gauge, per model and lead day
aquascope bulletin 2026-09 --out bulletin         # last month's state of the rivers: HydroSOS classes, HTML and Markdown
aquascope warnings --bbox -10 35 30 60            # floods ahead: reaches expected to reach their 2-year flow in 15 days
aquascope flowstatus --bbox -10 35 30 60          # every big reach's flow today against its normal for the month
aquascope watch usgs/USGS-01350000 river:230260670 --since 2026-10-01   # what changed since then: data, status, forecast, floods
aquascope assess 51.415 -0.308 --problem flood_risk   # what can be answered here: gauges in reach, catchment, which methods the record supports
aquascope context 51.86 5.95                      # flood history, surface water, flood depth, dams, rain gauge, ET and soil at a place
aquascope caravan export --source uk_ea --out caravan_gb   # a Caravan-format large-sample dataset from the archive
aquascope export --to hec-ssp --station usgs/01134500   # inputs for HEC-HMS/RAS/SSP, SWMM, MODFLOW 6, Delft-FEWS or Raven
aquascope gym run --basin uk_ea/013054a3-670e-49ee-afda-e0865a449197   # AquaScope Bench: calibrate GR4J on a real basin as a gym episode
aquascope layers frames precip --start 2024-05-01 --end 2024-05-20   # a time-lapse of a dated map layer: dates and tile URLs
aquascope map "trace the Nile to the sea" --resolve   # plain words to the Explorer's map actions (keyless rules)
aquascope mesh my_run.p01.hdf                     # a HEC-RAS 2D or UGRID mesh result: cells, times, largest depth and speed
aquascope whatif --lat 47.598 --lon 8.682 "20% less rain and 2 degrees warmer"   # GR4J for the basin, the change carried downstream
aquascope swmm model.inp model.out --crs 32617   # an EPA SWMM model and results: network, flooding nodes, largest flows
aquascope layers scout --bbox -10 35 30 60           # what stands out on the map, each with its reason (the Explorer's Scout)
aquascope groundwater surface --mode change      # the head surface of the Archive's wells and its change, masked where wells are sparse
aquascope modflow model.hds --grid model.dis.grb --crs EPSG:27700   # your MODFLOW heads on the map: cells in lon/lat, per layer and time
aquascope mcp                                     # serve the same tools to Claude / Cursor over MCP
aquascope ask "100-year flood of the Seine at Paris?"   # the analyst: tools + a cited Markdown report
aquascope ingest agency_export.csv --unit cfs     # any CSV/Excel -> clean daily series + QA report
aquascope collect --source usgs --days 365
aquascope collect --source wapor --bbox 30.5,29.8,31.1,30.2 --variable RET --start-date 2026-04-01

# Hydrological analysis
aquascope hydro --analysis flood-freq --file discharge.csv
aquascope hydro --analysis baseflow --file discharge.csv --method eckhardt -o baseflow.json

# Agriculture planning
aquascope agri plan --crop maize --planting-date 2026-04-01 --lat 30.0 --lon 31.25 -o plan.csv

# AI recommendation + natural-language problem solving
aquascope recommend --parameters DO,BOD5,COD --goal "pollution trend detection" -o recommendations.json
aquascope solve "Design flow for a road crossing, 100-year return period" --lat 51.415 --lon -0.308
aquascope studio                                     # the crew: asks where and what, then brief, plan, run, bundle
aquascope studio "Design flow for a road crossing, 100-year, and how sure can we be" --at "Thames Kingston" --out kingston/
aquascope studio kingston/ --exclude-years 2014 --sign checked="A. Name"  # the Study Desk: revise, review, sign
aquascope eval score kingston/                       # how the crew did: gates, Critic, report quality, time, cost
aquascope eval stats studies/ --by model             # many studies at once: grades, gate failures, cost per study
aquascope area-study --bbox=-0.9,51.2,0.3,51.8       # a flood study over every gauge in a box: Q100, flood trends, a regional curve
aquascope paint --bbox=-8,49.8,2,59                    # paint every Archive gauge in a box with its flood trend, plus field significance
aquascope replay --at 51.95 -2.2 --month 2024-11       # a past flood as a story: rain, gauge vs model, forecast, wave, flood
aquascope route flood 230301988 --peak 2000             # a flood wave down GEOGLOWS's river network
aquascope flood 230567275 --flow 954 --levee ring.geojson --crest 15.1   # a screening flood of the valley, with a drawn levee
aquascope well --at 4.88 45.76 --rate 1000 --days 90   # Theis drawdown round a pumping well
aquascope landuse --polygon '[[4.955,45.70],[4.97,45.70],[4.97,45.708],[4.955,45.708]]' --use residential_1_8 --rain 100

# Interactive Streamlit dashboard — multipage workspace with 38 live sources,
# smart auto-insights, and fully interactive Plotly charts
aquascope dashboard

# Shell tab-completion
eval "$(aquascope completion bash)"   # add this to ~/.bashrc (or .zshrc / config.fish)
```

すべてのコマンドを確認するには、`aquascope --help` を実行してください。
