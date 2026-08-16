# AquaScope, démarrage rapide

[English](../index.md)

## ⚡ Installation
```bash
pip install aquascope              # noyau — collecteurs + hydrologie
pip install "aquascope[all]"       # tout — ML, viz, spatial, tableau de bord
```
Extras par groupe de fonctionnalités :
```bash
pip install "aquascope[ml]"           # sklearn, xgboost, statsmodels
pip install "aquascope[viz]"          # matplotlib, seaborn, folium
pip install "aquascope[scientific]"   # xarray, netcdf4, h5py
pip install "aquascope[interop]"      # xarray + geopandas (collecter en as_xarray / as_geodataframe)
pip install "aquascope[spatial]"      # rasterio, geopandas, shapely
pip install "aquascope[dashboard]"    # streamlit
pip install "aquascope[forecast]"     # prophet, torch (pour LSTM)
```
Pour le développement :
```bash
git clone https://github.com/Rekin226/aquascope.git
cd aquascope
pip install -e ".[all,dev]"
```
---
## 🚀 Exemples
### 1. Analyse de fréquence des crues (Bulletin 17C)
```python
from aquascope.api import flood_analysis
result = flood_analysis(daily_discharge, method="gev", return_periods=[10, 50, 100])
print(result.return_periods)
# {10: 1840.2, 50: 2530.7, 100: 2870.4}
print(result.confidence_intervals)
# {10: (1690.4, 2010.6), 50: (2280.1, 2820.9), 100: (2540.6, 3260.5)}
```
Changez `method` pour `"lp3"`, `"gumbel"`, `"gev_lmoments"`, ou `"gpd"`. La GEV non stationnaire (`fit_nonstationary_gev`) et l'algorithme EMA du Bulletin 17C pour les séries censurées (`expected_moments_algorithm`) sont disponibles dans `aquascope.hydrology.flood_frequency`.
### 2. Séparation de l'écoulement de base + signatures hydrologiques
```python
from aquascope.api import baseflow_analysis, compute_all_signatures
bf  = baseflow_analysis(daily_discharge, method="eckhardt")   # ou "lyne_hollick"
sig = compute_all_signatures(daily_discharge)
print(bf.bfi)                  # indice d'écoulement de base, ex. 0.42
print(sig.q5, sig.q95)         # dépassements de hautes eaux / basses eaux
print(sig.flashiness_index)    # indice de flashiness de Richards-Baker
```
21 signatures couvrant l'amplitude, la variabilité, la temporalité, la récession et la flashiness — voir [docs/features.md](../features.md#hydrological-analysis).

## 💻 CLI
AquaScope propose une CLI de 19 commandes pour les workflows les plus courants :
```bash
# Collecter des données
aquascope collect --source usgs --days 365
aquascope collect --source wapor --bbox 30.5,29.8,31.1,30.2 --variable RET --start-date 2026-04-01
# Analyse hydrologique
aquascope hydro --analysis flood-freq --file discharge.csv
aquascope hydro --analysis baseflow --file discharge.csv --method eckhardt
# Planification agricole
aquascope agri plan --crop maize --planting-date 2026-04-01 --lat 30.0 --lon 31.25
# Recommandation IA + résolution de problèmes en langage naturel
aquascope recommend --parameters DO,BOD5,COD --goal "pollution trend detection"
aquascope solve --problem "Assess flood risk for a 100-year return period"
# Tableau de bord interactif Streamlit — espace de travail multipage avec 21 sources
# en direct, insights automatiques intelligents, et graphiques Plotly entièrement interactifs
aquascope dashboard
```
Exécutez `aquascope --help` pour la liste complète des commandes.
