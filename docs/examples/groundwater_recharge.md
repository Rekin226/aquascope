# Groundwater recharge and GRACE anomalies

Explore GRACE total-water-storage anomalies, recharge estimation, aquifer hydraulics, and trend plots with AquaScope.

Runtime: ~15 seconds on a laptop. source note: [`Open the tutorial notebook`](https://github.com/Rekin226/aquascope/blob/main/notebooks/09_groundwater_recharge.ipynb)

!!! note "Synthetic data"
Every series in this notebook is synthetic, generated from a fixed random seed. The true values (−10 mm/year storage trend, an aquifer with T = 500 m²/day and S = 0.001) are known in advance, so each step can be checked against the truth. Nothing here describes a real aquifer or basin.

## The scenario

Groundwater is hard to observe directly. This tutorial shows three complementary views of the same problem, using the aquascope.groundwater module:

1. **Recharge** is the water that reaches the water table. It is estimated from a well record with the Water Table Fluctuation (WTF) method.

2. **Aquifer hydraulics** describe how the aquifer responds to pumping. A synthetic pumping test is analysed with the Theis and Cooper-Jacob solutions.

3. **GRACE storage anomalies** give the regional trend. Groundwater storage is isolated as GWS = TWS − SM − SW, then a depletion rate, a trend line and anomalies are computed.

## What this shows about AquaScope
- **`water_table_fluctuation`** converts water-level rises into a recharge rate (mm/year) from a specific yield.
- **`theis_drawdown, cooper_jacob and estimate_transmissivity`** simulate a pumping test and recover transmissivity and storativity from noisy observations.
- **`safe_yield`** compares recharge with extraction to assess sustainability.
- **`GRACEProcessor`** computes groundwater storage, the depletion rate with a 95 % confidence interval, the trend and anomalous months.

aquascope groundwater exposes the same recharge and Theis calculations from the command line.


## Run it

Requires Python 3.10 or later.
#1. install all the dependecies 
``` bash

pip install aquascope matplotlib jupyter

```
#2. Download the tutorial notebook
Download 09_groundwater_recharge.ipynb, then open it with Jupyter and choose Restart & Run All:

``` bash
jupyter notebook 09_groundwater_recharge.ipynb

```
The notebook also runs unchanged on Colab or Binder

## Verified results

With the fixed seed, the notebook prints:

Step	      | Quantity	   | Result	         |Truth

Recharge (WTF)|	Recharge rate  |146.4 mm/year	 |150 mm/year

Pumping test  |Drawdown 	   |1.50 m           |
              |after 1 day

Pumping test  |Transmissivity  |497 m²/day	     |500 m²/day

Pumping test  |Storativity	   |0.00102	         |0.001

Sustainability|Extraction /    |1.09             | --	
               recharge        |(unsustainable)

GRACE	      |Depletion rate  |−10.1 mm/year (95|−10 mm/year
                                %CI −10.7 to −9.6)	

GRACE         |	Volume change  |−0.36 km³/year	 | ----

GRACE         |	Anomalous      |	3	
                 months

The notebook draws three figures: the synthetic water level, the pumping-test drawdown curves, and the groundwater-storage series with its linear trend and anomalies.
