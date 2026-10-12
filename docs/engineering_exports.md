# Engineering exports: hand a record to HEC, SWMM, MODFLOW, FEWS and Raven

Practitioners work in HEC-HMS, HEC-RAS, HEC-SSP, SWMM, MODFLOW 6, Delft-FEWS and Raven.
None of them runs in a browser, but each reads plain inputs. AquaScope writes those inputs
from any gauge record, ready to open, so you leave with something that fits your workflow.

The same code (`aquascope.io.engineering`) is behind every face:

- **Explorer**: open a gauge, then pick a tool in **Export for…** next to Download CSV.
  You get a zip with the files and a README that says how to load them.
- **Studio**: a finished study's bundle carries an `engineering/` folder with the
  inputs for every tool that takes its main record.
- **CLI**: `aquascope export --to hec-ssp --station usgs/01134500` (or `--file record.csv`).
- **MCP**: `engineering_export(source, station_id, tool)` returns the files' text.

```bash
aquascope export --to all --station usgs/01134500 -o moose/          # every tool that fits
aquascope export --to hec-ssp --station usgs/01134500 --regional-skew 0.44 --regional-skew-mse 0.078
aquascope export --to modflow6 --file stage.csv --variable water_level --cell 1 12 30 --cond 250 --rbot 101.5
pip install "aquascope[engineering]"   # optional: real .dss files (hecdss) and the FloPy writer
```

## What each tool gets

| Tool | Records | Files | Written against |
|---|---|---|---|
| HEC-HMS | flow, rainfall | the gage record as DSS, with the steps to attach it in the Time-Series Data Manager | HEC-HMS keeps gage data in DSS |
| HEC-RAS | flow, stage | the `Flow Hydrograph=` (or `Stage Hydrograph=`) block of a `.u##` file, and the DSS record | [ras-commander](https://github.com/gpt-cmdr/ras-commander) (MIT); HEC does not publish the file layout |
| HEC-SSP | flow, stage | `annual_peaks.csv` for the Data Importer, the DSS record, `b17c_settings.txt` with AquaScope's own Bulletin 17C result to compare | [HEC-SSP Data Importer](https://www.hec.usace.army.mil/confluence/sspdocs/sspum/latest/using-the-hec-ssp-data-importer/developing-a-new-data-set) |
| HEC-DSS | any | the record as a DSS time series | [hecdss](https://github.com/HydrologicEngineeringCenter/hec-dss-python) (MIT, HEC) |
| SWMM 5 | flow, rainfall, stage | a time series file, the same data as an inline `[TIMESERIES]` block, and the `[INFLOWS]` or `[RAINGAGES]` line | [SWMM 5.2 User's Manual](https://www.epa.gov/system/files/documents/2022-04/swmm-users-manual-version-5.2.pdf), section 11.6 and Appendix D |
| MODFLOW 6 | stage, groundwater level, flow | a River (stage) or Well (flow) package with one stress period per time step, and its TDIS file | [MODFLOW 6 Input Guide](https://modflow6.readthedocs.io/en/stable/_mf6io/gwf-riv.html) |
| Delft-FEWS | any | a PI-XML time series | [pi_timeseries.xsd](https://fewsdocs.deltares.nl/schemas/version1.0/pi-schemas/pi_timeseries.xsd) |
| Raven | flow, stage, rainfall | an `:ObservationData` block (`HYDROGRAPH`, `WATER_LEVEL`) or a `:Data PRECIP` block for an `.rvt` file | [Raven manual v3.8](https://raven.uwaterloo.ca/files/v3.8/RavenManual_v3.8.pdf), section A.4.2 |

How the formats were checked: the PI-XML output validates against the published XSD; the
MODFLOW 6 files load in FloPy with the right stress periods; the DSS CSV reads back with
HEC's own `hecdss` reader; the SWMM, Raven and HEC-RAS outputs are parsed back in the tests.
The tools themselves were not run.

### Read before you use them

- **DSS.** HEC's `hecdss` ships its native library for Linux and Windows. There, and with the
  `engineering` extra, you get a real `.dss` file. Elsewhere, and always in the browser, you
  get the CSV layout `hecdss` reads, which becomes a `.dss` with
  `HecDss('out.dss').put(RegularTimeSeries.read_csv('file.dss.csv'))`.
- **Period stamps.** Daily means and rainfall totals go into DSS as `PER-AVER` and `PER-CUM`,
  stamped at the end of the period (a day's mean at 2400), HEC's convention. Raven gets
  period-starting times, as its manual asks. The PI-XML keeps the times as the source gave
  them and says so in a comment.
- **Annual peaks** are the maxima of daily means by water year (October to September by
  default), not instantaneous peaks. Where the agency publishes a peak-flow file, that is the
  better input for a design value.
- **Gaps.** HEC-RAS and MODFLOW 6 cannot hold a missing value. Gaps of up to 10 steps are filled
  by linear interpolation; a longer gap is not invented, so the file starts after the last one.
  The README says what was filled or cut. DSS, PI-XML and Raven keep gaps as missing values.
  The SWMM files leave them out: SWMM interpolates a flow or level across a gap and reads
  missing rain as none.
- **MODFLOW 6.** The cell is layer 1, row 1, column 1 unless you pass `--cell`. Without
  `--cond` and `--rbot` the River package carries placeholders (conductance 1, bottom 1 m
  below the lowest stage), flagged in the file. Flow becomes m3/d for a Well package.
- **Not written:** HEC-HMS project files (`.gage`, `.basin`) and HEC-SSP analysis files,
  which HEC does not document. The README in each folder gives the few clicks instead.
- **Licence.** Each README names the source's data licence and attribution. The files are for
  your own models; the source's terms still apply to the data in them.

## Bulletin 17C check

Engineers will not trust a Bulletin 17C number until it agrees with HEC-SSP. The reference
cases are the worked examples of [Bulletin 17C itself](https://doi.org/10.3133/tm4B5)
(England and others, 2018, Appendix 10), computed with USGS PeakFQ v7.1 (the figures say so). HEC's documentation
of the same data sets ([HEC-SSP Bulletin 17C examples](https://www.hec.usace.army.mil/confluence/sspdocs/sspexamples/latest/bulletin-17c-examples))
states that HEC-SSP gives the same results as PeakFQ for the first six. Every published number
below was copied from the Bulletin's tables; the AquaScope column is
`aquascope.hydrology.b17c_check.run()`, and a test checks this page carries exactly its output.

**Bulletin 17C Example 1 (systematic record)**: USGS 01134500 Moose River at Victory, Vermont, 68 peaks. Source: Tables 10-2 (peaks) and 10-5 (quantiles); moments in the text of Example 1.

| | Mean of log10 | Std of log10 | Skew |
|---|---|---|---|
| Published (weighted skew) | 3.3286 | 0.1403 | 0.421 |
| AquaScope | 3.3286 | 0.1403 | 0.422 |

| AEP | Published (ft3/s) | AquaScope (ft3/s) | Difference |
|---|---|---|---|
| 0.1 | 3,261 | 3,262 | +0.0% |
| 0.04 | 3,911 | 3,921 | +0.2% |
| 0.02 | 4,422 | 4,440 | +0.4% |
| 0.01 | 4,957 | 4,986 | +0.6% |
| 0.005 | 5,519 | 5,561 | +0.8% |
| 0.002 | 6,313 | 6,376 | +1.0% |

**Bulletin 17C Example 2 (potentially influential low floods)**: USGS 11274500 Orestimba Creek near Newman, California, 82 peaks. Source: Tables 10-6 (peaks) and 10-9 (quantiles); moments and PILF threshold in the text of Example 2.

| | Mean of log10 | Std of log10 | Skew |
|---|---|---|---|
| Published (station skew) | 3.0227 | 0.6821 | -0.929 |
| AquaScope | 2.5822 | 1.3996 | -0.705 |

Low outliers: published threshold 782 ft3/s with 30 peaks censored; AquaScope threshold 39.4 ft3/s with 15 censored.

| AEP | Published (ft3/s) | AquaScope (ft3/s) | Difference |
|---|---|---|---|
| 0.5 | 1,339 | 556 | -58.4% |
| 0.2 | 4,026 | 6,049 | +50.2% |
| 0.1 | 6,328 | 17,271 | +172.9% |
| 0.04 | 9,426 | 45,984 | +387.8% |
| 0.02 | 11,690 | 80,520 | +588.8% |
| 0.01 | 13,820 | 127,284 | +821.0% |
| 0.005 | 15,800 | 186,626 | +1081.2% |
| 0.002 | 18,150 | 283,876 | +1464.1% |

**What this says.** On a plain systematic record with weighted skew (Example 1) AquaScope's
EMA matches: the mean and standard deviation agree to the published digits, the weighted skew
is 0.422 against 0.421, and the quantiles are within 1%, the gap growing toward the rare floods.

On a record with zero flows and potentially influential low floods (Example 2) it does not.
The published Multiple Grubbs-Beck test censors 30 peaks below 782 ft3/s; AquaScope's
version stops at 39.4 ft3/s with 15 censored. With the wrong peaks censored, the standard
deviation of the logs comes out at about twice the published one (1.40 against 0.68) and the
skew at -0.705 against -0.929, so the median flood is less than half the published value and
the rare floods are several times too large. Until that is fixed, use
HEC-SSP or PeakFQ for any record with zeros or low outliers; the `b17c_settings.txt` that
the HEC-SSP export writes says so.

**Not run yet.** Bulletin 17C Examples 3 to 7 (a broken record, historical data,
crest-stage censoring, historic data with low outliers, paleoflood data) need flow intervals
and perception thresholds, which AquaScope's EMA does not take yet.
