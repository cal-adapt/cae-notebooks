Climate Profiles Notebooks
==========================

This folder will house notebooks that can be used to generate custom Climate Profiles hourly datasets (8760s), including Standard Year Profile (single variable), Typical Meteorological Years (TMY, multivariable), and 8760s.

## Notebooks

[custom_climate_profiles.ipynb](custom_climate_profiles.ipynb): Generate a custom hourly climate profile a single location.  *In development*

*For example: How do I generate a TMY in .epw format for the Los Angeles International Airport location using climate model data?*


[typical_meteorological_year_methodology.ipynb](typical_meteorological_year_methodology.ipynb): Explore the full methodology of a Typical Meteorological Year to represent the typical climatological conditions over one year of hourly data.  

*For example: What will an average year of hourly temperature data look like in a 2°C warmer world?*

## Filename Conventions

### Standard Year

stdyr\_`VARIABLE`\_`PERCENTILE`\_`STATION_NAME`\_`GWL_PERIOD`\_`DELTA`\_`WINDOW`\_`APPROACH`\_`CENTERED_YEAR`\_`SCENARIO`\_`BA_MODELS`.csv

**Examples:**
  - stdyr_t2_50ptile_sacramento_executive_airport_ksac_present-day_delta_from_historical_30yr_window_warming_level.csv
  - stdyr_t2_50ptile_sacramento_executive_airport_ksac_mid-late-century_delta_from_historical_30yr_window.csv
  - stdyr_t2_50ptile_san_diego_lindbergh_field_ksan_30yr_window_time_2020_ssp370.csv
  - stdyr_prec_75ptile_35-5N_122-5W_delta_from_historical_10yr_window_time_2016_ssp370_ba_models.csv
  - stdyr_rh_derived_50ptile_santa_barbara_municipal_airport_ksba_delta_from_historical_30yr_window_time_2015_ssp370.csv



#### Filename Components

| Component | Description | Options |
|--------|-------------|--------|
| `VARIABLE` | Climate variable measured | `t2` – Air temperature at 2m<br>`rh_derived` – Relative humidity<br>`wind_speed_derived` – Wind speed<br>`swdnb` – Solar radiation / Shortwave downward normal beam radiation<br>`noaa_heat_index_derived` – NOAA heat index |
| `PERCENTILE` | Statistical percentile | `05ptile` – 5th percentile<br>`50ptile` – 50th percentile / median<br>`95ptile` – 95th percentile |
| `STATION_NAME` | Weather station identifier | e.g., `palm_springs_regional_airport_kpsp` |
| `GWL_PERIOD` | Global warming level 30-year period | `present-day` – 1.2°C GWL<br>`near-future` – 1.5°C GWL<br>`mid-century` – 2.0°C GWL<br>`mid-late-century` – 2.5°C GWL |
| `DELTA` | Whether the difference from baseline (delta) was taken | `no_delta` - no delta taken <br>`delta_from_historical` - delta taken |
| `WINDOW` | Years around the year in which the input GWL is reached | e.g., `30yr` (corresponding to a 15yr window size) |
| `APPROACH` | Climate profile approach used | `time`<br>`warming_level` |
| `CENTERED_YEAR` | For approach='Time', the year used to find a corresponding warming level | e.g., `2015` |
| `SCENARIO` | SSP scenario (default: `ssp370`) | `ssp245` – SSP 2-4.5<br>`ssp370` – SSP 3-7.0<br>`ssp585` – SSP 5-8.5 |
| `BA_MODELS` | If only bias-adjusted models were used (default: FALSE) | `ba_models` - TRUE |


### Typical Meteorological Year

tmy\_`STATION_NAME`\_wrf_`SOURCE_ID`\_`GWL_PERIOD`.`EXTENSION`

**Examples:**
  - tmy_palm_springs_regional_airport_kpsp_wrf_mpi-esm1-2-hr_r3i1p1f1_mid-century.csv
  - tmy_palm_springs_regional_airport_kpsp_wrf_mpi-esm1-2-hr_r3i1p1f1_mid-century.epw
  - tmy_palm_springs_regional_airport_kpsp_wrf_mpi-esm1-2-hr_r3i1p1f1_mid-century.tmy

#### Filename Components

| Component | Description | Options |
|--------|-------------|--------|
| `STATION_NAME` | Weather station identifier | e.g., `palm_springs_regional_airport_kpsp` |
| `SOURCE_ID` | Climate model source | e.g., `mpi-esm1-2-hr`, `miroc6`, `ec-earth3`, `taiesm1` |
| `GWL_PERIOD` | Global warming level 30-year period | `present-day` – 1.2°C GWL<br>`near-future` – 1.5°C GWL<br>`mid-century` – 2.0°C GWL<br>`mid-late-century` – 2.5°C GWL |

