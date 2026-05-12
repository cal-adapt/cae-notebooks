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

stdyr\_[VARIABLE]\_[PERCENTILE]\_[STATION\_NAME]\_[GWL\_PERIOD]\_[DELTA]\_[WINDOW]\_[APPROACH]\_[CENTERED\_YEAR]\_[SCENARIO]\_[BA\_MODELS].[EXTENSION]

Examples:
  stdyr_t2_50ptile_san_diego_lindbergh_field_ksan_30yr_window_time_2020_ssp370.csv
  stdyr_prec_75ptile_35-5N_122-5W_delta_from_historical_10yr_window_time_2016_ssp370_ba_models.csv
  stdyr_rh_derived_50ptile_santa_barbara_municipal_airport_ksba_delta_from_historical_30yr_window_time_2015_ssp370.epw


#### Components
STATION_NAME:   Weather station identifier (e.g., palm_springs_regional_airport_kpsp)

VARIABLE:       Climate variable measured
                - t2 (Air temperature at 2m)
                - rh_derived (Relative humidity)
                - wind_speed_derived (Wind speed)
                - swdnb (Solar radiation / Shortwave downward normal beam radiation)
                - noaa_heat_index_derived (NOAA heat index)

PERCENTILE:     Statistical percentile
                - 05ptile (5th percentile)
                - 50ptile (50th percentile/median)
                - 95ptile (95th percentile)

GWL_PERIOD:     Global warming level 30-year period
                - present-day (1.2 degC GWL)
                - near-future (1.5 degC GWL)
                - mid-century (2.0 degC GWL)
                - mid-late-century (2.5 degC GWL)

DELTA:          Weather or not the difference from baseline (delta) was taken
                - no_delta
                - delta_from_historical

WINDOW:          Years around the year in which the input Global Warming Level is reached. 
                - ex: 30yr (corresponding to a 15yr window size)
APPROACH:        The climate profile approach that was used.
                - time
                - warming_level

CENTERED_YEAR:  For approach="Time", the year for which to find a corresponding warming level
                - ex: 2015

SCENARIO       SSP scenario from ["SSP 3-7.0", "SSP 2-4.5","SSP 5-8.5"]
                - ssp245
                - ssp370
                - ssp585

BA_MODELS     Option to return only bias-adjusted models. Default = False
                - ba_models 

### Typical Meteorological Year

tmy\_[STATION\_NAME]\_[ACTIVITY\_ID]\_[SOURCE\_ID]\_[MEMBER\_ID]\_[GWL\_PERIOD].[EXTENSION]

Examples:

  tmy_palm_springs_regional_airport_kpsp_wrf_mpi-esm1-2-hr_r3i1p1f1_mid-century.csv

  tmy_palm_springs_regional_airport_kpsp_wrf_mpi-esm1-2-hr_r3i1p1f1_mid-century.epw

  tmy_palm_springs_regional_airport_kpsp_wrf_mpi-esm1-2-hr_r3i1p1f1_mid-century.tmy

#### Components

STATION_NAME:   Weather station identifier (e.g., palm_springs_regional_airport_kpsp)

SOURCE_ID:      Climate model source (e.g., mpi-esm1-2-hr, miroc6, ec-earth3, taiesm1)

GWL_PERIOD:     Global warming level 30-year period

                - present-day (1.2 degC GWL)

                - near-future (1.5 degC GWL)

                - mid-century (2.0 degC GWL)

                - mid-late-century (2.5 degC GWL)

Additional metadata in filenames and intake catalog:

ACTIVITY_ID:    Downscaling method 

MEMBER_ID:      Ensemble member identifier (e.g., r1i1p1f1, r3i1p1f1)

