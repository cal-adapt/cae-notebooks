"""Currently a big pile of pseudocode for a TMY class"""

from cdf_functions import (
    get_cdf,
    plot_one_var_cdf,
    get_cdf_monthly,
    fs_statistic,
    compute_weighted_fs,
)
from climakitae.core.constants import UNSET
import climakitae as ck
from climakitae.util.utils import (
    convert_to_local_time,
    get_closest_gridcell,
)
from climakitae.core.data_export import write_tmy_file
from climakitae.core.data_interface import get_data
import climakitaegui as ckg

import pandas as pd
import xarray as xr
import numpy as np
import pkg_resources
from tqdm.auto import tqdm  # Progress bar


class TMY:
    """Encapsulate the code needed to generate Typical Meteorological Year (TMY) files.

    Uses WRF hourly data to produce TMYs.
    """
    def __init__(self, stn_lat, stn_lon, start_year, end_year, verbose=False):
        # Set variables
        self.stn_lat = stn_lat
        self.stn_lon = stn_lon
        # TODO: are buffers general?
        self.latitude = (stn_lat - 0.05, stn_lat + 0.05)
        self.longitude = (stn_lon - 0.06, stn_lon + 0.06)
        self.start_year = start_year
        self.end_year = end_year
        self.data_models = [
            "WRF_EC-Earth3_r1i1p1f1",
            "WRF_MPI-ESM1-2-HR_r3i1p1f1",
            "WRF_TaiESM1_r1i1p1f1",
            "WRF_MIROC6_r1i1p1f1",
        ]
        self.stn_name = ""
        self.stn_code = ""
        self.stn_state = ""
        self.scenario = ["Historical Climate", "SSP 3-7.0"]
        self.verbose = verbose
        self.cdf_climatology = UNSET
        self.cdf_monthly = UNSET
        self.weighted_fs_sum = UNSET
        self.top_df = UNSET
        self.all_vars = UNSET

    def _vprint(self,msg):
        if self.verbose:
            print(msg)

    def generate_tmy(self):
        # This runs the whole workflow at once
        print("Running TMY workflow. Expected overall runtime: 40 minutes")
        self.load_all_variables()
        self.get_candidate_months()
        self.run_tmy_analysis()
        self.export_tmy_data_epw()
        return

    def get_tmy_variable(self, varname, units, stats):
        """Run get_data and resampling calls to get data."""
        if self.end_year == 2100:
            print("End year is 2100. The final day in timeseries may be incomplete after data is converted to local time.")
            new_end_year = self.end_year
        else:
            new_end_year = self.end_year + 1
        
        data = get_data(
            variable=varname,
            resolution="9 km",
            timescale="hourly",
            data_type="Gridded",
            units=units,
            latitude=self.latitude,
            longitude=self.longitude,
            area_average="Yes",
            scenario=self.scenario,
            time_slice=(self.start_year, new_end_year),
        )

        data = convert_to_local_time(
            data, self.stn_lon, self.stn_lat
        )  # convert to local timezone, provide lon/lat because area average data lacks coordinates
        data = data.sel({"time": slice(f"{self.start_year}-01-01", f"{self.end_year}-12-31")})
        data = data.sel(simulation=self.data_models)

        returned_data = []

        if "max" in stats:
            # max air temp
            max_data = data.resample(time="1D").max()
            returned_data.append(max_data)

        if "min" in stats:
            # min air temp
            min_data = data.resample(time="1D").min()
            returned_data.append(min_data)

        if "mean" in stats:
            # mean air temp
            mean_data = data.resample(time="1D").mean() 
            returned_data.append(mean_data)

        if "sum" in stats:
            sum_data = data.resample(time="1D").sum()
            returned_data.append(sum_data)

        return returned_data

    def load_all_variables(self):
        self._vprint("Loading variables. Expected runtime: 7 minutes")

        self._vprint("Getting air temperature.")
        airtemp_data = self.get_tmy_variable(
            "Air Temperature at 2m", "degC", ["max", "min", "mean"]
        )
        # unpack and rename
        max_airtemp_data = airtemp_data[0]
        max_airtemp_data.name = "Daily max air temperature"
        min_airtemp_data = airtemp_data[1]
        min_airtemp_data.name = "Daily min air temperature"
        mean_airtemp_data = airtemp_data[2]
        mean_airtemp_data.name = "Daily mean air temperature"

        self._vprint("Getting dew point temperature.")
        # dew point temperature
        dewpt_data = self.get_tmy_variable(
            "Dew point temperature", "degC", ["max", "min", "mean"]
        )
        # unpack and rename
        max_dewpt_data = dewpt_data[0]
        max_dewpt_data.name = "Daily max dewpoint temperature"
        min_dewpt_data = dewpt_data[1]
        min_dewpt_data.name = "Daily min dewpoint temperature"
        mean_dewpt_data = dewpt_data[2]
        mean_dewpt_data.name = "Daily mean dewpoint temperature"

        # wind speed
        self._vprint("Getting wind speed.")
        wndspd_data = self.get_tmy_variable("Wind speed at 10m", "m s-1", ["max", "mean"])
        # unpack and rename
        max_windspd_data = wndspd_data[0]
        max_windspd_data.name = "Daily max wind speed"
        mean_windspd_data = wndspd_data[1]
        mean_windspd_data.name = "Daily mean wind speed"

        # global irradiance
        self._vprint("Getting global irradiance.")
        total_ghi_data = self.get_tmy_variable(
            "Instantaneous downwelling shortwave flux at bottom", "W/m2", ["sum"]
        )
        total_ghi_data = total_ghi_data[0]
        total_ghi_data.name = "Global horizontal irradiance"

        # direct normal irradiance
        self._vprint("Getting direct normal irradiance.")
        total_dni_data = self.get_tmy_variable(
            "Shortwave surface downward direct normal irradiance", "W/m2", ["sum"]
        )
        total_dni_data = total_dni_data[0]
        total_dni_data.name = "Direct normal irradiance"

        self._vprint("Loading all variables into memory.")
        all_vars = xr.merge(
            [
                max_airtemp_data.squeeze(),
                min_airtemp_data.squeeze(),
                mean_airtemp_data.squeeze(),
                max_dewpt_data.squeeze(),
                min_dewpt_data.squeeze(),
                mean_dewpt_data.squeeze(),
                max_windspd_data.squeeze(),
                mean_windspd_data.squeeze(),
                total_ghi_data.squeeze(),
                total_dni_data.squeeze(),
            ]
        )

        # load all indices in
        self.all_vars = all_vars.compute()
        print("All TMY variables loaded.")

    def set_cdf_climatology(self):
        if self.all_vars is UNSET:
            self.load_all_variables()
        self._vprint("Calculating CDF climatology.")
        self.cdf_climatology = get_cdf(self.all_vars)
        return

    def set_cdf_monthly(self):
        if self.all_vars is UNSET:
            self.load_all_variables()
        self._vprint("Calculating monthly CDF.")
        self.cdf_monthly = get_cdf_monthly(self.all_vars)
        # Remove the years for the Pinatubo eruption
        self.cdf_monthly = self.cdf_monthly.where(
            (~self.cdf_monthly.year.isin([1991, 1992, 1993, 1994])), np.nan, drop=True
        )
        return

    def set_weighted_statistic(self):
        if self.cdf_climatology is UNSET:
            self.set_cdf_climatology()
        if self.cdf_monthly is UNSET:
            self.set_cdf_monthly()
        self._vprint("Calculating weighted FS statistic.")
        all_vars_fs = fs_statistic(self.cdf_climatology, self.cdf_monthly)
        weighted_fs = compute_weighted_fs(all_vars_fs)

        # Sum
        self.weighted_fs_sum = (
            weighted_fs.to_array().sum(dim=["variable", "bin_number"]).drop(["data"])
        )
        return

    def set_top_df(self):
        # Pass the weighted F-S sum data for simplicity
        if self.weighted_fs_sum is UNSET:
            self.set_weighted_statistic()
        ds = self.weighted_fs_sum

        df_list = []
        num_values = (
            1  # Selecting the top value for now, persistence statistics calls for top 5
        )
        for sim in ds.simulation.values:
            for mon in ds.month.values:
                da_i = ds.sel(month=mon, simulation=sim)
                top_xr = da_i.sortby(da_i, ascending=True)[:num_values].expand_dims(
                    ["month", "simulation"]
                )
                top_df_i = top_xr.to_dataframe(name="top_values")
                df_list.append(top_df_i)

        # Concatenate list together for all months and simulations
        self.top_df = pd.concat(df_list).drop(columns=["top_values"]).reset_index()
        self._vprint("Top months:")
        self._vprint(self.top_df)
        return

    def get_candidate_months(self):
        """Run CDF functions to get top candidates."""
        self._vprint("Getting top months for TMY. Expected runtime: < 1 min")
        self.set_cdf_climatology()
        self.set_cdf_monthly()
        self.set_weighted_statistic()
        self.set_top_df()
        print("Done.")
        return

    def show_cdf_plot_one_var(self, var="Daily max air temperature"):
        # Make the plot
        if self.cdf_climatology is UNSET:
            self.set_cdf_climatology()
        cdf_plot = plot_one_var_cdf(self.cdf_climatology, var)
        display(cdf_plot)

    def show_cdf_plot_monthly(self, var):
        # Make the plot
        if self.cdf_monthly is UNSET:
            self.set_cdf_monthly()
        # Make the plot
        cdf_plot_mon_yr = plot_one_var_cdf(self.cdf_monthly, var)
        display(cdf_plot_mon_yr)
        return

    def show_top_df(self):
        if self.top_df is UNSET:
            print("Top months not available.")
            print(
                "Please run TMY.generate_tmy() or TMY.get_candidate_months() to generate dataframe."
            )
        print(self.top_df)
        return

    def show_tmy_data_to_export(self, simulation="WRF_MPI-ESM1-2-HR_r3i1p1f1"):
        if self.tmy_data_to_export is UNSET:
            print("No TMY data generated.")
            print("Please run TMY.generate_tmy() to create TMY data for viewing.")
            return

        self.tmy_data_to_export[simulation].plot(
            x="time",
            y=[
                "Air Temperature at 2m",
                "Dew point temperature",
                "Relative humidity",
                "Instantaneous downwelling shortwave flux at bottom",
                "Shortwave surface downward direct normal irradiance",
                "Shortwave surface downward diffuse irradiance",
                "Instantaneous downwelling longwave flux at bottom",
                "Wind speed at 10m",
                "Wind direction at 10m",
                "Surface Pressure",
            ],
            title=f"Typical Meteorological Year ({simulation})",
            subplots=True,
            figsize=(10, 8),
            legend=True,
        )
        return

    def run_tmy_analysis(self):
        """Generate typical meteorological year data
        Output will be a list of dataframes per simulation.
        Print statements throughout the function indicate to the user the progress of the computatioconvert_to_local_time   Parameters
        -----------
        top_df: pd.DataFrame
            Table with column values month, simulation, and year
            Each month-sim-yr combo represents the top candidate that has the lowest weighted sum from the FS statistic
    
        Returns
        --------
        dict of str:pd.DataFrame
            Dictionary in the format of {simulation:TMY corresponding to that simulation}
    
        """
        self._vprint("Generating TMY data to export. Expected runtime: 30 minutes")
    
    
        ## ================== GET DATA FROM CATALOG ==================
        vars_and_units = {
            "Air Temperature at 2m": "degC",
            "Dew point temperature": "degC",
            "Relative humidity": "[0 to 100]",
            "Instantaneous downwelling shortwave flux at bottom": "W/m2",
            "Shortwave surface downward direct normal irradiance": "W/m2",
            "Shortwave surface downward diffuse irradiance": "W/m2",
            "Instantaneous downwelling longwave flux at bottom": "W/m2",
            "Wind speed at 10m": "m s-1",
            "Wind direction at 10m": "degrees",
            "Surface Pressure": "Pa",
        }

        if self.end_year == 2100:
            new_end_year = 2100
        else:
            new_end_year = self.end_year + 1
        
        # Loop through each variable and grab data from catalog
        all_vars_list = []
        print("STEP 1: RETRIEVING HOURLY DATA FROM CATALOG\n")
        for var, units in vars_and_units.items():
            print(f"Retrieving data for {var}", end="... ")
            data_by_var = get_data(
                variable=var,
                resolution="9 km",
                timescale="hourly",
                data_type="Gridded",
                units=units,
                latitude=self.latitude,
                longitude=self.longitude,
                area_average="No",
                scenario=self.scenario,
                time_slice=(self.start_year, self.end_year + 1),
            )
            data_by_var = convert_to_local_time(data_by_var)  # convert to local timezone.
            data_by_var = data_by_var.sel(
                {"time": slice(f"{self.start_year}-01-01", f"{new_end_year}-12-31")}
            )  # get desired time slice in local time
            data_by_var = get_closest_gridcell(
                data_by_var, self.stn_lat, self.stn_lon, print_coords=False
            )  # retrieve only closest gridcell
            data_by_var = data_by_var.sel(
                simulation=self.data_models
            )  # Subset for only the models that have solar variables
    
            # Drop unwanted coords
            data_by_var = data_by_var.squeeze().drop(
                ["lakemask", "landmask", "x", "y", "Lambert_Conformal"]
            )
    
            all_vars_list.append(data_by_var)  # Append to list
            print("complete!")
    
        # Merge data from all variables into a single xr.Dataset object
        all_vars_ds = xr.merge(all_vars_list)
    
        ## ================== CONSTRUCT TMY ==================
        print(
            "\nSTEP 2: CALCULATING TYPICAL METEOROLOGICAL YEAR PER MODEL SIMULATION\nProgress bar shows code looping through each month in the year.\n"
        )
        tmy_df_all = {}
        for sim in all_vars_ds.simulation.values:
            df_list = []
            print(f"Calculating TMY for simulation: {sim}")
            for mon in tqdm(np.arange(1, 13, 1)):
                # Get year corresponding to month and simulation combo
                year = self.top_df.loc[
                    (self.top_df["month"] == mon) & (self.top_df["simulation"] == sim)
                ].year.item()
    
                # Select data for unique month, year, and simulation
                data_at_stn_mon_sim_yr = all_vars_ds.sel(
                    simulation=sim, time=f"{mon}-{year}"
                ).expand_dims("simulation")
    
                # Reformat as dataframe
                df_by_mon_sim_yr = data_at_stn_mon_sim_yr.to_dataframe()
                df_by_mon_sim_yr = df_by_mon_sim_yr.reset_index()
    
                # Reformat time index to remove seconds
                df_by_mon_sim_yr["time"] = pd.to_datetime(
                    df_by_mon_sim_yr["time"].values
                ).strftime("%Y-%m-%d %H:%M")
                df_list.append(df_by_mon_sim_yr)
    
            # Concatenate all DataFrames together
            tmy_df_by_sim = pd.concat(df_list)
            tmy_df_all[sim] = tmy_df_by_sim
    
        self.tmy_data_to_export = tmy_df_all  # Return dict of TMY by simulation

    def export_tmy_data_epw(self):
        self._vprint("Exporting TMY to file.")
        for sim, tmy in self.tmy_data_to_export.items():
            filename = "TMY_{0}_{1}".format(
                stn_name.replace(" ", "_").replace("(", "").replace(")", ""), sim
            ).lower()
            write_tmy_file(
                filename,
                self.tmy_data_to_export[sim],
                (self.start_year, self.end_year),
                self.stn_name,
                self.stn_code,
                self.stn_lat,
                self.stn_lon,
                self.stn_state,
                file_ext="epw",
            )
            if self.verbose:
                print("  Wrote", filename)
        return
