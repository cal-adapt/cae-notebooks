"""Currently a big pile of pseudocode for a TMY class"""

from cdf_functions import (
    get_cdf,
    plot_one_var_cdf,
    get_cdf_monthly,
    fs_statistic,
    compute_weighted_fs,
    generate_tmy_data,
)
from climakitae.core.constants import UNSET


class TMY:
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
        self.scenario = ["Historical Climate", "SSP 3-7.0"]
        self.verbose = verbose
        self.cdf_climatology = UNSET
        self.cdf_monthly = UNSET
        self.weighted_fs_sum = UNSET
        self.top_df = UNSET
        self.all_vars = UNSET

    def generate_tmy(self):
        # This runs the whole workflow at once
        self.load_all_variables()
        self.get_candidate_months()
        self.run_tmy_analysis()
        self.export_tmy_data_epw()
        return

    def get_tmy_variable(self, varname, units, stats):
        """Run get_data and resampling calls to get data."""
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
            time_slice=(self.start_year, self.end_year + 1),
        )  # TODO: handle if end_year is 2011

        data = convert_to_local_time(
            data, stn_lon, stn_lat
        )  # convert to local timezone, provide lon/lat because area average data lacks coordinates
        data = data.sel({"time": slice(f"{start_year}-01-01", f"{end_year}-12-31")})
        data = data.sel(simulation=data_models)

        returned_data = []

        if "max" in stats:
            # max air temp
            max_data = data.resample(time="1D").max()  # daily max air temp
            returned_data.append(max_data)

        if "min" in stats:
            # min air temp
            min_data = data.resample(time="1D").min()  # daily min air temp
            returned_data.append(min_data)

        if "mean" in stats:
            # mean air temp
            mean_data = data.resample(time="1D").mean()  # daily mean air temp
            returned_data.append(mean_data)

        if "sum" in stats:
            sum_data = data.resample(time="1D").sum()
            returned_data.append(sum_data)

        return returned_data

    def load_all_variables(self):
        print("Loading variables")
        print("Expected runtime:")

        if self.verbose:
            print("Getting air temperature.")
        airtemp_data = load_tmy_variable(
            "Air Temperature at 2m", "degC", ["max", "min", "mean"]
        )
        # unpack and rename
        max_airtemp_data = airtemp_data[0]
        max_airtemp_data.name = "Daily max air temperature"
        min_airtemp_data = airtemp_data[1]
        min_airtemp_data.name = "Daily min air temperature"
        mean_airtemp_data = airtemp_data[2]
        mean_airtemp_data.name = "Daily mean air temperature"

        if self.verbose:
            print("Getting dew point temperature.")
        # dew point temperature
        dewpt_data = load_tmy_variable(
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
        if self.verbose:
            print("Getting wind speed.")
        wndspd_data = load_tmy_variable("Wind speed at 10m", "m s-1", ["max", "mean"])
        # unpack and rename
        max_windspd_data = wndspd_data[0]
        max_windspd_data.name = "Daily max wind speed"
        mean_wndspd_data = wndspd_data[1]
        mean_wndspd_data.name = "Daily mean wind speed"

        # global irradiance
        if self.verbose:
            print("Getting global Irradiance.")
        total_ghi_data = load_tmy_variable(
            "Instantaneous downwelling shortwave flux at bottom", ["sum"]
        )
        total_ghi_data = total_ghi_data[0]
        total_ghi_data.name = "Global horizontal irradiance"

        # direct normal irradiance
        if self.verbose:
            print("Getting direct normal irradiance.")
        total_dni_data = load_tmy_variable(
            "Shortwave surface downward direct normal irradiance", ["sum"]
        )
        total_dni_data = total_dni_data[0]
        total_dni_data.name = "Direct normal irradiance"

        if self.verbose:
            print("Loading all variables into memory.")
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
        if self.all_vars == UNSET:
            self.load_all_variables()
        if self.verbose:
            print("Calculating CDF climatology.")
        self.cdf_climatology = get_cdf(self.all_vars)
        return

    def set_cdf_monthly(self):
        if all_vars == UNSET:
            self.load_all_variables()
        if self.verbose:
            print("Calculating monthly CDF.")
        self.cdf_monthly = get_cdf_monthly(self.all_vars)
        # Remove the years for the Pinatubo eruption
        self.cdf_monthly = self.cdf_monthly.where(
            (~cdf_monthly.year.isin([1991, 1992, 1993, 1994])), np.nan, drop=True
        )
        return

    def set_weighted_statistic(self):
        if self.cdf_climatology == UNSET:
            self.set_cdf_climatology()
        if self.cdf_monthly == UNSET:
            self.set_cdf_monthly()
        if self.verbose:
            print("Calculating weighted FS statistic.")
        all_vars_fs = fs_statistic(self.cdf_climatology, self.cdf_monthly)
        weighted_fs = compute_weighted_fs(all_vars_fs)

        # Sum
        self.weighted_fs_sum = (
            weighted_fs.to_array().sum(dim=["variable", "bin_number"]).drop(["data"])
        )
        return

    def set_top_df(self):
        # Pass the weighted F-S sum data for simplicity
        if self.weighted_fs_sum == UNSET:
            self.set_weighted_statistic()
        ds = self.weighted_fs_sum()

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
        if self.verbose:
            print("Top months:")
            print(self.top_df)
        return

    def get_candidate_months(self):
        """Run CDF functions to get top candidates."""
        print("Getting top months for TMY.")
        self.set_cdf_climatology()
        self.set_cdf_monthly()
        self.set_weighted_statistic()
        self.set_top_df()
        print("Done.")
        return

    def show_cdf_plot_one_var(self, var="Daily max air temperature"):
        # Make the plot
        if self.cdf_climatology == UNSET:
            self.set_cdf_climatology()
        cdf_plot = plot_one_var_cdf(self.cdf_climatology, var)
        display(cdf_plot)

    def show_cdf_plot_monthly(self, var):
        # Make the plot
        if self.cdf_monthly == UNSET:
            self.set_cdf_monthly()
        # Make the plot
        cdf_plot_mon_yr = plot_one_var_cdf(self.cdf_monthly, var)
        display(cdf_plot_mon_yr)
        return

    def show_top_df(self):
        if top_df == UNSET:
            print("Top months not available.")
            print(
                "Please run TMY.generate_tmy() or TMY.get_candidate_months() to generate dataframe."
            )
        print(top_df)
        return

    def show_tmy_data_to_export(self, simulation="WRF_MPI-ESM1-2-HR_r3i1p1f1"):
        if self.tmy_data_to_export == UNSET:
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
        # TODO: Wrap or rewrite this function
        if self.verbose:
            print("Generating TMY data to export.")
        return generate_tmy_data(
            top_df,
            latitude,
            longitude,
            stn_lat,
            stn_lon,
            start_year,
            end_year,
            data_models,
        )

    def export_tmy_data_epw(self):
        if self.verbose:
            print("Exporting TMY to file.")
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
