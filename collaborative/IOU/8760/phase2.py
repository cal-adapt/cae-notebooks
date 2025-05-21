import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import os
from typing import List, Union, Tuple
from datetime import timedelta
from climakitae.explore.vulnerability import cava_data
from climakitae.util.utils import add_dummy_time_to_wl
from climakitae.explore.threshold_tools import (
    get_block_maxima,
    get_return_value,
)

random.seed(42)


def plot_retvals(calc_data: xr.Dataset, time_axis: bool = False) -> None:
    """
    Plot return values (or Julian day equivalents) from a calculation dataset.

    Args:
        calc_data (xr.Dataset):
            An xarray Dataset containing dimensions including 'one_in_x',
            'simulation', and 'location', with return value data.
        time_axis (bool, optional):
            If True, divides the data by 24 to plot in Julian days.
            If False, plots the raw return values. Defaults to False.

    Returns:
        None
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    for j, xval in enumerate(calc_data.one_in_x.values):
        idx = np.where(calc_data["one_in_x"].values == xval)[0].item()
        data = calc_data.isel(one_in_x=idx)

        if time_axis:
            data = data / 24

        # Let xarray create the legend only on the first plot
        data.plot.line(ax=ax[j], hue="location", add_legend=(j == 0))

        labels = [sim.item().split("_")[1] for sim in calc_data.simulation.values]
        ax[j].set_xticks(range(len(labels)))
        ax[j].set_xticklabels(labels, rotation=45, ha="right")

        ax[j].set_title(f"1-in-{xval} by GCM")
        ax[j].set_xlabel("GCM")

        if j == 0:
            if time_axis:
                ax[j].set_ylabel("Median Julian Day")
            else:
                ax[j].set_ylabel("Max Return Value")
        else:
            ax[j].set_ylabel("")

    plt.tight_layout()
    plt.show()


def plot_med_val_by_locs(calc_data: xr.Dataset, time_axis: bool = False) -> None:
    """
    Plot median return values by location for each return period.

    Args:
        calc_data (xr.Dataset):
            An xarray Dataset containing dimensions 'simulation', 'one_in_x', and 'location'.
            Expected to have return value data across simulations and locations.
        time_axis (bool, optional):
            If True, converts values from hours to Julian days by dividing by 24.
            If False, plots the raw return values. Defaults to False.

    Returns:
        None; just generates the figure
    """
    # Median return values
    med = calc_data.median(dim="simulation")

    one_in_x_vals = med["one_in_x"].values
    locations = med["location"].values

    n_groups = len(one_in_x_vals)
    n_locs = len(locations)

    bar_width = 0.1  # smaller width = visible space between bars
    group_width = n_locs * bar_width
    x = np.linspace(0, 0.5, n_groups)  # base x for each group

    fig, ax = plt.subplots(figsize=(6, 5))

    for i, loc in enumerate(locations):
        y = med.sel(location=loc).values
        if time_axis:
            y = y / 24
        offset = (i - (n_locs - 1) / 2) * bar_width  # center the group
        bar_positions = x + offset
        bars = ax.bar(bar_positions, y, width=bar_width - 0.01, label=str(loc))

        # Add text labels above each bar
        for xpos, height in zip(bar_positions, y):
            ax.text(
                xpos,
                height + 0.5,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ymax = med.max().item()
    if time_axis:
        ymax = ymax / 24

    ax.set_ylim(top=ymax * 1.1)

    # Center group labels
    ax.set_xticks(x)
    ax.set_xticklabels([f"1-in-{v}" for v in one_in_x_vals])

    # Labels and formatting
    ax.set_xlabel("Return Period")
    if time_axis:
        ax.set_ylabel("Median Julian Day")
    else:
        ax.set_ylabel("1-in-X Max Value")
    ax.set_title("Median Return Values by Location")
    ax.legend(title="Location", loc="upper right", bbox_to_anchor=(1.35, 1))
    plt.show()


def clean_raw_data(raw_data: List[xr.Dataset], loc_names: List[str]) -> xr.Dataset:
    """
    Combines raw datasets together by new `location` dimension, creates a dummy time axis, and creates an hour-of-year coordinate.

    Args:
        raw_data (List[xr.Dataset]):
            A list of xarray Datasets, one for each location.
        loc_names (List[str]):
            A list of location names corresponding to the datasets.

    Returns:
        xr.Dataset:
            A single combined xarray Dataset with cleaned time and added 'hour_of_year' coordinate.
    """
    # raw_data, calc_data = cava_data_retval['raw_data'], cava_data_retval['calc_data']
    total_raw = xr.concat(raw_data, dim="location").assign_coords(location=loc_names)
    # total_calc = xr.concat(calc_data, dim='location').assign_coords(location=loc_names)
    total_raw = add_dummy_time_to_wl(total_raw)

    # Making new time axis without leap days
    total_raw["time"] = xr.cftime_range(
        start=f"{total_raw.time.dt.year[0].item()}-01-01",
        periods=total_raw.sizes["time"],
        freq="H",
        calendar="noleap",
    )

    # Make new dimension for `hour_of_year`
    hour_of_year = (total_raw["time"].dt.dayofyear - 1) * 24 + total_raw["time"].dt.hour
    total_raw = total_raw.assign_coords(hour_of_year=hour_of_year)

    return total_raw


def get_one_in_x(
    da: xr.DataArray,
    one_in_x: Union[int, float],
    event_duration: Tuple[int, str],
    distr: str,
) -> xr.DataArray:
    """
    Calculate the 1-in-X year return value for a given dataset and event duration.

    Args:
        da (xr.DataArray):
            Input xarray DataArray containing the data to analyze.
        one_in_x (Union[int, float]):
            Return period (e.g., 10 for 1-in-10 years) to calculate.
        event_duration (Tuple[int, str]):
            Duration grouping for block maxima, given as (amount, unit)
            e.g., (1, 'hour'), (3, 'day').
        distr (str):
            Name of the statistical distribution to fit (e.g., 'gev', 'gumbel').

    Returns:
        xr.DataArray:
            DataArray of the 1-in-X return values calculated across simulations or points.
    """
    ams = get_block_maxima(
        da.squeeze(),
        extremes_type="max",
        groupby=event_duration,
        check_ess=False,
    ).squeeze()

    return get_return_value(
        ams,
        return_period=one_in_x,
        multiple_points=False,
        distr=distr,
    )


def find_med_hrs(raw_data: xr.Dataset, all_one_in_x: xr.Dataset) -> xr.DataArray:
    """
    Finds the median hour-of-year of times with temperatures that fall between confidence interval bounds.

    Args:
        raw_data (xr.Dataset):
            Raw climate or observational data with dimensions including 'time', 'location', and 'simulation'.
        all_one_in_x (xr.Dataset):
            Dataset containing confidence interval bounds ('conf_int_lower_limit' and 'conf_int_upper_limit')
            for each 'one_in_x' return period.

    Returns:
        xr.DataArray:
            Median hour-of-year for each location, simulation, and return period ('one_in_x').
    """
    subset = raw_data.resample(time="1D").max()

    # Removing leap days
    subset["time"] = xr.cftime_range(
        start=f"{subset.time.dt.year[0].item()}-01-01",
        periods=subset.sizes["time"],
        freq="D",
        calendar="noleap",
    )

    tmp = subset.where(
        (subset > all_one_in_x["conf_int_lower_limit"])
        & (subset < all_one_in_x["conf_int_upper_limit"])
    )

    hour_of_year = (tmp["time"].dt.dayofyear - 1) * 24 + tmp["time"].dt.hour
    tmp = tmp.assign_coords(hour_of_year=hour_of_year)

    med_hrs = tmp.groupby(["location", "simulation", "one_in_x"]).apply(
        lambda x: x.dropna(dim="time").hour_of_year.median()
    )

    med_hrs = med_hrs.assign_coords(simulation=med_hrs["simulation"].astype(str))
    med_hrs = med_hrs.assign_coords(location=med_hrs["location"].astype(str))

    # Re-order `med_hrs` to have the same location order as `raw_data` input
    med_hrs = med_hrs.reindex(location=raw_data.location.values)

    return med_hrs


def insert_at_hrs(
    median8760: xr.DataArray, med_hr: xr.DataArray, val: xr.DataArray, window: int = 1
) -> xr.DataArray:
    """
    Insert values into a DataArray at specific hours, with optional window.

    Args:
        median8760 (xr.DataArray):
            Array of baseline values with dimensions ('location', 'hour_of_year').
        med_hr (xr.DataArray):
            Target hours to modify, with dimensions ('location', 'one_in_x').
        val (xr.DataArray):
            Values to insert at the target hours, with dimensions ('location', 'one_in_x').
        window (int, optional):
            Number of hours before and after each med_hr to also replace.
            Defaults to 1.

    Returns:
        xr.DataArray:
            Mutated DataArray with dimensions ('location', 'one_in_x', 'hour_of_year').
    """
    broadcasted = xr.broadcast(val, median8760)[
        1
    ]  # shape: (location, one_in_x, hour_of_year)

    result = broadcasted.copy()
    for offset in range(-window, window + 1):
        target_hr = med_hr + offset  # shape: (location, one_in_x)
        mask = broadcasted["hour_of_year"] == target_hr
        result = xr.where(mask, val, result)

    return result


# ### Grabbing appropriate datasets


def retrieve_data(locs, latlons, files, label):
    def files_exist(file_list):
        return all(os.path.exists(f) for f in file_list)

    def try_load():
        print(f"Attempting to load saved {label} files...")
        data = (
            xr.open_mfdataset(files, concat_dim="location", combine="nested")
            .to_array()
            .squeeze("variable")
        )
        data.attrs["frequency"] = "hourly"
        return data.assign_coords({"location": locs}).compute()

    def fetch_with_cava():
        print(f"Using `cava_data` to fetch {label} data instead...")
        data = cava_data(
            latlons,
            variable="Air Temperature at 2m",
            units="degF",
            downscaling_method="Dynamical",
            approach="Warming Level",
            warming_level=2.0,
            wrf_bias_adjust=False,
            metric_calc="max",
            one_in_x=[10, 100],
            event_duration=(1, "day"),
            export_method="raw",
            file_format="NetCDF",
        )
        return data

    try:
        if files_exist(files):
            return clean_raw_data(try_load(), locs)
        else:
            raise FileNotFoundError("One or more files missing.")
    except Exception as e:
        print(f"Falling back to cava_data for {label} due to error: {e}")

    data = fetch_with_cava()

    # Cleaning the raw data from `cava_data`
    return clean_raw_data(data, locs)


def make_clean_daily(raw_data: xr.DataArray) -> xr.DataArray:
    """
    Resamples hourly data to daily maximum values and assigns a 'hour_of_year' coordinate
    based on the resulting no-leap calendar day timestamps.

    Parameters:
        raw_data (xr.DataArray): Hourly data with a 'time' dimension.

    Returns:
        xr.DataArray: Daily max data with updated 'time' and 'hour_of_year' coordinates.
    """
    # Resample to daily max
    daily = raw_data.resample(time="1D").max()

    # Reset time to a clean noleap calendar range starting Jan 1 of first year
    start_year = daily.time.dt.year[0].item()
    daily["time"] = xr.cftime_range(
        start=f"{start_year}-01-01",
        periods=daily.sizes["time"],
        freq="D",
        calendar="noleap",
    )

    # Assign hour_of_year (since it's daily, this will be 0, 24, 48, ...)
    # hour_of_year = (daily['time'].dt.dayofyear - 1) * 24
    hour_of_year = (daily["time"].dt.dayofyear - 1) * 24 + daily["time"].dt.hour
    daily = daily.assign_coords(hour_of_year=hour_of_year)

    return daily


def combine_ds(daily_da: xr.DataArray, one_in_x_da: xr.Dataset) -> xr.Dataset:
    """
    Combines a daily DataArray with confidence interval bounds from a dataset into a single Dataset.

    Parameters:
        daily_da (xr.DataArray): Daily values (e.g. temperature or metric output).
        one_in_x_da (xr.Dataset): Dataset containing 'conf_int_lower_limit' and 'conf_int_upper_limit'.

    Returns:
        xr.Dataset: Dataset with three aligned DataArrays: 'vals', 'lower', and 'upper'.
    """
    # Align daily data with confidence bounds
    lower = xr.broadcast(daily_da, one_in_x_da["conf_int_lower_limit"])[1]
    upper = xr.broadcast(daily_da, one_in_x_da["conf_int_upper_limit"])[1]

    # Combine into a new Dataset
    combined = xr.Dataset({"vals": daily_da, "lower": lower, "upper": upper})

    return combined


def find_valid_times(timeseries: xr.Dataset) -> np.ndarray:
    """
    Finds all times where the 'vals' are strictly between the 'lower' and 'upper' bounds.

    Parameters:
        timeseries (xr.Dataset): Dataset containing 'vals', 'lower', and 'upper' variables with a 'time' dimension.
        t (int): Currently unused, but could represent window size or context for future filtering.

    Returns:
        np.ndarray: Array of time values where the condition is met.
    """
    # Identify time points where 'vals' fall between 'lower' and 'upper'
    valid_mask = (timeseries["vals"] > timeseries["lower"]) & (
        timeseries["vals"] < timeseries["upper"]
    )

    # Extract valid time values
    valid_times = timeseries["time"].where(valid_mask).dropna("time").time.values

    return valid_times


def gather_valid_times(ds: xr.Dataset) -> pd.DataFrame:
    """
    Applies `find_valid_times` to each (location, simulation, one_in_x) group in the dataset,
    and returns a DataFrame of valid times per group.

    Parameters:
        ds (xr.Dataset): Input dataset with dimensions including 'location', 'simulation', 'one_in_x', and 'time'.
        t (int): Window size passed to `find_valid_times`.

    Returns:
        pd.DataFrame: A DataFrame with columns: location, simulation, one_in_x, and valid_times (as np.ndarray).
    """
    results = []
    grouped = ds.squeeze().groupby(["location", "simulation", "one_in_x"])

    for (loc, sim, oneinx), subset in grouped:
        valid = find_valid_times(subset)
        results.append(
            {
                "location": loc,
                "simulation": sim,
                "one_in_x": oneinx,
                "valid_times": valid,
            }
        )

    return pd.DataFrame(results)


def extract_event_windows(
    timeseries: xr.DataArray, df: pd.DataFrame, t: int
) -> xr.DataArray:
    """
    Extracts non-overlapping time windows of length (2t + 1) days around valid event timestamps,
    for a specific (location, simulation, one_in_x) combination.

    Parameters:
        timeseries (xr.DataArray): Time series with dimensions including 'time'.
        df (pd.DataFrame): DataFrame containing valid event timestamps in 'valid_times'.
        t (int): Half-window size in days (default is 3, for a 7-day window).

    Returns:
        xr.DataArray: Median of all extracted windows with time replaced by relative indices.
    """
    # Filter dataframe for matching metadata
    df_subset = df[
        (df["location"] == timeseries.location.item())
        & (df["simulation"] == timeseries.simulation.item())
        & (df["one_in_x"] == timeseries.one_in_x.item())
    ]

    # Get valid timestamps
    event_times = np.atleast_1d(df_subset.valid_times.item())

    # Define time slices: [t days before, t days after] each event
    time_slices = [
        slice(dt - timedelta(days=t), dt + timedelta(days=t + 1) - timedelta(hours=1))
        for dt in event_times
    ]

    # Extract windows and normalize time axis
    windows = [
        timeseries.sel(time=ts).assign_coords(
            time=np.arange(timeseries.sel(time=ts).sizes["time"])
        )
        for ts in time_slices
    ]

    # Take median across all extracted windows
    return xr.concat(windows, dim="window").median(dim="window")


def generate_data_to_insert(
    times_to_insert_at: xr.DataArray,
    clean_raw_data: xr.DataArray,
    metadata_df: pd.DataFrame,
    t: int,
) -> xr.DataArray:
    """
    Extracts event-based time windows from raw data using group-wise metadata and computes
    the median pattern to insert for each (location, one_in_x) across simulations.

    Parameters:
        times_to_insert_at (xr.DataArray): Time DataArray with shape matching metadata (used to align with raw data).
        clean_raw_data (xr.DataArray): Full daily/hourly input data.
        metadata_df (pd.DataFrame): DataFrame containing valid times per group.
        t (int): Half-window size in days for extraction (default: 3).

    Returns:
        xr.DataArray: Median event response per group, collapsed across simulations.
    """
    # Broadcast clean raw data to shape of insertion times
    total_da = xr.broadcast(times_to_insert_at, clean_raw_data)[1]

    # Extract and reduce across simulations
    to_insert = (
        total_da.squeeze()
        .groupby(["location", "simulation", "one_in_x"])
        .apply(lambda timeseries: extract_event_windows(timeseries, metadata_df, t))
        .median(dim="simulation")
    )
    return to_insert


def insert_data(
    to_insert: xr.DataArray, center_time: int, orig_data: xr.DataArray, t: int
) -> xr.DataArray:
    """
    Inserts a short time window (`to_insert`) into a copy of the original data array,
    centered at a specified hour index.

    Parameters:
        to_insert (xr.DataArray): Data to insert (length should be 24 * (2t + 1)).
        center_time (int): Center index (in hours) for the insertion.
        orig_data (xr.DataArray): The full-length time series to modify.
        t (int): Half-window size in days.

    Returns:
        xr.DataArray: Copy of `orig_data` with `to_insert` injected into the specified time window.
    """
    start = int(center_time) - 24 * t
    end = int(center_time) + 24 * (t + 1)

    result = orig_data.copy()
    result[start:end] = to_insert

    return result


def plot_modified8760s(modified8760: xr.DataArray, figsize: tuple = (15, 8)) -> None:
    """
    Plot modified 8760-hour data with two rows (one_in_x) and three columns (locations).

    Args:
        modified8760 (xr.DataArray):
            DataArray with dimensions ('hour_of_year', 'location', 'one_in_x').
        figsize (tuple, optional):
            Size of the entire figure. Defaults to (15, 8).

    Returns:
        None
    """
    plot = modified8760.plot.line(
        x="hour_of_year",
        row="one_in_x",
        col="location",
        sharey=False,
        aspect=2,
        figsize=figsize,
    )

    # Adjust titles for each column
    for ax, loc in zip(plot.axs[0], modified8760.location.values):
        ax.set_title(loc, fontsize=16)

    # Adjust y-axis labels
    for row_i, one_in_x_val in enumerate(modified8760.one_in_x.values):
        for col_i, ax in enumerate(plot.axs[row_i]):
            if col_i == 0:
                # First column only: bold "1-in-X" label horizontally
                ax.set_ylabel(
                    rf"$\bf{{1\text{{-}}in\text{{-}}{int(one_in_x_val)}}}$"
                    + "\n"
                    + "degF",
                    fontsize=16,
                    rotation=0,
                    labelpad=40,
                    ha="center",
                    va="center",
                )

    # plt.xlim(left=4000, right=6000)
    plt.tight_layout()
    plt.show()


def create_modified_8760(ds, one_in_x_vals, t, custom_times=None):
    """
    Create a modified 8760 timeseries with 1-in-X event data inserted into the median profile.

    Parameters:
        ds (xr.Dataset): Original hourly dataset with 'simulation' dimension.
        one_in_x_vals (xr.DataArray): Return value data for insertion.
        t (int): Duration (in days) of the event to insert.

    Returns:
        xr.DataArray: Modified 8760 profile with events inserted.
    """
    # Prepare daily and median 8760 data
    daily_ds = combine_ds(make_clean_daily(ds), one_in_x_vals)
    df = gather_valid_times(daily_ds)
    median_8760 = ds.groupby("hour_of_year").median().median(dim="simulation").squeeze()

    # Identify event insertion times
    if isinstance(custom_times, xr.DataArray):
        insert_times = xr.where(
            ~xr.ufuncs.isnan(custom_times),
            custom_times,
            find_med_hrs(ds, one_in_x_vals).median(dim="simulation"),
        )
    else:
        insert_times = find_med_hrs(ds, one_in_x_vals).median(dim="simulation")

    # Generate data to insert into the 8760
    insert_vals = generate_data_to_insert(insert_times, ds, df, t)

    # Broadcast median 8760 to include 1-in-X dimension
    base_8760 = xr.broadcast(insert_times, median_8760)[1]

    # Apply insertions
    modified_8760 = xr.apply_ufunc(
        insert_data,
        insert_vals,
        insert_times,
        base_8760,
        input_core_dims=[["time"], [], ["hour_of_year"]],
        output_core_dims=[["hour_of_year"]],
        kwargs={"t": t},
        vectorize=True,
    )

    return modified_8760


def create_empty_da(
    da: xr.DataArray, keep_dims: list, fill_value=np.nan
) -> xr.DataArray:
    """
    Create an empty DataArray with only the specified dimensions from the original DataArray.

    Parameters:
    - da: xarray.DataArray, the original array
    - keep_dims: tuple of dimension names to keep (default: ("location", "one_in_x"))
    - fill_value: value to initialize the new DataArray with (default: np.nan)

    Returns:
    - xr.DataArray with shape (len(da[dim]) for dim in keep_dims)
    """
    coords = {dim: da[dim] for dim in keep_dims}
    shape = tuple(len(coords[dim]) for dim in keep_dims)

    return xr.DataArray(
        np.full(shape, fill_value),
        dims=keep_dims,
        coords=coords,
        name="custom_1_in_x_times",
    )


def set_custom_times(da: xr.DataArray, updates) -> xr.DataArray:
    """
    Set values in a DataArray either by applying a single scalar value to all elements,
    or using a list of coordinate-value update dictionaries.

    Parameters:
    - da: xarray.DataArray
        The array to modify.
    - updates: scalar or list of dicts
        - If scalar: fills the entire DataArray with this value.
        - If list of dicts: each dict should specify coordinate labels and a 'value' key.
          Example: {'location': 'goleta', 'one_in_x': 10, 'value': 3000}

    Returns:
    - xr.DataArray
        A modified copy with updated values.
    """
    da_mod = da.copy(deep=True)

    if np.isscalar(updates):
        da_mod.data = np.full_like(da_mod, updates)
    elif isinstance(updates, list):
        for entry in updates:
            entry = entry.copy()
            value = entry.pop("value")
            da_mod.loc[entry] = value
    else:
        raise ValueError(
            "`updates` must be either a scalar or a list of coordinate-value dicts."
        )

    return da_mod
