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

import warnings
warnings.filterwarnings("ignore")

def compute_cdf(da):
    """Compute the cumulative density function for an input DataArray"""
    da_np = da.values  # Get numpy array of values
    num_samples = 1024  # Number of samples to generate
    count, bins_count = np.histogram(  # Create a numpy histogram of the values
        da_np,
        bins=np.linspace(
            da_np.min(),  # Start at the minimum value of the array
            da_np.max(),  # End at the maximum value of the array
            num_samples,
        ),
    )
    cdf_np = np.cumsum(count / sum(count))  # Compute the CDF

    # Turn the CDF array into xarray DataArray
    # New dimension is the bin values
    cdf_da = xr.DataArray(
        [bins_count[1:], cdf_np],
        dims=["data", "bin_number"],
        coords={
            "data": ["bins", "probability"],
        },
    )
    cdf_da.name = da.name
    return cdf_da


def get_cdf_by_sim(da):
    # Group the DataArray by simulation
    return da.groupby("simulation").apply(compute_cdf)


def get_cdf_by_mon_and_sim(da):
    # Group the DataArray by month in the year
    return da.groupby("time.month").apply(get_cdf_by_sim)


def get_cdf(ds):
    """Get the cumulative density function.

    Parameters
    -----------
    ds: xr.Dataset
        Input data to compute CDF for
    Returns
    -------
    xr.Dataset
    """
    return ds.apply(get_cdf_by_mon_and_sim)

def get_cdf_monthly(ds):
    """Get the cumulative density function by unique mon-yr combos

    Parameters
    -----------
    ds: xr.Dataset
        Input data to compute CDF for
    Returns
    -------
    xr.Dataset
    """

    def get_cdf_mon_yr(da):
        return da.groupby("time.year").apply(get_cdf_by_mon_and_sim)

    return ds.apply(get_cdf_mon_yr)

def fs_statistic(cdf_climatology, cdf_monthly):
    """
    Calculates the Finkelstein-Schafer statistic:
    Absolute difference between long-term climatology and candidate CDF, divided by number of days in month
    """
    days_per_mon = xr.DataArray(
        data=[31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
        coords={"month": np.arange(1, 13)},
    )
    fs_stat = abs(cdf_monthly - cdf_climatology).sel(data="probability") / days_per_mon
    return fs_stat

def compute_weighted_fs(da_fs):
    """Weights the F-S statistics based on TMY3 methodology"""
    weights_per_var = {
        "Daily max air temperature": 1 / 20,
        "Daily min air temperature": 1 / 20,
        "Daily mean air temperature": 2 / 20,
        "Daily max dewpoint temperature": 1 / 20,
        "Daily min dewpoint temperature": 1 / 20,
        "Daily mean dewpoint temperature": 2 / 20,
        "Daily max wind speed": 1 / 20,
        "Daily mean wind speed": 1 / 20,
        "Global horizontal irradiance": 5 / 20,
        "Direct normal irradiance": 5 / 20,
    }

    for var, weight in weights_per_var.items():
        # Multiply each variable by it's appropriate weight
        da_fs[var] = da_fs[var] * weight
    return da_fs


def plot_one_var_cdf(cdf_da ,var):
    """Plot CDF for a single variable
    Written to function for the unique configuration of the CDF DataArray object
    Silences an annoying hvplot warning
    Will show every simulation together on the plot

    Parameters
    -----------
    cdf: xr.DataArray
       Cumulative density function for a single variable

    Returns
    -------
    panel.layout.base.Column
        Hvplot lineplot

    """
    cdf_da = cdf_da[var]
    prob_da = cdf_da.sel(data="probability", drop=True).rename(
        "probability"
    )  # Grab only probability da
    bins_da = cdf_da.sel(data="bins", drop=True).rename("bins")  # Grab just bin values
    ds = xr.merge([prob_da, bins_da])  # Merge the two to form a single Dataset object
    cdf_pl = ds.hvplot(
        "bins",
        "probability",
        by="simulation",  # Simulations should all be displayed together
        widget_location="bottom",
        grid=True,
        xlabel="{0} ({1})".format(var, cdf_da.attrs["units"]),
        xlim=(
            bins_da.min().item(),
            bins_da.max().item(),
        ),  # Fix the x-limits for all months
        ylabel="Probability (0-1)",
    )
    return cdf_pl