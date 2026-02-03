"""This script contains functions used to calculate the geostrophic wind
in the santa_ana_metrics notebook. These functions will eventually be
moved to climakitae."""

import numpy as np
import pyproj
import xarray as xr
from climakitae.tools.derived_variables import compute_wind_dir
from climakitae.util.utils import add_dummy_time_to_wl, get_closest_gridcell
from pyproj import CRS, Geod, Proj


def _wrf_deltas(h: xr.DataArray) -> tuple[xr.DataArray]:
    """Get the actual x and y spacing in meters.

    Find the distance between lat/lon points on a great circle. Assumes a
    spherical geoid. The returned deltas are assigned the coordinates of
    the terminus point of the delta.

    Parameters
    ----------
    h : xr.DataArray
        DataArray with x and y dimensions on WRF grid

    Returns
    -------
    Tuple[xr.DataArray]
        X and Y direction deltas.
    """
    g = Geod(ellps="sphere")
    forward_az, _, dy = g.inv(
        h.lon[0:-1, :], h.lat[0:-1, :], h.lon[1:, :], h.lat[1:, :]
    )
    dy[(forward_az < -90.0) | (forward_az > 90.0)] *= -1

    forward_az, _, dx = g.inv(
        h.lon[:, 0:-1], h.lat[:, 0:-1], h.lon[:, 1:], h.lat[:, 1:]
    )
    dx[(forward_az < -90.0) | (forward_az > 90.0)] *= -1
    # Convert to data array with coordinates of terminus point
    dx = xr.DataArray(
        data=dx,
        dims=["y", "x"],
        coords={
            "y": (["y"], h.y.data),
            "x": (["x"], h.x.data[1:]),
            "lon": (["y", "x"], h.lon.data[:, 1:]),
            "lat": (["y", "x"], h.lat.data[:, 1:]),
        },
    )
    dy = xr.DataArray(
        data=dy,
        dims=["y", "x"],
        coords={
            "y": (["y"], h.y.data[1:]),
            "x": (["x"], h.x.data),
            "lon": (["y", "x"], h.lon.data[1:, :]),
            "lat": (["y", "x"], h.lat.data[1:, :]),
        },
    )
    return dx, dy


def _get_dhdx(h: xr.DataArray, center_point: tuple[float]) -> xr.DataArray:
    """Get the spatial derivative in the x direction around a single point
    on the WRF grid.

    Parameters
    ----------
    h : xr.DataArray
        Data on WRF grid
    center : xr.DataArray
        Target point extracted from h dataset.

    Returns
    -------
    xr.DataArray
        Derivative of h with respect to x
    """
    nominal_spacing = 45000.0  # WRF projection
    delta_x, _ = _wrf_deltas(h)
    back_one = h.sel(
        x=(center_point[0] - nominal_spacing), y=center_point[1], method="nearest"
    )
    center = h.sel(x=center_point[0], y=center_point[1], method="nearest")
    forward_one = h.sel(
        x=(center_point[0] + nominal_spacing), y=center_point[1], method="nearest"
    )

    # delta coordinates are for terminus point of delta
    diff_one = delta_x.sel(x=center_point[0], y=center_point[1], method="nearest")
    diff_two = delta_x.sel(
        x=center_point[0] + nominal_spacing, y=center_point[1], method="nearest"
    )

    # Method for taking derivative on unevenly spaced grid. See MetPy first_derivative()
    derivative = (
        (-diff_two) / ((diff_one + diff_two) * diff_one) * back_one
        + (diff_two - diff_one) / (diff_one * diff_two) * center
        + (diff_one) / ((diff_one + diff_two) * diff_two) * forward_one
    )
    return derivative


def _get_dhdy(h: xr.DataArray, center_point: tuple[float]) -> xr.DataArray:
    """Get the spatial derivative in the y direction for a single point
    on the WRF grid.

    Parameters
    ----------
    h : xr.DataArray
        Data on WRF grid
    center : xr.DataArray
        Target point extracted from h dataset.

    Returns
    -------
    xr.DataArray
        Derivative of h with respect to y
    """

    nominal_spacing = 45000.0  # WRF projection
    _, delta_y = _wrf_deltas(h)
    back_one = h.sel(
        x=center_point[0], y=(center_point[1] - nominal_spacing), method="nearest"
    )
    center = h.sel(x=center_point[0], y=center_point[1], method="nearest")
    forward_one = h.sel(
        x=center_point[0], y=(center_point[1] + nominal_spacing), method="nearest"
    )

    # delta coordinates are for terminus point of delta
    diff_one = delta_y.sel(x=center_point[0], y=center_point[1], method="nearest")
    diff_two = delta_y.sel(
        x=center_point[0], y=center_point[1] + nominal_spacing, method="nearest"
    )

    # Method for taking derivative on unevenly spaced grid. See MetPy first_derivative()
    derivative = (
        (-diff_two) / ((diff_one + diff_two) * diff_one) * back_one
        + (diff_two - diff_one) / (diff_one * diff_two) * center
        + (diff_one) / ((diff_one + diff_two) * diff_two) * forward_one
    )
    return derivative


def _get_rotated_geostrophic_wind(
    u: xr.DataArray, v: xr.DataArray, point: tuple[float], gridlabel: str
) -> tuple[xr.DataArray]:
    """Convert WRF-relative winds to Earth-relative winds.

    Parameters
    ----------
    u : xr.DataArray
        U component of wind
    v : xr.DataArray
        V component of wind
    point : tuple[float]
        x, y coordinates of point (meters)
    gridlabel : str
        Grid label (e.g. "d01")

    Returns
    -------
    tuple[xr.DataArray]
        Earth-relative U and V wind components
    """
    # Read in the appropriate file depending on the data resolution
    # This file contains sinalpha and cosalpha for the WRF grid
    wrf_angles_ds = xr.open_zarr(
        "s3://cadcat/tmp/era/wrf/wrf_angles_{}.zarr/".format(gridlabel),
        storage_options={"anon": True},
    )
    wrf_angles_ds = wrf_angles_ds.sel(x=point[0], y=point[1], method="nearest")
    sinalpha = wrf_angles_ds.SINALPHA
    cosalpha = wrf_angles_ds.COSALPHA

    # Wind components
    Uearth = u * cosalpha - v * sinalpha
    Vearth = v * cosalpha + u * sinalpha

    # Add variable name
    Uearth.name = "u"
    Vearth.name = "v"

    return Uearth, Vearth


def geostrophic_wind_single_point(
    geopotential_height: xr.DataArray, point: tuple[float]
) -> tuple[xr.DataArray]:
    """Calculate the geostrophic wind at a single point on a constant pressure surface.

    Parameters
    ----------
    geopotential_height : xr.DataArray
        Geopotential height in meters on WRF grid. May include multiple pressure levels
    point : tuple[float]
        Lat, lon coordinates for point in degrees

    Returns
    -------
    tuple[xr.DataArray]
        Earth-relative U and V components of the geostrophic wind.
    """
    center = get_closest_gridcell(geopotential_height.compute(), point[0], point[1])

    lat_rad = center.lat.data * np.pi / 180
    omega = 7292115e-11  # rad/s
    g = 9.81  # m/s2
    f = 2 * omega * np.sin(lat_rad)
    norm_factor = g / f

    # Get the actual x,y coordinates at this point
    wrf_point = (center.x.compute().data, center.y.compute().data)

    dhdx = _get_dhdx(geopotential_height, wrf_point)
    dhdy = _get_dhdy(geopotential_height, wrf_point)

    # These components are u and v on the WRF grid
    geo_u, geo_v = -norm_factor * dhdy, norm_factor * dhdx

    # Rotate these components to an earth-relative E/W orientation
    geo_u_earth, geo_v_earth = _get_rotated_geostrophic_wind(
        geo_u, geo_v, wrf_point, "d01"
    )

    # Update attributes for results
    geo_u_earth.name = "u"
    geo_u_earth.attrs["long_name"] = "Geostrophic Wind U Component"
    geo_v_earth.name = "v"
    geo_v_earth.attrs["long_name"] = "Geostrophic Wind V Component"

    return geo_u_earth, geo_v_earth
