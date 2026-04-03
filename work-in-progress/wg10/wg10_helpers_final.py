"""Shared utility and figure functions for WG10 compound event notebooks (EPC-23-024).

Extends wg10_helpers.py with figure-generation functions so that final notebooks
only need a single function call per figure.
"""

import math
import os

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar
from matplotlib.collections import PolyCollection
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ── Carried over from wg10_helpers.py ────────────────────────────────────────
def pull_wrf(
    cd,
    variable,
    gwls,
    territory,
    resolution="d03",
    convert_units=None,
    add_dummy_time=False,
    warming_level_window=15,
):
    """Pull 5 bias-adjusted WRF sims for given variable/GWLs/territory.

    Parameters
    ----------
    add_dummy_time : bool
        If True, adds a dummy DatetimeIndex (via the warming_level processor's
        add_dummy_time option) so the output has a 'time' dim compatible with
        climakitae threshold tools like get_block_maxima/get_return_value.
    warming_level_window : int
        Half-width of the warming level window in years (default 15 → ±15 yr = 30 yr total).
    """
    cd.reset()
    procs = {
        "warming_level": {
            "warming_levels": gwls,
            "add_dummy_time": add_dummy_time,
            "warming_level_window": warming_level_window,
        },
        "clip": territory,
        "filter_unadjusted_models": "yes",
    }
    if convert_units:
        procs["convert_units"] = convert_units
    return (
        cd.catalog("cadcat")
        .activity_id("WRF")
        .institution_id("UCLA")
        .table_id("day")
        .grid_label(resolution)
        .variable(variable)
        .processes(procs)
        .get()
    )


def compute_mask(ds, variable):
    """Boolean mask (True = invalid/ocean) from first sim/time/level."""
    with ProgressBar():
        return (
            ds.isel(warming_level=0, time_delta=0, sim=0)[variable].isnull().compute()
        )


def empirical_return_value(da, n_years=30.0, return_period=2, dim="time_delta"):
    """
    Empirical return value via annual block maxima.

    1-in-N year = (1 - 1/N) quantile of annual maxima.

    Parameters
    ----------
    da : xr.DataArray
        Input data with a ``dim`` axis of length ``n_years * block``.
    n_years : float
        Number of years (blocks) in the window.
    return_period : int
        Return period in years (e.g. 5 → 1-in-5 year event).
    dim : str
        Time dimension to split into annual blocks.

    Returns
    -------
    xr.DataArray : same shape as ``da`` minus ``dim``.
    """
    n = int(n_years)
    block = len(da[dim]) // n
    annual_max = xr.concat(
        [
            da.isel({dim: slice(b * block, (b + 1) * block)}).max(dim=dim)
            for b in range(n)
        ],
        dim="year",
    )
    return annual_max.quantile(1 - 1 / return_period, dim="year")


def spatial_hit_fraction(hit_da, spatial_dims=("y", "x"), threshold=0.75):
    """
    Return True where >= threshold fraction of spatial grid boxes are True.

    Parameters
    ----------
    hit_da : xr.DataArray
        Boolean array with spatial dims ``y`` and ``x``.
    spatial_dims : tuple of str
        Names of the spatial dimensions to average over.
    threshold : float
        Minimum fraction of grid boxes that must be True.

    Returns
    -------
    xr.DataArray : ``hit_da`` with spatial dims removed, dtype bool.
    """
    return hit_da.mean(dim=list(spatial_dims)) >= threshold


def save_outputs(maps_dict, out_dir="wg10"):
    """
    Save dict of {filename_stem: DataArray} as NetCDF files under ``out_dir``.

    Overwrites any existing files with the same name.
    """
    os.makedirs(out_dir, exist_ok=True)
    for stem, da in maps_dict.items():
        path = f"{out_dir}/{stem}.nc"
        if os.path.exists(path):
            os.remove(path)
        da.to_netcdf(path)
    print(f"Saved {list(maps_dict.keys())} to {out_dir}/")


# ── Shared utilities ─────────────────────────────────────────────────────────


def get_gwl_year_map(gwls, ssp="SSP 3-7.0"):
    """Return {gwl: year} dict of mean crossing years for a list of GWLs.

    Falls back to Historical SSP if the primary SSP has no crossing year.

    Parameters
    ----------
    gwls : list of float
        Global warming levels (°C) to look up.
    ssp : str
        SSP scenario string passed to ``climakitae.util.warming_levels.get_year_at_gwl``.

    Returns
    -------
    dict : {gwl: int}
    """
    from climakitae.util.warming_levels import get_year_at_gwl

    years = {}
    for gwl in gwls:
        year = get_year_at_gwl(gwl, ssp=ssp)["Mean"].item()
        if pd.isna(year):
            year = get_year_at_gwl(gwl, ssp="Historical")["Mean"].item()
        years[gwl] = int(year)
    return years


def plot_gwl_bar_chart(
    bars_by_label,
    title,
    ylabel,
    out_path,
    colors=None,
    ymin=None,
    ymax=None,
    figsize=(10, 5),
):
    """Generic bar chart for GWL-indexed scalar values.

    Parameters
    ----------
    bars_by_label : dict
        Ordered ``{x_label: value}`` mapping. Labels become x-tick labels.
    title : str
        Figure title.
    ylabel : str
        Y-axis label.
    out_path : str
        Full path (including filename) to save the PNG.
    colors : list or None
        Bar face colors, one per bar. Defaults to the default matplotlib cycle.
    ymin, ymax : float or None
        Y-axis limits; omit to use matplotlib auto-scaling.
    figsize : tuple
        Figure size in inches (width, height).
    """
    labels = list(bars_by_label.keys())
    values = list(bars_by_label.values())

    fig, ax = plt.subplots(figsize=figsize)

    bar_kw = dict(edgecolor="black", linewidth=0.6, width=0.7, zorder=2)
    if colors is not None:
        for i, (label, val) in enumerate(zip(labels, values)):
            ax.bar(label, val, color=colors[i] if i < len(colors) else None, **bar_kw)
    else:
        ax.bar(labels, values, **bar_kw)

    if ymin is not None or ymax is not None:
        ax.set_ylim(
            ymin if ymin is not None else ax.get_ylim()[0],
            ymax if ymax is not None else ax.get_ylim()[1],
        )

    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlabel("Global Warming Level", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight", transparent=True)
    plt.show()
    print(f"Saved → {out_path}")


def plot_gwl_maps(
    data_by_gwl,
    title,
    colorbar_label,
    cmap,
    out_path,
    years=None,
    boundary=None,
    vmin=None,
    vmax=None,
    marker_lon=None,
    marker_lat=None,
    marker_label="★",
    ncols=5,
):
    """N-panel grid of GWL spatial maps, one panel per GWL, with OSM basemap.

    Parameters
    ----------
    data_by_gwl : dict
        ``{gwl: xr.DataArray}`` mapping. Each DataArray must have ``lon`` and
        ``lat`` 2-D coordinate arrays.
    title : str
        Figure suptitle.
    colorbar_label : str
        Label for the shared colorbar.
    cmap : str
        Matplotlib colormap name.
    out_path : str
        Full path (including filename) to save the PNG.
    years : dict or None
        ``{gwl: year}`` — if provided, each panel title shows "GWL X°C - YEAR".
    boundary : GeoDataFrame or None
        Territory boundary to draw on each panel.
    vmin, vmax : float or None
        Colorbar limits. Derived from data if not provided.
    marker_lon, marker_lat : float or None
        If both provided, a star marker is plotted at this location on each panel.
    marker_label : str
        Legend label for the marker (shown in suptitle note).
    ncols : int
        Number of columns in the grid layout.
    """
    import cartopy
    import cartopy.crs as ccrs
    import cartopy.io.img_tiles as cimgt

    cartopy.config["cache_dir"] = os.path.expanduser("~/.cartopy_cache")

    if not data_by_gwl:
        print("No data to plot.")
        return

    gwls = list(data_by_gwl.keys())
    nrows = math.ceil(len(gwls) / ncols)

    if vmax is None:
        vmax = max(float(np.nanmax(da.values)) for da in data_by_gwl.values())
    if vmin is None:
        vmin = 0

    _sample = next(iter(data_by_gwl.values()))
    _lon = _sample.lon.values if "lon" in _sample.coords else _sample.x.values
    _lat = _sample.lat.values if "lat" in _sample.coords else _sample.y.values
    _valid = ~np.isnan(_sample.values)
    LON_MIN = float(_lon[_valid].min()) - 0.15
    LON_MAX = float(_lon[_valid].max()) + 0.15
    LAT_MIN = float(_lat[_valid].min()) - 0.15
    LAT_MAX = float(_lat[_valid].max()) + 0.15

    fig = plt.figure(figsize=(4 * ncols, 4.5 * nrows + 0.5))
    gs = GridSpec(
        nrows,
        ncols,
        figure=fig,
        hspace=0.15,
        wspace=0.02,
        left=0.02,
        right=0.88,
        top=0.88,
        bottom=0.05,
    )
    subtitle = f"{marker_label} = {marker_label}" if marker_lon is not None else ""
    fig.text(0.5, 0.97, title, ha="center", va="top", fontsize=14)
    if marker_lon is not None and marker_label == "★":
        fig.text(0.5, 0.935, "★ = Sacramento", ha="center", va="top", fontsize=13)

    for i, gwl in enumerate(gwls):
        tiler = cimgt.OSM()
        ax = fig.add_subplot(gs[i // ncols, i % ncols], projection=tiler.crs)

        if gwl not in data_by_gwl:
            ax.set_visible(False)
            continue

        da = data_by_gwl[gwl]
        lon = da.lon.values if "lon" in da.coords else da.x.values
        lat = da.lat.values if "lat" in da.coords else da.y.values

        ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
        ax.add_image(tiler, 9, zorder=1)
        ax.pcolormesh(
            lon,
            lat,
            da.values,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            transform=ccrs.PlateCarree(),
            alpha=0.8,
            zorder=2,
        )

        if boundary is not None:
            boundary.to_crs("EPSG:4326").boundary.plot(
                ax=ax,
                color="black",
                linewidth=1.0,
                transform=ccrs.PlateCarree(),
                zorder=3,
            )

        if marker_lon is not None and marker_lat is not None:
            ax.plot(
                marker_lon,
                marker_lat,
                marker="*",
                color="#000000",
                markersize=7,
                transform=ccrs.PlateCarree(),
                zorder=5,
            )

        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=True,
            linewidth=0.3,
            color="gray",
            alpha=0.5,
        )
        gl.top_labels = False
        gl.right_labels = False
        gl.left_labels = i % ncols == 0
        gl.bottom_labels = i // ncols == nrows - 1
        gl.xlabel_style = {"size": 7}
        gl.ylabel_style = {"size": 7}

        panel_title = f"GWL {gwl}°C"
        if years is not None and gwl in years:
            panel_title += f" - {years[gwl]}"
        ax.set_title(panel_title, fontsize=12)

    cax = fig.add_axes([0.90, 0.15, 0.015, 0.65])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    fig.colorbar(sm, cax=cax, label=colorbar_label)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight", transparent=True)
    plt.show()
    print(f"Saved → {out_path}")


# ── Event 1: Compound Event Asset Prioritization ─────────────────────────────


def _build_polys(da, boundary_gdf):
    """Build a grid-cell polygon list for cells inside ``boundary_gdf``.

    Parameters
    ----------
    da : xr.DataArray
        Any 2-D DataArray with ``lon`` and ``lat`` 2-D coordinate arrays.
    boundary_gdf : GeoDataFrame
        Territory boundary in any CRS (converted to EPSG:4326 internally).

    Returns
    -------
    list of list of (lon, lat) tuples
        One quadrilateral per grid cell whose centre falls inside the boundary.
    """
    from shapely.vectorized import contains as shp_contains

    geom = boundary_gdf.to_crs("EPSG:4326").geometry.union_all()
    lons_flat = da.lon.values.ravel()
    lats_flat = da.lat.values.ravel()
    valid = ~np.isnan(lons_flat)
    inside = np.zeros(len(lons_flat), dtype=bool)
    inside[valid] = shp_contains(geom, lons_flat[valid], lats_flat[valid])
    mask_2d = inside.reshape(da.lon.values.shape)

    lons, lats = da.lon.values, da.lat.values
    ny, nx = lons.shape
    polys = []
    for i in range(ny - 1):
        for j in range(nx - 1):
            if mask_2d[i, j]:
                polys.append(
                    [
                        (lons[i, j], lats[i, j]),
                        (lons[i, j + 1], lats[i, j + 1]),
                        (lons[i + 1, j + 1], lats[i + 1, j + 1]),
                        (lons[i + 1, j], lats[i + 1, j]),
                    ]
                )
    return polys


def _three_panel_osm(
    panels,
    tx_lines,
    sce_boundary,
    lon_extent,
    lat_extent,
    suptitle,
    out_path,
    basemap_scale=7,
    pair_unit_label=None,
    delta_unit_label=None,
):
    """Internal: draw the standard 3-panel (GWL08 | GWL20 | delta) map with OSM tiles."""
    import cartopy
    import cartopy.crs as ccrs
    import cartopy.io.img_tiles as cimgt

    cartopy.config["cache_dir"] = os.path.expanduser("~/.cartopy_cache")

    LON_MIN, LON_MAX = lon_extent
    LAT_MIN, LAT_MAX = lat_extent
    tiler = cimgt.OSM()

    # Build grid-cell polygons from the first data panel and the boundary
    grid_polys = []
    if sce_boundary is not None and not sce_boundary.empty:
        grid_polys = _build_polys(panels[0][0], sce_boundary)

    fig = plt.figure(figsize=(25, 9))
    fig.patch.set_alpha(0)
    outer_gs = GridSpec(1, 2, figure=fig, wspace=0.1, width_ratios=[2, 1])
    pair_gs = outer_gs[0].subgridspec(1, 2, wspace=0.02)

    tx_legend_handle = Line2D(
        [0], [0], color="#888888", linewidth=1.5, label="Transmission lines"
    )

    specs = [pair_gs[0], pair_gs[1], outer_gs[1]]
    axes, ims = [], []
    for col, ((data, title, cmap, vmin, vmax_panel), spec) in enumerate(
        zip(panels, specs)
    ):
        ax = fig.add_subplot(spec, projection=tiler.crs)
        ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
        ax.add_image(tiler, basemap_scale, zorder=1)

        im = ax.pcolormesh(
            data.lon.values,
            data.lat.values,
            data.values,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax_panel,
            transform=ccrs.PlateCarree(),
            alpha=0.8,
            zorder=2,
        )
        ims.append(im)

        if grid_polys:
            ax.add_collection(
                PolyCollection(
                    grid_polys,
                    facecolor="none",
                    edgecolor="#666666",
                    linewidth=0.08,
                    transform=ccrs.PlateCarree(),
                    zorder=3,
                )
            )

        if tx_lines is not None:
            tx_lines.plot(
                ax=ax,
                color="#888888",
                linewidth=0.7,
                alpha=0.9,
                transform=ccrs.PlateCarree(),
                zorder=4,
            )

        if sce_boundary is not None and not sce_boundary.empty:
            sce_boundary.to_crs("EPSG:4326").boundary.plot(
                ax=ax,
                color="black",
                linewidth=1.4,
                transform=ccrs.PlateCarree(),
                zorder=5,
            )
        else:
            territory_mask = (~np.isnan(data.values)).astype(float)
            ax.contour(
                data.lon.values,
                data.lat.values,
                territory_mask,
                levels=[0.5],
                colors=["black"],
                linewidths=1.4,
                transform=ccrs.PlateCarree(),
                zorder=5,
            )

        gl = ax.gridlines(
            draw_labels=True,
            crs=ccrs.PlateCarree(),
            linewidth=0.3,
            color="gray",
            alpha=0.5,
        )
        gl.top_labels = False
        gl.right_labels = False
        gl.left_labels = col == 0
        gl.xlabel_style = {"size": 11}
        gl.ylabel_style = {"size": 11}
        for spine in ax.spines.values():
            spine.set_zorder(7)

        ax.set_title(title, fontsize=14, pad=4)
        if tx_lines is not None:
            ax.legend(
                handles=[tx_legend_handle],
                loc="lower left",
                fontsize=11,
                framealpha=0.7,
            )
        axes.append(ax)

    fig.canvas.draw()
    p1 = axes[1].get_position()
    p2 = axes[2].get_position()
    cbar_h = p1.height * 0.85
    cbar_y = p1.y0 + (p1.height - cbar_h) / 2

    cax_pair = fig.add_axes([p1.x1 + 0.005, cbar_y, 0.010, cbar_h])
    fig.colorbar(ims[0], cax=cax_pair).ax.tick_params(labelsize=10)
    cax_pair.set_ylabel(pair_unit_label or "", fontsize=11)

    cax_delta = fig.add_axes([p2.x1 + 0.005, cbar_y, 0.010, cbar_h])
    fig.colorbar(ims[2], cax=cax_delta).ax.tick_params(labelsize=10)
    cax_delta.set_ylabel(delta_unit_label or pair_unit_label or "", fontsize=11)

    fig.suptitle(suptitle, fontsize=20, y=0.95)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight", transparent=True)
    plt.show()
    print(f"Saved → {out_path}")


def plot_compound_event_maps(
    baseline,
    future,
    delta,
    tx_lines,
    sce_boundary,
    out_path,
    lon_extent=(-121.0, -114.0),
    lat_extent=(33.0, 38.5),
    title="Compound Event Frequency: High Winds During Extreme 3-Day Precipitation",
    basemap_scale=7,
):
    """3-panel OSM map: compound event frequency at GWL 0.8, GWL 2.0, and delta.

    Grid-cell polygons are generated automatically from ``baseline`` and ``sce_boundary``.

    Parameters
    ----------
    baseline, future, delta : xr.DataArray
        2-D arrays (y, x) with ``lon``/``lat`` coordinate arrays.
    tx_lines : GeoDataFrame
        Transmission line geometries.
    sce_boundary : GeoDataFrame
        SCE service territory boundary.
    out_path : str
        Full output file path.
    lon_extent, lat_extent : tuple of (min, max)
        Map extent in degrees.
    title : str
        Figure suptitle.
    basemap_scale : int
        OSM zoom level (higher = more detail, slower).
    """
    vmax = max(float(baseline.max()), float(future.max()))
    if vmax == 0:
        vmax = 0.01
    delta_abs = max(abs(float(delta.min())), abs(float(delta.max())), 0.001)

    panels = [
        (baseline, "GWL 0.8°C", "YlOrRd", 0, vmax),
        (future, "GWL 2.0°C", "YlOrRd", 0, vmax),
        (delta, "Delta (2.0°C − 0.8°C)", "RdBu_r", -delta_abs, delta_abs),
    ]
    _three_panel_osm(
        panels,
        tx_lines,
        sce_boundary,
        lon_extent,
        lat_extent,
        suptitle=title,
        out_path=out_path,
        basemap_scale=basemap_scale,
        pair_unit_label="Avg compound events / year",
    )


def plot_metric_frequency_maps(
    freq_gwl08,
    freq_gwl20,
    tx_lines,
    sce_boundary,
    metric_label,
    cmap,
    unit_label,
    out_path,
    lon_extent=(-121.0, -114.0),
    lat_extent=(33.0, 38.5),
    basemap_scale=7,
):
    """3-panel OSM map for a single sub-condition metric (wind or precip frequency).

    Grid-cell polygons are generated automatically from ``freq_gwl08`` and ``sce_boundary``.

    Parameters
    ----------
    freq_gwl08, freq_gwl20 : xr.DataArray
        Annual frequency maps at GWL 0.8°C and 2.0°C.
    tx_lines : GeoDataFrame
        Transmission line geometries.
    sce_boundary : GeoDataFrame
        SCE service territory boundary.
    metric_label : str
        Figure suptitle (e.g. "Days with Max Wind Speed > 20 mph").
    cmap : str
        Colormap for the pair panels (delta always uses "RdBu_r").
    unit_label : str
        Colorbar label (e.g. "Days / year").
    out_path : str
        Full output file path.
    lon_extent, lat_extent : tuple
        Map extent in degrees.
    basemap_scale : int
        OSM zoom level.
    """
    delta_m = freq_gwl20 - freq_gwl08
    vmax = max(float(freq_gwl08.max()), float(freq_gwl20.max()))
    if vmax == 0:
        vmax = 0.01
    delta_abs = max(abs(float(delta_m.min())), abs(float(delta_m.max())), 0.001)

    panels = [
        (freq_gwl08, "GWL 0.8°C", cmap, 0, vmax),
        (freq_gwl20, "GWL 2.0°C", cmap, 0, vmax),
        (delta_m, "Delta (2.0°C − 0.8°C)", "RdBu_r", -delta_abs, delta_abs),
    ]
    _three_panel_osm(
        panels,
        tx_lines,
        sce_boundary,
        lon_extent,
        lat_extent,
        suptitle=metric_label,
        out_path=out_path,
        basemap_scale=basemap_scale,
        pair_unit_label=unit_label,
    )


def plot_example_compound_day(
    precip_2d,
    wind_2d,
    gwl_label,
    out_path,
    tx_lines,
    sce_boundary,
    precip_threshold,
    wind_threshold,
    lon_extent=(-121.0, -114.0),
    lat_extent=(33.0, 38.5),
    basemap_scale=7,
):
    """2-panel map for a single example compound event day.

    Left panel: 3-day rolling precip (in). Right panel: daily max wind speed (mph).
    Red overlay marks cells where both conditions are simultaneously met.
    Grid-cell polygons are generated automatically from ``precip_2d`` and ``sce_boundary``.

    Parameters
    ----------
    precip_2d, wind_2d : xr.DataArray
        2-D slices for one example day.
    gwl_label : str
        Panel title suffix (e.g. "GWL 0.8°C").
    out_path : str
        Full output file path.
    tx_lines : GeoDataFrame
        Transmission line geometries.
    sce_boundary : GeoDataFrame
        SCE service territory boundary.
    precip_threshold, wind_threshold : float
        Thresholds used to identify compound cells (for red overlay).
    lon_extent, lat_extent : tuple
        Map extent in degrees.
    basemap_scale : int
        OSM zoom level.
    """
    import cartopy
    import cartopy.crs as ccrs
    import cartopy.io.img_tiles as cimgt

    cartopy.config["cache_dir"] = os.path.expanduser("~/.cartopy_cache")

    LON_MIN, LON_MAX = lon_extent
    LAT_MIN, LAT_MAX = lat_extent
    tiler = cimgt.OSM()

    # Grid-cell polygons for the SCE territory outline mesh
    grid_polys = []
    if sce_boundary is not None and not sce_boundary.empty:
        grid_polys = _build_polys(precip_2d, sce_boundary)

    # Red overlay polygons for cells where both conditions are simultaneously met
    compound_mask_2d = (precip_2d > precip_threshold) & (wind_2d > wind_threshold)
    compound_polys = []
    lons, lats = precip_2d.lon.values, precip_2d.lat.values
    compound_vals = compound_mask_2d.values
    ny, nx = lons.shape
    for i in range(ny - 1):
        for j in range(nx - 1):
            if compound_vals[i, j]:
                compound_polys.append(
                    [
                        (lons[i, j], lats[i, j]),
                        (lons[i, j + 1], lats[i, j + 1]),
                        (lons[i + 1, j + 1], lats[i + 1, j + 1]),
                        (lons[i + 1, j], lats[i + 1, j]),
                    ]
                )

    panels = [
        (
            precip_2d,
            "YlGnBu",
            f"3-Day Rolling Precip (in) — {gwl_label}",
            "3-Day Precip (in)",
        ),
        (
            wind_2d,
            "Blues",
            f"Daily Max Wind Speed (mph) — {gwl_label}",
            "Wind Speed (mph)",
        ),
    ]

    tx_legend_handle = Line2D(
        [0], [0], color="#888888", linewidth=1.5, label="Transmission lines"
    )
    compound_legend_handle = Line2D(
        [0],
        [0],
        color="red",
        linewidth=0,
        marker="s",
        markersize=8,
        label="Compound hit",
        alpha=0.5,
    )

    fig = plt.figure(figsize=(20, 8))
    fig.patch.set_alpha(0)
    gs = GridSpec(1, 2, figure=fig, wspace=0.05)

    axes, ims = [], []
    for col, (data, cmap, title, cbar_label) in enumerate(panels):
        ax = fig.add_subplot(gs[col], projection=tiler.crs)
        ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
        ax.add_image(tiler, basemap_scale, zorder=1)

        im = data.plot(
            ax=ax,
            x="lon",
            y="lat",
            cmap=cmap,
            add_colorbar=False,
            add_labels=False,
            transform=ccrs.PlateCarree(),
            alpha=0.8,
            edgecolors="none",
            zorder=2,
        )
        ims.append((im, cbar_label))

        if grid_polys:
            ax.add_collection(
                PolyCollection(
                    grid_polys,
                    facecolor="none",
                    edgecolor="#666666",
                    linewidth=0.08,
                    transform=ccrs.PlateCarree(),
                    zorder=3,
                )
            )

        if compound_polys:
            ax.add_collection(
                PolyCollection(
                    compound_polys,
                    facecolor="red",
                    edgecolor="none",
                    alpha=0.4,
                    transform=ccrs.PlateCarree(),
                    zorder=4,
                )
            )

        if tx_lines is not None:
            tx_lines.plot(
                ax=ax,
                color="#888888",
                linewidth=0.7,
                alpha=0.9,
                transform=ccrs.PlateCarree(),
                zorder=5,
            )

        if sce_boundary is not None and not sce_boundary.empty:
            sce_boundary.to_crs("EPSG:4326").boundary.plot(
                ax=ax,
                color="black",
                linewidth=1.4,
                transform=ccrs.PlateCarree(),
                zorder=6,
            )

        gl = ax.gridlines(
            draw_labels=True,
            crs=ccrs.PlateCarree(),
            linewidth=0.3,
            color="gray",
            alpha=0.5,
        )
        gl.top_labels = False
        gl.right_labels = False
        gl.left_labels = col == 0
        gl.xlabel_style = {"size": 11}
        gl.ylabel_style = {"size": 11}
        for spine in ax.spines.values():
            spine.set_zorder(8)

        ax.set_title(title, fontsize=14, pad=4)
        ax.legend(
            handles=[tx_legend_handle, compound_legend_handle],
            loc="lower left",
            fontsize=11,
            framealpha=0.7,
        )
        axes.append(ax)

    fig.canvas.draw()
    for ax, (im, cbar_label) in zip(axes, ims):
        p = ax.get_position()
        cbar_h = p.height * 0.85
        cbar_y = p.y0 + (p.height - cbar_h) / 2
        cax = fig.add_axes([p.x1 + 0.005, cbar_y, 0.010, cbar_h])
        fig.colorbar(im, cax=cax).ax.tick_params(labelsize=10)
        cax.set_ylabel(cbar_label, fontsize=11)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight", transparent=True)
    plt.show()
    print(f"Saved → {out_path}")


# ── Event 3: Heatwave + No-Wind ───────────────────────────────────────────────


def plot_subcondition_frequency_maps(
    hot_day_by_gwl,
    warm_night_by_gwl,
    tmax_threshold,
    tmin_threshold,
    years,
    boundary,
    out_dir,
):
    """Plot hot day and warm night rate maps for all GWLs.

    Calls ``plot_gwl_maps`` twice: once for hot days, once for warm nights.

    Parameters
    ----------
    hot_day_by_gwl, warm_night_by_gwl : dict
        ``{gwl: xr.DataArray}`` of per-grid-cell annual frequencies.
    tmax_threshold, tmin_threshold : float
        Temperature thresholds (°F) used in the panel titles.
    years : dict
        ``{gwl: year}`` from ``get_gwl_year_map()``.
    boundary : GeoDataFrame
        Territory boundary to overlay on each panel.
    out_dir : str
        Directory in which to save the two PNG files.
    """
    plot_gwl_maps(
        hot_day_by_gwl,
        title=f"Hot Day Rate — tmax > {tmax_threshold:.0f}°F (Events per Grid Cell per Year)",
        colorbar_label="Hot days / year",
        cmap="YlOrRd",
        out_path=f"{out_dir}/hot_day_rate_maps.png",
        years=years,
        boundary=boundary,
        vmin=60,
        vmax=150,
    )
    plot_gwl_maps(
        warm_night_by_gwl,
        title=f"Warm Night Rate — tmin > {tmin_threshold:.0f}°F (Events per Grid Cell per Year)",
        colorbar_label="Warm nights / year",
        cmap="YlOrRd",
        out_path=f"{out_dir}/warm_night_rate_maps.png",
        years=years,
        boundary=boundary,
    )


def plot_smud_solano_grid_map(
    smud_boundary, solano_gdf, smud_grid, solano_grid, out_path
):
    """Territory overview map showing SMUD service territory and Solano wind farm grid cells.

    Parameters
    ----------
    smud_boundary : GeoDataFrame
        SMUD service territory boundary polygon(s).
    solano_gdf : GeoDataFrame
        Solano wind farm area polygon(s).
    smud_grid : xr.DataArray
        A 2-D WRF grid variable covering the SMUD domain (used for lon/lat coords).
    solano_grid : xr.DataArray
        A 2-D WRF grid variable covering the Solano wind farm domain.
    out_path : str
        Full output file path.
    """
    import cartopy
    import cartopy.crs as ccrs
    import cartopy.io.img_tiles as cimgt
    from shapely import intersects_xy

    cartopy.config["cache_dir"] = os.path.expanduser("~/.cartopy_cache")

    def _lonlat(da):
        if "lon" in da.coords:
            return da.lon.values, da.lat.values
        return da.x.values, da.y.values

    smud_geom = smud_boundary.union_all()
    lon2d, lat2d = _lonlat(smud_grid)
    ny, nx = lon2d.shape
    lon2d_solano, lat2d_solano = _lonlat(solano_grid)

    flat_lon = lon2d.ravel().astype(float)
    flat_lat = lat2d.ravel().astype(float)
    valid = ~np.isnan(flat_lon)
    smud_mask_flat = np.zeros(len(flat_lon), dtype=bool)
    smud_mask_flat[valid] = intersects_xy(smud_geom, flat_lon[valid], flat_lat[valid])
    smud_mask = smud_mask_flat.reshape(ny, nx)

    smud_b = smud_boundary.total_bounds  # (minx, miny, maxx, maxy)
    pad = 0.25
    LAT_MIN = smud_b[1] - pad
    LAT_MAX = smud_b[3] + pad
    LON_MIN = smud_b[0] - pad - 0.18
    LON_MAX = smud_b[2] + pad - 0.18

    lat_mid = (LAT_MIN + LAT_MAX) / 2
    lon_range_km = (LON_MAX - LON_MIN) * 111 * math.cos(math.radians(lat_mid))
    lat_range_km = (LAT_MAX - LAT_MIN) * 111
    fig_w = 10
    fig_ha = fig_w / (lon_range_km / lat_range_km)

    tiler = cimgt.OSM()
    fig, ax = plt.subplots(
        figsize=(fig_w, fig_ha), subplot_kw={"projection": tiler.crs}
    )
    ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
    ax.add_image(tiler, 10, zorder=1)

    pc_kw = dict(transform=ccrs.PlateCarree(), zorder=3)
    smud_fill = np.where(smud_mask, 1.0, np.nan)
    ax.pcolormesh(
        lon2d,
        lat2d,
        smud_fill,
        cmap=mcolors.ListedColormap(["#1f78b4"]),
        vmin=0.5,
        vmax=1.5,
        alpha=0.35,
        **pc_kw,
    )
    ax.pcolormesh(
        lon2d_solano,
        lat2d_solano,
        solano_grid.values,
        cmap=mcolors.ListedColormap(["#FF8C00"]),
        vmin=0.5,
        vmax=1.5,
        alpha=0.35,
        **pc_kw,
    )

    ax.add_geometries(
        smud_boundary.geometry,
        crs=ccrs.PlateCarree(),
        facecolor="none",
        edgecolor="#1f78b4",
        linewidth=2.0,
        zorder=5,
    )
    ax.add_geometries(
        solano_gdf.geometry,
        crs=ccrs.PlateCarree(),
        facecolor="none",
        edgecolor="#FF8C00",
        linewidth=2.0,
        zorder=5,
    )

    ax.set_title("SMUD Service Territory & Solano Wind Farm Area", fontsize=16)
    legend_elements = [
        Patch(
            facecolor="#1f78b4",
            alpha=0.35,
            edgecolor="#1f78b4",
            label="SMUD Service Territory",
        ),
        Patch(
            facecolor="#FF8C00",
            alpha=0.35,
            edgecolor="#FF8C00",
            label="Solano Wind Farm Area",
        ),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=14, framealpha=0.85)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight", transparent=True)
    plt.show()
    print(f"Saved → {out_path}")


# ── Event 5: Compound Heatwave ────────────────────────────────────────────────


def plot_example_heatwave_day(
    gwl,
    ca_day,
    sce_tmax_day,
    sce_tmin_day,
    dates_df,
    tmax_threshold,
    tmin_threshold,
    out_path,
):
    """3-panel spatial map for one example compound heatwave event day.

    Panels: CA tmax (d01, 45 km) | SCE tmax (d03, 3 km) | SCE tmin (d03, 3 km).
    Threshold contours and fraction-above-threshold annotations are added to each panel.

    Parameters
    ----------
    gwl : float
        Global warming level (e.g. 0.8 or 2.0).
    ca_day, sce_tmax_day, sce_tmin_day : xr.DataArray
        2-D temperature snapshots for the example day.
    dates_df : pd.DataFrame
        Must have columns ``gwl``, ``sim``, ``time_delta_idx``.
    tmax_threshold, tmin_threshold : float
        Temperature thresholds (°F).
    out_path : str
        Full output file path.
    """

    def _lonlat(da):
        if "lon" in da.coords:
            return da.lon.values, da.lat.values
        return da.x.values, da.y.values

    ca_frac = float((ca_day.values[ca_day.values > -999] > tmax_threshold).mean())
    sce_tmax_vals = sce_tmax_day.values[~np.isnan(sce_tmax_day.values)]
    sce_tmin_vals = sce_tmin_day.values[~np.isnan(sce_tmin_day.values)]
    sce_tmax_frac = float((sce_tmax_vals > tmax_threshold).mean())
    sce_tmin_frac = float((sce_tmin_vals > tmin_threshold).mean())

    gwl_rows = dates_df[(dates_df["gwl"] == gwl) & (dates_df["sim"] == 0)]
    t_idx = int(gwl_rows.iloc[0]["time_delta_idx"]) if len(gwl_rows) else "?"

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"Example Compound Event Day — GWL {gwl}°C  |  sim 0, time_delta={t_idx}\n"
        f"All 3 conditions met: "
        f"CA tmax >{tmax_threshold:.0f}°F in {ca_frac:.0%} of grids  ·  "
        f"SCE tmax >{tmax_threshold:.0f}°F in {sce_tmax_frac:.0%}  ·  "
        f"SCE tmin >{tmin_threshold:.0f}°F in {sce_tmin_frac:.0%}",
        fontsize=10,
    )

    # Panel 1: CA tmax (d01)
    ax = axes[0]
    lon, lat = _lonlat(ca_day)
    im = ax.pcolormesh(lon, lat, ca_day.values, cmap="RdYlBu_r", vmin=50, vmax=115)
    ca_mask = (ca_day.values > tmax_threshold).astype(float)
    if ca_mask.max() > 0:
        ax.contour(lon, lat, ca_mask, levels=[0.5], colors="black", linewidths=1.2)
    plt.colorbar(im, ax=ax, label="°F", shrink=0.85)
    ax.set_title(
        f"CA tmax (d01, 45 km)\n>{tmax_threshold:.0f}°F in {ca_frac:.0%} of CA land grids",
        fontsize=10,
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # Panel 2: SCE tmax (d03)
    ax = axes[1]
    lon_s, lat_s = _lonlat(sce_tmax_day)
    im2 = ax.pcolormesh(
        lon_s, lat_s, sce_tmax_day.values, cmap="RdYlBu_r", vmin=50, vmax=115
    )
    sce_tmax_mask = (~np.isnan(sce_tmax_day.values)) & (
        sce_tmax_day.values > tmax_threshold
    )
    if sce_tmax_mask.astype(float).max() > 0:
        ax.contour(
            lon_s,
            lat_s,
            sce_tmax_mask.astype(float),
            levels=[0.5],
            colors="black",
            linewidths=1.2,
        )
    plt.colorbar(im2, ax=ax, label="°F", shrink=0.85)
    ax.set_title(
        f"SCE tmax (d03, 3 km)\n>{tmax_threshold:.0f}°F in {sce_tmax_frac:.0%} of SCE land grids",
        fontsize=10,
    )
    ax.set_xlabel("Longitude")

    # Panel 3: SCE tmin (d03)
    ax = axes[2]
    im3 = ax.pcolormesh(
        lon_s, lat_s, sce_tmin_day.values, cmap="RdYlBu_r", vmin=50, vmax=95
    )
    sce_tmin_mask = (~np.isnan(sce_tmin_day.values)) & (
        sce_tmin_day.values > tmin_threshold
    )
    if sce_tmin_mask.astype(float).max() > 0:
        ax.contour(
            lon_s,
            lat_s,
            sce_tmin_mask.astype(float),
            levels=[0.5],
            colors="black",
            linewidths=1.2,
        )
    plt.colorbar(im3, ax=ax, label="°F", shrink=0.85)
    ax.set_title(
        f"SCE tmin (d03, 3 km)\n>{tmin_threshold:.0f}°F in {sce_tmin_frac:.0%} of SCE land grids",
        fontsize=10,
    )
    ax.set_xlabel("Longitude")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved → {out_path}")
