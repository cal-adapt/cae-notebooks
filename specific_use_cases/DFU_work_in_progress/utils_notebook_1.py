from shapely.geometry import Point
import xarray as xr 
import hvplot.xarray
import hvplot.pandas
import rioxarray 

def clip_data_to_dfz(gridded_data, dfzs_df, station_lat, station_lon): 
    # Shapely Point object of the weather station 
    shapely_Point = Point(station_lon, station_lat) 

    # Get the name of the DFZ that contains that point 
    forecast_zone = dfzs_df.where(dfzs_df.contains(shapely_Point)).dropna()
    forecast_zone_name = forecast_zone.FZ_Name.item()
    print("Demand forecast zone: {0}".format(forecast_zone_name))
    
    # Clip data 
    clipped_data = gridded_data.rio.clip(
        geometries=[forecast_zone.geometry.item()], 
        crs=4326, 
        drop=True # Drop cells outside the forecast zone 
    )
    return clipped_data


def compute_annual_aggreggate(data, name, num_grid_cells):  
    annual_ag = data.squeeze().groupby('time.year').sum(['time']) # Aggregate annually
    annual_ag = annual_ag/num_grid_cells # Divide by number of gridcells 
    annual_ag.name = name # Give new name to dataset
    return annual_ag


def compute_multimodel_stats(hdd_annual, cdd_annual): 
    # Compute mean across simulation dimensions and add is as a coordinate
    hdd_sim_mean = hdd_annual.mean(dim="simulation").assign_coords({"simulation":"simulation mean"}).expand_dims("simulation") 
    cdd_sim_mean = cdd_annual.mean(dim="simulation").assign_coords({"simulation":"simulation mean"}).expand_dims("simulation") 

    # Compute multimodel min 
    hdd_sim_min = hdd_annual.min(dim="simulation").assign_coords({"simulation":"simulation min"}).expand_dims("simulation") 
    cdd_sim_min = cdd_annual.min(dim="simulation").assign_coords({"simulation":"simulation min"}).expand_dims("simulation") 

    # Compute multimodel max 
    hdd_sim_max = hdd_annual.max(dim="simulation").assign_coords({"simulation":"simulation max"}).expand_dims("simulation") 
    cdd_sim_max = cdd_annual.max(dim="simulation").assign_coords({"simulation":"simulation max"}).expand_dims("simulation") 

    # Add to main dataset
    hdd_concat = xr.concat([hdd_annual, hdd_sim_mean, hdd_sim_min, hdd_sim_max], dim="simulation")
    cdd_concat =  xr.concat([cdd_annual, cdd_sim_mean, cdd_sim_min, cdd_sim_max], dim="simulation")
    return hdd_concat, cdd_concat


def trendline(data): 
    data_sim_mean = data.sel(simulation="simulation mean") 
    m, b = data_sim_mean.polyfit(dim="year", deg=1).polyfit_coefficients.values
    trendline = m*data_sim_mean.year + b # y = mx + b 
    trendline.name = "trendline" 
    return trendline


def hdd_cdd_lineplot(annual_data, trendline, title="title"): 
    return annual_data.hvplot.line(
        x="year", by="simulation", 
        width=800, height=350,
        title=title,
        yformatter='%.0f' # Remove scientific notation
    ) * trendline.hvplot.line(  # Add trendline
        x="year", 
        color="black", 
        line_dash='dashed', 
        label="trendline"
    ) 