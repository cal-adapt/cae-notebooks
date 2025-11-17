import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import Colormap, ListedColormap
from matplotlib import cm
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from climakitae.core.data_interface import (
    DataInterface,
)
from climakitae.util.utils import read_csv_file

from climakitae.core.paths import (
    HIST_FILE,
    SSP119_FILE,
    SSP126_FILE,
    SSP245_FILE,
    SSP370_FILE,
    SSP585_FILE,
    GWL_1850_1900_TIMEIDX_FILE
)

def lighten_cmap(cmap: str, factor: float = 0.3) -> ListedColormap:
    """
    Lightens a colormap by blending the colors with white.

    Parameters:i wa
    cmap (str): The colormap name to lighten (e.g., 'inferno').
    factor (float): The factor by which to lighten the colormap (default is 0.3).

    Returns:
    ListedColormap: A new colormap that is lighter than the original.
    """
    cmap = cm.get_cmap(cmap, 256)  # Get the existing colormap
    new_colors = cmap(np.linspace(0, 1, 256))

    # Lighten the colors by blending with white
    white = np.array([1, 1, 1, 1])
    new_colors = (1 - factor) * new_colors + factor * white

    return ListedColormap(new_colors)

lighter_r_rev = lighten_cmap("inferno_r", factor=0.3)


def fig1(arr, diff):
    
    fig = plt.figure(figsize=(13, 4))
    
    # Grid layout:
    # [ WL1 | WL2 | cbar1 | spacer | diff | cbar2 ]
    gs = gridspec.GridSpec(
        1, 6,
        width_ratios=[1, 1, 0.03, 0.2, 1, 0.03],  # spacer + two colorbars
        wspace=0.15,
        figure=fig
    )
    
    # Axes
    ax1 = fig.add_subplot(gs[0, 0])  # first warming level
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)  # second warming level
    cax1 = fig.add_subplot(gs[0, 2])  # colorbar for WL plots
    ax_diff = fig.add_subplot(gs[0, 4])  # difference plot
    cax2 = fig.add_subplot(gs[0, 5])  # colorbar for diff plot
    
    # --- Plot first two warming levels ---
    sim_name = arr.sim.item().split('_')[2]
    for i, (ax, wl) in enumerate(zip([ax1, ax2], arr.warming_level.values[:2])):
        pcm = arr.sel(warming_level=wl).plot(ax=ax, add_colorbar=False, cmap=lighter_r_rev)
        ax.set_title(f"Avg Max Temp for Sim\n{sim_name} at WL {wl}")
        if i > 0:
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)
        if i != 1:
            # ax.set_xlabel("")
            pass
    
    # Shared colorbar hugging the 2nd plot
    cb1 = fig.colorbar(pcm, cax=cax1)
    cb1.set_label("Max Temp (°F)")
    
    # --- Difference plot ---
    pcm_diff = diff.plot(ax=ax_diff, add_colorbar=False, cmap='RdBu_r', vmin=-6, vmax=6)
    ax_diff.set_title(
        f"Difference Plot:\nWL {arr.warming_level.values[-1]} - WL {arr.warming_level.values[0]}"
    )
    ax_diff.tick_params(labelleft=False)
    ax_diff.set_ylabel("")
    # ax_diff.set_xlabel("")
    
    # Second colorbar for the difference plot
    cb2 = fig.colorbar(pcm_diff, cax=cax2)
    cb2.set_label("Δ Max Temp (°F)")
    
    fig.tight_layout()
    plt.show()



def fig2(arrs):
    fig = plt.figure(figsize=(13, 4))

    # Grid layout:
    # [ min | mean | max | cbar ]

    shared_vmin = -20
    shared_vmax = 20
    gs = gridspec.GridSpec(
        1, 4,
        width_ratios=[1, 1, 1, 0.03],  # spacer + two colorbars
        wspace=0.15,
        figure=fig
    )

    # Axes
    ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    ax2 = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree()) 
    ax3 = fig.add_subplot(gs[0, 2], projection=ccrs.PlateCarree())  
    cax1 = fig.add_subplot(gs[0, 3])  # colorbar 


    pcm = arrs[0].prec.plot(
        ax=ax1,
        x="lon",
        y="lat",
        transform=ccrs.PlateCarree(),
        cmap="BrBG",
        vmin=shared_vmin,
        vmax=shared_vmax,
        add_colorbar=False,
    )

    pcm = arrs[1].prec.plot(
        ax=ax2,
        x="lon",
        y="lat",
        transform=ccrs.PlateCarree(),
        cmap="BrBG",
        vmin=shared_vmin,
        vmax=shared_vmax,
        add_colorbar=False
    )

    pcm = arrs[2].prec.plot(
        ax=ax3,
        x="lon",
        y="lat",
        transform=ccrs.PlateCarree(),
        cmap="BrBG",
        vmin=shared_vmin,
        vmax=shared_vmax,
        add_colorbar=False
    )

    ax1.set_title('Multi-model minumum change')
    ax2.set_title('Multi-model mean change')
    ax3.set_title('Multi-model maximum change')

    # Shared colorbar hugging the 2nd plot
    cb1 = fig.colorbar(pcm, cax=cax1)
    cb1.set_label("Δ Precipitation (mm/month)")

    for ax in [ax1, ax2, ax3]:
        ax.set_extent([-125, -105, 28, 50])
        ax.add_feature(cfeature.STATES.with_scale("10m"), edgecolor="black",
                linewidth=0.1)
        ax.add_feature(cfeature.COASTLINE.with_scale("10m"), edgecolor="black")


    fig.tight_layout()
    plt.show()






### GWL illustration setup ###

colors = {'ssp119': "#00a9cf", 'ssp126': "#003466", 'ssp245': "#f69320", 'ssp370': "#df0000", 'ssp585': "#980002",
            'hist':'#222222'}

scenarios = ['ssp585','ssp370', 'ssp245']  # Order matters for stacking

def _load_gwl_illustration_data():
    ssp119_data = read_csv_file(SSP119_FILE, index_col="Year")
    ssp126_data = read_csv_file(SSP126_FILE, index_col="Year")
    ssp245_data = read_csv_file(SSP245_FILE, index_col="Year")
    ssp370_data = read_csv_file(SSP370_FILE, index_col="Year")
    ssp585_data = read_csv_file(SSP585_FILE, index_col="Year")
    hist_data = read_csv_file(HIST_FILE, index_col="Year")

    ssp_data = {
    'ssp119': ssp119_data,
    'ssp126': ssp126_data,
    'ssp245': ssp245_data,
    'ssp370': ssp370_data,
    'ssp585': ssp585_data}
    return hist_data, ssp_data

### data loading ###

def _load_warming_trajectories():
    data_interface = DataInterface()
    df = data_interface.data_catalog.df

    columns_of_interest = ['activity_id', 'source_id', 'experiment_id', 'member_id']
    unique_combinations = df[columns_of_interest].drop_duplicates()
    simulations_df = unique_combinations.reset_index(drop=True)

    warming_trajectories = read_csv_file(
            GWL_1850_1900_TIMEIDX_FILE, index_col="time", parse_dates=True
        )
    return warming_trajectories, simulations_df

def _filter_warming_trajectories(simulations_df, warming_trajectories, activity):
    columns_to_keep = []
    
    # Filter simulations_df for the specific activity
    activity_simulations = simulations_df[simulations_df['activity_id'] == activity]
    
    # Iterate through each row in the filtered simulations_df
    for _, row in activity_simulations.iterrows():
        # Construct the column name pattern
        column_pattern = f"{row['source_id']}_{row['member_id']}_{row['experiment_id']}"
        
        # Find matching columns in warming_trajectories
        matching_columns = [col for col in warming_trajectories.columns if column_pattern in col]
        
        # Add matching columns to our list
        columns_to_keep.extend(matching_columns)

    # Create a new DataFrame with only the relevant columns
    filtered_trajectories = warming_trajectories[columns_to_keep]
    
    return filtered_trajectories


###helpers###

def plot_trajectories(ax, df, linestyle, scenario =None):




    if scenario:
        cols_to_plot = [col for col in df.columns if scenario in col]
    else:
        cols_to_plot = df.columns


    for column in cols_to_plot:
        scenario = next((s for s in scenarios if s in column), None)
        if scenario:
            ax.plot(df.index, df[column], color=colors[scenario], linestyle=linestyle, alpha=0.3)


def plot_warming_level_period(ax,df,sim,wl):
    scenario=sim.split('_')[2]
    ts = df[df[sim]>2.0].index.min()
    start_ts = df.index.get_loc(ts)-(12*15)
    end_ts = df.index.get_loc(ts)+(12*14)
    subset = df.iloc[start_ts:end_ts]
    ax.plot(subset.index, subset[sim], color=colors[scenario], linestyle='-', alpha=1,linewidth=4)

###figures###

def gwl_fig1():


    warming_trajectories, simulations_df = _load_warming_trajectories()

    loca2_warming_trajectories = _filter_warming_trajectories(simulations_df, warming_trajectories, "LOCA2")

    # Set the style for the plots
    plt.style.use('seaborn-v0_8-whitegrid')
    # Define colors and scenarios


    fig = plt.figure(figsize=(15, 8))
    gs = gridspec.GridSpec(
        2, 2,
        width_ratios=[1, 1,],  # spacer + two colorbars
        height_ratios=[1, 1,],
        wspace=0.15,
        figure=fig
    )

    ax1 = fig.add_subplot(gs[0, 0])  
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax1, sharex=ax1)  
    ax3 = fig.add_subplot(gs[1, 0], sharey=ax1, sharex=ax1) 
    ax4 = fig.add_subplot(gs[1, 1], sharey=ax1, sharex=ax1)   
    ax1.set_xlim(np.datetime64("1950", 'Y'), np.datetime64("2100", 'Y'))

    ax1.set_title("Target-year planning")
    plot_trajectories(ax1,loca2_warming_trajectories,linestyle='-',scenario='ssp370')
    ax1.axvline(x=np.datetime64("2050", 'Y'), color='m', linestyle='--', alpha=0.7)
    legend_elements = [
    plt.Line2D([0], [0], color=colors['ssp370'], label='SSP3-7.0'),
    ]
    ax1.legend(handles=legend_elements, loc='upper left', fontsize=10)


    ax2.set_title("GWL planning")
    plot_trajectories(ax2,loca2_warming_trajectories,linestyle='-')
    ax2.axhline(y=2, color='m', linestyle='--', alpha=0.7)
    legend_elements = [
    plt.Line2D([0], [0], color=colors['ssp245'], label='SSP2-4.5'),
    plt.Line2D([0], [0], color=colors['ssp370'], label='SSP3-7.0'),
    plt.Line2D([0], [0], color=colors['ssp585'], label='SSP5-8.5'),
    ]
    ax2.legend(handles=legend_elements, loc='upper left', fontsize=10)


    ax3.set_title("Example target-year climatologies")
    plot_trajectories(ax3,loca2_warming_trajectories[['MIROC6_r2i1p1f1_ssp370','ACCESS-CM2_r2i1p1f1_ssp370']], linestyle='-')
    subset = loca2_warming_trajectories.loc[np.datetime64("2035", 'Y'):np.datetime64("2065", 'Y')]
    ax3.plot(subset.index, subset['MIROC6_r2i1p1f1_ssp370'], color=colors['ssp370'], linestyle='-', alpha=1,linewidth=4)
    ax3.plot(subset.index, subset['ACCESS-CM2_r2i1p1f1_ssp370'], color=colors['ssp370'], linestyle='-', alpha=1,linewidth=4)

    ax3.axvline(x=np.datetime64("2050", 'Y'), color='m', linestyle='--', alpha=0.7)



    ax4.set_title("Example GWL climatologies")
    plot_trajectories(ax4,loca2_warming_trajectories[['MIROC6_r2i1p1f1_ssp370','EC-Earth3_r3i1p1f1_ssp585']], linestyle='-')
    plot_warming_level_period(ax4,loca2_warming_trajectories,'MIROC6_r2i1p1f1_ssp370',2.0)
    plot_warming_level_period(ax4,loca2_warming_trajectories,'EC-Earth3_r3i1p1f1_ssp585',2.0) 
    ax4.axhline(y=2, color='m', linestyle='--', alpha=0.7)  



    #set title for figure
    fig.suptitle("LOCA2 and WRF Global Warming Trajectories", fontsize=16)
    plt.savefig("./gwl_figure1.png", dpi=300)

def gwl_fig2():

    warming_trajectories, simulations_df = _load_warming_trajectories()
    loca2_warming_trajectories = _filter_warming_trajectories(simulations_df, warming_trajectories, "LOCA2")

    hist_data, ssp_data = _load_gwl_illustration_data()

    # Set the style for the plots
    plt.style.use('seaborn-v0_8-whitegrid')

    df = loca2_warming_trajectories
    selected_scenario = 'ssp370'
    plot_ipcc=True
    plt.figure(figsize=(6,5))
    
    scenarios_to_plot = [selected_scenario] if selected_scenario else scenarios

                
    if plot_ipcc:
        context_data = hist_data
        context_data.index = pd.to_datetime(context_data.index, format='%Y')
        # plt.fill_between(context_data.index, context_data['5%'], context_data['95%'], 
        #                          color='#0000AA', alpha=0.5, zorder=8)
        plt.plot(context_data.index, context_data['Mean'], color='k', 
                         linewidth=2, linestyle='--',zorder=8)
        
        
        for scenario in ['ssp370']:           
            # Plot context data
            if scenario in ssp_data:
                context_data = ssp_data[scenario]
                context_data.index = context_data.index.astype('int')
                context_data.index = pd.to_datetime(context_data.index, format='%Y')
                plt.fill_between(context_data.index, context_data['5%'], context_data['95%'], 
                                 color=colors[scenario], alpha=0.4, zorder=4, label='SSP3-7.0 90% Range')
                plt.plot(context_data.index, context_data['Mean'], color=colors[scenario], 
                         linewidth=2, linestyle='--', label=f'{scenario.upper()} (IPCC Best Estimate)', zorder=8)
                
                
    for i, scenario in enumerate(scenarios_to_plot):
        if selected_scenario:
            plot_color='k'
        else:
            plot_color = colors[scenario]
            

        # Filter columns for the current scenario
        scenario_cols = [col for col in df.columns if scenario in col]
        if scenario_cols:
            # Plot each trajectory
            for col in scenario_cols:
                plt.plot(df.index, df[col], color='k', alpha=0.4, zorder=5+i, linewidth=0.8)
                


    # Customize the plot
    plt.title(f"LOCA2 & WRF Warming Trajectories", fontsize=16)
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Global Warming Level (°C)", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    for i, scenario in enumerate(scenarios_to_plot): 
        plt.plot([0], [0], color='k', label=f'{scenario} simulations'),

    plt.legend(loc='upper left', fontsize=10)
    
    # Set x-axis limits
    plt.xlim(np.datetime64("1950", 'Y'), np.datetime64("2100", 'Y'))
    
    #plt.axvline(x=np.datetime64("2047", 'Y'), color='m', linestyle='dotted')
    #plt.axhspan(ymin=1.6,ymax=3.3, color='m', alpha=0.1, zorder=2)
    
    plt.axhline(y=2, color='m', linestyle='dotted')
    #plt.axvline(x=np.datetime64("2037", 'Y'), color='m', linestyle='dashed',alpha=0.2)
    #plt.axvline(x=np.datetime64("2061", 'Y'), color='m', linestyle='dashed',alpha=0.2)

    plt.axvspan(xmin=np.datetime64("2037", 'Y'), xmax=np.datetime64("2061", 'Y'), color="m", alpha=0.1)
    plt.axvline(x=np.datetime64("2047", 'Y'), color='m', linestyle='dotted')
    
    plt.tight_layout()
    plt.savefig("./gwl_figure2.png", dpi=300)


