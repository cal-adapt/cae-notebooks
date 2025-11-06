import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import Colormap, ListedColormap
from matplotlib import cm

from climakitae.core.data_interface import (
    DataInterface,
    DataParameters,
    _get_variable_options_df,
    _get_user_options,
)
from climakitae.util.utils import read_csv_file

from climakitae.core.paths import (
    HIST_FILE,
    SSP119_FILE,
    SSP126_FILE,
    SSP245_FILE,
    SSP370_FILE,
    SSP585_FILE,
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
    pcm_diff = diff.plot(ax=ax_diff, add_colorbar=False, cmap='RdBu_r')
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




## 1.1 find the simulations in the catalog
data_interface = DataInterface()
gcms = data_interface.data_catalog.df.source_id.unique()
df = data_interface.data_catalog.df

columns_of_interest = ['activity_id', 'source_id', 'experiment_id', 'member_id']
unique_combinations = df[columns_of_interest].drop_duplicates()
simulations_df = unique_combinations.reset_index(drop=True)

## 1.2 Load the warming trajectories dataframe, columns for each simulation like "ACCESS-CM2_r3i1p1f1_ssp585"
## and rows for each month like "1860-01-01"
warming_trajectories = read_csv_file(
        "/home/jovyan/src/climakitae/climakitae/data/gwl_1850-1900ref_timeidx.csv", index_col="time", parse_dates=True
    )

def filter_warming_trajectories(simulations_df, warming_trajectories, activity):
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

# Filter for LOCA2 simulations
loca2_warming_trajectories = filter_warming_trajectories(simulations_df, warming_trajectories, "LOCA2")

# Filter for WRF simulations
wrf_warming_trajectories = filter_warming_trajectories(simulations_df, warming_trajectories, "WRF")


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

ssp119_data = read_csv_file(SSP119_FILE, index_col="Year")
ssp126_data = read_csv_file(SSP126_FILE, index_col="Year")
ssp245_data = read_csv_file(SSP245_FILE, index_col="Year")
ssp370_data = read_csv_file(SSP370_FILE, index_col="Year")
ssp585_data = read_csv_file(SSP585_FILE, index_col="Year")
hist_data = read_csv_file(HIST_FILE, index_col="Year")

# Set the style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

# Define colors and scenarios

colors = {'ssp119': "#00a9cf", 'ssp126': "#003466", 'ssp245': "#f69320", 'ssp370': "#df0000", 'ssp585': "#980002",
          'hist':'#222222'}
scenarios = ['ssp585','ssp370', 'ssp245']  # Order matters for stacking

# Function to combine historical and scenario data
def combine_hist_scenario(hist_data, scenario_data):
    combined = pd.concat([hist_data, scenario_data])
    combined = combined[~combined.index.duplicated(keep='first')]
    return combined

# Combine historical and scenario data
ssp_data = {
    'ssp119': ssp119_data,
    'ssp126': ssp126_data,
    'ssp245': ssp245_data,
    'ssp370': ssp370_data,
    'ssp585': ssp585_data
}

def plot_trajectories(df, title, selected_scenario=None, plot_ipcc=False):
    plt.figure(figsize=(4,3))
    
    scenarios_to_plot = [selected_scenario] if selected_scenario else scenarios

                
    if plot_ipcc:
        # context_data = hist_data
        # context_data.index = pd.to_datetime(context_data.index, format='%Y')
        # plt.fill_between(context_data.index, context_data['5%'], context_data['95%'], 
        #                          color='#0000AA', alpha=0.5, zorder=8)
        # plt.plot(context_data.index, context_data['Mean'], color='k', 
        #                  linewidth=2, linestyle='--',zorder=8)
        
        
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

        # Filter columns for the current scenario
        scenario_cols = [col for col in df.columns if scenario in col]
        if scenario_cols:
            # Plot each trajectory
            for col in scenario_cols:
                plt.plot(df.index, df[col], color='k', alpha=0.4, zorder=5+i, linewidth=0.8)
                


    # Customize the plot
    plt.title(f"{title} & WRF Warming Trajectories", fontsize=16)
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
    plt.axvline(x=np.datetime64("2037", 'Y'), color='m', linestyle='dashed',alpha=0.2)
    plt.axvline(x=np.datetime64("2061", 'Y'), color='m', linestyle='dashed',alpha=0.2)

    #plt.axvspan(xmin=np.datetime64("2040", 'Y'), xmax=np.datetime64("2060", 'Y'), color="m", alpha=0.1)
    
    
    plt.tight_layout()
    plt.savefig(f"{title}_fig.png")
    plt.show()

# Example usage:
# Plot all scenarios for LOCA2
#plot_trajectories(loca2_warming_trajectories, "LOCA2")

loca2_warming_trajectory = 

plot_trajectories(loca2_warming_trajectories, "LOCA2", selected_scenario='ssp370', plot_ipcc=True)



