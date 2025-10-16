import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import Colormap, ListedColormap
from matplotlib import cm


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
    pcm_diff = diff.plot(ax=ax_diff, add_colorbar=False, cmap=lighter_r_rev)
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