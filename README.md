Cal-Adapt: Analytics Engine Notebooks
====================================

The cae-notebooks repository consists of Jupyter notebooks that provide step-by-step functionality to access, analyze, and visualize climate data availale on the Cal-Adapt: Analytics Engine. Tools included in the notebooks provide examples for how to work with both the **historical and projection data** on the platform, and demonstrate how to move from the climate variables provided through the Analytics Engine to **actionable information that can inform decision-making and evaluate risks**.

Notebooks are organized by similar-themed notebooks:<br>
- **Exploratory notebooks** investigate a climate data topic but have no specific tools associated with the notebooks
    - Including *model_uncertainty.ipynb* and *internal_variability.ipynb* <br>
- **Tool notebooks** highlight a specific tool
    - Including *warming_levels.ipynb*, *timeseries_transformations.ipynb*, and *threshold_exceedance.ipynb*<br>
- **Collaborative notebooks** were co-produced with stakeholders for a specific application, and may be of interest to all users
    - Including *localization_methodology.ipynb*, *annual_consumption_model.ipynb* and *station_hourly_profiles.ipynb (8760s)*

New to the Analytics Engine? Try `getting_started.ipynb` for a step-through of the basic functionality.

The notebooks are designed to be used as-is or serve as a starting point to adapt to specific needs, workflows, and particular applications. 


----
Example notebooks in the **tools** folder:
- `timeseries_transformations.ipynb`:
    Demonstrates pulling up a toolkit for examining and processing time-series data. Motivated by Hourly climate  use case (for future time periods), for inputs into production cost, energy load forecasting, and other models.
- `threshold_basics.ipynb`:
    Demonstrates some functionality for calculating extreme value statistics. Motivated by the 'threshold-based analytics' use case for asset-by-asset vulnerability assessments and updating design standards.
