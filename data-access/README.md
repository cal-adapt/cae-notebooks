Data Access Notebooks
=====================

Analytics Engine catalog data can be accessed in multiple different ways, each with distinct benefits. These notebooks demonstrate these different ways. The appropriate method for you will depend on your unique use case and workflow.

[basic_data_access.ipynb](basic_data_access.ipynb): Overview to accessing, spatially and temporally subsetting, and exporting climate data from the AE data catalog using helper functions in climakitae.

[interactive_data_access_and_viz.ipynb](interactive_data_access_and_viz.ipynb): Retrieve, subset, and visualize data options using a simple and intuitive interactive graphical user interface (GUI). This notebook leverages functionality from both climakitae and our visualizations library climakitaegui.

[outside_AE_data_access.ipynb](outside_AE_data_access.ipynb): Access and export AE catalog data without using the helper functions from AE's python libraries. This notebook instead leverages the python library intake for interfacing with the data catalog. This method may be useful for users accessing the data outside of the Analytics Engine and who don't want to set up climakitae in their computational environment.

[renewables_data_access](renewables_data_access.ipynb): Data access for our derived renewables data is still a work in progress as we build a data catalog and continue generating data products. For the time being, here's the best way to access this data using python.

[weather_station_data_access](weather_station_data_access.ipynb): Access and plot quality controlled, standardized historical weather station data. This notebook leverages the Python library `intake` for interfacing with the historical weather station data catalog, which is separate from the larger AE catalog. Eventually, data access will be fully integrated into `climakitae`, but for now, this is the best way to access the data.