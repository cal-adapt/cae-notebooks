# Cal-Adapt: Analytics Engine Notebooks

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

The `cae-notebooks` repository consists of Jupyter notebooks that provide step-by-step 
functionality to access, analyze, and visualize available data from California's Fifth Climate Change Assessment. Tools included in the notebooks provide examples for how to work with the **historical and projection climate model data** on the platform, and demonstrate how to move from the climate variables provided through the Analytics Engine to **actionable information that can inform decision-making and evaluate risks**.

These notebooks are designed to be used as-is or as a starting point for users to adapt to their specific needs, workflows, and applications. 

> [!WARNING]
> The `climakitae` package (a crucial dependency of almost all notebooks) is under active development. APIs may change between versions and notebooks in this repository may change accordingly.

Please refer to [climakitae](https://github.com/cal-adapt/climakitae) for installation instructions, documentation, and updates.

## Navigating

New to [Cal-Adapt: Analytics Engine](https://analytics.cal-adapt.org/), JupyterHub, or `climakitae`? Take a peek our [navigation guide](https://github.com/cal-adapt/cae-notebooks/blob/main/AE_navigation_guide.ipynb)

Want a basic example of how to retrieve, visualize, and export data using climakitae? Check out the [`interactive_data_access_and_viz.ipynb`](https://github.com/cal-adapt/cae-notebooks/blob/main/data-access/interactive_data_access_and_viz.ipynb) notebook.

Looking for a specific type of notebook?  
Notebooks are organized by their theme:
- [**Data access notebooks**](https://github.com/cal-adapt/cae-notebooks/tree/improve/readme/data-access) highlight various ways of accessing California's Fifth Climate Assessment data, including:
    - [`interactive_data_and_viz.ipynb`](https://github.com/cal-adapt/cae-notebooks/blob/improve/readme/data-access/interactive_data_access_and_viz.ipynb)
    - [`renewables_data_access.ipynb`](https://github.com/cal-adapt/cae-notebooks/blob/improve/readme/data-access/renewables_data_access.ipynb)

- [**Climate profile notebooks**](https://github.com/cal-adapt/cae-notebooks/tree/main/climate-profiles) generate climate profiles and explore profile generation methodology:
    - [`custom_climate_profiles.ipynb`](https://github.com/cal-adapt/cae-notebooks/blob/main/climate-profiles/custom_climate_profiles.ipynb)
    - [`typical_meteorological_year_methodology.ipynb`](https://github.com/cal-adapt/cae-notebooks/blob/c90a214fcf653713392ddbcd68d4a98618a4df8b//climate-profiles/typical_meteorological_year_methodology.ipynb)

- [**Analysis notebooks**](https://github.com/cal-adapt/cae-notebooks/tree/improve/readme/analysis) investigate a climate data topic but have no specific tools associated with the notebooks, including:
    - [`warming_levels.ipynb`](https://github.com/cal-adapt/cae-notebooks/blob/improve/readme/analysis/warming_levels.ipynb)
    - [`timeseries_transformations.ipynb`](https://github.com/cal-adapt/cae-notebooks/blob/improve/readme/analysis/timeseries_transformations.ipynb)
    - [`threshold_exceedance.ipynb` ](https://github.com/cal-adapt/cae-notebooks/blob/improve/readme/analysis/threshold_exceedance.ipynb)

- [**Collaborative notebooks**](https://github.com/cal-adapt/cae-notebooks/tree/improve/readme/collaborative) were co-produced with industry partners for a specific application, and may be of interest to all users, including:
    - [`vulnerability_assessment.ipynb`](https://github.com/cal-adapt/cae-notebooks/tree/improve/readme/collaborative/IOU/vulnerability_assessment)
    - [`degree_days.ipynb`](https://github.com/cal-adapt/cae-notebooks/blob/improve/readme/collaborative/DFU/degree_days.ipynb)

- [**In progress notebooks**](https://github.com/cal-adapt/cae-notebooks/tree/improve/readme/work-in-progress) are currently in development for a specific application, and may have frequent updates before they are moved to another folder, including:
    - [`typical_meteorological_year_methodology.ipynb`](https://github.com/cal-adapt/cae-notebooks/blob/improve/readme/work-in-progress/typical_meteorological_year_methodology.ipynb)


## About Cal-Adapt


[Cal-Adapt](https://cal-adapt.org/) encompasses a range of research and development efforts designed to provide access to California climate data. Cal-Adapt includes the Cal-Adapt: Analytics Engine and Cal-Adapt: Data Explorer. The Cal-Adapt: Analytics Engine is designed for complex and detailed analyses, requiring extensive data and technical or scientific expertise. The Cal-Adapt: Data Explorer is particularly useful for quick access to interactive maps and tools, providing a valuable overview of how climate change may impact various regions of the state. In our efforts to provide actionable and transparent insight using climate data, the following documentation is available:
- [**`climakitae` API documentation**](https://climakitae.readthedocs.io/en/latest/)
- [**Guidance for working with climate data**](https://analytics.cal-adapt.org/guidance/)
- [**Contribution: development guidelines**](https://climakitae.readthedocs.io/en/latest/contribute.html)

## Contributing

We welcome contributions! Please see our [contributing guidelines](https://climakitae.readthedocs.io/en/latest/contribute.html) for details on:

- 🐛 Reporting bugs
- 💡 Requesting features  
- 🔧 Submitting code changes
- 📖 Improving documentation

## License

This project is licensed under the BSD 3-Clause License - see the [LICENSE](LICENSE) file for details.

## Related Projects

- [**climakitae**](https://github.com/cal-adapt/climakitae) - API for data access, exploration, manipulation, and visualization
- [**climakitaegui**](https://github.com/cal-adapt/climakitaegui) - Interactive GUI tools for climakitae

## Support

- 📧 **Email**: [analytics@cal-adapt.org](mailto:analytics@cal-adapt.org)
- 🐛 **Issues**: [GitHub Issues](https://github.com/cal-adapt/cae-notebooks/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/cal-adapt/cae-notebooks/discussions)

---

## Contributors

[![Contributors](https://contrib.rocks/image?repo=cal-adapt/cae-notebooks)](https://github.com/cal-adapt/cae-notebooks/graphs/contributors)