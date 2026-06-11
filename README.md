# Cal-Adapt: Analytics Engine Notebooks

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This repository contains the actively maintained notebooks for the Analytics Engine. It consists of Jupyter Notebooks that provide step-by-step functionality to access, analyze, and visualize available data from California's Fifth Climate Change Assessment. Tools included in the notebooks provide examples for how to work with the **historical and projection climate model data** on the platform, and demonstrate how to move from the climate variables provided through the Analytics Engine to **actionable information that can inform decision-making and evaluate risks**. 

These notebooks are designed to be used as-is or as a starting point for users to adapt to their specific needs, workflows, and applications. 

> [!WARNING]
> The `climakitae` package (a crucial dependency of almost all notebooks) is under active development. APIs may change between versions and notebooks in this repository may change accordingly.

Please refer to [climakitae](https://github.com/cal-adapt/climakitae) for installation instructions, documentation, and updates.

## Notebooks 
Each notebook is labeled with a type to help you find the right resource for your needs:

| Type | Description |
|------|-------------|
| **Data Access** | Data Access notebooks demonstrate how to retrieve, subset, and visualize existing climate data and derived data products using available tools and workflows. |
| **Data Generation** | Data Generation notebooks show how to create new custom data products, profiles, or metrics by transforming and combining source data. |
| **Tool/Methods** | Tool/Methods notebooks teach specific tools, methodologies, or analytical approaches, with hands-on examples of how to apply them. |

- [`basic_data_access.ipynb`](https://github.com/cal-adapt/cae-notebooks/blob/main/basic_data_access.ipynb): Access, subset, and export climate data using `climakitae`. Notebook type: **Data Access**
- [`custom_climate_profiles.ipynb`](https://github.com/cal-adapt/cae-notebooks/blob/main/custom_climate_profiles.ipynb): Generate annualized hourly climate profiles for energy system modeling and planning. Notebook type: **Data Generation**
- [`derived_variables_demo.ipynb`](https://github.com/cal-adapt/cae-notebooks/blob/main/derived_variables_demo.ipynb): Define and use custom derived climate metrics with `register_user_function`. Notebook type: **Tool/Methods**
- [`renewables_data_access.ipynb`](https://github.com/cal-adapt/cae-notebooks/blob/main/renewables_data_access.ipynb): Access and plot derived renewables data products. Notebook type: **Data Access**
- [`threshold_tools.ipynb`](https://github.com/cal-adapt/cae-notebooks/blob/main/threshold_tools.ipynb): Define extreme events and analyze their likelihood using extreme value theory. Notebook type: **Tool/Methods**
- [`vulnerability_assessment.ipynb`](https://github.com/cal-adapt/cae-notebooks/blob/main/vulnerability_assessment.ipynb): Generate data-informed answers for vulnerability assessments through a customizeable metric builder. Notebook type: **Data Generation**
- [`warming_level_methods.ipynb`](https://github.com/cal-adapt/cae-notebooks/blob/main/warming_level_methods.ipynb): Compare SSP time-based and Global Warming Levels approaches. Notebook type: **Tool/Methods**
- [`weather_station_data_access.ipynb`](https://github.com/cal-adapt/cae-notebooks/blob/main/weather_station_data_access.ipynb): Access quality controlled historical weather station data. Notebook type: **Data Access**

Looking for older or experimental notebooks? See the [cae-archives](https://github.com/cal-adapt/cae-archives) repository.


## About Cal-Adapt

[Cal-Adapt](https://cal-adapt.org/) encompasses a range of research and development efforts designed to provide access to California climate data. Cal-Adapt includes the Cal-Adapt: Analytics Engine and Cal-Adapt: Data Explorer. 
- **The Cal-Adapt: Analytics Engine** is designed for complex and detailed analyses, requiring extensive data and technical or scientific expertise.
- **The Cal-Adapt: Data Explorer** is particularly useful for quick access to interactive maps and tools, providing a valuable overview of how climate change may impact various regions of the state.

We also maintain an open source python library, `climakitae`, which can be used to query, process, and analyze downscaled climate projections for California. See the package on GitHub [here](https://github.com/cal-adapt/climakitae), and the documentation [here](https://cal-adapt.github.io/climakitae/dev/). 

The Cal Adapt team produces a lot of notebooks outside of the ones actively maintained here. If you're looking for a notebook you've previously used on the Analytics Engine that is not available in this repository, check out our archive repository, [cae-archives](https://github.com/cal-adapt/cae-archives), for our full catalog of notebooks. Note that these notebooks are not actively maintained by the team. 

## Contributing 
Find an issue with one of our notebooks? Please let us know by submitting an Issue in this repository! 

## License

This project is licensed under the BSD 3-Clause License - see the [LICENSE](LICENSE) file for details.

## Support

- 📧 **Email**: [analytics@cal-adapt.org](mailto:analytics@cal-adapt.org)
- 🐛 **Issues**: [GitHub Issues](https://github.com/cal-adapt/cae-notebooks/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/cal-adapt/cae-notebooks/discussions)

---

## Contributors

[![Contributors](https://contrib.rocks/image?repo=cal-adapt/cae-notebooks)](https://github.com/cal-adapt/cae-notebooks/graphs/contributors)
