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

New to [Cal-Adapt: Analytics Engine](https://analytics.cal-adapt.org/), JupyterHub, or `climakitae`? Start with our [navigation guide](https://github.com/cal-adapt/cae-notebooks/blob/main/AE_navigation_guide.ipynb).

This repository contains the actively maintained notebooks for the Analytics Engine:

**Data access**
- [`basic_data_access.ipynb`](https://github.com/cal-adapt/cae-notebooks/blob/main/basic_data_access.ipynb) — Access, subset, and export climate data using `climakitae`
- [`renewables_data_access.ipynb`](https://github.com/cal-adapt/cae-notebooks/blob/main/renewables_data_access.ipynb) — Access and plot derived renewables data products
- [`weather_station_data_access.ipynb`](https://github.com/cal-adapt/cae-notebooks/blob/main/weather_station_data_access.ipynb) — Access quality controlled historical weather station data

**Analysis**
- [`derived_variables_demo.ipynb`](https://github.com/cal-adapt/cae-notebooks/blob/main/derived_variables_demo.ipynb) — Define and use custom derived climate metrics with `register_user_function`
- [`warming_level_methods.ipynb`](https://github.com/cal-adapt/cae-notebooks/blob/main/warming_level_methods.ipynb) — Compare SSP time-based and Global Warming Levels approaches

**Threshold tools**
- [`threshold_tools.ipynb`](https://github.com/cal-adapt/cae-notebooks/blob/main/threshold_tools.ipynb) — Define extreme events and analyze their likelihood using extreme value theory

Looking for older or experimental notebooks? See the [cae-archives](https://github.com/cal-adapt/cae-archives) repository.


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
