# Cal-Adapt: Analytics Engine Notebooks

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

The `cae-notebooks` repository consists of Jupyter notebooks that provide step-by-step 
functionality to access, analyze, and visualize climate data available on the [Cal-Adapt: Analytics Engine](https://analytics.cal-adapt.org/). Tools included in the notebooks provide examples for how to work with the **historical and projection climate model data** on the platform, and demonstrate how to move from the climate variables provided through the Analytics Engine to **actionable information that can inform decision-making and evaluate risks**.

These notebooks are designed to be used as-is or as a starting point for users to adapt to their specific needs, workflows, and applications. 

> [!WARNING]
> The `climakitae` package (a crucial dependency of almost all notebooks) is under active development. APIs may change between versions and notebooks in this repository may change accordingly.

Please refer to [climakitae](https://github.com/cal-adapt/climakitae) for installation instructions, documentation, and updates.

## Navigating

New to Cal-Adapt: Analytics Engine, JupyterHub, or `climakitae`? Take a peek our [navigation guide](https://github.com/cal-adapt/cae-notebooks/blob/main/AE_navigation_guide.ipynb)

Want a basic example of how to use `climakitae`? Check out [this notebook](https://github.com/cal-adapt/cae-notebooks/blob/main/data-access/interactive_data_access_and_viz.ipynb)

Looking for a specific type notebook?

Notebooks are organized by their theme:<br>
- **Data access notebooks** highlight various ways of accessing California's Fifth Climate Assessment data
    - Including `interactive_data_and_viz.ipynb` and `renewables_data_access.ipynb`<br>
- **Analysis notebooks** investigate a climate data topic but have no specific tools associated with the notebooks
    - Including `warming_levels.ipynb`, `timeseries_transformations.ipynb`, and `threshold_exceedance.ipynb` <br>
- **Collaborative notebooks** were co-produced with industry partners for a specific application, and may be of interest to all users
    - Including `vulnerability_assessment.ipynb` and `degree_days.ipynb`<br>
- **In progress notebooks** are currently in development for a specific application, and may have frequent updates before they are moved to another folder
    - Including `typical_meteorological_year_methodology.ipynb`


## About Cal-Adapt

Climakitae is developed as part of the [Cal-Adapt Analytics Engine](https://analytics.cal-adapt.org), a platform for California climate data and tools. Cal-Adapt provides access to cutting-edge climate science to support adaptation planning and decision-making.


## Documentation

| Resource | Description |
|----------|-------------|
| [**Getting Started**](https://github.com/cal-adapt/cae-notebooks/blob/main/getting_started.ipynb) | Analytics Engine navigation |
| [**API Reference**](https://climakitae.readthedocs.io/en/latest/) | `climakitae` API documentation |
| [**Contributing**](https://climakitae.readthedocs.io/en/latest/contribute.html) | Development guidelines |

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