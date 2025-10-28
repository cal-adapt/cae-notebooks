# User Contributions

This folder contains notebooks and projects contributed by users of the Cal-Adapt: Analytics Engine (CAE / AE). We welcome and appreciate community contributions that demonstrate innovative uses of climate data and analytics.

Many tools in this folder have been designed around the `climakitae` package: a powerful Python toolkit for climate data analysis and retrieval from the Cal-Adapt: Analytics Engine. If you are new to this package, please head over to the  [`climakitae` Github Repository](https://github.com/cal-adapt/climakitae) to learn about it.

> [!NOTE]
> While we are happy to host user contributed code, the notebooks and projects in this folder have not been vetted by the CAE team for best practices, coding standards, and adherance to CAE guidance. 

## Using These Contributions

When using code from this folder:

- Review the code carefully before running it in your environment
- Test thoroughly with your specific use case
- Consider the code as examples or starting points rather than production-ready solutions
- Feel free to adapt and improve upon the contributed work

> [!NOTE]
> Be aware that functions and scientific approaches may differ from `climakitae` or AE best practices.

## 📋 Requirements for All Submissions

Thank you for your interest in contributing to the CAE Notebooks repository! These guidelines ensure that all contributions maintain high quality and are immediately usable by the community.

### 1. Minimum Viable Product (MVP)
- **Keep it focused**: Submit only the essential code needed to demonstrate your solution
- **No extras**: Remove any experimental features, debugging code, or unnecessary dependencies
- **Clear purpose**: Each contribution should solve one specific problem or demonstrate one technique
- **Clean code**: Remove commented-out code blocks and unused imports

### 2. Contact Information
Every submission **must** include contact information for support and questions:
- Add a markdown cell at the beginning of your notebook with:
  ```
  **Author**: [Your Name]
  **Email**: [Your Email]
  **Team/Department**: [Optional]
  **Date**: [Submission Date]
  ```
- This ensures users can reach out if they encounter issues or have questions

> [!WARNING]
> The `climakitae` and `cae-notebooks` development teams will not respond to requests for
> assistance with user-submitted notebooks. You must contact the developer of the submitted
> code for assistance.
>
> Your code will not be accepted for submission without appropriate contact information.

### 3. Environment Compatibility
#### A. JupyterHub
If your code was developed on or with the Cal-Adapt: Analytics Engine JupyterHub, your code **must** run on the default environment without modifications, unless explicitly stated.
- ✅ Test your notebook on the JupyterHub before submission
- ✅ Use only pre-installed libraries or include installation instructions
- ✅ Ensure all file paths are relative or use environment variables
- ❌ No hardcoded absolute paths
- ❌ No dependencies on local files not included in the submission
- ❌ No requirements for special permissions or access tokens

#### B. Other Environments
If your code was developed external to the Cal-Adapt: Analytics Engine JupyterHub, please provide explicit instructions on how to set up the environment necessary to run your project.
- ✅ Test your environment install instructions before submission
- ✅ Include installation instructions
- ✅ Ensure all file paths are relative or use environment variables
- ❌ No hardcoded absolute paths
- ❌ No dependencies on local files not included in the submission
- ❌ No requirements for special permissions or access tokens

#### Nice to Have's
In addition to the above, some nice to have metrics for notebooks include:
- Memory usage
- Cell-by-cell Run times
- Infrastructure notes (What kind of machine? How many cores? How much RAM?)

### 4. Output Management
- **Clear all outputs** from cells that:
  - Generate large datasets
  - Contain sensitive information
  - Produce verbose logging
  - Create intermediate results
- **Keep outputs only for**:
  - The primary demonstration notebook showing usage examples
  - Visualization outputs that help understand the results
  - Key metrics or summary statistics that validate the approach
- Use `Cell → All Output → Clear` in Jupyter before submission (except for demonstration outputs)

### 5. Use Existing `climakitae` Functionality
Your code **must** leverage existing `climakitae` capabilities rather than re-inventing them:
- ✅ **Do use** `climakitae` functions for:
  - Data retrieval and loading
  - Common data transformations
  - Standard visualization methods
  - Statistical calculations already available
  - Unit conversions and data processing utilities
- ❌ **Don't re-invent** functionality that already exists in `climakitae`
- 📚 Review the [climakitae documentation](https://climakitae.readthedocs.io/) before writing custom functions
- 💡 If you find yourself writing utility functions, first check if `climakitae` already provides them
- 🔧 If `climakitae` is missing functionality you need, consider contributing to `climakitae` directly

**Examples of what to avoid:**
- Writing custom functions to load Cal-Adapt data (use `climakitae.DataInterface`)
- Creating custom visualization functions for standard plots (use `climakitae` plotting methods)
- Implementing unit conversions that `climakitae` already handles

## 📝 Submission Checklist

Before submitting, ensure:
- [ ] **No duplication of existing `climakitae` functionality**
- [ ] **Uses `climakitae` functions wherever applicable**
- [ ] Code follows MVP principle - minimal and focused
- [ ] Contact information is included at the top of the notebook
- [ ] Tested successfully on Analytics Engine (AE) JupyterHub
- [ ] All unnecessary outputs have been cleared
- [ ] Documentation clearly explains what the code does
- [ ] Example usage is demonstrated with preserved outputs
- [ ] No sensitive data or credentials are included
- [ ] File paths are relative or environment-based

## 🚀 How to Submit

1. **Fork the repository**
   - Go to the `cae-notebooks` repository
   - Click the "Fork" button to create your own copy

2. **Create your contribution**
   - Clone your forked repository locally
   - Create a new branch for your contribution: `git checkout -b your-feature-name`
   - Add your notebook(s) following all guidelines above
   - Commit your changes with clear, descriptive messages

3. **Test thoroughly**
   - Ensure your code runs on the Analytics Engine (AE) JupyterHub or installation instructions have been provided.
   - Verify all requirements are met using the checklist above

4. **Open a Pull Request (PR)**
   - Push your branch to your forked repository
   - Navigate to the original `cae-notebooks` repository
   - Click "New Pull Request"
   - Select your fork and branch as the source
   - Provide a clear description of your contribution including:
     - What problem it solves
     - How to use it
     - Any dependencies or special requirements
   - Reference any related issues if applicable

5. **PR Review Process**
   - Maintainers will review your submission
   - Address any feedback or requested changes
   - Once approved, your contribution will be merged

## 📂 File Structure

All submissions must follow this organizational structure:

```
user-contributions/
└── [your-organization]/
    └── [project-name]/
        └── [project-files]
```

**For single notebook submissions:**
```
user-contributions/
└── your-organization/
    └── your-project-name/
        ├── your_notebook.ipynb
        └── README.md (optional but recommended)
```

**For multi-file submissions:**
```
user-contributions/
└── your-organization/
    └── your-project-name/
        ├── main_notebook.ipynb
        ├── src/
        │   └── helper_functions.py
        ├── data/
        │   └── sample_data.csv (if needed)
        ├── pyproject.toml (recommended for dependencies)
        └── README.md (required)
```

**Dependency Management:**
- Use `pyproject.toml` for specifying project dependencies and metadata
- This follows modern Python packaging standards
- Example minimal `pyproject.toml`:
  ```toml
  [project]
  name = "your-project-name"
  version = "0.1.0"
  dependencies = [
      "pandas>=1.3.0",
      "numpy>=1.21.0",
      "climakitae>=1.0.0",
  ]
  ```

**Code Organization:**
- Place all helper functions and modules in a `src/` directory
- This follows Python best practices for source code organization
- Import from `src/` in your notebooks: `from src.helper_functions import my_function`

**Examples:**
- `user-contributions/ucla/wildfire-analysis/wildfire_risk_notebook.ipynb`
- `user-contributions/lbl/energy-modeling/building_energy_analysis.ipynb`
- `user-contributions/cal-adapt/extreme-heat/heat_wave_projections.ipynb`

**Naming conventions:**
- Use lowercase for organization and project names
- Use hyphens (-) instead of spaces or underscores in folder names
- Keep project names descriptive but concise

## ⚠️ Common Pitfalls to Avoid

- **Over-engineering**: Don't include "nice-to-have" features
- **Missing documentation**: Always explain what your code does and how to use it
- **Untested code**: Never submit without testing on the target environment
- **Large files**: Don't include large datasets; provide download instructions instead
- **Personal information**: Remove any personal or sensitive data from code and outputs
- **Duplicating `climakitae`**: Don't rewrite functionality that already exists in the library

## 💡 Best Practices

1. **Documentation First**: Start with clear documentation of what problem you're solving
2. **Examples Matter**: Include at least one working example with visible output
3. **Error Handling**: Include basic error handling for common issues
4. **Version Information**: Note any specific version requirements for libraries
5. **Performance Notes**: If applicable, mention expected runtime for large datasets

## 🤝 Support

If you have questions about these guidelines or need help preparing your submission:
- Contact: [New Issue](https://github.com/cal-adapt/cae-notebooks/issues/new/choose)

---

By following these guidelines, you help maintain a high-quality, accessible repository that benefits the entire community. Thank you for your contribution!
