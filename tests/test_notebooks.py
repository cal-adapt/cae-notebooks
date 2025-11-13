import os
import sys
import pytest

# Ensure the tests/ directory is importable for helper modules
sys.path.insert(0, os.path.abspath('tests'))

from notebook_test_utils import NotebookTester


tester = NotebookTester(manifest_path='testing/test-manifest.yml')


def pytest_addoption(parser):
    parser.addoption(
        "--category",
        action="store",
        default="quick",
        help="Test category to run: quick, medium, long",
    )


def pytest_generate_tests(metafunc):
    if "notebook_config" in metafunc.fixturenames:
        category = metafunc.config.getoption("--category")
        notebooks = tester.get_notebooks_by_category(category)
        ids = [nb['path'] if isinstance(nb, dict) else str(nb) for nb in notebooks]
        metafunc.parametrize("notebook_config", notebooks, ids=ids)


@pytest.mark.notebook
def test_notebook_execution(notebook_config):
    """Execute a notebook (or its smoke cells) according to the manifest.

    Tests will be skipped when notebooks are explicitly marked as skipped in
    the manifest or when required data files are missing.
    """
    notebook_path = notebook_config['path']

    should_skip, reason = tester.should_skip_notebook(notebook_path)
    if should_skip:
        pytest.skip(reason)

    has_data, missing = tester.validate_data_dependencies(notebook_config)
    if not has_data:
        pytest.skip(f"Missing data dependencies: {missing}")

    smoke_cells = notebook_config.get('smoke_test_cells')
    timeout = notebook_config.get('timeout', 600)

    success, message = tester.execute_notebook(
        notebook_path, timeout=timeout, cell_indices=smoke_cells
    )

    assert success, message


@pytest.mark.smoke
def test_notebook_smoke(notebook_config):
    """Explicit smoke test that executes only the cells listed under
    `smoke_test_cells` in the manifest. If no smoke cells are defined the
    test is skipped.
    """
    if 'smoke_test_cells' not in notebook_config:
        pytest.skip('No smoke test cells defined')

    notebook_path = notebook_config['path']
    smoke_cells = notebook_config['smoke_test_cells']

    success, message = tester.execute_notebook(
        notebook_path, timeout=300, cell_indices=smoke_cells
    )

    assert success, message
