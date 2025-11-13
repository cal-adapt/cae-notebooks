"""Utilities to run and manage notebook tests for CI.

This module is intentionally minimal and keeps file paths relative to the
repository root. It expects the test manifest to live at
`testing/test-manifest.yml` by default.
"""
from __future__ import annotations

import fnmatch
import traceback
from pathlib import Path
from typing import List, Optional, Tuple

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import yaml


class NotebookTester:
    """Small helper to run notebooks according to a YAML manifest.

    The manifest path defaults to `testing/test-manifest.yml`.
    """

    def __init__(self, manifest_path: str = "testing/test-manifest.yml"):
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        with open(self.manifest_path, "r", encoding="utf-8") as fh:
            self.manifest = yaml.safe_load(fh) or {}

        # repository root (used for resolving relative paths)
        self.root = Path('.').resolve()

    def _category_key(self, category: str) -> str:
        mapping = {
            'quick': 'quick_tests',
            'medium': 'medium_tests',
            'long': 'long_tests',
            'manual': 'manual_tests',
        }
        return mapping.get(category, f"{category}_tests")

    def get_notebooks_by_category(self, category: str) -> List[dict]:
        key = self._category_key(category)
        return self.manifest.get(key, []) or []

    def should_skip_notebook(self, notebook_path: str) -> Tuple[bool, str]:
        """Return (True, reason) if manifest marks the notebook as skipped."""
        skip_list = self.manifest.get('skip_tests', []) or []
        for entry in skip_list:
            if isinstance(entry, dict):
                pat = entry.get('path')
                reason = entry.get('reason', 'Marked as skip')
            else:
                pat = entry
                reason = 'Marked as skip'

            if not pat:
                continue

            # Pattern match against the path as stored in the manifest
            if fnmatch.fnmatch(notebook_path, pat):
                return True, reason

        return False, ''

    def validate_data_dependencies(self, notebook_config: dict) -> Tuple[bool, List[str]]:
        """Return (True, []) if required data files exist; otherwise returns missing list.

        Tests should skip notebooks with missing data rather than fail CI.
        """
        if not notebook_config.get('requires_data'):
            return True, []

        missing = []
        for d in notebook_config.get('data_dependencies', []):
            p = Path(d)
            if not p.exists():
                missing.append(str(d))

        return (len(missing) == 0, missing)

    def execute_notebook(
        self,
        notebook_path: str,
        timeout: int = 600,
        cell_indices: Optional[List[int]] = None,
    ) -> Tuple[bool, str]:
        """Execute a notebook and return (success, message).

        The method will try to find the notebook path relative to repo root
        if the provided path does not exist literally.
        """
        nb_path = Path(notebook_path)
        if not nb_path.exists():
            # try to locate by filename (useful when manifest contains relative paths)
            candidates = list(self.root.rglob(nb_path.name))
            if candidates:
                nb_path = candidates[0]
            else:
                return False, f"Notebook file not found: {notebook_path}"

        try:
            nb = nbformat.read(str(nb_path), as_version=4)

            if cell_indices:
                cells = nb.cells
                nb.cells = [cells[i] for i in cell_indices if i < len(cells)]

            ep = ExecutePreprocessor(timeout=timeout, kernel_name='python3')
            ep.preprocess(nb, {'metadata': {'path': str(nb_path.parent)}})
            return True, 'Success'

        except Exception:
            tb = traceback.format_exc()
            return False, tb


__all__ = ["NotebookTester"]
