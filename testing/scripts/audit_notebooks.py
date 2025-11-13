#!/usr/bin/env python3
"""
Audit Jupyter notebooks in a repository and produce a small JSON report.

Usage:
  python scripts/audit_notebooks.py --output notebook_audit.json

This script counts code/markdown cells, extracts simple import names,
and looks for obvious data file references (csv, nc, geojson, etc.).
"""
import os
import subprocess
import re
import json
import argparse
from pathlib import Path
from collections import Counter
from datetime import datetime

import nbformat


IMPORT_RE = re.compile(r"^\s*(?:from|import)\s+([A-Za-z0-9_\.]+)")
FILEPATH_RE = re.compile(
    r"(?:r?['\"])([^'\"]+\.(?:csv|nc|nc4|json|geojson|txt|zip|parquet|feather|pkl|npz|npy))(?:['\"])",
    re.IGNORECASE,
)


def extract_imports(code: str) -> list[str]:
    """Extract import names from Python code.

    Parameters
    ----------
    code : str
        Python source code as a string

    Returns
    -------
    list[str]
        List of top-level module names found in import statements
    """
    imports = []
    for line in code.splitlines():
        m = IMPORT_RE.match(line)
        if m:
            imports.append(m.group(1).split(".")[0])
    return imports


def find_data_refs(code: str) -> list[str]:
    """Find data file references and data access patterns in Python code.

    Parameters
    ----------
    code : str
        Python source code as a string

    Returns
    -------
    list[str]
        List of data file paths found in the code and common data access
        patterns (e.g., 'read_csv', 'open', 'xarray', 'parquet')
    """
    refs = FILEPATH_RE.findall(code)
    # also flag presence of common data-reading calls
    hints = []
    if "read_csv" in code or "pd.read_csv" in code:
        hints.append("read_csv")
    if "open(" in code:
        hints.append("open")
    if "xr.open_dataset" in code or "xarray" in code:
        hints.append("xarray")
    if "read_parquet" in code or "to_parquet" in code:
        hints.append("parquet")
    return list(set(refs + hints))


def audit_notebook(nb_path: Path) -> dict[str, object]:
    """
    Audit a Jupyter notebook and extract basic metadata.

    Parameters
    ----------
    nb_path : pathlib.Path
        Path to the notebook (.ipynb) file to audit.

    Returns
    -------
    dict[str, object]
        Dictionary with the following keys:
        - path (str): Path to the notebook
        - code_cells (int): Number of code cells in the notebook
        - markdown_cells (int): Number of markdown cells in the notebook
        - imports (list[str]): Sorted list of unique import names found in code cells
        - data_references (list[str]): Sorted list of unique data file paths and hints found
        - status (str): 'ok' when audit succeeded, 'error' on failure
        - error (str or None): Error message when status is 'error'
        - last_modified (str): ISO-formatted last modification time (present when status is 'ok')
    """
    result: dict[str, object] = {"path": str(nb_path)}
    try:
        nb = nbformat.read(nb_path, as_version=4)
        code_cells = [c for c in nb.cells if c.cell_type == "code"]
        md_cells = [c for c in nb.cells if c.cell_type == "markdown"]

        imports: list[str] = []
        data_refs: list[str] = []
        for cell in code_cells:
            src: str = cell.get("source", "")
            imports += extract_imports(src)
            data_refs += find_data_refs(src)

        result.update(
            {
                "code_cells": len(code_cells),
                "markdown_cells": len(md_cells),
                "imports": sorted(set(imports)),
                "data_references": sorted(set(data_refs)),
                "status": "ok",
                "error": None,
                "last_modified": datetime.fromtimestamp(
                    nb_path.stat().st_mtime
                ).isoformat(),
            }
        )
    except Exception as e:
        result.update({"status": "error", "error": str(e)})
    return result


def main(output: str = "notebook_audit.json", root: str = ".") -> int:
    """
    Audit Jupyter notebooks under a repository root and write a JSON summary.

    Parameters
    ----------
    output : str, optional
        Path to the JSON output file (default is 'notebook_audit.json').
    root : str, optional
        Repository root to search for notebooks (default is current directory).

    Returns
    -------
    int
        Exit code (0 on success).

    Raises
    ------
    OSError
        If writing the output file fails.
    """
    root_path: Path = Path(root)
    notebooks: list[Path] = [
        p for p in root_path.rglob("*.ipynb") if ".ipynb_checkpoints" not in str(p)
    ]
    results: list[dict[str, object]] = []
    import_counter: Counter[str] = Counter()
    for nb in sorted(notebooks):
        print(f"Auditing {nb}")
        r: dict[str, object] = audit_notebook(nb)
        results.append(r)
        for imp in r.get("imports", []):
            import_counter[imp] += 1

    out: dict[str, object] = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "repo_root": str(Path(root).resolve()),
        "total_notebooks": len(results),
        "results": results,
        "top_imports": import_counter.most_common(50),
    }

    with open(output, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote audit to {output}")
    print(f"Total notebooks: {len(results)}")
    errors: list[dict[str, object]] = [r for r in results if r.get("status") != "ok"]
    print(f"Notebooks with errors: {len(errors)}")
    return 0


if __name__ == "__main__":
    # determine repository root (prefer git); fall back to finding a parent with common project files
    _script_dir = Path(__file__).resolve().parent
    try:
        _git_top = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=_script_dir,
            stderr=subprocess.DEVNULL,
        )
        repo_root = Path(_git_top.decode().strip())
    except Exception:
        repo_root = _script_dir
        for _parent in [_script_dir] + list(_script_dir.parents):
            if (
                (_parent / ".git").exists()
                or (_parent / "pyproject.toml").exists()
                or (_parent / "setup.py").exists()
            ):
                repo_root = _parent
                break

    # use the repo root as the working directory so relative paths default to the repository base
    os.chdir(repo_root)
    parser = argparse.ArgumentParser(
        description="Audit Jupyter notebooks and extract metadata"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="testing/outputs/notebook_audit.json",
        help="Path to JSON output",
    )
    parser.add_argument(
        "--root", "-r", default=".", help="Repository root to search for notebooks"
    )
    args = parser.parse_args()
    raise SystemExit(main(output=args.output, root=args.root))
