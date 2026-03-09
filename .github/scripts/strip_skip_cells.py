"""Remove cells tagged 'skip' from a notebook before passing to papermill."""
import sys
import nbformat

input_path, output_path = sys.argv[1], sys.argv[2]
nb = nbformat.read(input_path, as_version=4)
nb.cells = [c for c in nb.cells if "skip" not in c.get("metadata", {}).get("tags", [])]
nbformat.write(nb, output_path)
