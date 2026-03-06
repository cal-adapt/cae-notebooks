#!/usr/bin/env python3
"""Check for broken links in notebook markdown cells."""

import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

DIRS = ["data-access", "analysis", "climate-profiles", "collaborative"]

# Strip trailing punctuation that gets caught in URL regex
URL_RE = re.compile(r"https?://[^\s\)\]\"\'<>]+")

# These status codes don't indicate a broken link
OK_STATUSES = {200, 201, 203, 204, 206, 301, 302, 303, 307, 308, 403, 429}

HEADERS = {"User-Agent": "Mozilla/5.0"}


def extract_urls(notebook_path):
    with open(notebook_path) as f:
        nb = json.load(f)
    urls = set()
    for cell in nb["cells"]:
        if cell["cell_type"] == "markdown":
            source = "".join(cell["source"])
            for url in URL_RE.findall(source):
                urls.add(url.rstrip(".,;:)"))
    return urls


def check_url(url):
    for attempt in range(2):
        try:
            resp = requests.head(url, timeout=60, allow_redirects=True, headers=HEADERS)
            if resp.status_code in (404, 405):
                # Some servers (e.g. PDFs, government sites) reject HEAD requests
                # with 404 or 405; fall back to GET which will also follow redirects.
                # For example, when NREL renamed to NRL, old links to the NREL website
                # redirect to NRL and are still valid.
                resp = requests.get(url, timeout=60, allow_redirects=True, headers=HEADERS)
            ok = resp.status_code in OK_STATUSES
            return url, resp.status_code, ok
        except requests.RequestException as e:
            last_error = e
    return url, str(last_error), False


def main():
    broken = {}

    for nb_path in sorted(nb for d in DIRS for nb in Path(d).rglob("*.ipynb")):
        urls = extract_urls(nb_path)
        if not urls:
            continue

        print(f"Checking {nb_path} ({len(urls)} URLs)...")
        nb_broken = []

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(check_url, url): url for url in urls}
            for future in as_completed(futures):
                url, status, ok = future.result()
                if not ok:
                    print(f"  BROKEN [{status}]: {url}")
                    nb_broken.append((url, status))

        if nb_broken:
            broken[str(nb_path)] = nb_broken

    if broken:
        print(f"\n{len(broken)} notebook(s) have broken links:")
        for nb, links in broken.items():
            print(f"  {nb}:")
            for url, status in links:
                print(f"    [{status}] {url}")
        sys.exit(1)

    print("\nAll links OK!")


if __name__ == "__main__":
    main()
