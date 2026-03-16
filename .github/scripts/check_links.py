#!/usr/bin/env python3
"""Check for broken links in notebook markdown cells."""

import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

# Strip trailing punctuation that gets caught in URL regex
URL_RE = re.compile(r"https?://[^\s\)\]\"\'<>]+")

# These status codes don't indicate a broken link
OK_STATUSES = {200, 201, 203, 204, 206, 301, 302, 303, 307, 308, 403, 429}

HEADERS = {"User-Agent": "Mozilla/5.0"}

# URLs that are known-good but return non-OK status codes for programmatic requests
URL_EXCEPTIONS = {
    "https://docs.nlr.gov/docs/fy08osti/43156.pdf",
}


def extract_urls(notebook_path):
    with open(notebook_path) as f:
        nb = json.load(f)
    urls = set()
    for cell in nb["cells"]:
        if cell["cell_type"] == "markdown":
            source = "".join(cell["source"])
            for url in URL_RE.findall(source):
                url = url.rstrip(".,;:)")
                urls.add(url)
    return urls


def check_url(url):
    errors = []
    for attempt in range(2):
        if attempt > 0:
            time.sleep(2)
        try:
            resp = requests.head(url, timeout=60, allow_redirects=True, headers=HEADERS)
            if resp.status_code in (404, 405):
                # Some servers reject HEAD requests with 405 (Method Not Allowed);
                # fall back to GET which will also follow redirects.
                resp = requests.get(url, timeout=60, allow_redirects=True, headers=HEADERS)
            ok = resp.status_code in OK_STATUSES
            return url, resp.status_code, ok
        except requests.RequestException as e:
            errors.append(str(e))
    return url, "; ".join(errors), False


def main():
    broken = {}

    for nb_path in sorted(Path(".").rglob("*.ipynb")):
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
