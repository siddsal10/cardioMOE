"""Download the Mendeley "ECG Images dataset of Cardiac Patients" (gwbz3fsgp8 v2)
through Mendeley Data's public API (no login). Equivalent to the manual
"Download All" zip: one sub-folder per class, original filenames, sha256-verified.

    python scripts/fetch_mendeley.py data/raw/ecg_images_pk
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

API = "https://data.mendeley.com/public-api/datasets/gwbz3fsgp8"
VERSION = 2
HEADERS = {"User-Agent": "cardiomoe-downloader/0.1 (curl-compatible)", "Accept": "*/*"}


def _open(url: str, timeout: int):
    return urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=timeout)


def get_json(url: str):
    with _open(url, 60) as r:
        return json.load(r)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def fetch(item: dict, dest: Path, retries: int = 6) -> None:
    cd = item["content_details"]
    if dest.exists() and sha256(dest) == cd["sha256_hash"]:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            with _open(cd["download_url"], 120) as r:
                dest.write_bytes(r.read())
            if sha256(dest) != cd["sha256_hash"]:
                raise IOError(f"checksum mismatch: {dest}")
            return
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dest")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--debug", action="store_true", help="first 50 images per class")
    args = ap.parse_args()

    folders = get_json(f"{API}/folders/{VERSION}")
    jobs = []
    for folder in folders:
        files = get_json(f"{API}/files?folder_id={folder['id']}&version={VERSION}")
        files = sorted(files, key=lambda f: f["filename"])[: 50 if args.debug else None]
        print(f"{folder['name']}: {len(files)} files", flush=True)
        jobs += [(f, Path(args.dest) / folder["name"] / f["filename"]) for f in files]

    with ThreadPoolExecutor(args.workers) as ex:
        errors = [e for e in ex.map(lambda j: _safe(fetch, *j), jobs) if e]
    if errors:
        sys.exit(f"{len(errors)} failures, e.g. {errors[:3]}; re-run to resume")
    print(f"complete: {len(jobs)} images")


def _safe(fn, *a):
    try:
        fn(*a)
    except Exception as e:  # collect, report at the end
        return repr(e)


if __name__ == "__main__":
    main()
