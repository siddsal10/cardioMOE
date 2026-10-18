"""Parallel, resumable downloader for open-access PhysioNet projects.

Uses the public AWS mirror (s3://physionet-open, anonymous HTTPS) because it is
much faster than physionet.org for tens of thousands of small WFDB files.
Files that already exist locally with the right size are skipped, so re-running
resumes an interrupted download.

    python scripts/fetch_physionet.py ptb-xl/1.0.3 data/raw/ptbxl --exclude records500/
"""
from __future__ import annotations

import argparse
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BUCKET = "https://physionet-open.s3.amazonaws.com"
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


def list_objects(prefix: str, delimiter: str | None = None) -> tuple[list[tuple[str, int]], list[str]]:
    """Return ([(key, size)], [sub-prefixes]) under prefix (ListObjectsV2, paginated)."""
    out, subdirs, token = [], [], None
    while True:
        q = {"list-type": "2", "prefix": prefix}
        if delimiter:
            q["delimiter"] = delimiter
        if token:
            q["continuation-token"] = token
        with urllib.request.urlopen(f"{BUCKET}/?{urllib.parse.urlencode(q)}", timeout=60) as r:
            root = ET.fromstring(r.read())
        for c in root.findall("s3:Contents", NS):
            out.append((c.find("s3:Key", NS).text, int(c.find("s3:Size", NS).text)))
        subdirs += [c.find("s3:Prefix", NS).text for c in root.findall("s3:CommonPrefixes", NS)]
        if root.find("s3:IsTruncated", NS).text != "true":
            return out, subdirs
        token = root.find("s3:NextContinuationToken", NS).text


def walk(prefix: str, skip) -> list[tuple[str, int]]:
    """Top-level files plus everything under each top-level folder not rejected by
    skip(rel_folder). Excluded folders (e.g. records500/) are never listed."""
    files, subdirs = list_objects(prefix, delimiter="/")
    for sd in subdirs:
        if not skip(sd[len(prefix):]):
            files += list_objects(sd)[0]
    return files


def fetch(key: str, dest: Path, size: int, retries: int = 8) -> int:
    if dest.exists() and dest.stat().st_size == size:
        return 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    url = f"{BUCKET}/{urllib.parse.quote(key)}"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            if tmp.stat().st_size != size:
                raise IOError(f"size mismatch for {key}")
            tmp.replace(dest)
            return size
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("project", help="e.g. ptb-xl/1.0.3")
    ap.add_argument("dest")
    ap.add_argument("--exclude", nargs="*", default=[], help="relative path prefixes to skip")
    ap.add_argument("--include", nargs="*", default=None, help="only these relative path prefixes")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--debug", action="store_true", help="metadata + first ~200 records only")
    args = ap.parse_args()

    prefix = args.project.rstrip("/") + "/"
    rel = lambda k: k[len(prefix):]
    excluded = lambda r: any(r.startswith(e) for e in args.exclude)
    objs = [(k, s) for k, s in walk(prefix, excluded) if not k.endswith("/")]
    objs = [(k, s) for k, s in objs if not any(rel(k).startswith(e) for e in args.exclude)]
    if args.include is not None:
        objs = [(k, s) for k, s in objs if any(rel(k).startswith(i) for i in args.include)]
    if args.debug:
        top = [o for o in objs if "/" not in rel(o[0])]
        recs = [o for o in objs if "/" in rel(o[0])][:400]  # ~200 records (.hea + data)
        objs = top + recs

    dest = Path(args.dest)
    total = sum(s for _, s in objs)
    print(f"{args.project}: {len(objs)} files, {total / 1e9:.2f} GB -> {dest}", flush=True)
    done = got = 0
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(fetch, k, dest / rel(k), s) for k, s in objs]
        for f in as_completed(futs):
            got += f.result()
            done += 1
            if done % 2000 == 0 or done == len(futs):
                rate = got / max(time.time() - t0, 1e-6) / 1e6
                print(f"  {done}/{len(futs)} files, {got / 1e9:.2f} GB new, {rate:.1f} MB/s", flush=True)
    bad = [k for k, s in objs if not (dest / rel(k)).exists() or (dest / rel(k)).stat().st_size != s]
    if bad:
        sys.exit(f"{len(bad)} files missing or incomplete, e.g. {bad[:3]}; re-run to resume")
    print("complete")


if __name__ == "__main__":
    main()
