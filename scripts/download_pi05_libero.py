#!/usr/bin/env python3
"""Mirror the public ``pi05_libero`` checkpoint from openpi's asset bucket.

openpi distributes checkpoints on Google Cloud Storage. The bucket is public, so
plain HTTPS works and neither ``gsutil`` nor credentials are needed. The
checkpoint is an Orbax/OCDBT directory: the relative layout is load-bearing and
must be preserved exactly, which is why this mirrors object paths rather than
flattening them.

Roughly 12.4 GB across ~16 objects. Downloads are resumable and idempotent --
re-running skips objects already present at the expected size, so an interrupted
transfer is repaired by running the same command again.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

BUCKET = "openpi-assets"
PREFIX = "checkpoints/pi05_libero/"
LIST_URL = "https://storage.googleapis.com/storage/v1/b/{bucket}/o"
OBJECT_URL = "https://storage.googleapis.com/{bucket}/{name}"


def parse_args() -> argparse.Namespace:
    default_dest = Path(__file__).resolve().parents[2] / "checkpoints"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", default=str(default_dest))
    parser.add_argument("--bucket", default=BUCKET)
    parser.add_argument("--prefix", default=PREFIX)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="check sizes of already-downloaded objects and exit",
    )
    return parser.parse_args()


def list_objects(bucket: str, prefix: str) -> list[tuple[str, int]]:
    query = urllib.parse.urlencode(
        {"prefix": prefix, "fields": "items(name,size),nextPageToken", "maxResults": 1000}
    )
    url = f"{LIST_URL.format(bucket=bucket)}?{query}"
    objects: list[tuple[str, int]] = []
    while True:
        with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310
            payload = json.load(response)
        for item in payload.get("items", []):
            objects.append((item["name"], int(item["size"])))
        token = payload.get("nextPageToken")
        if not token:
            return objects
        url = f"{LIST_URL.format(bucket=bucket)}?{query}&pageToken={token}"


def download(bucket: str, name: str, destination: Path, *, retries: int) -> None:
    url = OBJECT_URL.format(bucket=bucket, name=urllib.parse.quote(name))
    partial = destination.with_suffix(destination.suffix + ".part")
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310
                with partial.open("wb") as handle:
                    shutil.copyfileobj(response, handle, length=1 << 20)
            partial.replace(destination)
            return
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            last_error = error
            print(f"    attempt {attempt}/{retries} failed: {error}", flush=True)
            partial.unlink(missing_ok=True)
    raise SystemExit(f"failed to download {name}: {last_error}")


def main() -> None:
    args = parse_args()
    dest = Path(args.dest).expanduser().resolve()
    objects = list_objects(args.bucket, args.prefix)
    if not objects:
        raise SystemExit(f"no objects under gs://{args.bucket}/{args.prefix}")

    total_bytes = sum(size for _, size in objects)
    print(f"{len(objects)} objects, {total_bytes / 2**30:.1f} GiB -> {dest}")

    missing = 0
    for name, size in objects:
        target = dest / name
        if target.exists() and target.stat().st_size == size:
            print(f"ok   {name}")
            continue
        if args.verify_only:
            print(f"MISSING/SHORT {name}")
            missing += 1
            continue
        print(f"get  {name}  ({size / 2**20:.1f} MiB)", flush=True)
        download(args.bucket, name, target, retries=args.retries)
        actual = target.stat().st_size
        if actual != size:
            raise SystemExit(f"size mismatch for {name}: {actual} != {size}")

    if args.verify_only and missing:
        print(f"\n{missing} object(s) missing or truncated; re-run without --verify-only")
        sys.exit(1)

    root = dest / args.prefix.rstrip("/")
    print(f"\ncheckpoint ready: {root}")
    print(f"serve it with:  scripts/serve_pi05.sh <openpi_dir> {root}")


if __name__ == "__main__":
    main()
