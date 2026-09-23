"""Fetch only allowlisted stage-one assets, with a persistent 110 GB download quota."""

import argparse
import fcntl
import json
import os
import shutil
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "data/preparation"
LIMIT = 110_000_000_000
SOURCES = (
    (
        "datasets",
        "yifengzhu-hf/LIBERO-datasets",
        "main",
        "data/libero_raw",
    ),
    (
        "models",
        "allenai/MolmoAct2-LIBERO",
        "main",
        "checkpoints/base/MolmoAct2-LIBERO",
    ),
)
SUITES = {"libero_spatial", "libero_object", "libero_goal", "libero_10"}


def inside(path):
    path = Path(path)
    if not path.resolve().is_relative_to(ROOT):
        raise ValueError("asset path escapes repository")
    return path


def save(path, value):
    path = inside(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = inside(path.with_suffix(path.suffix + ".tmp"))
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def inventory():
    cached = ROOT / "experiments/stage1_assets.json"
    if cached.exists():
        value = json.loads(cached.read_text())
        rows = value["files"]
        if (
            value["limit_bytes"] != LIMIT
            or len(rows) != 59
            or sum(r["size"] for r in rows) != 55_566_258_025
        ):
            raise ValueError("cached inventory differs from reviewed assets")
        for row in rows:
            source = next((s for s in SOURCES if s[1] == row["source"]), None)
            if (
                source is None
                or row["revision"] != source[2]
                or not row["path"].startswith(source[3] + "/")
            ):
                raise ValueError("cached asset source mismatch")
            inside(ROOT / row["path"])
            prefix = "datasets/" if source[0] == "datasets" else ""
            filename = row["path"][len(source[3]) + 1 :]
            if (
                row["url"]
                != f"https://huggingface.co/{prefix}{source[1]}/resolve/{source[2]}/{filename}?download=true"
            ):
                raise ValueError("cached asset URL mismatch")
        return rows
    rows = []
    for kind, repo, revision, destination in SOURCES:
        url = f"https://huggingface.co/api/{kind}/{repo}/tree/{revision}?recursive=true&limit=1000"
        with urllib.request.urlopen(url, timeout=60) as response:
            if response.headers.get("Link"):
                raise RuntimeError(
                    "unexpected pagination; review inventory before downloading"
                )
            entries = json.load(response)
        for entry in entries:
            path = PurePosixPath(entry["path"])
            if entry["type"] != "file":
                continue
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("unsafe upstream path")
            if kind == "datasets":
                if (
                    len(path.parts) != 2
                    or path.parts[0] not in SUITES
                    or path.suffix != ".hdf5"
                ):
                    continue
            elif len(path.parts) != 1 or path.suffix not in {
                ".json",
                ".py",
                ".safetensors",
                ".model",
                ".txt",
                ".jinja",
            }:
                continue
            prefix = "datasets/" if kind == "datasets" else ""
            rows.append(
                {
                    "path": f"{destination}/{path}",
                    "size": entry["size"],
                    "url": f"https://huggingface.co/{prefix}{repo}/resolve/{revision}/{path}?download=true",
                    "source": repo,
                    "revision": revision,
                }
            )
    data = [r for r in rows if r["source"] == SOURCES[0][1]]
    if len(data) != 40 or sum(r["size"] for r in data) != 33_784_856_577:
        raise RuntimeError("LIBERO allowlist differs from reviewed inventory")
    if sum(r["size"] for r in rows) > LIMIT:
        raise RuntimeError("assets exceed stage-one download quota")
    rows.sort(
        key=lambda r: (r["size"] > 10_000_000, "drawer" not in r["path"], r["size"])
    )
    save(ROOT / "experiments/stage1_assets.json", {"limit_bytes": LIMIT, "files": rows})
    return rows


class Ledger:
    def __init__(self):
        self.path = inside(STATE / "asset_ledger.json")
        self.lock = threading.Lock()
        self.value = (
            json.loads(self.path.read_text())
            if self.path.exists()
            else {
                "reserved_download_bytes": 0,
                "verified": {},
                "note": "Conservative transfer bound; every resumed attempt is reserved before reading. Crashes do not refund quota.",
            }
        )

    def reserve(self, size):
        with self.lock:
            if self.value["reserved_download_bytes"] + size > LIMIT:
                raise RuntimeError("110 GB cumulative download quota exhausted")
            self.value["reserved_download_bytes"] += size
            save(self.path, self.value)

    def verified(self, row):
        with self.lock:
            self.value["verified"][row["path"]] = {
                "size": row["size"],
            }
            save(self.path, self.value)


def fetch(row, ledger):
    destination = inside(ROOT / row["path"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = inside(destination.with_suffix(destination.suffix + ".partial"))
    if destination.exists():
        if destination.stat().st_size != row["size"]:
            raise RuntimeError(f"existing file failed validation: {row['path']}")
        ledger.verified(row)
        return
    failures = 0
    while True:
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > row["size"]:
            raise RuntimeError("oversized partial file")
        try:
            if offset < row["size"]:
                end = min(offset + 64 * 1024 * 1024, row["size"]) - 1
                request = urllib.request.Request(
                    row["url"], headers={"Range": f"bytes={offset}-{end}"}
                )
                with urllib.request.urlopen(request, timeout=120) as response:
                    if response.status == 206:
                        if (
                            response.headers.get("Content-Range")
                            != f"bytes {offset}-{end}/{row['size']}"
                        ):
                            raise RuntimeError("server returned a different byte range")
                    elif offset or end + 1 != row["size"]:
                        raise RuntimeError("server ignored bounded range")
                    length = response.headers.get("Content-Length")
                    if length and int(length) != end + 1 - offset:
                        raise RuntimeError("upstream length differs from inventory")
                    ledger.reserve(end + 1 - offset)
                    with partial.open("ab") as stream:
                        while offset <= end:
                            block = response.read(min(1024 * 1024, end + 1 - offset))
                            if not block:
                                raise OSError("truncated response")
                            stream.write(block)
                            offset += len(block)
                        stream.flush()
                        os.fsync(stream.fileno())
                failures = 0
                if offset < row["size"]:
                    continue
            partial.replace(destination)
            ledger.verified(row)
            print(f"VERIFIED {row['path']} {row['size']} bytes", flush=True)
            return
        except OSError as error:
            failures += 1
            print(f"RETRY {failures} {row['path']}: {error}", flush=True)
            if failures == 3:
                raise
            time.sleep(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    inside(STATE).mkdir(parents=True, exist_ok=True)
    with inside(STATE / "assets.lock").open("w") as lockfile:
        fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows = inventory()
        total = sum(r["size"] for r in rows)
        print(f"PINNED {len(rows)} files {total} bytes", flush=True)
        if not args.download:
            return
        footprint = 0
        for directory, dirs, files in os.walk(ROOT, followlinks=False):
            dirs[:] = [d for d in dirs if not (Path(directory) / d).is_symlink()]
            footprint += sum(
                (Path(directory) / f).stat().st_size
                for f in files
                if not (Path(directory) / f).is_symlink()
            )
        if (
            footprint + total > 800_000_000_000
            or shutil.disk_usage(ROOT).free < total + 10_000_000_000
        ):
            raise RuntimeError("asset preparation exceeds project/disk budget")
        ledger = Ledger()
        errors = []
        # Keep two slots for the model so large weights are not queued behind
        # every demonstration file; the first audited subset can train sooner.
        with (
            ThreadPoolExecutor(max_workers=2) as data_pool,
            ThreadPoolExecutor(max_workers=2) as model_pool,
        ):
            jobs = {
                (data_pool if row["source"] == SOURCES[0][1] else model_pool).submit(
                    fetch, row, ledger
                ): row
                for row in rows
            }
            for job in as_completed(jobs):
                try:
                    job.result()
                except (OSError, RuntimeError, ValueError) as error:
                    errors.append({"path": jobs[job]["path"], "error": str(error)})
                    print(f"ERROR {errors[-1]}", flush=True)
        save(
            STATE / "asset_status.json",
            {
                "expected_files": len(rows),
                "verified_files": len(ledger.value["verified"]),
                "errors": errors,
            },
        )
        if errors:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
