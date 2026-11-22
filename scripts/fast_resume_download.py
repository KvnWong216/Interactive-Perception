#!/usr/bin/env python3
"""Bulletproof resumable downloader for ActiveArena assets & checkpoints.

Wraps `curl -L -C -` in an outer retry loop so proxy SSL EOFs (`curl exit 56` / `35`)
automatically resume from the exact byte offset until exact target size (and optional
SHA-256) is reached, and runs two independent streams in parallel (assets + weights).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import subprocess
import time
import traceback


PROXY = "http://127.0.0.1:7897"
SIM_REPO = Path("/home/ubuntu/ygwang/IP-VLA/baseline-study/external/activearena")
SIM_PY = "/home/ubuntu/ygwang/IP-VLA/baseline-study/.venvs/activearena-sim/bin/python"
WEIGHTS_ROOT = Path("/home/ubuntu/ygwang/IP-VLA/baseline-study/checkpoints/ActiveArena-VLA")

ACTIVEARENA_SUPPLEMENTARY = (
    (
        "005_button.zip",
        "a6340c04fbd107e6728c21bfff3be58f6c6862093a7acb197ae41f06322b0a63",
        SIM_REPO / "assets" / "objects" / "005_button",
    ),
    (
        "astribot_descriptions_texture_20260630_174911.zip",
        "49be39c500e763d6d0194d4449751967bff16df5a2c044f1f53f7e9b7adffcb7",
        SIM_REPO / "assets" / "embodiments" / "astribot_descriptions_texture",
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_remote_size(url: str) -> int:
    for _ in range(15):
        res = subprocess.run(
            ["curl", "-sI", "-L", "--proxy", PROXY, "--connect-timeout", "20", url],
            capture_output=True,
            text=True,
        )
        sizes = []
        for line in res.stdout.splitlines():
            if line.lower().startswith("content-length:"):
                try:
                    sizes.append(int(line.split(":", 1)[1].strip()))
                except ValueError:
                    pass
        if sizes and sizes[-1] > 10_000:
            return sizes[-1]
        time.sleep(2)
    return 0


def download_with_infinite_resume(
    url: str,
    dest: Path,
    label: str,
    expected_bytes: int = 0,
    expected_sha256: str = "",
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if expected_sha256 and dest.is_file() and sha256_file(dest) == expected_sha256:
        print(f"[{label}] VERIFIED CACHE ({dest.stat().st_size / (1024**2):.1f} MB)", flush=True)
        return

    expected = expected_bytes or get_remote_size(url)
    attempt = 0
    while True:
        cur = dest.stat().st_size if dest.is_file() else 0
        if expected > 0 and cur > expected:
            print(f"[{label}] Local file larger than expected ({cur} > {expected}); resetting.", flush=True)
            dest.unlink(missing_ok=True)
            cur = 0
        if expected > 0 and cur == expected:
            if expected_sha256:
                got_sha = sha256_file(dest)
                if got_sha != expected_sha256:
                    print(f"[{label}] SHA-256 mismatch ({got_sha} != {expected_sha256}); redownloading.", flush=True)
                    dest.unlink(missing_ok=True)
                    cur = 0
                else:
                    print(f"[{label}] COMPLETE & SHA-256 VERIFIED ({cur / (1024**2):.1f} MB)", flush=True)
                    return
            else:
                print(f"[{label}] COMPLETE ({cur / (1024**3):.2f} GB)", flush=True)
                return

        attempt += 1
        print(
            f"[{label}] attempt={attempt} resuming at {cur / (1024**2):.1f} MB / "
            f"{expected / (1024**2):.1f} MB...",
            flush=True,
        )
        proc = subprocess.run(
            [
                "curl",
                "-L",
                "-C",
                "-",
                "--fail",
                "--silent",
                "--show-error",
                "--connect-timeout",
                "30",
                "--speed-time",
                "45",
                "--speed-limit",
                "5000",
                "--proxy",
                PROXY,
                "-o",
                str(dest),
                url,
            ]
        )
        cur_after = dest.stat().st_size if dest.is_file() else 0
        if proc.returncode == 0 and (expected == 0 or cur_after == expected):
            if expected_sha256:
                got_sha = sha256_file(dest)
                if got_sha != expected_sha256:
                    print(f"[{label}] SHA-256 mismatch after finish; redownloading.", flush=True)
                    dest.unlink(missing_ok=True)
                    time.sleep(2)
                    continue
            print(f"[{label}] FINISHED ({cur_after / (1024**3):.2f} GB)", flush=True)
            return
        if expected == 0:
            expected = get_remote_size(url)
        time.sleep(3)


def ensure_supplementary_activearena_assets() -> None:
    assets_dir = SIM_REPO / "assets"
    cache_dir = assets_dir / ".downloads"
    cache_dir.mkdir(parents=True, exist_ok=True)

    supp_rev = "819632dfd545b569657e195cc47aae3af7fce5f1"
    supp_base = f"https://huggingface.co/datasets/leeibo/ActiveArena-Assets/resolve/{supp_rev}"

    for arc, sha256, target_dir in ACTIVEARENA_SUPPLEMENTARY:
        if target_dir.is_dir():
            print(f"[supp:{arc}] Already installed at {target_dir}", flush=True)
            continue
        # Remove any stale .part left by upstream script
        (cache_dir / arc).with_suffix(".zip.part").unlink(missing_ok=True)
        download_with_infinite_resume(
            f"{supp_base}/{arc}",
            cache_dir / arc,
            f"supp:{arc}",
            expected_sha256=sha256,
        )

    subprocess.run([SIM_PY, "./script/update_embodiment_config_path.py"], cwd=str(SIM_REPO), check=True)
    env = dict(
        PATH="/usr/local/bin:/usr/bin:/bin",
        HOME=str(Path.home()),
        HTTP_PROXY=PROXY,
        HTTPS_PROXY=PROXY,
    )
    subprocess.run([SIM_PY, "./script/download_activearena_assets.py"], cwd=str(SIM_REPO), env=env, check=True)
    print("[assets] Core objects + Astribot + 005_button READY for ID rollouts!", flush=True)


def download_assets_stream() -> None:
    while True:
        try:
            rev = "981c92aa34d8f94d4cff47e0d5bc2f7d4e0af042"
            base = f"https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/resolve/{rev}"
            assets_dir = SIM_REPO / "assets"
            assets_dir.mkdir(parents=True, exist_ok=True)

            # 1. Ensure objects.zip + embodiments.zip are extracted first
            for arc, folder in [("embodiments.zip", "embodiments"), ("objects.zip", "objects")]:
                target_dir = assets_dir / folder
                if not target_dir.is_dir():
                    zip_path = assets_dir / arc
                    download_with_infinite_resume(f"{base}/{arc}", zip_path, f"asset:{arc}")
                    print(f"[asset:{arc}] Extracting...", flush=True)
                    subprocess.run(["unzip", "-q", "-o", str(zip_path), "-d", str(assets_dir)], check=True)
                    assert target_dir.is_dir(), f"Missing {target_dir} after unzip"
                    zip_path.unlink(missing_ok=True)
                    print(f"[asset:{arc}] Extracted OK!", flush=True)

            # 2. Pre-cache & install 005_button + astribot immediately (unlocks ID setting!)
            ensure_supplementary_activearena_assets()

            # 3. Download & extract background_texture.zip (unlocks OOD setting!)
            bg_dir = assets_dir / "background_texture"
            if not bg_dir.is_dir():
                bg_zip = assets_dir / "background_texture.zip"
                download_with_infinite_resume(f"{base}/background_texture.zip", bg_zip, "asset:background_texture.zip")
                print("[asset:background_texture.zip] Extracting...", flush=True)
                subprocess.run(["unzip", "-q", "-o", str(bg_zip), "-d", str(assets_dir)], check=True)
                assert bg_dir.is_dir(), f"Missing {bg_dir} after unzip"
                bg_zip.unlink(missing_ok=True)
                print("[asset:background_texture.zip] Extracted OK!", flush=True)

            print("[assets] ALL SIMULATION ASSETS (ID + OOD) 100% READY!", flush=True)
            return
        except Exception:
            print("[assets] Unexpected error in assets stream; retrying in 5s:\n" + traceback.format_exc(), flush=True)
            time.sleep(5)


def download_weights_stream() -> None:
    while True:
        try:
            rev = "c265fe089dc23656e7ff82eb0fe8012adc0644aa"
            base = f"https://huggingface.co/leeibo/ActiveArena-VLA/resolve/{rev}"
            expected_pt_bytes = 4_967_033_387
            # Download oft_subtask_action_12_ws FIRST (already partially downloaded; used by Oracle & IP-VLA),
            # then oft_instruction_action_12_ws (native end-to-end ASA baseline).
            for model_name in ("oft_subtask_action_12_ws", "oft_instruction_action_12_ws"):
                rel = f"{model_name}/checkpoints/steps_100000_pytorch_model.pt"
                download_with_infinite_resume(
                    f"{base}/{rel}",
                    WEIGHTS_ROOT / rel,
                    f"ckpt:{model_name}",
                    expected_bytes=expected_pt_bytes,
                )
            print("[weights] ALL ACTIVEARENA-VLA CHECKPOINTS 100% READY!", flush=True)
            return
        except Exception:
            print("[weights] Unexpected error in weights stream; retrying in 5s:\n" + traceback.format_exc(), flush=True)
            time.sleep(5)


def main() -> None:
    with ThreadPoolExecutor(max_workers=2) as pool:
        f1 = pool.submit(download_assets_stream)
        f2 = pool.submit(download_weights_stream)
        f1.result()
        f2.result()
    print("ALL_DOWNLOADS_100_PERCENT_COMPLETE!", flush=True)


if __name__ == "__main__":
    main()
