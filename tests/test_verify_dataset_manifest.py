import hashlib
import json
import subprocess
import sys
from pathlib import Path


def test_verify_dataset_manifest(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    dataset = tmp_path / "rows.jsonl"
    dataset.write_text('{"x": 1}\n{"x": 2}\n')
    manifest = tmp_path / "rows.manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset": str(dataset),
                "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
                "samples": 2,
            }
        )
    )
    completed = subprocess.run(
        [sys.executable, str(root / "scripts/verify_dataset_manifest.py"), str(manifest)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "verified" in completed.stdout
