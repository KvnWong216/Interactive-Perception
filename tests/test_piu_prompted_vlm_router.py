from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _server(metadata: dict, selected: str) -> tuple[HTTPServer, threading.Thread]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            assert self.path == "/metadata"
            payload = json.dumps(metadata).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self) -> None:
            assert self.path == "/route"
            size = int(self.headers["Content-Length"])
            request = json.loads(self.rfile.read(size))
            response = {
                "schema_version": "piu.prompted-vlm-router-response.v1",
                "request_sha256": _canonical_sha256(request),
                "selected_candidate_id": selected,
            }
            payload = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_: object) -> None:
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    image = tmp_path / "observation.png"
    pixels = np.zeros((4, 4, 3), dtype=np.uint8)
    pixels[:, :, 1] = 64
    Image.fromarray(pixels).save(image)
    artifact = {
        "path": str(image),
        "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
        "pixel_sha256": hashlib.sha256(
            np.ascontiguousarray(pixels).tobytes()
        ).hexdigest(),
    }
    observation = {
        "images": {"agentview": artifact},
        "public_robot_state": [0.0],
    }
    public = tmp_path / "public.jsonl"
    public.write_text(
        json.dumps(
            {
                "schema_version": "piu.public-transition.v1",
                "sample_id": "sample",
                "initial_state_group": "group",
                "split": "development",
                "prompt": "Place the butter in the basket",
                "observations": {
                    "pre_interaction": observation,
                    "post_interaction": observation,
                },
                "public_action_history": {
                    "initial_observation": True,
                    "last_executed_candidate": None,
                },
                "candidate_actions": [
                    {
                        "candidate_id": "open_drawer",
                        "primitive": "OPEN",
                        "target": "middle drawer",
                    },
                    {
                        "candidate_id": "stop",
                        "primitive": "STOP",
                        "target": "task",
                    },
                ],
                "online_oracle_inputs": [],
            }
        )
        + "\n"
    )
    metadata = {
        "schema_version": "piu.prompted-vlm-router-server.v1",
        "model_id": "fixture-frozen-vlm",
        "revision": "fixture-revision",
        "capabilities": ["public_candidate_routing_v1"],
    }
    identity = tmp_path / "identity.json"
    identity.write_text(
        json.dumps(
            {
                "schema_version": "piu.prompted-vlm-router-identity.v1",
                "server_metadata": metadata,
            }
        )
    )
    return public, identity


def _run_router(
    *, tmp_path: Path, public: Path, identity: Path, selected: str, stem: str
) -> dict:
    metadata = json.loads(identity.read_text())["server_metadata"]
    server, thread = _server(metadata, selected)
    output = tmp_path / f"{stem}.json"
    try:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/pipeline/run_piu_prompted_vlm_router.py"),
                "--public-transition",
                str(public),
                "--sample-id",
                "sample",
                "--router-identity",
                str(identity),
                "--host",
                "127.0.0.1",
                "--port",
                str(server.server_port),
                "--expected-split",
                "development",
                "--output",
                str(output),
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
    return json.loads(output.read_text())


def test_prompted_vlm_router_uses_only_bound_public_candidate_response(
    tmp_path: Path,
) -> None:
    public, identity = _inputs(tmp_path)
    report = _run_router(
        tmp_path=tmp_path,
        public=public,
        identity=identity,
        selected="open_drawer",
        stem="selected",
    )
    assert report["method_id"] == "B1"
    assert report["evaluator_labels_loaded"] is False
    assert report["online_oracle_inputs"] == []
    assert report["decisions"][0]["decision_kind"] == "INTERACT"
    assert report["manual_confidence_threshold"] is None
    dispatch = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/pipeline/execute_piu_controller_decision.py"),
            "--controller-report",
            str(tmp_path / "selected.json"),
            "--sample-id",
            "sample",
            "--scenario-config",
            str(ROOT / "configs/scenarios/original_drawer.yaml"),
            "--seed",
            "7",
            "--host",
            "pi05.internal",
            "--run-dir",
            str(tmp_path / "dispatch"),
            "--dry-run",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(dispatch.stdout)
    assert plan["candidate_id"] == "open_drawer"
    assert plan["physical_dispatch_allowed"] is False


def test_prompted_vlm_router_hallucinated_candidate_becomes_abstain(
    tmp_path: Path,
) -> None:
    public, identity = _inputs(tmp_path)
    report = _run_router(
        tmp_path=tmp_path,
        public=public,
        identity=identity,
        selected="oracle_hidden_candidate",
        stem="abstain",
    )
    assert report["decisions"][0]["decision_kind"] == "ABSTAIN"
    assert report["decisions"][0]["selected_candidate_id"] is None


def test_prompted_vlm_closed_loop_plan_keeps_router_and_pi05_external(
    tmp_path: Path,
) -> None:
    _, identity = _inputs(tmp_path)
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(
        json.dumps(
            {
                "schema_version": "piu.public-candidate-set.v1",
                "sample_id": "sample",
                "initial_state_group": "group",
                "split": "development",
                "candidates": [
                    {"candidate_id": "open_drawer", "primitive": "OPEN"}
                ],
            }
        )
        + "\n"
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/pipeline/run_piu_prompted_vlm_closed_loop.py"),
            "--scenario-config",
            str(ROOT / "configs/scenarios/original_drawer.yaml"),
            "--candidate-set",
            str(candidates),
            "--initial-sample-id",
            "sample",
            "--seed",
            "7",
            "--router-identity",
            str(identity),
            "--router-host",
            "router.internal",
            "--router-port",
            "9000",
            "--pi05-host",
            "pi05.internal",
            "--output-dir",
            str(tmp_path / "closed_loop"),
            "--dry-run",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(completed.stdout)
    assert plan["method_id"] == "B1"
    assert plan["local_models_loaded"] == []
    assert plan["router_endpoint"] == "router.internal:9000"
    assert plan["pi05_endpoint"] == "pi05.internal:8002"
    assert [Path(row[1]).name for row in plan["first_decision_commands"]] == [
        "capture_piu_initial_observation.py",
        "run_piu_prompted_vlm_router.py",
    ]
    assert plan["execution_ready"] is False
