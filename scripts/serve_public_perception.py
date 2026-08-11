#!/usr/bin/env python3
"""RGB-only evidence adapter for an OpenAI-compatible local vision model."""

from __future__ import annotations

import argparse
import json
import os
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import yaml


def evidence_prompt(task: dict[str, Any]) -> str:
    locations = [str(item["label"]) for item in task.get("search_locations", [])]
    occluders = [str(item["label"]) for item in task.get("occluder_actions", [])]
    return f"""Inspect only the supplied robot RGB image. Do not infer hidden simulator state.
Task: {task['prompt']}
Return one JSON object and no prose with exactly these fields:
- target_visible: boolean; whether the named target is visibly identifiable.
- target_sufficient: boolean; whether visibility is sufficient to attempt the final grasp.
- locations: object mapping each label to closed, open_unsearched, or searched_empty.
- occluders: object mapping each label to blocking or cleared.
- confidence: number from 0 to 1.
Location labels: {locations}
Occluder labels: {occluders}
Use searched_empty only when the location is visibly open and its relevant interior is visibly exhausted.
Use low confidence when the image cannot support a state. Never claim knowledge of a closed container's contents."""


def parse_model_json(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("vision model response must be a JSON object")
    return value


class EvidenceHandler(BaseHTTPRequestHandler):
    tasks: dict[str, dict[str, Any]] = {}
    upstream: str = ""
    model: str = ""
    api_key: str | None = None
    timeout_s: float = 60.0

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/public-evidence":
            self.send_error(404)
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            request_value = json.loads(self.rfile.read(size).decode("utf-8"))
            task_id = str(request_value["task_id"])
            task = self.tasks[task_id]
            image_b64 = str(request_value["rgb_png_base64"])
            upstream_payload = {
                "model": self.model,
                "temperature": 0,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": evidence_prompt(task)},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                            },
                        ],
                    }
                ],
            }
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            upstream_request = urllib.request.Request(
                self.upstream.rstrip("/") + "/v1/chat/completions",
                data=json.dumps(upstream_payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(upstream_request, timeout=self.timeout_s) as response:
                upstream_value = json.loads(response.read().decode("utf-8"))
            content = upstream_value["choices"][0]["message"]["content"]
            result = parse_model_json(str(content))
            encoded = json.dumps(result).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
        except Exception as error:  # noqa: BLE001 - HTTP boundary reports errors
            encoded = json.dumps(
                {"error": f"{type(error).__name__}: {error}"}
            ).encode("utf-8")
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[public-perception] {format % args}", flush=True)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spec",
        type=Path,
        default=root / "benchmarks/interactive_manipulation_v0/benchmark.yaml",
    )
    parser.add_argument("--upstream", default="http://127.0.0.1:8001")
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8101)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()
    spec = yaml.safe_load(args.spec.read_text(encoding="utf-8"))
    EvidenceHandler.tasks = {str(task["id"]): task for task in spec["tasks"]}
    EvidenceHandler.upstream = args.upstream
    EvidenceHandler.model = args.model
    EvidenceHandler.api_key = os.environ.get("VLM_API_KEY")
    EvidenceHandler.timeout_s = args.timeout
    server = ThreadingHTTPServer((args.host, args.port), EvidenceHandler)
    print(
        f"public evidence server on http://{args.host}:{args.port}/v1/public-evidence "
        f"using model={args.model!r} upstream={args.upstream!r}",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
