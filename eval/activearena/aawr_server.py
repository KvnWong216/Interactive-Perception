#!/usr/bin/env python3
"""Localhost HTTP inference server for the unprivileged AAWR policy (`POST /act`, `GET /health`).

Strictly evaluates ONLY the unprivileged actor `pi_theta(a_{t:t+H} | rgb_hist, qpos_18, instruction)`
with ZERO privileged simulator state (`_Q_encoder` / `_V_encoder` are never invoked at deployment).
"""
from __future__ import annotations

import argparse
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import threading
import time
from typing import Any

import numpy as np
from PIL import Image
import torch

from eval.activearena.aawr_network import AAWRTrainer, hash_instruction_to_vec


class AAWRPolicyService:
    def __init__(self, ckpt_dir: Path, device: str = "cuda:0") -> None:
        self.ckpt_dir = ckpt_dir.resolve()
        self.device = device if torch.cuda.is_available() else "cpu"
        self.trainer, self.stats = AAWRTrainer.load_for_inference(self.ckpt_dir, device=self.device)
        self.cfg = self.trainer.cfg
        self._lock = threading.Lock()

        self.qpos_mean = np.asarray(self.stats["qpos_mean"], dtype=np.float32)
        self.qpos_std = np.maximum(np.asarray(self.stats["qpos_std"], dtype=np.float32), 1e-4)
        self.act_min = np.asarray(self.stats["act_min"], dtype=np.float32)
        self.act_max = np.asarray(self.stats["act_max"], dtype=np.float32)

    def predict(self, instruction: str, images_b64: list[str], qpos_18: list[float]) -> dict[str, Any]:
        t0 = time.perf_counter()
        sz = self.cfg.image_size
        frames = []
        for b64 in images_b64:
            raw = base64.b64decode(b64)
            with Image.open(io.BytesIO(raw)) as im:
                rgb = im.convert("RGB").resize((sz, sz), Image.Resampling.BILINEAR)
                arr = np.asarray(rgb, dtype=np.float32) / 255.0
                frames.append(np.transpose(arr, (2, 0, 1)))

        while len(frames) < self.cfg.history_frames:
            frames.insert(0, frames[0])
        frames = frames[-self.cfg.history_frames :]

        rgb_t = torch.from_numpy(np.stack(frames, axis=0))[None, ...]  # (1, T, 3, H, W)
        q_norm = (np.asarray(qpos_18, dtype=np.float32) - self.qpos_mean) / self.qpos_std
        q_t = torch.from_numpy(q_norm)[None, ...]
        txt_t = torch.from_numpy(hash_instruction_to_vec(instruction, self.cfg.text_hash_dim))[None, ...]

        with self._lock:
            norm_chunk = self.trainer.predict_normalized_chunk(rgb_t, q_t, txt_t)[0]  # (H, 18) in [-1, 1]

        # Unnormalize [-1, 1] -> physical 18-DoF joint targets
        mid = 0.5 * (self.act_max + self.act_min)
        half = np.maximum(0.5 * (self.act_max - self.act_min), 1e-5)
        actions_18d = norm_chunk * half[None, :] + mid[None, :]
        actions_18d[:, 7] = np.clip(actions_18d[:, 7], 0.0, 1.0)
        actions_18d[:, 15] = np.clip(actions_18d[:, 15], 0.0, 1.0)

        return {
            "actions_18d": actions_18d.tolist(),
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }


def make_handler(svc: AAWRPolicyService):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:
            return

        def _json(self, code: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._json(200, {"status": "ok", "ckpt": str(svc.ckpt_dir), "device": svc.device})
                return
            self._json(404, {"error": "not_found"})

        def do_POST(self) -> None:
            if self.path != "/act":
                self._json(404, {"error": "not_found"})
                return
            try:
                n = int(self.headers.get("Content-Length", "0"))
                req = json.loads(self.rfile.read(n).decode("utf-8"))
                out = svc.predict(
                    instruction=str(req.get("instruction", "")),
                    images_b64=list(req["images_b64"]),
                    qpos_18=list(req["qpos_18"]),
                )
                self._json(200, out)
            except Exception as exc:
                self._json(500, {"error": f"{type(exc).__name__}: {exc}"})

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="AAWR Unprivileged Policy HTTP Server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=19400)
    parser.add_argument("--aawr-ckpt", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    svc = AAWRPolicyService(args.aawr_ckpt, device=args.device)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(svc))
    print(
        json.dumps(
            {
                "event": "aawr_server_ready",
                "url": f"http://{args.host}:{args.port}/act",
                "ckpt": str(svc.ckpt_dir),
            }
        ),
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
