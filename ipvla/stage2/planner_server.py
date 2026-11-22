#!/usr/bin/env python3
"""Autonomous Stage-1/Stage-2 VLM Subtask Planner & Calibrated Confidence HTTP Server.

Serves `POST /plan` and `GET /health` so any simulator environment (including
ActiveArena's isolated `activearena-sim` virtualenv) can query our frozen
`Qwen3-VL-2B-Instruct` (+ Stage-1 v2 / Stage-2 LoRA + Ridge-Temp confidence head)
over localhost with zero Python dependency conflict.

Request JSON (`POST /plan`):
{
  "instruction": "Inspect the gray block and place it on the matching colored pad.",
  "images_b64": ["<jpeg-b64-1>", "<jpeg-b64-2>", "<jpeg-b64-3>", "<jpeg-b64-4>"],
  "max_new_tokens": 28
}

Response JSON:
{
  "subtask": "Place the gray block on the red pad",
  "confidence": 0.7842,
  "token_prob": 0.8513,
  "latency_ms": 172.4,
  "planner_tag": "stage1_v2_ep3"
}
"""
from __future__ import annotations

import argparse
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import math
from pathlib import Path
import threading
import time
from typing import Any, Optional

import numpy as np
from PIL import Image

from ipvla.config import load_paths_config
from ipvla.stage1.confidence import RidgeTempConfidenceHead


def merge_custom_lora_safetensors(model, safetensors_path: Path, alpha: float = 64.0) -> int:
    """Merge IP-VLA Stage-1 `.a` / `.b` LoRA tensors directly into base Linear weights."""
    import torch
    from safetensors.torch import load_file

    tensors = load_file(str(safetensors_path), device="cpu")
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for key, val in tensors.items():
        if key.endswith(".a"):
            pairs.setdefault(key[:-2], {})["a"] = val
        elif key.endswith(".b"):
            pairs.setdefault(key[:-2], {})["b"] = val

    modules = dict(model.named_modules())
    merged = 0
    with torch.no_grad():
        for mod_name, ab in pairs.items():
            if "a" not in ab or "b" not in ab:
                continue
            mod = modules.get(mod_name)
            if mod is None or not hasattr(mod, "weight"):
                raise KeyError(f"LoRA target module not found in base VLM: {mod_name}")
            a = ab["a"].to(device=mod.weight.device, dtype=torch.float32)
            b = ab["b"].to(device=mod.weight.device, dtype=torch.float32)
            rank = int(a.shape[0])
            scale = float(alpha) / max(rank, 1)
            delta = (b @ a) * scale
            mod.weight.data.add_(delta.to(dtype=mod.weight.dtype))
            merged += 1
    return merged


class AutonomousVLMPlanner:
    """Loads Qwen3-VL-2B (+ optional Stage-1/2 LoRA + Ridge-Temp confidence head)."""

    def __init__(
        self,
        base_vlm_dir: str | Path,
        lora_path: Optional[str | Path] = None,
        lora_alpha: float = 64.0,
        confidence_head_path: Optional[str | Path] = None,
        image_width: int = 448,
        image_height: int = 336,
        device: str = "cuda:0",
        planner_tag: str = "stage1_v2_ep3",
    ) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.base_vlm_dir = str(base_vlm_dir)
        self.image_width = int(image_width)
        self.image_height = int(image_height)
        self.device = device
        self.planner_tag = planner_tag
        self._lock = threading.Lock()

        self.processor = AutoProcessor.from_pretrained(self.base_vlm_dir, local_files_only=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.base_vlm_dir,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        )
        self.merged_adapters = 0
        if lora_path and Path(lora_path).is_file():
            self.merged_adapters = merge_custom_lora_safetensors(
                self.model, Path(lora_path), alpha=lora_alpha
            )
        self.model.eval()

        self.conf_head: Optional[RidgeTempConfidenceHead] = None
        if confidence_head_path and Path(confidence_head_path).is_file():
            self.conf_head = RidgeTempConfidenceHead.from_checkpoint(confidence_head_path)

    def plan(
        self,
        instruction: str,
        images: list[Image.Image],
        max_new_tokens: int = 28,
    ) -> dict[str, Any]:
        import torch

        t0 = time.perf_counter()
        resized = [
            im.convert("RGB").resize((self.image_width, self.image_height), Image.Resampling.LANCZOS)
            for im in images
        ]
        prompt_text = (
            f"Goal: {instruction.strip()}\n"
            "The views are ordered oldest to newest. "
            "State one feasible immediate next task as a short imperative sentence."
        )
        messages = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image"} for _ in resized],
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        encoded = self.processor(images=resized, text=[prompt], return_tensors="pt")
        encoded = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in encoded.items()}
        prompt_len = int(encoded["input_ids"].shape[-1])

        with self._lock, torch.no_grad():
            gen = self.model.generate(
                **encoded,
                max_new_tokens=int(max_new_tokens),
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
                output_hidden_states=True,
            )

        seq = gen.sequences[0]
        new_tokens = seq[prompt_len:]
        text = self.processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        text = text.splitlines()[0].strip().strip("\"'")

        # Mean token probability across generated tokens.
        logprobs = []
        for step_idx, scores in enumerate(gen.scores):
            if step_idx >= new_tokens.shape[0]:
                break
            tok_id = int(new_tokens[step_idx].item())
            lp = float(torch.log_softmax(scores[0].float(), dim=-1)[tok_id].item())
            logprobs.append(lp)
        mean_lp = float(sum(logprobs) / max(len(logprobs), 1))
        token_prob = float(math.exp(max(mean_lp, -20.0)))

        # Calibrated decision confidence from last generated token's final-layer hidden state.
        confidence = token_prob
        if self.conf_head is not None and gen.hidden_states:
            last_step_layers = gen.hidden_states[-1]
            last_hidden = last_step_layers[-1][0, -1, :].detach().float().cpu().numpy()
            confidence = self.conf_head.predict_proba(last_hidden)

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "subtask": text,
            "confidence": round(float(confidence), 6),
            "token_prob": round(float(token_prob), 6),
            "mean_token_logprob": round(float(mean_lp), 6),
            "latency_ms": round(latency_ms, 2),
            "planner_tag": self.planner_tag,
        }


def make_handler(planner: AutonomousVLMPlanner):
    class PlannerHTTPHandler(BaseHTTPRequestHandler):
        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args) -> None:
            return

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send_json(
                    200,
                    {
                        "status": "ok",
                        "planner_tag": planner.planner_tag,
                        "merged_adapters": planner.merged_adapters,
                        "has_confidence_head": planner.conf_head is not None,
                    },
                )
                return
            self._send_json(404, {"error": f"Unknown route: {self.path}"})

        def do_POST(self) -> None:
            if self.path != "/plan":
                self._send_json(404, {"error": f"Unknown route: {self.path}"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                req = json.loads(self.rfile.read(length).decode("utf-8"))
                instruction = str(req["instruction"])
                imgs_b64 = req["images_b64"]
                max_new_tokens = int(req.get("max_new_tokens", 28))
                images = []
                for b64 in imgs_b64:
                    raw = base64.b64decode(b64)
                    with Image.open(io.BytesIO(raw)) as im:
                        images.append(im.convert("RGB"))
                result = planner.plan(instruction, images, max_new_tokens=max_new_tokens)
                self._send_json(200, result)
            except Exception as exc:
                self._send_json(500, {"error": f"{type(exc).__name__}: {exc}"})

    return PlannerHTTPHandler


def main() -> None:
    paths = load_paths_config()
    parser = argparse.ArgumentParser(description="IP-VLA Autonomous VLM Subtask Planner Server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=19100)
    parser.add_argument("--base-vlm", default=paths.get("base_vlm_dir", ""))
    parser.add_argument("--lora", default=paths.get("stage1_lora_path", ""))
    parser.add_argument("--no-lora", action="store_true", help="Run base Qwen3-VL-2B without LoRA")
    parser.add_argument("--lora-alpha", type=float, default=64.0)
    parser.add_argument("--confidence-head", default=paths.get("stage1_confidence_head_path", ""))
    parser.add_argument("--no-confidence-head", action="store_true")
    parser.add_argument("--image-width", type=int, default=448)
    parser.add_argument("--image-height", type=int, default=336)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tag", default="stage1_v2_ep3")
    args = parser.parse_args()

    lora_path = None if args.no_lora else args.lora
    conf_path = None if args.no_confidence_head else args.confidence_head
    planner = AutonomousVLMPlanner(
        base_vlm_dir=args.base_vlm,
        lora_path=lora_path,
        lora_alpha=args.lora_alpha,
        confidence_head_path=conf_path,
        image_width=args.image_width,
        image_height=args.image_height,
        device=args.device,
        planner_tag=args.tag,
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(planner))
    print(
        json.dumps(
            {
                "event": "planner_server_ready",
                "url": f"http://{args.host}:{args.port}/plan",
                "planner_tag": planner.planner_tag,
                "merged_adapters": planner.merged_adapters,
                "has_confidence_head": planner.conf_head is not None,
            }
        ),
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
