#!/usr/bin/env python3
"""Standalone localhost HTTP inference server for upstream ASA (`EgoActiveFlowModelv2`).

Runs inside `asa-iteration/.venv` (`transformers==4.57.1`) and exposes:
  - `GET  /health` -> model & checkpoint metadata
  - `POST /act`    -> exact ASA tokenization (`224x224`, dummy-history span masking, `get_rope_index_3`,
                      `memory_buffer_proprio` / `label` / `mask`) + 10-step Flow-Matching `sample_actions`,
                      returning `view_actions (30, 9)`, `mani_actions (30, 2, 10)`, and `p_complete`.
"""
from __future__ import annotations

import argparse
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import sys
import time
from typing import Any

from PIL import Image
import torch
from transformers import AutoProcessor


VISION_START_INDEX = 151652
VISION_END_INDEX = 151653
IMAGE_TOKEN_INDEX = 151655


def _decode_rgb_224(b64_str: str) -> Image.Image:
    raw = base64.b64decode(b64_str)
    with Image.open(io.BytesIO(raw)) as im:
        return im.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)


class ASAInferenceEngine:
    def __init__(
        self,
        *,
        asa_repo: Path,
        asa_ckpt: Path,
        base_vlm: Path,
        device: str = "cuda",
        flow_steps: int = 10,
    ) -> None:
        if str(asa_repo) not in sys.path:
            sys.path.insert(0, str(asa_repo))

        from data.rope2d import get_rope_index_3
        from model.vla_flow_arch_memory import EgoActiveFlowConfigv2, EgoActiveFlowModelv2

        self._get_rope_index_3 = get_rope_index_3
        self.device = torch.device(device)
        self.flow_steps = int(flow_steps)
        self.asa_ckpt = str(asa_ckpt)
        self.base_vlm = str(base_vlm)

        cfg = EgoActiveFlowConfigv2.from_pretrained(str(asa_ckpt))
        cfg.model_name_or_path = str(base_vlm)
        self.cfg = cfg

        self.processor = AutoProcessor.from_pretrained(str(base_vlm))
        ip = self.processor.image_processor
        if hasattr(ip, "min_pixels") and hasattr(ip, "max_pixels"):
            ip.min_pixels = 784
            ip.max_pixels = 384 * 384
        if hasattr(ip, "size") and isinstance(ip.size, dict):
            ip.size["shortest_edge"] = 784
            ip.size["longest_edge"] = 384 * 384
        self.merge_size = getattr(ip, "merge_size", 2)

        self.model = (
            EgoActiveFlowModelv2.from_pretrained(str(asa_ckpt), config=cfg, torch_dtype=torch.bfloat16)
            .to(self.device)
            .eval()
        )

    @torch.no_grad()
    def predict(
        self,
        *,
        instruction: str,
        current_image_b64: str,
        history_images_b64: list[str],
        current_proprio29: list[float],
        history_proprio29: list[list[float]],
        history_complete: list[int] | None = None,
        num_steps: int | None = None,
    ) -> dict[str, Any]:
        t0 = time.monotonic()
        hist_target_len = int(self.cfg.history_obs_step)  # 5

        hist_imgs = [_decode_rgb_224(b) for b in history_images_b64[-hist_target_len:]]
        cur_img = _decode_rgb_224(current_image_b64)
        num_padded_imgs = max(0, hist_target_len - len(hist_imgs))
        if num_padded_imgs > 0:
            dummy = Image.new("RGB", (224, 224), (0, 0, 0))
            hist_imgs = [dummy] * num_padded_imgs + hist_imgs

        content = [{"type": "image", "image": im} for im in hist_imgs]
        content.append({"type": "image", "image": cur_img})
        content.append({"type": "text", "text": str(instruction)})

        full = self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        input_ids = full["input_ids"]
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
        seq_len = int(input_ids.size(1))
        image_grid_thw = full["image_grid_thw"]

        pos_ids, _ = self._get_rope_index_3(self.merge_size, input_ids, image_grid_thw=image_grid_thw)

        horizon = int(self.cfg.action_horizon)  # 30
        extra_ids = torch.tensor(
            [[int(self.cfg.label_token_idx)] + [int(self.cfg.action_token_idx)] * horizon],
            dtype=torch.long,
        )
        extra_pos = torch.arange(seq_len, seq_len + 1 + horizon, dtype=pos_ids.dtype).view(1, 1, -1).expand(3, 1, -1)
        input_ids = torch.cat([input_ids, extra_ids], dim=-1)
        pos_ids = torch.cat([pos_ids, extra_pos], dim=-1)

        attn_mask = torch.ones_like(input_ids, dtype=torch.bool)
        if num_padded_imgs > 0:
            v_starts = (input_ids == VISION_START_INDEX).nonzero(as_tuple=True)[1]
            v_ends = (input_ids == VISION_END_INDEX).nonzero(as_tuple=True)[1]
            if len(v_starts) > 0 and len(v_ends) > 0:
                img_tok_len = int(v_ends[0] - v_starts[0] + 1)
                img_tok_idx = torch.isin(
                    input_ids,
                    torch.tensor([VISION_START_INDEX, IMAGE_TOKEN_INDEX, VISION_END_INDEX]),
                ).nonzero(as_tuple=True)[1]
                attn_mask[0, img_tok_idx[: img_tok_len * num_padded_imgs]] = False

        # Pack 5 history + 1 current 29D proprio buffer (`memory_alone=True` convention)
        hist_p = [list(p) for p in history_proprio29[-hist_target_len:]]
        valid_len = len(hist_p)
        hist_lbl_raw = (history_complete or [0] * valid_len)[-valid_len:] if valid_len > 0 else []
        mem_label = torch.tensor(
            [[2] * (hist_target_len - valid_len) + [int(x) for x in hist_lbl_raw] + [0]],
            dtype=torch.long,
            device=self.device,
        )
        mem_mask = mem_label == 2  # (1, 6), current slot [-1] is False (unmasked)

        mem_proprio = torch.zeros((1, hist_target_len + 1, int(self.cfg.proprio_size)), dtype=torch.float32)
        if valid_len > 0:
            mem_proprio[0, hist_target_len - valid_len : hist_target_len] = torch.tensor(hist_p, dtype=torch.float32)
        mem_proprio[0, -1] = torch.tensor(current_proprio29, dtype=torch.float32)
        mem_proprio = mem_proprio.to(self.device, dtype=torch.bfloat16)

        pixel_values = full["pixel_values"]
        if pixel_values.dim() == 2:
            pixel_values = pixel_values.unsqueeze(0)

        steps = int(num_steps or self.flow_steps)
        x_view, x_mani, p_complete = self.model.sample_actions(
            input_ids=input_ids.to(self.device),
            attention_mask=attn_mask.to(self.device),
            position_ids=pos_ids.to(self.device),
            pixel_values=pixel_values.to(self.device, dtype=torch.bfloat16),
            image_grid_thw=image_grid_thw.to(self.device),
            view_proprio=mem_proprio[:, -1:, :9],
            ee_proprio=torch.stack([mem_proprio[:, -1, 9:18], mem_proprio[:, -1, 18:27]], dim=1),
            gripper_proprio=torch.stack([mem_proprio[:, -1, 27:28], mem_proprio[:, -1, 28:29]], dim=1),
            memory_buffer_proprio=mem_proprio,
            memory_buffer_label=mem_label,
            memory_buffer_mask=mem_mask,
            device=self.device,
            num_steps=steps,
        )

        latency_ms = (time.monotonic() - t0) * 1000.0
        return {
            "view_actions": x_view[0].float().cpu().tolist(),   # (30, 9)
            "mani_actions": x_mani[0].float().cpu().tolist(),   # (30, 2, 10)
            "p_complete": float(p_complete.reshape(-1)[0].float().cpu().item()),
            "action_horizon": horizon,
            "future_view_horizon": int(self.cfg.future_view_horizon),
            "latency_ms": round(latency_ms, 2),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="ASA EgoActiveFlowModelv2 Inference Server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=19200)
    parser.add_argument("--asa-repo", default="/home/ubuntu/ygwang/IP-VLA/asa-iteration")
    parser.add_argument(
        "--asa-ckpt",
        default="/home/ubuntu/ygwang/IP-VLA/asa-iteration/checkpoints/asa-human-stage2",
    )
    parser.add_argument(
        "--base-vlm",
        default="/home/ubuntu/ygwang/IP-VLA/asa-iteration/checkpoints/Qwen3-VL-2B-Instruct",
    )
    parser.add_argument("--flow-steps", type=int, default=10)
    args = parser.parse_args()

    engine = ASAInferenceEngine(
        asa_repo=Path(args.asa_repo).resolve(),
        asa_ckpt=Path(args.asa_ckpt).resolve(),
        base_vlm=Path(args.base_vlm).resolve(),
        flow_steps=args.flow_steps,
    )

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *log_args: Any) -> None:
            return

        def do_GET(self) -> None:
            if self.path == "/health":
                body = json.dumps(
                    {
                        "status": "ok",
                        "model": "EgoActiveFlowModelv2",
                        "asa_ckpt": engine.asa_ckpt,
                        "action_horizon": int(engine.cfg.action_horizon),
                    }
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_error(404)

        def do_POST(self) -> None:
            if self.path != "/act":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            out = engine.predict(
                instruction=payload["instruction"],
                current_image_b64=payload["current_image_b64"],
                history_images_b64=payload.get("history_images_b64", []),
                current_proprio29=payload["current_proprio29"],
                history_proprio29=payload.get("history_proprio29", []),
                history_complete=payload.get("history_complete"),
                num_steps=payload.get("num_steps"),
            )
            body = json.dumps(out).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[asa_server] Ready on http://{args.host}:{args.port} (ckpt={engine.asa_ckpt})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
