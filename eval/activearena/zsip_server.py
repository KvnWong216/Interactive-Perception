#!/usr/bin/env python3
"""Zero-Shot Interactive Perception (ZS-IP, arXiv:2602.18374) — Frozen VLM Server.

Uses 100% frozen pretrained `Qwen3-VL-2B-Instruct` (0 training, 0 LoRA, matching budget parity)
to select discrete robotic manipulation primitives over Enhanced Observation (`K_i`, `P_i->D_i`, `G_i`)
and pinned episodic reveal images.
"""
from __future__ import annotations

import argparse
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import re
import time
from typing import Any

from PIL import Image
import torch


VALID_PRIMITIVES = {"SCAN_HEAD", "PUSH", "GRASP", "INSPECT_HELD", "PLACE", "DONE"}

SYSTEM_PROMPT = """You are the Zero-Shot Interactive Perception (ZS-IP) VLM planner controlling a bimanual tabletop robot.
You receive:
1. Optional [MEMORY REVEAL IMAGE(S)] captured earlier when inspecting the hidden backside of a held object;
2. The [CURRENT ENHANCED OBSERVATION] image annotated with Keypoints (K1, K2, ...), Pushlines (P1->D1, ...), and Workspace Grid sectors (G1..G6);
3. The task instruction, Enhanced Observation anchor catalog, and Memory Context Lookup.

Available discrete action primitives and preconditions:
- SCAN_HEAD(left | center | right) : pan head & waist camera to find unseen target objects or destination color pads not yet visible in the current image (avoid repeating already-scanned empty directions).
- PUSH(P1 | P2 | ...)              : push an occluding clutter object along pushline Pi->Di if clutter blocks the target.
- GRASP(K1 | K2 | ...)             : ONLY when grippers are EMPTY, pick up the movable target object specified by the instruction (e.g. gray block at mid_workspace_object or cola can near center; do NOT grasp flat near-zone sorting pads).
- INSPECT_HELD(self)               : ONLY when HOLDING an object with backside_inspected=False, rotate the held object in front of the head camera to reveal its hidden backside color patch or production date label.
- PLACE(K1 | K2 | ... | G1..G6)    : ONLY when HOLDING an object with backside_inspected=True, carefully read the [MEMORY REVEAL IMAGE] (color patch or PROD DATE vs today's date & shelf life), locate the matching destination pad Ki (or SCAN_HEAD first if that color pad is outside current view), and place the object onto Ki.
- DONE(self)                       : ONLY after PLACE has succeeded and grippers are EMPTY.

Output strictly a single JSON object:
{"thought": "<brief 1-sentence reasoning>", "primitive": "<PRIMITIVE_NAME>", "target": "<ANCHOR_OR_DIR>"}"""


def _decode_pil(b64_str: str) -> Image.Image:
    raw = base64.b64decode(b64_str)
    with Image.open(io.BytesIO(raw)) as im:
        return im.convert("RGB")


def parse_zsip_vlm_output(raw_text: str) -> dict[str, str]:
    text = raw_text.strip()
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            prim = str(obj.get("primitive", "")).strip().upper()
            tgt = str(obj.get("target", "center")).strip()
            if prim in VALID_PRIMITIVES:
                return {
                    "thought": str(obj.get("thought", ""))[:200],
                    "primitive": prim,
                    "target": tgt,
                }
        except Exception:
            pass

    for prim in ("INSPECT_HELD", "GRASP", "PLACE", "PUSH", "SCAN_HEAD", "DONE"):
        if prim in text.upper():
            tgt_m = re.search(r"\b(K\d+|P\d+|G\d+|left|right|center|self)\b", text, flags=re.IGNORECASE)
            return {
                "thought": text[:160],
                "primitive": prim,
                "target": tgt_m.group(1) if tgt_m else "self",
            }
    return {"thought": text[:160], "primitive": "SCAN_HEAD", "target": "center"}


class ZSIPFrozenVLMServer:
    def __init__(self, base_vlm: Path, device: str = "cuda") -> None:
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.device = torch.device(device)
        self.base_vlm = str(base_vlm)
        self.processor = AutoProcessor.from_pretrained(self.base_vlm)
        self.model = (
            Qwen3VLForConditionalGeneration.from_pretrained(
                self.base_vlm,
                attn_implementation="sdpa",
                torch_dtype=torch.bfloat16,
            )
            .to(self.device)
            .eval()
        )

    @torch.no_grad()
    def decide(
        self,
        *,
        instruction: str,
        eo_image_b64: str,
        reveal_images_b64: list[str],
        eo_catalog: str,
        memory_context: str,
        max_new_tokens: int = 96,
    ) -> dict[str, Any]:
        t0 = time.monotonic()
        content: list[dict[str, Any]] = []
        for idx, b64 in enumerate(reveal_images_b64[-2:], 1):
            content.append({"type": "text", "text": f"[MEMORY REVEAL IMAGE #{idx} (captured during INSPECT_HELD)]:"})
            content.append({"type": "image", "image": _decode_pil(b64)})

        content.append({"type": "text", "text": "[CURRENT ENHANCED OBSERVATION]:"})
        content.append({"type": "image", "image": _decode_pil(eo_image_b64)})
        user_prompt = (
            f"Task Instruction: {instruction}\n\n"
            f"{eo_catalog}\n\n"
            f"{memory_context}\n\n"
            "Select the single best next primitive as JSON."
        )
        content.append({"type": "text", "text": user_prompt})

        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": content},
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)

        gen_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        out_ids = gen_ids[0, inputs["input_ids"].shape[1] :]
        raw_text = self.processor.decode(out_ids, skip_special_tokens=True)
        parsed = parse_zsip_vlm_output(raw_text)
        parsed["raw_text"] = raw_text
        parsed["latency_ms"] = round((time.monotonic() - t0) * 1000.0, 2)
        return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="ZS-IP Frozen Qwen3-VL-2B Server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=19300)
    parser.add_argument(
        "--base-vlm",
        type=Path,
        default=Path("/home/ubuntu/ygwang/IP-VLA/asa-iteration/checkpoints/Qwen3-VL-2B-Instruct"),
    )
    args = parser.parse_args()

    engine = ZSIPFrozenVLMServer(base_vlm=args.base_vlm.resolve())

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *log_args: Any) -> None:
            return

        def do_GET(self) -> None:
            if self.path == "/health":
                body = json.dumps({"status": "ok", "method": "ZS-IP", "base_vlm": engine.base_vlm}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_error(404)

        def do_POST(self) -> None:
            if self.path != "/decide":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            out = engine.decide(
                instruction=payload["instruction"],
                eo_image_b64=payload["eo_image_b64"],
                reveal_images_b64=payload.get("reveal_images_b64", []),
                eo_catalog=payload["eo_catalog"],
                memory_context=payload["memory_context"],
            )
            body = json.dumps(out, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[zsip_server] Ready on http://{args.host}:{args.port} (frozen VLM={engine.base_vlm})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
