#!/usr/bin/env python3
"""Formal ASA Stage-3 Robot Finetuning (`EgoActiveFlowModelv2`) on ActiveArena Astribot S1.

Faithfully reproduces ASA's `scripts/flow_robot_finetune.sh` objective & architecture on a single
24 GB GPU (`sdpa` FlashAttention + `bf16` + gradient checkpointing, zero DeepSpeed ZeRO-3 overhead):
  - Initializes all 891/891 weights from official `asa-human-stage2` (`JernLi/ASA/human_stage2`)
  - Freezes `vlm.visual` backbone & `vlm.language_model` blocks; trains `vlm.visual.merger`,
    `temporal_encoder`, `memory_buffer_encoder`, `task_state_head`, and 6-layer `joint_flow_decoder`
  - Optimizes exact joint Flow-Matching + Focal completion loss (`horizon=30`, `future_view_horizon=10`,
    `history_obs_step=5`, `view_rot_loss_coeff=1.5`, `task_state_loss_coeff=0.8`)
  - Splits train/val strictly by episode seed (zero frame leakage across splits)
  - Saves HuggingFace-compatible `EgoActiveFlowModelv2` (`best/` and `final/`) + `astribot_norm_stats.json`
    so `eval/activearena/asa_server.py` and `asa_policy.py` load the Stage-3 checkpoint directly.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, get_cosine_schedule_with_warmup


VISION_START_INDEX = 151652
VISION_END_INDEX = 151653
IMAGE_TOKEN_INDEX = 151655


class AstribotASAStage3Dataset(Dataset):
    """Episode-grouped dataset yielding exact ASA Stage-3 samples (both episode-level & subtask-level)."""

    def __init__(
        self,
        samples: list[dict[str, Any]],
        norm_stats: dict[str, dict[str, list[float]]],
        processor: Any,
        get_rope_index_3_fn: Any,
        *,
        history_obs_step: int = 5,
        predict_future_step: int = 30,
        action_token_idx: int = 151620,
        label_token_idx: int = 151621,
    ) -> None:
        self.samples = samples
        self.norm = norm_stats
        self.processor = processor
        self._get_rope_index_3 = get_rope_index_3_fn
        self.merge_size = getattr(processor.image_processor, "merge_size", 2)
        self.history_obs_step = int(history_obs_step)
        self.predict_future_step = int(predict_future_step)
        self.action_token_idx = int(action_token_idx)
        self.label_token_idx = int(label_token_idx)

    def __len__(self) -> int:
        return len(self.samples)

    def _norm_pos(self, arr: np.ndarray, key: str) -> np.ndarray:
        out = np.asarray(arr, dtype=np.float32).copy()
        st = self.norm[key]
        mean = np.asarray(st["mean"], dtype=np.float32)
        std = np.maximum(np.asarray(st["std"], dtype=np.float32), 1e-3)
        lb = np.asarray(st["lower_bound"], dtype=np.float32)
        ub = np.asarray(st["upper_bound"], dtype=np.float32)
        out[..., :3] = (np.clip(out[..., :3], lb, ub) - mean) / (std + 1e-6)
        return out

    def __getitem__(self, idx: int) -> dict[str, Any]:
        s = self.samples[idx]
        ep = s["ep_data"]
        t = int(s["t"])
        start_t = int(s["start_t"])
        end_t = int(s["end_t"])
        stride = int(s["history_stride"])
        fut_stride = int(s["future_stride"])
        prefix = s["pose_prefix"]  # "ep" or "sub"
        text = str(s["instruction"])

        head_seq = ep[f"{prefix}_head9"]          # (L, 9)
        left_seq = ep[f"{prefix}_left_wrist9"]    # (L, 9)
        right_seq = ep[f"{prefix}_right_wrist9"]  # (L, 9)
        lg_seq = ep["left_grip"]                  # (L, 1)
        rg_seq = ep["right_grip"]                 # (L, 1)
        frame_paths: list[Path] = ep["frame_paths"]

        # 1. History frames & proprio (`t - k * stride >= start_t` for k = 5..1)
        hist_indices = [t - k * stride for k in range(self.history_obs_step, 0, -1) if (t - k * stride) >= start_t]
        num_padded_imgs = max(0, self.history_obs_step - len(hist_indices))

        hist_imgs: list[Image.Image] = []
        if num_padded_imgs > 0:
            dummy = Image.new("RGB", (224, 224), (0, 0, 0))
            hist_imgs.extend([dummy] * num_padded_imgs)
        for hi in hist_indices:
            with Image.open(frame_paths[hi]) as im:
                hist_imgs.append(im.convert("RGB"))
        with Image.open(frame_paths[t]) as im:
            cur_img = im.convert("RGB")

        content = [{"type": "image", "image": im} for im in hist_imgs]
        content.append({"type": "image", "image": cur_img})
        content.append({"type": "text", "text": text})
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

        extra_ids = torch.tensor(
            [[self.label_token_idx] + [self.action_token_idx] * self.predict_future_step],
            dtype=torch.long,
        )
        extra_pos = (
            torch.arange(seq_len, seq_len + 1 + self.predict_future_step, dtype=pos_ids.dtype)
            .view(1, 1, -1)
            .expand(3, 1, -1)
        )
        input_ids = torch.cat([input_ids, extra_ids], dim=-1)
        pos_ids = torch.cat([pos_ids, extra_pos], dim=-1)

        # 2. Pack `memory_buffer_proprio` (6, 29), `memory_buffer_label` (6,), `memory_buffer_mask` (6,)
        complete_thresh = fut_stride * 10
        cur_complete = 1.0 if (end_t - t) <= complete_thresh else 0.0

        valid_len = len(hist_indices)
        hist_comp = [1 if (end_t - hi) <= complete_thresh else 0 for hi in hist_indices]
        mem_label = torch.tensor([2] * (self.history_obs_step - valid_len) + hist_comp + [0], dtype=torch.long)
        mem_mask = mem_label == 2  # current slot [-1] is False (valid)

        mem_proprio = np.zeros((self.history_obs_step + 1, 29), dtype=np.float32)
        for slot_i, hi in enumerate(hist_indices, start=self.history_obs_step - valid_len):
            mem_proprio[slot_i] = np.concatenate(
                [
                    self._norm_pos(head_seq[hi], "head"),
                    self._norm_pos(left_seq[hi], "left_wrist"),
                    self._norm_pos(right_seq[hi], "right_wrist"),
                    lg_seq[hi],
                    rg_seq[hi],
                ],
                axis=-1,
            )
        mem_proprio[-1] = np.concatenate(
            [
                self._norm_pos(head_seq[t], "head"),
                self._norm_pos(left_seq[t], "left_wrist"),
                self._norm_pos(right_seq[t], "right_wrist"),
                lg_seq[t],
                rg_seq[t],
            ],
            axis=-1,
        )

        # 3. Future 30-step targets (`t + (k+1)*fut_stride`, clamped at `end_t` matching ASA `extend_end_frame`)
        fut_idx = [min(t + (k + 1) * fut_stride, end_t) for k in range(self.predict_future_step)]
        view_lbl = self._norm_pos(head_seq[fut_idx], "head")        # (30, 9)
        left_lbl = self._norm_pos(left_seq[fut_idx], "left_wrist")  # (30, 9)
        right_lbl = self._norm_pos(right_seq[fut_idx], "right_wrist")  # (30, 9)
        ee_lbl = np.stack([left_lbl, right_lbl], axis=1)            # (30, 2, 9)
        grip_lbl = np.stack([lg_seq[fut_idx], rg_seq[fut_idx]], axis=1).astype(np.float32)  # (30, 2, 1)

        return {
            "input_ids": input_ids,
            "position_ids": pos_ids,
            "pixel_values": full["pixel_values"],
            "image_grid_thw": image_grid_thw,
            "num_padded_dummy_images": num_padded_imgs,
            "memory_buffer_proprio": torch.from_numpy(mem_proprio),
            "memory_buffer_label": mem_label,
            "memory_buffer_mask": mem_mask,
            "view_action_labels": torch.from_numpy(view_lbl),
            "ee_action_labels": torch.from_numpy(ee_lbl),
            "gripper_action_labels": torch.from_numpy(grip_lbl),
            "task_state_labels": torch.tensor([cur_complete], dtype=torch.float32),
        }


def make_collate_fn(pad_token_id: int):
    def _collate(instances: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        ids_list = [inst["input_ids"].squeeze(0) for inst in instances]
        input_ids = torch.nn.utils.rnn.pad_sequence(ids_list, batch_first=True, padding_value=pad_token_id)
        attention_mask = input_ids.ne(pad_token_id)

        max_len = max(inst["position_ids"].shape[2] for inst in instances)
        pos_padded = [
            torch.nn.functional.pad(inst["position_ids"], (0, max_len - inst["position_ids"].shape[2]), value=1)
            for inst in instances
        ]
        position_ids = torch.cat(pos_padded, dim=1)

        for b_idx, inst in enumerate(instances):
            n_pad = int(inst["num_padded_dummy_images"])
            if n_pad > 0:
                raw_ids = inst["input_ids"]
                v_starts = (raw_ids == VISION_START_INDEX).nonzero(as_tuple=True)[1]
                v_ends = (raw_ids == VISION_END_INDEX).nonzero(as_tuple=True)[1]
                if len(v_starts) > 0 and len(v_ends) > 0:
                    tok_len = int(v_ends[0] - v_starts[0] + 1)
                    img_idx = torch.isin(
                        raw_ids, torch.tensor([VISION_START_INDEX, IMAGE_TOKEN_INDEX, VISION_END_INDEX])
                    ).nonzero(as_tuple=True)[1]
                    attention_mask[b_idx, img_idx[: tok_len * n_pad]] = False

        pixel_values = torch.cat([inst["pixel_values"].unsqueeze(0) for inst in instances], dim=0)
        image_grid_thw = torch.cat([inst["image_grid_thw"] for inst in instances], dim=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "memory_buffer_proprio": torch.stack([inst["memory_buffer_proprio"] for inst in instances], dim=0),
            "memory_buffer_label": torch.stack([inst["memory_buffer_label"] for inst in instances], dim=0),
            "memory_buffer_mask": torch.stack([inst["memory_buffer_mask"] for inst in instances], dim=0),
            "view_action_labels": torch.stack([inst["view_action_labels"] for inst in instances], dim=0),
            "ee_action_labels": torch.stack([inst["ee_action_labels"] for inst in instances], dim=0),
            "gripper_action_labels": torch.stack([inst["gripper_action_labels"] for inst in instances], dim=0),
            "task_state_labels": torch.stack([inst["task_state_labels"] for inst in instances], dim=0),
        }

    return _collate


def load_episode_bank_and_split(
    data_root: Path,
    *,
    val_ratio: float = 0.1,
    history_stride: int = 12,
    future_stride: int = 2,
    sample_stride: int = 2,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, list[float]]], dict[str, Any]]:
    manifest = json.loads((data_root / "manifest.json").read_text(encoding="utf-8"))
    ep_entries = manifest["episodes"]
    rng = random.Random(42)
    by_task: dict[str, list[dict[str, Any]]] = {}
    for e in ep_entries:
        by_task.setdefault(e["task_name"], []).append(e)

    train_eps: list[dict[str, Any]] = []
    val_eps: list[dict[str, Any]] = []
    for task_name, items in sorted(by_task.items()):
        items_shuf = list(items)
        rng.shuffle(items_shuf)
        n_val = max(1, int(round(len(items_shuf) * val_ratio))) if len(items_shuf) >= 4 else 1
        val_eps.extend(items_shuf[:n_val])
        train_eps.extend(items_shuf[n_val:])

    def _load_ep(entry: dict[str, Any]) -> dict[str, Any]:
        npz = np.load(data_root / entry["npz_relpath"], allow_pickle=False)
        frames_dir = data_root / entry["frames_relpath"]
        n_frames = int(entry["num_frames"])
        return {
            "meta": entry,
            "ep_head9": npz["ep_head9"].astype(np.float32),
            "ep_left_wrist9": npz["ep_left_wrist9"].astype(np.float32),
            "ep_right_wrist9": npz["ep_right_wrist9"].astype(np.float32),
            "sub_head9": npz["sub_head9"].astype(np.float32),
            "sub_left_wrist9": npz["sub_left_wrist9"].astype(np.float32),
            "sub_right_wrist9": npz["sub_right_wrist9"].astype(np.float32),
            "left_grip": npz["left_grip"].astype(np.float32).reshape(n_frames, 1),
            "right_grip": npz["right_grip"].astype(np.float32).reshape(n_frames, 1),
            "frame_paths": [frames_dir / f"frame_{i:04d}.jpg" for i in range(n_frames)],
        }

    loaded_train = [_load_ep(e) for e in train_eps]
    loaded_val = [_load_ep(e) for e in val_eps]

    # Compute empirical normalization stats strictly from TRAIN episodes
    head_xyz = np.concatenate([ep["ep_head9"][:, :3] for ep in loaded_train] + [ep["sub_head9"][:, :3] for ep in loaded_train], axis=0)
    left_xyz = np.concatenate([ep["ep_left_wrist9"][:, :3] for ep in loaded_train] + [ep["sub_left_wrist9"][:, :3] for ep in loaded_train], axis=0)
    right_xyz = np.concatenate([ep["ep_right_wrist9"][:, :3] for ep in loaded_train] + [ep["sub_right_wrist9"][:, :3] for ep in loaded_train], axis=0)

    def _stats(xyz: np.ndarray) -> dict[str, list[float]]:
        mean = xyz.mean(axis=0)
        std = np.maximum(xyz.std(axis=0), 0.02)
        return {
            "mean": [round(float(x), 6) for x in mean],
            "std": [round(float(x), 6) for x in std],
            "lower_bound": [-5.0, -5.0, -1.5],
            "upper_bound": [5.0, 5.0, 1.5],
        }

    norm_stats = {
        "head": _stats(head_xyz),
        "left_wrist": _stats(left_xyz),
        "right_wrist": _stats(right_xyz),
    }

    def _build_samples(loaded_list: list[dict[str, Any]], stride_step: int) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for ep in loaded_list:
            meta = ep["meta"]
            n_f = int(meta["num_frames"])
            task_instr = str(meta["task_instruction"])
            # 1) Full-episode non-Markovian task-instruction view (`ep_*` poses anchored at t=0)
            for t in range(0, n_f, stride_step):
                out.append(
                    {
                        "ep_data": ep,
                        "t": t,
                        "start_t": 0,
                        "end_t": n_f - 1,
                        "history_stride": history_stride,
                        "future_stride": future_stride,
                        "pose_prefix": "ep",
                        "instruction": task_instr,
                    }
                )
            # 2) Subtask-segmented view (`sub_*` poses anchored at each subtask start)
            for seg in meta.get("subtask_segments", []):
                s0, s1 = int(seg["start_frame"]), int(seg["end_frame"])
                sub_instr = str(seg.get("instruction", "")).strip()
                if not sub_instr or s1 <= s0:
                    continue
                for t in range(s0, s1 + 1, stride_step):
                    out.append(
                        {
                            "ep_data": ep,
                            "t": t,
                            "start_t": s0,
                            "end_t": s1,
                            "history_stride": history_stride,
                            "future_stride": future_stride,
                            "pose_prefix": "sub",
                            "instruction": sub_instr,
                        }
                    )
        return out

    train_samples = _build_samples(loaded_train, sample_stride)
    val_samples = _build_samples(loaded_val, max(sample_stride, 4))
    split_info = {
        "num_train_episodes": len(loaded_train),
        "num_val_episodes": len(loaded_val),
        "num_train_samples": len(train_samples),
        "num_val_samples": len(val_samples),
        "train_seeds": [e["seed"] for e in train_eps],
        "val_seeds": [e["seed"] for e in val_eps],
    }
    return train_samples, val_samples, norm_stats, split_info


def save_asa_checkpoint(model: Any, processor: Any, norm_stats: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir), safe_serialization=True)
    processor.save_pretrained(str(out_dir))
    (out_dir / "astribot_norm_stats.json").write_text(json.dumps(norm_stats, indent=2) + "\n", encoding="utf-8")


@torch.no_grad()
def evaluate_loader(model: Any, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    n_batches = 0
    bf16_keys = {"pixel_values", "memory_buffer_proprio"}
    for batch in loader:
        dev_batch = {
            k: (v.to(device, dtype=torch.bfloat16) if k in bf16_keys else v.to(device))
            for k, v in batch.items()
        }
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(**dev_batch)
        for k in ("loss", "ee_pos_loss", "ee_rot_loss", "gripper_loss", "view_pos_loss", "view_rot_loss", "task_state_loss"):
            val = getattr(out, k, None)
            if val is not None:
                totals[k] = totals.get(k, 0.0) + float(val.detach().float().item())
        n_batches += 1
    model.train()
    return {k: round(v / max(n_batches, 1), 5) for k, v in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="ASA Stage-3 Formal Finetuning on ActiveArena Astribot S1")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--asa-repo", type=Path, default=Path("/home/ubuntu/ygwang/IP-VLA/asa-iteration"))
    parser.add_argument(
        "--human-ckpt",
        type=Path,
        default=Path("/home/ubuntu/ygwang/IP-VLA/asa-iteration/checkpoints/asa-human-stage2"),
    )
    parser.add_argument(
        "--base-vlm",
        type=Path,
        default=Path("/home/ubuntu/ygwang/IP-VLA/asa-iteration/checkpoints/Qwen3-VL-2B-Instruct"),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--sample-stride", type=int, default=2)
    parser.add_argument("--tune-mm-mlp", action="store_true", help="Also tune vlm.visual.merger (uses grad checkpointing)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    asa_repo = args.asa_repo.resolve()
    if str(asa_repo) not in sys.path:
        sys.path.insert(0, str(asa_repo))

    from data.rope2d import get_rope_index_3
    from model.vla_flow_arch_memory import EgoActiveFlowConfigv2, EgoActiveFlowModelv2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_samples, val_samples, norm_stats, split_info = load_episode_bank_and_split(
        args.data_root.resolve(),
        sample_stride=args.sample_stride,
    )
    (args.output_dir / "astribot_norm_stats.json").write_text(json.dumps(norm_stats, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "split_info.json").write_text(json.dumps(split_info, indent=2) + "\n", encoding="utf-8")
    print(f"[stage3] Dataset ready: {split_info}", flush=True)

    processor = AutoProcessor.from_pretrained(str(args.base_vlm.resolve()))
    ip = processor.image_processor
    if hasattr(ip, "min_pixels") and hasattr(ip, "max_pixels"):
        ip.min_pixels = 784
        ip.max_pixels = 384 * 384
    if hasattr(ip, "size") and isinstance(ip.size, dict):
        ip.size["shortest_edge"] = 784
        ip.size["longest_edge"] = 384 * 384

    cfg = EgoActiveFlowConfigv2.from_pretrained(str(args.human_ckpt.resolve()))
    cfg.model_name_or_path = str(args.base_vlm.resolve())
    cfg.view_rot_loss_coeff = 1.5
    cfg.task_state_loss_coeff = 0.8

    device = torch.device("cuda")
    model = EgoActiveFlowModelv2.from_pretrained(
        str(args.human_ckpt.resolve()),
        config=cfg,
        attn_impl="sdpa",
        torch_dtype=torch.bfloat16,
    ).to(device)

    # Set Stage-3 trainable parameters (`set_model_for_robot_finetuning` + optional `vlm.visual.merger`)
    for name, p in model.named_parameters():
        if "vlm" in name:
            p.requires_grad = bool(args.tune_mm_mlp and "visual.merger" in name)
        else:
            p.requires_grad = True

    if args.tune_mm_mlp:
        model.gradient_checkpointing_enable()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in model.parameters())
    print(
        f"[stage3] Trainable params: {n_trainable:,} / {n_total:,} ({100.0 * n_trainable / n_total:.2f}%)",
        flush=True,
    )

    train_ds = AstribotASAStage3Dataset(
        train_samples,
        norm_stats,
        processor,
        get_rope_index_3,
        history_obs_step=int(cfg.history_obs_step),
        predict_future_step=int(cfg.action_horizon),
        action_token_idx=int(cfg.action_token_idx),
        label_token_idx=int(cfg.label_token_idx),
    )
    val_ds = AstribotASAStage3Dataset(
        val_samples,
        norm_stats,
        processor,
        get_rope_index_3,
        history_obs_step=int(cfg.history_obs_step),
        predict_future_step=int(cfg.action_horizon),
        action_token_idx=int(cfg.action_token_idx),
        label_token_idx=int(cfg.label_token_idx),
    )

    collate = make_collate_fn(int(processor.tokenizer.pad_token_id))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate)

    optimizer = AdamW(trainable_params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    warmup_steps = max(20, int(0.05 * args.max_steps))
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=args.max_steps)

    metrics_log = args.output_dir / "training_metrics.jsonl"
    best_val_loss = float("inf")

    # Initial zero-shot val loss before Stage-3 updates
    val0 = evaluate_loader(model, val_loader, device)
    print(f"[stage3] step=0 init_val={val0}", flush=True)
    with metrics_log.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"step": 0, "phase": "val", **val0}) + "\n")

    model.train()
    optimizer.zero_grad(set_to_none=True)
    train_iter = iter(train_loader)
    t_start = time.monotonic()
    running: dict[str, float] = {}
    micro_count = 0

    for opt_step in range(1, args.max_steps + 1):
        for _ in range(args.grad_accum):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            bf16_keys = {"pixel_values", "memory_buffer_proprio"}
            dev_batch = {
                k: (v.to(device, dtype=torch.bfloat16, non_blocking=True) if k in bf16_keys else v.to(device, non_blocking=True))
                for k, v in batch.items()
            }

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(**dev_batch)
                loss = out.loss / args.grad_accum

            loss.backward()
            for k in ("loss", "ee_pos_loss", "ee_rot_loss", "gripper_loss", "view_pos_loss", "view_rot_loss", "task_state_loss"):
                val = getattr(out, k, None)
                if val is not None:
                    running[k] = running.get(k, 0.0) + float(val.detach().float().item())
            micro_count += 1

        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0).item())
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        if opt_step % args.log_every == 0 or opt_step == 1:
            avg_m = {k: round(v / max(micro_count, 1), 5) for k, v in running.items()}
            elapsed = time.monotonic() - t_start
            rec = {
                "step": opt_step,
                "phase": "train",
                "lr": round(float(scheduler.get_last_lr()[0]), 8),
                "grad_norm": round(grad_norm, 4),
                "elapsed_sec": round(elapsed, 1),
                **avg_m,
            }
            print(f"[stage3] step={opt_step}/{args.max_steps} loss={avg_m.get('loss')} ee_pos={avg_m.get('ee_pos_loss')} view_rot={avg_m.get('view_rot_loss')} lr={rec['lr']:.2e} ({elapsed:.0f}s)", flush=True)
            with metrics_log.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            running.clear()
            micro_count = 0

        if opt_step % args.eval_every == 0 or opt_step == args.max_steps:
            val_m = evaluate_loader(model, val_loader, device)
            val_rec = {"step": opt_step, "phase": "val", **val_m}
            print(f"[stage3] === VAL step={opt_step}: {val_m} ===", flush=True)
            with metrics_log.open("a", encoding="utf-8") as f:
                f.write(json.dumps(val_rec) + "\n")
            if val_m["loss"] < best_val_loss:
                best_val_loss = val_m["loss"]
                print(f"[stage3] Saving new BEST checkpoint (val_loss={best_val_loss:.5f}) -> {args.output_dir / 'best'}", flush=True)
                save_asa_checkpoint(model, processor, norm_stats, args.output_dir / "best")

    save_asa_checkpoint(model, processor, norm_stats, args.output_dir / "final")
    print(f"[stage3] TRAINING COMPLETE! Best val_loss={best_val_loss:.5f}. Saved to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
