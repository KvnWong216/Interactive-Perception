"""Versioned configuration, independent of the retired PSR protocol."""

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class VLAConfig:
    schema: str = "geometry-history-action-jepa-v1"
    model_id: str = "allenai/MolmoAct2-LIBERO"
    history_frames: int = 3
    execute_steps: int = 5
    prediction_steps: int = 10
    total_steps: int = 300
    flow_steps: int = 10
    num_flow_samples: int = 8
    train_action_expert: bool = True
    upper_vlm_layers: int = 8
    lora_rank: int = 16
    lora_alpha: float = 32.0
    predictor_kind: str = "absolute"
    predictor_dim: int = 256
    predictor_heads: int = 8
    predictor_layers: int = 2
    geometry: bool = True
    prediction_weight: float = 1.0
    language_weight: float = 0.1
    learning_rate: float = 0.0001
    vlm_learning_rate: float = 0.00002
    action_expert_learning_rate: float = 0.00005
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    accumulation: int = 16
    manual_seed: int = 17
    dtype: str = "bfloat16"
    max_context_tokens: int = 16384
    response_tokens: int = 64

    def __post_init__(self):
        import math

        if self.schema != "geometry-history-action-jepa-v1":
            raise ValueError("unsupported predictive VLA configuration")
        if self.model_id != "allenai/MolmoAct2-LIBERO":
            raise ValueError("the native bridge requires MolmoAct2-LIBERO")
        for name in (
            "history_frames",
            "execute_steps",
            "prediction_steps",
            "total_steps",
            "flow_steps",
            "num_flow_samples",
            "upper_vlm_layers",
            "lora_rank",
            "predictor_dim",
            "predictor_heads",
            "predictor_layers",
            "accumulation",
            "max_context_tokens",
            "response_tokens",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            self.predictor_dim % self.predictor_heads
            or not self.execute_steps <= self.prediction_steps <= self.total_steps
        ):
            raise ValueError("incompatible attention dimensions or execution budget")
        if self.predictor_kind not in {"absolute", "transport", "local_transport"}:
            raise ValueError("unsupported predictor kind")
        if type(self.train_action_expert) is not bool:
            raise ValueError("train_action_expert must be boolean")
        if type(self.geometry) is not bool or self.dtype not in {"float32", "bfloat16"}:
            raise ValueError("invalid geometry/dtype setting")
        if type(self.manual_seed) is not int or not 0 <= self.manual_seed < 2**32:
            raise ValueError("manual_seed must be an integer in [0, 2**32)")
        for name in (
            "lora_alpha",
            "learning_rate",
            "vlm_learning_rate",
            "action_expert_learning_rate",
            "grad_clip",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name in ("prediction_weight", "language_weight", "weight_decay"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative and finite")

    def to_dict(self):
        return asdict(self)


def load_config(path: str | Path) -> VLAConfig:
    import yaml

    value = yaml.safe_load(Path(path).read_text())
    if not isinstance(value, dict):
        raise TypeError("configuration must be a mapping")
    return VLAConfig(**value)


def set_manual_seed(manual_seed: int):
    """Initialize Python, NumPy and PyTorch from the experiment's one seed."""
    import random

    import numpy as np
    import torch

    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(manual_seed)
