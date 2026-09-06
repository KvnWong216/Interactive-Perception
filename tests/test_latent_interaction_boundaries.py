import ast
from pathlib import Path

import numpy as np
import pytest

from latent_interaction.executor import (
    FrozenVLAActionChunk,
    FrozenVLAObservation,
    validate_frozen_executor,
)
from latent_interaction.token_provider import validate_frozen_token_provider


class _Executor:
    frozen = True
    executor_id = "molmoact2-libero-frozen"

    def act(self, observation, *, subtask_text):
        assert subtask_text
        return FrozenVLAActionChunk(np.zeros((4, 7)), self.executor_id)


class _TokenProvider:
    frozen = True
    provider_id = "qwen3-vl-screen-candidate"
    output_width = 32

    def encode_context(self, **kwargs):
        raise NotImplementedError

    def encode_primitives(self, **kwargs):
        raise NotImplementedError


def test_frozen_executor_protocol_has_no_model_download_side_effect() -> None:
    executor = _Executor()
    validate_frozen_executor(executor)
    observation = FrozenVLAObservation(np.zeros((32, 32, 3), dtype=np.float32))
    chunk = executor.act(observation, subtask_text="Open the middle drawer.")
    assert chunk.actions.shape == (4, 7)


def test_mutable_executor_fails_closed() -> None:
    executor = _Executor()
    executor.frozen = False
    with pytest.raises(ValueError, match="must be frozen"):
        validate_frozen_executor(executor)


def test_frozen_token_provider_protocol_is_model_agnostic() -> None:
    provider = _TokenProvider()
    validate_frozen_token_provider(provider)
    provider.frozen = False
    with pytest.raises(ValueError, match="must be frozen"):
        validate_frozen_token_provider(provider)


def test_new_package_has_no_legacy_or_s03_imports() -> None:
    package = Path(__file__).parents[1] / "src" / "latent_interaction"
    forbidden = {
        "interaction_uncertainty",
        "piu",
        "calibrated_interaction",
        "s03",
    }
    for path in package.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            imported: list[str] = []
            if isinstance(node, ast.Import):
                imported = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported = [node.module]
            for name in imported:
                assert name.split(".", 1)[0] not in forbidden, (path, name)
