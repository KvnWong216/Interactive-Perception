from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from grounded_interaction.psr.molmo_backend import (
    MolmoCapabilities,
    MolmoPSRBackend,
    _find_subsequence,
    _make_toggleable_lora,
    _postprocess_conditioned_actions,
    content_seed,
)


def test_toggleable_lora_is_exact_bypass_and_trainable_delta() -> None:
    torch.manual_seed(3)
    base = torch.nn.Linear(4, 5)
    original = base(torch.randn(2, 4)).detach()
    enabled = False
    wrapped = _make_toggleable_lora(base, rank=2, alpha=4.0, enabled=lambda: enabled)
    inputs = torch.randn(2, 4)
    assert torch.equal(wrapped(inputs), base(inputs))
    assert all(not parameter.requires_grad for parameter in base.parameters())
    enabled = True
    loss = wrapped(inputs).square().mean()
    loss.backward()
    # Up is initialized at zero: it receives the first useful gradient; down
    # receives one after up moves. Both are registered trainable parameters.
    assert wrapped.up.weight.grad is not None
    assert torch.isfinite(wrapped.up.weight.grad).all()
    assert original.shape == (2, 5)


def test_content_seed_and_boundary_are_content_stable() -> None:
    assert content_seed(episode_seed=7, global_step=50, attempt=1) == content_seed(
        episode_seed=7, global_step=50, attempt=1
    )
    assert content_seed(episode_seed=7, global_step=50, attempt=1) != content_seed(
        episode_seed=7, global_step=50, attempt=2
    )
    assert _find_subsequence([1, 2, 3, 2, 3, 4], [2, 3]) == 3


def test_content_seed_rejects_negative_public_temporal_identity() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        content_seed(episode_seed=-1, global_step=0, attempt=0)


def test_prefix_metadata_follows_embedding_segments_without_trigger() -> None:
    # Native fields: [user0, user1, assistant0, assistant1, trigger, tail].
    value = torch.tensor([[10, 11, 12, 13, 14, 15]])
    assembled = MolmoPSRBackend._insert_prefix_field(
        value,
        assistant=2,
        trigger=4,
        history_visual_length=2,
        history_text_length=1,
        state_length=2,
        anchor_length=1,
        fill=0,
    )
    assert assembled.tolist() == [[10, 11, 0, 0, 0, 12, 13, 0, 0, 0]]


def test_role_boundaries_keep_history_inside_last_user_message() -> None:
    class Tokenizer:
        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            assert add_special_tokens is False
            return {
                "<|im_end|>": [7, 8],
                "<|im_start|>assistant\n": [9, 10],
            }[text]

    backend = object.__new__(MolmoPSRBackend)
    backend.tokenizer = Tokenizer()
    backend.capabilities = MolmoCapabilities(  # type: ignore[assignment]
        hidden_dim=16,
        num_vlm_layers=2,
        num_kv_heads=2,
        head_dim=8,
        max_action_horizon=8,
        max_action_dim=7,
        action_trigger_id=99,
        checkpoint_revision="a" * 40,
        upstream_revision="b" * 40,
    )
    input_ids = torch.tensor([[1, 7, 8, 2, 7, 8, 9, 10, 3, 99]])
    user_end, assistant, trigger = backend._role_boundaries(input_ids)
    assert (user_end, assistant, trigger) == (4, 6, 9)


def test_conditioned_actions_use_native_observation_prefix_window() -> None:
    np = pytest.importorskip("numpy")
    calls: list[tuple[object, ...]] = []

    class Outer:
        config = type("Config", (), {"n_obs_steps": 2})()

        @staticmethod
        def _slice_action_dim(actions: object, action_dim: int) -> object:
            calls.append(("dim", action_dim))
            return actions[:, :, :action_dim]

        @staticmethod
        def _slice_action_chunk(
            actions: object, n_obs_steps: int, n_action_steps: int
        ) -> object:
            calls.append(("window", n_obs_steps, n_action_steps))
            return actions[:, n_obs_steps : n_obs_steps + n_action_steps]

    class Stats:
        @staticmethod
        def unnormalize_action(actions: object, tag: str) -> object:
            calls.append(("unnormalize", tag))
            return actions + 100

    raw = np.arange(1 * 6 * 9).reshape(1, 6, 9)
    result = _postprocess_conditioned_actions(
        outer=Outer(),
        actions=raw,
        stats=Stats(),
        tag="libero",
        action_dim=7,
        n_action_steps=3,
    )
    assert calls == [("dim", 7), ("window", 2, 3), ("unnormalize", "libero")]
    np.testing.assert_array_equal(result, raw[:, 2:5, :7] + 100)
