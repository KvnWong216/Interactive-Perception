"""Candidate-conditioned action-effect contracts and lightweight predictor."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import Split
from .spatial_prefix import validate_feature_arrays

EFFECT_FACTORS = (
    "execution_succeeded",
    "task_progress_succeeded",
    "task_relevant_change",
    "target_revealed",
    "identity_resolved_post",
    "candidate_rejected",
    "region_confirmed_empty",
    "task_information_sufficient_post",
)


@dataclasses.dataclass(frozen=True)
class EffectLabel:
    """Next-action outcomes joined to an earlier public decision state."""

    sample_id: str
    initial_state_group: str
    split: Split
    candidate_id: str
    candidate_primitive: str
    decision_observation_sha256: str
    outcome_observation_sha256: Mapping[str, str]
    selection_correct: bool
    eligible_for_execution: bool
    executed: bool
    exact_null_transition: bool
    factors: Mapping[str, bool | None]
    simulator_teacher_only: bool

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> EffectLabel:
        if value.get("schema_version") != "piu.action-effect-label.v1":
            raise ValueError("unsupported action-effect label schema")
        factors = dict(value.get("factors", {}))
        if set(factors) != set(EFFECT_FACTORS):
            raise ValueError("effect label must contain every declared factor")
        for name, factor in factors.items():
            if factor is not None and not isinstance(factor, bool):
                raise TypeError(f"effect factor {name} must be boolean or null")
        executed = value.get("executed")
        selection_correct = value.get("selection_correct")
        exact_null = value.get("exact_null_transition")
        eligible = value.get("eligible_for_execution", True)
        teacher = value.get("simulator_teacher_only")
        if not all(
            isinstance(item, bool)
            for item in (executed, exact_null, eligible, teacher, selection_correct)
        ):
            raise TypeError(
                "effect eligibility/execution/null/teacher/selection flags must be booleans"
            )
        primitive = " ".join(str(value.get("candidate_primitive", "")).split()).upper()
        decision_digest = str(value.get("decision_observation_sha256", ""))
        outcome_hashes = dict(value.get("outcome_observation_sha256", {}))
        digests = (
            decision_digest,
            outcome_hashes.get("pre"),
            outcome_hashes.get("post"),
        )
        if any(
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            for digest in digests
        ):
            raise ValueError(
                "effect decision/outcome observations require lowercase SHA-256"
            )
        if decision_digest != outcome_hashes["pre"]:
            raise ValueError(
                "effect rollout must start from the public decision observation"
            )
        if exact_null:
            if (
                primitive not in {"STOP", "REPORT_NOT_FOUND"}
                or executed
                or not eligible
            ):
                raise ValueError(
                    "exact null transitions must be unexecuted terminal candidates"
                )
            if outcome_hashes["pre"] != outcome_hashes["post"]:
                raise ValueError(
                    "exact null transition must preserve public observation"
                )
            if factors["execution_succeeded"] is not None:
                raise ValueError("terminal candidates have no physical execution event")
            for name in (
                "task_progress_succeeded",
                "task_relevant_change",
                "target_revealed",
                "candidate_rejected",
                "region_confirmed_empty",
            ):
                if factors[name] is not False:
                    raise ValueError(f"exact null transition requires {name}=false")
        elif not executed:
            if eligible or primitive in {"STOP", "REPORT_NOT_FOUND"}:
                raise ValueError(
                    "unexecuted nonterminal labels must be context-ineligible"
                )
            if outcome_hashes["pre"] != outcome_hashes["post"]:
                raise ValueError(
                    "ineligible candidate must preserve the decision observation"
                )
            if any(value is not None for value in factors.values()):
                raise ValueError("ineligible candidate effects must all be null")
        elif not eligible:
            raise ValueError("an ineligible candidate cannot have an executed outcome")
        if teacher is not True:
            raise ValueError("effect labels must be evaluator-only supervision")
        sample_id = " ".join(str(value.get("sample_id", "")).split())
        group = " ".join(str(value.get("initial_state_group", "")).split())
        candidate_id = " ".join(str(value.get("candidate_id", "")).split())
        if not sample_id or not group or not candidate_id or not primitive:
            raise ValueError("effect sample/group/candidate fields are required")
        return cls(
            sample_id=sample_id,
            initial_state_group=group,
            split=Split(str(value.get("split", ""))),
            candidate_id=candidate_id,
            candidate_primitive=primitive,
            decision_observation_sha256=decision_digest,
            outcome_observation_sha256=outcome_hashes,
            selection_correct=selection_correct,
            eligible_for_execution=eligible,
            executed=executed,
            exact_null_transition=exact_null,
            factors=factors,
            simulator_teacher_only=True,
        )


def load_effect_labels(path: Path) -> list[EffectLabel]:
    labels = [
        EffectLabel.from_mapping(json.loads(line))
        for line in path.read_text().splitlines()
        if line
    ]
    if not labels:
        raise ValueError("effect label file is empty")
    keys = [(row.sample_id, row.candidate_id) for row in labels]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate sample/candidate effect label")
    group_splits: dict[str, Split] = {}
    for row in labels:
        previous = group_splits.setdefault(row.initial_state_group, row.split)
        if previous is not row.split:
            raise ValueError("action-effect initial-state group leakage")
    return labels


@dataclasses.dataclass(frozen=True)
class EffectInputs:
    """Label-free public arrays accepted by online effect inference."""

    sample_id: tuple[str, ...]
    initial_state_group: tuple[str, ...]
    split: tuple[Split, ...]
    belief_token: np.ndarray
    candidate_prompt_tokens: np.ndarray
    candidate_prompt_valid_mask: np.ndarray
    candidate_valid_mask: np.ndarray
    candidate_id: np.ndarray
    candidate_primitive: np.ndarray
    candidate_payload: np.ndarray
    candidate_action_id: np.ndarray


@dataclasses.dataclass(frozen=True)
class EffectArrays(EffectInputs):
    """Training/evaluation arrays after a separate evaluator-label join."""

    route_target: np.ndarray
    effect_target: np.ndarray
    effect_support_mask: np.ndarray
    executed_mask: np.ndarray


def build_effect_inputs(
    *,
    feature_arrays: Mapping[str, Any],
    binding_predictions: Mapping[str, Any],
    action_vocabulary: Sequence[str],
) -> EffectInputs:
    """Build an online effect batch without accepting evaluator labels."""

    forbidden = {
        "patch_target",
        "target_present",
        "task_sufficient",
        "task_sufficient_mask",
        "holding_requested_target",
        "holding_requested_target_mask",
        "region_confirmed_empty",
        "region_confirmed_empty_mask",
        "task_complete",
        "task_complete_mask",
        "route_target",
        "effect_target",
        "effect_support_mask",
        "selection_correct",
    }
    leaked = (forbidden & set(feature_arrays)) | (forbidden & set(binding_predictions))
    if leaked:
        raise ValueError(
            f"online effect inputs contain evaluator targets {sorted(leaked)}"
        )
    validate_feature_arrays(feature_arrays)
    required_candidates = {
        "candidate_prompt_tokens",
        "candidate_prompt_valid_mask",
        "candidate_valid_mask",
        "candidate_id",
        "candidate_primitive",
        "candidate_payload",
        "decision_observation_sha256",
    }
    if not required_candidates <= set(feature_arrays):
        raise ValueError("effect training requires retained candidate prompt tokens")
    sample_id = np.asarray(feature_arrays["sample_id"]).astype(str)
    group = np.asarray(feature_arrays["initial_state_group"]).astype(str)
    split = np.asarray(feature_arrays["split"]).astype(str)
    if not np.array_equal(
        sample_id, np.asarray(binding_predictions["sample_id"]).astype(str)
    ):
        raise ValueError("binding predictions and prefix samples differ")
    if not np.array_equal(
        group, np.asarray(binding_predictions["initial_state_group"]).astype(str)
    ) or not np.array_equal(
        split, np.asarray(binding_predictions["split"]).astype(str)
    ):
        raise ValueError("binding predictions and prefix group/split differ")
    belief = np.asarray(binding_predictions["target_token"], dtype=np.float32)
    if belief.ndim != 2 or belief.shape[0] != len(sample_id):
        raise ValueError("target belief token shape mismatch")
    candidate_tokens = np.asarray(
        feature_arrays["candidate_prompt_tokens"], dtype=np.float32
    )
    candidate_mask = np.asarray(
        feature_arrays["candidate_prompt_valid_mask"], dtype=bool
    )
    count, candidate_count, time_count, token_count, width = candidate_tokens.shape
    flattened_tokens = candidate_tokens.reshape(
        count, candidate_count, time_count * token_count, width
    )
    flattened_mask = candidate_mask.reshape(
        count, candidate_count, time_count * token_count
    )
    valid = np.asarray(feature_arrays["candidate_valid_mask"], dtype=bool)
    candidate_ids = np.asarray(feature_arrays["candidate_id"]).astype(str)
    primitives = np.char.upper(
        np.asarray(feature_arrays["candidate_primitive"]).astype(str)
    )
    payloads = np.asarray(feature_arrays["candidate_payload"]).astype(str)
    decision_digests = np.asarray(feature_arrays["decision_observation_sha256"]).astype(
        str
    )
    if decision_digests.shape != (count,) or any(
        len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        for digest in decision_digests.tolist()
    ):
        raise ValueError(
            "effect inputs require one decision-observation SHA-256 per sample"
        )
    vocabulary = tuple(str(value).upper() for value in action_vocabulary)
    if len(set(vocabulary)) != len(vocabulary):
        raise ValueError("effect action vocabulary has duplicates")
    action_to_id = {value: index for index, value in enumerate(vocabulary)}
    action_ids = np.zeros((count, candidate_count), dtype=np.int64)
    for row, column in zip(*np.nonzero(valid), strict=True):
        primitive = primitives[row, column]
        if primitive not in action_to_id:
            raise ValueError(f"unknown effect action {primitive!r}")
        action_ids[row, column] = action_to_id[primitive]
    return EffectInputs(
        sample_id=tuple(sample_id.tolist()),
        initial_state_group=tuple(group.tolist()),
        split=tuple(Split(value) for value in split),
        belief_token=belief,
        candidate_prompt_tokens=flattened_tokens,
        candidate_prompt_valid_mask=flattened_mask,
        candidate_valid_mask=valid,
        candidate_id=candidate_ids,
        candidate_primitive=primitives,
        candidate_payload=payloads,
        candidate_action_id=action_ids,
    )


def join_effect_features(
    *,
    feature_arrays: Mapping[str, Any],
    binding_predictions: Mapping[str, Any],
    labels: Sequence[EffectLabel],
    action_vocabulary: Sequence[str],
) -> EffectArrays:
    """Join label-free effect inputs to separately stored evaluator targets."""

    inputs = build_effect_inputs(
        feature_arrays=feature_arrays,
        binding_predictions=binding_predictions,
        action_vocabulary=action_vocabulary,
    )
    by_key = {(label.sample_id, label.candidate_id): label for label in labels}
    expected_keys = {
        (inputs.sample_id[row], inputs.candidate_id[row, column])
        for row, column in zip(*np.nonzero(inputs.candidate_valid_mask), strict=True)
    }
    if set(by_key) != expected_keys:
        raise ValueError("effect labels and public candidate matrix differ")
    count, candidate_count = inputs.candidate_valid_mask.shape
    targets = np.zeros((count, candidate_count, len(EFFECT_FACTORS)), dtype=np.float32)
    support = np.zeros_like(targets, dtype=bool)
    executed = np.zeros((count, candidate_count), dtype=bool)
    route_target = np.zeros((count, candidate_count), dtype=np.float32)
    for row, column in zip(*np.nonzero(inputs.candidate_valid_mask), strict=True):
        label = by_key[(inputs.sample_id[row], inputs.candidate_id[row, column])]
        if (
            label.initial_state_group != inputs.initial_state_group[row]
            or label.split is not inputs.split[row]
            or label.candidate_primitive != inputs.candidate_primitive[row, column]
        ):
            raise ValueError("effect label provenance differs from public candidate")
        decision_digest = str(feature_arrays["decision_observation_sha256"][row])
        if label.decision_observation_sha256 != decision_digest:
            raise ValueError(
                "effect label outcome does not begin at the feature decision state"
            )
        executed[row, column] = label.executed
        route_target[row, column] = float(label.selection_correct)
        for factor_index, factor_name in enumerate(EFFECT_FACTORS):
            value = label.factors[factor_name]
            if value is not None:
                targets[row, column, factor_index] = float(value)
                support[row, column, factor_index] = True
    if not np.all(route_target.sum(axis=1) == 1.0):
        raise ValueError("every effect sample requires exactly one correct route")
    return EffectArrays(
        **{
            field.name: getattr(inputs, field.name)
            for field in dataclasses.fields(EffectInputs)
        },
        route_target=route_target,
        effect_target=targets,
        effect_support_mask=support,
        executed_mask=executed,
    )


try:
    import torch
    from torch import nn
    from torch.nn import functional
except ModuleNotFoundError:  # pragma: no cover - optional learned dependency
    torch = None
    nn = None
    functional = None


if nn is not None:

    class CandidateConditionedEffectPredictor(nn.Module):
        def __init__(
            self,
            *,
            belief_width: int,
            vlm_width: int,
            model_width: int,
            num_heads: int,
            maximum_action_types: int,
            dropout: float = 0.0,
        ) -> None:
            super().__init__()
            if (
                min(
                    belief_width,
                    vlm_width,
                    model_width,
                    num_heads,
                    maximum_action_types,
                )
                < 1
            ):
                raise ValueError("effect model dimensions must be positive")
            if model_width % num_heads:
                raise ValueError("effect model width must be divisible by heads")
            self.belief_projection = nn.Linear(belief_width, model_width)
            self.candidate_norm = nn.LayerNorm(vlm_width, elementwise_affine=False)
            self.candidate_projection = nn.Linear(vlm_width, model_width)
            self.action_embedding = nn.Embedding(maximum_action_types, model_width)
            self.candidate_seed = nn.Parameter(torch.empty(1, 1, model_width))
            nn.init.normal_(self.candidate_seed, std=model_width**-0.5)
            self.candidate_readout = nn.MultiheadAttention(
                model_width, num_heads, dropout=dropout, batch_first=True
            )
            self.joint = nn.Sequential(
                nn.LayerNorm(2 * model_width),
                nn.Linear(2 * model_width, model_width),
                nn.GELU(),
            )
            self.factor_head = nn.Linear(model_width, len(EFFECT_FACTORS))
            self.route_head = nn.Linear(model_width + len(EFFECT_FACTORS), 1)

        def forward(
            self,
            belief_token: torch.Tensor,
            candidate_prompt_tokens: torch.Tensor,
            *,
            candidate_prompt_valid_mask: torch.Tensor,
            candidate_valid_mask: torch.Tensor,
            candidate_action_ids: torch.Tensor,
            effect_backprop_to_shared: bool = True,
            route_use_predicted_effects: bool = True,
        ) -> dict[str, torch.Tensor]:
            if belief_token.ndim != 2 or candidate_prompt_tokens.ndim != 4:
                raise ValueError("effect belief/candidate tensors have invalid rank")
            batch, candidates, tokens, _ = candidate_prompt_tokens.shape
            if belief_token.shape[0] != batch:
                raise ValueError("effect belief/candidate batches differ")
            if candidate_prompt_valid_mask.shape != (batch, candidates, tokens):
                raise ValueError("candidate prompt mask shape mismatch")
            if candidate_valid_mask.shape != (batch, candidates):
                raise ValueError("candidate valid mask shape mismatch")
            if candidate_action_ids.shape != (batch, candidates):
                raise ValueError("candidate action IDs shape mismatch")
            valid = candidate_valid_mask.bool()
            if not valid.any(dim=1).all():
                raise ValueError("every sample needs a valid candidate")
            if not candidate_prompt_valid_mask[valid].bool().any(dim=1).all():
                raise ValueError("every valid candidate needs a prompt token")
            if candidate_prompt_valid_mask[~valid].bool().any():
                raise ValueError("padded candidates cannot have prompt tokens")
            row_index, candidate_index = valid.nonzero(as_tuple=True)
            tokens_valid = self.candidate_projection(
                self.candidate_norm(candidate_prompt_tokens[row_index, candidate_index])
            )
            action = self.action_embedding(
                candidate_action_ids[row_index, candidate_index]
            )
            seed = self.candidate_seed.expand(len(row_index), -1, -1) + action[:, None]
            candidate_value, _ = self.candidate_readout(
                seed,
                tokens_valid,
                tokens_valid,
                key_padding_mask=~candidate_prompt_valid_mask[
                    row_index, candidate_index
                ].bool(),
                need_weights=False,
            )
            belief = self.belief_projection(belief_token[row_index])
            joint_valid = self.joint(
                torch.cat((belief, candidate_value.squeeze(1)), dim=-1)
            )
            factor_source = (
                joint_valid if effect_backprop_to_shared else joint_valid.detach()
            )
            factor_valid = self.factor_head(factor_source)
            # The route is learned from the interaction representation and the
            # predicted factor distribution. This is not a hand-written
            # utility: route loss learns every downstream coefficient.
            route_factors = (
                torch.sigmoid(factor_valid)
                if route_use_predicted_effects
                else torch.zeros_like(factor_valid).detach()
            )
            route_valid = self.route_head(
                torch.cat((joint_valid, route_factors), dim=-1)
            ).squeeze(-1)
            factor_logits = factor_valid.new_zeros(
                batch, candidates, len(EFFECT_FACTORS)
            )
            factor_logits[row_index, candidate_index] = factor_valid
            route_logits = route_valid.new_full((batch, candidates), float("-inf"))
            route_logits[row_index, candidate_index] = route_valid
            return {"factor_logits": factor_logits, "route_logits": route_logits}

    def effect_objectives(
        outputs: Mapping[str, torch.Tensor],
        *,
        route_target_distribution: torch.Tensor,
        factor_target: torch.Tensor,
        factor_support_mask: torch.Tensor,
        candidate_valid_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        factor_logits = outputs["factor_logits"]
        route_logits = outputs["route_logits"]
        if (
            factor_logits.shape != factor_target.shape
            or factor_logits.shape != factor_support_mask.shape
        ):
            raise ValueError("effect factor tensor shapes differ")
        if factor_logits.shape[:2] != candidate_valid_mask.shape:
            raise ValueError("candidate valid mask shape mismatch")
        if route_logits.shape != candidate_valid_mask.shape:
            raise ValueError("route logits and candidate mask differ")
        if route_target_distribution.shape != route_logits.shape:
            raise ValueError("route target shape mismatch")
        route_target = route_target_distribution.masked_fill(
            ~candidate_valid_mask.bool(), 0.0
        )
        denominator = route_target.sum(dim=1, keepdim=True)
        if not torch.allclose(denominator, torch.ones_like(denominator)):
            raise ValueError("route target must select exactly one valid candidate")
        route_log_probabilities = functional.log_softmax(route_logits, dim=1)
        route_log_probabilities = route_log_probabilities.masked_fill(
            ~candidate_valid_mask.bool(), 0.0
        )
        losses = {
            "route_selection": -(route_target * route_log_probabilities)
            .sum(dim=1)
            .mean()
        }
        for factor_index, factor_name in enumerate(EFFECT_FACTORS):
            supported = (
                factor_support_mask[:, :, factor_index].bool()
                & candidate_valid_mask.bool()
            )
            losses[factor_name] = (
                functional.binary_cross_entropy_with_logits(
                    factor_logits[:, :, factor_index][supported],
                    factor_target[:, :, factor_index][supported].float(),
                )
                if supported.any()
                else factor_logits.sum() * 0.0
            )
        return losses

    class LearnedEffectObjective(nn.Module):
        objective_names = ("route_selection", *EFFECT_FACTORS)

        def __init__(self) -> None:
            super().__init__()
            self.log_variances = nn.Parameter(torch.zeros(len(self.objective_names)))

        def forward(
            self,
            objectives: Mapping[str, torch.Tensor],
            *,
            supported: Mapping[str, bool],
        ) -> dict[str, torch.Tensor]:
            if set(objectives) != set(self.objective_names) or set(supported) != set(
                self.objective_names
            ):
                raise ValueError("effect objective family mismatch")
            terms = []
            for index, name in enumerate(self.objective_names):
                if supported[name]:
                    terms.append(
                        torch.exp(-self.log_variances[index]) * objectives[name]
                        + self.log_variances[index]
                    )
            if not terms:
                raise ValueError("at least one effect factor must be labeled")
            return {
                "loss": torch.stack(terms).sum(),
                "learned_log_variances": self.log_variances,
            }

else:

    class CandidateConditionedEffectPredictor:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any) -> None:
            raise RuntimeError(
                "effect predictor requires optional learned dependencies"
            )

    def effect_objectives(*_: Any, **__: Any) -> Any:
        raise RuntimeError("effect objectives require optional learned dependencies")

    class LearnedEffectObjective:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any) -> None:
            raise RuntimeError(
                "effect objective requires optional learned dependencies"
            )
