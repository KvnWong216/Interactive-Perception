"""Evaluation utilities for Method-V1 predictions and closed-loop episodes.

The module distinguishes three quantities that are often conflated:

* probability quality on actually executed branches (NLL/Brier/reliability),
* an *offline branch-matrix estimate* obtained by selecting among outcomes
  collected from the same reset, and
* final task success from newly executed closed-loop policy episodes.

Only the third is deployment evidence.  The CLI never fabricates missing
rollouts and every report carries an explicit evidence scope.  Its ``select``
command is narrower still: it emits an identity-bound pre-execution decision
from one frozen manifest/cache/checkpoint and does not claim a rollout result.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import random
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Any

from .contracts import Primitive, canonical_json_bytes, canonical_sha256

_SHA256_HEX = frozenset("0123456789abcdef")


def _clean_text(value: Any, *, name: str) -> str:
    result = " ".join(str(value or "").split())
    if not result:
        raise ValueError(f"{name} must be non-empty")
    return result


def _sha256(value: Any, *, name: str) -> str:
    result = str(value)
    if len(result) != 64 or any(character not in _SHA256_HEX for character in result):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return result


def _probability(value: Any, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return result


def _file_sha256(path: str | Path) -> str:
    source = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_mapping(value: Any, *, keys: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    observed = set(value)
    if observed != keys:
        raise ValueError(
            f"{name} keys mismatch; missing={sorted(keys - observed)}, "
            f"extra={sorted(observed - keys)}"
        )
    return value


@dataclasses.dataclass(frozen=True)
class ReliabilityBin:
    lower: float
    upper: float
    count: int
    mean_probability: float | None
    empirical_success: float | None
    absolute_gap: float | None


@dataclasses.dataclass(frozen=True)
class ProbabilityMetrics:
    """Calibration metrics over receipt-backed executed outcomes."""

    count: int
    nll: float
    brier: float
    expected_calibration_error: float
    bins: tuple[ReliabilityBin, ...]


def probability_metrics(
    probabilities: Sequence[float],
    outcomes: Sequence[bool | int],
    *,
    num_bins: int = 10,
    epsilon: float = 1e-12,
) -> ProbabilityMetrics:
    """Compute binary NLL, Brier, and equal-width reliability bins."""

    values = tuple(_probability(value, name="probability") for value in probabilities)
    labels = tuple(int(value) for value in outcomes)
    if not values or len(values) != len(labels):
        raise ValueError("probabilities and outcomes must be non-empty and aligned")
    if any(label not in {0, 1} for label in labels):
        raise ValueError("outcomes must be binary")
    if not isinstance(num_bins, int) or isinstance(num_bins, bool) or num_bins < 1:
        raise ValueError("num_bins must be a positive integer")
    if not math.isfinite(epsilon) or not 0 < epsilon < 0.5:
        raise ValueError("epsilon must be finite and in (0, 0.5)")

    nll_terms: list[float] = []
    brier_terms: list[float] = []
    assignments: list[list[int]] = [[] for _ in range(num_bins)]
    for index, (probability, label) in enumerate(zip(values, labels, strict=True)):
        clipped = min(max(probability, epsilon), 1.0 - epsilon)
        nll_terms.append(
            -(label * math.log(clipped) + (1 - label) * math.log(1.0 - clipped))
        )
        brier_terms.append((probability - label) ** 2)
        bin_index = min(int(probability * num_bins), num_bins - 1)
        assignments[bin_index].append(index)

    bins: list[ReliabilityBin] = []
    weighted_gap = 0.0
    for bin_index, indices in enumerate(assignments):
        lower = bin_index / num_bins
        upper = (bin_index + 1) / num_bins
        if not indices:
            bins.append(
                ReliabilityBin(
                    lower=lower,
                    upper=upper,
                    count=0,
                    mean_probability=None,
                    empirical_success=None,
                    absolute_gap=None,
                )
            )
            continue
        mean_probability = sum(values[index] for index in indices) / len(indices)
        empirical_success = sum(labels[index] for index in indices) / len(indices)
        gap = abs(mean_probability - empirical_success)
        weighted_gap += len(indices) * gap / len(values)
        bins.append(
            ReliabilityBin(
                lower=lower,
                upper=upper,
                count=len(indices),
                mean_probability=mean_probability,
                empirical_success=empirical_success,
                absolute_gap=gap,
            )
        )
    return ProbabilityMetrics(
        count=len(values),
        nll=sum(nll_terms) / len(nll_terms),
        brier=sum(brier_terms) / len(brier_terms),
        expected_calibration_error=weighted_gap,
        bins=tuple(bins),
    )


@dataclasses.dataclass(frozen=True)
class BootstrapInterval:
    estimate: float
    lower: float
    upper: float
    confidence_level: float
    bootstrap_samples: int
    group_count: int


def group_bootstrap_interval(
    values_by_group: Mapping[str, float],
    *,
    seed: int = 0,
    bootstrap_samples: int = 10_000,
    confidence_level: float = 0.95,
    statistic: Callable[[Sequence[float]], float] | None = None,
) -> BootstrapInterval:
    """Percentile bootstrap that resamples reset groups, never frames/branches."""

    groups = tuple(sorted(values_by_group))
    if not groups:
        raise ValueError("group bootstrap requires at least one group")
    values = tuple(float(values_by_group[group]) for group in groups)
    if any(not math.isfinite(value) for value in values):
        raise ValueError("bootstrap values must be finite")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0, 1)")
    if statistic is None:

        def statistic(sample: Sequence[float]) -> float:
            return sum(sample) / len(sample)

    estimate = float(statistic(values))
    rng = random.Random(seed)
    replicates = sorted(
        float(statistic(tuple(rng.choice(values) for _ in groups)))
        for _ in range(bootstrap_samples)
    )
    tail = (1.0 - confidence_level) / 2.0

    def percentile(probability: float) -> float:
        if len(replicates) == 1:
            return replicates[0]
        position = probability * (len(replicates) - 1)
        lower_index = math.floor(position)
        upper_index = math.ceil(position)
        fraction = position - lower_index
        return (
            replicates[lower_index] * (1.0 - fraction)
            + replicates[upper_index] * fraction
        )

    return BootstrapInterval(
        estimate=estimate,
        lower=percentile(tail),
        upper=percentile(1.0 - tail),
        confidence_level=confidence_level,
        bootstrap_samples=bootstrap_samples,
        group_count=len(groups),
    )


def split_group_bootstrap_interval(
    values_by_decision_group: Mapping[str, float],
    split_group_by_decision_group: Mapping[str, str],
    *,
    seed: int = 0,
    bootstrap_samples: int = 10_000,
    confidence_level: float = 0.95,
) -> BootstrapInterval:
    """Cluster-bootstrap split groups while estimating a decision-group mean.

    The point estimand gives every decision group equal weight.  Bootstrap draws
    operate on the broader, frozen ``split_group_id``: when one split group is
    sampled, all decision groups belonging to it move together.  This preserves
    dependence between prompt/hidden variants from one scene-layout family.
    """

    decision_ids = tuple(sorted(values_by_decision_group))
    if not decision_ids:
        raise ValueError("split-group bootstrap requires at least one decision group")
    if set(split_group_by_decision_group) != set(decision_ids):
        raise ValueError(
            "every decision group requires exactly one split-group identity"
        )
    values = {
        decision_id: float(values_by_decision_group[decision_id])
        for decision_id in decision_ids
    }
    if any(not math.isfinite(value) for value in values.values()):
        raise ValueError("bootstrap values must be finite")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0, 1)")

    decisions_by_split: dict[str, list[str]] = defaultdict(list)
    for decision_id in decision_ids:
        split_group_id = _clean_text(
            split_group_by_decision_group[decision_id], name="split_group_id"
        )
        decisions_by_split[split_group_id].append(decision_id)
    split_groups = tuple(sorted(decisions_by_split))
    estimate = sum(values.values()) / len(values)
    rng = random.Random(seed)
    replicates: list[float] = []
    for _ in range(bootstrap_samples):
        sampled_decisions: list[str] = []
        for _ in split_groups:
            sampled_split = rng.choice(split_groups)
            sampled_decisions.extend(decisions_by_split[sampled_split])
        replicates.append(
            sum(values[decision_id] for decision_id in sampled_decisions)
            / len(sampled_decisions)
        )
    replicates.sort()
    tail = (1.0 - confidence_level) / 2.0

    def percentile(probability: float) -> float:
        if len(replicates) == 1:
            return replicates[0]
        position = probability * (len(replicates) - 1)
        lower_index = math.floor(position)
        upper_index = math.ceil(position)
        fraction = position - lower_index
        return (
            replicates[lower_index] * (1.0 - fraction)
            + replicates[upper_index] * fraction
        )

    return BootstrapInterval(
        estimate=estimate,
        lower=percentile(tail),
        upper=percentile(1.0 - tail),
        confidence_level=confidence_level,
        bootstrap_samples=bootstrap_samples,
        group_count=len(split_groups),
    )


@dataclasses.dataclass(frozen=True)
class BranchPrediction:
    """One executed branch with predictions available before that execution."""

    record_id: str
    decision_group_id: str
    split_group_id: str
    candidate_id: str
    candidate_fingerprint: str
    primitive: Primitive
    repeat_index: int
    observed_outcome: bool
    scorer_probability: float | None
    ensemble_probability: float | None = None
    frozen_vlm_rank: int | None = None
    feasible: bool = True

    def __post_init__(self) -> None:
        for name in (
            "record_id",
            "decision_group_id",
            "split_group_id",
            "candidate_id",
        ):
            object.__setattr__(self, name, _clean_text(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "candidate_fingerprint",
            _sha256(self.candidate_fingerprint, name="candidate_fingerprint"),
        )
        if not isinstance(self.primitive, Primitive):
            raise TypeError("primitive must be a Primitive")
        if (
            not isinstance(self.repeat_index, int)
            or isinstance(self.repeat_index, bool)
            or self.repeat_index < 0
        ):
            raise ValueError("repeat_index must be a non-negative integer")
        if not isinstance(self.observed_outcome, bool):
            raise TypeError("observed_outcome must be a bool")
        for name in ("scorer_probability", "ensemble_probability"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _probability(value, name=name))
        if self.frozen_vlm_rank is not None and (
            not isinstance(self.frozen_vlm_rank, int)
            or isinstance(self.frozen_vlm_rank, bool)
            or self.frozen_vlm_rank < 0
        ):
            raise ValueError("frozen_vlm_rank must be a non-negative integer")
        if not isinstance(self.feasible, bool):
            raise TypeError("feasible must be a bool")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> BranchPrediction:
        required = {
            "record_id",
            "decision_group_id",
            "split_group_id",
            "candidate_id",
            "candidate_fingerprint",
            "primitive",
            "repeat_index",
            "observed_outcome",
            "scorer_probability",
        }
        optional = {"ensemble_probability", "frozen_vlm_rank", "feasible"}
        if not required.issubset(value) or set(value) - required - optional:
            raise ValueError("branch-prediction keys do not match the public schema")
        return cls(
            record_id=str(value["record_id"]),
            decision_group_id=str(value["decision_group_id"]),
            split_group_id=str(value["split_group_id"]),
            candidate_id=str(value["candidate_id"]),
            candidate_fingerprint=str(value["candidate_fingerprint"]),
            primitive=Primitive(str(value["primitive"])),
            repeat_index=int(value["repeat_index"]),
            observed_outcome=value["observed_outcome"],
            scorer_probability=(
                None
                if value["scorer_probability"] is None
                else float(value["scorer_probability"])
            ),
            ensemble_probability=(
                None
                if value.get("ensemble_probability") is None
                else float(value["ensemble_probability"])
            ),
            frozen_vlm_rank=(
                None
                if value.get("frozen_vlm_rank") is None
                else int(value["frozen_vlm_rank"])
            ),
            feasible=bool(value.get("feasible", True)),
        )


@dataclasses.dataclass(frozen=True)
class CheckpointEvidence:
    """Byte and semantic identity of one scorer used for prediction."""

    checkpoint_sha256: str
    identity_sha256: str
    seed: int
    experiment_id: str
    configuration_sha256: str
    proposal_provider_id: str
    provider_id: str
    outcome_contract_sha256: str
    training_dataset_sha256: str
    training_admission_evidence_sha256: str
    collection_plan_sha256: str
    scorer_verifier_auth_key_id: str
    training_split_group_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "checkpoint_sha256",
            "identity_sha256",
            "configuration_sha256",
            "outcome_contract_sha256",
            "training_dataset_sha256",
            "training_admission_evidence_sha256",
            "collection_plan_sha256",
            "scorer_verifier_auth_key_id",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name=name))
        object.__setattr__(
            self, "provider_id", _clean_text(self.provider_id, name="provider_id")
        )
        for name in ("experiment_id", "proposal_provider_id"):
            object.__setattr__(self, name, _clean_text(getattr(self, name), name=name))
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TypeError("checkpoint seed must be an integer")
        groups = tuple(
            _clean_text(value, name="training split group")
            for value in self.training_split_group_ids
        )
        if not groups or len(set(groups)) != len(groups):
            raise ValueError(
                "checkpoint training split groups must be non-empty and unique"
            )
        object.__setattr__(self, "training_split_group_ids", groups)

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)

    @classmethod
    def from_mapping(cls, value: Any) -> CheckpointEvidence:
        mapping = _strict_mapping(
            value,
            keys={
                "checkpoint_sha256",
                "identity_sha256",
                "seed",
                "experiment_id",
                "configuration_sha256",
                "proposal_provider_id",
                "provider_id",
                "outcome_contract_sha256",
                "training_dataset_sha256",
                "training_admission_evidence_sha256",
                "collection_plan_sha256",
                "scorer_verifier_auth_key_id",
                "training_split_group_ids",
            },
            name="checkpoint evidence",
        )
        raw_groups = mapping["training_split_group_ids"]
        if not isinstance(raw_groups, list):
            raise TypeError("checkpoint training_split_group_ids must be a list")
        return cls(
            checkpoint_sha256=str(mapping["checkpoint_sha256"]),
            identity_sha256=str(mapping["identity_sha256"]),
            seed=mapping["seed"],
            experiment_id=str(mapping["experiment_id"]),
            configuration_sha256=str(mapping["configuration_sha256"]),
            proposal_provider_id=str(mapping["proposal_provider_id"]),
            provider_id=str(mapping["provider_id"]),
            outcome_contract_sha256=str(mapping["outcome_contract_sha256"]),
            training_dataset_sha256=str(mapping["training_dataset_sha256"]),
            training_admission_evidence_sha256=str(
                mapping["training_admission_evidence_sha256"]
            ),
            collection_plan_sha256=str(mapping["collection_plan_sha256"]),
            scorer_verifier_auth_key_id=str(mapping["scorer_verifier_auth_key_id"]),
            training_split_group_ids=tuple(raw_groups),
        )


@dataclasses.dataclass(frozen=True)
class CanonicalBranchPrediction:
    """One receipt-backed branch plus every frozen scorer probability."""

    record_id: str
    branch_sha256: str
    execution_receipt_id: str
    manifest_sha256: str
    token_cache_sha256: str
    outcome_contract_sha256: str
    decision_group_id: str
    initial_state_group: str
    split_group_id: str
    split: str
    candidate_id: str
    candidate_fingerprint: str
    primitive: Primitive
    repeat_index: int
    observed_outcome: bool
    frozen_vlm_rank: int
    checkpoint_probabilities: tuple[float, ...]

    def __post_init__(self) -> None:
        for name in (
            "record_id",
            "execution_receipt_id",
            "decision_group_id",
            "initial_state_group",
            "split_group_id",
            "split",
            "candidate_id",
        ):
            object.__setattr__(self, name, _clean_text(getattr(self, name), name=name))
        for name in (
            "branch_sha256",
            "manifest_sha256",
            "token_cache_sha256",
            "outcome_contract_sha256",
            "candidate_fingerprint",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name=name))
        if self.split not in {"train", "validation", "calibration", "test"}:
            raise ValueError("canonical prediction has an unsupported split")
        if not isinstance(self.primitive, Primitive):
            raise TypeError("canonical prediction primitive must be Primitive")
        for name in ("repeat_index", "frozen_vlm_rank"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not isinstance(self.observed_outcome, bool):
            raise TypeError("observed_outcome must be bool")
        probabilities = tuple(
            _probability(value, name="checkpoint_probability")
            for value in self.checkpoint_probabilities
        )
        if not probabilities:
            raise ValueError("canonical prediction requires at least one scorer output")
        object.__setattr__(self, "checkpoint_probabilities", probabilities)

    def to_dict(self) -> dict[str, object]:
        value = dataclasses.asdict(self)
        value["primitive"] = self.primitive.value
        value["checkpoint_probabilities"] = list(self.checkpoint_probabilities)
        return value

    @classmethod
    def from_mapping(cls, value: Any) -> CanonicalBranchPrediction:
        mapping = _strict_mapping(
            value,
            keys={
                "record_id",
                "branch_sha256",
                "execution_receipt_id",
                "manifest_sha256",
                "token_cache_sha256",
                "outcome_contract_sha256",
                "decision_group_id",
                "initial_state_group",
                "split_group_id",
                "split",
                "candidate_id",
                "candidate_fingerprint",
                "primitive",
                "repeat_index",
                "observed_outcome",
                "frozen_vlm_rank",
                "checkpoint_probabilities",
            },
            name="canonical branch prediction",
        )
        probabilities = mapping["checkpoint_probabilities"]
        if not isinstance(probabilities, Sequence) or isinstance(probabilities, str):
            raise TypeError("checkpoint_probabilities must be a sequence")
        return cls(
            record_id=str(mapping["record_id"]),
            branch_sha256=str(mapping["branch_sha256"]),
            execution_receipt_id=str(mapping["execution_receipt_id"]),
            manifest_sha256=str(mapping["manifest_sha256"]),
            token_cache_sha256=str(mapping["token_cache_sha256"]),
            outcome_contract_sha256=str(mapping["outcome_contract_sha256"]),
            decision_group_id=str(mapping["decision_group_id"]),
            initial_state_group=str(mapping["initial_state_group"]),
            split_group_id=str(mapping["split_group_id"]),
            split=str(mapping["split"]),
            candidate_id=str(mapping["candidate_id"]),
            candidate_fingerprint=str(mapping["candidate_fingerprint"]),
            primitive=Primitive(str(mapping["primitive"])),
            repeat_index=mapping["repeat_index"],
            observed_outcome=mapping["observed_outcome"],
            frozen_vlm_rank=mapping["frozen_vlm_rank"],
            checkpoint_probabilities=tuple(probabilities),
        )


@dataclasses.dataclass(frozen=True)
class CanonicalPredictionArtifact:
    """Hash-bound predictions generated from real caches and checkpoints."""

    dataset_id: str
    dataset_sha256: str
    dataset_admission_evidence_sha256: str
    collection_plan_sha256: str
    scorer_verifier_auth_key_id: str
    outcome_contract_sha256: str
    temperature: float
    checkpoints: tuple[CheckpointEvidence, ...]
    records: tuple[CanonicalBranchPrediction, ...]
    schema_version: str = "method-v1-canonical-predictions-v3"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "dataset_id", _clean_text(self.dataset_id, name="dataset_id")
        )
        for name in (
            "dataset_sha256",
            "dataset_admission_evidence_sha256",
            "collection_plan_sha256",
            "scorer_verifier_auth_key_id",
            "outcome_contract_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name=name))
        if self.schema_version != "method-v1-canonical-predictions-v3":
            raise ValueError("unsupported canonical prediction schema")
        temperature = float(self.temperature)
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("prediction temperature must be positive and finite")
        object.__setattr__(self, "temperature", temperature)
        checkpoints = tuple(self.checkpoints)
        records = tuple(self.records)
        if not checkpoints or any(
            not isinstance(item, CheckpointEvidence) for item in checkpoints
        ):
            raise ValueError("canonical predictions require checkpoint evidence")
        if len({item.checkpoint_sha256 for item in checkpoints}) != len(checkpoints):
            raise ValueError("checkpoint files must be unique")
        if len({item.seed for item in checkpoints}) != len(checkpoints):
            raise ValueError("checkpoint seeds must be unique")
        if not records or any(
            not isinstance(item, CanonicalBranchPrediction) for item in records
        ):
            raise ValueError("canonical predictions require branch records")
        if len({item.record_id for item in records}) != len(records):
            raise ValueError("canonical prediction record IDs must be unique")
        if any(
            len(item.checkpoint_probabilities) != len(checkpoints) for item in records
        ):
            raise ValueError("every record must contain one probability per checkpoint")
        object.__setattr__(self, "checkpoints", checkpoints)
        object.__setattr__(self, "records", records)

    @property
    def records_sha256(self) -> str:
        return canonical_sha256([item.to_dict() for item in self.records])

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "dataset_sha256": self.dataset_sha256,
            "dataset_admission_evidence_sha256": (
                self.dataset_admission_evidence_sha256
            ),
            "collection_plan_sha256": self.collection_plan_sha256,
            "scorer_verifier_auth_key_id": self.scorer_verifier_auth_key_id,
            "outcome_contract_sha256": self.outcome_contract_sha256,
            "temperature": self.temperature,
            "checkpoints": [item.to_dict() for item in self.checkpoints],
            "records": [item.to_dict() for item in self.records],
            "records_sha256": self.records_sha256,
        }

    @property
    def artifact_sha256(self) -> str:
        return canonical_sha256(self._payload())

    def to_dict(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

    @classmethod
    def from_mapping(cls, value: Any) -> CanonicalPredictionArtifact:
        mapping = _strict_mapping(
            value,
            keys={
                "schema_version",
                "dataset_id",
                "dataset_sha256",
                "dataset_admission_evidence_sha256",
                "collection_plan_sha256",
                "scorer_verifier_auth_key_id",
                "outcome_contract_sha256",
                "temperature",
                "checkpoints",
                "records",
                "records_sha256",
                "artifact_sha256",
            },
            name="canonical prediction artifact",
        )
        raw_checkpoints = mapping["checkpoints"]
        raw_records = mapping["records"]
        if not isinstance(raw_checkpoints, list) or not isinstance(raw_records, list):
            raise TypeError("canonical checkpoint and record collections must be lists")
        result = cls(
            schema_version=str(mapping["schema_version"]),
            dataset_id=str(mapping["dataset_id"]),
            dataset_sha256=str(mapping["dataset_sha256"]),
            dataset_admission_evidence_sha256=str(
                mapping["dataset_admission_evidence_sha256"]
            ),
            collection_plan_sha256=str(mapping["collection_plan_sha256"]),
            scorer_verifier_auth_key_id=str(mapping["scorer_verifier_auth_key_id"]),
            outcome_contract_sha256=str(mapping["outcome_contract_sha256"]),
            temperature=float(mapping["temperature"]),
            checkpoints=tuple(
                CheckpointEvidence.from_mapping(item) for item in raw_checkpoints
            ),
            records=tuple(
                CanonicalBranchPrediction.from_mapping(item) for item in raw_records
            ),
        )
        if mapping["records_sha256"] != result.records_sha256:
            raise ValueError("canonical prediction record digest mismatch")
        if mapping["artifact_sha256"] != result.artifact_sha256:
            raise ValueError("canonical prediction artifact digest mismatch")
        return result


def _checkpoint_evidence(
    checkpoint_path: str | Path, identity: Mapping[str, object]
) -> CheckpointEvidence:
    required = {
        "seed",
        "experiment_id",
        "configuration_sha256",
        "proposal_provider_id",
        "provider_id",
        "outcome_contract_sha256",
        "training_dataset_sha256",
        "training_admission_evidence_sha256",
        "collection_plan_sha256",
        "scorer_verifier_auth_key_id",
        "training_split_group_ids",
    }
    if not required.issubset(identity):
        raise ValueError("checkpoint identity lacks canonical Method-V1 fields")
    raw_groups = identity["training_split_group_ids"]
    if not isinstance(raw_groups, list):
        raise TypeError("checkpoint training_split_group_ids must be a list")
    return CheckpointEvidence(
        checkpoint_sha256=_file_sha256(checkpoint_path),
        identity_sha256=canonical_sha256(identity),
        seed=identity["seed"],
        experiment_id=str(identity["experiment_id"]),
        configuration_sha256=str(identity["configuration_sha256"]),
        proposal_provider_id=str(identity["proposal_provider_id"]),
        provider_id=str(identity["provider_id"]),
        outcome_contract_sha256=str(identity["outcome_contract_sha256"]),
        training_dataset_sha256=str(identity["training_dataset_sha256"]),
        training_admission_evidence_sha256=str(
            identity["training_admission_evidence_sha256"]
        ),
        collection_plan_sha256=str(identity["collection_plan_sha256"]),
        scorer_verifier_auth_key_id=str(identity["scorer_verifier_auth_key_id"]),
        training_split_group_ids=tuple(raw_groups),
    )


def _generate_canonical_prediction_artifact(
    dataset: object,
    *,
    cache_roots: Sequence[str | Path],
    checkpoint_paths: Sequence[str | Path],
    device: str = "cpu",
    temperature: float = 1.0,
) -> CanonicalPredictionArtifact:
    """Run every real checkpoint once per frozen decision candidate set."""

    from .train_outcomes import (
        _candidate_field_to_device,
        load_checkpoint_model,
        load_qwen_token_fields_from_roots,
        method_v1_training_dataset_identity,
        require_receipt_backed_dataset,
    )

    admitted_dataset = dataset
    dataset = require_receipt_backed_dataset(admitted_dataset)
    paths = tuple(Path(path).expanduser().resolve() for path in checkpoint_paths)
    if not paths:
        raise ValueError("at least one checkpoint is required")
    if len(set(paths)) != len(paths):
        raise ValueError("checkpoint paths must be unique")
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature != 1.0:
        raise ValueError(
            "formal Method-V1 prediction currently admits raw temperature=1.0 only; "
            "a calibrated artifact requires a separately frozen calibration contract"
        )
    try:
        import torch
    except ImportError as error:  # pragma: no cover - environment dependent.
        raise RuntimeError("canonical prediction requires PyTorch") from error

    models = []
    checkpoint_rows: list[CheckpointEvidence] = []
    for path in paths:
        model, identity = load_checkpoint_model(path, device=device)
        models.append(model)
        checkpoint_rows.append(_checkpoint_evidence(path, identity))
    if len({item.provider_id for item in checkpoint_rows}) != 1:
        raise ValueError("canonical prediction cannot mix feature providers")
    if len({item.outcome_contract_sha256 for item in checkpoint_rows}) != 1:
        raise ValueError("canonical prediction cannot mix outcome contracts")
    if len({item.training_dataset_sha256 for item in checkpoint_rows}) != 1:
        raise ValueError("canonical prediction cannot mix training datasets")
    if len({item.training_admission_evidence_sha256 for item in checkpoint_rows}) != 1:
        raise ValueError("canonical prediction cannot mix training admissions")
    if len({item.collection_plan_sha256 for item in checkpoint_rows}) != 1:
        raise ValueError("canonical prediction cannot mix collection plans")
    if len({item.scorer_verifier_auth_key_id for item in checkpoint_rows}) != 1:
        raise ValueError("canonical prediction cannot mix scorer verifier trust roots")
    if len({item.training_split_group_ids for item in checkpoint_rows}) != 1:
        raise ValueError("canonical prediction cannot mix training split groups")

    manifest_by_group = {
        manifest.decision_group_id: manifest for manifest in dataset.manifests
    }
    contract_ids = {
        manifest.outcome_contract.fingerprint() for manifest in dataset.manifests
    }
    provider_ids = {manifest.token_provider_id for manifest in dataset.manifests}
    experiment_ids = {manifest.experiment_id for manifest in dataset.manifests}
    proposal_provider_ids = {
        manifest.proposal_provider_id for manifest in dataset.manifests
    }
    configurations = {manifest.configuration_sha256 for manifest in dataset.manifests}
    if any(
        len(values) != 1
        for values in (
            contract_ids,
            provider_ids,
            experiment_ids,
            proposal_provider_ids,
            configurations,
        )
    ):
        raise ValueError(
            "prediction dataset mixes contracts, providers, experiments, or "
            "resolved configurations"
        )
    contract_sha256 = next(iter(contract_ids))
    provider_id = next(iter(provider_ids))
    if any(item.provider_id != provider_id for item in checkpoint_rows):
        raise ValueError("checkpoint provider identity differs from prediction dataset")
    if any(item.outcome_contract_sha256 != contract_sha256 for item in checkpoint_rows):
        raise ValueError("checkpoint outcome contract differs from prediction dataset")
    expected_identity = {
        "experiment_id": next(iter(experiment_ids)),
        "proposal_provider_id": next(iter(proposal_provider_ids)),
        "configuration_sha256": next(iter(configurations)),
    }
    for item in checkpoint_rows:
        for name, expected in expected_identity.items():
            if getattr(item, name) != expected:
                raise ValueError(f"checkpoint {name} differs from prediction dataset")
    expected_training_dataset_sha256 = method_v1_training_dataset_identity(
        admitted_dataset, cache_roots=cache_roots
    )
    if any(
        item.training_dataset_sha256 != expected_training_dataset_sha256
        for item in checkpoint_rows
    ):
        raise ValueError(
            "checkpoint training dataset differs from the train/validation partitions "
            "inside the evaluation dataset"
        )
    if any(
        item.training_admission_evidence_sha256
        != admitted_dataset.evidence_for_splits(("train", "validation"))
        for item in checkpoint_rows
    ):
        raise ValueError(
            "checkpoint admission evidence differs from the receipt-backed dataset"
        )
    if any(
        item.collection_plan_sha256 != admitted_dataset.collection_plan_sha256
        for item in checkpoint_rows
    ):
        raise ValueError("checkpoint collection plan differs from evaluation dataset")
    if any(
        item.scorer_verifier_auth_key_id != admitted_dataset.scorer_verifier_auth_key_id
        for item in checkpoint_rows
    ):
        raise ValueError(
            "checkpoint scorer-verifier trust root differs from evaluation dataset"
        )

    fields = load_qwen_token_fields_from_roots(
        dataset.manifests, cache_roots=cache_roots
    )
    probabilities_by_group: dict[str, tuple[tuple[float, ...], ...]] = {}
    for manifest in dataset.manifests:
        field = fields[manifest.token_cache_sha256]
        moved = _candidate_field_to_device(field, device)
        model_probabilities: list[tuple[float, ...]] = []
        for model in models:
            with torch.no_grad():
                output = model(moved)
                probabilities = (
                    torch.sigmoid(output.task_success_logits / temperature)
                    .detach()
                    .cpu()[0]
                )
            expected_fingerprints = tuple(
                candidate.fingerprint() for candidate in manifest.candidates
            )
            if output.context_fingerprints != (manifest.context.fingerprint(),):
                raise ValueError("checkpoint prediction changed context identity")
            if output.candidate_fingerprints != (expected_fingerprints,):
                raise ValueError("checkpoint prediction changed candidate identities")
            model_probabilities.append(
                tuple(
                    float(probabilities[index])
                    for index in range(len(manifest.candidates))
                )
            )
        probabilities_by_group[manifest.decision_group_id] = tuple(
            tuple(model_row[candidate_index] for model_row in model_probabilities)
            for candidate_index in range(len(manifest.candidates))
        )

    records: list[CanonicalBranchPrediction] = []
    for branch in sorted(dataset.observed_branches, key=lambda item: item.branch_id):
        manifest = manifest_by_group[branch.decision_group_id]
        candidate_fingerprints = tuple(
            candidate.fingerprint() for candidate in manifest.candidates
        )
        candidate_index = candidate_fingerprints.index(branch.candidate_fingerprint)
        records.append(
            CanonicalBranchPrediction(
                record_id=branch.branch_id,
                branch_sha256=branch.public_fingerprint(),
                execution_receipt_id=branch.execution_receipt_id,
                manifest_sha256=manifest.fingerprint(),
                token_cache_sha256=manifest.token_cache_sha256,
                outcome_contract_sha256=manifest.outcome_contract.fingerprint(),
                decision_group_id=manifest.decision_group_id,
                initial_state_group=manifest.initial_state_group,
                split_group_id=manifest.split_group_id,
                split=manifest.split,
                candidate_id=branch.candidate_id,
                candidate_fingerprint=branch.candidate_fingerprint,
                primitive=branch.executed_intervention.primitive,
                repeat_index=branch.repeat_index,
                observed_outcome=branch.observed_outcome,
                frozen_vlm_rank=candidate_index,
                checkpoint_probabilities=probabilities_by_group[
                    manifest.decision_group_id
                ][candidate_index],
            )
        )
    return CanonicalPredictionArtifact(
        dataset_id=dataset.dataset_id,
        dataset_sha256=dataset.fingerprint(),
        dataset_admission_evidence_sha256=admitted_dataset.evidence_sha256,
        collection_plan_sha256=admitted_dataset.collection_plan_sha256,
        scorer_verifier_auth_key_id=admitted_dataset.scorer_verifier_auth_key_id,
        outcome_contract_sha256=contract_sha256,
        temperature=temperature,
        checkpoints=tuple(checkpoint_rows),
        records=tuple(records),
    )


def build_canonical_prediction_artifact(
    dataset: object,
    *,
    cache_roots: Sequence[str | Path],
    checkpoint_paths: Sequence[str | Path],
    device: str = "cpu",
    temperature: float = 1.0,
) -> CanonicalPredictionArtifact:
    """Generate receipt-backed predictions; no replay score is accepted."""

    return _generate_canonical_prediction_artifact(
        dataset,
        cache_roots=cache_roots,
        checkpoint_paths=checkpoint_paths,
        device=device,
        temperature=temperature,
    )


def _validate_artifact_against_dataset(
    artifact: CanonicalPredictionArtifact, dataset: object
) -> None:
    from .train_outcomes import require_receipt_backed_dataset

    if not isinstance(artifact, CanonicalPredictionArtifact):
        raise TypeError("artifact must be a CanonicalPredictionArtifact")
    admitted_dataset = dataset
    dataset = require_receipt_backed_dataset(admitted_dataset)
    if artifact.dataset_id != dataset.dataset_id:
        raise ValueError("prediction artifact dataset ID mismatch")
    if artifact.dataset_sha256 != dataset.fingerprint():
        raise ValueError("prediction artifact dataset digest mismatch")
    if artifact.dataset_admission_evidence_sha256 != admitted_dataset.evidence_sha256:
        raise ValueError("prediction artifact dataset admission evidence mismatch")
    if artifact.collection_plan_sha256 != admitted_dataset.collection_plan_sha256:
        raise ValueError("prediction artifact collection plan mismatch")
    if (
        artifact.scorer_verifier_auth_key_id
        != admitted_dataset.scorer_verifier_auth_key_id
    ):
        raise ValueError("prediction artifact scorer verifier trust root mismatch")
    contracts = {
        manifest.outcome_contract.fingerprint() for manifest in dataset.manifests
    }
    if contracts != {artifact.outcome_contract_sha256}:
        raise ValueError("prediction artifact outcome contract mismatch")

    manifest_by_group = {
        manifest.decision_group_id: manifest for manifest in dataset.manifests
    }
    expected_branches = {
        branch.branch_id: branch for branch in dataset.observed_branches
    }
    observed_records = {record.record_id: record for record in artifact.records}
    if set(observed_records) != set(expected_branches):
        raise ValueError("prediction records differ from outcome-evaluated branches")
    for record_id, branch in expected_branches.items():
        record = observed_records[record_id]
        manifest = manifest_by_group[branch.decision_group_id]
        candidate_fingerprints = tuple(
            candidate.fingerprint() for candidate in manifest.candidates
        )
        candidate_index = candidate_fingerprints.index(branch.candidate_fingerprint)
        expected = {
            "branch_sha256": branch.public_fingerprint(),
            "execution_receipt_id": branch.execution_receipt_id,
            "manifest_sha256": manifest.fingerprint(),
            "token_cache_sha256": manifest.token_cache_sha256,
            "outcome_contract_sha256": manifest.outcome_contract.fingerprint(),
            "decision_group_id": manifest.decision_group_id,
            "initial_state_group": manifest.initial_state_group,
            "split_group_id": manifest.split_group_id,
            "split": manifest.split,
            "candidate_id": branch.candidate_id,
            "candidate_fingerprint": branch.candidate_fingerprint,
            "primitive": branch.executed_intervention.primitive,
            "repeat_index": branch.repeat_index,
            "observed_outcome": branch.observed_outcome,
            "frozen_vlm_rank": candidate_index,
        }
        for name, value in expected.items():
            if getattr(record, name) != value:
                raise ValueError(f"canonical prediction changed branch field {name}")


def validate_canonical_prediction_artifact(
    artifact: CanonicalPredictionArtifact,
    dataset: object,
    *,
    cache_roots: Sequence[str | Path],
    checkpoint_paths: Sequence[str | Path],
    device: str = "cpu",
) -> CanonicalPredictionArtifact:
    """Re-run frozen scorers and return the checkpoint-derived prediction values.

    Metadata and declared probabilities are checked for consistency, but formal
    reports consume this regenerated object.  A numerically tiny tolerated GPU
    replay difference can therefore never alter the selected candidate.
    """

    _validate_artifact_against_dataset(artifact, dataset)
    regenerated = _generate_canonical_prediction_artifact(
        dataset,
        cache_roots=cache_roots,
        checkpoint_paths=checkpoint_paths,
        device=device,
        temperature=artifact.temperature,
    )
    if artifact.checkpoints != regenerated.checkpoints:
        raise ValueError("canonical prediction checkpoint evidence mismatch")
    declared_by_id = {item.record_id: item for item in artifact.records}
    for expected in regenerated.records:
        declared = declared_by_id[expected.record_id]
        declared_metadata = declared.to_dict()
        expected_metadata = expected.to_dict()
        declared_probabilities = declared_metadata.pop("checkpoint_probabilities")
        expected_probabilities = expected_metadata.pop("checkpoint_probabilities")
        if declared_metadata != expected_metadata:
            raise ValueError("canonical prediction metadata mismatch")
        if any(
            not math.isclose(float(left), float(right), rel_tol=1e-6, abs_tol=1e-7)
            for left, right in zip(
                declared_probabilities, expected_probabilities, strict=True
            )
        ):
            raise ValueError(
                "canonical prediction probabilities differ from checkpoint"
            )
    return regenerated


class Baseline(str, Enum):
    FROZEN_VLM = "frozen_vlm"
    ALWAYS_DIRECT = "always_direct"
    ALWAYS_OPEN = "always_open"
    RANDOM = "random"
    SCORER = "single_scorer"
    ENSEMBLE = "deep_ensemble"


@dataclasses.dataclass(frozen=True)
class CandidateAggregate:
    candidate_id: str
    candidate_fingerprint: str
    primitive: Primitive
    empirical_success: float
    repetitions: int
    scorer_probability: float | None
    ensemble_probability: float | None
    frozen_vlm_rank: int | None
    feasible: bool


def _aggregate_candidates(
    rows: Sequence[BranchPrediction],
) -> dict[str, tuple[CandidateAggregate, ...]]:
    by_group_and_candidate: dict[tuple[str, str], list[BranchPrediction]] = defaultdict(
        list
    )
    decision_to_split_group: dict[str, str] = {}
    record_ids: set[str] = set()
    for row in rows:
        if row.record_id in record_ids:
            raise ValueError("branch prediction record IDs must be unique")
        record_ids.add(row.record_id)
        previous = decision_to_split_group.setdefault(
            row.decision_group_id, row.split_group_id
        )
        if previous != row.split_group_id:
            raise ValueError("inconsistent split-group identity")
        by_group_and_candidate[
            (row.decision_group_id, row.candidate_fingerprint)
        ].append(row)

    grouped: dict[str, list[CandidateAggregate]] = defaultdict(list)
    for (group_id, fingerprint), candidate_rows in by_group_and_candidate.items():
        candidate_ids = {row.candidate_id for row in candidate_rows}
        primitives = {row.primitive for row in candidate_rows}
        feasible = {row.feasible for row in candidate_rows}
        if len(candidate_ids) != 1 or len(primitives) != 1 or len(feasible) != 1:
            raise ValueError("candidate identity changes across repeated executions")
        repeats = [row.repeat_index for row in candidate_rows]
        if len(repeats) != len(set(repeats)):
            raise ValueError("candidate repeat indices must be unique")

        def fixed_optional(
            name: str,
            repeated_rows: Sequence[BranchPrediction] = candidate_rows,
        ) -> float | int | None:
            values = {getattr(row, name) for row in repeated_rows}
            if len(values) != 1:
                raise ValueError(f"{name} changes across repeated executions")
            return next(iter(values))

        grouped[group_id].append(
            CandidateAggregate(
                candidate_id=next(iter(candidate_ids)),
                candidate_fingerprint=fingerprint,
                primitive=next(iter(primitives)),
                empirical_success=(
                    sum(row.observed_outcome for row in candidate_rows)
                    / len(candidate_rows)
                ),
                repetitions=len(candidate_rows),
                scorer_probability=fixed_optional("scorer_probability"),  # type: ignore[arg-type]
                ensemble_probability=fixed_optional("ensemble_probability"),  # type: ignore[arg-type]
                frozen_vlm_rank=fixed_optional("frozen_vlm_rank"),  # type: ignore[arg-type]
                feasible=next(iter(feasible)),
            )
        )
    return {
        group_id: tuple(
            sorted(candidates, key=lambda candidate: candidate.candidate_fingerprint)
        )
        for group_id, candidates in grouped.items()
    }


def select_baseline_candidate(
    candidates: Sequence[CandidateAggregate],
    *,
    baseline: Baseline,
    decision_group_id: str,
    random_seed: int = 0,
) -> CandidateAggregate | None:
    """Select one feasible candidate with a fully specified baseline rule."""

    feasible = tuple(candidate for candidate in candidates if candidate.feasible)
    if not feasible:
        return None
    if baseline is Baseline.ALWAYS_DIRECT:
        direct = tuple(
            candidate
            for candidate in feasible
            if candidate.primitive is Primitive.DIRECT
        )
        if any(candidate.frozen_vlm_rank is None for candidate in direct):
            raise ValueError("always-DIRECT requires frozen VLM rank")
        return (
            min(direct, key=lambda candidate: int(candidate.frozen_vlm_rank))
            if direct
            else None
        )
    if baseline is Baseline.ALWAYS_OPEN:
        opened = tuple(
            candidate for candidate in feasible if candidate.primitive is Primitive.OPEN
        )
        direct = tuple(
            candidate
            for candidate in feasible
            if candidate.primitive is Primitive.DIRECT
        )
        eligible = opened or direct
        if any(candidate.frozen_vlm_rank is None for candidate in eligible):
            raise ValueError("always-OPEN requires frozen VLM rank")
        return (
            min(eligible, key=lambda candidate: int(candidate.frozen_vlm_rank))
            if eligible
            else None
        )
    if baseline is Baseline.FROZEN_VLM:
        ranked = tuple(
            candidate for candidate in feasible if candidate.frozen_vlm_rank is not None
        )
        if len(ranked) != len(feasible):
            raise ValueError(
                "frozen-VLM baseline requires a recorded rank for every candidate"
            )
        ranks = [candidate.frozen_vlm_rank for candidate in ranked]
        if len(set(ranks)) != len(ranks):
            raise ValueError("frozen-VLM ranks must be unique within a decision group")
        return min(ranked, key=lambda candidate: int(candidate.frozen_vlm_rank))
    if baseline is Baseline.SCORER:
        if any(candidate.scorer_probability is None for candidate in feasible):
            raise ValueError(
                "single-scorer baseline requires every candidate probability"
            )
        return max(
            feasible,
            key=lambda candidate: (
                float(candidate.scorer_probability),
                -int(candidate.candidate_fingerprint, 16),
            ),
        )
    if baseline is Baseline.ENSEMBLE:
        if any(candidate.ensemble_probability is None for candidate in feasible):
            raise ValueError("ensemble baseline requires every candidate probability")
        return max(
            feasible,
            key=lambda candidate: (
                float(candidate.ensemble_probability),
                -int(candidate.candidate_fingerprint, 16),
            ),
        )
    if baseline is Baseline.RANDOM:
        digest = hashlib.sha256(f"{random_seed}:{decision_group_id}".encode()).digest()
        index = int.from_bytes(digest[:8], "big") % len(feasible)
        return sorted(feasible, key=lambda candidate: candidate.candidate_fingerprint)[
            index
        ]
    raise ValueError(f"unsupported baseline: {baseline}")


@dataclasses.dataclass(frozen=True)
class OfflineBranchMatrixEstimate:
    """Selection estimate over already collected paired branch outcomes."""

    evidence_scope: str
    baseline: str
    decision_group_count: int
    selected_group_count: int
    selection_coverage: float
    task_success: BootstrapInterval
    information_action_rate: BootstrapInterval
    mean_repetitions_per_selected_candidate: float
    primitive_outcome_coverage: Mapping[str, Any] | None = None


def evaluate_offline_branch_matrix(
    rows: Sequence[BranchPrediction],
    *,
    baseline: Baseline,
    random_seed: int = 0,
    bootstrap_seed: int = 0,
    bootstrap_samples: int = 10_000,
) -> OfflineBranchMatrixEstimate:
    """Estimate selection from paired real branches without claiming deployment.

    Candidate outcomes are averaged across their repeated model seeds.  An
    absent baseline choice is counted as zero success and zero information
    action, and is also reflected in selection coverage.
    """

    groups = _aggregate_candidates(rows)
    if not groups:
        raise ValueError("offline evaluation requires branch predictions")
    successes: dict[str, float] = {}
    information: dict[str, float] = {}
    repetitions: list[int] = []
    selected_count = 0
    for group_id, candidates in groups.items():
        selected = select_baseline_candidate(
            candidates,
            baseline=baseline,
            decision_group_id=group_id,
            random_seed=random_seed,
        )
        if selected is None:
            successes[group_id] = 0.0
            information[group_id] = 0.0
            continue
        selected_count += 1
        successes[group_id] = selected.empirical_success
        information[group_id] = float(selected.primitive.is_information_action)
        repetitions.append(selected.repetitions)
    return OfflineBranchMatrixEstimate(
        evidence_scope="NON_FORMAL_UNVERIFIED_BRANCH_PREDICTIONS",
        baseline=baseline.value,
        decision_group_count=len(groups),
        selected_group_count=selected_count,
        selection_coverage=selected_count / len(groups),
        task_success=group_bootstrap_interval(
            successes, seed=bootstrap_seed, bootstrap_samples=bootstrap_samples
        ),
        information_action_rate=group_bootstrap_interval(
            information,
            seed=bootstrap_seed + 1,
            bootstrap_samples=bootstrap_samples,
        ),
        mean_repetitions_per_selected_candidate=(
            sum(repetitions) / len(repetitions) if repetitions else 0.0
        ),
    )


def _canonical_rows_for_split(
    artifact: CanonicalPredictionArtifact,
    dataset: object,
    *,
    split: str,
    single_scorer_index: int,
) -> tuple[tuple[BranchPrediction, ...], dict[str, str]]:
    """Require a full candidate-by-seed matrix for one held-out split."""

    from .method_v1_data import CollectionAttemptStatus
    from .train_outcomes import require_receipt_backed_dataset

    _validate_artifact_against_dataset(artifact, dataset)
    dataset = require_receipt_backed_dataset(dataset)
    split = _clean_text(split, name="split").lower()
    if split != "test":
        raise ValueError(
            "formal Method-V1 reporting is restricted to the frozen test split; "
            "validation/calibration are development data"
        )
    if (
        not isinstance(single_scorer_index, int)
        or isinstance(single_scorer_index, bool)
        or not 0 <= single_scorer_index < len(artifact.checkpoints)
    ):
        raise ValueError("single_scorer_index is outside checkpoint evidence")

    manifests = tuple(item for item in dataset.manifests if item.split == split)
    if not manifests:
        raise ValueError(f"dataset has no {split!r} decision groups")
    training_groups = set(artifact.checkpoints[0].training_split_group_ids)
    leaked = training_groups & {item.split_group_id for item in manifests}
    if leaked:
        raise ValueError(
            "formal test split groups overlap checkpoint training groups: "
            f"{sorted(leaked)}"
        )
    manifest_ids = {item.manifest_id for item in manifests}
    expected_entries = {
        item.entry_id: item
        for item in dataset.schedule
        if item.manifest_id in manifest_ids
    }
    attempts = {
        item.schedule_entry.entry_id: item
        for item in dataset.attempts
        if item.schedule_entry.manifest_id in manifest_ids
    }
    if set(attempts) != set(expected_entries):
        raise ValueError("formal branch matrix is incomplete for the requested split")
    if any(
        item.status is not CollectionAttemptStatus.OUTCOME_EVALUATED
        for item in attempts.values()
    ):
        raise ValueError(
            "formal branch matrix contains infrastructure failures without outcomes"
        )
    attempts_by_entry = {entry_id: attempts[entry_id] for entry_id in expected_entries}
    expected_record_ids = {
        attempt.branch.branch_id
        for attempt in attempts_by_entry.values()
        if attempt.branch is not None
    }
    records = {item.record_id: item for item in artifact.records if item.split == split}
    if set(records) != expected_record_ids:
        raise ValueError("formal prediction records do not cover the frozen schedule")

    split_groups: dict[str, str] = {}
    rows: list[BranchPrediction] = []
    for entry_id in sorted(expected_entries):
        entry = expected_entries[entry_id]
        branch = attempts_by_entry[entry_id].branch
        if branch is None:  # pragma: no cover - guarded by status above.
            raise RuntimeError("outcome-evaluated attempt lost its branch")
        record = records[branch.branch_id]
        if record.repeat_index != entry.repeat_index:
            raise ValueError("formal prediction changed a registered repeat index")
        previous = split_groups.setdefault(
            record.decision_group_id, record.split_group_id
        )
        if previous != record.split_group_id:
            raise ValueError("decision group changes split-group identity")
        rows.append(
            BranchPrediction(
                record_id=record.record_id,
                decision_group_id=record.decision_group_id,
                split_group_id=record.split_group_id,
                candidate_id=record.candidate_id,
                candidate_fingerprint=record.candidate_fingerprint,
                primitive=record.primitive,
                repeat_index=record.repeat_index,
                observed_outcome=record.observed_outcome,
                scorer_probability=record.checkpoint_probabilities[single_scorer_index],
                ensemble_probability=(
                    sum(record.checkpoint_probabilities)
                    / len(record.checkpoint_probabilities)
                ),
                frozen_vlm_rank=record.frozen_vlm_rank,
            )
        )
    return tuple(rows), split_groups


def evaluate_formal_branch_matrix(
    artifact: CanonicalPredictionArtifact,
    dataset: object,
    *,
    cache_roots: Sequence[str | Path],
    checkpoint_paths: Sequence[str | Path],
    baseline: Baseline,
    split: str = "test",
    single_scorer_index: int = 0,
    device: str = "cpu",
    random_seed: int = 0,
    bootstrap_seed: int = 0,
    bootstrap_samples: int = 10_000,
) -> OfflineBranchMatrixEstimate:
    """Evaluate verified inference over a complete held-out branch matrix."""

    verified_artifact = validate_canonical_prediction_artifact(
        artifact,
        dataset,
        cache_roots=cache_roots,
        checkpoint_paths=checkpoint_paths,
        device=device,
    )
    rows, split_group_by_decision = _canonical_rows_for_split(
        verified_artifact,
        dataset,
        split=split,
        single_scorer_index=single_scorer_index,
    )
    coverage = _formal_primitive_outcome_coverage(dataset, split=split)
    return dataclasses.replace(
        _evaluate_verified_rows(
            rows,
            split_group_by_decision=split_group_by_decision,
            baseline=baseline,
            random_seed=random_seed,
            bootstrap_seed=bootstrap_seed,
            bootstrap_samples=bootstrap_samples,
        ),
        primitive_outcome_coverage=coverage,
    )


@dataclasses.dataclass(frozen=True)
class FormalProbabilityEvaluation:
    """Probability quality from a verified, complete held-out branch matrix."""

    evidence_scope: str
    dataset_id: str
    dataset_sha256: str
    prediction_artifact_sha256: str
    split: str
    prediction_source: str
    checkpoint_indices: tuple[int, ...]
    metrics: ProbabilityMetrics
    primitive_outcome_coverage: Mapping[str, Any]


def _formal_primitive_outcome_coverage(
    dataset: object, *, split: str
) -> dict[str, Any]:
    """Expose a held-out coverage validity gate only after test unblinding."""

    from .method_v1_data import validate_primitive_outcome_coverage
    from .train_outcomes import require_receipt_backed_dataset

    split_name = _clean_text(split, name="split").lower()
    if split_name != "test":
        raise ValueError("formal outcome coverage is restricted to the test split")
    admitted = require_receipt_backed_dataset(dataset)
    report = validate_primitive_outcome_coverage(
        admitted, required_splits=(split_name,)
    )
    return dict(report["splits"][split_name])


def evaluate_formal_probabilities(
    artifact: CanonicalPredictionArtifact,
    dataset: object,
    *,
    cache_roots: Sequence[str | Path],
    checkpoint_paths: Sequence[str | Path],
    split: str = "test",
    prediction_source: str = "single_scorer",
    single_scorer_index: int = 0,
    num_bins: int = 10,
    device: str = "cpu",
) -> FormalProbabilityEvaluation:
    """Re-run real checkpoints, require full coverage, then score probabilities."""

    verified_artifact = validate_canonical_prediction_artifact(
        artifact,
        dataset,
        cache_roots=cache_roots,
        checkpoint_paths=checkpoint_paths,
        device=device,
    )
    rows, _ = _canonical_rows_for_split(
        verified_artifact,
        dataset,
        split=split,
        single_scorer_index=single_scorer_index,
    )
    coverage = _formal_primitive_outcome_coverage(dataset, split=split)
    if prediction_source == "single_scorer":
        probabilities = tuple(float(row.scorer_probability) for row in rows)
        checkpoint_indices = (single_scorer_index,)
    elif prediction_source == "deep_ensemble":
        probabilities = tuple(float(row.ensemble_probability) for row in rows)
        checkpoint_indices = tuple(range(len(verified_artifact.checkpoints)))
    else:
        raise ValueError("prediction_source must be 'single_scorer' or 'deep_ensemble'")
    return FormalProbabilityEvaluation(
        evidence_scope="FORMAL_CHECKPOINT_REPLAYED_EXECUTED_BRANCH_PROBABILITIES",
        dataset_id=verified_artifact.dataset_id,
        dataset_sha256=verified_artifact.dataset_sha256,
        prediction_artifact_sha256=verified_artifact.artifact_sha256,
        split=_clean_text(split, name="split").lower(),
        prediction_source=prediction_source,
        checkpoint_indices=checkpoint_indices,
        metrics=probability_metrics(
            probabilities,
            [row.observed_outcome for row in rows],
            num_bins=num_bins,
        ),
        primitive_outcome_coverage=coverage,
    )


def _evaluate_verified_rows(
    rows: Sequence[BranchPrediction],
    *,
    split_group_by_decision: Mapping[str, str],
    baseline: Baseline,
    random_seed: int,
    bootstrap_seed: int,
    bootstrap_samples: int,
) -> OfflineBranchMatrixEstimate:
    groups = _aggregate_candidates(rows)
    if set(groups) != set(split_group_by_decision):
        raise ValueError("split-group identities do not cover every decision group")
    successes: dict[str, float] = {}
    information: dict[str, float] = {}
    repetitions: list[int] = []
    selected_count = 0
    for group_id, candidates in groups.items():
        selected = select_baseline_candidate(
            candidates,
            baseline=baseline,
            decision_group_id=group_id,
            random_seed=random_seed,
        )
        if selected is None:
            successes[group_id] = 0.0
            information[group_id] = 0.0
            continue
        selected_count += 1
        successes[group_id] = selected.empirical_success
        information[group_id] = float(selected.primitive.is_information_action)
        repetitions.append(selected.repetitions)
    return OfflineBranchMatrixEstimate(
        evidence_scope="FORMAL_OFFLINE_RESET_CONTROLLED_BRANCH_MATRIX",
        baseline=baseline.value,
        decision_group_count=len(groups),
        selected_group_count=selected_count,
        selection_coverage=selected_count / len(groups),
        task_success=split_group_bootstrap_interval(
            successes,
            split_group_by_decision,
            seed=bootstrap_seed,
            bootstrap_samples=bootstrap_samples,
        ),
        information_action_rate=split_group_bootstrap_interval(
            information,
            split_group_by_decision,
            seed=bootstrap_seed + 1,
            bootstrap_samples=bootstrap_samples,
        ),
        mean_repetitions_per_selected_candidate=(
            sum(repetitions) / len(repetitions) if repetitions else 0.0
        ),
    )


@dataclasses.dataclass(frozen=True)
class PolicyEpisodeResult:
    """One newly executed policy episode under a frozen policy identity."""

    episode_id: str
    reset_group_id: str
    policy_id: str
    final_task_success: bool
    selected_primitive: Primitive | None
    control_steps: int
    vlm_calls: int
    vla_calls: int
    latency_seconds: float
    proposal_covered_successful_candidate: bool | None = None

    def __post_init__(self) -> None:
        for name in ("episode_id", "reset_group_id", "policy_id"):
            object.__setattr__(self, name, _clean_text(getattr(self, name), name=name))
        if not isinstance(self.final_task_success, bool):
            raise TypeError("final_task_success must be a bool")
        if self.selected_primitive is not None and not isinstance(
            self.selected_primitive, Primitive
        ):
            raise TypeError("selected_primitive must be Primitive or None")
        for name in ("control_steps", "vlm_calls", "vla_calls"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not math.isfinite(self.latency_seconds) or self.latency_seconds < 0:
            raise ValueError("latency_seconds must be finite and non-negative")
        if self.proposal_covered_successful_candidate is not None and not isinstance(
            self.proposal_covered_successful_candidate, bool
        ):
            raise TypeError("proposal coverage must be bool or None")


@dataclasses.dataclass(frozen=True)
class ClosedLoopPolicySummary:
    evidence_scope: str
    policy_id: str
    episode_count: int
    reset_group_count: int
    final_task_success: BootstrapInterval
    information_action_rate: float
    mean_control_steps: float
    mean_vlm_calls: float
    mean_vla_calls: float
    mean_latency_seconds: float
    proposal_coverage: float | None


def summarize_closed_loop_episodes(
    episodes: Sequence[PolicyEpisodeResult],
    *,
    bootstrap_seed: int = 0,
    bootstrap_samples: int = 10_000,
) -> ClosedLoopPolicySummary:
    """Summarize actual closed-loop executions, clustered by reset group."""

    rows = tuple(episodes)
    if not rows:
        raise ValueError("closed-loop summary requires at least one episode")
    policies = {row.policy_id for row in rows}
    if len(policies) != 1:
        raise ValueError("one summary cannot mix policy identities")
    episode_ids = [row.episode_id for row in rows]
    if len(episode_ids) != len(set(episode_ids)):
        raise ValueError("episode IDs must be unique")
    by_group: dict[str, list[PolicyEpisodeResult]] = defaultdict(list)
    for row in rows:
        by_group[row.reset_group_id].append(row)
    group_success = {
        group_id: sum(row.final_task_success for row in group_rows) / len(group_rows)
        for group_id, group_rows in by_group.items()
    }
    coverages = [
        row.proposal_covered_successful_candidate
        for row in rows
        if row.proposal_covered_successful_candidate is not None
    ]
    return ClosedLoopPolicySummary(
        evidence_scope="LIVE_CLOSED_LOOP_POLICY_EPISODES",
        policy_id=next(iter(policies)),
        episode_count=len(rows),
        reset_group_count=len(by_group),
        final_task_success=group_bootstrap_interval(
            group_success,
            seed=bootstrap_seed,
            bootstrap_samples=bootstrap_samples,
        ),
        information_action_rate=(
            sum(
                row.selected_primitive is not None
                and row.selected_primitive.is_information_action
                for row in rows
            )
            / len(rows)
        ),
        mean_control_steps=sum(row.control_steps for row in rows) / len(rows),
        mean_vlm_calls=sum(row.vlm_calls for row in rows) / len(rows),
        mean_vla_calls=sum(row.vla_calls for row in rows) / len(rows),
        mean_latency_seconds=sum(row.latency_seconds for row in rows) / len(rows),
        proposal_coverage=(
            sum(bool(value) for value in coverages) / len(coverages)
            if coverages
            else None
        ),
    )


def _load_jsonl(path: str | Path) -> tuple[Mapping[str, object], ...]:
    source = Path(path).expanduser().resolve()
    rows: list[Mapping[str, object]] = []
    for line_number, line in enumerate(
        source.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise TypeError(f"JSONL line {line_number} must be an object")
        rows.append(value)
    if not rows:
        raise ValueError("JSONL input is empty")
    return tuple(rows)


def _load_json_object(path: str | Path) -> Mapping[str, object]:
    source = Path(path).expanduser().resolve()
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"{source} must contain one JSON object")
    return value


def _write_json_once(path: str | Path, value: Mapping[str, object]) -> None:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as handle:
        handle.write(canonical_json_bytes(value) + b"\n")


def _jsonable(value: object) -> object:
    if dataclasses.is_dataclass(value):
        return {
            field.name: _jsonable(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _jsonable(child) for key, child in value.items()}
    return value


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate Method-V1 recorded evidence")
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser(
        "export-predictions",
        help="run real checkpoints and freeze canonical receipt-backed predictions",
    )

    def add_receipt_dataset_inputs(command_parser: argparse.ArgumentParser) -> None:
        command_parser.add_argument("--collection-plan", required=True)
        command_parser.add_argument(
            "--config",
            required=True,
            help="Method-V1 source config; path is a locator and is not plan identity",
        )
        command_parser.add_argument(
            "--freeze-dir",
            action="append",
            required=True,
            help="collector decision freeze; repeat for every decision group",
        )
        command_parser.add_argument(
            "--attempt",
            action="append",
            required=True,
            help="sealed collector attempt.json; repeat for every scheduled branch",
        )
        command_parser.add_argument("--dataset-id", required=True)

    add_receipt_dataset_inputs(export_parser)
    export_parser.add_argument("--cache-root", action="append", required=True)
    export_parser.add_argument("--checkpoint", action="append", required=True)
    export_parser.add_argument("--device", default="cpu")
    export_parser.add_argument("--output", required=True)

    def add_formal_inputs(command_parser: argparse.ArgumentParser) -> None:
        command_parser.add_argument("--artifact", required=True)
        add_receipt_dataset_inputs(command_parser)
        command_parser.add_argument("--cache-root", action="append", required=True)
        command_parser.add_argument("--checkpoint", action="append", required=True)
        command_parser.add_argument("--device", default="cpu")

    verify_parser = subparsers.add_parser(
        "verify-predictions",
        help="re-run checkpoints and verify a canonical prediction artifact",
    )
    add_formal_inputs(verify_parser)

    formal_probability_parser = subparsers.add_parser(
        "formal-probabilities",
        help="evaluate checkpoint-replayed probabilities on a complete held-out matrix",
    )
    add_formal_inputs(formal_probability_parser)
    formal_probability_parser.add_argument("--split", default="test")
    formal_probability_parser.add_argument(
        "--prediction-source",
        choices=("single_scorer", "deep_ensemble"),
        default="single_scorer",
    )
    formal_probability_parser.add_argument("--single-scorer-index", type=int, default=0)
    formal_probability_parser.add_argument("--num-bins", type=int, default=10)
    formal_probability_parser.add_argument("--output")

    formal_matrix_parser = subparsers.add_parser(
        "formal-branch-matrix",
        help="evaluate a verified full candidate-by-seed held-out matrix",
    )
    add_formal_inputs(formal_matrix_parser)
    formal_matrix_parser.add_argument("--split", default="test")
    formal_matrix_parser.add_argument("--single-scorer-index", type=int, default=0)
    formal_matrix_parser.add_argument(
        "--baseline", required=True, choices=[baseline.value for baseline in Baseline]
    )
    formal_matrix_parser.add_argument("--random-seed", type=int, default=0)
    formal_matrix_parser.add_argument("--bootstrap-seed", type=int, default=0)
    formal_matrix_parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    formal_matrix_parser.add_argument("--output")

    probability_parser = subparsers.add_parser(
        "diagnostic-probabilities",
        help="UNVERIFIED JSONL diagnostic; never use as formal evidence",
    )
    probability_parser.add_argument("--predictions", required=True)
    probability_parser.add_argument("--num-bins", type=int, default=10)

    matrix_parser = subparsers.add_parser(
        "diagnostic-branch-matrix",
        help="UNVERIFIED JSONL diagnostic; never use as formal evidence",
    )
    matrix_parser.add_argument("--predictions", required=True)
    matrix_parser.add_argument(
        "--baseline", required=True, choices=[baseline.value for baseline in Baseline]
    )
    matrix_parser.add_argument("--random-seed", type=int, default=0)
    matrix_parser.add_argument("--bootstrap-seed", type=int, default=0)
    matrix_parser.add_argument("--bootstrap-samples", type=int, default=10_000)

    args = parser.parse_args(argv)
    if args.command == "export-predictions":
        from .train_outcomes import build_method_v1_outcome_dataset

        dataset = build_method_v1_outcome_dataset(
            collection_plan_path=args.collection_plan,
            collection_config_path=args.config,
            freeze_dirs=tuple(args.freeze_dir),
            attempt_paths=tuple(args.attempt),
            dataset_id=args.dataset_id,
        )
        artifact = build_canonical_prediction_artifact(
            dataset,
            cache_roots=tuple(args.cache_root),
            checkpoint_paths=tuple(args.checkpoint),
            device=args.device,
        )
        _write_json_once(args.output, artifact.to_dict())
        print(
            json.dumps(
                {
                    "schema_version": "method-v1-prediction-export-report-v1",
                    "evidence_scope": "CANONICAL_CHECKPOINT_DERIVED_PREDICTIONS",
                    "output": str(Path(args.output).expanduser().resolve()),
                    "artifact_sha256": artifact.artifact_sha256,
                    "dataset_sha256": artifact.dataset_sha256,
                    "admission_evidence_sha256": dataset.evidence_sha256,
                    "collection_plan_sha256": dataset.collection_plan_sha256,
                    "scorer_verifier_auth_key_id": (
                        dataset.scorer_verifier_auth_key_id
                    ),
                    "checkpoint_count": len(artifact.checkpoints),
                    "record_count": len(artifact.records),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.command in {
        "verify-predictions",
        "formal-probabilities",
        "formal-branch-matrix",
    }:
        from .train_outcomes import build_method_v1_outcome_dataset

        dataset = build_method_v1_outcome_dataset(
            collection_plan_path=args.collection_plan,
            collection_config_path=args.config,
            freeze_dirs=tuple(args.freeze_dir),
            attempt_paths=tuple(args.attempt),
            dataset_id=args.dataset_id,
        )
        artifact = CanonicalPredictionArtifact.from_mapping(
            _load_json_object(args.artifact)
        )
        if args.command == "verify-predictions":
            verified = validate_canonical_prediction_artifact(
                artifact,
                dataset,
                cache_roots=tuple(args.cache_root),
                checkpoint_paths=tuple(args.checkpoint),
                device=args.device,
            )
            report: Mapping[str, object] = {
                "schema_version": "method-v1-prediction-verification-v1",
                "status": "VERIFIED_BY_CHECKPOINT_REPLAY",
                "artifact_sha256": verified.artifact_sha256,
                "dataset_sha256": verified.dataset_sha256,
                "admission_evidence_sha256": dataset.evidence_sha256,
                "checkpoint_count": len(verified.checkpoints),
                "record_count": len(verified.records),
            }
        elif args.command == "formal-probabilities":
            result = evaluate_formal_probabilities(
                artifact,
                dataset,
                cache_roots=tuple(args.cache_root),
                checkpoint_paths=tuple(args.checkpoint),
                split=args.split,
                prediction_source=args.prediction_source,
                single_scorer_index=args.single_scorer_index,
                num_bins=args.num_bins,
                device=args.device,
            )
            report = {
                "schema_version": "method-v1-formal-probability-evaluation-v1",
                "admission_evidence_sha256": dataset.evidence_sha256,
                "result": _jsonable(result),
            }
        else:
            result = evaluate_formal_branch_matrix(
                artifact,
                dataset,
                cache_roots=tuple(args.cache_root),
                checkpoint_paths=tuple(args.checkpoint),
                baseline=Baseline(args.baseline),
                split=args.split,
                single_scorer_index=args.single_scorer_index,
                device=args.device,
                random_seed=args.random_seed,
                bootstrap_seed=args.bootstrap_seed,
                bootstrap_samples=args.bootstrap_samples,
            )
            report = {
                "schema_version": "method-v1-formal-branch-matrix-evaluation-v1",
                "warning": (
                    "Verified paired-branch estimate; not new closed-loop policy execution."
                ),
                "prediction_artifact_sha256": artifact.artifact_sha256,
                "dataset_sha256": artifact.dataset_sha256,
                "admission_evidence_sha256": dataset.evidence_sha256,
                "split": args.split,
                "result": _jsonable(result),
            }
        output_path = getattr(args, "output", None)
        if output_path is not None:
            _write_json_once(output_path, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    rows = tuple(
        BranchPrediction.from_mapping(value) for value in _load_jsonl(args.predictions)
    )
    if args.command == "diagnostic-probabilities":
        if any(row.scorer_probability is None for row in rows):
            raise ValueError(
                "probability metrics require scorer_probability on every row"
            )
        report: object = {
            "schema_version": "method-v1-probability-evaluation-v1",
            "evidence_scope": "NON_FORMAL_UNVERIFIED_JSONL",
            "warning": "Input rows are trusted as supplied and are not formal evidence.",
            "metrics": _jsonable(
                probability_metrics(
                    [float(row.scorer_probability) for row in rows],
                    [row.observed_outcome for row in rows],
                    num_bins=args.num_bins,
                )
            ),
        }
    else:
        report = {
            "schema_version": "method-v1-offline-policy-evaluation-v1",
            "warning": (
                "UNVERIFIED JSONL diagnostic only; missing candidate/seed branches "
                "are not detected and this is not policy evidence."
            ),
            "result": _jsonable(
                evaluate_offline_branch_matrix(
                    rows,
                    baseline=Baseline(args.baseline),
                    random_seed=args.random_seed,
                    bootstrap_seed=args.bootstrap_seed,
                    bootstrap_samples=args.bootstrap_samples,
                )
            ),
        }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI.
    raise SystemExit(_main())
