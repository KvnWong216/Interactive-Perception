"""Fail-closed command line for the frozen PSR-VLA V1 lifecycle.

The CLI owns validation, training loops, calibration, collection receipts and
metric computation.  Site-specific model/LIBERO construction is deliberately
loaded through an explicit ``module:factory`` seam; there is no replay or
synthetic fallback.  A factory receives keyword arguments and must return the
strict component mapping documented by the corresponding command's help.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
import json
import math
import os
import random
import shutil
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from grounded_interaction.contracts import canonical_sha256

from .collection import (
    ImmutableJSONLReceiptWriter,
    SnapshotCollectionPlan,
    collect_snapshot_rollouts,
    verify_immutable_receipts,
)
from .config import load_psr_config
from .data import RecordKind, load_jsonl, validate_records
from .evaluation import (
    CandidateOutcome,
    TrialOutcome,
    binary_probability_metrics,
    candidate_ranking_metrics,
    execution_metrics,
)
from .preflight import cpu_preflight, real_model_preflight
from .training import (
    CHECKPOINT_SCHEMA,
    SNAPSHOT_SCHEMA,
    CheckpointIdentity,
    CheckpointStage,
    FrozenExecutionSnapshot,
    StageCBatch,
    TemperatureCalibration,
    assert_disjoint_stage_c_parameters,
    assert_execution_snapshot_unchanged,
    compose_stage_a_loss,
    compose_stage_c_loss,
    fit_positive_temperature,
    load_training_checkpoint,
    object_sha256,
    save_training_checkpoint,
)
from .types import PublicHistory

COLLECTION_PLAN_SCHEMA = "psr-v1-collection-plan-v1"
EVALUATION_PLAN_SCHEMA = "psr-v1-evaluation-plan-v1"
CALIBRATION_ROW_SCHEMA = "psr-v1-calibration-prediction-v1"
DRIVER_ENV = "IP_PSR_DRIVER"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read valid JSON from {path}") from error


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _exact(value: Any, expected: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a JSON object")
    if set(value) != expected:
        raise ValueError(
            f"{name} fields differ from the frozen schema; "
            f"missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )
    return value


def _new_output(path: str | Path) -> Path:
    destination = Path(path).expanduser().resolve()
    try:
        destination.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise FileExistsError(
            f"refusing to overwrite or mix an existing output: {destination}"
        ) from error
    return destination


def _resolve_file(path: str | Path, names: Sequence[str]) -> Path:
    source = Path(path).expanduser().resolve()
    if source.is_file():
        return source
    if source.is_dir():
        matches = [source / name for name in names if (source / name).is_file()]
        if len(matches) == 1:
            return matches[0]
    raise FileNotFoundError(
        f"expected one existing file at {source} or one of {list(names)} inside it"
    )


def _load_driver(reference: str) -> Callable[..., Any]:
    value = " ".join(str(reference or "").split())
    if ":" not in value:
        raise ValueError("driver must use the explicit Python module:factory form")
    module_name, attribute = value.split(":", 1)
    if not module_name or not attribute:
        raise ValueError("driver must use the explicit Python module:factory form")
    try:
        factory = getattr(importlib.import_module(module_name), attribute)
    except (ImportError, AttributeError) as error:
        raise RuntimeError(f"cannot load PSR integration factory {value!r}") from error
    if not callable(factory):
        raise TypeError("PSR integration factory is not callable")
    return factory


def _driver_reference(
    args: argparse.Namespace, plan: Mapping[str, Any] | None = None
) -> str:
    value = getattr(args, "driver", None) or os.environ.get(DRIVER_ENV)
    if value is None and plan is not None:
        value = plan.get("component_factory")
    if not value:
        raise RuntimeError(
            "this command requires real model/environment components; pass "
            f"--driver module:factory or set {DRIVER_ENV}. There is no replay "
            "or synthetic fallback."
        )
    return str(value)


def _components(value: Any, expected: set[str], command: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        observed = set(value) if isinstance(value, Mapping) else set()
        raise TypeError(
            f"{command} factory must return exactly {sorted(expected)}; "
            f"received {sorted(observed)}"
        )
    return value


def _seed(seed: int, device: str) -> Any:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("training/calibration requires the learned extra") from error
    random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        torch.cuda.manual_seed_all(seed)
    return torch


def _optimizer_parameters(optimizer: Any) -> list[Any]:
    parameters: list[Any] = []
    seen: set[int] = set()
    for group in getattr(optimizer, "param_groups", ()):
        for parameter in group.get("params", ()):
            if id(parameter) not in seen and parameter.requires_grad:
                parameters.append(parameter)
                seen.add(id(parameter))
    if not parameters:
        raise ValueError("factory optimizer has no trainable parameters")
    return parameters


def _step_accumulated(
    *,
    optimizer: Any,
    parameters: Sequence[Any],
    count: int,
    accumulation: int,
    clip: float,
) -> float:
    import torch

    if count < accumulation:
        scale = accumulation / count
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.mul_(scale)
    norm = torch.nn.utils.clip_grad_norm_(parameters, clip)
    if not bool(torch.isfinite(norm)):
        optimizer.zero_grad(set_to_none=True)
        raise RuntimeError("training gradient norm is non-finite")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return float(norm.detach().cpu().item())


def _snapshot_from_path(path: str | Path) -> FrozenExecutionSnapshot:
    source = _resolve_file(path, ("execution_snapshot.json", "snapshot.json"))
    value = _exact(
        _json(source),
        {
            "schema_version",
            "snapshot_id",
            "parameter_sha256",
            "state_sha256",
            "parameter_count",
            "parameter_names",
            "trainable_before_freeze",
            "identity",
        },
        "execution snapshot",
    )
    identity = CheckpointIdentity.from_mapping(value["identity"])
    snapshot = FrozenExecutionSnapshot(
        schema_version=str(value["schema_version"]),
        snapshot_id=str(value["snapshot_id"]),
        parameter_sha256=str(value["parameter_sha256"]),
        state_sha256=str(value["state_sha256"]),
        parameter_count=int(value["parameter_count"]),
        parameter_names=tuple(value["parameter_names"]),
        trainable_before_freeze=tuple(value["trainable_before_freeze"]),
        identity=identity,
    )
    if snapshot.schema_version != SNAPSHOT_SCHEMA:
        raise ValueError("unsupported execution snapshot schema")
    for name in ("snapshot_id", "parameter_sha256", "state_sha256"):
        digest = getattr(snapshot, name)
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError(f"execution snapshot {name} is not a lowercase SHA-256")
    if snapshot.parameter_count < 1 or not snapshot.parameter_names:
        raise ValueError("execution snapshot has no registered parameters")
    expected_id = canonical_sha256(
        {
            "schema_version": SNAPSHOT_SCHEMA,
            "parameter_sha256": snapshot.parameter_sha256,
            "state_sha256": snapshot.state_sha256,
            "parameter_names": snapshot.parameter_names,
            "trainable_before_freeze": snapshot.trainable_before_freeze,
            "identity": snapshot.identity.to_dict(),
        }
    )
    if snapshot.snapshot_id != expected_id:
        raise ValueError("execution snapshot identity does not match its contents")
    return snapshot


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    config = load_psr_config(args.config)
    report = (
        real_model_preflight(config, device=args.device)
        if args.load_model
        else cpu_preflight(config, device=args.device)
    )
    return report.to_dict()


def _import_data(args: argparse.Namespace) -> dict[str, Any]:
    source = Path(args.source).expanduser().resolve()
    records_path = _resolve_file(source, ("records.jsonl", "branches.jsonl"))
    records = load_jsonl(
        str(records_path), expected_kind=RecordKind.WARMUP_DEMONSTRATIONS
    )
    summary = validate_records(records, expected_kind=RecordKind.WARMUP_DEMONSTRATIONS)
    if source.is_dir():
        if (source / "import_manifest.json").exists():
            raise ValueError(
                "source already contains an import manifest; use the raw real data"
            )
        files = [item for item in sorted(source.rglob("*")) if item.is_file()]
        if any(item.is_symlink() for item in source.rglob("*")):
            raise ValueError("real-data import refuses symbolic links")
    destination = _new_output(args.output)
    if source.is_dir():
        for item in files:
            relative = item.relative_to(source)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(item, target)
    else:
        shutil.copyfile(source, destination / "records.jsonl")
    imported_records = _resolve_file(destination, ("records.jsonl", "branches.jsonl"))
    if _sha256(imported_records) != _sha256(records_path):
        raise RuntimeError("imported record bytes differ from the real source")
    files = [item for item in sorted(destination.rglob("*")) if item.is_file()]
    manifest = {
        "schema_version": "psr-v1-import-manifest-v1",
        "source": str(source),
        "dataset": summary.to_dict(),
        "files": [
            {
                "path": item.relative_to(destination).as_posix(),
                "size": item.stat().st_size,
                "sha256": _sha256(item),
            }
            for item in files
        ],
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    _write_json(destination / "import_manifest.json", manifest)
    return manifest


def _run_warmup(args: argparse.Namespace) -> dict[str, Any]:
    config = load_psr_config(args.config)
    data = Path(args.data).expanduser().resolve()
    records = load_jsonl(
        str(_resolve_file(data, ("records.jsonl", "branches.jsonl"))),
        expected_kind=RecordKind.WARMUP_DEMONSTRATIONS,
    )
    dataset = validate_records(records, expected_kind=RecordKind.WARMUP_DEMONSTRATIONS)
    if not {"train", "validation"}.issubset(dataset.splits):
        raise ValueError("Stage A requires group-disjoint train and validation splits")
    dataset = validate_records(records, expected_kind=RecordKind.WARMUP_DEMONSTRATIONS)
    if not {"train", "validation"}.issubset(dataset.splits):
        raise ValueError("Stage A requires disjoint train and validation records")
    torch = _seed(int(config.section("training")["seed"]), args.device)
    factory = _load_driver(_driver_reference(args))
    parts = _components(
        factory(command="train-warmup", config=config, data=data, device=args.device),
        {
            "execution_model",
            "backend",
            "optimizer",
            "train_batches",
            "validation_batches",
            "identity",
        },
        "train-warmup",
    )
    if not isinstance(parts["identity"], CheckpointIdentity):
        raise TypeError("train-warmup identity must be CheckpointIdentity")
    if parts["identity"].config_sha256 != config.fingerprint:
        raise ValueError("train-warmup checkpoint identity does not match config")
    if parts["identity"].split_sha256 != dataset.identity_sha256:
        raise ValueError("train-warmup checkpoint identity does not match data")
    if parts["identity"].config_sha256 != config.fingerprint:
        raise ValueError("Stage-A identity does not match the frozen config")
    if parts["identity"].split_sha256 != dataset.identity_sha256:
        raise ValueError("Stage-A identity does not match the imported split data")
    if not callable(parts["train_batches"]) or not callable(
        parts["validation_batches"]
    ):
        raise TypeError("training batch providers must be callable")
    output = _new_output(args.output)
    model, optimizer = parts["execution_model"], parts["optimizer"]
    if args.resume:
        load_training_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            expected_stage=CheckpointStage.WARMUP,
            restore_rng=True,
        )
    accumulation = int(config.section("training")["effective_batch"]) // int(
        config.section("training")["microbatch"]
    )
    parameters = _optimizer_parameters(optimizer)
    best = math.inf
    stale = 0
    logs: list[dict[str, Any]] = []
    best_path = output / "warmup.pt"
    for epoch in range(args.max_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_values: list[float] = []
        pending = 0
        for batch in parts["train_batches"](epoch, records):
            if not isinstance(batch, Mapping):
                raise TypeError("Stage-A batches must be compose_stage_a_loss kwargs")
            if batch.get("backend") is not parts["backend"]:
                raise ValueError(
                    "Stage-A batch did not use the registered real backend"
                )
            if batch.get("backend") is not parts["backend"]:
                raise RuntimeError("Stage-A batch substituted a different backend")
            loss = compose_stage_a_loss(**dict(batch))
            (loss.total / accumulation).backward()
            train_values.append(float(loss.total.detach().cpu().item()))
            pending += 1
            if pending == accumulation:
                _step_accumulated(
                    optimizer=optimizer,
                    parameters=parameters,
                    count=pending,
                    accumulation=accumulation,
                    clip=float(config.section("training")["grad_clip_norm"]),
                )
                pending = 0
        if pending:
            _step_accumulated(
                optimizer=optimizer,
                parameters=parameters,
                count=pending,
                accumulation=accumulation,
                clip=float(config.section("training")["grad_clip_norm"]),
            )
        if not train_values:
            raise RuntimeError("real Stage-A training provider yielded no batches")
        model.eval()
        validation: list[float] = []
        with torch.no_grad():
            for batch in parts["validation_batches"](epoch, records):
                if not isinstance(batch, Mapping):
                    raise TypeError(
                        "Stage-A validation batches must be loss-argument mappings"
                    )
                if batch.get("backend") is not parts["backend"]:
                    raise ValueError(
                        "Stage-A validation did not use the registered real backend"
                    )
                if batch.get("backend") is not parts["backend"]:
                    raise RuntimeError("Stage-A validation substituted another backend")
                loss = compose_stage_a_loss(**dict(batch))
                validation.append(float(loss.total.detach().cpu().item()))
        if not validation:
            raise RuntimeError("real Stage-A validation provider yielded no batches")
        score = sum(validation) / len(validation)
        logs.append(
            {
                "epoch": epoch,
                "train_loss": sum(train_values) / len(train_values),
                "validation_loss": score,
            }
        )
        if score < best:
            best, stale = score, 0
            save_training_checkpoint(
                best_path,
                stage=CheckpointStage.WARMUP,
                model=model,
                optimizer=optimizer,
                identity=parts["identity"],
            )
        else:
            stale += 1
            if stale >= args.patience:
                break
    load_training_checkpoint(best_path, model=model, expected_stage="warmup")
    from .training import freeze_execution_snapshot

    snapshot = freeze_execution_snapshot(model, identity=parts["identity"])
    save_training_checkpoint(
        output / "s0.pt",
        stage=CheckpointStage.EXECUTION_SNAPSHOT,
        model=model,
        identity=dataclasses.replace(
            parts["identity"], execution_snapshot_id=snapshot.snapshot_id
        ),
    )
    _write_json(output / "execution_snapshot.json", snapshot.to_dict())
    _write_json(output / "training_log.json", logs)
    return {
        "stage": "S0",
        "records": len(records),
        "epochs": len(logs),
        "best_validation_loss": best,
        "snapshot_id": snapshot.snapshot_id,
    }


def _collection_plan(
    path: str | Path, snapshot_id: str
) -> tuple[dict[str, Any], tuple[SnapshotCollectionPlan, ...]]:
    source = Path(path).expanduser().resolve()
    root = _exact(
        _json(source),
        {"schema_version", "component_factory", "entries"},
        "collection plan",
    )
    if root["schema_version"] != COLLECTION_PLAN_SCHEMA:
        raise ValueError("unsupported collection plan schema")
    if not isinstance(root["entries"], list) or not root["entries"]:
        raise ValueError("collection plan requires real decision-point entries")
    plans = []
    keys = {
        "episode_id",
        "group_id",
        "split",
        "initial_reset_ref",
        "public_history_path",
        "episode_seed",
        "behavior_seed",
        "selection_mode",
        "candidate_generation_version",
        "continuation_id",
        "target_encoder_id",
    }
    for index, raw in enumerate(root["entries"]):
        row = _exact(raw, keys, f"collection plan entry {index}")
        history_path = Path(str(row["public_history_path"])).expanduser()
        if not history_path.is_absolute() or not history_path.is_file():
            raise FileNotFoundError(
                "public_history_path must be an existing absolute JSON file"
            )
        plans.append(
            SnapshotCollectionPlan(
                episode_id=row["episode_id"],
                group_id=row["group_id"],
                split=row["split"],
                initial_reset_ref=row["initial_reset_ref"],
                public_history=PublicHistory.from_mapping(_json(history_path)),
                episode_seed=row["episode_seed"],
                behavior_seed=row["behavior_seed"],
                selection_mode=row["selection_mode"],
                candidate_generation_version=row["candidate_generation_version"],
                execution_snapshot_id=snapshot_id,
                continuation_id=row["continuation_id"],
                target_encoder_id=row["target_encoder_id"],
            )
        )
    return dict(root), tuple(plans)


def _collect(args: argparse.Namespace) -> dict[str, Any]:
    config = load_psr_config(args.config)
    snapshot = _snapshot_from_path(args.snapshot)
    plan_root, plans = _collection_plan(args.plan, snapshot.snapshot_id)
    factory = _load_driver(_driver_reference(args, plan_root))
    output = _new_output(args.output)
    parts = _components(
        factory(
            command="collect",
            config=config,
            snapshot=snapshot,
            plan_path=Path(args.plan).expanduser().resolve(),
            artifact_root=output / "artifacts",
            device=args.device,
        ),
        {
            "policy",
            "environment_factory",
            "evaluator",
            "artifact_sink",
            "execution_snapshot_id",
        },
        "collect",
    )
    if parts["execution_snapshot_id"] != snapshot.snapshot_id:
        raise ValueError("collection policy does not attest the requested S0")
    receipt_path = output / "snapshot_rollouts.receipts.jsonl"
    summaries = []
    with ImmutableJSONLReceiptWriter(receipt_path) as writer:
        for plan in plans:
            summaries.append(
                collect_snapshot_rollouts(
                    plan=plan,
                    policy=parts["policy"],
                    environment_factory=parts["environment_factory"],
                    evaluator=parts["evaluator"],
                    artifact_sink=parts["artifact_sink"],
                    receipt_writer=writer,
                )
            )
    records = verify_immutable_receipts(receipt_path)
    dataset = validate_records(
        records,
        expected_kind=RecordKind.SNAPSHOT_ROLLOUTS,
        expected_snapshot_id=snapshot.snapshot_id,
    )
    records_path = output / "records.jsonl"
    with records_path.open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(
                json.dumps(record.to_dict(), sort_keys=True, allow_nan=False) + "\n"
            )
        stream.flush()
        os.fsync(stream.fileno())
    report = {
        "schema_version": "psr-v1-collection-report-v1",
        "plans": len(plans),
        "proposed_candidates": sum(item.proposed_candidates for item in summaries),
        "selected_branches": sum(item.selected_branches for item in summaries),
        "dataset": dataset.to_dict(),
        "receipt_sha256": _sha256(receipt_path),
        "records_sha256": _sha256(records_path),
    }
    _write_json(output / "collection_report.json", report)
    return report


def _run_outcomes(args: argparse.Namespace) -> dict[str, Any]:
    config = load_psr_config(args.config)
    snapshot = _snapshot_from_path(args.snapshot)
    records = load_jsonl(
        str(_resolve_file(args.data, ("records.jsonl", "branches.jsonl"))),
        expected_kind=RecordKind.SNAPSHOT_ROLLOUTS,
        expected_snapshot_id=snapshot.snapshot_id,
    )
    dataset = validate_records(
        records,
        expected_kind=RecordKind.SNAPSHOT_ROLLOUTS,
        expected_snapshot_id=snapshot.snapshot_id,
    )
    if not {"train", "validation"}.issubset(dataset.splits):
        raise ValueError("Stage C requires group-disjoint train and validation splits")
    dataset = validate_records(
        records,
        expected_kind=RecordKind.SNAPSHOT_ROLLOUTS,
        expected_snapshot_id=snapshot.snapshot_id,
    )
    if not {"train", "validation"}.issubset(dataset.splits):
        raise ValueError("Stage C requires disjoint train and validation records")
    torch = _seed(int(config.section("training")["seed"]), args.device)
    factory = _load_driver(_driver_reference(args))
    parts = _components(
        factory(
            command="train-outcomes",
            config=config,
            snapshot=snapshot,
            data=Path(args.data).expanduser().resolve(),
            device=args.device,
        ),
        {
            "execution_model",
            "predictor",
            "optimizer",
            "train_batches",
            "validation_batches",
            "identity",
        },
        "train-outcomes",
    )
    if not isinstance(parts["identity"], CheckpointIdentity):
        raise TypeError("train-outcomes identity must be CheckpointIdentity")
    if parts["identity"].config_sha256 != config.fingerprint:
        raise ValueError("Stage-C identity does not match the frozen config")
    if parts["identity"].split_sha256 != dataset.identity_sha256:
        raise ValueError("Stage-C identity does not match the rollout split data")
    if not callable(parts["train_batches"]) or not callable(
        parts["validation_batches"]
    ):
        raise TypeError("training batch providers must be callable")
    if parts["identity"].execution_snapshot_id != snapshot.snapshot_id:
        raise ValueError("predictor identity does not match S0")
    if parts["identity"].config_sha256 != config.fingerprint:
        raise ValueError("predictor identity does not match the frozen config")
    if parts["identity"].split_sha256 != dataset.identity_sha256:
        raise ValueError("predictor identity does not match the rollout data")
    execution, predictor, optimizer = (
        parts["execution_model"],
        parts["predictor"],
        parts["optimizer"],
    )
    assert_execution_snapshot_unchanged(execution, snapshot)
    assert_disjoint_stage_c_parameters(execution, optimizer)
    output = _new_output(args.output)
    best_path = output / "predictor.pt"
    best, stale = math.inf, 0
    logs = []
    parameters = _optimizer_parameters(optimizer)
    accumulation = int(config.section("training")["effective_batch"])
    for epoch in range(args.max_epochs):
        predictor.train()
        optimizer.zero_grad(set_to_none=True)
        pending = 0
        train_values = []
        for batch in parts["train_batches"](epoch, records):
            if not isinstance(batch, StageCBatch):
                raise TypeError("Stage-C provider must yield StageCBatch values")
            loss = compose_stage_c_loss(batch.forward(predictor), batch)
            if not loss.has_supervision:
                continue
            assert loss.total is not None
            (loss.total / accumulation).backward()
            train_values.append(float(loss.total.detach().cpu().item()))
            pending += 1
            if pending == accumulation:
                _step_accumulated(
                    optimizer=optimizer,
                    parameters=parameters,
                    count=pending,
                    accumulation=accumulation,
                    clip=float(config.section("training")["grad_clip_norm"]),
                )
                pending = 0
        if pending:
            _step_accumulated(
                optimizer=optimizer,
                parameters=parameters,
                count=pending,
                accumulation=accumulation,
                clip=float(config.section("training")["grad_clip_norm"]),
            )
        if not train_values:
            raise RuntimeError("Stage-C provider yielded no supervised training batch")
        predictor.eval()
        validation = []
        with torch.no_grad():
            for batch in parts["validation_batches"](epoch, records):
                if not isinstance(batch, StageCBatch):
                    raise TypeError(
                        "Stage-C validation provider must yield StageCBatch values"
                    )
                loss = compose_stage_c_loss(batch.forward(predictor), batch)
                if loss.total is not None:
                    validation.append(float(loss.total.detach().cpu().item()))
        if not validation:
            raise RuntimeError(
                "Stage-C provider yielded no supervised validation batch"
            )
        score = sum(validation) / len(validation)
        logs.append(
            {
                "epoch": epoch,
                "train_loss": sum(train_values) / len(train_values),
                "validation_loss": score,
            }
        )
        if score < best:
            best, stale = score, 0
            save_training_checkpoint(
                best_path,
                stage=CheckpointStage.OUTCOME_PREDICTOR,
                model=predictor,
                optimizer=optimizer,
                identity=parts["identity"],
            )
        else:
            stale += 1
            if stale >= args.patience:
                break
        assert_execution_snapshot_unchanged(execution, snapshot)
    assert_execution_snapshot_unchanged(execution, snapshot)
    _write_json(output / "training_log.json", logs)
    return {
        "stage": "predictor",
        "records": len(records),
        "epochs": len(logs),
        "best_validation_loss": best,
    }


def _verified_checkpoint_payload(
    path: Path, expected_stage: str
) -> tuple[dict[str, Any], CheckpointIdentity]:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "checkpoint verification requires the learned extra"
        ) from error
    payload = torch.load(path, map_location="cpu", weights_only=False)
    keys = {
        "schema_version",
        "stage",
        "identity",
        "model_state_sha256",
        "optimizer_state_sha256",
        "rng_state_sha256",
        "calibration",
        "payload_sha256",
        "model_state",
        "optimizer_state",
        "rng_state",
    }
    value = dict(_exact(payload, keys, "PSR checkpoint"))
    if value["schema_version"] != CHECKPOINT_SCHEMA or value["stage"] != expected_stage:
        raise ValueError(f"expected a verified {expected_stage} PSR checkpoint")
    identity = CheckpointIdentity.from_mapping(value["identity"])
    calibration = value["calibration"]
    if expected_stage == CheckpointStage.CALIBRATED_PREDICTOR.value:
        if not isinstance(calibration, Mapping):
            raise ValueError("calibrated predictor lacks calibration metadata")
        TemperatureCalibration(**dict(calibration))
    elif calibration is not None:
        raise ValueError("uncalibrated predictor unexpectedly contains calibration")
    fields = {
        "schema_version": CHECKPOINT_SCHEMA,
        "stage": expected_stage,
        "identity": identity.to_dict(),
        "model_state_sha256": object_sha256(value["model_state"]),
        "optimizer_state_sha256": None
        if value["optimizer_state"] is None
        else object_sha256(value["optimizer_state"]),
        "rng_state_sha256": object_sha256(value["rng_state"]),
        "calibration": calibration,
    }
    if any(
        value[name] != fields[name]
        for name in ("model_state_sha256", "optimizer_state_sha256", "rng_state_sha256")
    ):
        raise ValueError("checkpoint state digest mismatch")
    if value["payload_sha256"] != canonical_sha256(fields):
        raise ValueError("checkpoint payload digest mismatch")
    return value, identity


class _StateCarrier:
    def __init__(self, state: Mapping[str, Any]) -> None:
        self._state = state

    def named_parameters(self) -> tuple[Any, ...]:
        return ()

    def state_dict(self) -> Mapping[str, Any]:
        return self._state


def _calibrate(args: argparse.Namespace) -> dict[str, Any]:
    predictor = _resolve_file(args.predictor, ("predictor.pt",))
    payload, identity = _verified_checkpoint_payload(predictor, "predictor")
    snapshot = _snapshot_from_path(args.snapshot)
    if identity.execution_snapshot_id != snapshot.snapshot_id:
        raise ValueError("predictor was not fitted for the supplied S0")
    data = _resolve_file(args.data, ("calibration_predictions.jsonl",))
    rows = []
    for number, line in enumerate(data.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = _exact(
            json.loads(line),
            {
                "schema_version",
                "record_id",
                "split",
                "execution_snapshot_id",
                "predictor_file_sha256",
                "failure_logit",
                "failure_target",
                "cost_valid",
            },
            f"calibration row {number}",
        )
        if (
            row["schema_version"] != CALIBRATION_ROW_SCHEMA
            or row["split"] != "calibration"
        ):
            raise ValueError(
                "calibration rows require the calibration split and V1 schema"
            )
        if row["execution_snapshot_id"] != snapshot.snapshot_id or row[
            "predictor_file_sha256"
        ] != _sha256(predictor):
            raise ValueError("calibration row model/S0 identity mismatch")
        valid = row["cost_valid"]
        if not isinstance(valid, bool):
            raise TypeError("cost_valid must be bool")
        if valid:
            if (
                isinstance(row["failure_logit"], bool)
                or not isinstance(row["failure_logit"], (int, float))
                or not math.isfinite(float(row["failure_logit"]))
            ):
                raise ValueError("valid calibration logit must be finite")
            if row["failure_target"] not in (0, 1, False, True):
                raise ValueError("valid calibration target must be binary")
        elif row["failure_logit"] is not None or row["failure_target"] is not None:
            raise ValueError("invalid calibration rows cannot carry predictions/labels")
        rows.append(dict(row))
    if len({row["record_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate calibration record_id")
    valid_rows = [row for row in rows if row["cost_valid"]]
    if not valid_rows:
        raise ValueError("calibration data contains no valid real outcomes")
    import torch

    calibration_identity = canonical_sha256(
        {"schema": CALIBRATION_ROW_SCHEMA, "rows": rows}
    )
    calibration = fit_positive_temperature(
        torch.tensor([float(row["failure_logit"]) for row in valid_rows]),
        torch.tensor([float(row["failure_target"]) for row in valid_rows]),
        calibration_sha256=calibration_identity,
    )
    output = _new_output(args.output)
    extra = dict(identity.extra)
    extra["calibration_data_sha256"] = calibration_identity
    calibrated_identity = dataclasses.replace(identity, extra=extra)
    manifest = save_training_checkpoint(
        output / "calibrated_predictor.pt",
        stage=CheckpointStage.CALIBRATED_PREDICTOR,
        model=_StateCarrier(payload["model_state"]),
        identity=calibrated_identity,
        calibration=calibration,
    )
    report = {
        "schema_version": "psr-v1-calibration-report-v1",
        "checkpoint_file_sha256": manifest.file_sha256,
        "calibration": dataclasses.asdict(calibration),
    }
    _write_json(output / "calibration_report.json", report)
    return report


@dataclasses.dataclass(frozen=True)
class EvaluationTrialSpec:
    trial_id: str
    reset_family: str
    initial_reset_ref: str
    task: str
    episode_seed: int

    def __post_init__(self) -> None:
        for name in ("trial_id", "reset_family", "initial_reset_ref", "task"):
            value = " ".join(str(getattr(self, name) or "").split())
            if not value:
                raise ValueError(f"{name} must be non-empty")
            object.__setattr__(self, name, value)
        if (
            not isinstance(self.episode_seed, int)
            or isinstance(self.episode_seed, bool)
            or self.episode_seed < 0
        ):
            raise ValueError("episode_seed must be a non-negative integer")


@dataclasses.dataclass(frozen=True)
class EvaluationTrialResult:
    trial_id: str
    outcome: TrialOutcome
    probabilities: tuple[float, ...] = ()
    failure_targets: tuple[int, ...] = ()
    candidate_outcomes: tuple[CandidateOutcome, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.trial_id, str) or not self.trial_id.strip():
            raise TypeError("evaluation result requires a non-empty trial_id")
        if not isinstance(self.outcome, TrialOutcome):
            raise TypeError("evaluation result requires a TrialOutcome")
        if len(self.probabilities) != len(self.failure_targets):
            raise ValueError("probability predictions and targets must align")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
            for value in self.probabilities
        ):
            raise ValueError("evaluation probabilities must be finite in [0,1]")
        if any(value not in (0, 1, False, True) for value in self.failure_targets):
            raise ValueError("evaluation failure targets must be binary")
        if any(
            not isinstance(value, CandidateOutcome) for value in self.candidate_outcomes
        ):
            raise TypeError("candidate_outcomes must contain CandidateOutcome values")


def _evaluate(args: argparse.Namespace) -> dict[str, Any]:
    config = load_psr_config(args.config)
    plan_path = Path(args.plan).expanduser().resolve()
    plan = _exact(
        _json(plan_path),
        {"schema_version", "component_factory", "trials"},
        "evaluation plan",
    )
    if (
        plan["schema_version"] != EVALUATION_PLAN_SCHEMA
        or not isinstance(plan["trials"], list)
        or not plan["trials"]
    ):
        raise ValueError("evaluation plan requires V1 schema and real trials")
    specs = []
    for index, raw in enumerate(plan["trials"]):
        row = _exact(
            raw,
            {"trial_id", "reset_family", "initial_reset_ref", "task", "episode_seed"},
            f"evaluation trial {index}",
        )
        specs.append(EvaluationTrialSpec(**row))
    snapshot = predictor = None
    if args.mode == "psr":
        if not args.snapshot or not args.predictor:
            raise ValueError("psr evaluation requires --snapshot and --predictor")
        snapshot = _snapshot_from_path(args.snapshot)
        predictor = _resolve_file(args.predictor, ("calibrated_predictor.pt",))
        _, identity = _verified_checkpoint_payload(predictor, "calibrated_predictor")
        if identity.execution_snapshot_id != snapshot.snapshot_id:
            raise ValueError("calibrated predictor does not match S0")
    factory = _load_driver(_driver_reference(args, plan))
    output = _new_output(args.output)
    parts = _components(
        factory(
            command="evaluate",
            config=config,
            mode=args.mode,
            snapshot=snapshot,
            predictor=predictor,
            plan_path=plan_path,
            artifact_root=output / "artifacts",
            device=args.device,
        ),
        {"run_trial", "attestation"},
        "evaluate",
    )
    if not callable(parts["run_trial"]):
        raise TypeError("evaluate factory run_trial must be callable")
    attestation = _exact(
        parts["attestation"],
        {
            "mode",
            "base_model_id",
            "base_revision",
            "execution_snapshot_id",
            "predictor_file_sha256",
        },
        "evaluation driver attestation",
    )
    base = config.section("base")
    expected_attestation = {
        "mode": args.mode,
        "base_model_id": base["model_id"],
        "base_revision": base["revision"],
        "execution_snapshot_id": None if snapshot is None else snapshot.snapshot_id,
        "predictor_file_sha256": None if predictor is None else _sha256(predictor),
    }
    if dict(attestation) != expected_attestation:
        raise ValueError(
            "evaluation driver attestation does not match the frozen run identity"
        )
    results = []
    with (output / "trial_results.jsonl").open("x", encoding="utf-8") as stream:
        for spec in specs:
            result = parts["run_trial"](spec)
            if (
                not isinstance(result, EvaluationTrialResult)
                or result.trial_id != spec.trial_id
            ):
                raise TypeError(
                    "run_trial must return the matching EvaluationTrialResult"
                )
            if len(result.probabilities) != len(result.failure_targets):
                raise ValueError(
                    "evaluation probability predictions/targets are unaligned"
                )
            if args.mode == "native" and (
                result.probabilities or result.candidate_outcomes
            ):
                raise ValueError(
                    "native evaluation cannot masquerade as PSR prediction evidence"
                )
            stream.write(
                json.dumps(dataclasses.asdict(result), sort_keys=True, allow_nan=False)
                + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
            results.append(result)
    execution = execution_metrics([result.outcome for result in results])
    probabilities = [value for result in results for value in result.probabilities]
    targets = [value for result in results for value in result.failure_targets]
    candidates = [value for result in results for value in result.candidate_outcomes]
    report = {
        "schema_version": "psr-v1-evaluation-report-v1",
        "mode": args.mode,
        "execution": dataclasses.asdict(execution),
        "probability": None
        if not probabilities
        else binary_probability_metrics(probabilities, targets).to_dict(),
        "candidate_ranking": None
        if not candidates
        else candidate_ranking_metrics(candidates).to_dict(),
        "unavailable": [
            name
            for name, value in (
                ("probability", probabilities),
                ("candidate_ranking", candidates),
            )
            if not value
        ],
    }
    _write_json(output / "evaluation_report.json", report)
    return report


def _add_driver(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--driver",
        help=f"real integration module:factory (or set {DRIVER_ENV}); no synthetic fallback",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m grounded_interaction.psr",
        description="Frozen PSR-VLA V1 lifecycle; real data/model/environment commands fail closed.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser(
        "preflight",
        help="validate config/dependencies; optionally run the real pinned-model canary",
    )
    preflight.add_argument("--config", required=True)
    preflight.add_argument("--device", choices=("cpu", "cuda"), required=True)
    preflight.add_argument(
        "--load-model",
        action="store_true",
        help="load the real pinned MolmoAct2 checkpoint; CUDA only",
    )

    imported = commands.add_parser(
        "import-data",
        help="validate and byte-copy user-provided real warmup records/assets",
    )
    imported.add_argument(
        "--source",
        required=True,
        help="real records JSONL or directory containing records.jsonl",
    )
    imported.add_argument(
        "--output",
        required=True,
        help="new output directory; existing paths are rejected",
    )

    warmup = commands.add_parser(
        "train-warmup",
        help="run Stage A on real batches and freeze execution snapshot S0",
    )
    warmup.add_argument("--config", required=True)
    warmup.add_argument("--data", required=True, help="imported real warmup data")
    warmup.add_argument("--output", required=True)
    warmup.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    warmup.add_argument("--max-epochs", type=_positive_int, default=20)
    warmup.add_argument("--patience", type=_positive_int, default=3)
    warmup.add_argument("--resume", help="verified Stage-A checkpoint")
    _add_driver(warmup)

    collect = commands.add_parser(
        "collect", help="execute real same-reset S0 branches and seal receipts"
    )
    collect.add_argument("--config", required=True)
    collect.add_argument("--snapshot", required=True)
    collect.add_argument(
        "--plan",
        required=True,
        help="plan referencing real reset assets and public histories",
    )
    collect.add_argument("--output", required=True)
    collect.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    _add_driver(collect)

    outcomes = commands.add_parser(
        "train-outcomes", help="fit E/C only from actual branches under frozen S0"
    )
    outcomes.add_argument("--config", required=True)
    outcomes.add_argument("--snapshot", required=True)
    outcomes.add_argument("--data", required=True, help="real snapshot_rollouts JSONL")
    outcomes.add_argument("--output", required=True)
    outcomes.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    outcomes.add_argument("--max-epochs", type=_positive_int, default=20)
    outcomes.add_argument("--patience", type=_positive_int, default=3)
    _add_driver(outcomes)

    calibrate = commands.add_parser(
        "calibrate",
        help="fit held-out positive temperature and emit a bound calibrated checkpoint",
    )
    calibrate.add_argument("--snapshot", required=True)
    calibrate.add_argument("--predictor", required=True)
    calibrate.add_argument(
        "--data",
        required=True,
        help="real calibration_predictions.jsonl; both outcome classes required",
    )
    calibrate.add_argument("--output", required=True)

    evaluate = commands.add_parser(
        "evaluate",
        help="execute real native or PSR trials, then compute repository metrics",
    )
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--mode", choices=("native", "psr"), required=True)
    evaluate.add_argument("--snapshot")
    evaluate.add_argument("--predictor")
    evaluate.add_argument(
        "--plan",
        required=True,
        help="plan referencing real reset/assets; it contains no outcomes",
    )
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    _add_driver(evaluate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "preflight": _preflight,
        "import-data": _import_data,
        "train-warmup": _run_warmup,
        "collect": _collect,
        "train-outcomes": _run_outcomes,
        "calibrate": _calibrate,
        "evaluate": _evaluate,
    }
    try:
        result = handlers[args.command](args)
    except Exception as error:  # noqa: BLE001 - CLI boundary must fail closed.
        print(f"ip-psr: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    if args.command == "preflight" and not result.get("passed", False):
        return 2
    return 0


__all__ = [
    "CALIBRATION_ROW_SCHEMA",
    "COLLECTION_PLAN_SCHEMA",
    "EVALUATION_PLAN_SCHEMA",
    "EvaluationTrialResult",
    "EvaluationTrialSpec",
    "build_parser",
    "main",
]
