"""Hash-bound offline release and explicit external-gate audit."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

_LOCAL_PACKAGE_ROOTS = frozenset(
    ("calibrated_interaction", "interactive_perception", "piu")
)
_REPOSITORY_FILE_LITERAL = re.compile(
    r"^(?:configs|results/diagnostics|scenarios|scripts|src)/[^\0]+$"
)
_SHELL_PROJECT_FILE = re.compile(
    r"\$PROJECT_ROOT/(scripts/[A-Za-z0-9_./-]+\.(?:py|sh))"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _files(value: Any, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence")
    result = tuple(str(item) for item in value)
    if not result or len(set(result)) != len(result):
        raise ValueError(f"{name} must be non-empty and duplicate-free")
    if any(Path(item).is_absolute() or ".." in Path(item).parts for item in result):
        raise ValueError(f"{name} must contain repository-relative paths")
    return result


def _validate_parse(path: Path) -> None:
    if path.suffix == ".py":
        ast.parse(path.read_text(), filename=str(path))
    elif path.suffix in {".yaml", ".yml"}:
        if yaml.safe_load(path.read_text()) is None:
            raise ValueError(f"empty YAML artifact: {path}")
    elif path.suffix == ".json":
        json.loads(path.read_text())


def _module_files(module: str, *, repository_root: Path) -> set[Path]:
    parts = module.split(".")
    if not parts or parts[0] not in _LOCAL_PACKAGE_ROOTS:
        return set()
    source_root = repository_root / "src"
    stem = source_root.joinpath(*parts)
    candidates = (stem.with_suffix(".py"), stem / "__init__.py")
    result = {candidate for candidate in candidates if candidate.is_file()}
    for depth in range(1, len(parts) + 1):
        package_init = source_root.joinpath(*parts[:depth]) / "__init__.py"
        if package_init.is_file():
            result.add(package_init)
    return result


def _local_python_dependencies(
    path: Path, *, repository_root: Path
) -> set[Path]:
    tree = ast.parse(path.read_text(), filename=str(path))
    relative = path.relative_to(repository_root / "src")
    current_package = list(relative.parent.parts)
    dependencies: set[Path] = set()
    for node in ast.walk(tree):
        modules: list[str] = []
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                keep = len(current_package) - (node.level - 1)
                if keep < 0:
                    continue
                base = current_package[:keep]
                if node.module:
                    modules.append(".".join((*base, *node.module.split("."))))
                else:
                    modules.extend(
                        ".".join((*base, alias.name)) for alias in node.names
                    )
            elif node.module:
                modules.append(node.module)
                modules.extend(f"{node.module}.{alias.name}" for alias in node.names)
        for module in modules:
            dependencies.update(
                _module_files(module, repository_root=repository_root)
            )
    return dependencies


def _literal_repository_dependencies(
    path: Path, *, repository_root: Path
) -> set[Path]:
    tree = ast.parse(path.read_text(), filename=str(path))
    result = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        relative = node.value
        if _REPOSITORY_FILE_LITERAL.fullmatch(relative):
            candidate = repository_root / relative
            if candidate.is_file():
                result.add(candidate)
    return result


def _script_module_files(module: str, *, repository_root: Path) -> set[Path]:
    if not module or "." in module:
        return set()
    candidates = list((repository_root / "scripts").glob(f"**/{module}.py"))
    return set(candidates) if len(candidates) == 1 else set()


def _shell_repository_dependencies(
    path: Path, *, repository_root: Path
) -> set[Path]:
    result = set()
    for line in path.read_text().splitlines():
        if line.lstrip().startswith("#"):
            continue
        for relative in _SHELL_PROJECT_FILE.findall(line):
            dependency = repository_root / relative
            if dependency.is_file():
                result.add(dependency)
    return result


def _audit_dependency_closure(
    required: Sequence[str],
    *,
    repository_root: Path,
    exceptions: Mapping[str, str],
) -> list[str]:
    required_set = set(required)
    errors = []
    for relative in required:
        path = repository_root / relative
        if not path.is_file():
            continue
        if path.suffix == ".sh":
            dependencies = _shell_repository_dependencies(
                path, repository_root=repository_root
            )
            for dependency in sorted(dependencies):
                dependency_relative = str(dependency.relative_to(repository_root))
                if (
                    dependency_relative not in required_set
                    and dependency_relative not in exceptions
                ):
                    errors.append(
                        f"untracked claim-critical dependency: {relative} -> "
                        f"{dependency_relative}"
                    )
            continue
        if path.suffix != ".py":
            continue
        dependencies = set()
        if (repository_root / "src") in path.parents:
            dependencies.update(
                _local_python_dependencies(path, repository_root=repository_root)
            )
        else:
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                modules: list[str] = []
                if isinstance(node, ast.Import):
                    modules.extend(alias.name for alias in node.names)
                elif (
                    isinstance(node, ast.ImportFrom)
                    and node.level == 0
                    and node.module
                ):
                    modules.append(node.module)
                    modules.extend(
                        f"{node.module}.{alias.name}" for alias in node.names
                    )
                for module in modules:
                    dependencies.update(
                        _module_files(module, repository_root=repository_root)
                    )
                    dependencies.update(
                        _script_module_files(
                            module, repository_root=repository_root
                        )
                    )
        dependencies.update(
            _literal_repository_dependencies(path, repository_root=repository_root)
        )
        for dependency in sorted(dependencies):
            dependency_relative = str(dependency.relative_to(repository_root))
            if (
                dependency_relative not in required_set
                and dependency_relative not in exceptions
            ):
                errors.append(
                    f"untracked claim-critical dependency: {relative} -> "
                    f"{dependency_relative}"
                )
    return errors


def _dependency_exceptions(value: Any) -> dict[str, str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("dependency closure exceptions must be a sequence")
    result = {}
    for row in value:
        if not isinstance(row, Mapping):
            raise TypeError("dependency closure exception must be a mapping")
        path = str(row.get("path", ""))
        rationale = " ".join(str(row.get("rationale", "")).split())
        if (
            not path
            or Path(path).is_absolute()
            or ".." in Path(path).parts
            or not rationale
            or path in result
        ):
            raise ValueError("dependency closure exceptions are malformed")
        result[path] = rationale
    return result


def _inventory_digest(inventory: Sequence[Mapping[str, str]]) -> str:
    canonical = json.dumps(list(inventory), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def validate_repro_lock(
    lock_path: Path, *, manifest_path: Path, repository_root: Path
) -> dict[str, Any]:
    """Verify that a release lock still identifies every current offline file."""

    value = json.loads(lock_path.read_text())
    if value.get("schema_version") != "piu.offline-repro-audit.v1":
        raise ValueError("unsupported PIU offline reproduction lock")
    if value.get("offline_ready") is not True or value.get("errors") != []:
        raise ValueError("PIU offline reproduction lock is not ready")
    if value.get("local_gpu_actions_performed") is not False:
        raise ValueError("PIU offline reproduction lock performed local GPU actions")
    if value.get("manifest", {}).get("sha256") != sha256_file(manifest_path):
        raise ValueError("PIU reproduction manifest differs from the release lock")
    inventory = value.get("offline_inventory")
    if not isinstance(inventory, list) or not inventory:
        raise ValueError("PIU reproduction lock has no offline inventory")
    for row in inventory:
        if not isinstance(row, Mapping):
            raise TypeError("PIU reproduction inventory row must be a mapping")
        relative = str(row.get("path", ""))
        path = repository_root / relative
        if not relative or not path.is_file() or sha256_file(path) != row.get("sha256"):
            raise ValueError(f"offline release file differs from lock: {relative}")
    if value.get("offline_inventory_sha256") != _inventory_digest(inventory):
        raise ValueError("PIU reproduction inventory digest differs")
    return value


def audit_repro_manifest(
    manifest_path: Path,
    *,
    repository_root: Path,
    reference_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = yaml.safe_load(manifest_path.read_text())
    if not isinstance(manifest, Mapping):
        raise ValueError("unsupported PIU reproducibility manifest")
    if manifest.get("schema_version") == "piu.offline-repro-manifest-extension.v2":
        base_relative = str(manifest.get("base_manifest", ""))
        base_path = repository_root / base_relative
        base = yaml.safe_load(base_path.read_text())
        if (
            not isinstance(base, Mapping)
            or base.get("schema_version") != "piu.offline-repro-manifest.v1"
        ):
            raise ValueError("v2 offline extension lacks a valid retained v1 base")
        if manifest.get("resource_contract") != base.get("resource_contract"):
            raise ValueError("v2 offline extension rewrites the retained resource contract")
        if manifest.get("claim_contract") != base.get("claim_contract"):
            raise ValueError("v2 offline extension weakens the retained claim contract")
        manifest = {
            **base,
            "schema_version": "piu.offline-repro-manifest.v1",
            "id": str(manifest.get("id", "")),
            "status": str(manifest.get("status", "")),
            "empirical_stage_dag": str(manifest.get("empirical_stage_dag", "")),
            "dependency_closure_exceptions": [
                *base.get("dependency_closure_exceptions", ()),
                *manifest.get("added_dependency_closure_exceptions", ()),
            ],
            "required_files": [
                *base.get("required_files", ()),
                *manifest.get("added_required_files", ()),
            ],
            "python_entrypoints": [
                *base.get("python_entrypoints", ()),
                *manifest.get("added_python_entrypoints", ()),
            ],
            "external_empirical_gates": [
                *base.get("external_empirical_gates", ()),
                *manifest.get("added_external_empirical_gates", ()),
            ],
        }
    elif manifest.get("schema_version") != "piu.offline-repro-manifest.v1":
        raise ValueError("unsupported PIU reproducibility manifest")
    resource = manifest.get("resource_contract", {})
    if int(resource.get("local_gpu_memory_mib_max", -1)) != 1500:
        raise ValueError("repro manifest must preserve the 1500 MiB GPU cap")
    if resource.get("local_pi05_checkpoint_load_allowed") is not False:
        raise ValueError("local pi0.5 checkpoint loading must remain forbidden")
    if resource.get("pi05_execution_location") != "identified_external_server_only":
        raise ValueError("pi0.5 execution must remain external-only")
    claim = manifest.get("claim_contract", {})
    if any(
        claim.get(name) is not False
        for name in (
            "synthetic_tests_are_performance_evidence",
            "missing_external_gate_can_be_imputed",
            "old_256_pixel_labels_can_train_main_method",
            "unsealed_results_can_enter_main_table",
            "unqualified_primitive_can_dispatch",
        )
    ):
        raise ValueError("repro manifest weakens a claim firewall")
    required = _files(manifest.get("required_files"), name="required files")
    dependency_exceptions = _dependency_exceptions(
        manifest.get("dependency_closure_exceptions", ())
    )
    inventory = []
    errors = []
    for relative in sorted(required):
        path = repository_root / relative
        if not path.is_file():
            errors.append(f"missing required file: {relative}")
            continue
        try:
            _validate_parse(path)
        except (SyntaxError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"invalid required file {relative}: {exc}")
            continue
        inventory.append(
            {
                "path": relative,
                "sha256": sha256_file(path),
                "bytes": str(path.stat().st_size),
            }
        )
    errors.extend(
        _audit_dependency_closure(
            required,
            repository_root=repository_root,
            exceptions=dependency_exceptions,
        )
    )
    entrypoints = _files(manifest.get("python_entrypoints"), name="Python entrypoints")
    entrypoint_rows = []
    for relative in entrypoints:
        path = repository_root / relative
        executable = path.is_file() and os.access(path, os.X_OK)
        syntax_valid = False
        if path.is_file():
            try:
                ast.parse(path.read_text(), filename=str(path))
                syntax_valid = True
            except SyntaxError as exc:
                errors.append(f"invalid entrypoint {relative}: {exc}")
        if not executable:
            errors.append(f"entrypoint is not executable: {relative}")
        entrypoint_rows.append(
            {"path": relative, "executable": executable, "syntax_valid": syntax_valid}
        )
    external_rows = []
    for gate in manifest.get("external_empirical_gates", ()):
        relative = str(gate["artifact"])
        path = repository_root / relative
        external_rows.append(
            {
                "id": str(gate["id"]),
                "artifact": relative,
                "status": (
                    "PRESENT_UNVERIFIED" if path.is_file() else "PENDING_EXTERNAL"
                ),
                "sha256": sha256_file(path) if path.is_file() else None,
            }
        )
    empirical_dag = None
    dag_relative = manifest.get("empirical_stage_dag")
    if dag_relative is not None:
        dag_path = repository_root / str(dag_relative)
        if not dag_path.is_file():
            errors.append(f"missing empirical stage DAG: {dag_relative}")
        else:
            try:
                from .empirical_dag import evaluate_empirical_dag

                empirical_dag = evaluate_empirical_dag(
                    dag_path, repository_root=repository_root
                )
            except Exception as exc:
                errors.append(f"invalid empirical stage DAG: {exc}")
    inventory_sha = _inventory_digest(inventory)
    reference_match: bool | None = None
    if reference_report is not None:
        expected = str(reference_report.get("offline_inventory_sha256", ""))
        reference_match = bool(expected) and expected == inventory_sha
        if not reference_match:
            errors.append("offline inventory differs from the reference release")
    offline_ready = not errors and len(inventory) == len(required)
    empirical_ready = bool(
        offline_ready
        and empirical_dag is not None
        and empirical_dag.get("empirical_pipeline_complete") is True
    )
    paper_claim_ready = bool(
        empirical_ready and empirical_dag.get("paper_claim_ready") is True
    )
    return {
        "schema_version": "piu.offline-repro-audit.v1",
        "manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "mode": (
            "VERIFY_REFERENCE"
            if reference_report is not None
            else "CREATE_RELEASE_LOCK"
        ),
        "offline_ready": offline_ready,
        "empirical_ready": empirical_ready,
        "paper_claim_ready": paper_claim_ready,
        "paper_claim_ready_reason": (
            "the machine-verified empirical DAG and offline release are complete"
            if paper_claim_ready
            else (
                "external artifacts require the empirical DAG's identity, split, "
                "authorization, hash-chain, and evaluator validation"
            )
        ),
        "local_gpu_actions_performed": False,
        "offline_inventory": inventory,
        "offline_inventory_sha256": inventory_sha,
        "dependency_closure_exceptions": dependency_exceptions,
        "reference_inventory_match": reference_match,
        "entrypoints": entrypoint_rows,
        "external_empirical_gates": external_rows,
        "empirical_stage_dag": empirical_dag,
        "errors": errors,
    }
