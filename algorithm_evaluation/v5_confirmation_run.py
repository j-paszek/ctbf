"""Run the frozen seven-method portfolio on a CTBF v5 confirmation bank."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence

import networkx as nx

from algorithm_evaluation.paper_pipeline_contract import (
    aligned_distance_submatrix,
    fixed_label_ad_f1,
    file_sha256,
    read_json,
)
from algorithm_evaluation.paper_pipeline_runner import (
    deserialize_tree,
    measured_stage,
    serialize_tree,
    validate_reconstruction_input,
)
from algorithm_evaluation.process_isolation import (
    CASE_ARM_WORKER_UNIT,
    FreshSpawnTaskPool,
    fresh_process_contract,
)
from algorithm_evaluation.v5_algorithm_development_common import (
    ARM_SPEC_BY_ID,
    DevelopmentArmSpec,
    observed_labels,
    reconstruct_development_arm,
    tree_summary,
    truth_prefix,
)
from algorithm_evaluation.v5_algorithm_development_run import run_semantic_gate
from algorithm_evaluation.v5_confirmation_bank import (
    TRUTH_SCHEMA_VERSION,
    deserialize_confirmation_distance,
    load_confirmation_bank,
)
from algorithm_evaluation.v5_confirmation_common import (
    BANK_CHECKSUM_NAME,
    BIOPSY_COUNT_CONDITION_IDS,
    DEFAULT_MANIFEST_PATH,
    PAPER_ARM_IDS,
    RUN_CHECKSUM_NAME,
    RUN_MANIFEST_NAME,
    RUN_SCHEMA_VERSION,
    ensure_new_output_root,
    record_path,
    validate_checksum_closure,
    validate_internal_runtime,
    validate_manifest,
    write_checksum_closure,
    write_json,
)
from ctbs import DistanceMatrix
from distance_semantics import validate_distance_label_coverage, validate_distance_matrix
from evaluation_contract import evaluate_tree_pair_result, validate_evaluation_result


RECORD_SCHEMA_VERSION = "ctbf-v5-confirmatory-arm-record-v1"
DEFAULT_RECONSTRUCTION_TIMEOUT_SECONDS = 300
DEFAULT_EVALUATION_TIMEOUT_SECONDS = 300
DEFAULT_RSS_LIMIT_BYTES = 4 * 1024**3
DEFAULT_RECORD_WORKERS = 8
MAX_RECORD_WORKERS = 8
WORKER_GRACE_SECONDS = 90


def _typed_error(error: BaseException, stage: str) -> dict[str, Any]:
    return {
        "stage": stage,
        "type": type(error).__name__,
        "message": str(error)[:4096],
    }


def _resource_audit_error(resources: Mapping[str, Any], stage: str) -> RuntimeError | None:
    memory = resources.get("memory")
    if not isinstance(memory, Mapping) or memory.get("peak_rss_bytes") is None:
        return RuntimeError(f"{stage} has no auditable process-tree peak RSS.")
    return None


def _record_base(case: Mapping[str, Any], spec: DevelopmentArmSpec) -> dict[str, Any]:
    return {
        "schema_version": RECORD_SCHEMA_VERSION,
        "case_id": str(case["case_id"]),
        "block_index": int(case["block_index"]),
        "condition_id": str(case["condition_id"]),
        "condition_family": str(case["condition_family"]),
        "height": int(case["height"]),
        "generations": [int(value) for value in case["generations"]],
        "arm_id": spec.arm_id,
        "family": spec.family,
        "problem": spec.problem,
        "primary_metric": spec.primary_metric,
    }


def _dependency_record(
    case: Mapping[str, Any],
    spec: DevelopmentArmSpec,
    *,
    dependency: str,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        **_record_base(case, spec),
        "status": "not_run_dependency",
        "failure": {
            "stage": "dependency",
            "type": "DependencyFailure",
            "message": f"Record was not run because {dependency} was unavailable.",
            "dependency": dependency,
            "details": None if details is None else dict(details),
        },
        "metrics": None,
        "native_evaluation": None,
        "fixed_label_evaluation": None,
        "observation_coverage": None,
        "tree": None,
        "tree_summary": None,
        "reconstruction_metadata": None,
        "resources": {"reconstruction": None, "evaluation": None},
    }


def _failure_record(
    case: Mapping[str, Any],
    spec: DevelopmentArmSpec,
    *,
    error: BaseException,
    stage: str,
    reconstruction_resources: Mapping[str, Any] | None,
    evaluation_resources: Mapping[str, Any] | None,
    tree: nx.DiGraph | None = None,
    reconstruction_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        **_record_base(case, spec),
        "status": "failure",
        "failure": _typed_error(error, stage),
        "metrics": None,
        "native_evaluation": None,
        "fixed_label_evaluation": None,
        "observation_coverage": None,
        "tree": None if tree is None else serialize_tree(tree),
        "tree_summary": None if tree is None else tree_summary(tree),
        "reconstruction_metadata": (
            None if reconstruction_metadata is None else dict(reconstruction_metadata)
        ),
        "resources": {
            "reconstruction": reconstruction_resources,
            "evaluation": evaluation_resources,
        },
    }


def _load_observable_assets(
    bank_root: Path,
    case: Mapping[str, Any],
) -> tuple[dict[str, Any], DistanceMatrix, dict[str, Any]]:
    payload = read_json(bank_root / str(case["input_path"]))
    validate_reconstruction_input(payload)
    if payload.get("case_id") != case.get("case_id"):
        raise ValueError("Stored reconstruction-input case id changed.")
    metadata = read_json(bank_root / str(case["metadata_path"]))
    if metadata.get("case_id") != case.get("case_id"):
        raise ValueError("Stored case-metadata id changed.")
    maximum = deserialize_confirmation_distance(
        read_json(bank_root / str(case["distance_path"]))
    )
    labels = observed_labels(payload)
    ids, matrix = aligned_distance_submatrix(maximum.ids, maximum.matrix, labels)
    validate_distance_matrix(ids, matrix)
    validate_distance_label_coverage(ids, labels, allow_extra=False)
    return (
        payload,
        DistanceMatrix(ids=ids, matrix=matrix, provenance=maximum.provenance),
        metadata,
    )


def _load_truth_after_reconstruction(
    bank_root: Path,
    case: Mapping[str, Any],
) -> nx.DiGraph:
    payload = read_json(bank_root / str(case["truth_path"]))
    if payload.get("schema_version") != TRUTH_SCHEMA_VERSION:
        raise ValueError("Unknown confirmation truth schema.")
    if payload.get("block_index") != int(case["block_index"]):
        raise ValueError("Stored confirmation truth block changed.")
    return truth_prefix(deserialize_tree(payload["tree"]), int(case["height"]))


def _evaluate_bundle(
    truth: nx.DiGraph,
    reconstruction: nx.DiGraph,
    labels: Sequence[Any],
    fixed_labels: Sequence[Any] | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    native = evaluate_tree_pair_result(truth, reconstruction, labels)
    fixed = (
        None
        if fixed_labels is None
        else fixed_label_ad_f1(truth, reconstruction, fixed_labels)
    )
    return native, fixed


def _execute_record(
    bank_root: str,
    case: Mapping[str, Any],
    spec: DevelopmentArmSpec,
    fixed_label_reference_path: str | None,
    reconstruction_timeout_seconds: int,
    evaluation_timeout_seconds: int,
    rss_limit_bytes: int,
) -> dict[str, Any]:
    root = Path(bank_root)
    try:
        reconstruction_input, distance, metadata = _load_observable_assets(root, case)
    except Exception as error:
        return _failure_record(
            case,
            spec,
            error=error,
            stage="input",
            reconstruction_resources=None,
            evaluation_resources=None,
        )
    reconstruction, reconstruction_resources, reconstruction_error = measured_stage(
        lambda: reconstruct_development_arm(
            spec,
            reconstruction_input,
            distance,
            reconstruction_seed=int(metadata["reconstruction_seed"]),
        ),
        timeout_seconds=reconstruction_timeout_seconds,
        rss_limit_bytes=rss_limit_bytes,
    )
    if reconstruction_error is None:
        reconstruction_error = _resource_audit_error(
            reconstruction_resources,
            "reconstruction",
        )
    if reconstruction_error is not None or reconstruction is None:
        return _failure_record(
            case,
            spec,
            error=reconstruction_error or RuntimeError("Reconstruction returned no result."),
            stage="reconstruction",
            reconstruction_resources=reconstruction_resources,
            evaluation_resources=None,
        )
    tree, _levels, returned_root, reconstruction_metadata = reconstruction
    if not nx.is_arborescence(tree):
        return _failure_record(
            case,
            spec,
            error=ValueError("Reconstruction is not a directed arborescence."),
            stage="reconstruction_validation",
            reconstruction_resources=reconstruction_resources,
            evaluation_resources=None,
            tree=tree,
            reconstruction_metadata=reconstruction_metadata,
        )

    # Truth is intentionally unavailable until reconstruction has terminated.
    try:
        truth = _load_truth_after_reconstruction(root, case)
        labels = observed_labels(reconstruction_input)
        fixed_labels = None
        if fixed_label_reference_path is not None and spec.problem != "partial":
            fixed_input = read_json(root / fixed_label_reference_path)
            validate_reconstruction_input(fixed_input)
            fixed_labels = observed_labels(fixed_input)
        evaluations, evaluation_resources, evaluation_error = measured_stage(
            lambda: _evaluate_bundle(truth, tree, labels, fixed_labels),
            timeout_seconds=evaluation_timeout_seconds,
            rss_limit_bytes=rss_limit_bytes,
        )
        if evaluation_error is None:
            evaluation_error = _resource_audit_error(
                evaluation_resources,
                "evaluation",
            )
        if evaluation_error is not None or evaluations is None:
            raise evaluation_error or RuntimeError("Evaluation returned no result.")
        native, fixed = evaluations
        validate_evaluation_result(native)
        if native.get("status") != "success":
            raise ValueError(f"Native evaluator failed: {native.get('failure')}.")
        coverage = native["inputs"]["observation_label_coverage"]
        if float(coverage["fraction"]) < 1.0:
            raise ValueError("Reconstructed observation-label coverage is below one.")
        if fixed is not None and fixed.get("status") != "success":
            raise ValueError(f"Fixed-label evaluator failed: {fixed.get('failure')}.")
    except Exception as error:
        return _failure_record(
            case,
            spec,
            error=error,
            stage="evaluation",
            reconstruction_resources=reconstruction_resources,
            evaluation_resources=locals().get("evaluation_resources"),
            tree=tree,
            reconstruction_metadata=reconstruction_metadata,
        )

    metrics = native["metrics"]
    declared_metrics = {
        name: float(metrics[name])
        for name in (spec.primary_metric, *spec.complementary_metrics)
    }
    return {
        **_record_base(case, spec),
        "status": "success",
        "failure": None,
        "metrics": declared_metrics,
        "native_evaluation": native,
        "fixed_label_evaluation": fixed,
        "observation_coverage": {
            "required_unique_label_count": coverage["required_unique_label_count"],
            "reconstructed_unique_label_count": coverage[
                "reconstructed_unique_label_count"
            ],
            "fraction": float(coverage["fraction"]),
        },
        "tree": serialize_tree(tree),
        "tree_summary": tree_summary(tree),
        "reconstruction_metadata": {
            **dict(reconstruction_metadata),
            "returned_root": returned_root,
            "truth_loaded_only_after_reconstruction": True,
        },
        "resources": {
            "reconstruction": reconstruction_resources,
            "evaluation": evaluation_resources,
        },
    }


def _execute_declared_task(
    ordinal: int,
    mode: str,
    case: Mapping[str, Any],
    spec: DevelopmentArmSpec,
    context: Any,
    arguments: Sequence[Any] | None,
) -> tuple[int, dict[str, Any]]:
    """Execute one picklable fresh-worker task or materialize its dependency."""
    if mode == "dependency":
        record = _dependency_record(
            case,
            spec,
            dependency="bank_case",
            details=context if isinstance(context, Mapping) else None,
        )
    elif mode == "semantic_gate":
        record = _dependency_record(
            case,
            spec,
            dependency="semantic_gate",
            details=context if isinstance(context, Mapping) else None,
        )
    else:
        if arguments is None:  # pragma: no cover - guarded by declaration
            raise RuntimeError("Executable record has no arguments.")
        record = _execute_record(*arguments)
    return ordinal, record


def _declared_cases(bank: Mapping[str, Any]) -> list[dict[str, Any]]:
    available = {str(case["case_id"]): dict(case) for case in bank["cases"]}
    result = []
    for declaration in bank["condition_inventory"]:
        case = dict(declaration)
        if case["status"] == "available":
            case.update(available[str(case["case_id"])])
        result.append(case)
    return sorted(
        result,
        key=lambda case: (int(case["block_index"]), int(case["condition_ordinal"])),
    )


def _new_run_manifest(
    *,
    validation: Mapping[str, Any],
    bank_root: Path,
    bank: Mapping[str, Any],
    technical_smoke: bool,
    resources: Mapping[str, Any],
    semantic_gate: Mapping[str, Any],
    internal_runtime: Mapping[str, Any],
    created_at_utc: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "run_id": (
            "ctbf-v5-g2-01a-technical-smoke-run-v1"
            if technical_smoke
            else "ctbf-v5-g2-01a-independent-confirmation-run-v1"
        ),
        "status": "in_progress",
        "scientific_role": (
            "technical_smoke_not_accuracy_evidence"
            if technical_smoke
            else "prospectively_locked_independent_confirmation"
        ),
        "created_at_utc": created_at_utc or datetime.now(timezone.utc).isoformat(),
        "design_manifest_sha256": validation["manifest_sha256"],
        "protocol_sha256": validation["protocol_sha256"],
        "source_lock_sha256": validation["source_lock"].get("sha256"),
        "bank_id": bank["bank_id"],
        "bank_root": str(bank_root),
        "bank_checksum_sha256": bank["checksum_closure"]["sha256"],
        "arm_ids": list(PAPER_ARM_IDS),
        "arm_specs": [ARM_SPEC_BY_ID[arm_id].as_record() for arm_id in PAPER_ARM_IDS],
        "declared_condition_count": int(bank["declared_condition_count"]),
        "expected_record_count": int(bank["declared_condition_count"]) * len(PAPER_ARM_IDS),
        "resource_contract": dict(resources),
        "internal_runtime": dict(internal_runtime),
        "record_execution": fresh_process_contract(CASE_ARM_WORKER_UNIT),
        "truth_available_to_reconstruction": False,
        "semantic_gate_by_arm": dict(semantic_gate),
        "completed_record_count": 0,
        "record_inventory": [],
        "status_counts": {},
    }


def _resume_run(
    root: Path,
    *,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    path = root / RUN_MANIFEST_NAME
    if not path.is_file():
        raise ValueError("A resumed confirmation run requires run_manifest.json.")
    run = read_json(path)
    for field in (
        "schema_version",
        "run_id",
        "scientific_role",
        "design_manifest_sha256",
        "protocol_sha256",
        "source_lock_sha256",
        "bank_id",
        "bank_root",
        "bank_checksum_sha256",
        "arm_ids",
        "arm_specs",
        "declared_condition_count",
        "expected_record_count",
        "resource_contract",
        "internal_runtime",
        "record_execution",
        "truth_available_to_reconstruction",
        "semantic_gate_by_arm",
    ):
        if run.get(field) != expected.get(field):
            raise ValueError(f"Cannot resume because stored {field} changed.")
    if run.get("status") not in {"in_progress", "interrupted", "failure"}:
        raise ValueError("Only an incomplete confirmation run can resume.")
    inventory = run.get("record_inventory")
    if not isinstance(inventory, list):
        raise ValueError("Stored confirmation record inventory is invalid.")
    for record in inventory:
        path = root / str(record.get("path"))
        if not path.is_file() or file_sha256(path) != record.get("sha256"):
            raise ValueError("A stored confirmation record changed before resume.")
    run["status"] = "in_progress"
    run.pop("runner_failure", None)
    run.setdefault("resume_history", []).append(
        {
            "resumed_at_utc": datetime.now(timezone.utc).isoformat(),
            "preserved_record_count": len(inventory),
        }
    )
    write_json(root / RUN_MANIFEST_NAME, run)
    return run


def run_confirmation(
    *,
    bank_root: Path | str,
    output_root: Path | str,
    manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
    record_workers: int = DEFAULT_RECORD_WORKERS,
    reconstruction_timeout_seconds: int = DEFAULT_RECONSTRUCTION_TIMEOUT_SECONDS,
    evaluation_timeout_seconds: int = DEFAULT_EVALUATION_TIMEOUT_SECONDS,
    rss_limit_bytes: int = DEFAULT_RSS_LIMIT_BYTES,
    technical_smoke: bool = False,
    resume: bool = False,
    progress: bool = False,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    if not 1 <= record_workers <= MAX_RECORD_WORKERS:
        raise ValueError(f"record_workers must be in [1,{MAX_RECORD_WORKERS}].")
    for name, value in (
        ("reconstruction_timeout_seconds", reconstruction_timeout_seconds),
        ("evaluation_timeout_seconds", evaluation_timeout_seconds),
        ("rss_limit_bytes", rss_limit_bytes),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    validation = validate_manifest(
        manifest_path,
        require_approved=not technical_smoke,
    )
    internal_runtime = validate_internal_runtime(validation["manifest"])
    bank_path, bank = load_confirmation_bank(
        bank_root,
        manifest_path=manifest_path,
        allow_smoke=technical_smoke,
    )
    is_smoke_bank = bank["scientific_role"] == "technical_smoke_not_accuracy_evidence"
    if is_smoke_bank != technical_smoke:
        raise ValueError("Run kind does not match the supplied bank scientific role.")
    resources = {
        "reconstruction_timeout_seconds": reconstruction_timeout_seconds,
        "evaluation_timeout_seconds": evaluation_timeout_seconds,
        "rss_limit_bytes": rss_limit_bytes,
        "requested_record_workers": record_workers,
    }
    specs = tuple(ARM_SPEC_BY_ID[arm_id] for arm_id in PAPER_ARM_IDS)
    root = Path(output_root).expanduser().resolve()
    if resume:
        if not root.is_dir():
            raise ValueError("A resumed confirmation run requires an existing root.")
        stored = read_json(root / RUN_MANIFEST_NAME)
        semantic_gate = stored.get("semantic_gate_by_arm")
        if not isinstance(semantic_gate, Mapping):
            raise ValueError("Stored confirmation run lacks its semantic gate.")
        expected = _new_run_manifest(
            validation=validation,
            bank_root=bank_path,
            bank=bank,
            technical_smoke=technical_smoke,
            resources=resources,
            semantic_gate=semantic_gate,
            internal_runtime=internal_runtime,
            created_at_utc=stored.get("created_at_utc"),
        )
        run = _resume_run(root, expected=expected)
    else:
        root = ensure_new_output_root(root)
        shutil.copyfile(validation["manifest_path"], root / "design_manifest.snapshot.json")
        semantic_gate = run_semantic_gate(
            specs,
            timeout_seconds=reconstruction_timeout_seconds,
            rss_limit_bytes=rss_limit_bytes,
        )
        run = _new_run_manifest(
            validation=validation,
            bank_root=bank_path,
            bank=bank,
            technical_smoke=technical_smoke,
            resources=resources,
            semantic_gate=semantic_gate,
            internal_runtime=internal_runtime,
            created_at_utc=created_at_utc,
        )
        write_json(root / RUN_MANIFEST_NAME, run)

    cases = _declared_cases(bank)
    expected_keys = [
        (str(case["case_id"]), spec.arm_id)
        for case in cases
        for spec in specs
    ]
    completed = [
        (str(record["case_id"]), str(record["arm_id"]))
        for record in run["record_inventory"]
    ]
    if completed != expected_keys[: len(completed)]:
        raise ValueError("Stored record inventory is not the exact declared prefix.")
    start = len(completed)
    case_by_block_condition = {
        (int(case["block_index"]), str(case["condition_id"])): case
        for case in cases
    }

    def pending_arguments():
        for ordinal, (case, spec) in enumerate(
            (pair for case in cases for pair in ((case, spec) for spec in specs))
        ):
            if ordinal < start:
                continue
            gate = semantic_gate[spec.arm_id]
            if case.get("status") != "available":
                yield (ordinal, "dependency", case, spec, case.get("failure"), None)
                continue
            if gate.get("status") != "passed":
                yield (ordinal, "semantic_gate", case, spec, gate, None)
                continue
            fixed_reference = None
            if case["condition_id"] in BIOPSY_COUNT_CONDITION_IDS:
                reference = case_by_block_condition[
                    (int(case["block_index"]), "H24_L2")
                ]
                if reference.get("status") == "available":
                    fixed_reference = str(reference["input_path"])
            yield (
                ordinal,
                "execute",
                case,
                spec,
                fixed_reference,
                (
                    str(bank_path),
                    case,
                    spec,
                    fixed_reference,
                    reconstruction_timeout_seconds,
                    evaluation_timeout_seconds,
                    rss_limit_bytes,
                ),
            )

    try:
        if start < len(expected_keys):
            with FreshSpawnTaskPool(record_workers, buffer_factor=2) as pool:
                for ordinal, record in pool.map_ordered(
                    _execute_declared_task,
                    pending_arguments(),
                    timeout_seconds=(
                        reconstruction_timeout_seconds
                        + evaluation_timeout_seconds
                        + WORKER_GRACE_SECONDS
                    ),
                ):
                    if ordinal != len(run["record_inventory"]):
                        raise RuntimeError("Parallel results changed declared record order.")
                    relative = record_path(record["case_id"], record["arm_id"])
                    write_json(root / relative, record)
                    run["record_inventory"].append(
                        {
                            "case_id": record["case_id"],
                            "arm_id": record["arm_id"],
                            "status": record["status"],
                            "path": relative,
                            "sha256": file_sha256(root / relative),
                        }
                    )
                    run["completed_record_count"] = len(run["record_inventory"])
                    write_json(root / RUN_MANIFEST_NAME, run)
                    if progress:
                        print(
                            f"{record['case_id']} {record['arm_id']}: {record['status']}",
                            file=sys.stderr,
                            flush=True,
                        )
    except BaseException as error:
        run["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failure"
        run["runner_failure"] = _typed_error(error, "run_runner")
        write_json(root / RUN_MANIFEST_NAME, run)
        raise

    status_counts: dict[str, int] = {}
    for record in run["record_inventory"]:
        status = str(record["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    run["status_counts"] = status_counts
    if validate_internal_runtime(validation["manifest"]) != internal_runtime:
        raise RuntimeError("Internal runtime identity changed during reconstruction.")
    run["status"] = "complete"
    run["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    run.pop("runner_failure", None)
    write_json(root / RUN_MANIFEST_NAME, run)
    write_checksum_closure(root, RUN_CHECKSUM_NAME)
    return run


def load_confirmation_run(
    run_root: Path | str,
    *,
    manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
    allow_smoke: bool = False,
) -> tuple[Path, dict[str, Any]]:
    root = Path(run_root).expanduser().resolve()
    run = read_json(root / RUN_MANIFEST_NAME)
    if run.get("schema_version") != RUN_SCHEMA_VERSION or run.get("status") != "complete":
        raise ValueError("Confirmation run is not complete under the current schema.")
    is_smoke = run.get("scientific_role") == "technical_smoke_not_accuracy_evidence"
    if is_smoke and not allow_smoke:
        raise ValueError("Technical-smoke output cannot enter confirmatory analysis.")
    validation = validate_manifest(
        manifest_path,
        require_approved=not is_smoke,
    )
    if run.get("design_manifest_sha256") != validation["manifest_sha256"]:
        raise ValueError("Confirmation run design digest changed.")
    validate_checksum_closure(Path(run["bank_root"]), BANK_CHECKSUM_NAME)
    closure = validate_checksum_closure(root, RUN_CHECKSUM_NAME)
    run["checksum_closure"] = closure
    expected = int(run["declared_condition_count"]) * len(PAPER_ARM_IDS)
    if run.get("expected_record_count") != expected:
        raise ValueError("Confirmation run expected-record count changed.")
    if len(run.get("record_inventory", ())) != expected:
        raise ValueError("Confirmation run record inventory is incomplete.")
    return root, run


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--record-workers", type=int, default=DEFAULT_RECORD_WORKERS)
    parser.add_argument(
        "--reconstruction-timeout-seconds",
        type=int,
        default=DEFAULT_RECONSTRUCTION_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--evaluation-timeout-seconds",
        type=int,
        default=DEFAULT_EVALUATION_TIMEOUT_SECONDS,
    )
    parser.add_argument("--rss-limit-bytes", type=int, default=DEFAULT_RSS_LIMIT_BYTES)
    parser.add_argument("--technical-smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = run_confirmation(
        bank_root=arguments.bank_root,
        output_root=arguments.output_root,
        manifest_path=arguments.manifest,
        record_workers=arguments.record_workers,
        reconstruction_timeout_seconds=arguments.reconstruction_timeout_seconds,
        evaluation_timeout_seconds=arguments.evaluation_timeout_seconds,
        rss_limit_bytes=arguments.rss_limit_bytes,
        technical_smoke=arguments.technical_smoke,
        resume=arguments.resume,
        progress=arguments.progress,
    )
    print(
        f"complete: {result['completed_record_count']} arm records; "
        f"{result['status_counts'].get('failure', 0)} failures; "
        f"{result['status_counts'].get('not_run_dependency', 0)} dependency failures"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "RECORD_SCHEMA_VERSION",
    "load_confirmation_run",
    "main",
    "run_confirmation",
]
