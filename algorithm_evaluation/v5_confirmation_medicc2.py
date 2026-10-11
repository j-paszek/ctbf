"""Run the frozen MEDICC2 contextual arm on standard confirmation cases."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence

import networkx as nx

from algorithm_evaluation.medicc2_reconstruction_preflight import (
    run_fixture_preflight,
)
from algorithm_evaluation.paper_pipeline_contract import file_sha256, read_json
from algorithm_evaluation.paper_pipeline_runner import deserialize_tree, measured_stage, serialize_tree
from algorithm_evaluation.process_isolation import (
    CASE_ARM_WORKER_UNIT,
    FreshSpawnPerTaskExecutor,
    FreshSpawnTaskPool,
    fresh_process_contract,
)
from algorithm_evaluation.v5_algorithm_development_common import (
    observed_labels,
    tree_summary,
    truth_prefix,
)
from algorithm_evaluation.v5_confirmation_bank import (
    TRUTH_SCHEMA_VERSION,
    load_confirmation_bank,
)
from algorithm_evaluation.v5_confirmation_common import (
    DEFAULT_MANIFEST_PATH,
    M2_ID,
    STANDARD_CONDITION_IDS,
    ensure_new_output_root,
    validate_internal_runtime,
    validate_manifest,
    write_checksum_closure,
    write_json,
)
from algorithm_evaluation.v5_medicc2_development_run import (
    MEDICC2_ARM_SPEC,
    MEDICC2_DEVELOPMENT_ARM_ID,
    Medicc2ExecutionError,
    _auditable_external_memory_error,
    _normalized_command_prefix,
    inspect_medicc2_runtime,
    reconstruct_medicc2_case,
)
from algorithm_evaluation.medicc2_reconstruction_preflight import inspect_source_checkout
from evaluation_contract import evaluate_tree_pair_result, validate_evaluation_result


GATE_SCHEMA_VERSION = "ctbf-v5-confirmatory-medicc2-resource-gate-v1"
RUN_SCHEMA_VERSION = "ctbf-v5-confirmatory-medicc2-run-v1"
RECORD_SCHEMA_VERSION = "ctbf-v5-confirmatory-medicc2-record-v1"
GATE_NAME = "resource_gate.json"
RUN_NAME = "run_manifest.json"
GATE_CHECKSUM_NAME = "gate_checksums.sha256"
RUN_CHECKSUM_NAME = "run_checksums.sha256"
DEFAULT_RECONSTRUCTION_TIMEOUT_SECONDS = 300
DEFAULT_EVALUATION_TIMEOUT_SECONDS = 300
DEFAULT_RSS_LIMIT_BYTES = 4 * 1024**3
DEFAULT_CAPTURE_LIMIT_BYTES = 1_048_576
DEFAULT_WORKERS = 4
MAX_WORKERS = 4
WORKER_GRACE_SECONDS = 120


def _typed_error(error: BaseException, stage: str) -> dict[str, Any]:
    if isinstance(error, Medicc2ExecutionError):
        failure = error.record.get("failure") or {}
        return {
            "stage": stage,
            "type": "Medicc2ExecutionError",
            "code": failure.get("code", "external_failure"),
            "message": str(failure.get("message") or error)[:4096],
            "external_execution": error.record,
        }
    return {
        "stage": stage,
        "type": type(error).__name__,
        "message": str(error)[:4096],
    }


def _standard_cases(bank: Mapping[str, Any]) -> list[dict[str, Any]]:
    available = {str(case["case_id"]): dict(case) for case in bank["cases"]}
    cases = []
    for declaration in bank["condition_inventory"]:
        if declaration["condition_id"] not in STANDARD_CONDITION_IDS:
            continue
        case = dict(declaration)
        if case["status"] == "available":
            case.update(available[str(case["case_id"])])
        cases.append(case)
    return sorted(
        cases,
        key=lambda case: (
            int(case["block_index"]),
            STANDARD_CONDITION_IDS.index(str(case["condition_id"])),
        ),
    )


def _observable_occurrence_count(bank_root: Path, case: Mapping[str, Any]) -> int:
    payload = read_json(bank_root / str(case["input_path"]))
    return sum(len(level["states"]) for level in payload["levels"])


def _gate_cases(bank_root: Path, bank: Mapping[str, Any]) -> list[dict[str, Any]]:
    available = [case for case in _standard_cases(bank) if case["status"] == "available"]
    if not available:
        raise ValueError("No standard confirmation case is available for MEDICC2 gating.")
    ranked = sorted(
        available,
        key=lambda case: (
            _observable_occurrence_count(bank_root, case),
            str(case["case_id"]),
        ),
    )
    selected = [ranked[0]]
    if ranked[-1]["case_id"] != ranked[0]["case_id"]:
        selected.append(ranked[-1])
    return selected


def _run_gate_case(
    bank_root: str,
    case: Mapping[str, Any],
    case_root: str,
    command_prefix: Sequence[str],
    timeout_seconds: int,
    rss_limit_bytes: int,
    capture_limit_bytes: int,
) -> dict[str, Any]:
    root = Path(bank_root)
    payload = read_json(root / str(case["input_path"]))
    try:
        tree, _returned_root, metadata, resources = reconstruct_medicc2_case(
            payload,
            case_root,
            command_prefix=command_prefix,
            timeout_seconds=timeout_seconds,
            rss_limit_bytes=rss_limit_bytes,
            capture_limit_bytes=capture_limit_bytes,
        )
        memory_error = _auditable_external_memory_error(resources)
        if memory_error is not None:
            raise memory_error
    except BaseException as error:
        return {
            "case_id": case["case_id"],
            "status": "failure",
            "failure": _typed_error(error, "resource_gate"),
            "resources": locals().get("resources"),
        }
    return {
        "case_id": case["case_id"],
        "status": "success",
        "failure": None,
        "observable_occurrence_count": _observable_occurrence_count(root, case),
        "tree_summary": tree_summary(tree),
        "occurrence_input": metadata["occurrence_input"],
        "resources": resources,
        "accuracy_evaluated": False,
        "truth_loaded": False,
    }


def run_resource_gate(
    *,
    bank_root: Path | str,
    output_root: Path | str,
    command_prefix: Sequence[Path | str],
    source_root: Path | str,
    manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
    reconstruction_timeout_seconds: int = DEFAULT_RECONSTRUCTION_TIMEOUT_SECONDS,
    rss_limit_bytes: int = DEFAULT_RSS_LIMIT_BYTES,
    capture_limit_bytes: int = DEFAULT_CAPTURE_LIMIT_BYTES,
) -> dict[str, Any]:
    validation = validate_manifest(manifest_path, require_approved=True)
    internal_runtime = validate_internal_runtime(validation["manifest"])
    bank_path, bank = load_confirmation_bank(bank_root, manifest_path=manifest_path)
    source = inspect_source_checkout(source_root)
    runtime = inspect_medicc2_runtime(command_prefix, source)
    normalized_command, _interpreter = _normalized_command_prefix(command_prefix)
    root = ensure_new_output_root(output_root)
    preflight = run_fixture_preflight(
        command_prefix=normalized_command,
        source_root=source_root,
        output_root=root / "semantic_preflight",
        timeout_seconds=reconstruction_timeout_seconds,
        rss_limit_bytes=rss_limit_bytes,
        capture_limit_bytes=capture_limit_bytes,
        require_recursive_rss=True,
    )
    selected = _gate_cases(bank_path, bank)
    records = []
    if preflight["status"] == "success":
        with FreshSpawnPerTaskExecutor() as executor:
            for case in selected:
                records.append(
                    executor.run(
                        _run_gate_case,
                        str(bank_path),
                        case,
                        str(root / "cases" / str(case["case_id"])),
                        normalized_command,
                        reconstruction_timeout_seconds,
                        rss_limit_bytes,
                        capture_limit_bytes,
                        timeout_seconds=reconstruction_timeout_seconds
                        + WORKER_GRACE_SECONDS,
                    )
                )
    result = {
        "schema_version": GATE_SCHEMA_VERSION,
        "status": (
            "success"
            if preflight["status"] == "success"
            and len(records) == len(selected)
            and all(record["status"] == "success" for record in records)
            else "failure"
        ),
        "scientific_role": "output_blind_resource_gate_not_accuracy_evidence",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "design_manifest_sha256": validation["manifest_sha256"],
        "source_lock_sha256": validation["source_lock"]["sha256"],
        "bank_id": bank["bank_id"],
        "bank_root": str(bank_path),
        "bank_checksum_sha256": bank["checksum_closure"]["sha256"],
        "medicc2_source": source,
        "medicc2_runtime": runtime,
        "ctbf_internal_runtime": internal_runtime,
        "arm_spec": dict(MEDICC2_ARM_SPEC),
        "semantic_preflight": preflight,
        "selection": {
            "rule": "smallest_and_largest_observable_standard_case",
            "case_ids": [case["case_id"] for case in selected],
            "truth_or_accuracy_used": False,
        },
        "resources": {
            "reconstruction_timeout_seconds": reconstruction_timeout_seconds,
            "rss_limit_bytes": rss_limit_bytes,
            "capture_limit_bytes": capture_limit_bytes,
        },
        "records": records,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(root / GATE_NAME, result)
    write_checksum_closure(root, GATE_CHECKSUM_NAME)
    return result


def _load_gate(
    gate_path: Path | str,
    *,
    validation: Mapping[str, Any],
    bank: Mapping[str, Any],
    bank_root: Path,
    source: Mapping[str, Any],
    runtime: Mapping[str, Any],
    internal_runtime: Mapping[str, Any],
    resources: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    path = Path(gate_path).expanduser().resolve()
    root = path if path.is_dir() else path.parent
    record_path = root / GATE_NAME if path.is_dir() else path
    gate = read_json(record_path)
    if gate.get("schema_version") != GATE_SCHEMA_VERSION or gate.get("status") != "success":
        raise ValueError("MEDICC2 confirmation resource gate did not pass.")
    expected = {
        "design_manifest_sha256": validation["manifest_sha256"],
        "source_lock_sha256": validation["source_lock"]["sha256"],
        "bank_id": bank["bank_id"],
        "bank_root": str(bank_root),
        "bank_checksum_sha256": bank["checksum_closure"]["sha256"],
        "medicc2_source": source,
        "medicc2_runtime": runtime,
        "ctbf_internal_runtime": internal_runtime,
        "resources": resources,
    }
    for field, value in expected.items():
        if gate.get(field) != value:
            raise ValueError(f"MEDICC2 resource-gate {field} changed.")
    from algorithm_evaluation.v5_confirmation_common import validate_checksum_closure

    validate_checksum_closure(root, GATE_CHECKSUM_NAME)
    return record_path, gate


def _medicc2_failure_record(
    case: Mapping[str, Any],
    error: BaseException,
    *,
    stage: str,
    reconstruction_resources: Mapping[str, Any] | None,
    evaluation_resources: Mapping[str, Any] | None = None,
    tree: nx.DiGraph | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": RECORD_SCHEMA_VERSION,
        "case_id": case["case_id"],
        "block_index": int(case["block_index"]),
        "condition_id": case["condition_id"],
        "height": int(case["height"]),
        "arm_id": MEDICC2_DEVELOPMENT_ARM_ID,
        "problem": "partial",
        "primary_metric": "grf",
        "status": "failure",
        "failure": _typed_error(error, stage),
        "metrics": None,
        "native_evaluation": None,
        "tree": None if tree is None else serialize_tree(tree),
        "tree_summary": None if tree is None else tree_summary(tree),
        "resources": {
            "reconstruction": reconstruction_resources,
            "evaluation": evaluation_resources,
        },
    }


def _run_medicc2_record(
    bank_root: str,
    case: Mapping[str, Any],
    case_root: str,
    command_prefix: Sequence[str],
    reconstruction_timeout_seconds: int,
    evaluation_timeout_seconds: int,
    rss_limit_bytes: int,
    capture_limit_bytes: int,
) -> dict[str, Any]:
    bank_path = Path(bank_root)
    payload = read_json(bank_path / str(case["input_path"]))
    reconstruction_resources = None
    try:
        tree, _returned_root, metadata, reconstruction_resources = reconstruct_medicc2_case(
            payload,
            case_root,
            command_prefix=command_prefix,
            timeout_seconds=reconstruction_timeout_seconds,
            rss_limit_bytes=rss_limit_bytes,
            capture_limit_bytes=capture_limit_bytes,
        )
        memory_error = _auditable_external_memory_error(reconstruction_resources)
        if memory_error is not None:
            raise memory_error
    except BaseException as error:
        return _medicc2_failure_record(
            case,
            error,
            stage="reconstruction",
            reconstruction_resources=reconstruction_resources,
        )

    # The external program has terminated before truth is loaded.
    try:
        truth_payload = read_json(bank_path / str(case["truth_path"]))
        if truth_payload.get("schema_version") != TRUTH_SCHEMA_VERSION:
            raise ValueError("Unknown confirmation truth schema.")
        truth = truth_prefix(
            deserialize_tree(truth_payload["tree"]),
            int(case["height"]),
        )
        labels = observed_labels(payload)
        evaluation, evaluation_resources, evaluation_error = measured_stage(
            lambda: evaluate_tree_pair_result(truth, tree, labels),
            timeout_seconds=evaluation_timeout_seconds,
            rss_limit_bytes=rss_limit_bytes,
        )
        if evaluation_error is not None or evaluation is None:
            raise evaluation_error or RuntimeError("Evaluation returned no result.")
        validate_evaluation_result(evaluation)
        if evaluation["status"] != "success":
            raise ValueError(f"MEDICC2 evaluator failed: {evaluation.get('failure')}.")
        coverage = evaluation["inputs"]["observation_label_coverage"]
        if float(coverage["fraction"]) < 1.0:
            raise ValueError("MEDICC2 observation-label coverage is below one.")
        projected = [
            attributes.get("cell_id")
            for _node, attributes in tree.nodes(data=True)
            if attributes.get("cell_id") is not None
        ]
        occurrences = [
            state["state_label"]
            for level in payload["levels"]
            for state in level["states"]
        ]
        if Counter(projected) != Counter(occurrences):
            raise ValueError("MEDICC2 projection changed observation multiplicity.")
    except BaseException as error:
        return _medicc2_failure_record(
            case,
            error,
            stage="evaluation",
            reconstruction_resources=reconstruction_resources,
            evaluation_resources=locals().get("evaluation_resources"),
            tree=tree,
        )
    return {
        "schema_version": RECORD_SCHEMA_VERSION,
        "case_id": case["case_id"],
        "block_index": int(case["block_index"]),
        "condition_id": case["condition_id"],
        "height": int(case["height"]),
        "arm_id": MEDICC2_DEVELOPMENT_ARM_ID,
        "problem": "partial",
        "primary_metric": "grf",
        "status": "success",
        "failure": None,
        "metrics": {"grf": float(evaluation["metrics"]["grf"])},
        "native_evaluation": evaluation,
        "tree": serialize_tree(tree),
        "tree_summary": tree_summary(tree),
        "reconstruction_metadata": {
            **metadata,
            "truth_loaded_only_after_reconstruction": True,
        },
        "resources": {
            "reconstruction": reconstruction_resources,
            "evaluation": evaluation_resources,
        },
    }


def _dependency_record(case: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": RECORD_SCHEMA_VERSION,
        "case_id": case["case_id"],
        "block_index": int(case["block_index"]),
        "condition_id": case["condition_id"],
        "height": int(case["height"]),
        "arm_id": MEDICC2_DEVELOPMENT_ARM_ID,
        "problem": "partial",
        "primary_metric": "grf",
        "status": "not_run_dependency",
        "failure": {
            "stage": "dependency",
            "type": "BankCaseUnavailable",
            "message": "The standard confirmation case is unavailable.",
            "details": case.get("failure"),
        },
        "metrics": None,
        "native_evaluation": None,
        "tree": None,
        "tree_summary": None,
        "resources": {"reconstruction": None, "evaluation": None},
    }


def _execute_medicc2_task(
    ordinal: int,
    bank_root: str,
    run_root: str,
    case: Mapping[str, Any],
    command_prefix: Sequence[str],
    reconstruction_timeout_seconds: int,
    evaluation_timeout_seconds: int,
    rss_limit_bytes: int,
    capture_limit_bytes: int,
) -> tuple[int, dict[str, Any], str]:
    relative_root = (Path("cases") / str(case["case_id"])).as_posix()
    target = Path(run_root) / relative_root
    if case["status"] != "available":
        return ordinal, _dependency_record(case), relative_root
    record = _run_medicc2_record(
        bank_root,
        case,
        str(target),
        command_prefix,
        reconstruction_timeout_seconds,
        evaluation_timeout_seconds,
        rss_limit_bytes,
        capture_limit_bytes,
    )
    return ordinal, record, relative_root


def _quarantine_uncommitted_case_roots(
    root: Path,
    committed_case_ids: set[str],
) -> None:
    cases_root = root / "cases"
    if not cases_root.is_dir():
        return
    orphan_root = root / "orphaned_interrupted_attempts"
    for path in sorted(cases_root.iterdir(), key=lambda value: value.name):
        if not path.is_dir() or path.name in committed_case_ids:
            continue
        orphan_root.mkdir(exist_ok=True)
        suffix = 1
        destination = orphan_root / path.name
        while destination.exists():
            suffix += 1
            destination = orphan_root / f"{path.name}-{suffix}"
        os.replace(path, destination)


def run_medicc2_confirmation(
    *,
    bank_root: Path | str,
    output_root: Path | str,
    gate_path: Path | str,
    command_prefix: Sequence[Path | str],
    source_root: Path | str,
    manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
    workers: int = DEFAULT_WORKERS,
    reconstruction_timeout_seconds: int = DEFAULT_RECONSTRUCTION_TIMEOUT_SECONDS,
    evaluation_timeout_seconds: int = DEFAULT_EVALUATION_TIMEOUT_SECONDS,
    rss_limit_bytes: int = DEFAULT_RSS_LIMIT_BYTES,
    capture_limit_bytes: int = DEFAULT_CAPTURE_LIMIT_BYTES,
    resume: bool = False,
    progress: bool = False,
) -> dict[str, Any]:
    if not 1 <= workers <= MAX_WORKERS:
        raise ValueError(f"workers must be in [1,{MAX_WORKERS}].")
    validation = validate_manifest(manifest_path, require_approved=True)
    internal_runtime = validate_internal_runtime(validation["manifest"])
    bank_path, bank = load_confirmation_bank(bank_root, manifest_path=manifest_path)
    source = inspect_source_checkout(source_root)
    runtime = inspect_medicc2_runtime(command_prefix, source)
    normalized_command, _interpreter = _normalized_command_prefix(command_prefix)
    resources = {
        "reconstruction_timeout_seconds": reconstruction_timeout_seconds,
        "evaluation_timeout_seconds": evaluation_timeout_seconds,
        "rss_limit_bytes": rss_limit_bytes,
        "capture_limit_bytes": capture_limit_bytes,
    }
    gate_file, gate = _load_gate(
        gate_path,
        validation=validation,
        bank=bank,
        bank_root=bank_path,
        source=source,
        runtime=runtime,
        internal_runtime=internal_runtime,
        resources={
            "reconstruction_timeout_seconds": reconstruction_timeout_seconds,
            "rss_limit_bytes": rss_limit_bytes,
            "capture_limit_bytes": capture_limit_bytes,
        },
    )
    cases = _standard_cases(bank)
    root = Path(output_root).expanduser().resolve()
    expected_base = {
        "schema_version": RUN_SCHEMA_VERSION,
        "run_id": "ctbf-v5-g2-01a-medicc2-confirmation-v1",
        "scientific_role": "separate_topology_only_contextual_confirmation",
        "design_manifest_sha256": validation["manifest_sha256"],
        "source_lock_sha256": validation["source_lock"]["sha256"],
        "bank_id": bank["bank_id"],
        "bank_root": str(bank_path),
        "bank_checksum_sha256": bank["checksum_closure"]["sha256"],
        "resource_gate_path": str(gate_file),
        "resource_gate_sha256": file_sha256(gate_file),
        "medicc2_source": source,
        "medicc2_runtime": runtime,
        "ctbf_internal_runtime": internal_runtime,
        "arm_spec": dict(MEDICC2_ARM_SPEC),
        "resources": resources,
        "record_execution": fresh_process_contract(CASE_ARM_WORKER_UNIT),
        "expected_record_count": len(cases),
    }
    if resume:
        if not (root / RUN_NAME).is_file():
            raise ValueError("A resumed MEDICC2 confirmation needs run_manifest.json.")
        run = read_json(root / RUN_NAME)
        for field, value in expected_base.items():
            if run.get(field) != value:
                raise ValueError(f"Cannot resume because stored {field} changed.")
        if run.get("status") not in {"in_progress", "interrupted", "failure"}:
            raise ValueError("Only an incomplete MEDICC2 confirmation can resume.")
        committed_case_ids = {
            str(record["case_id"]) for record in run.get("record_inventory", [])
        }
        _quarantine_uncommitted_case_roots(root, committed_case_ids)
        run["status"] = "in_progress"
        run.pop("runner_failure", None)
        run.setdefault("resume_history", []).append(
            {
                "resumed_at_utc": datetime.now(timezone.utc).isoformat(),
                "preserved_record_count": len(committed_case_ids),
            }
        )
    else:
        root = ensure_new_output_root(root)
        shutil.copyfile(validation["manifest_path"], root / "design_manifest.snapshot.json")
        run = {
            **expected_base,
            "status": "in_progress",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "completed_record_count": 0,
            "record_inventory": [],
        }
    write_json(root / RUN_NAME, run)
    start = len(run["record_inventory"])
    expected_ids = [str(case["case_id"]) for case in cases]
    observed_ids = [str(record["case_id"]) for record in run["record_inventory"]]
    if observed_ids != expected_ids[:start]:
        raise ValueError("Stored MEDICC2 records are not the declared prefix.")
    arguments = (
        (
            ordinal,
            str(bank_path),
            str(root),
            case,
            normalized_command,
            reconstruction_timeout_seconds,
            evaluation_timeout_seconds,
            rss_limit_bytes,
            capture_limit_bytes,
        )
        for ordinal, case in enumerate(cases)
        if ordinal >= start
    )
    try:
        if start < len(cases):
            with FreshSpawnTaskPool(workers, buffer_factor=1) as pool:
                for ordinal, record, relative_root in pool.map_ordered(
                    _execute_medicc2_task,
                    arguments,
                    timeout_seconds=(
                        reconstruction_timeout_seconds
                        + evaluation_timeout_seconds
                        + WORKER_GRACE_SECONDS
                    ),
                ):
                    if ordinal != len(run["record_inventory"]):
                        raise RuntimeError("MEDICC2 result order changed.")
                    case_root = root / relative_root
                    case_root.mkdir(parents=True, exist_ok=True)
                    record_path = case_root / "record.json"
                    write_json(record_path, record)
                    run["record_inventory"].append(
                        {
                            "case_id": record["case_id"],
                            "condition_id": record["condition_id"],
                            "status": record["status"],
                            "path": record_path.relative_to(root).as_posix(),
                            "sha256": file_sha256(record_path),
                        }
                    )
                    run["completed_record_count"] = len(run["record_inventory"])
                    write_json(root / RUN_NAME, run)
                    if progress:
                        print(
                            f"{record['case_id']} {MEDICC2_DEVELOPMENT_ARM_ID}: "
                            f"{record['status']}",
                            file=sys.stderr,
                            flush=True,
                        )
    except BaseException as error:
        run["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failure"
        run["runner_failure"] = _typed_error(error, "runner")
        write_json(root / RUN_NAME, run)
        raise
    statuses: dict[str, int] = {}
    for record in run["record_inventory"]:
        statuses[record["status"]] = statuses.get(record["status"], 0) + 1
    run["status_counts"] = statuses
    if validate_internal_runtime(validation["manifest"]) != internal_runtime:
        raise RuntimeError("Internal runtime identity changed during MEDICC2 execution.")
    run["status"] = "complete"
    run["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(root / RUN_NAME, run)
    write_checksum_closure(root, RUN_CHECKSUM_NAME)
    return run


def load_medicc2_confirmation(
    run_root: Path | str,
    *,
    validation: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    from algorithm_evaluation.v5_confirmation_common import validate_checksum_closure

    root = Path(run_root).expanduser().resolve()
    run = read_json(root / RUN_NAME)
    if run.get("schema_version") != RUN_SCHEMA_VERSION or run.get("status") != "complete":
        raise ValueError("MEDICC2 confirmation run is not complete.")
    if run.get("design_manifest_sha256") != validation["manifest_sha256"]:
        raise ValueError("MEDICC2 confirmation design digest changed.")
    validate_checksum_closure(root, RUN_CHECKSUM_NAME)
    if len(run.get("record_inventory", ())) != run.get("expected_record_count"):
        raise ValueError("MEDICC2 confirmation record inventory is incomplete.")
    return root, run


def _command_prefix(arguments: argparse.Namespace) -> list[str]:
    values = [arguments.medicc2_executable]
    if arguments.medicc2_python:
        values.insert(0, arguments.medicc2_python)
    return values


def _add_external_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--medicc2-executable", required=True)
    parser.add_argument("--medicc2-python")
    parser.add_argument("--source-root", required=True)
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
    parser.add_argument("--capture-limit-bytes", type=int, default=DEFAULT_CAPTURE_LIMIT_BYTES)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    gate = subparsers.add_parser("resource-gate")
    _add_external_arguments(gate)
    run = subparsers.add_parser("run")
    _add_external_arguments(run)
    run.add_argument("--resource-gate", required=True)
    run.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--progress", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.mode == "resource-gate":
        result = run_resource_gate(
            bank_root=arguments.bank_root,
            output_root=arguments.output_root,
            command_prefix=_command_prefix(arguments),
            source_root=arguments.source_root,
            manifest_path=arguments.manifest,
            reconstruction_timeout_seconds=arguments.reconstruction_timeout_seconds,
            rss_limit_bytes=arguments.rss_limit_bytes,
            capture_limit_bytes=arguments.capture_limit_bytes,
        )
        print(
            f"complete: MEDICC2 resource gate {result['status']}; "
            f"{len(result['records'])} bank cases"
        )
    else:
        result = run_medicc2_confirmation(
            bank_root=arguments.bank_root,
            output_root=arguments.output_root,
            gate_path=arguments.resource_gate,
            command_prefix=_command_prefix(arguments),
            source_root=arguments.source_root,
            manifest_path=arguments.manifest,
            workers=arguments.workers,
            reconstruction_timeout_seconds=arguments.reconstruction_timeout_seconds,
            evaluation_timeout_seconds=arguments.evaluation_timeout_seconds,
            rss_limit_bytes=arguments.rss_limit_bytes,
            capture_limit_bytes=arguments.capture_limit_bytes,
            resume=arguments.resume,
            progress=arguments.progress,
        )
        print(
            f"complete: {result['completed_record_count']} MEDICC2 records; "
            f"{result['status_counts'].get('failure', 0)} failures"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "GATE_SCHEMA_VERSION",
    "RUN_SCHEMA_VERSION",
    "load_medicc2_confirmation",
    "main",
    "run_medicc2_confirmation",
    "run_resource_gate",
]
