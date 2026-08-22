"""Run a declared CTBF v5 shortlist arm set on a robustness bank."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import multiprocessing
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from algorithm_evaluation.paper_pipeline_contract import read_json
from algorithm_evaluation.process_isolation import (
    CASE_ARM_WORKER_UNIT,
    FreshSpawnTaskPool,
    fresh_process_contract,
)
from algorithm_evaluation.v5_algorithm_development_run import (
    DEFAULT_EVALUATION_TIMEOUT_SECONDS,
    DEFAULT_RECONSTRUCTION_TIMEOUT_SECONDS,
    DEFAULT_RSS_LIMIT_BYTES,
    _failed_record,
    _run_one,
    _run_resource_contract,
    _typed_error,
    run_semantic_gate,
)
from algorithm_evaluation.v5_algorithm_development_common import DevelopmentArmSpec
from algorithm_evaluation.v5_shortlist_robustness_common import (
    ARM_SET_BY_NAME,
    PREVIOUS_RUN_SCHEMA_VERSION,
    PLACEMENT_POLICIES,
    RECORD_EXECUTION_SEGMENT_SCHEMA_VERSION,
    RESULT_NAME,
    RUN_SCHEMA_VERSION,
    ensure_new_output_root,
    inferred_serial_record_execution_segment,
    load_bank_manifest,
    read_case_assets,
    resolve_shortlist_arm_ids,
    shortlist_specs,
    validate_positive_integer,
    validate_record_execution_segments,
    write_json,
)


DEFAULT_RECORD_WORKERS = 1
MAX_RECORD_WORKERS = 8


def _ordered_cases(bank: Mapping[str, Any]) -> list[dict[str, Any]]:
    policy_order = {name: index for index, name in enumerate(PLACEMENT_POLICIES)}
    return sorted(
        (dict(case) for case in bank["cases"]),
        key=lambda case: (
            int(case["block_index"]),
            int(case["height"]),
            policy_order[str(case["placement_policy"])],
        ),
    )


def _execute_isolated_shortlist_case_arm(
    source_root: str,
    case: Mapping[str, Any],
    spec: DevelopmentArmSpec,
    reconstruction_timeout_seconds: int,
    evaluation_timeout_seconds: int,
    rss_limit_bytes: int,
) -> dict[str, Any]:
    reconstruction_input, distance, truth_tree, metadata = read_case_assets(
        Path(source_root),
        case,
    )
    return _run_one(
        case=case,
        spec=spec,
        reconstruction_input=reconstruction_input,
        distance=distance,
        truth_tree=truth_tree,
        reconstruction_seed=int(metadata["reconstruction_seed"]),
        reconstruction_timeout_seconds=reconstruction_timeout_seconds,
        evaluation_timeout_seconds=evaluation_timeout_seconds,
        rss_limit_bytes=rss_limit_bytes,
    )


def _execute_isolated_shortlist_record(
    case_index: int,
    source_root: str,
    case: Mapping[str, Any],
    spec: DevelopmentArmSpec,
    semantic_gate: Mapping[str, Any],
    reconstruction_timeout_seconds: int,
    evaluation_timeout_seconds: int,
    rss_limit_bytes: int,
) -> tuple[int, dict[str, Any]]:
    if semantic_gate.get("status") != "passed":
        record = _failed_record(
            case=case,
            spec=spec,
            error={
                "stage": "d0_semantic_gate",
                "type": "D0SemanticGateFailure",
                "message": "Arm was not admitted to robustness screening.",
                "gate_failure": semantic_gate.get("failure"),
            },
            stage="d0_semantic_gate",
            reconstruction_runtime=semantic_gate.get("resources"),
            evaluation_runtime=None,
        )
    else:
        record = _execute_isolated_shortlist_case_arm(
            source_root,
            case,
            spec,
            reconstruction_timeout_seconds,
            evaluation_timeout_seconds,
            rss_limit_bytes,
        )
    return case_index, record


def _resume_run(
    *,
    output_root: Path | str,
    run_id: str,
    source_root: Path,
    bank: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
    specs: Sequence[DevelopmentArmSpec],
    arm_set: str,
    resources: Mapping[str, Any],
) -> tuple[Path, dict[str, Any], set[tuple[str, str]]]:
    root = Path(output_root).expanduser().resolve()
    result_path = root / RESULT_NAME
    if not root.is_dir() or not result_path.is_file():
        raise ValueError("A resumed shortlist run requires an existing result root.")
    result = read_json(result_path)
    stored_schema_version = result.get("schema_version")
    if stored_schema_version not in {
        RUN_SCHEMA_VERSION,
        PREVIOUS_RUN_SCHEMA_VERSION,
    }:
        raise ValueError(
            "Only current or immediately previous shortlist runs can resume."
        )
    exact_fields = {
        "run_id": run_id,
        "arm_set": arm_set,
        "arm_ids": [spec.arm_id for spec in specs],
        "bank_id": bank["bank_id"],
        "bank_root": str(source_root),
        "block_count": int(bank["block_count"]),
        "declared_condition_count": int(bank["declared_condition_count"]),
        "available_condition_count": len(cases),
        "arm_specs": [spec.as_record() for spec in specs],
        "expected_record_count": len(cases) * len(specs),
        "resources": dict(resources),
    }
    for field, expected in exact_fields.items():
        if result.get(field) != expected:
            raise ValueError(
                f"Cannot resume because stored {field} does not match this command."
            )
    if result.get("status") == "complete":
        raise ValueError("A completed shortlist run cannot be resumed.")
    if result.get("status") not in {"in_progress", "failure"}:
        raise ValueError("Only an interrupted or failed shortlist run can resume.")
    records = result.get("records")
    if not isinstance(records, list) or len(records) > result["expected_record_count"]:
        raise ValueError("Stored shortlist run has an invalid record inventory.")
    expected_prefix = [
        (str(case["case_id"]), spec.arm_id)
        for case in cases
        for spec in specs
    ][: len(records)]
    observed_prefix = [
        (str(record.get("case_id")), str(record.get("arm_id")))
        for record in records
        if isinstance(record, Mapping)
    ]
    if len(observed_prefix) != len(records) or observed_prefix != expected_prefix:
        raise ValueError("Stored shortlist records are not the exact run-order prefix.")
    if stored_schema_version == PREVIOUS_RUN_SCHEMA_VERSION:
        result["schema_version"] = RUN_SCHEMA_VERSION
        result["record_execution_segments"] = [
            inferred_serial_record_execution_segment(
                record_count=len(records),
                source_schema_version=stored_schema_version,
            )
        ]
    else:
        segments = validate_record_execution_segments(
            result.get("record_execution_segments"),
            record_count=len(records),
            allow_in_progress=result.get("status") == "in_progress",
        )
        if segments and segments[-1]["status"] == "in_progress":
            segments[-1]["status"] = "interrupted"
            segments[-1]["completed_at_utc"] = datetime.now(
                timezone.utc
            ).isoformat()
        result["record_execution_segments"] = segments
    history = result.setdefault("resume_history", [])
    if not isinstance(history, list):
        raise ValueError("Stored shortlist resume history is invalid.")
    history.append(
        {
            "resumed_at_utc": datetime.now(timezone.utc).isoformat(),
            "previous_status": result["status"],
            "previous_schema_version": stored_schema_version,
            "migrated_to_schema_version": (
                RUN_SCHEMA_VERSION
                if stored_schema_version != RUN_SCHEMA_VERSION
                else None
            ),
            "preserved_record_count": len(records),
            "previous_runner_failure": result.pop("runner_failure", None),
        }
    )
    result["status"] = "in_progress"
    result["completed_record_count"] = len(records)
    result["completed_available_condition_count"] = len(records) // len(specs)
    result.pop("completed_at_utc", None)
    result.pop("success_count", None)
    result.pop("failure_count", None)
    write_json(result_path, result)
    return root, result, set(observed_prefix)


def _begin_record_execution_segment(
    result: dict[str, Any],
    *,
    requested_worker_count: int,
    remaining_record_count: int,
    origin: str,
) -> dict[str, Any] | None:
    if remaining_record_count <= 0:
        return None
    segments = result.setdefault("record_execution_segments", [])
    if not isinstance(segments, list):
        raise ValueError("Stored record-execution segments are invalid.")
    effective_worker_count = min(requested_worker_count, remaining_record_count)
    segment = {
        "schema_version": RECORD_EXECUTION_SEGMENT_SCHEMA_VERSION,
        "segment_index": len(segments),
        "status": "in_progress",
        "record_start_index": len(result["records"]),
        "record_end_index_exclusive": len(result["records"]),
        "requested_worker_count": int(requested_worker_count),
        "effective_worker_count": int(effective_worker_count),
        "machine_cpu_count": multiprocessing.cpu_count(),
        "scheduler": (
            "sequential_declared_order"
            if effective_worker_count == 1
            else "bounded_parallel_declared_order"
        ),
        "result_collection_order": "declared_case_arm_order",
        "checkpoint_policy": "completed_case_prefix",
        "worker_lifecycle": fresh_process_contract(CASE_ARM_WORKER_UNIT),
        "origin": origin,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed_at_utc": None,
        "failure": None,
    }
    segments.append(segment)
    return segment


def _pending_record_arguments(
    *,
    cases: Sequence[Mapping[str, Any]],
    specs: Sequence[DevelopmentArmSpec],
    completed_keys: set[tuple[str, str]],
    semantic_gate: Mapping[str, Mapping[str, Any]],
    source_root: Path,
    reconstruction_timeout_seconds: int,
    evaluation_timeout_seconds: int,
    rss_limit_bytes: int,
):
    for case_index, case in enumerate(cases):
        for spec in specs:
            record_key = (str(case["case_id"]), spec.arm_id)
            if record_key in completed_keys:
                continue
            yield (
                case_index,
                str(source_root),
                dict(case),
                spec,
                dict(semantic_gate[spec.arm_id]),
                reconstruction_timeout_seconds,
                evaluation_timeout_seconds,
                rss_limit_bytes,
            )


def run_shortlist(
    *,
    bank_root: Path | str,
    output_root: Path | str,
    run_id: str,
    reconstruction_timeout_seconds: int = DEFAULT_RECONSTRUCTION_TIMEOUT_SECONDS,
    evaluation_timeout_seconds: int = DEFAULT_EVALUATION_TIMEOUT_SECONDS,
    rss_limit_bytes: int = DEFAULT_RSS_LIMIT_BYTES,
    record_workers: int = DEFAULT_RECORD_WORKERS,
    arm_set: str = "abcd",
    expected_block_count: int | None = None,
    created_at_utc: str | None = None,
    progress: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    if not run_id or not run_id.strip():
        raise ValueError("run_id must be nonempty.")
    for field, value in (
        ("reconstruction_timeout_seconds", reconstruction_timeout_seconds),
        ("evaluation_timeout_seconds", evaluation_timeout_seconds),
        ("rss_limit_bytes", rss_limit_bytes),
        ("record_workers", record_workers),
    ):
        validate_positive_integer(value, field)
    if record_workers > MAX_RECORD_WORKERS:
        raise ValueError(
            f"record_workers may not exceed {MAX_RECORD_WORKERS}."
        )
    if expected_block_count is not None:
        validate_positive_integer(expected_block_count, "expected_block_count")
    arm_ids = resolve_shortlist_arm_ids(arm_set)
    specs = shortlist_specs(arm_ids)
    if tuple(spec.arm_id for spec in specs) != arm_ids:
        raise RuntimeError("The selected shortlist arm order changed.")
    source_root, bank = load_bank_manifest(
        bank_root,
        expected_block_count=expected_block_count,
    )
    cases = _ordered_cases(bank)
    resources = _run_resource_contract(
        reconstruction_timeout_seconds,
        evaluation_timeout_seconds,
        rss_limit_bytes,
    )
    if resume:
        root, result, completed_keys = _resume_run(
            output_root=output_root,
            run_id=run_id,
            source_root=source_root,
            bank=bank,
            cases=cases,
            specs=specs,
            arm_set=arm_set,
            resources=resources,
        )
        semantic_gate = result["semantic_gate_by_arm"]
    else:
        root = ensure_new_output_root(output_root)
        semantic_gate = run_semantic_gate(
            specs,
            timeout_seconds=reconstruction_timeout_seconds,
            rss_limit_bytes=rss_limit_bytes,
        )
        result = {
            "schema_version": RUN_SCHEMA_VERSION,
            "run_id": run_id,
            "arm_set": arm_set,
            "arm_ids": [spec.arm_id for spec in specs],
            "status": "in_progress",
            "created_at_utc": created_at_utc or datetime.now(timezone.utc).isoformat(),
            "scientific_role": bank["scientific_role"],
            "bank_id": bank["bank_id"],
            "bank_root": str(source_root),
            "block_count": int(bank["block_count"]),
            "declared_condition_count": int(bank["declared_condition_count"]),
            "available_condition_count": len(cases),
            "unavailable_condition_count": int(bank["unavailable_condition_count"]),
            "arm_specs": [spec.as_record() for spec in specs],
            "semantic_gate_by_arm": semantic_gate,
            "expected_record_count": len(cases) * len(specs),
            "resources": resources,
            "record_execution_segments": [],
            "records": [],
        }
        completed_keys: set[tuple[str, str]] = set()
        write_json(root / RESULT_NAME, result)

    remaining_record_count = result["expected_record_count"] - len(
        result["records"]
    )
    execution_segment = _begin_record_execution_segment(
        result,
        requested_worker_count=record_workers,
        remaining_record_count=remaining_record_count,
        origin="resume" if resume else "new_run",
    )
    if execution_segment is not None:
        write_json(root / RESULT_NAME, result)
        if progress:
            print(
                "record execution: "
                f"{remaining_record_count} pending records with "
                f"{execution_segment['effective_worker_count']} fresh worker(s); "
                "results commit in declared order",
                file=sys.stderr,
                flush=True,
            )

    try:
        if execution_segment is not None:
            effective_worker_count = int(
                execution_segment["effective_worker_count"]
            )
            arguments = _pending_record_arguments(
                cases=cases,
                specs=specs,
                completed_keys=completed_keys,
                semantic_gate=semantic_gate,
                source_root=source_root,
                reconstruction_timeout_seconds=reconstruction_timeout_seconds,
                evaluation_timeout_seconds=evaluation_timeout_seconds,
                rss_limit_bytes=rss_limit_bytes,
            )
            with FreshSpawnTaskPool(effective_worker_count) as record_pool:
                outcomes = record_pool.map_ordered(
                    _execute_isolated_shortlist_record,
                    arguments,
                    timeout_seconds=resources[
                        "record_worker_outer_timeout_seconds"
                    ],
                )
                for case_index, record in outcomes:
                    case = cases[case_index]
                    record_key = (str(record["case_id"]), str(record["arm_id"]))
                    if record_key in completed_keys:
                        raise RuntimeError(
                            "Parallel shortlist execution duplicated a record."
                        )
                    record["placement_policy"] = str(case["placement_policy"])
                    record["generations"] = [
                        int(value) for value in case["generations"]
                    ]
                    result["records"].append(record)
                    completed_keys.add(record_key)
                    execution_segment["record_end_index_exclusive"] = len(
                        result["records"]
                    )
                    if progress:
                        print(
                            f"{case['case_id']} {record['arm_id']}: "
                            f"{record['status']}",
                            file=sys.stderr,
                            flush=True,
                        )
                    if len(result["records"]) % len(specs) == 0:
                        result["completed_available_condition_count"] = (
                            len(result["records"]) // len(specs)
                        )
                        result["completed_record_count"] = len(result["records"])
                        write_json(root / RESULT_NAME, result)
    except BaseException as error:
        if execution_segment is not None:
            execution_segment["status"] = "failure"
            execution_segment["record_end_index_exclusive"] = len(
                result["records"]
            )
            execution_segment["completed_at_utc"] = datetime.now(
                timezone.utc
            ).isoformat()
            execution_segment["failure"] = _typed_error(error, "runner")
        result["status"] = "failure"
        result["runner_failure"] = _typed_error(error, "runner")
        result["completed_record_count"] = len(result["records"])
        write_json(root / RESULT_NAME, result)
        raise

    if execution_segment is not None:
        execution_segment["status"] = "complete"
        execution_segment["record_end_index_exclusive"] = len(result["records"])
        execution_segment["completed_at_utc"] = datetime.now(
            timezone.utc
        ).isoformat()
    validate_record_execution_segments(
        result["record_execution_segments"],
        record_count=len(result["records"]),
    )
    result["status"] = "complete"
    result["completed_record_count"] = len(result["records"])
    result["success_count"] = sum(
        record["status"] == "success" for record in result["records"]
    )
    result["failure_count"] = len(result["records"]) - result["success_count"]
    result["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(root / RESULT_NAME, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--arm-set",
        choices=tuple(ARM_SET_BY_NAME),
        default="abcd",
        help=(
            "Named development arm roster. selected-all is the complete "
            "21-method labeled roster for fresh sensitivity banks."
        ),
    )
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
    parser.add_argument(
        "--record-workers",
        type=int,
        default=DEFAULT_RECORD_WORKERS,
        help=(
            "Concurrent fresh case-arm workers (1-8). Every worker still "
            "executes exactly one record before process exit."
        ),
    )
    parser.add_argument("--expected-block-count", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = run_shortlist(
        bank_root=arguments.bank_root,
        output_root=arguments.output_root,
        run_id=arguments.run_id,
        arm_set=arguments.arm_set,
        reconstruction_timeout_seconds=arguments.reconstruction_timeout_seconds,
        evaluation_timeout_seconds=arguments.evaluation_timeout_seconds,
        rss_limit_bytes=arguments.rss_limit_bytes,
        record_workers=arguments.record_workers,
        expected_block_count=arguments.expected_block_count,
        resume=arguments.resume,
        progress=arguments.progress,
    )
    print(
        f"complete: {result['completed_record_count']} arm records; "
        f"{result['failure_count']} failures"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
