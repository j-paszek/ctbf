"""Replay selected shortlist cases under the fresh-process resource boundary."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Sequence

from algorithm_evaluation.process_isolation import FreshSpawnPerTaskExecutor
from algorithm_evaluation.v5_algorithm_development_run import (
    DEFAULT_EVALUATION_TIMEOUT_SECONDS,
    DEFAULT_RECONSTRUCTION_TIMEOUT_SECONDS,
    DEFAULT_RSS_LIMIT_BYTES,
    _failed_record,
    _run_resource_contract,
    _typed_error,
    run_semantic_gate,
)
from algorithm_evaluation.v5_shortlist_robustness_common import (
    DEFAULT_BLOCK_COUNT,
    RESULT_NAME,
    ensure_new_output_root,
    load_bank_manifest,
    shortlist_specs,
    validate_positive_integer,
    write_json,
)
from algorithm_evaluation.v5_shortlist_robustness_run import (
    _execute_isolated_shortlist_case_arm,
)


PROBE_SCHEMA_VERSION = "ctbf-v5-shortlist-resource-isolation-probe-v1"
DEFAULT_TRIGGER_CASE_IDS = (
    "short-b018-H38-late",
    "short-b018-H38-random",
    "short-b067-H38-late",
    "short-b067-H38-random",
)


def run_probe(
    *,
    bank_root: Path | str,
    output_root: Path | str,
    case_ids: Sequence[str] = DEFAULT_TRIGGER_CASE_IDS,
    expected_block_count: int = DEFAULT_BLOCK_COUNT,
    reconstruction_timeout_seconds: int = DEFAULT_RECONSTRUCTION_TIMEOUT_SECONDS,
    evaluation_timeout_seconds: int = DEFAULT_EVALUATION_TIMEOUT_SECONDS,
    rss_limit_bytes: int = DEFAULT_RSS_LIMIT_BYTES,
    created_at_utc: str | None = None,
    progress: bool = False,
) -> dict[str, Any]:
    for field, value in (
        ("expected_block_count", expected_block_count),
        ("reconstruction_timeout_seconds", reconstruction_timeout_seconds),
        ("evaluation_timeout_seconds", evaluation_timeout_seconds),
        ("rss_limit_bytes", rss_limit_bytes),
    ):
        validate_positive_integer(value, field)
    normalized_ids = tuple(str(value).strip() for value in case_ids)
    if not normalized_ids or any(not value for value in normalized_ids):
        raise ValueError("Resource-isolation probe case ids must be nonempty.")
    if len(set(normalized_ids)) != len(normalized_ids):
        raise ValueError("Resource-isolation probe case ids must be unique.")

    source_root, bank = load_bank_manifest(
        bank_root,
        expected_block_count=expected_block_count,
    )
    case_by_id = {str(case["case_id"]): dict(case) for case in bank["cases"]}
    missing = [case_id for case_id in normalized_ids if case_id not in case_by_id]
    if missing:
        raise ValueError(f"Resource-isolation probe cases are absent: {missing}.")
    cases = [case_by_id[case_id] for case_id in normalized_ids]
    specs = shortlist_specs()
    resources = _run_resource_contract(
        reconstruction_timeout_seconds,
        evaluation_timeout_seconds,
        rss_limit_bytes,
    )
    semantic_gate = run_semantic_gate(
        specs,
        timeout_seconds=reconstruction_timeout_seconds,
        rss_limit_bytes=rss_limit_bytes,
    )
    root = ensure_new_output_root(output_root)
    result: dict[str, Any] = {
        "schema_version": PROBE_SCHEMA_VERSION,
        "status": "in_progress",
        "created_at_utc": created_at_utc or datetime.now(timezone.utc).isoformat(),
        "scientific_role": (
            "resource_isolation_implementation_smoke_not_accuracy_evidence"
        ),
        "bank_id": bank["bank_id"],
        "bank_root": str(source_root),
        "case_ids": list(normalized_ids),
        "arm_specs": [spec.as_record() for spec in specs],
        "semantic_gate_by_arm": semantic_gate,
        "resources": resources,
        "expected_record_count": len(cases) * len(specs),
        "records": [],
    }
    write_json(root / RESULT_NAME, result)

    try:
        with FreshSpawnPerTaskExecutor() as executor:
            for case in cases:
                for spec in specs:
                    gate = semantic_gate[spec.arm_id]
                    if gate["status"] != "passed":
                        record = _failed_record(
                            case=case,
                            spec=spec,
                            error={
                                "stage": "d0_semantic_gate",
                                "type": "D0SemanticGateFailure",
                                "message": (
                                    "Arm was not admitted to the isolation probe."
                                ),
                                "gate_failure": gate.get("failure"),
                            },
                            stage="d0_semantic_gate",
                            reconstruction_runtime=gate.get("resources"),
                            evaluation_runtime=None,
                        )
                    else:
                        record = executor.run(
                            _execute_isolated_shortlist_case_arm,
                            str(source_root),
                            case,
                            spec,
                            reconstruction_timeout_seconds,
                            evaluation_timeout_seconds,
                            rss_limit_bytes,
                            timeout_seconds=resources[
                                "record_worker_outer_timeout_seconds"
                            ],
                        )
                    record["placement_policy"] = str(case["placement_policy"])
                    record["generations"] = [
                        int(value) for value in case["generations"]
                    ]
                    result["records"].append(record)
                    result["completed_record_count"] = len(result["records"])
                    write_json(root / RESULT_NAME, result)
                    if progress:
                        print(
                            f"{case['case_id']} {spec.arm_id}: {record['status']}",
                            file=sys.stderr,
                            flush=True,
                        )
    except BaseException as error:
        result["status"] = "failure"
        result["runner_failure"] = _typed_error(error, "runner")
        write_json(root / RESULT_NAME, result)
        raise

    result["status"] = "complete"
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
    parser.add_argument(
        "--case-ids",
        nargs="+",
        default=DEFAULT_TRIGGER_CASE_IDS,
        help=(
            "Stored case ids in execution order; defaults to the block-18 and "
            "block-67 H38 late/random RSS-trigger sequence."
        ),
    )
    parser.add_argument("--expected-block-count", type=int, default=DEFAULT_BLOCK_COUNT)
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
    parser.add_argument("--progress", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = run_probe(
        bank_root=arguments.bank_root,
        output_root=arguments.output_root,
        case_ids=arguments.case_ids,
        expected_block_count=arguments.expected_block_count,
        reconstruction_timeout_seconds=arguments.reconstruction_timeout_seconds,
        evaluation_timeout_seconds=arguments.evaluation_timeout_seconds,
        rss_limit_bytes=arguments.rss_limit_bytes,
        progress=arguments.progress,
    )
    print(
        f"complete: {result['completed_record_count']} arm records; "
        f"{result['failure_count']} failures"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_TRIGGER_CASE_IDS",
    "PROBE_SCHEMA_VERSION",
    "run_probe",
]
