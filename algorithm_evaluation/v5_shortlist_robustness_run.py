"""Run the fixed A--D shortlist on a generated robustness bank."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from algorithm_evaluation.paper_pipeline_contract import read_json
from algorithm_evaluation.process_isolation import FreshSpawnPerTaskExecutor
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
    PLACEMENT_POLICIES,
    RESULT_NAME,
    RUN_SCHEMA_VERSION,
    SHORTLIST_ARM_IDS,
    ensure_new_output_root,
    load_bank_manifest,
    read_case_assets,
    shortlist_specs,
    validate_positive_integer,
    write_json,
)


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


def _resume_run(
    *,
    output_root: Path | str,
    run_id: str,
    source_root: Path,
    bank: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
    resources: Mapping[str, Any],
) -> tuple[Path, dict[str, Any], set[tuple[str, str]]]:
    root = Path(output_root).expanduser().resolve()
    result_path = root / RESULT_NAME
    if not root.is_dir() or not result_path.is_file():
        raise ValueError("A resumed shortlist run requires an existing result root.")
    result = read_json(result_path)
    specs = shortlist_specs()
    exact_fields = {
        "schema_version": RUN_SCHEMA_VERSION,
        "run_id": run_id,
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
    history = result.setdefault("resume_history", [])
    if not isinstance(history, list):
        raise ValueError("Stored shortlist resume history is invalid.")
    history.append(
        {
            "resumed_at_utc": datetime.now(timezone.utc).isoformat(),
            "previous_status": result["status"],
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


def run_shortlist(
    *,
    bank_root: Path | str,
    output_root: Path | str,
    run_id: str,
    reconstruction_timeout_seconds: int = DEFAULT_RECONSTRUCTION_TIMEOUT_SECONDS,
    evaluation_timeout_seconds: int = DEFAULT_EVALUATION_TIMEOUT_SECONDS,
    rss_limit_bytes: int = DEFAULT_RSS_LIMIT_BYTES,
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
    ):
        validate_positive_integer(value, field)
    if expected_block_count is not None:
        validate_positive_integer(expected_block_count, "expected_block_count")
    specs = shortlist_specs()
    if tuple(spec.arm_id for spec in specs) != SHORTLIST_ARM_IDS:
        raise RuntimeError("The fixed shortlist arm order changed.")
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
            "records": [],
        }
        completed_keys: set[tuple[str, str]] = set()
        write_json(root / RESULT_NAME, result)

    try:
        with FreshSpawnPerTaskExecutor() as record_executor:
            for case_index, case in enumerate(cases):
                case_keys = {
                    (str(case["case_id"]), spec.arm_id) for spec in specs
                }
                if case_keys <= completed_keys:
                    continue
                for spec in specs:
                    record_key = (str(case["case_id"]), spec.arm_id)
                    if record_key in completed_keys:
                        continue
                    gate = semantic_gate[spec.arm_id]
                    if gate["status"] != "passed":
                        record = _failed_record(
                            case=case,
                            spec=spec,
                            error={
                                "stage": "d0_semantic_gate",
                                "type": "D0SemanticGateFailure",
                                "message": (
                                    "Arm was not admitted to robustness screening."
                                ),
                                "gate_failure": gate.get("failure"),
                            },
                            stage="d0_semantic_gate",
                            reconstruction_runtime=gate.get("resources"),
                            evaluation_runtime=None,
                        )
                    else:
                        record = record_executor.run(
                            _execute_isolated_shortlist_case_arm,
                            str(source_root),
                            dict(case),
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
                    completed_keys.add(record_key)
                    if progress:
                        print(
                            f"{case['case_id']} {spec.arm_id}: {record['status']}",
                            file=sys.stderr,
                            flush=True,
                        )
                result["completed_available_condition_count"] = case_index + 1
                result["completed_record_count"] = len(result["records"])
                write_json(root / RESULT_NAME, result)
    except BaseException as error:
        result["status"] = "failure"
        result["runner_failure"] = _typed_error(error, "runner")
        result["completed_record_count"] = len(result["records"])
        write_json(root / RESULT_NAME, result)
        raise

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
        reconstruction_timeout_seconds=arguments.reconstruction_timeout_seconds,
        evaluation_timeout_seconds=arguments.evaluation_timeout_seconds,
        rss_limit_bytes=arguments.rss_limit_bytes,
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
