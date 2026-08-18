"""Bounded owner-run CUTED/EPS preflight for a closed CTBF v5 result root.

The module never discovers result rows recursively.  It first verifies the
completed checksum closure, derives a fixed regime-by-arm sample from the
versioned expected inventory, and then reads only the selected tree pairs.
External metrics run only behind explicit command-line switches.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

from algorithm_evaluation.paper_pipeline_contract import (
    EXPECTED_INVENTORY_SCHEMA_VERSION,
    REGISTERED_ARM_SPECS,
    file_sha256,
    read_checksum_file,
    read_json,
    validate_checksum_closure,
    validate_status_record,
    write_json_atomic,
)
from algorithm_evaluation.paper_pipeline_runner import deserialize_tree
from evaluation_contract import validate_evaluation_result
from repeated_label_evaluation import (
    DEFAULT_EXTERNAL_EDIST_ROOT,
    DEFAULT_EXTERNAL_EPS_ROOT,
    EXTERNAL_CUTED_DEFAULT_TIMEOUT_SECONDS,
    EXTERNAL_EPS_DEFAULT_TIMEOUT_SECONDS,
    evaluate_external_cuted_tree_pair_result,
    evaluate_external_eps_approx_tree_pair_result,
    inspect_external_edist_source,
    inspect_external_eps_source,
    validate_repeated_label_result,
)


RUNTIME_PREFLIGHT_SCHEMA_VERSION = "ctbf-repeated-label-runtime-preflight-v1"
RUNTIME_PREFLIGHT_SAMPLING_VERSION = (
    "ctbf-regime-arm-midpoint-condition-cycle-v1"
)

_ANALYSIS_DIRECTORY = "analysis"


def _bytewise(value: str) -> bytes:
    return value.encode("utf-8")


def _safe_token(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
    ):
        raise ValueError(f"{field} must be a non-empty path-safe token.")
    return value


def _path_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(path, str) or not path for path in value
    ):
        raise ValueError(f"{field} must be a list of non-empty relative paths.")
    if len(set(value)) != len(value):
        raise ValueError(f"{field} contains duplicate paths.")
    return value


def _non_analysis_entries(
    entries: Iterable[tuple[str, str]],
) -> list[tuple[str, str]]:
    return [
        (relative, digest)
        for relative, digest in entries
        if not Path(relative).parts
        or Path(relative).parts[0] != _ANALYSIS_DIRECTORY
    ]


def audit_completed_output_root(
    output_root: Path | str,
) -> tuple[Path, dict[str, Any], set[str], dict[str, Any]]:
    """Verify one completed closure pass and the raw/complete relationship.

    ``complete_checksums.sha256`` is verified against every eligible file once.
    The raw closure is then proven by comparing its entries with the non-analysis
    subset of that already-verified complete closure, avoiding a second full
    hashing pass over the large result root.
    """
    root = Path(output_root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Completed output root is not a directory: {root}")
    for filename in (
        "expected_inventory.json",
        "run_status.json",
        "raw_checksums.sha256",
        "complete_checksums.sha256",
    ):
        if not (root / filename).is_file():
            raise ValueError(f"Completed output root is missing {filename}.")

    validate_checksum_closure(
        root,
        "complete_checksums.sha256",
        include_analysis=True,
    )
    raw_entries = read_checksum_file(root / "raw_checksums.sha256")
    complete_entries = read_checksum_file(root / "complete_checksums.sha256")
    if raw_entries != _non_analysis_entries(complete_entries):
        raise ValueError(
            "raw_checksums.sha256 is not the exact non-analysis subset of the "
            "verified complete closure."
        )

    inventory = read_json(root / "expected_inventory.json")
    if inventory.get("schema_version") != EXPECTED_INVENTORY_SCHEMA_VERSION:
        raise ValueError("Unknown expected-inventory schema.")
    raw_paths = {relative for relative, _digest in raw_entries}
    declared_raw_paths = set(_path_list(inventory.get("raw_files"), "raw_files"))
    missing_from_closure = sorted(declared_raw_paths - raw_paths, key=_bytewise)
    if missing_from_closure:
        raise ValueError(
            "Expected inventory names raw files outside the closed raw set: "
            f"{missing_from_closure[:5]}"
        )

    required_root_files = _path_list(
        inventory.get("required_root_files"), "required_root_files"
    )
    missing_required = [
        relative for relative in required_root_files if not (root / relative).is_file()
    ]
    if missing_required:
        raise ValueError(f"Completed root files are missing: {missing_required}")

    run_status = read_json(root / "run_status.json")
    validate_status_record(run_status)
    if run_status.get("status") != "success" or run_status.get("code") != "raw_closed":
        raise ValueError("The experiment run status is not successful raw_closed.")

    closure = {
        "complete_closure_verified": True,
        "raw_is_complete_non_analysis_subset": True,
        "raw_checksum_file_sha256": file_sha256(root / "raw_checksums.sha256"),
        "complete_checksum_file_sha256": file_sha256(
            root / "complete_checksums.sha256"
        ),
        "raw_checksum_entry_count": len(raw_entries),
        "complete_checksum_entry_count": len(complete_entries),
    }
    return root, inventory, raw_paths, closure


def _validated_inventory_cases(inventory: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_cases = inventory.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("Expected inventory must contain at least one case.")
    cases = []
    seen_case_ids = set()
    for raw_case in raw_cases:
        if not isinstance(raw_case, dict):
            raise ValueError("Each expected-inventory case must be an object.")
        case_id = _safe_token(raw_case.get("case_id"), "case_id")
        regime_id = _safe_token(raw_case.get("regime_id"), "regime_id")
        replicate = raw_case.get("replicate")
        if isinstance(replicate, bool) or not isinstance(replicate, int) or replicate < 0:
            raise ValueError("case replicate must be a nonnegative integer.")
        condition_ids = [
            _safe_token(value, "condition_id")
            for value in _path_list(raw_case.get("condition_ids"), "condition_ids")
        ]
        if case_id in seen_case_ids:
            raise ValueError(f"Duplicate expected-inventory case id: {case_id}")
        seen_case_ids.add(case_id)
        cases.append(
            {
                "case_id": case_id,
                "regime_id": regime_id,
                "replicate": replicate,
                "condition_ids": sorted(condition_ids, key=_bytewise),
            }
        )
    return cases


def build_stratified_sample_plan(
    inventory: Mapping[str, Any],
    *,
    arm_specs: Sequence[tuple[str, str]] = REGISTERED_ARM_SPECS,
) -> list[dict[str, Any]]:
    """Select one deterministic pair for every regime-by-arm stratum."""
    if not arm_specs:
        raise ValueError("At least one arm specification is required.")
    arms = [
        (
            _safe_token(arm_id, "arm_id"),
            _safe_token(algorithm, "algorithm"),
        )
        for arm_id, algorithm in arm_specs
    ]
    if len({arm_id for arm_id, _algorithm in arms}) != len(arms):
        raise ValueError("Arm specifications contain duplicate arm ids.")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in _validated_inventory_cases(inventory):
        grouped[case["regime_id"]].append(case)
    regimes = sorted(grouped, key=_bytewise)
    plan = []
    arm_count = len(arms)
    for regime_index, regime_id in enumerate(regimes):
        cases = sorted(
            grouped[regime_id],
            key=lambda case: (case["replicate"], _bytewise(case["case_id"])),
        )
        for arm_index, (arm_id, algorithm) in enumerate(arms):
            case_bin = (arm_index + regime_index) % arm_count
            case_index = ((2 * case_bin + 1) * len(cases)) // (2 * arm_count)
            case_index = min(case_index, len(cases) - 1)
            case = cases[case_index]
            global_index = regime_index * arm_count + arm_index
            conditions = case["condition_ids"]
            condition = conditions[global_index % len(conditions)]
            plan.append(
                {
                    "sample_id": f"runtime-{global_index + 1:03d}",
                    "regime_id": regime_id,
                    "case_id": case["case_id"],
                    "replicate": case["replicate"],
                    "condition_id": condition,
                    "arm_id": arm_id,
                    "algorithm": algorithm,
                }
            )
    return plan


def _sample_relative_paths(sample: Mapping[str, Any]) -> dict[str, str]:
    case_id = sample["case_id"]
    condition_id = sample["condition_id"]
    arm_id = sample["arm_id"]
    arm_base = f"cases/{case_id}/conditions/{condition_id}/arms/{arm_id}"
    return {
        "truth": f"cases/{case_id}/truth.json",
        "reconstruction": f"{arm_base}/reconstruction.json",
        "evaluation": f"{arm_base}/evaluation.json",
    }


def _load_sample(
    root: Path,
    sample: Mapping[str, Any],
    raw_paths: set[str],
) -> tuple[Any, Any | None, dict[str, Any], str]:
    paths = _sample_relative_paths(sample)
    unclosed = sorted(set(paths.values()) - raw_paths, key=_bytewise)
    if unclosed:
        raise ValueError(f"Selected sample paths are outside raw closure: {unclosed}")

    native = read_json(root / paths["evaluation"])
    validate_evaluation_result(native)
    truth_record = read_json(root / paths["truth"])
    truth_status = truth_record.get("status_record")
    if not isinstance(truth_status, dict):
        raise ValueError(f"Selected truth record lacks status: {paths['truth']}")
    validate_status_record(truth_status)
    if truth_status.get("status") != "success" or "tree" not in truth_record:
        return None, None, native, "truth_unavailable"
    true_tree = deserialize_tree(truth_record["tree"])

    reconstruction = read_json(root / paths["reconstruction"])
    if reconstruction.get("status") != "success":
        return true_tree, None, native, str(reconstruction.get("status", "failure"))
    if "tree" not in reconstruction:
        raise ValueError(f"Successful reconstruction lacks tree: {paths['reconstruction']}")
    return true_tree, deserialize_tree(reconstruction["tree"]), native, "success"


def _tree_sizes(true_tree: Any, reconstructed_tree: Any | None) -> dict[str, Any]:
    true_nodes = None if true_tree is None else int(true_tree.number_of_nodes())
    true_edges = None if true_tree is None else int(true_tree.number_of_edges())
    reconstructed_nodes = (
        None
        if reconstructed_tree is None
        else int(reconstructed_tree.number_of_nodes())
    )
    reconstructed_edges = (
        None
        if reconstructed_tree is None
        else int(reconstructed_tree.number_of_edges())
    )
    return {
        "true_nodes": true_nodes,
        "true_edges": true_edges,
        "reconstructed_nodes": reconstructed_nodes,
        "reconstructed_edges": reconstructed_edges,
        "total_nodes": (
            None
            if true_nodes is None or reconstructed_nodes is None
            else true_nodes + reconstructed_nodes
        ),
    }


def _native_summary(native: Mapping[str, Any]) -> dict[str, Any]:
    if native.get("status") == "success":
        return {
            "status": "success",
            "ad_f1": float(native["metrics"]["ad_f1"]),
            "grf": float(native["metrics"]["grf"]),
            "observation_coverage": float(
                native["inputs"]["observation_label_coverage"]["fraction"]
            ),
        }
    failure = native.get("failure", {})
    return {
        "status": "failure",
        "failure_code": failure.get("code", "unknown_failure"),
    }


def _diagnostic_tail(result: Mapping[str, Any], limit: int = 600) -> str | None:
    failure = result.get("failure")
    if not isinstance(failure, dict):
        return None
    details = failure.get("details")
    if not isinstance(details, dict):
        return None
    fragments = []
    for key in ("stderr_tail", "stdout_tail", "message"):
        value = details.get(key)
        if isinstance(value, str) and value.strip():
            fragments.append(value.strip())
    if not fragments:
        return None
    return " | ".join(fragments)[-limit:]


def _compact_metric_result(
    result: Mapping[str, Any],
    *,
    wall_time_seconds: float,
) -> dict[str, Any]:
    compact = {
        "status": result["status"],
        "wall_time_seconds": float(wall_time_seconds),
    }
    if result["status"] == "failure":
        failure = result["failure"]
        compact.update(
            {
                "failure_code": failure["code"],
                "failure_stage": failure["stage"],
                "failure_message": failure["message"],
            }
        )
        diagnostic = _diagnostic_tail(result)
        if diagnostic is not None:
            compact["backend_diagnostic_tail"] = diagnostic
        return compact

    metric = result["metric"]
    compact.update(
        {
            "raw_value": float(metric["raw_value"]),
            "normalization_denominator": float(
                metric["normalization_denominator"]
            ),
            "value": None if metric["value"] is None else float(metric["value"]),
            "degeneracy": metric["degeneracy"],
        }
    )
    external = result.get("external_execution", {})
    directions = external.get("directional_raw_values")
    if isinstance(directions, dict):
        compact["directional_raw_values"] = {
            direction: float(value)
            for direction, value in sorted(directions.items())
        }
        values = tuple(compact["directional_raw_values"].values())
        compact["directional_disagreement"] = len(values) == 2 and not math.isclose(
            values[0], values[1], rel_tol=0.0, abs_tol=1e-12
        )
    durations = external.get("directional_duration_seconds")
    if isinstance(durations, dict):
        compact["backend_directional_seconds"] = {
            direction: float(value)
            for direction, value in sorted(durations.items())
        }
    return compact


def _number_summary(values: Iterable[float]) -> dict[str, float] | None:
    values = [float(value) for value in values]
    if not values:
        return None
    return {
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
        "total": sum(values),
    }


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    left_delta = [value - left_mean for value in left]
    right_delta = [value - right_mean for value in right]
    denominator = math.sqrt(
        sum(value * value for value in left_delta)
        * sum(value * value for value in right_delta)
    )
    if denominator == 0.0:
        return None
    return sum(a * b for a, b in zip(left_delta, right_delta)) / denominator


def _metric_summary(
    records: Sequence[Mapping[str, Any]], metric_id: str
) -> dict[str, Any]:
    attempts = [
        (record, record["candidate_metrics"][metric_id])
        for record in records
        if metric_id in record["candidate_metrics"]
    ]
    statuses = Counter(result["status"] for _record, result in attempts)
    failures = Counter(
        result["failure_code"]
        for _record, result in attempts
        if result["status"] == "failure"
    )
    not_run_reasons = Counter(
        result["reason"]
        for _record, result in attempts
        if result["status"] == "not_run"
    )
    successful = [
        (record, result)
        for record, result in attempts
        if result["status"] == "success"
    ]
    failed = [
        (record, result)
        for record, result in attempts
        if result["status"] == "failure"
    ]
    diagnostics = " ".join(
        result.get("backend_diagnostic_tail", "") for _record, result in failed
    ).lower()
    failure_examples = {}
    for record, result in failed:
        code = result["failure_code"]
        if code in failure_examples:
            continue
        failure_examples[code] = {
            "sample_id": record["sample_id"],
            "case_id": record["case_id"],
            "condition_id": record["condition_id"],
            "arm_id": record["arm_id"],
            "tree_sizes": record["tree_sizes"],
            "failure_stage": result["failure_stage"],
            "failure_message": result["failure_message"],
            "backend_diagnostic_tail": result.get("backend_diagnostic_tail"),
        }
    return {
        "attempt_count": len(attempts),
        "status_counts": dict(sorted(statuses.items())),
        "failure_code_counts": dict(sorted(failures.items())),
        "not_run_reason_counts": dict(sorted(not_run_reasons.items())),
        "first_failure_by_code": dict(sorted(failure_examples.items())),
        "attempt_wall_time_seconds": _number_summary(
            result["wall_time_seconds"] for _record, result in attempts
        ),
        "successful_value": _number_summary(
            result["value"]
            for _record, result in successful
            if result.get("value") is not None
        ),
        "largest_successful_total_nodes": max(
            (record["tree_sizes"]["total_nodes"] for record, _result in successful),
            default=None,
        ),
        "smallest_failed_total_nodes": min(
            (record["tree_sizes"]["total_nodes"] for record, _result in failed),
            default=None,
        ),
        "license_or_capacity_text_detected": (
            "license" in diagnostics or "model too large" in diagnostics
        ),
        "directional_disagreement_count": sum(
            result.get("directional_disagreement") is True
            for _record, result in successful
        ),
    }


def _comparison_summary(
    records: Sequence[Mapping[str, Any]], metric_id: str
) -> dict[str, Any]:
    aligned_values = []
    ad_f1_values = []
    grf_values = []
    for record in records:
        native = record["native"]
        candidate = record["candidate_metrics"].get(metric_id)
        if (
            native.get("status") != "success"
            or candidate is None
            or candidate.get("status") != "success"
            or candidate.get("value") is None
        ):
            continue
        value = float(candidate["value"])
        aligned_values.append(1.0 - value if metric_id == "cuted_edist" else value)
        ad_f1_values.append(float(native["ad_f1"]))
        grf_values.append(float(native["grf"]))
    return {
        "direction_aligned_candidate_score": (
            "one_minus_normalized_distance"
            if metric_id == "cuted_edist"
            else "normalized_similarity"
        ),
        "paired_success_count": len(aligned_values),
        "pearson_with_native_ad_f1": _pearson(aligned_values, ad_f1_values),
        "pearson_with_native_grf": _pearson(aligned_values, grf_values),
    }


def _not_run_metric(status: str) -> dict[str, Any]:
    return {
        "status": "not_run",
        "reason": status,
        "wall_time_seconds": 0.0,
    }


def run_bounded_preflight(
    output_root: Path | str,
    *,
    run_external_cuted: bool = False,
    run_external_eps_approx: bool = False,
    edist_repository_root: Path | str = DEFAULT_EXTERNAL_EDIST_ROOT,
    edist_python: Path | str | None = None,
    edist_timeout_seconds: float = EXTERNAL_CUTED_DEFAULT_TIMEOUT_SECONDS,
    eps_repository_root: Path | str = DEFAULT_EXTERNAL_EPS_ROOT,
    eps_python: Path | str | None = None,
    eps_timeout_seconds: float = EXTERNAL_EPS_DEFAULT_TIMEOUT_SECONDS,
    arm_specs: Sequence[tuple[str, str]] = REGISTERED_ARM_SPECS,
) -> dict[str, Any]:
    """Verify closure and run explicitly selected metrics on the fixed sample."""
    if run_external_cuted and edist_python is None:
        raise ValueError("External CUTED execution requires edist_python.")
    if run_external_eps_approx and eps_python is None:
        raise ValueError("External EPS execution requires eps_python.")

    root, inventory, raw_paths, closure = audit_completed_output_root(output_root)
    plan = build_stratified_sample_plan(inventory, arm_specs=arm_specs)
    requested_metrics = []
    source_audits = {}
    runner_options: dict[str, tuple[Callable[..., dict[str, Any]], dict[str, Any]]] = {}
    if run_external_cuted:
        requested_metrics.append("cuted_edist")
        audit = inspect_external_edist_source(edist_repository_root)
        source_audits["cuted_edist"] = {
            "status": audit["status"],
            "revision": audit.get("revision"),
        }
        runner_options["cuted_edist"] = (
            evaluate_external_cuted_tree_pair_result,
            {
                "repository_root": edist_repository_root,
                "python_executable": edist_python,
                "timeout_seconds": edist_timeout_seconds,
                "source_audit": audit,
            },
        )
    if run_external_eps_approx:
        requested_metrics.append("eps_approx_external")
        audit = inspect_external_eps_source(eps_repository_root)
        source_audits["eps_approx_external"] = {
            "status": audit["status"],
            "revision": audit.get("revision"),
        }
        runner_options["eps_approx_external"] = (
            evaluate_external_eps_approx_tree_pair_result,
            {
                "repository_root": eps_repository_root,
                "python_executable": eps_python,
                "timeout_seconds": eps_timeout_seconds,
                "source_audit": audit,
            },
        )

    records = []
    for sample in plan:
        true_tree, reconstructed_tree, native, reconstruction_status = _load_sample(
            root, sample, raw_paths
        )
        record = {
            **sample,
            "reconstruction_status": reconstruction_status,
            "tree_sizes": _tree_sizes(true_tree, reconstructed_tree),
            "native": _native_summary(native),
            "candidate_metrics": {},
        }
        for metric_id in requested_metrics:
            if true_tree is None or reconstructed_tree is None:
                record["candidate_metrics"][metric_id] = _not_run_metric(
                    reconstruction_status
                )
                continue
            runner, options = runner_options[metric_id]
            started_ns = time.perf_counter_ns()
            result = runner(true_tree, reconstructed_tree, **options)
            wall_time_seconds = (time.perf_counter_ns() - started_ns) / 1e9
            validate_repeated_label_result(result)
            record["candidate_metrics"][metric_id] = _compact_metric_result(
                result,
                wall_time_seconds=wall_time_seconds,
            )
        records.append(record)

    native_statuses = Counter(record["native"]["status"] for record in records)
    reconstruction_statuses = Counter(
        record["reconstruction_status"] for record in records
    )
    metric_summaries = {
        metric_id: _metric_summary(records, metric_id)
        for metric_id in requested_metrics
    }
    comparisons = {
        metric_id: _comparison_summary(records, metric_id)
        for metric_id in requested_metrics
    }
    report = {
        "schema_version": RUNTIME_PREFLIGHT_SCHEMA_VERSION,
        "run_kind": "bounded_external_evaluator_runtime_and_capacity_preflight",
        "source_output_root": str(root),
        "source_experiment_id": inventory.get("experiment_id"),
        "closure": closure,
        "sampling": {
            "version": RUNTIME_PREFLIGHT_SAMPLING_VERSION,
            "sample_count": len(plan),
            "regimes": sorted({row["regime_id"] for row in plan}, key=_bytewise),
            "arms": [arm_id for arm_id, _algorithm in arm_specs],
            "condition_ids_represented": sorted(
                {row["condition_id"] for row in plan}, key=_bytewise
            ),
            "selection_uses_result_values_or_tree_sizes": False,
        },
        "requested_metrics": requested_metrics,
        "source_audits": source_audits,
        "records": records,
        "summary": {
            "sample_count": len(records),
            "reconstruction_status_counts": dict(
                sorted(reconstruction_statuses.items())
            ),
            "native_status_counts": dict(sorted(native_statuses.items())),
            "candidate_metrics": metric_summaries,
            "descriptive_comparisons": comparisons,
            "inferential_claim_allowed": False,
            "full_corpus_candidate_evaluation_performed": False,
        },
    }
    json.dumps(report, allow_nan=False, sort_keys=True)
    return report


def _output_destination(output_root: Path, output: Path | None) -> Path | None:
    if output is None:
        return None
    destination = output.expanduser().resolve()
    if destination.is_relative_to(output_root):
        raise ValueError(
            "The preflight report must be outside the immutable completed result root."
        )
    if destination.exists():
        raise ValueError(f"Refusing to overwrite existing report: {destination}")
    return destination


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Explicit completed CTBF v5 output root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional new compact JSON report path outside the completed root.",
    )
    parser.add_argument("--run-external-cuted", action="store_true")
    parser.add_argument(
        "--edist-repository-root", type=Path, default=DEFAULT_EXTERNAL_EDIST_ROOT
    )
    parser.add_argument("--edist-python", type=Path)
    parser.add_argument(
        "--edist-timeout-seconds",
        type=float,
        default=EXTERNAL_CUTED_DEFAULT_TIMEOUT_SECONDS,
    )
    parser.add_argument("--run-external-eps-approx", action="store_true")
    parser.add_argument(
        "--eps-repository-root", type=Path, default=DEFAULT_EXTERNAL_EPS_ROOT
    )
    parser.add_argument("--eps-python", type=Path)
    parser.add_argument(
        "--eps-timeout-seconds",
        type=float,
        default=EXTERNAL_EPS_DEFAULT_TIMEOUT_SECONDS,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    output_root = args.output_root.expanduser().resolve()
    try:
        destination = _output_destination(output_root, args.output)
        report = run_bounded_preflight(
            output_root,
            run_external_cuted=args.run_external_cuted,
            run_external_eps_approx=args.run_external_eps_approx,
            edist_repository_root=args.edist_repository_root,
            edist_python=args.edist_python,
            edist_timeout_seconds=args.edist_timeout_seconds,
            eps_repository_root=args.eps_repository_root,
            eps_python=args.eps_python,
            eps_timeout_seconds=args.eps_timeout_seconds,
        )
        if destination is not None:
            write_json_atomic(destination, report)
        console = {
            "schema_version": report["schema_version"],
            "source_output_root": report["source_output_root"],
            "closure": report["closure"],
            "sampling": report["sampling"],
            "source_audits": report["source_audits"],
            "summary": report["summary"],
            "report_path": None if destination is None else str(destination),
        }
        print(json.dumps(console, indent=2, sort_keys=True, allow_nan=False))
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "RUNTIME_PREFLIGHT_SAMPLING_VERSION",
    "RUNTIME_PREFLIGHT_SCHEMA_VERSION",
    "audit_completed_output_root",
    "build_stratified_sample_plan",
    "main",
    "run_bounded_preflight",
]
