"""Paired 50%-versus-capped reconstruction probe for CTBF v5.

This non-paper probe reuses the twelve H14/H24/H34 Rule-Y truth blocks and
the exact nested 50% representative-state samples selected by the completed
truth-only fraction probe.  It runs production minimum-bidirectional cnp2cnp
once per case and all six established reconstruction arms, then compares each
arm only with itself under the paired capped-six reference condition.

The output is compact: it contains no CNP vectors, matrices, trees, or node
identities.  Production execution is authorized only by the registered
compact-solver worst-case preflight.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Callable, Iterable, Mapping, Sequence

from algorithm_evaluation import simulator_dense_reconstruction_preflight as preflight
from algorithm_evaluation.hypothesis_height_ambiguity_trend import (
    CASE_METRICS as AMBIGUITY_CASE_METRICS,
)
from algorithm_evaluation.hypothesis_trend_distance_preflight import (
    _number_summary,
)
from algorithm_evaluation.paper_pipeline_contract import (
    PROJECT_ROOT,
    REGISTERED_ARM_SPECS,
    canonical_json_sha256,
    read_json,
    write_json_atomic,
)
from algorithm_evaluation.simulator_growth_probe import (
    _file_sha256,
    _validate_standard_base_config,
)
from algorithm_evaluation.simulator_reconstruction_intuition_probe import (
    APPROVED_SCHEDULES,
    ARM_ENDPOINTS,
    ARM_IDS,
    DEFAULT_BASE_CONFIG,
    DEFAULT_BASE_SEED,
    RECONSTRUCTION_AGGREGATE_METRICS,
    _arm_metric,
    _prefix_consistency,
    _wins_ties_losses,
)
from algorithm_evaluation.simulator_sampling_fraction_truth_probe import (
    DEFAULT_LOWER_BOUND,
    SAMPLING_VERSION,
    hybrid_sample_size,
)
from ctbs import Cnp2CnpFileDistanceProvider, DistanceMatrix, load_ctbs_runtime_config
from distance_semantics import CNP2CNP_SEMANTICS_VERSION
from reconstructor_temporal import TEMPORAL_ARBORESCENCE_SOLVER_VERSION


SCHEMA_VERSION = "ctbf-v5-simulator-dense-reconstruction-probe-v1"
ANALYSIS_ROLE = "nonpaper_fraction50_vs_capped_reconstruction_probe"
CONDITION_ID = preflight.CONDITION_ID
TARGET_FRACTION = preflight.TARGET_FRACTION

EXPECTED_PREFLIGHT_SHA256 = (
    "431fddca7d73d9da0ead156d1dd4cc1a1a2085d5b0a5b0bbf3a81fed8b7144e9"
)
EXPECTED_MAX_SELECTED_OCCURRENCE_COUNT = 395
EXPECTED_MAX_UNIQUE_STATE_COUNT = preflight.EXPECTED_UNIQUE_STATE_COUNT
EXPECTED_MAX_DISTANCE_MATRIX_CELL_COUNT = 108_241
EXPECTED_MAX_ORDERED_PAIR_COUNT = preflight.EXPECTED_ORDERED_PAIR_COUNT

DEFAULT_REPLICATES = 12
MAX_REPLICATES = 12
DEFAULT_MAX_CASE_DEPENDENCY_FAILURES = 6

DEFAULT_SIMULATION_TIMEOUT_SECONDS = preflight.DEFAULT_SIMULATION_TIMEOUT_SECONDS
DEFAULT_DISTANCE_TIMEOUT_SECONDS = preflight.DEFAULT_DISTANCE_TIMEOUT_SECONDS
DEFAULT_DIAGNOSTIC_TIMEOUT_SECONDS = preflight.DEFAULT_DIAGNOSTIC_TIMEOUT_SECONDS
DEFAULT_RECONSTRUCTION_TIMEOUT_SECONDS = (
    preflight.DEFAULT_RECONSTRUCTION_TIMEOUT_SECONDS
)
DEFAULT_EVALUATION_TIMEOUT_SECONDS = preflight.DEFAULT_EVALUATION_TIMEOUT_SECONDS
DEFAULT_CNP2CNP_PROCESS_TIMEOUT_SECONDS = (
    preflight.DEFAULT_CNP2CNP_PROCESS_TIMEOUT_SECONDS
)
DEFAULT_RSS_LIMIT_BYTES = 4 * 1024**3
DEFAULT_CAPTURE_LIMIT_BYTES = preflight.DEFAULT_CAPTURE_LIMIT_BYTES


DistanceCompute = Callable[[Sequence[Any]], DistanceMatrix]


TRUTH_COMPARISON_DIRECTIONS = {
    "all_pair_incomparable_fraction": "lower",
    "all_pair_hidden_fork_fraction": "lower",
    "cross_biopsy_ancestor_pair_fraction": "higher",
    "cross_biopsy_incomparable_pair_fraction": "lower",
    "adjacent_sampled_ancestor_coverage_fraction": "higher",
    "any_earlier_sampled_ancestor_coverage_fraction": "higher",
    "lineage_linked_occurrence_fraction": "higher",
    "mean_hidden_internal_nodes_to_nearest_sampled_ancestor": "lower",
    "minimal_sampled_occurrence_fraction": "lower",
    "minimum_invented_edge_fraction": "lower",
}


def _validate_positive_integer(value: Any, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer.")


def _validate_preflight_authorization(
    report: Mapping[str, Any],
    *,
    base_config_sha256: str,
    base_seed: int,
) -> None:
    preflight.validate_report(report)
    if (
        report.get("status") != "complete"
        or report.get("preflight_verdict")
        != "pass_all_registered_stages_pending_owner_runtime_review"
    ):
        raise ValueError("Dense preflight did not pass its registered gate.")
    if report.get("scientific_role", {}).get("injected_distance_for_test"):
        raise ValueError("Injected-distance preflight cannot authorize production.")
    if report.get("input", {}).get("base_config_sha256") != base_config_sha256:
        raise ValueError("Dense preflight uses a different base config.")
    if report.get("input", {}).get("base_seed") != base_seed:
        raise ValueError("Dense preflight uses a different base seed.")
    if report.get("resource_bound", {}).get("rss_limit_bytes_per_stage") != (
        DEFAULT_RSS_LIMIT_BYTES
    ):
        raise ValueError("Dense preflight used a different RSS bound.")
    arms = report.get("case", {}).get("arms", [])
    if tuple(arm.get("arm_id") for arm in arms) != ARM_IDS or any(
        arm.get("status") != "success" for arm in arms
    ):
        raise ValueError("Dense preflight did not pass all six registered arms.")
    solver = report.get("provenance", {}).get("temporal_solver", {})
    if solver.get("implementation_version") != TEMPORAL_ARBORESCENCE_SOLVER_VERSION:
        raise ValueError("Dense preflight used a different temporal solver.")
    if solver.get("source_module_sha256") != _file_sha256(
        PROJECT_ROOT / "reconstructor_temporal.py"
    ):
        raise ValueError("Dense preflight temporal-solver source is stale.")
    if report.get("provenance", {}).get("source_module_sha256") != _file_sha256(
        Path(preflight.__file__)
    ):
        raise ValueError("Dense preflight runner source is stale.")
    references = report.get("references", {})
    if references.get("fraction_truth", {}).get("sha256") != (
        preflight.EXPECTED_FRACTION_TRUTH_SHA256
    ):
        raise ValueError("Dense preflight used a different fraction reference.")
    if references.get("sparse_reconstruction", {}).get("sha256") != (
        preflight.EXPECTED_SPARSE_RECONSTRUCTION_SHA256
    ):
        raise ValueError("Dense preflight used a different capped reference.")


def _load_preflight_authorization(
    path: Path | str | None,
    *,
    base_config_sha256: str,
    base_seed: int,
    enforce_registered_hash: bool,
) -> dict[str, Any]:
    if path is None:
        return {"provided": False, "registered_hash_enforced": False}
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"Dense preflight is not a file: {resolved}.")
    digest = _file_sha256(resolved)
    if enforce_registered_hash and digest != EXPECTED_PREFLIGHT_SHA256:
        raise ValueError("Dense preflight is not the owner-reviewed artifact.")
    report = read_json(resolved)
    _validate_preflight_authorization(
        report,
        base_config_sha256=base_config_sha256,
        base_seed=base_seed,
    )
    return {
        "provided": True,
        "path": str(resolved),
        "sha256": digest,
        "schema_version": report["schema_version"],
        "verdict": report["preflight_verdict"],
        "registered_hash_enforced": enforce_registered_hash,
        "owner_review_recorded": True,
    }


def _truth_reference(fraction_case: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if fraction_case is None:
        return None
    conditions = fraction_case["simulation_summary"]["conditions"]
    dense = conditions[CONDITION_ID]["summary"]
    capped = conditions["capped_six_control"]["summary"]
    paired = {}
    for metric, preferred_direction in TRUTH_COMPARISON_DIRECTIONS.items():
        dense_value = dense["scalar_metrics"].get(metric)
        capped_value = capped["scalar_metrics"].get(metric)
        difference = (
            None
            if dense_value is None or capped_value is None
            else float(dense_value) - float(capped_value)
        )
        paired[metric] = {
            "preferred_direction": preferred_direction,
            "fraction50": dense_value,
            "capped_six": capped_value,
            "difference": difference,
        }
    return {
        "fraction50_condition_summary_sha256": canonical_json_sha256(dense),
        "capped_six_condition_summary_sha256": canonical_json_sha256(capped),
        "fraction50_scalar_metrics": dict(dense["scalar_metrics"]),
        "capped_six_scalar_metrics": dict(capped["scalar_metrics"]),
        "fraction50_observed_only_arborescence_representable": bool(
            dense["cross_biopsy_relation_diagnostics"][
                "observed_only_occurrence_arborescence_representable"
            ]
        ),
        "capped_six_observed_only_arborescence_representable": bool(
            capped["cross_biopsy_relation_diagnostics"][
                "observed_only_occurrence_arborescence_representable"
            ]
        ),
        "paired_truth_metrics": paired,
    }


def _paired_reconstruction_comparison(
    dense_case: Mapping[str, Any],
    capped_case: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if capped_case is None:
        return None
    arms = {}
    for arm_id in ARM_IDS:
        dense_arm = next(
            (arm for arm in dense_case.get("arms", []) if arm.get("arm_id") == arm_id),
            None,
        )
        capped_arm = next(
            (arm for arm in capped_case.get("arms", []) if arm.get("arm_id") == arm_id),
            None,
        )
        metric_rows = {}
        for metric in ARM_ENDPOINTS[arm_id]["declared_metrics"]:
            dense_value = _arm_metric(dense_case, arm_id, metric)
            capped_value = _arm_metric(capped_case, arm_id, metric)
            metric_rows[metric] = {
                "preferred_direction": "higher",
                "fraction50": dense_value,
                "capped_six": capped_value,
                "difference": (
                    None
                    if dense_value is None or capped_value is None
                    else dense_value - capped_value
                ),
            }
        arms[arm_id] = {
            "fraction50_status": None if dense_arm is None else dense_arm.get("status"),
            "capped_six_status": None if capped_arm is None else capped_arm.get("status"),
            "paired_metrics": metric_rows,
        }
    return {
        "capped_case_key": capped_case["case_key"],
        "capped_case_summary_sha256": canonical_json_sha256(
            {
                "simulation_summary": capped_case["simulation_summary"],
                "arms": capped_case["arms"],
            }
        ),
        "arms": arms,
    }


def _direction_counts(values: Sequence[float], preferred_direction: str) -> dict[str, int]:
    if preferred_direction not in {"higher", "lower"}:
        raise ValueError("preferred_direction must be higher or lower.")
    tolerance = 1e-12
    if preferred_direction == "higher":
        improvements = sum(value > tolerance for value in values)
        worsenings = sum(value < -tolerance for value in values)
    else:
        improvements = sum(value < -tolerance for value in values)
        worsenings = sum(value > tolerance for value in values)
    return {
        "improvements": improvements,
        "ties": len(values) - improvements - worsenings,
        "worsenings": worsenings,
    }


def _runtime_values(
    arms: Sequence[Mapping[str, Any]],
    field: str,
    value: str,
) -> list[float]:
    values = []
    for arm in arms:
        runtime = arm.get(field)
        if runtime is None:
            continue
        if value == "seconds":
            values.append(runtime["wall_time_ns"] / 1_000_000_000)
        elif value == "peak_rss_bytes":
            peak = runtime.get("memory", {}).get("peak_rss_bytes")
            if peak is not None:
                values.append(float(peak))
        else:  # pragma: no cover - internal invariant
            raise ValueError(f"Unknown runtime value {value!r}.")
    return values


def _height_block(height: int, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    complete = [record for record in records if record.get("status") == "complete"]
    sampling = {
        metric: _number_summary(
            [float(record["simulation_summary"][metric]) for record in complete]
        )
        for metric in (
            "selected_occurrence_count",
            "selected_unique_state_count",
            "distance_matrix_cell_count",
            "bidirectional_ordered_pair_bound",
        )
    }
    biopsy_levels = []
    for biopsy_level, generation in enumerate(APPROVED_SCHEDULES[height]):
        rows = [record["simulation_summary"]["sampling"][biopsy_level] for record in complete]
        biopsy_levels.append(
            {
                "biopsy_level": biopsy_level,
                "generation": generation,
                "available_distinct_state_count": _number_summary(
                    [float(row["available_distinct_state_count"]) for row in rows]
                ),
                "realized_occurrence_count": _number_summary(
                    [float(row["realized_occurrence_count"]) for row in rows]
                ),
                "sample_size_driver_counts": dict(
                    sorted(Counter(row["sample_size_driver"] for row in rows).items())
                ),
            }
        )

    truth_comparisons = {}
    for metric, preferred_direction in TRUTH_COMPARISON_DIRECTIONS.items():
        paired_rows = [
            record["truth_reference"]["paired_truth_metrics"][metric]
            for record in complete
            if record.get("truth_reference") is not None
        ]
        differences = [
            float(row["difference"])
            for row in paired_rows
            if row["difference"] is not None
        ]
        truth_comparisons[metric] = {
            "preferred_direction": preferred_direction,
            "fraction50": _number_summary(
                [float(row["fraction50"]) for row in paired_rows if row["fraction50"] is not None]
            ),
            "capped_six": _number_summary(
                [float(row["capped_six"]) for row in paired_rows if row["capped_six"] is not None]
            ),
            "difference": _number_summary(differences),
            "improvement_tie_worsening": _direction_counts(
                differences,
                preferred_direction,
            ),
        }

    ambiguity = {
        metric: _number_summary(
            [
                float(record["ambiguity"]["case_metrics"][metric])
                for record in complete
                if record.get("ambiguity") is not None
                and record["ambiguity"]["case_metrics"].get(metric) is not None
            ]
        )
        for metric in AMBIGUITY_CASE_METRICS
    }
    distance_seconds = _number_summary(
        [
            record["distance"]["runtime"]["wall_time_ns"] / 1_000_000_000
            for record in complete
            if record.get("distance") is not None
        ]
    )
    distance_peak_rss = _number_summary(
        [
            float(record["distance"]["runtime"]["memory"]["peak_rss_bytes"])
            for record in complete
            if record.get("distance") is not None
            and record["distance"]["runtime"]["memory"]["peak_rss_bytes"] is not None
        ]
    )

    arm_blocks = {}
    for arm_id in ARM_IDS:
        arm_rows = [
            arm
            for record in complete
            for arm in record.get("arms", [])
            if arm.get("arm_id") == arm_id
        ]
        paired_rows = [
            record["capped_comparison"]["arms"][arm_id]
            for record in complete
            if record.get("capped_comparison") is not None
        ]
        metric_blocks = {}
        for metric in ARM_ENDPOINTS[arm_id]["declared_metrics"]:
            dense_values = [
                value
                for record in complete
                for value in [_arm_metric(record, arm_id, metric)]
                if value is not None
            ]
            capped_values = [
                float(row["paired_metrics"][metric]["capped_six"])
                for row in paired_rows
                if row["paired_metrics"][metric]["capped_six"] is not None
            ]
            differences = [
                float(row["paired_metrics"][metric]["difference"])
                for row in paired_rows
                if row["paired_metrics"][metric]["difference"] is not None
            ]
            metric_blocks[metric] = {
                "fraction50": _number_summary(dense_values),
                "capped_six": _number_summary(capped_values),
                "difference": _number_summary(differences),
                "wins_ties_losses": _wins_ties_losses(differences),
            }
        arm_blocks[arm_id] = {
            "problem": ARM_ENDPOINTS[arm_id]["problem"],
            "declared_metrics": list(ARM_ENDPOINTS[arm_id]["declared_metrics"]),
            "status_counts": dict(
                sorted(Counter(arm.get("status") for arm in arm_rows).items())
            ),
            "error_type_counts": dict(
                sorted(
                    Counter(
                        arm["error"]["type"]
                        for arm in arm_rows
                        if arm.get("error") is not None
                    ).items()
                )
            ),
            "paired_sampling_metrics": metric_blocks,
            "reconstruction_summaries": {
                metric: _number_summary(
                    [
                        float(arm["reconstruction_summary"][metric])
                        for arm in arm_rows
                        if arm.get("reconstruction_summary") is not None
                    ]
                )
                for metric in RECONSTRUCTION_AGGREGATE_METRICS
            },
            "reconstruction_wall_time_seconds": _number_summary(
                _runtime_values(arm_rows, "reconstruction_runtime", "seconds")
            ),
            "reconstruction_peak_rss_bytes": _number_summary(
                _runtime_values(arm_rows, "reconstruction_runtime", "peak_rss_bytes")
            ),
            "evaluation_wall_time_seconds": _number_summary(
                _runtime_values(arm_rows, "evaluation_runtime", "seconds")
            ),
            "evaluation_peak_rss_bytes": _number_summary(
                _runtime_values(arm_rows, "evaluation_runtime", "peak_rss_bytes")
            ),
        }

    within_fraction = {}
    for name, left, right, metric in (
        (
            "temporal_minus_no_time_ad_f1",
            "temporal_minimum",
            "temporal_minimum_no_time",
            "ad_f1",
        ),
        (
            "temporal_minus_no_time_grf",
            "temporal_minimum",
            "temporal_minimum_no_time",
            "grf",
        ),
        (
            "biopsy_guided_minus_classical_partial_grf",
            "biopsy_guided_classical",
            "classical_partial",
            "grf",
        ),
    ):
        differences = []
        for record in complete:
            left_value = _arm_metric(record, left, metric)
            right_value = _arm_metric(record, right, metric)
            if left_value is not None and right_value is not None:
                differences.append(left_value - right_value)
        within_fraction[name] = {
            "left_arm": left,
            "right_arm": right,
            "metric": metric,
            "difference": _number_summary(differences),
            "wins_ties_losses": _wins_ties_losses(differences),
        }

    return {
        "attempted_case_count": len(records),
        "status_counts": dict(
            sorted(Counter(record.get("status") for record in records).items())
        ),
        "error_type_counts": dict(
            sorted(
                Counter(
                    record["error"]["type"]
                    for record in records
                    if record.get("error") is not None
                ).items()
            )
        ),
        "passing_reference_check_count": sum(
            record.get("reference_check", {}).get("passed") is True
            for record in records
        ),
        "sampling_metrics": sampling,
        "biopsy_levels": biopsy_levels,
        "paired_truth_metrics": truth_comparisons,
        "ambiguity_metrics": ambiguity,
        "distance_wall_time_seconds": distance_seconds,
        "distance_peak_rss_bytes": distance_peak_rss,
        "arms": arm_blocks,
        "within_fraction50_contrasts": within_fraction,
    }


def _paired_height_endpoint(
    cases: Sequence[Mapping[str, Any]],
    heights: Sequence[int],
) -> dict[str, Any]:
    low_height = min(heights)
    high_height = max(heights)
    by_replicate = {
        replicate_index: {
            case["height"]: case
            for case in cases
            if case["replicate_index"] == replicate_index
            and case.get("status") == "complete"
        }
        for replicate_index in sorted({case["replicate_index"] for case in cases})
    }
    arms = {}
    for arm_id in ARM_IDS:
        arms[arm_id] = {}
        for metric in ARM_ENDPOINTS[arm_id]["declared_metrics"]:
            differences = []
            if low_height != high_height:
                for block in by_replicate.values():
                    if low_height not in block or high_height not in block:
                        continue
                    low = _arm_metric(block[low_height], arm_id, metric)
                    high = _arm_metric(block[high_height], arm_id, metric)
                    if low is not None and high is not None:
                        differences.append(high - low)
            arms[arm_id][metric] = {
                "difference": _number_summary(differences),
                "wins_ties_losses": _wins_ties_losses(differences),
            }
    return {
        "contrast": f"H{high_height}_minus_H{low_height}",
        "distinct_endpoint_heights": low_height != high_height,
        "arm_metrics": arms,
    }


def aggregate_cases(
    cases: Sequence[Mapping[str, Any]],
    heights: Sequence[int],
) -> dict[str, Any]:
    return {
        "by_height": {
            str(height): _height_block(
                height,
                [case for case in cases if case["height"] == height],
            )
            for height in heights
        },
        "paired_height_endpoint_differences": _paired_height_endpoint(
            cases,
            heights,
        ),
        "common_seed_prefix_consistency": _prefix_consistency(cases),
    }


def _reference_bounds(
    fraction_index: Mapping[tuple[int, int], Mapping[str, Any]],
) -> dict[str, Any]:
    rows = []
    for (replicate_index, height), case in fraction_index.items():
        summary = case["simulation_summary"]["conditions"][CONDITION_ID]["summary"]
        rows.append(
            {
                "replicate_index": replicate_index,
                "height": height,
                "selected_occurrence_count": summary["selected_occurrence_count"],
                "selected_unique_state_count": summary["selected_unique_state_count"],
                "projected_distance_matrix_cell_count": summary[
                    "projected_distance_matrix_cell_count"
                ],
                "projected_bidirectional_ordered_pair_count": summary[
                    "projected_bidirectional_ordered_pair_count"
                ],
            }
        )
    if not rows:
        return {
            "maximum_selected_occurrence_count": None,
            "maximum_unique_state_count": None,
            "maximum_distance_matrix_cell_count": None,
            "maximum_bidirectional_ordered_pair_count": None,
        }
    bounds = {
        "maximum_selected_occurrence_count": max(
            row["selected_occurrence_count"] for row in rows
        ),
        "maximum_unique_state_count": max(
            row["selected_unique_state_count"] for row in rows
        ),
        "maximum_distance_matrix_cell_count": max(
            row["projected_distance_matrix_cell_count"] for row in rows
        ),
        "maximum_bidirectional_ordered_pair_count": max(
            row["projected_bidirectional_ordered_pair_count"] for row in rows
        ),
    }
    if (
        bounds["maximum_selected_occurrence_count"]
        != EXPECTED_MAX_SELECTED_OCCURRENCE_COUNT
        or bounds["maximum_unique_state_count"] != EXPECTED_MAX_UNIQUE_STATE_COUNT
        or bounds["maximum_distance_matrix_cell_count"]
        != EXPECTED_MAX_DISTANCE_MATRIX_CELL_COUNT
        or bounds["maximum_bidirectional_ordered_pair_count"]
        != EXPECTED_MAX_ORDERED_PAIR_COUNT
    ):
        raise ValueError("Registered fraction reference has different resource bounds.")
    return bounds


def run_probe(
    *,
    base_config_path: Path | str = DEFAULT_BASE_CONFIG,
    fraction_truth_report_path: Path | str | None = None,
    sparse_reconstruction_report_path: Path | str | None = None,
    dense_preflight_report_path: Path | str | None = None,
    replicates: int = DEFAULT_REPLICATES,
    heights: Sequence[int] = tuple(APPROVED_SCHEDULES),
    base_seed: int = DEFAULT_BASE_SEED,
    simulation_timeout_seconds: int = DEFAULT_SIMULATION_TIMEOUT_SECONDS,
    distance_timeout_seconds: int = DEFAULT_DISTANCE_TIMEOUT_SECONDS,
    diagnostic_timeout_seconds: int = DEFAULT_DIAGNOSTIC_TIMEOUT_SECONDS,
    reconstruction_timeout_seconds: int = DEFAULT_RECONSTRUCTION_TIMEOUT_SECONDS,
    evaluation_timeout_seconds: int = DEFAULT_EVALUATION_TIMEOUT_SECONDS,
    cnp2cnp_process_timeout_seconds: int = DEFAULT_CNP2CNP_PROCESS_TIMEOUT_SECONDS,
    rss_limit_bytes: int = DEFAULT_RSS_LIMIT_BYTES,
    capture_limit_bytes: int = DEFAULT_CAPTURE_LIMIT_BYTES,
    max_case_dependency_failures: int = DEFAULT_MAX_CASE_DEPENDENCY_FAILURES,
    distance_compute: DistanceCompute | None = None,
    enforce_registered_references: bool = True,
    created_at_utc: str | None = None,
    progress: bool = False,
) -> dict[str, Any]:
    base_config_path = Path(base_config_path).expanduser().resolve()
    if not base_config_path.is_file():
        raise ValueError(f"Base config is not a file: {base_config_path}.")
    base_config = read_json(base_config_path)
    _validate_standard_base_config(base_config)
    base_config_sha256 = _file_sha256(base_config_path)

    if isinstance(replicates, bool) or not isinstance(replicates, int):
        raise ValueError("replicates must be an integer.")
    if not 1 <= replicates <= MAX_REPLICATES:
        raise ValueError(f"replicates must be in [1,{MAX_REPLICATES}].")
    heights = tuple(heights)
    if not heights or len(set(heights)) != len(heights) or any(
        height not in APPROVED_SCHEDULES for height in heights
    ):
        raise ValueError("heights must be a nonempty unique approved subset.")
    if isinstance(base_seed, bool) or not isinstance(base_seed, int) or base_seed < 0:
        raise ValueError("base_seed must be a nonnegative integer.")
    for field, value in (
        ("simulation_timeout_seconds", simulation_timeout_seconds),
        ("distance_timeout_seconds", distance_timeout_seconds),
        ("diagnostic_timeout_seconds", diagnostic_timeout_seconds),
        ("reconstruction_timeout_seconds", reconstruction_timeout_seconds),
        ("evaluation_timeout_seconds", evaluation_timeout_seconds),
        ("cnp2cnp_process_timeout_seconds", cnp2cnp_process_timeout_seconds),
        ("rss_limit_bytes", rss_limit_bytes),
        ("capture_limit_bytes", capture_limit_bytes),
        ("max_case_dependency_failures", max_case_dependency_failures),
    ):
        _validate_positive_integer(value, field)

    production = distance_compute is None
    references_provided = (
        fraction_truth_report_path is not None
        and sparse_reconstruction_report_path is not None
    )
    if (fraction_truth_report_path is None) != (
        sparse_reconstruction_report_path is None
    ):
        raise ValueError("Both compact comparison references must be provided together.")
    if production and (
        replicates != DEFAULT_REPLICATES
        or heights != tuple(APPROVED_SCHEDULES)
        or base_seed != DEFAULT_BASE_SEED
    ):
        raise ValueError("Production dense probe is frozen to 12 H14/H24/H34 blocks.")
    if production and (
        not references_provided or dense_preflight_report_path is None
    ):
        raise ValueError("Production dense probe requires both references and preflight.")
    if production and rss_limit_bytes != DEFAULT_RSS_LIMIT_BYTES:
        raise ValueError("Production dense probe requires the registered 4 GiB bound.")

    fraction_index: dict[tuple[int, int], Mapping[str, Any]] = {}
    sparse_index: dict[tuple[int, int], Mapping[str, Any]] = {}
    reference_metadata: dict[str, Any] = {
        "provided": False,
        "registered_hashes_enforced": False,
    }
    if references_provided:
        fraction_index, sparse_index, loaded = preflight.load_references(
            fraction_truth_report_path=fraction_truth_report_path,
            sparse_reconstruction_report_path=sparse_reconstruction_report_path,
            base_config_sha256=base_config_sha256,
            base_seed=base_seed,
            enforce_registered_hashes=enforce_registered_references,
        )
        reference_metadata = {
            "provided": True,
            "registered_hashes_enforced": enforce_registered_references,
            **loaded,
        }
    authorization = _load_preflight_authorization(
        dense_preflight_report_path,
        base_config_sha256=base_config_sha256,
        base_seed=base_seed,
        enforce_registered_hash=(production or enforce_registered_references),
    )
    if production and not authorization["provided"]:
        raise ValueError("Production dense probe lacks preflight authorization.")

    if production:
        runtime_config = replace(
            load_ctbs_runtime_config(),
            cnp2cnp_timeout_seconds=float(cnp2cnp_process_timeout_seconds),
            cnp2cnp_capture_limit_bytes=int(capture_limit_bytes),
        )
        compute = Cnp2CnpFileDistanceProvider(runtime_config).compute
    else:
        compute = distance_compute

    cases = []
    dependency_failure_count = 0
    stopped_early = False
    for replicate_index in range(replicates):
        for height in heights:
            key = (replicate_index, height)
            record = preflight._run_case(
                base_config=base_config,
                height=height,
                replicate_index=replicate_index,
                base_seed=base_seed,
                fraction_case=fraction_index.get(key),
                sparse_case=sparse_index.get(key),
                distance_compute=compute,
                injected_distance=not production,
                simulation_timeout_seconds=simulation_timeout_seconds,
                distance_timeout_seconds=distance_timeout_seconds,
                diagnostic_timeout_seconds=diagnostic_timeout_seconds,
                reconstruction_timeout_seconds=reconstruction_timeout_seconds,
                evaluation_timeout_seconds=evaluation_timeout_seconds,
                rss_limit_bytes=rss_limit_bytes,
                progress=progress,
            )
            record["case_key"] = (
                f"v5-fraction50-H{height}-R{replicate_index + 1:03d}"
            )
            record["truth_reference"] = _truth_reference(fraction_index.get(key))
            record["capped_comparison"] = _paired_reconstruction_comparison(
                record,
                sparse_index.get(key),
            )
            cases.append(record)
            if progress:
                print(
                    json.dumps(
                        {
                            "case_key": record["case_key"],
                            "stage": "case_complete",
                            "status": record["status"],
                            "arm_status_counts": dict(
                                sorted(
                                    Counter(
                                        arm["status"] for arm in record.get("arms", [])
                                    ).items()
                                )
                            ),
                        },
                        sort_keys=True,
                    ),
                    file=sys.stderr,
                    flush=True,
                )
            if record["status"] != "complete":
                dependency_failure_count += 1
                if dependency_failure_count >= max_case_dependency_failures:
                    stopped_early = True
                    break
        if stopped_early:
            break

    aggregate = aggregate_cases(cases, heights)
    if not aggregate["common_seed_prefix_consistency"][
        "all_evaluable_common_seed_prefix_checks_passed"
    ]:
        raise ValueError("Common-seed height prefixes are inconsistent.")

    arm_failures = sum(
        arm.get("status") != "success"
        for case in cases
        for arm in case.get("arms", [])
    )
    reference_failures = sum(
        case.get("reference_check", {}).get("performed") is True
        and case["reference_check"].get("passed") is not True
        for case in cases
    )
    if reference_failures:
        status = "failed_reference_validation"
    elif stopped_early:
        status = "stopped_at_dependency_failure_cap"
    elif dependency_failure_count or arm_failures:
        status = "complete_with_typed_failures"
    else:
        status = "complete"

    identities: dict[str, dict[str, Any]] = {}
    for record in cases:
        distance = record.get("distance") or {}
        identity_hash = distance.get("identity_sha256")
        if identity_hash:
            identities.setdefault(
                identity_hash,
                {"case_count": 0, "identity": distance["identity"]},
            )["case_count"] += 1

    bounds = _reference_bounds(fraction_index)
    report = {
        "schema_version": SCHEMA_VERSION,
        "analysis_role": ANALYSIS_ROLE,
        "created_at_utc": created_at_utc or datetime.now(timezone.utc).isoformat(),
        "status": status,
        "scientific_role": {
            "paper_evidence_allowed": False,
            "discovery_only": True,
            "simulation_run": True,
            "cnp2cnp_run": production,
            "injected_distance_for_test": not production,
            "reconstruction_run": True,
            "evaluation_run": True,
            "paired_capped_six_comparison": references_provided,
            "selects_simulator_parameters_from_accuracy": False,
            "freezes_paper_height_or_sampling_design": False,
        },
        "question": {
            "primary": (
                "Does fixed 50% representative-state sampling improve each "
                "reconstruction arm relative to its paired capped-six result "
                "enough to justify the larger observation problem?"
            ),
            "sampling_unit": "distinct_representative_genotype_state",
            "not_a_physical_cell_or_abundance_fraction": True,
            "within_arm_sampling_contrasts_only": True,
            "no_cross_output_problem_algorithm_ranking": True,
            "no_significance_testing": True,
            "not_paper_evidence": True,
        },
        "input": {
            "base_config": (
                base_config_path.relative_to(PROJECT_ROOT).as_posix()
                if base_config_path.is_relative_to(PROJECT_ROOT)
                else str(base_config_path)
            ),
            "base_config_sha256": base_config_sha256,
            "heights": list(heights),
            "schedule_by_height": {
                str(height): list(APPROVED_SCHEDULES[height]) for height in heights
            },
            "rule_y_positions": [0.6, 0.8, 1.0],
            "rounding": "ceiling",
            "condition_id": CONDITION_ID,
            "target_representative_state_fraction": TARGET_FRACTION,
            "biopsy_lower_bound": DEFAULT_LOWER_BOUND,
            "sample_size_formula": "min(N,max(6,ceil(0.5*N)))",
            "sampling_version": SAMPLING_VERSION,
            "replicates": replicates,
            "base_seed": base_seed,
            "distance_semantics": CNP2CNP_SEMANTICS_VERSION,
            "one_distance_computation_per_case": True,
            "arm_portfolio": [
                {
                    "arm_id": arm_id,
                    "algorithm": algorithm,
                    "problem": ARM_ENDPOINTS[arm_id]["problem"],
                    "declared_metrics": list(
                        ARM_ENDPOINTS[arm_id]["declared_metrics"]
                    ),
                }
                for arm_id, algorithm in REGISTERED_ARM_SPECS
            ],
        },
        "references": {
            "comparison": reference_metadata,
            "operational_preflight": authorization,
        },
        "resource_bound": {
            "planned_case_count": replicates * len(heights),
            "attempted_case_count": len(cases),
            **bounds,
            "arm_count_per_complete_case": len(ARM_IDS),
            "sequential_execution": True,
            "simulation_timeout_seconds": simulation_timeout_seconds,
            "distance_timeout_seconds": distance_timeout_seconds,
            "diagnostic_timeout_seconds": diagnostic_timeout_seconds,
            "reconstruction_timeout_seconds_per_arm": reconstruction_timeout_seconds,
            "evaluation_timeout_seconds_per_arm": evaluation_timeout_seconds,
            "cnp2cnp_process_timeout_seconds": cnp2cnp_process_timeout_seconds,
            "rss_limit_bytes_per_stage": rss_limit_bytes,
            "capture_limit_bytes": capture_limit_bytes,
            "max_case_dependency_failures": max_case_dependency_failures,
        },
        "cases": cases,
        "aggregate": aggregate,
        "distance_identity_counts": [
            {"identity_sha256": identity_hash, **identities[identity_hash]}
            for identity_hash in sorted(identities)
        ],
        "provenance": {
            "source_module": Path(__file__).relative_to(PROJECT_ROOT).as_posix(),
            "source_module_sha256": _file_sha256(Path(__file__)),
            "case_kernel_module": (
                Path(preflight.__file__).relative_to(PROJECT_ROOT).as_posix()
            ),
            "case_kernel_module_sha256": _file_sha256(Path(preflight.__file__)),
            "temporal_solver": {
                "implementation_version": TEMPORAL_ARBORESCENCE_SOLVER_VERSION,
                "source_module": "reconstructor_temporal.py",
                "source_module_sha256": _file_sha256(
                    PROJECT_ROOT / "reconstructor_temporal.py"
                ),
            },
            "reads_completed_compact_reference_reports": references_provided,
            "writes_raw_profiles": False,
            "writes_truth_or_reconstructed_trees": False,
            "writes_distance_matrices": False,
            "writes_simulator_node_identities": False,
            "replaces_failures_or_cases": False,
        },
    }
    validate_report(report)
    return report


def _walk_keys(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key)
            yield from _walk_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_keys(item)


def validate_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unknown dense reconstruction probe schema.")
    if report.get("analysis_role") != ANALYSIS_ROLE:
        raise ValueError("Dense reconstruction probe has the wrong role.")
    if report.get("status") not in {
        "complete",
        "complete_with_typed_failures",
        "stopped_at_dependency_failure_cap",
        "failed_reference_validation",
    }:
        raise ValueError("Dense reconstruction probe has an unknown status.")
    role = report.get("scientific_role", {})
    expected_role = {
        "paper_evidence_allowed": False,
        "discovery_only": True,
        "simulation_run": True,
        "reconstruction_run": True,
        "evaluation_run": True,
        "selects_simulator_parameters_from_accuracy": False,
        "freezes_paper_height_or_sampling_design": False,
    }
    for field, expected in expected_role.items():
        if role.get(field) is not expected:
            raise ValueError(f"Scientific role has invalid {field}.")

    input_record = report.get("input", {})
    heights = input_record.get("heights", [])
    replicates = input_record.get("replicates")
    if (
        not isinstance(heights, list)
        or not heights
        or len(set(heights)) != len(heights)
        or any(height not in APPROVED_SCHEDULES for height in heights)
    ):
        raise ValueError("Dense reconstruction report has invalid heights.")
    if (
        not isinstance(replicates, int)
        or isinstance(replicates, bool)
        or not 1 <= replicates <= MAX_REPLICATES
    ):
        raise ValueError("Dense reconstruction report has invalid replicates.")
    if (
        input_record.get("condition_id") != CONDITION_ID
        or input_record.get("target_representative_state_fraction")
        != TARGET_FRACTION
        or input_record.get("biopsy_lower_bound") != DEFAULT_LOWER_BOUND
        or input_record.get("sample_size_formula")
        != "min(N,max(6,ceil(0.5*N)))"
        or input_record.get("sampling_version") != SAMPLING_VERSION
        or input_record.get("distance_semantics") != CNP2CNP_SEMANTICS_VERSION
        or input_record.get("one_distance_computation_per_case") is not True
        or input_record.get("schedule_by_height")
        != {
            str(height): list(APPROVED_SCHEDULES[height])
            for height in heights
        }
    ):
        raise ValueError("Dense reconstruction report has invalid sampling design.")
    expected_keys = {
        (replicate_index, height)
        for replicate_index in range(replicates)
        for height in heights
    }
    cases = report.get("cases", [])
    actual_keys = {
        (case.get("replicate_index"), case.get("height")) for case in cases
    }
    if len(actual_keys) != len(cases) or not actual_keys <= expected_keys:
        raise ValueError("Dense reconstruction cases have duplicate or unknown keys.")
    if report["status"] == "complete" and actual_keys != expected_keys:
        raise ValueError("Complete dense reconstruction report lacks planned cases.")

    references_provided = report.get("references", {}).get("comparison", {}).get(
        "provided"
    ) is True
    if role.get("paired_capped_six_comparison") is not references_provided:
        raise ValueError("Dense reconstruction report has inconsistent pairing role.")
    if role.get("cnp2cnp_run") is role.get("injected_distance_for_test"):
        raise ValueError("Dense reconstruction report has inconsistent distance role.")
    allowed_case_statuses = {
        "complete",
        "simulation_or_reference_failure",
        "distance_failure",
        "diagnostic_failure",
    }
    for case in cases:
        if case.get("status") not in allowed_case_statuses:
            raise ValueError("Dense reconstruction case has an unknown status.")
        if case["status"] != "complete":
            continue
        summary = case.get("simulation_summary")
        if summary is None or case.get("distance") is None:
            raise ValueError("Complete dense case lacks dependencies.")
        for row in summary["sampling"]:
            expected = hybrid_sample_size(
                row["available_distinct_state_count"],
                TARGET_FRACTION,
                DEFAULT_LOWER_BOUND,
            )
            if row["realized_occurrence_count"] != expected:
                raise ValueError("Dense reconstruction sample size is inconsistent.")
        if tuple(arm.get("arm_id") for arm in case.get("arms", [])) != ARM_IDS:
            raise ValueError("Complete dense case has the wrong arm portfolio.")
        for arm in case["arms"]:
            if arm.get("status") not in {
                "success",
                "reconstruction_failure",
                "evaluation_failure",
            }:
                raise ValueError("Dense reconstruction arm has an unknown status.")
            if arm["status"] != "success":
                continue
            expected_metric_fields = {"grf"}
            if "ad_f1" in ARM_ENDPOINTS[arm["arm_id"]]["declared_metrics"]:
                expected_metric_fields.update(
                    {
                        "ad_f1",
                        "ad_precision",
                        "ad_recall",
                        "ad_iou",
                        "ad_counts",
                        "ad_f1_degenerate",
                        "ad_f1_degeneracy",
                    }
                )
            if set(arm["evaluation"]["metrics"]) != expected_metric_fields:
                raise ValueError("Successful dense arm exposes undeclared metrics.")
        if references_provided:
            if not case.get("reference_check", {}).get("performed") or not case[
                "reference_check"
            ].get("passed"):
                raise ValueError("Complete dense case lacks a passing reference check.")
            if case.get("truth_reference") is None or case.get("capped_comparison") is None:
                raise ValueError("Complete dense case lacks its paired references.")

    if report["status"] == "complete" and any(
        arm.get("status") != "success"
        for case in cases
        for arm in case.get("arms", [])
    ):
        raise ValueError("Complete dense reconstruction report contains an arm failure.")

    resource_bound = report.get("resource_bound", {})
    if (
        resource_bound.get("planned_case_count") != len(expected_keys)
        or resource_bound.get("attempted_case_count") != len(cases)
        or resource_bound.get("arm_count_per_complete_case") != len(ARM_IDS)
    ):
        raise ValueError("Dense reconstruction report has inconsistent case bounds.")

    if role.get("cnp2cnp_run") is True:
        if (
            replicates != DEFAULT_REPLICATES
            or tuple(heights) != tuple(APPROVED_SCHEDULES)
            or report.get("input", {}).get("base_seed") != DEFAULT_BASE_SEED
        ):
            raise ValueError("Production dense report differs from its frozen design.")
        comparison = report.get("references", {}).get("comparison", {})
        authorization = report.get("references", {}).get(
            "operational_preflight", {}
        )
        if (
            comparison.get("fraction_truth", {}).get("sha256")
            != preflight.EXPECTED_FRACTION_TRUTH_SHA256
            or comparison.get("sparse_reconstruction", {}).get("sha256")
            != preflight.EXPECTED_SPARSE_RECONSTRUCTION_SHA256
            or authorization.get("sha256") != EXPECTED_PREFLIGHT_SHA256
            or not comparison.get("registered_hashes_enforced")
            or not authorization.get("registered_hash_enforced")
        ):
            raise ValueError("Production dense report lacks registered authorization.")
        if report.get("resource_bound", {}).get("rss_limit_bytes_per_stage") != (
            DEFAULT_RSS_LIMIT_BYTES
        ):
            raise ValueError("Production dense report used a different RSS limit.")
        expected_bounds = {
            "maximum_selected_occurrence_count": (
                EXPECTED_MAX_SELECTED_OCCURRENCE_COUNT
            ),
            "maximum_unique_state_count": EXPECTED_MAX_UNIQUE_STATE_COUNT,
            "maximum_distance_matrix_cell_count": (
                EXPECTED_MAX_DISTANCE_MATRIX_CELL_COUNT
            ),
            "maximum_bidirectional_ordered_pair_count": (
                EXPECTED_MAX_ORDERED_PAIR_COUNT
            ),
        }
        if any(
            report.get("resource_bound", {}).get(field) != expected
            for field, expected in expected_bounds.items()
        ):
            raise ValueError("Production dense report has different case bounds.")
        if report["status"] == "complete" and not report.get("aggregate", {}).get(
            "common_seed_prefix_consistency", {}
        ).get("all_planned_common_seed_prefix_checks_available_and_passed"):
            raise ValueError("Complete production report lacks all prefix checks.")

    provenance = report.get("provenance", {})
    expected_source = Path(__file__).relative_to(PROJECT_ROOT).as_posix()
    expected_kernel = Path(preflight.__file__).relative_to(PROJECT_ROOT).as_posix()
    if (
        provenance.get("source_module") != expected_source
        or provenance.get("source_module_sha256") != _file_sha256(Path(__file__))
        or provenance.get("case_kernel_module") != expected_kernel
        or provenance.get("case_kernel_module_sha256")
        != _file_sha256(Path(preflight.__file__))
    ):
        raise ValueError("Dense reconstruction report has stale source identity.")
    solver = provenance.get("temporal_solver", {})
    if (
        solver.get("implementation_version")
        != TEMPORAL_ARBORESCENCE_SOLVER_VERSION
        or solver.get("source_module") != "reconstructor_temporal.py"
        or solver.get("source_module_sha256")
        != _file_sha256(PROJECT_ROOT / "reconstructor_temporal.py")
    ):
        raise ValueError("Dense reconstruction report lacks solver identity.")
    forbidden = {"cnp", "genome", "tree", "matrix", "node_id", "nodes", "links"}
    present = forbidden & set(_walk_keys(report))
    if present:
        raise ValueError(
            "Compact dense reconstruction report contains forbidden raw fields: "
            + ", ".join(sorted(present))
        )
    json.dumps(report, sort_keys=True, allow_nan=False)


def compact_summary(report: Mapping[str, Any], output: Path) -> dict[str, Any]:
    by_height = {}
    for height, block in report["aggregate"]["by_height"].items():
        by_height[height] = {
            "status_counts": block["status_counts"],
            "sampling_metric_means": {
                metric: summary["mean"]
                for metric, summary in block["sampling_metrics"].items()
            },
            "sampling_metric_maxima": {
                metric: summary["maximum"]
                for metric, summary in block["sampling_metrics"].items()
            },
            "paired_truth_metrics": {
                metric: {
                    "preferred_direction": row["preferred_direction"],
                    "mean_difference": row["difference"]["mean"],
                    "improvement_tie_worsening": row[
                        "improvement_tie_worsening"
                    ],
                }
                for metric, row in block["paired_truth_metrics"].items()
            },
            "arms": {
                arm_id: {
                    "status_counts": arm["status_counts"],
                    "paired_metric_mean_differences": {
                        metric: row["difference"]["mean"]
                        for metric, row in arm["paired_sampling_metrics"].items()
                    },
                    "paired_metric_wins_ties_losses": {
                        metric: row["wins_ties_losses"]
                        for metric, row in arm["paired_sampling_metrics"].items()
                    },
                    "mean_reconstruction_seconds": arm[
                        "reconstruction_wall_time_seconds"
                    ]["mean"],
                    "maximum_reconstruction_peak_rss_bytes": arm[
                        "reconstruction_peak_rss_bytes"
                    ]["maximum"],
                }
                for arm_id, arm in block["arms"].items()
            },
            "within_fraction50_contrasts": block[
                "within_fraction50_contrasts"
            ],
        }
    return {
        "schema_version": report["schema_version"],
        "analysis_role": report["analysis_role"],
        "status": report["status"],
        "output": str(output.resolve()),
        "references": report["references"],
        "planned_case_count": report["resource_bound"]["planned_case_count"],
        "attempted_case_count": report["resource_bound"]["attempted_case_count"],
        "common_seed_prefix_consistency": report["aggregate"][
            "common_seed_prefix_consistency"
        ],
        "by_height": by_height,
        "paired_height_endpoint_differences": report["aggregate"][
            "paired_height_endpoint_differences"
        ],
        "next_stage": "owner_and_agent_review_before_any_paper_sampling_freeze",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the fixed 50% H14/H24/H34 CTBF v5 reconstruction probe "
            "paired with the registered capped-six results."
        )
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--fraction-truth-report", type=Path, required=True)
    parser.add_argument("--sparse-reconstruction-report", type=Path, required=True)
    parser.add_argument("--dense-preflight-report", type=Path, required=True)
    parser.add_argument(
        "--simulation-timeout-seconds",
        type=int,
        default=DEFAULT_SIMULATION_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--distance-timeout-seconds",
        type=int,
        default=DEFAULT_DISTANCE_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--diagnostic-timeout-seconds",
        type=int,
        default=DEFAULT_DIAGNOSTIC_TIMEOUT_SECONDS,
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
    parser.add_argument(
        "--cnp2cnp-process-timeout-seconds",
        type=int,
        default=DEFAULT_CNP2CNP_PROCESS_TIMEOUT_SECONDS,
    )
    parser.add_argument("--rss-limit-bytes", type=int, default=DEFAULT_RSS_LIMIT_BYTES)
    parser.add_argument(
        "--capture-limit-bytes",
        type=int,
        default=DEFAULT_CAPTURE_LIMIT_BYTES,
    )
    parser.add_argument(
        "--max-case-dependency-failures",
        type=int,
        default=DEFAULT_MAX_CASE_DEPENDENCY_FAILURES,
    )
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    output = arguments.output.expanduser().resolve()
    if output.exists() and not arguments.overwrite:
        raise FileExistsError(
            f"Output already exists: {output}. Choose a new path or pass --overwrite."
        )
    report = run_probe(
        base_config_path=arguments.base_config,
        fraction_truth_report_path=arguments.fraction_truth_report,
        sparse_reconstruction_report_path=arguments.sparse_reconstruction_report,
        dense_preflight_report_path=arguments.dense_preflight_report,
        simulation_timeout_seconds=arguments.simulation_timeout_seconds,
        distance_timeout_seconds=arguments.distance_timeout_seconds,
        diagnostic_timeout_seconds=arguments.diagnostic_timeout_seconds,
        reconstruction_timeout_seconds=arguments.reconstruction_timeout_seconds,
        evaluation_timeout_seconds=arguments.evaluation_timeout_seconds,
        cnp2cnp_process_timeout_seconds=arguments.cnp2cnp_process_timeout_seconds,
        rss_limit_bytes=arguments.rss_limit_bytes,
        capture_limit_bytes=arguments.capture_limit_bytes,
        max_case_dependency_failures=arguments.max_case_dependency_failures,
        progress=arguments.progress,
    )
    write_json_atomic(output, report)
    print(json.dumps(compact_summary(report, output), indent=2, sort_keys=True))
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
