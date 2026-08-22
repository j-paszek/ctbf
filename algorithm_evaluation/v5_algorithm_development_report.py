"""Merge and compare CTBF v5 algorithm-development runs descriptively."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import math
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from algorithm_evaluation.paper_pipeline_contract import read_json
from algorithm_evaluation.process_isolation import (
    CASE_ARM_WORKER_UNIT,
    fresh_process_contract,
)
from algorithm_evaluation.v5_algorithm_development_common import (
    BASELINE_BY_FAMILY,
    BIOPSY_GUIDED_FULL_BASELINE_ID,
    BIOPSY_GUIDED_FULL_BOTTOM_TOP_INTERACTION_ROLE,
    BIOPSY_GUIDED_FULL_COUNTERPART_BY_PARTIAL_ID,
    BIOPSY_GUIDED_FULL_DEFAULT_ID,
    BIOPSY_GUIDED_FULL_FAMILY,
    BIOPSY_GUIDED_FULL_INCUMBENT_ID,
    BIOPSY_GUIDED_FULL_ROOTED_Q_DEFERRED_ID,
    COMPARISON_FAMILIES,
    DEFAULT_BLOCK_COUNT,
    HEIGHT_SCHEDULES,
    INCUMBENT_BY_FAMILY,
    INFERRED_COPY_INCUMBENT_ID,
    LEGACY_RUN_SCHEMA_VERSION,
    PARTIAL_BOTTOM_CANDIDATE_ROLE,
    PARTIAL_BOTTOM_CONTROL_ID,
    PARTIAL_BOTTOM_EXTENSION_ARM_SPECS,
    PARTIAL_BOTTOM_TOP_INTERACTION_ARM_ID,
    PARTIAL_BOTTOM_TOP_INTERACTION_ROLE,
    REPORT_SCHEMA_VERSION,
    RUN_SCHEMA_VERSION,
    ensure_new_output_root,
    numeric_summary,
    write_json,
)
from algorithm_evaluation.v5_algorithm_development_run import RESULT_NAME


TIE_TOLERANCE = 1e-12
WORST_BLOCK_COUNT = 5
PARTIAL_TOP_CONTROL_ID = "biopsy_guided_classical_r2"
PARTIAL_TOP_CANDIDATE_ROLE = "top_reconstruction_candidate"
PARTIAL_TOP_INTERACTION_ROLE = "top_radius_interaction_candidate"
PARTIAL_CLASSICAL_BY_RADIUS = {
    2: "biopsy_guided_classical_r2",
    4: "biopsy_guided_classical_r4",
}
PARTIAL_BINARY_BY_RADIUS = {
    2: "biopsy_guided_top_anticentral_binary_r2",
    4: "biopsy_guided_top_anticentral_binary_r4",
}
PARTIAL_DEFERRED_BINARY_ID = (
    "biopsy_guided_top_anticentral_binary_r2_bottom_deferred_tie"
)
PARTIAL_BOTTOM_TOP_BY_POLICY = {
    "default": {
        "classical": PARTIAL_CLASSICAL_BY_RADIUS[2],
        "binary_anticentral": PARTIAL_BINARY_BY_RADIUS[2],
    },
    "deferred": {
        "classical": PARTIAL_BOTTOM_TOP_INTERACTION_ARM_ID,
        "binary_anticentral": PARTIAL_DEFERRED_BINARY_ID,
    },
}
BIOPSY_GUIDED_FULL_BOTTOM_TOP_BY_POLICY = {
    "default": {
        "rooted_labeled_q": BIOPSY_GUIDED_FULL_BASELINE_ID,
        "binary_anticentral": BIOPSY_GUIDED_FULL_DEFAULT_ID,
    },
    "deferred": {
        "rooted_labeled_q": BIOPSY_GUIDED_FULL_ROOTED_Q_DEFERRED_ID,
        "binary_anticentral": BIOPSY_GUIDED_FULL_INCUMBENT_ID,
    },
}
PARTIAL_BOTTOM_AUDIT_FIELDS = (
    "child_decision_count",
    "multiple_plausible_parent_count",
    "minimum_distance_tie_count",
    "tie_parent_selected_count",
    "tie_deferred_count",
    "copy_up_count",
    "shared_parent_group_count",
)


def _height_labels() -> list[str]:
    return [f"H{height}" for height in sorted(HEIGHT_SCHEDULES)]


def _load_result(path: Path | str) -> tuple[Path, dict[str, Any]]:
    root = Path(path).expanduser().resolve()
    result_path = root / RESULT_NAME if root.is_dir() else root
    result = read_json(result_path)
    if result.get("schema_version") not in {
        RUN_SCHEMA_VERSION,
        LEGACY_RUN_SCHEMA_VERSION,
    }:
        raise ValueError(f"Unknown development-run schema in {result_path}.")
    if result.get("status") != "complete":
        raise ValueError(f"Development run is not complete: {result_path}.")
    if result.get("completed_record_count") != result.get("expected_record_count"):
        raise ValueError(f"Development run inventory is incomplete: {result_path}.")
    return result_path, result


def _finite_metric(record: Mapping[str, Any], name: str) -> float:
    metrics = record.get("metrics")
    if not isinstance(metrics, Mapping) or name not in metrics:
        raise ValueError(
            f"Successful record {record.get('case_id')}/{record.get('arm_id')} "
            f"lacks declared metric {name}."
        )
    value = float(metrics[name])
    if not math.isfinite(value):
        raise ValueError("Development metrics must be finite.")
    return value


def _resource_value(record: Mapping[str, Any], field: str) -> float | None:
    resources = record.get("resources") or {}
    values = []
    for stage in ("reconstruction", "evaluation"):
        stage_record = resources.get(stage)
        if not isinstance(stage_record, Mapping):
            continue
        if field == "wall_time_seconds":
            value = stage_record.get("wall_time_ns")
            if value is not None:
                values.append(float(value) / 1_000_000_000)
        elif field == "peak_rss_bytes":
            value = (stage_record.get("memory") or {}).get("peak_rss_bytes")
            if value is not None:
                values.append(float(value))
    if not values:
        return None
    return sum(values) if field == "wall_time_seconds" else max(values)


def _resource_methods(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    methods = []
    for record in records:
        for stage in ("reconstruction", "evaluation"):
            stage_record = (record.get("resources") or {}).get(stage)
            if isinstance(stage_record, Mapping):
                method = (stage_record.get("memory") or {}).get("method")
                if method is not None:
                    methods.append(str(method))
    return _counts(methods)


def _biopsy_layer_audit_summary(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    audits = [
        (record.get("reconstruction_metadata") or {}).get(
            "biopsy_layer_decision_audit"
        )
        for record in records
        if record.get("status") == "success"
    ]
    audits = [audit for audit in audits if isinstance(audit, Mapping)]
    if not audits:
        return None
    return {
        "record_count": len(audits),
        **{
            field: numeric_summary(
                int(audit[field]) for audit in audits if field in audit
            )
            for field in PARTIAL_BOTTOM_AUDIT_FIELDS
        },
    }


def _effect_summary(values: Sequence[float]) -> dict[str, Any] | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    worst = ordered[: min(WORST_BLOCK_COUNT, len(ordered))]
    return {
        **(numeric_summary(ordered) or {}),
        "worst_five_or_available_mean": float(statistics.fmean(worst)),
        "positive_count": sum(value > TIE_TOLERANCE for value in ordered),
        "tie_count": sum(abs(value) <= TIE_TOLERANCE for value in ordered),
        "negative_count": sum(value < -TIE_TOLERANCE for value in ordered),
        "conditional_gain_mean": (
            float(statistics.fmean(value for value in ordered if value > TIE_TOLERANCE))
            if any(value > TIE_TOLERANCE for value in ordered)
            else None
        ),
        "conditional_loss_mean": (
            float(statistics.fmean(value for value in ordered if value < -TIE_TOLERANCE))
            if any(value < -TIE_TOLERANCE for value in ordered)
            else None
        ),
    }


def _wtl(deltas: Sequence[float]) -> dict[str, Any]:
    wins = sum(delta > TIE_TOLERANCE for delta in deltas)
    ties = sum(abs(delta) <= TIE_TOLERANCE for delta in deltas)
    losses = sum(delta < -TIE_TOLERANCE for delta in deltas)
    eligible = len(deltas)
    return {
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "eligible": eligible,
        "win_score": ((wins + 0.5 * ties) / eligible if eligible else None),
    }


def pairwise_comparison(
    arm_a: str,
    arm_b: str,
    records_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
    case_inventory: Sequence[Mapping[str, Any]],
    *,
    primary_metric: str,
    complementary_metrics: Sequence[str],
) -> dict[str, Any]:
    """Return directional A-minus-B effects on identical successful cases."""
    return _pairwise_comparison_impl(
        arm_a,
        arm_b,
        records_by_key,
        case_inventory,
        primary_metric=primary_metric,
        complementary_metrics=complementary_metrics,
    )


def factorial_interaction_comparison(
    records_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
    case_inventory: Sequence[Mapping[str, Any]],
    *,
    primary_metric: str,
) -> dict[str, Any]:
    """Compare the binary-top effect at r2 with the same effect at r4."""
    return _factorial_interaction_impl(
        records_by_key,
        case_inventory,
        primary_metric=primary_metric,
    )


def bottom_top_factorial_interaction_comparison(
    records_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
    case_inventory: Sequence[Mapping[str, Any]],
    *,
    primary_metric: str,
) -> dict[str, Any]:
    """Compare the binary-top effect under deferred versus default bottom."""
    return _paired_difference_in_differences_impl(
        records_by_key,
        case_inventory,
        primary_metric=primary_metric,
        first_contrast=(
            PARTIAL_BOTTOM_TOP_BY_POLICY["deferred"]["binary_anticentral"],
            PARTIAL_BOTTOM_TOP_BY_POLICY["deferred"]["classical"],
        ),
        second_contrast=(
            PARTIAL_BOTTOM_TOP_BY_POLICY["default"]["binary_anticentral"],
            PARTIAL_BOTTOM_TOP_BY_POLICY["default"]["classical"],
        ),
        definition=(
            "(binary top minus classical top with deferred bottom) minus "
            "(binary top minus classical top with default bottom)"
        ),
        positive_direction=(
            "binary_top_advantage_is_larger_with_deferred_bottom"
        ),
    )


def full_bottom_top_factorial_interaction_comparison(
    records_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
    case_inventory: Sequence[Mapping[str, Any]],
    *,
    primary_metric: str,
) -> dict[str, Any]:
    """Compare the binary-top effect under deferred versus default bottom."""
    return _paired_difference_in_differences_impl(
        records_by_key,
        case_inventory,
        primary_metric=primary_metric,
        first_contrast=(
            BIOPSY_GUIDED_FULL_BOTTOM_TOP_BY_POLICY["deferred"][
                "binary_anticentral"
            ],
            BIOPSY_GUIDED_FULL_BOTTOM_TOP_BY_POLICY["deferred"][
                "rooted_labeled_q"
            ],
        ),
        second_contrast=(
            BIOPSY_GUIDED_FULL_BOTTOM_TOP_BY_POLICY["default"][
                "binary_anticentral"
            ],
            BIOPSY_GUIDED_FULL_BOTTOM_TOP_BY_POLICY["default"][
                "rooted_labeled_q"
            ],
        ),
        definition=(
            "(binary anticentral top minus rooted-labeled Q top with deferred "
            "bottom) minus the same top effect with default bottom"
        ),
        positive_direction=(
            "binary_anticentral_top_advantage_is_larger_with_deferred_bottom"
        ),
    )


def _factorial_interaction_impl(
    records_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
    case_inventory: Sequence[Mapping[str, Any]],
    *,
    primary_metric: str,
) -> dict[str, Any]:
    return _paired_difference_in_differences_impl(
        records_by_key,
        case_inventory,
        primary_metric=primary_metric,
        first_contrast=(
            PARTIAL_BINARY_BY_RADIUS[2],
            PARTIAL_CLASSICAL_BY_RADIUS[2],
        ),
        second_contrast=(
            PARTIAL_BINARY_BY_RADIUS[4],
            PARTIAL_CLASSICAL_BY_RADIUS[4],
        ),
        definition=(
            "(binary top minus classical top at r2) minus "
            "(binary top minus classical top at r4)"
        ),
        positive_direction="binary_top_advantage_is_larger_at_r2",
    )


def _paired_difference_in_differences_impl(
    records_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
    case_inventory: Sequence[Mapping[str, Any]],
    *,
    primary_metric: str,
    first_contrast: tuple[str, str],
    second_contrast: tuple[str, str],
    definition: str,
    positive_direction: str,
) -> dict[str, Any]:
    deltas_by_height: dict[int, list[float]] = {
        int(height): [] for height in HEIGHT_SCHEDULES
    }
    deltas_by_block: dict[int, dict[int, float]] = {}
    condition_effects = []
    complete_case_count = 0
    incomplete_case_count = 0
    required_ids = (*first_contrast, *second_contrast)
    for case in case_inventory:
        case_key = str(case["case_id"])
        records = {
            arm_id: records_by_key[(case_key, arm_id)]
            for arm_id in required_ids
        }
        if not all(record.get("status") == "success" for record in records.values()):
            incomplete_case_count += 1
            continue
        complete_case_count += 1
        first_effect = _finite_metric(
            records[first_contrast[0]],
            primary_metric,
        ) - _finite_metric(
            records[first_contrast[1]],
            primary_metric,
        )
        second_effect = _finite_metric(
            records[second_contrast[0]],
            primary_metric,
        ) - _finite_metric(
            records[second_contrast[1]],
            primary_metric,
        )
        delta = first_effect - second_effect
        height = int(case["height"])
        block_index = int(case["block_index"])
        deltas_by_height[height].append(delta)
        deltas_by_block.setdefault(block_index, {})[height] = delta
        condition_effects.append(
            {
                "case_id": case_key,
                "block_index": block_index,
                "height": height,
                "delta": float(delta),
            }
        )

    combined = [
        delta
        for height in sorted(deltas_by_height)
        for delta in deltas_by_height[height]
    ]
    complete_block_effects = [
        {
            "block_index": block,
            "delta": float(
                statistics.fmean(
                    values[height] for height in sorted(HEIGHT_SCHEDULES)
                )
            ),
        }
        for block, values in sorted(deltas_by_block.items())
        if set(values) == set(HEIGHT_SCHEDULES)
    ]
    complete_block_deltas = [row["delta"] for row in complete_block_effects]
    return {
        "definition": definition,
        "positive_direction": positive_direction,
        "primary_metric": primary_metric,
        "complete_case_count": complete_case_count,
        "incomplete_case_count": incomplete_case_count,
        "by_height": {
            str(height): {
                "wtl": _wtl(deltas),
                "effect": _effect_summary(deltas),
            }
            for height, deltas in sorted(deltas_by_height.items())
        },
        "combined_conditions": {
            "dependence_warning": (
                f"{', '.join(_height_labels())} share a truth block; these are "
                "descriptive, not independent replicates."
            ),
            "wtl": _wtl(combined),
            "effect": _effect_summary(combined),
        },
        "independent_block_effect": {
            "definition": (
                "mean paired difference-in-differences across "
                f"{', '.join(_height_labels())} within a truth block"
            ),
            "complete_block_count": len(complete_block_deltas),
            "effect": _effect_summary(complete_block_deltas),
            "worst_five_or_available_blocks": sorted(
                complete_block_effects,
                key=lambda row: (row["delta"], row["block_index"]),
            )[: min(WORST_BLOCK_COUNT, len(complete_block_effects))],
        },
        "worst_five_or_available_conditions": sorted(
            condition_effects,
            key=lambda row: (row["delta"], row["block_index"], row["height"]),
        )[: min(WORST_BLOCK_COUNT, len(condition_effects))],
    }


def _pairwise_comparison_impl(
    arm_a: str,
    arm_b: str,
    records_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
    case_inventory: Sequence[Mapping[str, Any]],
    *,
    primary_metric: str,
    complementary_metrics: Sequence[str],
) -> dict[str, Any]:
    deltas_by_height: dict[int, list[float]] = {
        int(height): [] for height in HEIGHT_SCHEDULES
    }
    deltas_by_block: dict[int, dict[int, float]] = {}
    complementary: dict[str, list[float]] = {
        metric: [] for metric in complementary_metrics
    }
    dominance_count = 0
    condition_effects = []
    joint_success = 0
    a_only_success = 0
    b_only_success = 0
    both_failure = 0

    for case in case_inventory:
        case_key = str(case["case_id"])
        a = records_by_key[(case_key, arm_a)]
        b = records_by_key[(case_key, arm_b)]
        a_success = a.get("status") == "success"
        b_success = b.get("status") == "success"
        if a_success and not b_success:
            a_only_success += 1
            continue
        if b_success and not a_success:
            b_only_success += 1
            continue
        if not a_success and not b_success:
            both_failure += 1
            continue

        joint_success += 1
        height = int(case["height"])
        delta = _finite_metric(a, primary_metric) - _finite_metric(b, primary_metric)
        deltas_by_height[height].append(delta)
        deltas_by_block.setdefault(int(case["block_index"]), {})[height] = delta
        condition_effects.append(
            {
                "case_id": case_key,
                "block_index": int(case["block_index"]),
                "height": height,
                "delta": float(delta),
            }
        )

        declared_deltas = [delta]
        for metric in complementary_metrics:
            metric_delta = _finite_metric(a, metric) - _finite_metric(b, metric)
            complementary[metric].append(metric_delta)
            declared_deltas.append(metric_delta)
        if all(value >= -TIE_TOLERANCE for value in declared_deltas):
            dominance_count += 1

    combined = [
        delta
        for height in sorted(deltas_by_height)
        for delta in deltas_by_height[height]
    ]
    complete_block_effects = [
        {
            "block_index": block,
            "delta": float(
                statistics.fmean(
                    values[height] for height in sorted(HEIGHT_SCHEDULES)
                )
            ),
        }
        for block, values in sorted(deltas_by_block.items())
        if set(values) == set(HEIGHT_SCHEDULES)
    ]
    complete_block_deltas = [row["delta"] for row in complete_block_effects]
    worst_conditions = sorted(
        condition_effects,
        key=lambda row: (row["delta"], row["block_index"], row["height"]),
    )[: min(WORST_BLOCK_COUNT, len(condition_effects))]
    worst_blocks = sorted(
        complete_block_effects,
        key=lambda row: (row["delta"], row["block_index"]),
    )[: min(WORST_BLOCK_COUNT, len(complete_block_effects))]
    return {
        "arm_a": arm_a,
        "arm_b": arm_b,
        "delta_direction": "arm_a_minus_arm_b_higher_is_better",
        "primary_metric": primary_metric,
        "joint_success_count": joint_success,
        "a_only_success_count": a_only_success,
        "b_only_success_count": b_only_success,
        "both_failure_count": both_failure,
        "by_height": {
            str(height): {
                "wtl": _wtl(deltas),
                "effect": _effect_summary(deltas),
            }
            for height, deltas in sorted(deltas_by_height.items())
        },
        "combined_conditions": {
            "dependence_warning": (
                f"{', '.join(_height_labels())} share a truth block; these are "
                "descriptive, not independent replicates."
            ),
            "wtl": _wtl(combined),
            "effect": _effect_summary(combined),
        },
        "independent_block_effect": {
            "definition": (
                f"mean of paired {', '.join(_height_labels())} deltas within "
                "a truth block"
            ),
            "complete_block_count": len(complete_block_deltas),
            "effect": _effect_summary(complete_block_deltas),
            "worst_five_or_available_blocks": worst_blocks,
        },
        "worst_five_or_available_conditions": worst_conditions,
        "complementary_metric_effects": {
            metric: _effect_summary(values)
            for metric, values in complementary.items()
        },
        "all_declared_metric_nonstrict_dominance": {
            "count": dominance_count,
            "eligible": joint_success,
            "fraction": (
                dominance_count / joint_success if joint_success else None
            ),
            "definition": (
                "A is no worse than B within tolerance on the primary and every "
                "shared declared complementary metric for one condition."
            ),
        },
    }


def _algorithm_summary(
    arm_id: str,
    spec: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    semantic_gate: Mapping[str, Any],
) -> dict[str, Any]:
    successes = [record for record in records if record["status"] == "success"]
    failures = [record for record in records if record["status"] != "success"]
    primary = str(spec["primary_metric"])
    runtime_values = [
        value
        for record in records
        if (value := _resource_value(record, "wall_time_seconds")) is not None
    ]
    memory_values = [
        value
        for record in records
        if (value := _resource_value(record, "peak_rss_bytes")) is not None
    ]
    gate_fixtures = semantic_gate.get("fixture_results", [])
    return {
        "arm_id": arm_id,
        "family": spec["family"],
        "problem": spec["problem"],
        "role": spec.get("role"),
        "d0_status": semantic_gate.get("status", "not_recorded"),
        "d0_stability": {
            "fixture_count": len(gate_fixtures),
            "within_level_permutation_changed_fixture_ids": [
                row["fixture_id"]
                for row in gate_fixtures
                if not row["within_level_permutation_same_topology"]
            ],
            "different_seed_changed_fixture_ids": [
                row["fixture_id"]
                for row in gate_fixtures
                if not row["different_seed_same_topology"]
            ],
        },
        "primary_metric": primary,
        "record_count": len(records),
        "success_count": len(successes),
        "failure_count": len(failures),
        "failure_types": _counts(
            (
                (record.get("failure") or {}).get("type")
                or (record.get("failure") or {}).get("code")
                or "declared_evaluation_failure"
            )
            for record in failures
        ),
        "primary_score": {
            "combined": numeric_summary(
                _finite_metric(record, primary) for record in successes
            ),
            "by_height": {
                str(height): numeric_summary(
                    _finite_metric(record, primary)
                    for record in successes
                    if int(record["height"]) == int(height)
                )
                for height in HEIGHT_SCHEDULES
            },
        },
        "runtime_seconds": numeric_summary(runtime_values),
        "peak_rss_bytes": numeric_summary(memory_values),
        "resource_measurement_methods": _resource_methods(records),
        "reconstructed_node_count": numeric_summary(
            record["tree_summary"]["node_count"] for record in successes
        ),
        "reconstructed_leaf_count": numeric_summary(
            record["tree_summary"]["leaf_count"]
            for record in successes
            if "leaf_count" in record["tree_summary"]
        ),
        "reconstructed_maximum_depth": numeric_summary(
            record["tree_summary"]["maximum_depth"]
            for record in successes
            if "maximum_depth" in record["tree_summary"]
        ),
        "reconstructed_maximum_out_degree": numeric_summary(
            record["tree_summary"]["maximum_out_degree"]
            for record in successes
            if "maximum_out_degree" in record["tree_summary"]
        ),
        "unlabeled_node_count": numeric_summary(
            record["tree_summary"]["unlabeled_node_count"]
            for record in successes
            if "unlabeled_node_count" in record["tree_summary"]
        ),
        "inferred_copy_occurrence_count": numeric_summary(
            record["tree_summary"]["inferred_copy_occurrence_count"]
            for record in successes
        ),
        "top_created_node_count": numeric_summary(
            record["reconstruction_metadata"]["top_created_node_count"]
            for record in successes
            if "top_created_node_count" in record.get("reconstruction_metadata", {})
        ),
        "top_labels_cleared_count": numeric_summary(
            record["reconstruction_metadata"]["top_labels_cleared_count"]
            for record in successes
            if "top_labels_cleared_count" in record.get("reconstruction_metadata", {})
        ),
        "top_labels_retained_count": numeric_summary(
            record["reconstruction_metadata"]["top_labels_retained_count"]
            for record in successes
            if "top_labels_retained_count"
            in record.get("reconstruction_metadata", {})
        ),
        "biopsy_layer_decision_audit": _biopsy_layer_audit_summary(successes),
        "observation_coverage": numeric_summary(
            record["observation_coverage"]["fraction"] for record in successes
        ),
    }


def _counts(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _matrix(
    arm_ids: Sequence[str],
    pairwise: Mapping[tuple[str, str], Mapping[str, Any]],
    accessor,
) -> dict[str, dict[str, Any]]:
    return {
        arm_a: {
            arm_b: accessor(pairwise[(arm_a, arm_b)])
            for arm_b in arm_ids
        }
        for arm_a in arm_ids
    }


def _leaderboard_rows(
    family: str,
    arm_ids: Sequence[str],
    specs: Mapping[str, Mapping[str, Any]],
    summaries: Mapping[str, Mapping[str, Any]],
    pairwise: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    incumbent = INCUMBENT_BY_FAMILY[family]
    baseline = BASELINE_BY_FAMILY[family]
    if incumbent not in arm_ids:
        raise ValueError(
            f"A {family} report requires fixed incumbent {incumbent!r}."
        )
    if baseline not in arm_ids:
        raise ValueError(f"A {family} report requires baseline {baseline!r}.")
    rows = []
    for arm_id in arm_ids:
        summary = summaries[arm_id]
        comparison = pairwise[(arm_id, incumbent)]
        baseline_comparison = pairwise[(arm_id, baseline)]
        combined = comparison["combined_conditions"]
        block = comparison["independent_block_effect"]["effect"]
        runtime = summary["runtime_seconds"]
        row = {
            "arm_id": arm_id,
            "role": specs[arm_id].get("role"),
            "d0_status": summary["d0_status"],
            "d0_permutation_changed_fixture_count": len(
                summary["d0_stability"][
                    "within_level_permutation_changed_fixture_ids"
                ]
            ),
            "d0_seed_changed_fixture_count": len(
                summary["d0_stability"]["different_seed_changed_fixture_ids"]
            ),
            "primary_metric": specs[arm_id]["primary_metric"],
            "success_count": summary["success_count"],
            "failure_count": summary["failure_count"],
            **{
                f"h{height}_primary_mean": _nested(
                    summary,
                    "primary_score",
                    "by_height",
                    str(height),
                    "mean",
                )
                for height in sorted(HEIGHT_SCHEDULES)
            },
            "combined_primary_mean": _nested(summary, "primary_score", "combined", "mean"),
            "incumbent_id": incumbent,
            "baseline_id": baseline,
            "vs_baseline_wins": baseline_comparison["combined_conditions"]["wtl"][
                "wins"
            ],
            "vs_baseline_ties": baseline_comparison["combined_conditions"]["wtl"][
                "ties"
            ],
            "vs_baseline_losses": baseline_comparison["combined_conditions"][
                "wtl"
            ]["losses"],
            "vs_baseline_win_score": baseline_comparison["combined_conditions"][
                "wtl"
            ]["win_score"],
            "vs_baseline_mean_block_delta": _nested(
                baseline_comparison,
                "independent_block_effect",
                "effect",
                "mean",
            ),
            "vs_incumbent_wins": combined["wtl"]["wins"],
            "vs_incumbent_ties": combined["wtl"]["ties"],
            "vs_incumbent_losses": combined["wtl"]["losses"],
            "vs_incumbent_eligible": combined["wtl"]["eligible"],
            "vs_incumbent_win_score": combined["wtl"]["win_score"],
            "vs_incumbent_mean_block_delta": None if block is None else block["mean"],
            "vs_incumbent_median_block_delta": None if block is None else block["median"],
            "vs_incumbent_minimum_block_delta": None if block is None else block["minimum"],
            "vs_incumbent_worst_five_block_mean": (
                None if block is None else block["worst_five_or_available_mean"]
            ),
            "median_runtime_seconds": None if runtime is None else runtime["median"],
            "maximum_peak_rss_bytes": _nested(summary, "peak_rss_bytes", "maximum"),
        }
        rows.append(row)

    def descending(value: Any) -> float:
        return -float(value) if value is not None else math.inf

    rows.sort(
        key=lambda row: (
            0 if row["d0_status"] == "passed" else 1,
            int(row["failure_count"]),
            descending(row["vs_incumbent_win_score"]),
            descending(row["vs_incumbent_mean_block_delta"]),
            descending(row["vs_incumbent_worst_five_block_mean"]),
            (
                float(row["median_runtime_seconds"])
                if row["median_runtime_seconds"] is not None
                else math.inf
            ),
            str(row["arm_id"]).encode("utf-8"),
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["display_rank"] = rank
    return rows


def _pareto_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Describe a small no-worse frontier without converting it to a winner."""
    maximize = (
        "vs_incumbent_win_score",
        "vs_incumbent_mean_block_delta",
        "vs_incumbent_worst_five_block_mean",
    )
    minimize = (
        "failure_count",
        "median_runtime_seconds",
        "maximum_peak_rss_bytes",
    )

    def dominates(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
        left_passed = left["d0_status"] == "passed"
        right_passed = right["d0_status"] == "passed"
        if left_passed != right_passed:
            return left_passed
        if any(left[field] is None or right[field] is None for field in (*maximize, *minimize)):
            return False
        no_worse = all(
            float(left[field]) >= float(right[field]) - TIE_TOLERANCE
            for field in maximize
        ) and all(
            float(left[field]) <= float(right[field]) + TIE_TOLERANCE
            for field in minimize
        )
        strictly_better = any(
            float(left[field]) > float(right[field]) + TIE_TOLERANCE
            for field in maximize
        ) or any(
            float(left[field]) < float(right[field]) - TIE_TOLERANCE
            for field in minimize
        )
        return no_worse and strictly_better

    dominated_by = {
        str(row["arm_id"]): sorted(
            str(other["arm_id"])
            for other in rows
            if other is not row and dominates(other, row)
        )
        for row in rows
    }
    return {
        "definition": (
            "No other family arm is no worse on D0 status, failures, incumbent "
            "win score, mean and worst-five block effects, median runtime, and "
            "maximum peak RSS, with at least one strict improvement."
        ),
        "automatic_winner": False,
        "frontier_arm_ids": sorted(
            arm_id for arm_id, dominators in dominated_by.items() if not dominators
        ),
        "dominated_by": dominated_by,
    }


def _nested(value: Mapping[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if current is None or not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _partial_bottom_mechanism_row(
    arm_id: str,
    spec: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "arm_id": arm_id,
        "biopsy_preset": spec.get("biopsy_preset"),
        **{
            f"mean_{field}": _nested(
                summary,
                "biopsy_layer_decision_audit",
                field,
                "mean",
            )
            for field in PARTIAL_BOTTOM_AUDIT_FIELDS
        },
        "mean_reconstructed_node_count": _nested(
            summary,
            "reconstructed_node_count",
            "mean",
        ),
        "mean_reconstructed_maximum_out_degree": _nested(
            summary,
            "reconstructed_maximum_out_degree",
            "mean",
        ),
    }


def build_report(
    results: Sequence[Mapping[str, Any]],
    *,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    if not results:
        raise ValueError("At least one completed development run is required.")
    bank_ids = {str(result["bank_id"]) for result in results}
    bank_roots = {str(result["bank_root"]) for result in results}
    block_counts = {int(result["block_count"]) for result in results}
    run_ids = [str(result["run_id"]) for result in results]
    if len(bank_ids) != 1 or len(bank_roots) != 1 or len(block_counts) != 1:
        raise ValueError("All development runs must use the exact same stored bank.")
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("Development result run ids must be unique.")
    block_count = next(iter(block_counts))
    expected_record_execution = fresh_process_contract(CASE_ARM_WORKER_UNIT)
    resource_execution_by_run = [
        {
            "run_id": str(result["run_id"]),
            "record_execution": (
                dict(result["resources"]["record_execution"])
                if isinstance(result.get("resources"), Mapping)
                and isinstance(
                    result["resources"].get("record_execution"), Mapping
                )
                else None
            ),
            "fresh_process_qualified": (
                isinstance(result.get("resources"), Mapping)
                and result["resources"].get("record_execution")
                == expected_record_execution
            ),
        }
        for result in results
    ]
    all_resources_fresh_process_qualified = all(
        row["fresh_process_qualified"] for row in resource_execution_by_run
    )

    specs: dict[str, Mapping[str, Any]] = {}
    semantic_gates: dict[str, Mapping[str, Any]] = {}
    records_by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    cases: dict[str, dict[str, Any]] = {}
    for result in results:
        run_specs = {spec["arm_id"]: spec for spec in result["arm_specs"]}
        for arm_id, spec in run_specs.items():
            if arm_id in specs:
                raise ValueError(
                    f"Arm {arm_id!r} occurs in more than one merged run; use one result."
                )
            specs[arm_id] = spec
        run_gates = result.get("semantic_gate_by_arm", {})
        if run_gates and set(run_gates) != set(run_specs):
            raise ValueError("A development run has an incomplete D0 gate inventory.")
        for arm_id in run_specs:
            semantic_gates[arm_id] = run_gates.get(
                arm_id,
                {"status": "not_recorded"},
            )
        for record in result["records"]:
            arm_id = str(record["arm_id"])
            if arm_id not in run_specs:
                raise ValueError("A result record has no matching arm declaration.")
            case_key = str(record["case_id"])
            key = (case_key, arm_id)
            if key in records_by_key:
                raise ValueError(f"Duplicate merged result record {key!r}.")
            records_by_key[key] = record
            case_descriptor = {
                "case_id": case_key,
                "block_index": int(record["block_index"]),
                "height": int(record["height"]),
            }
            if case_key in cases and cases[case_key] != case_descriptor:
                raise ValueError(f"Case descriptor changed for {case_key!r}.")
            cases[case_key] = case_descriptor

    expected_case_count = block_count * len(HEIGHT_SCHEDULES)
    if len(cases) != expected_case_count:
        raise ValueError(
            f"Expected {expected_case_count} bank cases, observed {len(cases)}."
        )
    case_inventory = sorted(
        cases.values(),
        key=lambda case: (case["block_index"], case["height"]),
    )
    for arm_id in specs:
        observed = {
            case_id
            for case_id, candidate_id in records_by_key
            if candidate_id == arm_id
        }
        if observed != set(cases):
            raise ValueError(f"Arm {arm_id!r} does not cover every bank case once.")

    records_by_arm = {
        arm_id: [records_by_key[(case["case_id"], arm_id)] for case in case_inventory]
        for arm_id in specs
    }
    summaries = {
        arm_id: _algorithm_summary(
            arm_id,
            specs[arm_id],
            records,
            semantic_gates[arm_id],
        )
        for arm_id, records in records_by_arm.items()
    }
    families = {}
    for family in COMPARISON_FAMILIES:
        arm_ids = sorted(
            arm_id for arm_id, spec in specs.items() if spec["family"] == family
        )
        if not arm_ids:
            continue
        primary_metrics = {str(specs[arm_id]["primary_metric"]) for arm_id in arm_ids}
        if len(primary_metrics) != 1:
            raise ValueError(f"Family {family} has inconsistent primary metrics.")
        primary_metric = next(iter(primary_metrics))
        complementary = sorted(
            set.intersection(
                *(set(specs[arm_id].get("complementary_metrics", [])) for arm_id in arm_ids)
            )
        )
        pairwise = {
            (arm_a, arm_b): pairwise_comparison(
                arm_a,
                arm_b,
                records_by_key,
                case_inventory,
                primary_metric=primary_metric,
                complementary_metrics=complementary,
            )
            for arm_a in arm_ids
            for arm_b in arm_ids
        }
        leaderboard = _leaderboard_rows(
            family,
            arm_ids,
            specs,
            summaries,
            pairwise,
        )
        top_layer_comparison = None
        top_radius_interaction = None
        bottom_layer_comparison = None
        bottom_top_factorial = None
        full_bottom_top_factorial = None
        if family == "partial":
            top_candidate_ids = sorted(
                arm_id
                for arm_id in arm_ids
                if specs[arm_id].get("role") == PARTIAL_TOP_CANDIDATE_ROLE
            )
            if top_candidate_ids:
                if PARTIAL_TOP_CONTROL_ID not in arm_ids:
                    raise ValueError(
                        "The partial top-layer screen requires its fixed r2 "
                        "classical-Q control."
                    )
                top_layer_comparison = {
                    "control_id": PARTIAL_TOP_CONTROL_ID,
                    "candidate_ids": top_candidate_ids,
                    "pairwise_records": [
                        pairwise[(arm_id, PARTIAL_TOP_CONTROL_ID)]
                        for arm_id in top_candidate_ids
                    ],
                }
            observed_bottom_ids = {
                arm_id
                for arm_id in arm_ids
                if specs[arm_id].get("role") == PARTIAL_BOTTOM_CANDIDATE_ROLE
            }
            if observed_bottom_ids:
                expected_bottom_ids = {
                    spec.arm_id for spec in PARTIAL_BOTTOM_EXTENSION_ARM_SPECS
                }
                if observed_bottom_ids != expected_bottom_ids:
                    raise ValueError(
                        "The partial bottom-layer screen requires all six "
                        "approved candidate rows."
                    )
                bottom_candidate_ids = [
                    spec.arm_id for spec in PARTIAL_BOTTOM_EXTENSION_ARM_SPECS
                ]
                if PARTIAL_BOTTOM_CONTROL_ID not in arm_ids:
                    raise ValueError(
                        "The partial bottom-layer screen requires its fixed "
                        "default-r2 binary-top control."
                    )
                bottom_layer_comparison = {
                    "control_id": PARTIAL_BOTTOM_CONTROL_ID,
                    "fixed_radius": 2,
                    "fixed_top_reconstruction": "binary_anticentral",
                    "candidate_ids": bottom_candidate_ids,
                    "pairwise_records": [
                        pairwise[(arm_id, PARTIAL_BOTTOM_CONTROL_ID)]
                        for arm_id in bottom_candidate_ids
                    ],
                    "mechanism_rows": [
                        _partial_bottom_mechanism_row(
                            arm_id,
                            specs[arm_id],
                            summaries[arm_id],
                        )
                        for arm_id in (
                            PARTIAL_BOTTOM_CONTROL_ID,
                            *bottom_candidate_ids,
                        )
                    ],
                }
            bottom_top_interaction_ids = {
                arm_id
                for arm_id in arm_ids
                if specs[arm_id].get("role")
                == PARTIAL_BOTTOM_TOP_INTERACTION_ROLE
            }
            if bottom_top_interaction_ids:
                if bottom_top_interaction_ids != {
                    PARTIAL_BOTTOM_TOP_INTERACTION_ARM_ID
                }:
                    raise ValueError(
                        "The partial bottom-by-top factorial has an unexpected "
                        "interaction-arm inventory."
                    )
                required_bottom_top_ids = {
                    arm_id
                    for policy in PARTIAL_BOTTOM_TOP_BY_POLICY.values()
                    for arm_id in policy.values()
                }
                if not required_bottom_top_ids <= set(arm_ids):
                    raise ValueError(
                        "The partial bottom-by-top factorial requires the "
                        "complete default/deferred by classical/binary design."
                    )
                bottom_top_factorial = {
                    "factor_levels": {
                        "bottom_policy": ["default", "deferred"],
                        "top_reconstruction": [
                            "classical",
                            "binary_anticentral",
                        ],
                    },
                    "arm_ids": PARTIAL_BOTTOM_TOP_BY_POLICY,
                    "top_effect_by_bottom": {
                        bottom_policy: pairwise[
                            (
                                top_ids["binary_anticentral"],
                                top_ids["classical"],
                            )
                        ]
                        for bottom_policy, top_ids in (
                            PARTIAL_BOTTOM_TOP_BY_POLICY.items()
                        )
                    },
                    "bottom_effect_by_top": {
                        top_method: pairwise[
                            (
                                PARTIAL_BOTTOM_TOP_BY_POLICY["deferred"][
                                    top_method
                                ],
                                PARTIAL_BOTTOM_TOP_BY_POLICY["default"][
                                    top_method
                                ],
                            )
                        ]
                        for top_method in (
                            "classical",
                            "binary_anticentral",
                        )
                    },
                    "difference_in_differences": (
                        bottom_top_factorial_interaction_comparison(
                            records_by_key,
                            case_inventory,
                            primary_metric=primary_metric,
                        )
                    ),
                }
            interaction_candidate_ids = sorted(
                arm_id
                for arm_id in arm_ids
                if specs[arm_id].get("role") == PARTIAL_TOP_INTERACTION_ROLE
            )
            if interaction_candidate_ids:
                expected_interaction_ids = {PARTIAL_BINARY_BY_RADIUS[4]}
                if set(interaction_candidate_ids) != expected_interaction_ids:
                    raise ValueError(
                        "The partial radius interaction has an unexpected arm inventory."
                    )
                required_factorial_ids = {
                    *PARTIAL_CLASSICAL_BY_RADIUS.values(),
                    *PARTIAL_BINARY_BY_RADIUS.values(),
                }
                if not required_factorial_ids <= set(arm_ids):
                    raise ValueError(
                        "The partial radius interaction requires the complete "
                        "r2/r4 by classical/binary-top factorial."
                    )
                top_radius_interaction = {
                    "factor_levels": {
                        "radius": [2, 4],
                        "top_reconstruction": ["classical", "binary_anticentral"],
                    },
                    "arm_ids": {
                        "classical_by_radius": {
                            str(radius): arm_id
                            for radius, arm_id in PARTIAL_CLASSICAL_BY_RADIUS.items()
                        },
                        "binary_by_radius": {
                            str(radius): arm_id
                            for radius, arm_id in PARTIAL_BINARY_BY_RADIUS.items()
                        },
                    },
                    "top_effect_by_radius": {
                        str(radius): pairwise[
                            (
                                PARTIAL_BINARY_BY_RADIUS[radius],
                                PARTIAL_CLASSICAL_BY_RADIUS[radius],
                            )
                        ]
                        for radius in (2, 4)
                    },
                    "radius_effect_by_top": {
                        "classical": pairwise[
                            (
                                PARTIAL_CLASSICAL_BY_RADIUS[2],
                                PARTIAL_CLASSICAL_BY_RADIUS[4],
                            )
                        ],
                        "binary_anticentral": pairwise[
                            (
                                PARTIAL_BINARY_BY_RADIUS[2],
                                PARTIAL_BINARY_BY_RADIUS[4],
                            )
                        ],
                    },
                    "difference_in_differences": factorial_interaction_comparison(
                        records_by_key,
                        case_inventory,
                        primary_metric=primary_metric,
                    ),
                }
        if family == BIOPSY_GUIDED_FULL_FAMILY:
            full_interaction_ids = {
                arm_id
                for arm_id in arm_ids
                if specs[arm_id].get("role")
                == BIOPSY_GUIDED_FULL_BOTTOM_TOP_INTERACTION_ROLE
            }
            if full_interaction_ids:
                if full_interaction_ids != {
                    BIOPSY_GUIDED_FULL_ROOTED_Q_DEFERRED_ID
                }:
                    raise ValueError(
                        "The fully labeled bottom-by-top factorial has an "
                        "unexpected interaction-arm inventory."
                    )
                required_ids = {
                    arm_id
                    for policy in BIOPSY_GUIDED_FULL_BOTTOM_TOP_BY_POLICY.values()
                    for arm_id in policy.values()
                }
                if not required_ids <= set(arm_ids):
                    raise ValueError(
                        "The fully labeled bottom-by-top factorial requires "
                        "the complete default/deferred by rooted-Q/binary design."
                    )
                full_bottom_top_factorial = {
                    "factor_levels": {
                        "bottom_policy": ["default", "deferred"],
                        "top_reconstruction": [
                            "rooted_labeled_q",
                            "binary_anticentral",
                        ],
                    },
                    "arm_ids": BIOPSY_GUIDED_FULL_BOTTOM_TOP_BY_POLICY,
                    "top_effect_by_bottom": {
                        bottom_policy: pairwise[
                            (
                                top_ids["binary_anticentral"],
                                top_ids["rooted_labeled_q"],
                            )
                        ]
                        for bottom_policy, top_ids in (
                            BIOPSY_GUIDED_FULL_BOTTOM_TOP_BY_POLICY.items()
                        )
                    },
                    "bottom_effect_by_top": {
                        top_method: pairwise[
                            (
                                BIOPSY_GUIDED_FULL_BOTTOM_TOP_BY_POLICY[
                                    "deferred"
                                ][top_method],
                                BIOPSY_GUIDED_FULL_BOTTOM_TOP_BY_POLICY[
                                    "default"
                                ][top_method],
                            )
                        ]
                        for top_method in (
                            "rooted_labeled_q",
                            "binary_anticentral",
                        )
                    },
                    "difference_in_differences": (
                        full_bottom_top_factorial_interaction_comparison(
                            records_by_key,
                            case_inventory,
                            primary_metric=primary_metric,
                        )
                    ),
                }
        families[family] = {
            "primary_metric": primary_metric,
            "arm_ids": arm_ids,
            "incumbent_id": INCUMBENT_BY_FAMILY[family],
            "baseline_id": BASELINE_BY_FAMILY[family],
            "leaderboard": leaderboard,
            "pareto_summary": _pareto_summary(leaderboard),
            "top_layer_comparison": top_layer_comparison,
            "top_radius_interaction": top_radius_interaction,
            "bottom_layer_comparison": bottom_layer_comparison,
            "bottom_top_factorial": bottom_top_factorial,
            "full_bottom_top_factorial": full_bottom_top_factorial,
            "pairwise_records": [
                pairwise[(arm_a, arm_b)]
                for arm_a in arm_ids
                for arm_b in arm_ids
            ],
            "matrices": {
                "combined_wins": _matrix(
                    arm_ids,
                    pairwise,
                    lambda value: value["combined_conditions"]["wtl"]["wins"],
                ),
                "combined_ties": _matrix(
                    arm_ids,
                    pairwise,
                    lambda value: value["combined_conditions"]["wtl"]["ties"],
                ),
                "combined_losses": _matrix(
                    arm_ids,
                    pairwise,
                    lambda value: value["combined_conditions"]["wtl"]["losses"],
                ),
                "combined_mean_delta": _matrix(
                    arm_ids,
                    pairwise,
                    lambda value: _nested(
                        value, "combined_conditions", "effect", "mean"
                    ),
                ),
                "combined_median_delta": _matrix(
                    arm_ids,
                    pairwise,
                    lambda value: _nested(
                        value, "combined_conditions", "effect", "median"
                    ),
                ),
                "declared_metric_dominance_fraction": _matrix(
                    arm_ids,
                    pairwise,
                    lambda value: value[
                        "all_declared_metric_nonstrict_dominance"
                    ]["fraction"],
                ),
            },
        }

    contextual_ids = sorted(
        arm_id
        for arm_id, spec in specs.items()
        if spec["family"] not in COMPARISON_FAMILIES
    )
    contextual_matched = []
    temporal_pair = ("temporal_minimum", "temporal_minimum_no_time")
    if all(arm_id in specs for arm_id in temporal_pair):
        temporal_metrics = {
            str(specs[arm_id]["primary_metric"]) for arm_id in temporal_pair
        }
        if len(temporal_metrics) != 1:
            raise ValueError("The temporal/no-time pair has inconsistent metrics.")
        temporal_complementary = sorted(
            set.intersection(
                *(
                    set(specs[arm_id].get("complementary_metrics", []))
                    for arm_id in temporal_pair
                )
            )
        )
        contextual_matched.append(
            pairwise_comparison(
                temporal_pair[0],
                temporal_pair[1],
                records_by_key,
                case_inventory,
                primary_metric=next(iter(temporal_metrics)),
                complementary_metrics=temporal_complementary,
            )
        )
    biopsy_guided_full_vs_pooled_incumbent = None
    biopsy_guided_full_ids = sorted(
        arm_id
        for arm_id, spec in specs.items()
        if spec["family"] == BIOPSY_GUIDED_FULL_FAMILY
    )
    if biopsy_guided_full_ids:
        if INFERRED_COPY_INCUMBENT_ID not in specs:
            raise ValueError(
                "The fully labeled biopsy-guided extension requires the pooled "
                "inferred-copy incumbent for its declared cross-pipeline comparison."
            )
        if any(
            specs[arm_id]["primary_metric"] != "ad_f1"
            for arm_id in (
                INFERRED_COPY_INCUMBENT_ID,
                *biopsy_guided_full_ids,
            )
        ):
            raise ValueError(
                "The fully labeled biopsy-guided cross-pipeline comparison "
                "requires AD-F1 for every arm."
            )
        shared_complementary_metrics = sorted(
            set(specs[INFERRED_COPY_INCUMBENT_ID].get("complementary_metrics", []))
            & set.intersection(
                *(
                    set(specs[arm_id].get("complementary_metrics", []))
                    for arm_id in biopsy_guided_full_ids
                )
            )
        )
        biopsy_guided_full_vs_pooled_incumbent = {
            "primary_metric": "ad_f1",
            "pooled_incumbent_id": INFERRED_COPY_INCUMBENT_ID,
            "principal_biopsy_guided_id": BIOPSY_GUIDED_FULL_INCUMBENT_ID,
            "interpretation": (
                "Cross-pipeline observed-label ancestry comparison. The top "
                "solver is matched for the principal contrast, while ordered "
                "biopsy reconstruction, occurrence preservation, and tie "
                "deferral differ from pooled state reconstruction."
            ),
            "ad_f1_occurrence_limit": (
                "AD-F1 compares unique observed-label ancestor pairs and does "
                "not preserve occurrence multiplicity; GRF is complementary."
            ),
            "counterparts": [
                {
                    "partial_arm_id": partial_id,
                    "full_arm_id": full_id,
                    "transformation": (
                        "already_fully_labeled_redeclared_for_ad_f1"
                        if partial_id
                        == "biopsy_guided_top_rooted_labeled_q_r2"
                        else "retain_top_created_parent_copy_labels"
                    ),
                }
                for partial_id, full_id in (
                    BIOPSY_GUIDED_FULL_COUNTERPART_BY_PARTIAL_ID.items()
                )
                if partial_id in specs and full_id in specs
            ],
            "pairwise_records": [
                pairwise_comparison(
                    arm_id,
                    INFERRED_COPY_INCUMBENT_ID,
                    records_by_key,
                    case_inventory,
                    primary_metric="ad_f1",
                    complementary_metrics=shared_complementary_metrics,
                )
                for arm_id in biopsy_guided_full_ids
            ],
        }
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "complete",
        "created_at_utc": created_at_utc or datetime.now(timezone.utc).isoformat(),
        "scientific_role": "manual_method_development_not_paper_confirmation",
        "automatic_winner_declared": False,
        "formal_significance_tests_run": False,
        "bank_id": next(iter(bank_ids)),
        "bank_root": next(iter(bank_roots)),
        "block_count": block_count,
        "condition_count": len(case_inventory),
        "dependence_contract": {
            "independent_truth_block_count": block_count,
            "descriptive_condition_count": len(case_inventory),
            "paired_heights_per_block": sorted(HEIGHT_SCHEDULES),
        },
        "run_ids": run_ids,
        "resource_execution_by_run": resource_execution_by_run,
        "resource_interpretation": {
            "all_runs_fresh_process_qualified": (
                all_resources_fresh_process_qualified
            ),
            "runtime_and_memory_cross_run_comparison_qualified": (
                all_resources_fresh_process_qualified
            ),
            "unqualified_resource_records_are_historical_context_only": True,
        },
        "arm_count": len(specs),
        "algorithm_summaries": summaries,
        "semantic_gate_by_arm": semantic_gates,
        "families": families,
        "contextual_reference_arm_ids": contextual_ids,
        "contextual_reference_summaries": {
            arm_id: summaries[arm_id] for arm_id in contextual_ids
        },
        "contextual_matched_comparisons": contextual_matched,
        "biopsy_guided_full_vs_pooled_incumbent": (
            biopsy_guided_full_vs_pooled_incumbent
        ),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_matrix_csv(
    path: Path,
    matrix: Mapping[str, Mapping[str, Any]],
    arm_ids: Sequence[str],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow(["arm_a_minus_arm_b", *arm_ids])
        for arm_a in arm_ids:
            writer.writerow([arm_a, *(matrix[arm_a][arm_b] for arm_b in arm_ids)])


def _pairwise_csv_rows(family: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for record in family["pairwise_records"]:
        block = record["independent_block_effect"]["effect"]
        combined = record["combined_conditions"]
        height_fields = {
            f"h{height}_{outcome}": record["by_height"][str(height)]["wtl"][
                outcome
            ]
            for height in sorted(HEIGHT_SCHEDULES)
            for outcome in ("wins", "ties", "losses")
        }
        rows.append(
            {
                "arm_a": record["arm_a"],
                "arm_b": record["arm_b"],
                "joint_success_count": record["joint_success_count"],
                "a_only_success_count": record["a_only_success_count"],
                "b_only_success_count": record["b_only_success_count"],
                "both_failure_count": record["both_failure_count"],
                **height_fields,
                "combined_wins": combined["wtl"]["wins"],
                "combined_ties": combined["wtl"]["ties"],
                "combined_losses": combined["wtl"]["losses"],
                "combined_win_score": combined["wtl"]["win_score"],
                "combined_mean_delta": _nested(combined, "effect", "mean"),
                "combined_median_delta": _nested(combined, "effect", "median"),
                "complete_block_count": record["independent_block_effect"][
                    "complete_block_count"
                ],
                "mean_block_delta": None if block is None else block["mean"],
                "median_block_delta": None if block is None else block["median"],
                "minimum_block_delta": None if block is None else block["minimum"],
                "worst_five_block_mean": (
                    None if block is None else block["worst_five_or_available_mean"]
                ),
                "conditional_gain_mean": (
                    None if block is None else block["conditional_gain_mean"]
                ),
                "conditional_loss_mean": (
                    None if block is None else block["conditional_loss_mean"]
                ),
                "worst_condition_case_ids": ";".join(
                    row["case_id"]
                    for row in record["worst_five_or_available_conditions"]
                ),
                "worst_block_indices": ";".join(
                    str(row["block_index"])
                    for row in record["independent_block_effect"][
                        "worst_five_or_available_blocks"
                    ]
                ),
                "declared_metric_dominance_fraction": record[
                    "all_declared_metric_nonstrict_dominance"
                ]["fraction"],
            }
        )
    return rows


def _factorial_interaction_csv_row(
    interaction: Mapping[str, Any],
) -> dict[str, Any]:
    height_fields: dict[str, Any] = {}
    for height in sorted(HEIGHT_SCHEDULES):
        record = interaction["by_height"][str(height)]
        for outcome in ("wins", "ties", "losses"):
            height_fields[f"h{height}_{outcome}"] = record["wtl"][outcome]
        height_fields[f"h{height}_mean_delta"] = _nested(
            record,
            "effect",
            "mean",
        )
    combined = interaction["combined_conditions"]
    block = interaction["independent_block_effect"]["effect"]
    return {
        "definition": interaction["definition"],
        "positive_direction": interaction["positive_direction"],
        "complete_case_count": interaction["complete_case_count"],
        "incomplete_case_count": interaction["incomplete_case_count"],
        **height_fields,
        "combined_wins": combined["wtl"]["wins"],
        "combined_ties": combined["wtl"]["ties"],
        "combined_losses": combined["wtl"]["losses"],
        "combined_mean_delta": _nested(combined, "effect", "mean"),
        "combined_median_delta": _nested(combined, "effect", "median"),
        "complete_block_count": interaction["independent_block_effect"][
            "complete_block_count"
        ],
        "mean_block_delta": None if block is None else block["mean"],
        "median_block_delta": None if block is None else block["median"],
        "minimum_block_delta": None if block is None else block["minimum"],
        "worst_five_block_mean": (
            None if block is None else block["worst_five_or_available_mean"]
        ),
        "worst_condition_case_ids": ";".join(
            row["case_id"]
            for row in interaction["worst_five_or_available_conditions"]
        ),
        "worst_block_indices": ";".join(
            str(row["block_index"])
            for row in interaction["independent_block_effect"][
                "worst_five_or_available_blocks"
            ]
        ),
    }


def _markdown(report: Mapping[str, Any]) -> str:
    height_labels = _height_labels()
    lines = [
        "# CTBF v5 algorithm-development comparison",
        "",
        (
            f"This report combines {report['condition_count']} "
            f"{'/'.join(height_labels)} condition "
            f"outcomes from {report['block_count']} independent truth blocks. "
            "It is development evidence and declares no automatic winner."
        ),
        "",
        (
            "Resource comparison: `fresh-process qualified`."
            if report["resource_interpretation"][
                "all_runs_fresh_process_qualified"
            ]
            else "Resource comparison: `unqualified historical context`; at "
            "least one run used a long-lived or unrecorded execution boundary."
        ),
        "",
    ]
    for family_name, family in report["families"].items():
        lines.extend(
            [
                f"## {family_name}",
                "",
                f"Primary metric: `{family['primary_metric']}`. Fixed incumbent: "
                f"`{family['incumbent_id']}`. Baseline: `{family['baseline_id']}`.",
                "",
                "| Rank | Algorithm | D0 | Failures | "
                + " | ".join(f"{label} mean" for label in height_labels)
                + " | W/T/L vs incumbent | Mean block delta | Worst-five mean |",
                "|"
                + "|".join(
                    ["---:", "---", "---", "---:"]
                    + ["---:"] * len(height_labels)
                    + ["---:", "---:", "---:"]
                )
                + "|",
            ]
        )
        for row in family["leaderboard"]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["display_rank"]),
                        f"`{row['arm_id']}`",
                        str(row["d0_status"]),
                        str(row["failure_count"]),
                        *(
                            _format_number(row[f"h{height}_primary_mean"])
                            for height in sorted(HEIGHT_SCHEDULES)
                        ),
                        (
                            f"{row['vs_incumbent_wins']}/"
                            f"{row['vs_incumbent_ties']}/"
                            f"{row['vs_incumbent_losses']}"
                        ),
                        _format_number(row["vs_incumbent_mean_block_delta"]),
                        _format_number(row["vs_incumbent_worst_five_block_mean"]),
                    ]
                )
                + " |"
            )
        lines.extend(
            [
                "",
                "Descriptive Pareto frontier: "
                + ", ".join(
                    f"`{arm_id}`"
                    for arm_id in family["pareto_summary"]["frontier_arm_ids"]
                )
                + ".",
                "",
            ]
        )
        top_layer = family.get("top_layer_comparison")
        if top_layer is not None:
            lines.extend(
                [
                    "### Partial top-layer screen",
                    "",
                    f"Fixed bottom policy: default radius 2. Control: "
                    f"`{top_layer['control_id']}`.",
                    "",
                    "| Top candidate | "
                    + " | ".join(f"{label} W/T/L" for label in height_labels)
                    + " | Combined W/T/L | Mean block delta | Worst-five mean |",
                    "|"
                    + "|".join(
                        ["---"]
                        + ["---:"] * len(height_labels)
                        + ["---:", "---:", "---:"]
                    )
                    + "|",
                ]
            )
            for comparison in top_layer["pairwise_records"]:
                combined = comparison["combined_conditions"]["wtl"]
                block = comparison["independent_block_effect"]["effect"]
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            f"`{comparison['arm_a']}`",
                            *(
                                "/".join(
                                    str(
                                        comparison["by_height"][str(height)][
                                            "wtl"
                                        ][outcome]
                                    )
                                    for outcome in ("wins", "ties", "losses")
                                )
                                for height in sorted(HEIGHT_SCHEDULES)
                            ),
                            (
                                f"{combined['wins']}/{combined['ties']}/"
                                f"{combined['losses']}"
                            ),
                            _format_number(None if block is None else block["mean"]),
                            _format_number(
                                None
                                if block is None
                                else block["worst_five_or_available_mean"]
                            ),
                        ]
                    )
                    + " |"
                )
            lines.append("")
        bottom_layer = family.get("bottom_layer_comparison")
        if bottom_layer is not None:
            mechanism_by_arm = {
                row["arm_id"]: row
                for row in bottom_layer["mechanism_rows"]
            }
            lines.extend(
                [
                    "### Partial bottom-layer screen",
                    "",
                    "All candidates use radius 2 and the same projected binary "
                    "anticentral plausible-parsimony top reconstruction. Control: "
                    f"`{bottom_layer['control_id']}` (default bottom policy).",
                    "",
                    "| Bottom candidate | "
                    + " | ".join(f"{label} W/T/L" for label in height_labels)
                    + " | Combined W/T/L | Mean block delta | Worst-five mean | "
                    "Mean multi-parent decisions | Mean exact ties | "
                    "Mean deferred ties | Mean copy-ups |",
                    "|"
                    + "|".join(
                        ["---"]
                        + ["---:"] * len(height_labels)
                        + ["---:"] * 7
                    )
                    + "|",
                ]
            )
            for comparison in bottom_layer["pairwise_records"]:
                combined = comparison["combined_conditions"]["wtl"]
                block = comparison["independent_block_effect"]["effect"]
                mechanism = mechanism_by_arm[comparison["arm_a"]]
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            f"`{comparison['arm_a']}`",
                            *(
                                "/".join(
                                    str(
                                        comparison["by_height"][str(height)][
                                            "wtl"
                                        ][outcome]
                                    )
                                    for outcome in ("wins", "ties", "losses")
                                )
                                for height in sorted(HEIGHT_SCHEDULES)
                            ),
                            (
                                f"{combined['wins']}/{combined['ties']}/"
                                f"{combined['losses']}"
                            ),
                            _format_number(None if block is None else block["mean"]),
                            _format_number(
                                None
                                if block is None
                                else block["worst_five_or_available_mean"]
                            ),
                            _format_number(
                                mechanism["mean_multiple_plausible_parent_count"]
                            ),
                            _format_number(
                                mechanism["mean_minimum_distance_tie_count"]
                            ),
                            _format_number(mechanism["mean_tie_deferred_count"]),
                            _format_number(mechanism["mean_copy_up_count"]),
                        ]
                    )
                    + " |"
                )
            lines.extend(
                [
                    "",
                    "Positive deltas favor the candidate over the unchanged "
                    "default-r2 control. Mechanism counts are means per complete "
                    "reconstruction and are descriptive diagnostics, not scores.",
                    "",
                ]
            )
        bottom_top = family.get("bottom_top_factorial")
        if bottom_top is not None:
            lines.extend(
                [
                    "### Partial bottom x top-reconstruction check",
                    "",
                    "Binary-top effects within each bottom policy:",
                    "",
                    "| Bottom policy | Binary minus classical W/T/L | "
                    "Mean block delta | Worst-five mean |",
                    "|---|---:|---:|---:|",
                ]
            )
            for bottom_policy in ("default", "deferred"):
                comparison = bottom_top["top_effect_by_bottom"][bottom_policy]
                wtl = comparison["combined_conditions"]["wtl"]
                effect = comparison["independent_block_effect"]["effect"]
                lines.append(
                    f"| {bottom_policy} | "
                    f"{wtl['wins']}/{wtl['ties']}/{wtl['losses']} | "
                    f"{_format_number(None if effect is None else effect['mean'])} | "
                    f"{_format_number(None if effect is None else effect['worst_five_or_available_mean'])} |"
                )
            lines.extend(
                [
                    "",
                    "Deferred-bottom effects within each top method:",
                    "",
                    "| Top method | Deferred minus default W/T/L | "
                    "Mean block delta | Worst-five mean |",
                    "|---|---:|---:|---:|",
                ]
            )
            for top_method in ("classical", "binary_anticentral"):
                comparison = bottom_top["bottom_effect_by_top"][top_method]
                wtl = comparison["combined_conditions"]["wtl"]
                effect = comparison["independent_block_effect"]["effect"]
                lines.append(
                    f"| {top_method} | "
                    f"{wtl['wins']}/{wtl['ties']}/{wtl['losses']} | "
                    f"{_format_number(None if effect is None else effect['mean'])} | "
                    f"{_format_number(None if effect is None else effect['worst_five_or_available_mean'])} |"
                )
            interaction = bottom_top["difference_in_differences"]
            combined = interaction["combined_conditions"]["wtl"]
            block = interaction["independent_block_effect"]["effect"]
            by_height = ", ".join(
                f"H{height} "
                + "/".join(
                    str(interaction["by_height"][str(height)]["wtl"][outcome])
                    for outcome in ("wins", "ties", "losses")
                )
                for height in sorted(HEIGHT_SCHEDULES)
            )
            lines.extend(
                [
                    "",
                    "Difference-in-differences is `(binary-classical with "
                    "deferred bottom) - (binary-classical with default "
                    "bottom)`; positive values mean the binary-top advantage "
                    "is larger after tie deferral.",
                    "",
                    f"By-height interaction W/T/L: {by_height}. Combined: "
                    f"{combined['wins']}/{combined['ties']}/{combined['losses']}; "
                    f"mean block delta {_format_number(None if block is None else block['mean'])}; "
                    f"worst-five mean {_format_number(None if block is None else block['worst_five_or_available_mean'])}.",
                    "",
                ]
            )
        full_bottom_top = family.get("full_bottom_top_factorial")
        if full_bottom_top is not None:
            lines.extend(
                [
                    "### Fully labeled bottom x top-reconstruction check",
                    "",
                    "Binary-anticentral top effects relative to rooted-labeled Q "
                    "within each bottom policy:",
                    "",
                    "| Bottom policy | Binary anticentral minus rooted Q W/T/L | "
                    "Mean block delta | Worst-five mean |",
                    "|---|---:|---:|---:|",
                ]
            )
            for bottom_policy in ("default", "deferred"):
                comparison = full_bottom_top["top_effect_by_bottom"][bottom_policy]
                wtl = comparison["combined_conditions"]["wtl"]
                effect = comparison["independent_block_effect"]["effect"]
                lines.append(
                    f"| {bottom_policy} | "
                    f"{wtl['wins']}/{wtl['ties']}/{wtl['losses']} | "
                    f"{_format_number(None if effect is None else effect['mean'])} | "
                    f"{_format_number(None if effect is None else effect['worst_five_or_available_mean'])} |"
                )
            lines.extend(
                [
                    "",
                    "Deferred-bottom effects within each fully labeled top method:",
                    "",
                    "| Top method | Deferred minus default W/T/L | "
                    "Mean block delta | Worst-five mean |",
                    "|---|---:|---:|---:|",
                ]
            )
            for top_method in ("rooted_labeled_q", "binary_anticentral"):
                comparison = full_bottom_top["bottom_effect_by_top"][top_method]
                wtl = comparison["combined_conditions"]["wtl"]
                effect = comparison["independent_block_effect"]["effect"]
                lines.append(
                    f"| {top_method} | "
                    f"{wtl['wins']}/{wtl['ties']}/{wtl['losses']} | "
                    f"{_format_number(None if effect is None else effect['mean'])} | "
                    f"{_format_number(None if effect is None else effect['worst_five_or_available_mean'])} |"
                )
            interaction = full_bottom_top["difference_in_differences"]
            combined = interaction["combined_conditions"]["wtl"]
            block = interaction["independent_block_effect"]["effect"]
            by_height = ", ".join(
                f"H{height} "
                + "/".join(
                    str(interaction["by_height"][str(height)]["wtl"][outcome])
                    for outcome in ("wins", "ties", "losses")
                )
                for height in sorted(HEIGHT_SCHEDULES)
            )
            lines.extend(
                [
                    "",
                    "Difference-in-differences is `(binary anticentral-rooted "
                    "Q with deferred bottom) - (binary anticentral-rooted Q "
                    "with default bottom)`; positive values mean the binary "
                    "top advantage is larger after tie deferral.",
                    "",
                    f"By-height interaction W/T/L: {by_height}. Combined: "
                    f"{combined['wins']}/{combined['ties']}/{combined['losses']}; "
                    f"mean block delta {_format_number(None if block is None else block['mean'])}; "
                    f"worst-five mean {_format_number(None if block is None else block['worst_five_or_available_mean'])}.",
                    "",
                ]
            )
        radius_interaction = family.get("top_radius_interaction")
        if radius_interaction is not None:
            lines.extend(
                [
                    "### Partial radius x top-reconstruction check",
                    "",
                    "Binary-top effects within each bottom radius:",
                    "",
                    "| Radius | Binary minus classical W/T/L | Mean block delta | "
                    "Worst-five mean |",
                    "|---:|---:|---:|---:|",
                ]
            )
            for radius in (2, 4):
                comparison = radius_interaction["top_effect_by_radius"][str(radius)]
                wtl = comparison["combined_conditions"]["wtl"]
                effect = comparison["independent_block_effect"]["effect"]
                lines.append(
                    f"| {radius} | {wtl['wins']}/{wtl['ties']}/{wtl['losses']} | "
                    f"{_format_number(None if effect is None else effect['mean'])} | "
                    f"{_format_number(None if effect is None else effect['worst_five_or_available_mean'])} |"
                )
            lines.extend(
                [
                    "",
                    "Radius effects within each top method:",
                    "",
                    "| Top method | r2 minus r4 W/T/L | Mean block delta | "
                    "Worst-five mean |",
                    "|---|---:|---:|---:|",
                ]
            )
            for top_method in ("classical", "binary_anticentral"):
                comparison = radius_interaction["radius_effect_by_top"][top_method]
                wtl = comparison["combined_conditions"]["wtl"]
                effect = comparison["independent_block_effect"]["effect"]
                lines.append(
                    f"| {top_method} | {wtl['wins']}/{wtl['ties']}/{wtl['losses']} | "
                    f"{_format_number(None if effect is None else effect['mean'])} | "
                    f"{_format_number(None if effect is None else effect['worst_five_or_available_mean'])} |"
                )
            interaction = radius_interaction["difference_in_differences"]
            combined = interaction["combined_conditions"]["wtl"]
            block = interaction["independent_block_effect"]["effect"]
            by_height = ", ".join(
                f"H{height} "
                + "/".join(
                    str(interaction["by_height"][str(height)]["wtl"][outcome])
                    for outcome in ("wins", "ties", "losses")
                )
                for height in sorted(HEIGHT_SCHEDULES)
            )
            lines.extend(
                [
                    "",
                    "Difference-in-differences is `(binary-classical at r2) - "
                    "(binary-classical at r4)`; positive values mean the binary "
                    "top advantage is larger at r2.",
                    "",
                    f"By-height interaction W/T/L: {by_height}. Combined: "
                    f"{combined['wins']}/{combined['ties']}/{combined['losses']}; "
                    f"mean block delta {_format_number(None if block is None else block['mean'])}; "
                    f"worst-five mean {_format_number(None if block is None else block['worst_five_or_available_mean'])}.",
                    "",
                ]
            )
    full_cross = report.get("biopsy_guided_full_vs_pooled_incumbent")
    if full_cross is not None:
        lines.extend(
            [
                "## Fully labeled biopsy-guided versus pooled incumbent",
                "",
                full_cross["interpretation"],
                "",
                full_cross["ad_f1_occurrence_limit"],
                "",
                f"Pooled comparator: `{full_cross['pooled_incumbent_id']}`. "
                f"Principal ordered method: "
                f"`{full_cross['principal_biopsy_guided_id']}`.",
                "",
                "| Ordered full method | Principal | "
                + " | ".join(f"{label} W/T/L" for label in height_labels)
                + " | Combined W/T/L | Mean AD-F1 block delta | "
                "Worst-five mean | Mean GRF delta |",
                "|"
                + "|".join(
                    ["---", "---:"]
                    + ["---:"] * len(height_labels)
                    + ["---:"] * 4
                )
                + "|",
            ]
        )
        for comparison in full_cross["pairwise_records"]:
            combined = comparison["combined_conditions"]["wtl"]
            block = comparison["independent_block_effect"]["effect"]
            grf_effect = comparison["complementary_metric_effects"].get("grf")
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{comparison['arm_a']}`",
                        (
                            "yes"
                            if comparison["arm_a"]
                            == full_cross["principal_biopsy_guided_id"]
                            else ""
                        ),
                        *(
                            "/".join(
                                str(
                                    comparison["by_height"][str(height)]["wtl"][
                                        outcome
                                    ]
                                )
                                for outcome in ("wins", "ties", "losses")
                            )
                            for height in sorted(HEIGHT_SCHEDULES)
                        ),
                        (
                            f"{combined['wins']}/{combined['ties']}/"
                            f"{combined['losses']}"
                        ),
                        _format_number(None if block is None else block["mean"]),
                        _format_number(
                            None
                            if block is None
                            else block["worst_five_or_available_mean"]
                        ),
                        _format_number(
                            None if grf_effect is None else grf_effect["mean"]
                        ),
                    ]
                )
                + " |"
            )
        lines.append("")
    if report["contextual_reference_arm_ids"]:
        lines.extend(
            [
                "## Contextual references",
                "",
                "These arms are summarized but not inserted into an incompatible-family ranking: "
                + ", ".join(
                    f"`{arm_id}`" for arm_id in report["contextual_reference_arm_ids"]
                )
                + ".",
                "",
            ]
        )
    if report["contextual_matched_comparisons"]:
        lines.extend(["### Temporal versus no-time ablation", ""])
        for comparison in report["contextual_matched_comparisons"]:
            wtl = comparison["combined_conditions"]["wtl"]
            lines.append(
                f"`{comparison['arm_a']}` minus `{comparison['arm_b']}`: "
                f"{wtl['wins']}/{wtl['ties']}/{wtl['losses']} combined "
                "descriptive W/T/L."
            )
        lines.append("")
    return "\n".join(lines)


def _format_number(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.6f}"


def write_report(
    *,
    result_roots: Sequence[Path | str],
    output_root: Path | str,
    expected_block_count: int = DEFAULT_BLOCK_COUNT,
) -> dict[str, Any]:
    loaded = [_load_result(path) for path in result_roots]
    results = [result for _path, result in loaded]
    if any(int(result["block_count"]) != expected_block_count for result in results):
        raise ValueError(
            f"The development comparison requires {expected_block_count} truth blocks."
        )
    report = build_report(results)
    root = ensure_new_output_root(output_root)
    write_json(root / "summary.json", report)
    for family_name, family in report["families"].items():
        _write_csv(root / f"leaderboard_{family_name}.csv", family["leaderboard"])
        _write_csv(root / f"pairwise_{family_name}.csv", _pairwise_csv_rows(family))
        for matrix_name, matrix in family["matrices"].items():
            _write_matrix_csv(
                root / f"matrix_{family_name}_{matrix_name}.csv",
                matrix,
                family["arm_ids"],
            )
        if family.get("top_layer_comparison") is not None:
            _write_csv(
                root / "partial_top_layer_vs_r2_classical.csv",
                _pairwise_csv_rows(family["top_layer_comparison"]),
            )
        bottom_layer = family.get("bottom_layer_comparison")
        if bottom_layer is not None:
            _write_csv(
                root / "partial_bottom_layer_vs_default_r2.csv",
                _pairwise_csv_rows(bottom_layer),
            )
            _write_csv(
                root / "partial_bottom_mechanism_summary.csv",
                bottom_layer["mechanism_rows"],
            )
        bottom_top = family.get("bottom_top_factorial")
        if bottom_top is not None:
            _write_csv(
                root / "partial_binary_top_effect_by_bottom.csv",
                _pairwise_csv_rows(
                    {
                        "pairwise_records": [
                            bottom_top["top_effect_by_bottom"][bottom_policy]
                            for bottom_policy in ("default", "deferred")
                        ]
                    }
                ),
            )
            _write_csv(
                root / "partial_deferred_bottom_effect_by_top.csv",
                _pairwise_csv_rows(
                    {
                        "pairwise_records": [
                            bottom_top["bottom_effect_by_top"][top_method]
                            for top_method in (
                                "classical",
                                "binary_anticentral",
                            )
                        ]
                    }
                ),
            )
            _write_csv(
                root / "partial_bottom_top_difference_in_differences.csv",
                [
                    _factorial_interaction_csv_row(
                        bottom_top["difference_in_differences"]
                    )
                ],
            )
        full_bottom_top = family.get("full_bottom_top_factorial")
        if full_bottom_top is not None:
            _write_csv(
                root / "biopsy_guided_full_binary_top_effect_by_bottom.csv",
                _pairwise_csv_rows(
                    {
                        "pairwise_records": [
                            full_bottom_top["top_effect_by_bottom"][bottom_policy]
                            for bottom_policy in ("default", "deferred")
                        ]
                    }
                ),
            )
            _write_csv(
                root / "biopsy_guided_full_deferred_bottom_effect_by_top.csv",
                _pairwise_csv_rows(
                    {
                        "pairwise_records": [
                            full_bottom_top["bottom_effect_by_top"][top_method]
                            for top_method in (
                                "rooted_labeled_q",
                                "binary_anticentral",
                            )
                        ]
                    }
                ),
            )
            _write_csv(
                root
                / "biopsy_guided_full_bottom_top_difference_in_differences.csv",
                [
                    _factorial_interaction_csv_row(
                        full_bottom_top["difference_in_differences"]
                    )
                ],
            )
        radius_interaction = family.get("top_radius_interaction")
        if radius_interaction is not None:
            _write_csv(
                root / "partial_binary_top_effect_by_radius.csv",
                _pairwise_csv_rows(
                    {
                        "pairwise_records": [
                            radius_interaction["top_effect_by_radius"][str(radius)]
                            for radius in (2, 4)
                        ]
                    }
                ),
            )
            _write_csv(
                root / "partial_radius_effect_by_top.csv",
                _pairwise_csv_rows(
                    {
                        "pairwise_records": [
                            radius_interaction["radius_effect_by_top"][top_method]
                            for top_method in ("classical", "binary_anticentral")
                        ]
                    }
                ),
            )
            _write_csv(
                root / "partial_radius_top_difference_in_differences.csv",
                [
                    _factorial_interaction_csv_row(
                        radius_interaction["difference_in_differences"]
                    )
                ],
            )
    full_cross = report.get("biopsy_guided_full_vs_pooled_incumbent")
    if full_cross is not None:
        _write_csv(
            root / "biopsy_guided_full_vs_pooled_incumbent.csv",
            _pairwise_csv_rows(full_cross),
        )
        _write_csv(
            root / "biopsy_guided_full_counterparts.csv",
            full_cross["counterparts"],
        )
    if report["contextual_matched_comparisons"]:
        _write_csv(
            root / "pairwise_contextual_matched.csv",
            _pairwise_csv_rows(
                {"pairwise_records": report["contextual_matched_comparisons"]}
            ),
        )
    (root / "report.md").write_text(_markdown(report), encoding="utf-8")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-root",
        type=Path,
        action="append",
        required=True,
        help="Repeat for the initial run and every added algorithm run.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    report = write_report(
        result_roots=arguments.result_root,
        output_root=arguments.output_root,
    )
    print(
        f"complete: {report['arm_count']} algorithms on "
        f"{report['condition_count']} conditions"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
