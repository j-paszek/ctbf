"""Report the block-paired CTBF v5 shortlist depth-by-placement experiment."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import itertools
import math
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

from algorithm_evaluation.paper_pipeline_contract import read_json
from algorithm_evaluation.process_isolation import (
    CASE_ARM_WORKER_UNIT,
    TRUTH_BLOCK_SIMULATION_WORKER_UNIT,
    fresh_process_contract,
    validate_fresh_process_contract,
)
from algorithm_evaluation.v5_algorithm_development_common import (
    ARM_SPEC_BY_ID,
    numeric_summary,
)
from algorithm_evaluation.v5_shortlist_robustness_common import (
    ADAPTIVE_A_PRIME_ID,
    ADAPTIVE_B_PRIME_ID,
    ADAPTIVE_C_PRIME_ID,
    ADAPTIVE_D_PRIME_ID,
    ADAPTIVE_RADIUS_ARM_IDS,
    ARM_SET_BY_NAME,
    DECLARED_METRICS,
    DISTANCE_EXECUTION_SCHEMA_VERSION,
    FULL_DEVELOPMENT_ARM_IDS,
    FULL_V2_ARM_IDS,
    ORDERED_A_ID,
    ORDERED_B_ID,
    ORDERED_C_ID,
    PLACEMENT_POLICIES,
    PARTIAL_DECLARED_METRICS,
    PARTIAL_V2_ARM_IDS,
    POOLED_D_ID,
    PREVIOUS_RUN_SCHEMA_VERSION,
    REPORT_SCHEMA_VERSION,
    RESULT_NAME,
    RUN_SCHEMA_VERSION,
    SHORTLIST_ARM_IDS,
    SHORT_LABEL_BY_ARM,
    SUPPORTED_SHORTLIST_ARM_IDS,
    ensure_new_output_root,
    load_bank_manifest,
    write_json,
)


TIE_TOLERANCE = 1e-12
WORST_COUNT = 5
DEPTH_CONTRASTS = ((14, 24), (24, 34), (34, 38), (14, 38))
PLACEMENT_CONTRASTS = (
    ("late", "spread"),
    ("random", "spread"),
    ("random", "late"),
)
PRINCIPAL_PAIRS = (
    (ORDERED_A_ID, POOLED_D_ID),
    (ORDERED_A_ID, ORDERED_B_ID),
    (ORDERED_A_ID, ORDERED_C_ID),
)
FULL_PRINCIPAL_PAIRS = PRINCIPAL_PAIRS + tuple(
    (ORDERED_A_ID, arm_id) for arm_id in FULL_DEVELOPMENT_ARM_IDS[4:]
) + (
    (ORDERED_B_ID, ADAPTIVE_B_PRIME_ID),
    (ORDERED_C_ID, ADAPTIVE_C_PRIME_ID),
    (ADAPTIVE_A_PRIME_ID, ADAPTIVE_B_PRIME_ID),
    (ADAPTIVE_A_PRIME_ID, ADAPTIVE_C_PRIME_ID),
    (ADAPTIVE_C_PRIME_ID, ADAPTIVE_D_PRIME_ID),
    (ADAPTIVE_B_PRIME_ID, ADAPTIVE_D_PRIME_ID),
    (POOLED_D_ID, ADAPTIVE_A_PRIME_ID),
)
PARTIAL_PRINCIPAL_PAIRS = (
    (PARTIAL_V2_ARM_IDS[0], PARTIAL_V2_ARM_IDS[1]),  # X-Y
    (PARTIAL_V2_ARM_IDS[1], PARTIAL_V2_ARM_IDS[2]),  # Y-Z
    (PARTIAL_V2_ARM_IDS[1], PARTIAL_V2_ARM_IDS[3]),  # Y-V
    (PARTIAL_V2_ARM_IDS[2], PARTIAL_V2_ARM_IDS[5]),  # Z-U
    (PARTIAL_V2_ARM_IDS[3], PARTIAL_V2_ARM_IDS[5]),  # V-U
    (PARTIAL_V2_ARM_IDS[3], PARTIAL_V2_ARM_IDS[4]),  # V-W
)
SHORT_DESCRIPTION_BY_ARM = {
    ORDERED_A_ID: "deferred-bottom binary-anticentral r2 full",
    ORDERED_B_ID: "deferred-bottom rooted-Q r2 full",
    ORDERED_C_ID: "default-bottom binary-anticentral r4 full",
    POOLED_D_ID: "pooled plausible-parsimony full",
    FULL_V2_ARM_IDS[4]: "pooled baseline full",
    FULL_V2_ARM_IDS[5]: "pooled hybrid-opt-refined full",
    FULL_V2_ARM_IDS[6]: "default-bottom binary-anticentral r2 full",
    ADAPTIVE_A_PRIME_ID: (
        "deferred-bottom binary-anticentral transition-median full"
    ),
    ADAPTIVE_B_PRIME_ID: (
        "deferred-bottom rooted-Q transition-median full"
    ),
    ADAPTIVE_C_PRIME_ID: (
        "default-bottom binary-anticentral transition-median full"
    ),
    ADAPTIVE_D_PRIME_ID: "default-bottom rooted-Q transition-median full",
    PARTIAL_V2_ARM_IDS[0]: "pooled classical partial",
    PARTIAL_V2_ARM_IDS[1]: "deferred-bottom binary-anticentral r2 partial",
    PARTIAL_V2_ARM_IDS[2]: "deferred-bottom classical r2 partial",
    PARTIAL_V2_ARM_IDS[3]: "default-bottom binary-anticentral r2 partial",
    PARTIAL_V2_ARM_IDS[4]: "default-bottom binary-anticentral r4 partial",
    PARTIAL_V2_ARM_IDS[5]: "default-bottom classical r2 partial",
}


def _effect_summary(values: Sequence[float]) -> dict[str, Any] | None:
    if not values:
        return None
    if any(not math.isfinite(value) for value in values):
        raise ValueError("Effect values must be finite.")
    ordered = sorted(float(value) for value in values)
    positive = [value for value in ordered if value > TIE_TOLERANCE]
    negative = [value for value in ordered if value < -TIE_TOLERANCE]
    ties = len(ordered) - len(positive) - len(negative)
    return {
        "count": len(ordered),
        "minimum": ordered[0],
        "median": float(statistics.median(ordered)),
        "mean": float(statistics.fmean(ordered)),
        "maximum": ordered[-1],
        "positive_count": len(positive),
        "tie_count": ties,
        "negative_count": len(negative),
        "conditional_gain_mean": (
            float(statistics.fmean(positive)) if positive else None
        ),
        "conditional_loss_mean": (
            float(statistics.fmean(negative)) if negative else None
        ),
        "worst_five_or_available_mean": float(
            statistics.fmean(ordered[:WORST_COUNT])
        ),
    }


def _wtl(values: Sequence[float]) -> dict[str, Any]:
    wins = sum(value > TIE_TOLERANCE for value in values)
    losses = sum(value < -TIE_TOLERANCE for value in values)
    ties = len(values) - wins - losses
    return {
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "eligible": len(values),
    }


def _metric(record: Mapping[str, Any], metric: str) -> float:
    value = record.get("metrics", {}).get(metric)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Successful record has invalid {metric}.")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"Successful record has nonfinite {metric}.")
    return value


def _resource_value(record: Mapping[str, Any], field: str) -> float | None:
    resources = record.get("resources")
    if not isinstance(resources, Mapping):
        return None
    stages = [
        stage
        for stage in (resources.get("reconstruction"), resources.get("evaluation"))
        if isinstance(stage, Mapping)
    ]
    if field == "runtime_seconds":
        values = [
            float(stage["wall_time_ns"]) / 1_000_000_000.0
            for stage in stages
            if isinstance(stage.get("wall_time_ns"), (int, float))
        ]
        return sum(values) if values else None
    if field == "peak_rss_bytes":
        values = [
            float(stage["memory"]["peak_rss_bytes"])
            for stage in stages
            if isinstance(stage.get("memory"), Mapping)
            and isinstance(stage["memory"].get("peak_rss_bytes"), (int, float))
        ]
        return max(values) if values else None
    raise ValueError(f"Unknown resource field {field!r}.")


def _failure_type(record: Mapping[str, Any]) -> str:
    failure = record.get("failure")
    if not isinstance(failure, Mapping):
        return "unknown_failure"
    return str(failure.get("type", "unknown_failure"))


def _condition_key(record: Mapping[str, Any]) -> tuple[int, int, str]:
    return (
        int(record["block_index"]),
        int(record["height"]),
        str(record["placement_policy"]),
    )


def _record_index(
    records: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, int, str, str], Mapping[str, Any]]:
    index = {}
    for record in records:
        key = (*_condition_key(record), str(record["arm_id"]))
        if key in index:
            raise ValueError(f"Duplicate shortlist result record {key}.")
        index[key] = record
    return index


def _availability_summary(bank: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for height in bank["heights"]:
        for policy in bank["placement_policies"]:
            conditions = [
                row
                for row in bank["condition_inventory"]
                if int(row["height"]) == int(height)
                and row["placement_policy"] == policy
            ]
            failures: dict[str, int] = {}
            for row in conditions:
                if row["status"] == "available":
                    continue
                failure = row.get("failure")
                name = (
                    str(failure.get("type", "unknown_failure"))
                    if isinstance(failure, Mapping)
                    else "unknown_failure"
                )
                failures[name] = failures.get(name, 0) + 1
            rows.append(
                {
                    "height": int(height),
                    "placement_policy": policy,
                    "declared_count": len(conditions),
                    "available_count": sum(
                        row["status"] == "available" for row in conditions
                    ),
                    "unavailable_count": sum(
                        row["status"] != "available" for row in conditions
                    ),
                    "failure_types": failures,
                }
            )
    return rows


def _adaptive_radius_diagnostics(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    diagnostics = []
    for arm_id in ADAPTIVE_RADIUS_ARM_IDS:
        arm_records = [
            record
            for record in records
            if record.get("arm_id") == arm_id and record.get("status") == "success"
        ]
        if not arm_records:
            continue
        transition_rows = []
        for record in arm_records:
            metadata = record.get("reconstruction_metadata")
            audit = (
                metadata.get("biopsy_layer_decision_audit")
                if isinstance(metadata, Mapping)
                else None
            )
            expected_policy = ARM_SPEC_BY_ID[arm_id].radius_policy
            if (
                not isinstance(audit, Mapping)
                or audit.get("radius_policy") != expected_policy
                or not isinstance(audit.get("transition_records"), list)
            ):
                raise ValueError("Adaptive-radius reconstruction diagnostics changed.")
            for transition in audit["transition_records"]:
                counters = transition.get("decision_counters")
                if not isinstance(counters, Mapping):
                    raise ValueError("Adaptive transition counters are missing.")
                transition_rows.append(
                    {
                        "height": int(record["height"]),
                        "placement_policy": str(record["placement_policy"]),
                        **transition,
                    }
                )

        def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
            child_count = sum(
                int(row["decision_counters"]["child_decision_count"])
                for row in rows
            )
            selected_count = sum(
                int(row["decision_counters"]["selected_parent_count"])
                for row in rows
            )
            copy_up_count = sum(
                int(row["decision_counters"]["copy_up_count"])
                for row in rows
            )
            frozen_child_count = sum(int(row["child_snapshot_count"]) for row in rows)
            frozen_covered_count = sum(
                int(row["frozen_radius_candidate_covered_child_count"])
                for row in rows
            )
            radius_frequency = {}
            for row in rows:
                radius = int(row["effective_radius"])
                radius_frequency[radius] = radius_frequency.get(radius, 0) + 1
            return {
                "transition_count": len(rows),
                "effective_radius": numeric_summary(
                    float(row["effective_radius"]) for row in rows
                ),
                "effective_radius_frequency": [
                    {"radius": radius, "count": radius_frequency[radius]}
                    for radius in sorted(radius_frequency)
                ],
                "nearest_rank_quantile_distance": numeric_summary(
                    float(row["nearest_rank_quantile_distance"])
                    for row in rows
                ),
                "frozen_radius_candidate_coverage_fraction": (
                    float(frozen_covered_count / frozen_child_count)
                    if frozen_child_count
                    else None
                ),
                "child_decision_count": child_count,
                "selected_parent_count": selected_count,
                "copy_up_count": copy_up_count,
                "selected_parent_fraction": (
                    float(selected_count / child_count) if child_count else None
                ),
                "copy_up_fraction": (
                    float(copy_up_count / child_count) if child_count else None
                ),
            }

        by_cell = []
        for height in sorted({int(row["height"]) for row in transition_rows}):
            for policy in PLACEMENT_POLICIES:
                rows = [
                    row
                    for row in transition_rows
                    if int(row["height"]) == height
                    and row["placement_policy"] == policy
                ]
                if rows:
                    by_cell.append(
                        {
                            "height": height,
                            "placement_policy": policy,
                            **summarize(rows),
                        }
                    )
        diagnostics.append(
            {
                "arm_id": arm_id,
                "short_label": SHORT_LABEL_BY_ARM[arm_id],
                "radius_policy": ARM_SPEC_BY_ID[arm_id].radius_policy,
                "successful_case_count": len(arm_records),
                "overall": summarize(transition_rows),
                "by_height_and_placement": by_cell,
            }
        )
    return diagnostics


def _bank_resource_execution(bank: Mapping[str, Any]) -> dict[str, Any]:
    resource_contract = bank.get("resource_contract")
    simulation_execution = (
        resource_contract.get("simulation_execution")
        if isinstance(resource_contract, Mapping)
        else None
    )
    simulation_qualified = simulation_execution == fresh_process_contract(
        TRUTH_BLOCK_SIMULATION_WORKER_UNIT
    )
    distance_records = bank.get("distance_execution_by_block")
    if not isinstance(distance_records, list):
        distance_records = []
    qualified_distance_records = [
        record
        for record in distance_records
        if isinstance(record, Mapping)
        and record.get("schema_version") == DISTANCE_EXECUTION_SCHEMA_VERSION
        and record.get("worker_lifecycle")
        == "fresh_spawn_process_per_condition"
    ]
    distance_qualified = (
        len(distance_records) == int(bank["block_count"])
        and len(qualified_distance_records) == len(distance_records)
    )
    return {
        "bank_schema_version": bank.get("schema_version"),
        "simulation_execution": (
            dict(simulation_execution)
            if isinstance(simulation_execution, Mapping)
            else None
        ),
        "simulation_fresh_process_qualified": simulation_qualified,
        "distance_execution_semantics": bank.get(
            "distance_execution_semantics"
        ),
        "distance_execution_record_count": len(distance_records),
        "fresh_distance_execution_record_count": len(
            qualified_distance_records
        ),
        "distance_fresh_process_qualified": distance_qualified,
        "all_bank_resources_fresh_process_qualified": (
            simulation_qualified and distance_qualified
        ),
        "unqualified_bank_resources_are_historical_context_only": True,
    }


def _algorithm_cell_summaries(
    *,
    bank: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    arm_ids: Sequence[str] = SHORTLIST_ARM_IDS,
    metrics: Sequence[str] = DECLARED_METRICS,
) -> list[dict[str, Any]]:
    rows = []
    for height in bank["heights"]:
        for policy in bank["placement_policies"]:
            for arm_id in arm_ids:
                cell = [
                    record
                    for record in records
                    if int(record["height"]) == int(height)
                    and record["placement_policy"] == policy
                    and record["arm_id"] == arm_id
                ]
                successful = [record for record in cell if record["status"] == "success"]
                failures: dict[str, int] = {}
                for record in cell:
                    if record["status"] == "success":
                        continue
                    name = _failure_type(record)
                    failures[name] = failures.get(name, 0) + 1
                rows.append(
                    {
                        "height": int(height),
                        "placement_policy": policy,
                        "arm_id": arm_id,
                        "short_label": SHORT_LABEL_BY_ARM[arm_id],
                        "record_count": len(cell),
                        "success_count": len(successful),
                        "failure_count": len(cell) - len(successful),
                        "failure_types": failures,
                        "metrics": {
                            metric: numeric_summary(
                                _metric(record, metric) for record in successful
                            )
                            for metric in metrics
                        },
                        "runtime_seconds": numeric_summary(
                            value
                            for record in cell
                            if (value := _resource_value(record, "runtime_seconds"))
                            is not None
                        ),
                        "peak_rss_bytes": numeric_summary(
                            value
                            for record in cell
                            if (value := _resource_value(record, "peak_rss_bytes"))
                            is not None
                        ),
                    }
                )
    return rows


def _pairwise_cells(
    *,
    bank: Mapping[str, Any],
    index: Mapping[tuple[int, int, str, str], Mapping[str, Any]],
    arm_ids: Sequence[str] = SHORTLIST_ARM_IDS,
    metrics: Sequence[str] = DECLARED_METRICS,
) -> list[dict[str, Any]]:
    rows = []
    for arm_a, arm_b in itertools.combinations(arm_ids, 2):
        for metric in metrics:
            for height in bank["heights"]:
                for policy in bank["placement_policies"]:
                    values = []
                    a_only = b_only = both_failure = 0
                    for block_index in range(int(bank["block_count"])):
                        record_a = index.get((block_index, int(height), policy, arm_a))
                        record_b = index.get((block_index, int(height), policy, arm_b))
                        if record_a is None or record_b is None:
                            continue
                        success_a = record_a["status"] == "success"
                        success_b = record_b["status"] == "success"
                        if success_a and success_b:
                            values.append(
                                _metric(record_a, metric) - _metric(record_b, metric)
                            )
                        elif success_a:
                            a_only += 1
                        elif success_b:
                            b_only += 1
                        else:
                            both_failure += 1
                    rows.append(
                        {
                            "arm_a": arm_a,
                            "label_a": SHORT_LABEL_BY_ARM[arm_a],
                            "arm_b": arm_b,
                            "label_b": SHORT_LABEL_BY_ARM[arm_b],
                            "delta_direction": "arm_a_minus_arm_b",
                            "metric": metric,
                            "height": int(height),
                            "placement_policy": policy,
                            "effect": _effect_summary(values),
                            "wtl": _wtl(values),
                            "a_only_success_count": a_only,
                            "b_only_success_count": b_only,
                            "both_failure_count": both_failure,
                        }
                    )
    return rows


def _paired_delta(
    index: Mapping[tuple[int, int, str, str], Mapping[str, Any]],
    *,
    block_index: int,
    height: int,
    policy: str,
    arm_a: str,
    arm_b: str,
    metric: str,
) -> float | None:
    record_a = index.get((block_index, height, policy, arm_a))
    record_b = index.get((block_index, height, policy, arm_b))
    if (
        record_a is None
        or record_b is None
        or record_a["status"] != "success"
        or record_b["status"] != "success"
    ):
        return None
    return _metric(record_a, metric) - _metric(record_b, metric)


def _depth_interactions(
    *,
    bank: Mapping[str, Any],
    index: Mapping[tuple[int, int, str, str], Mapping[str, Any]],
    arm_ids: Sequence[str] = SHORTLIST_ARM_IDS,
    metrics: Sequence[str] = DECLARED_METRICS,
) -> list[dict[str, Any]]:
    available_heights = set(int(value) for value in bank["heights"])
    rows = []
    for arm_a, arm_b in itertools.combinations(arm_ids, 2):
        for metric in metrics:
            for policy in bank["placement_policies"]:
                for lower, upper in DEPTH_CONTRASTS:
                    if lower not in available_heights or upper not in available_heights:
                        continue
                    values = []
                    for block_index in range(int(bank["block_count"])):
                        lower_delta = _paired_delta(
                            index,
                            block_index=block_index,
                            height=lower,
                            policy=policy,
                            arm_a=arm_a,
                            arm_b=arm_b,
                            metric=metric,
                        )
                        upper_delta = _paired_delta(
                            index,
                            block_index=block_index,
                            height=upper,
                            policy=policy,
                            arm_a=arm_a,
                            arm_b=arm_b,
                            metric=metric,
                        )
                        if lower_delta is not None and upper_delta is not None:
                            values.append(upper_delta - lower_delta)
                    rows.append(
                        {
                            "arm_a": arm_a,
                            "label_a": SHORT_LABEL_BY_ARM[arm_a],
                            "arm_b": arm_b,
                            "label_b": SHORT_LABEL_BY_ARM[arm_b],
                            "metric": metric,
                            "placement_policy": policy,
                            "lower_height": lower,
                            "upper_height": upper,
                            "delta_direction": (
                                "(arm_a-arm_b)_upper_minus_(arm_a-arm_b)_lower"
                            ),
                            "effect": _effect_summary(values),
                            "wtl": _wtl(values),
                        }
                    )
    return rows


def _placement_interactions(
    *,
    bank: Mapping[str, Any],
    index: Mapping[tuple[int, int, str, str], Mapping[str, Any]],
    arm_ids: Sequence[str] = SHORTLIST_ARM_IDS,
    metrics: Sequence[str] = DECLARED_METRICS,
) -> list[dict[str, Any]]:
    policies = set(str(value) for value in bank["placement_policies"])
    rows = []
    for arm_a, arm_b in itertools.combinations(arm_ids, 2):
        for metric in metrics:
            for height in bank["heights"]:
                for policy_a, policy_b in PLACEMENT_CONTRASTS:
                    if policy_a not in policies or policy_b not in policies:
                        continue
                    values = []
                    for block_index in range(int(bank["block_count"])):
                        delta_a = _paired_delta(
                            index,
                            block_index=block_index,
                            height=int(height),
                            policy=policy_a,
                            arm_a=arm_a,
                            arm_b=arm_b,
                            metric=metric,
                        )
                        delta_b = _paired_delta(
                            index,
                            block_index=block_index,
                            height=int(height),
                            policy=policy_b,
                            arm_a=arm_a,
                            arm_b=arm_b,
                            metric=metric,
                        )
                        if delta_a is not None and delta_b is not None:
                            values.append(delta_a - delta_b)
                    rows.append(
                        {
                            "arm_a": arm_a,
                            "label_a": SHORT_LABEL_BY_ARM[arm_a],
                            "arm_b": arm_b,
                            "label_b": SHORT_LABEL_BY_ARM[arm_b],
                            "metric": metric,
                            "height": int(height),
                            "policy_a": policy_a,
                            "policy_b": policy_b,
                            "delta_direction": (
                                "(arm_a-arm_b)_policy_a_minus_"
                                "(arm_a-arm_b)_policy_b"
                            ),
                            "effect": _effect_summary(values),
                            "wtl": _wtl(values),
                        }
                    )
    return rows


def _diagnostic_summaries(
    *,
    bank_root: Path,
    bank: Mapping[str, Any],
) -> list[dict[str, Any]]:
    by_cell: dict[tuple[int, str], list[Mapping[str, Any]]] = {}
    for case in bank["cases"]:
        metadata = read_json(bank_root / case["metadata_path"])
        by_cell.setdefault(
            (int(case["height"]), str(case["placement_policy"])),
            [],
        ).append(metadata)
    rows = []
    diagnostic_fields = (
        "observation_to_truth_node_ratio",
        "incomparable_pair_fraction",
        "sampled_ancestor_coverage_fraction",
        "minimum_invented_edges_for_observed_only_arborescence",
        "normalized_minimum_invented_edge_fraction",
        "unique_state_label_coverage_fraction",
    )
    for height in bank["heights"]:
        for policy in bank["placement_policies"]:
            metadata_rows = by_cell.get((int(height), policy), [])
            rows.append(
                {
                    "height": int(height),
                    "placement_policy": policy,
                    "available_case_count": len(metadata_rows),
                    "selected_occurrence_count": numeric_summary(
                        float(row["selected_occurrence_count"])
                        for row in metadata_rows
                    ),
                    "selected_unique_state_count": numeric_summary(
                        float(row["selected_unique_state_count"])
                        for row in metadata_rows
                    ),
                    "truth_sampling_diagnostics": {
                        field: numeric_summary(
                            float(value)
                            for row in metadata_rows
                            if (
                                value := row["truth_sampling_diagnostics"].get(field)
                            )
                            is not None
                        )
                        for field in diagnostic_fields
                    },
                    "observed_only_representable_count": sum(
                        bool(
                            row["truth_sampling_diagnostics"].get(
                                "observed_only_occurrence_arborescence_representable"
                            )
                        )
                        for row in metadata_rows
                    ),
                    "distance_runtime_seconds": numeric_summary(
                        float(row["distance_runtime"]["wall_time_ns"])
                        / 1_000_000_000.0
                        for row in metadata_rows
                    ),
                    "distance_peak_rss_bytes": numeric_summary(
                        float(row["distance_runtime"]["memory"]["peak_rss_bytes"])
                        for row in metadata_rows
                        if row["distance_runtime"]["memory"].get("peak_rss_bytes")
                        is not None
                    ),
                }
            )
    return rows


def build_report(
    *,
    result_root: Path | str | None = None,
    result_roots: Sequence[Path | str] | None = None,
    expected_block_count: int | None = None,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    if result_roots is None:
        if result_root is None:
            raise ValueError("At least one shortlist result root is required.")
        normalized_roots = (result_root,)
    else:
        if result_root is not None:
            raise ValueError("Pass result_root or result_roots, not both.")
        normalized_roots = tuple(result_roots)
        if not normalized_roots:
            raise ValueError("At least one shortlist result root is required.")

    loaded = []
    for candidate_root in normalized_roots:
        root = Path(candidate_root).expanduser().resolve()
        result = read_json(root / RESULT_NAME)
        if result.get("schema_version") not in {
            RUN_SCHEMA_VERSION,
            PREVIOUS_RUN_SCHEMA_VERSION,
        }:
            raise ValueError("Unknown shortlist-robustness run schema.")
        if result.get("status") != "complete":
            raise ValueError("Shortlist-robustness run is not complete.")
        resources = result.get("resources")
        if not isinstance(resources, Mapping):
            raise ValueError("Shortlist run has no auditable resource contract.")
        validate_fresh_process_contract(
            resources.get("record_execution"),
            worker_unit=CASE_ARM_WORKER_UNIT,
        )
        loaded.append((root, result))

    run_ids = [str(result["run_id"]) for _root, result in loaded]
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("Shortlist result run ids must be unique.")
    bank_roots = {str(result["bank_root"]) for _root, result in loaded}
    bank_ids = {str(result["bank_id"]) for _root, result in loaded}
    if len(bank_roots) != 1 or len(bank_ids) != 1:
        raise ValueError("All shortlist runs must use the exact same stored bank.")

    first_result = loaded[0][1]
    bank_root, bank = load_bank_manifest(
        first_result["bank_root"],
        expected_block_count=expected_block_count,
    )
    expected_result_fields = {
        "bank_id": bank["bank_id"],
        "block_count": int(bank["block_count"]),
        "declared_condition_count": int(bank["declared_condition_count"]),
        "available_condition_count": int(bank["available_condition_count"]),
        "unavailable_condition_count": int(bank["unavailable_condition_count"]),
    }
    all_records: list[Mapping[str, Any]] = []
    declared_arm_ids: set[str] = set()
    semantic_gate_by_arm: dict[str, Any] = {}
    record_execution_by_run = []
    for root, result in loaded:
        for field, expected in expected_result_fields.items():
            if result.get(field) != expected:
                raise ValueError(
                    f"Shortlist run {field} does not match its immutable bank."
                )
        run_specs = result.get("arm_specs")
        if not isinstance(run_specs, list) or not run_specs:
            raise ValueError("Shortlist run has no arm declarations.")
        arm_ids = tuple(str(spec.get("arm_id")) for spec in run_specs)
        if len(set(arm_ids)) != len(arm_ids):
            raise ValueError("A shortlist run declares duplicate arms.")
        if any(arm_id not in SUPPORTED_SHORTLIST_ARM_IDS for arm_id in arm_ids):
            raise ValueError("A shortlist run declares an unknown development arm.")
        if declared_arm_ids.intersection(arm_ids):
            raise ValueError(
                "An arm occurs in more than one merged shortlist run; use one result."
            )
        expected_specs = [ARM_SPEC_BY_ID[arm_id].as_record() for arm_id in arm_ids]
        if run_specs != expected_specs:
            raise ValueError("Shortlist run arm declaration changed.")
        if result.get("schema_version") == RUN_SCHEMA_VERSION:
            if result.get("arm_ids") != list(arm_ids):
                raise ValueError("Shortlist run arm-id declaration changed.")
            arm_set = result.get("arm_set")
            if arm_set not in ARM_SET_BY_NAME or ARM_SET_BY_NAME[arm_set] != arm_ids:
                raise ValueError("Shortlist run arm-set declaration changed.")
        records = result.get("records")
        expected_record_count = len(bank["cases"]) * len(arm_ids)
        if (
            not isinstance(records, list)
            or result.get("expected_record_count") != expected_record_count
            or len(records) != expected_record_count
        ):
            raise ValueError("Shortlist run record inventory is incomplete.")
        expected_keys = {
            (
                int(case["block_index"]),
                int(case["height"]),
                str(case["placement_policy"]),
                arm_id,
            )
            for case in bank["cases"]
            for arm_id in arm_ids
        }
        run_index = _record_index(records)
        if set(run_index) != expected_keys:
            raise ValueError("Shortlist run records do not match available bank cases.")
        for record in records:
            if record["status"] != "success":
                continue
            spec = ARM_SPEC_BY_ID[str(record["arm_id"])]
            for metric in (spec.primary_metric, *spec.complementary_metrics):
                _metric(record, metric)
        gates = result.get("semantic_gate_by_arm")
        if not isinstance(gates, Mapping) or set(gates) != set(arm_ids):
            raise ValueError("Shortlist run has an incomplete D0 gate inventory.")
        semantic_gate_by_arm.update(gates)
        declared_arm_ids.update(arm_ids)
        all_records.extend(records)
        record_execution_by_run.append(
            {
                "run_id": str(result["run_id"]),
                "result_root": str(root),
                "record_execution": dict(result["resources"]["record_execution"]),
            }
        )

    report_arm_order = FULL_DEVELOPMENT_ARM_IDS + PARTIAL_V2_ARM_IDS
    ordered_arm_ids = tuple(
        arm_id for arm_id in report_arm_order if arm_id in declared_arm_ids
    )
    index = _record_index(all_records)

    def comparison_group(
        group_name: str,
        candidate_ids: Sequence[str],
        metrics: Sequence[str],
        primary_metric: str,
    ) -> dict[str, Any] | None:
        arm_ids = tuple(
            arm_id for arm_id in candidate_ids if arm_id in declared_arm_ids
        )
        if not arm_ids:
            return None
        return {
            "group": group_name,
            "arm_ids": list(arm_ids),
            "short_labels": {
                arm_id: SHORT_LABEL_BY_ARM[arm_id] for arm_id in arm_ids
            },
            "declared_metrics": list(metrics),
            "primary_metric": primary_metric,
            "algorithm_cell_summaries": _algorithm_cell_summaries(
                bank=bank,
                records=all_records,
                arm_ids=arm_ids,
                metrics=metrics,
            ),
            "pairwise_by_height_and_placement": _pairwise_cells(
                bank=bank,
                index=index,
                arm_ids=arm_ids,
                metrics=metrics,
            ),
            "depth_interactions": _depth_interactions(
                bank=bank,
                index=index,
                arm_ids=arm_ids,
                metrics=metrics,
            ),
            "placement_interactions": _placement_interactions(
                bank=bank,
                index=index,
                arm_ids=arm_ids,
                metrics=metrics,
            ),
        }

    full_group = comparison_group(
        "fully_labeled",
        FULL_DEVELOPMENT_ARM_IDS,
        DECLARED_METRICS,
        "ad_f1",
    )
    partial_group = comparison_group(
        "partial",
        PARTIAL_V2_ARM_IDS,
        PARTIAL_DECLARED_METRICS,
        "grf",
    )
    groups = {
        group["group"]: group
        for group in (full_group, partial_group)
        if group is not None
    }
    compatibility_group = full_group or partial_group
    if compatibility_group is None:  # pragma: no cover - arm validation prevents this
        raise ValueError("No reportable shortlist comparison group is present.")

    bank_resource_execution = _bank_resource_execution(bank)
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "complete",
        "created_at_utc": created_at_utc or datetime.now(timezone.utc).isoformat(),
        "scientific_role": bank["scientific_role"],
        "bank_id": bank["bank_id"],
        "bank_root": str(bank_root),
        "run_id": run_ids[0] if len(run_ids) == 1 else None,
        "run_ids": run_ids,
        "result_root": str(loaded[0][0]) if len(loaded) == 1 else None,
        "result_roots": [str(root) for root, _result in loaded],
        "block_count": int(bank["block_count"]),
        "declared_condition_count": int(bank["declared_condition_count"]),
        "available_condition_count": int(bank["available_condition_count"]),
        "unavailable_condition_count": int(bank["unavailable_condition_count"]),
        "arm_count": len(ordered_arm_ids),
        "record_count": len(all_records),
        "heights": list(bank["heights"]),
        "placement_policies": list(bank["placement_policies"]),
        "shortlist_arm_ids": list(ordered_arm_ids),
        "short_labels": {
            arm_id: SHORT_LABEL_BY_ARM[arm_id] for arm_id in ordered_arm_ids
        },
        "declared_metrics": list(compatibility_group["declared_metrics"]),
        "primary_metric": compatibility_group["primary_metric"],
        "comparison_groups": groups,
        "dependence_contract": {
            "independent_unit": "truth_block",
            "independent_block_count": int(bank["block_count"]),
            "conditions_within_block_are_correlated": True,
            "condition_records_are_not_independent_replicates": True,
        },
        "interpretation_contract": {
            "automatic_winner_declared": False,
            "pooled_1200_condition_ranking_generated": False,
            "ad_f1_and_grf_combined_into_one_score": False,
            "cross_output_family_comparisons_generated": False,
            "placement_is_practical_policy_not_fixed_budget_causal_effect": True,
            "h38_role": "development_boundary_stress_not_automatic_paper_height",
        },
        "semantic_gate_by_arm": semantic_gate_by_arm,
        "record_execution": dict(
            loaded[0][1]["resources"]["record_execution"]
        ),
        "record_execution_by_run": record_execution_by_run,
        "bank_resource_execution": bank_resource_execution,
        "availability": _availability_summary(bank),
        "adaptive_radius_diagnostics": _adaptive_radius_diagnostics(all_records),
        "algorithm_cell_summaries": compatibility_group[
            "algorithm_cell_summaries"
        ],
        "pairwise_by_height_and_placement": compatibility_group[
            "pairwise_by_height_and_placement"
        ],
        "depth_interactions": compatibility_group["depth_interactions"],
        "placement_interactions": compatibility_group[
            "placement_interactions"
        ],
        "condition_diagnostics": _diagnostic_summaries(
            bank_root=bank_root,
            bank=bank,
        ),
    }
    return report


def _flatten_summary(prefix: str, value: Mapping[str, Any] | None) -> dict[str, Any]:
    fields = ("count", "minimum", "median", "mean", "maximum")
    return {
        f"{prefix}_{field}": None if value is None else value.get(field)
        for field in fields
    }


def _flatten_effect(value: Mapping[str, Any] | None) -> dict[str, Any]:
    fields = (
        "count",
        "minimum",
        "median",
        "mean",
        "maximum",
        "conditional_gain_mean",
        "conditional_loss_mean",
        "worst_five_or_available_mean",
    )
    return {field: None if value is None else value.get(field) for field in fields}


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _comparison_csv_rows(
    comparison: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    metric_rows = []
    resource_rows = []
    for row in comparison["algorithm_cell_summaries"]:
        for metric, summary in row["metrics"].items():
            metric_rows.append(
                {
                    "height": row["height"],
                    "placement_policy": row["placement_policy"],
                    "arm_id": row["arm_id"],
                    "short_label": row["short_label"],
                    "metric": metric,
                    **_flatten_summary("score", summary),
                    "success_count": row["success_count"],
                    "failure_count": row["failure_count"],
                }
            )
        resource_rows.append(
            {
                "height": row["height"],
                "placement_policy": row["placement_policy"],
                "arm_id": row["arm_id"],
                "short_label": row["short_label"],
                **_flatten_summary("runtime_seconds", row["runtime_seconds"]),
                **_flatten_summary("peak_rss_bytes", row["peak_rss_bytes"]),
                "success_count": row["success_count"],
                "failure_count": row["failure_count"],
            }
        )
    pairwise_rows = []
    for row in comparison["pairwise_by_height_and_placement"]:
        pairwise_rows.append(
            {
                "arm_a": row["arm_a"],
                "label_a": row["label_a"],
                "arm_b": row["arm_b"],
                "label_b": row["label_b"],
                "metric": row["metric"],
                "height": row["height"],
                "placement_policy": row["placement_policy"],
                "wins": row["wtl"]["wins"],
                "ties": row["wtl"]["ties"],
                "losses": row["wtl"]["losses"],
                **_flatten_effect(row["effect"]),
                "a_only_success_count": row["a_only_success_count"],
                "b_only_success_count": row["b_only_success_count"],
                "both_failure_count": row["both_failure_count"],
            }
        )
    interaction_rows = {}
    for name in ("depth_interactions", "placement_interactions"):
        values = []
        for row in comparison[name]:
            base = {
                key: value
                for key, value in row.items()
                if key not in {"effect", "wtl"}
            }
            values.append(
                {
                    **base,
                    "wins": row["wtl"]["wins"],
                    "ties": row["wtl"]["ties"],
                    "losses": row["wtl"]["losses"],
                    **_flatten_effect(row["effect"]),
                }
            )
        interaction_rows[name] = values
    return {
        "metric_summary": metric_rows,
        "resource_summary": resource_rows,
        "pairwise_by_cell": pairwise_rows,
        "depth_interactions": interaction_rows["depth_interactions"],
        "placement_interactions": interaction_rows["placement_interactions"],
    }


def _csv_rows(report: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    rows = _comparison_csv_rows(report)
    diagnostic_rows = []
    for row in report["condition_diagnostics"]:
        diagnostic_rows.append(
            {
                "height": row["height"],
                "placement_policy": row["placement_policy"],
                "available_case_count": row["available_case_count"],
                **_flatten_summary(
                    "selected_occurrence_count", row["selected_occurrence_count"]
                ),
                **_flatten_summary(
                    "selected_unique_state_count", row["selected_unique_state_count"]
                ),
                **{
                    key: value
                    for field, summary in row["truth_sampling_diagnostics"].items()
                    for key, value in _flatten_summary(field, summary).items()
                },
                "observed_only_representable_count": row[
                    "observed_only_representable_count"
                ],
                **_flatten_summary(
                    "distance_runtime_seconds", row["distance_runtime_seconds"]
                ),
                **_flatten_summary(
                    "distance_peak_rss_bytes", row["distance_peak_rss_bytes"]
                ),
            }
        )
    adaptive_rows = []
    for diagnostic in report["adaptive_radius_diagnostics"]:
        for row in diagnostic["by_height_and_placement"]:
            adaptive_rows.append(
                {
                    "arm_id": diagnostic["arm_id"],
                    "short_label": diagnostic["short_label"],
                    "radius_policy": diagnostic["radius_policy"],
                    "height": row["height"],
                    "placement_policy": row["placement_policy"],
                    **_flatten_summary("effective_radius", row["effective_radius"]),
                    **_flatten_summary(
                        "nearest_rank_quantile_distance",
                        row["nearest_rank_quantile_distance"],
                    ),
                    "frozen_radius_candidate_coverage_fraction": row[
                        "frozen_radius_candidate_coverage_fraction"
                    ],
                    "child_decision_count": row["child_decision_count"],
                    "selected_parent_count": row["selected_parent_count"],
                    "selected_parent_fraction": row["selected_parent_fraction"],
                    "copy_up_count": row["copy_up_count"],
                    "copy_up_fraction": row["copy_up_fraction"],
                }
            )
    return {
        **rows,
        "adaptive_radius_by_cell": adaptive_rows,
        "condition_diagnostics": diagnostic_rows,
        "availability": [
            {
                **{key: value for key, value in row.items() if key != "failure_types"},
                "failure_types": ";".join(
                    f"{key}:{value}"
                    for key, value in sorted(row["failure_types"].items())
                ),
            }
            for row in report["availability"]
        ],
    }


def _format(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _mean_score_table(
    report: Mapping[str, Any],
    metric: str,
) -> list[str]:
    arm_ids = tuple(report.get("arm_ids", report["shortlist_arm_ids"]))
    index = {
        (row["height"], row["placement_policy"], row["arm_id"]): row
        for row in report["algorithm_cell_summaries"]
    }
    lines = [
        f"### Mean {metric}",
        "",
        "| Height | Placement | "
        + " | ".join(SHORT_LABEL_BY_ARM[arm_id] for arm_id in arm_ids)
        + " |",
        "|---:|---|" + "---:|" * len(arm_ids),
    ]
    for height in report["heights"]:
        for policy in report["placement_policies"]:
            values = []
            for arm_id in arm_ids:
                summary = index[(height, policy, arm_id)]["metrics"][metric]
                values.append(None if summary is None else summary["mean"])
            lines.append(
                f"| H{height} | {policy} | "
                + " | ".join(_format(value) for value in values)
                + " |"
            )
    return lines


def _principal_pair_table(
    report: Mapping[str, Any],
    metric: str,
    principal_pairs: Sequence[tuple[str, str]] = PRINCIPAL_PAIRS,
) -> list[str]:
    rows = [
        row
        for row in report["pairwise_by_height_and_placement"]
        if (row["arm_a"], row["arm_b"]) in principal_pairs
        and row["metric"] == metric
    ]
    lines = [
        f"### Principal paired {metric} contrasts",
        "",
        "Positive deltas favor the first method.",
        "",
        "| Pair | Height | Placement | W/T/L | Mean delta | Median | Worst five |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        effect = row["effect"]
        wtl = row["wtl"]
        lines.append(
            f"| {row['label_a']}-{row['label_b']} | H{row['height']} | "
            f"{row['placement_policy']} | {wtl['wins']}/{wtl['ties']}/{wtl['losses']} | "
            f"{_format(None if effect is None else effect['mean'])} | "
            f"{_format(None if effect is None else effect['median'])} | "
            f"{_format(None if effect is None else effect['worst_five_or_available_mean'])} |"
        )
    return lines


def _h34_h38_table(
    report: Mapping[str, Any],
    principal_pairs: Sequence[tuple[str, str]] = PRINCIPAL_PAIRS,
) -> list[str]:
    rows = [
        row
        for row in report["depth_interactions"]
        if (row["arm_a"], row["arm_b"]) in principal_pairs
        and row["metric"] in {"ad_f1", "grf"}
        and row["lower_height"] == 34
        and row["upper_height"] == 38
    ]
    if not rows:
        return []
    lines = [
        "### H34 to H38 change in method contrast",
        "",
        "Positive means the first method's advantage increased at H38.",
        "",
        "| Pair | Metric | Placement | W/T/L | Mean change | Median |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in rows:
        effect = row["effect"]
        wtl = row["wtl"]
        lines.append(
            f"| {row['label_a']}-{row['label_b']} | {row['metric']} | "
            f"{row['placement_policy']} | {wtl['wins']}/{wtl['ties']}/{wtl['losses']} | "
            f"{_format(None if effect is None else effect['mean'])} | "
            f"{_format(None if effect is None else effect['median'])} |"
        )
    return lines


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# CTBF v5 shortlist robustness report",
        "",
        f"Schema: `{report['schema_version']}`  ",
        f"Status: `{report['status']}`  ",
        f"Truth blocks: `{report['block_count']}`  ",
        f"Conditions: `{report['available_condition_count']}` available of "
        f"`{report['declared_condition_count']}` declared  ",
        f"Arm records: `{report['record_count']}`  ",
        "Record execution: `fresh spawned process per case-arm`",
        (
            "Bank resource execution: `fresh-process qualified`."
            if report["bank_resource_execution"][
                "all_bank_resources_fresh_process_qualified"
            ]
            else "Bank resource execution: `unqualified historical context`; "
            "stored inputs and distances remain reusable."
        ),
        "",
        "The truth block is the independent unit. Heights and placements within a "
        "block are correlated. This report deliberately provides no pooled "
        "all-condition ranking, no automatic winner, and no AD-F1/GRF composite.",
        "",
        "Fully labeled and partial outputs are reported as separate comparison "
        "families; no cross-family score or ordering is generated.",
        "",
        "## Availability",
        "",
        "| Height | Placement | Available | Unavailable | Failures |",
        "|---:|---|---:|---:|---|",
    ]
    for row in report["availability"]:
        failures = ", ".join(
            f"{key}={value}" for key, value in sorted(row["failure_types"].items())
        ) or "none"
        lines.append(
            f"| H{row['height']} | {row['placement_policy']} | "
            f"{row['available_count']} | {row['unavailable_count']} | {failures} |"
        )
    for group_name in ("fully_labeled", "partial"):
        group = report["comparison_groups"].get(group_name)
        if group is None:
            continue
        view = {
            **report,
            **group,
            "shortlist_arm_ids": group["arm_ids"],
        }
        principal_pairs = (
            FULL_PRINCIPAL_PAIRS
            if group_name == "fully_labeled"
            else PARTIAL_PRINCIPAL_PAIRS
        )
        lines.extend(
            [
                "",
                (
                    "## Fully labeled comparison"
                    if group_name == "fully_labeled"
                    else "## Partial comparison"
                ),
                "",
                "; ".join(
                    f"{SHORT_LABEL_BY_ARM[arm_id]} = "
                    f"{SHORT_DESCRIPTION_BY_ARM[arm_id]}"
                    for arm_id in group["arm_ids"]
                )
                + ".",
            ]
        )
        for metric in group["declared_metrics"]:
            lines.extend(["", *_mean_score_table(view, metric)])
        principal_metrics = (
            ("ad_f1", "grf")
            if group_name == "fully_labeled"
            else ("grf",)
        )
        for metric in principal_metrics:
            lines.extend(
                [
                    "",
                    *_principal_pair_table(
                        view,
                        metric,
                        principal_pairs=principal_pairs,
                    ),
                ]
            )
        h34_h38 = _h34_h38_table(
            view,
            principal_pairs=principal_pairs,
        )
        if h34_h38:
            lines.extend(["", *h34_h38])
    for diagnostic in report["adaptive_radius_diagnostics"]:
        lines.extend(
            [
                "",
                f"## Adaptive-radius diagnostics ({diagnostic['short_label']})",
                "",
                f"Policy: `{diagnostic['radius_policy']}`.",
                "",
                "| Height | Placement | Mean radius | Frozen candidate coverage | "
                "Selected-parent fraction | Copy-up fraction |",
                "|---:|---|---:|---:|---:|---:|",
            ]
        )
        for row in diagnostic["by_height_and_placement"]:
            lines.append(
                f"| H{row['height']} | {row['placement_policy']} | "
                f"{_format(row['effective_radius']['mean'])} | "
                f"{_format(row['frozen_radius_candidate_coverage_fraction'])} | "
                f"{_format(row['selected_parent_fraction'])} | "
                f"{_format(row['copy_up_fraction'])} |"
            )
    lines.extend(
        [
            "",
            "## Remaining outputs",
            "",
            "Complete within-family contrasts, depth and placement interactions, "
            "condition diagnostics, tails, failures, runtime, and memory are in "
            "the adjacent CSV files and `summary.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(
    *,
    result_root: Path | str | None = None,
    result_roots: Sequence[Path | str] | None = None,
    output_root: Path | str,
    expected_block_count: int | None = None,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    report = build_report(
        result_root=result_root,
        result_roots=result_roots,
        expected_block_count=expected_block_count,
        created_at_utc=created_at_utc,
    )
    root = ensure_new_output_root(output_root)
    write_json(root / "summary.json", report)
    for name, rows in _csv_rows(report).items():
        _write_csv(root / f"{name}.csv", rows)
    compatibility_group = (
        "fully_labeled"
        if "fully_labeled" in report["comparison_groups"]
        else "partial"
    )
    for group_name, group in report["comparison_groups"].items():
        if group_name == compatibility_group:
            continue
        for name, rows in _comparison_csv_rows(group).items():
            _write_csv(root / f"{group_name}_{name}.csv", rows)
    (root / "report.md").write_text(_markdown(report), encoding="utf-8")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-root",
        type=Path,
        action="append",
        required=True,
        help="Repeat for the completed A-D run and every non-overlapping extension run.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-block-count", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    report = write_report(
        result_roots=arguments.result_root,
        output_root=arguments.output_root,
        expected_block_count=arguments.expected_block_count,
    )
    print(
        f"complete: {report['arm_count']} algorithms on "
        f"{report['available_condition_count']} available conditions"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
