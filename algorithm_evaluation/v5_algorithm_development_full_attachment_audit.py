"""Audit hard biopsy-layer attachments in the CTBF v5 full-tree comparison.

This non-overwriting development diagnostic reads two completed immutable
algorithm-result roots.  It does not regenerate simulation, distances, or
reconstructions.  The audit asks whether the ordered full method's immediate
non-same-state biopsy-layer attachments become more burdensome with depth and
whether that burden is associated with its paired AD-F1 difference from the
pooled plausible-parsimony incumbent.

No formal significance test or automatic algorithm-replacement verdict is
produced.  H14, H24, and H34 from one block remain dependent observations.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import math
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from algorithm_evaluation.paper_pipeline_contract import read_json
from algorithm_evaluation.v5_algorithm_development_common import (
    ARM_SPEC_BY_ID,
    BIOPSY_GUIDED_FULL_INCUMBENT_ID,
    DEFAULT_BLOCK_COUNT,
    HEIGHT_SCHEDULES,
    INFERRED_COPY_INCUMBENT_ID,
    LEGACY_RUN_SCHEMA_VERSION,
    RUN_SCHEMA_VERSION,
    ensure_new_output_root,
    numeric_summary,
    write_json,
)
from algorithm_evaluation.v5_algorithm_development_run import (
    RESULT_NAME as RUN_RESULT_NAME,
)
from reconstructor_biopsy_blocks import (
    BIOPSY_GUIDED_AUDIT_COUNTERS,
    BIOPSY_GUIDED_AUDIT_SCHEMA_VERSION,
    FROZEN_TRANSITION_PARENT_ELIGIBILITY_POLICY,
)


FULL_ATTACHMENT_AUDIT_SCHEMA_VERSION = (
    "ctbf-v5-full-attachment-mechanism-audit-v2"
)
RESULT_NAME = "full_attachment_audit.json"
REPORT_NAME = "report.md"
TIE_TOLERANCE = 1e-12

ORDERED_ARM_ID = BIOPSY_GUIDED_FULL_INCUMBENT_ID
POOLED_ARM_ID = INFERRED_COPY_INCUMBENT_ID

COUNT_FIELDS = (
    "child_decision_count",
    "same_state_selected_count",
    "one_plausible_parent_count",
    "unique_minimum_parent_count",
    "non_same_state_hard_attachment_count",
    "minimum_distance_tie_count",
    "tie_deferred_count",
    "no_plausible_parent_count",
    "selected_parent_count",
    "copy_up_count",
)
FRACTION_FIELDS = (
    "same_state_attachment_fraction",
    "one_plausible_attachment_fraction",
    "unique_minimum_attachment_fraction",
    "non_same_state_hard_attachment_fraction",
    "minimum_distance_tie_fraction",
    "tie_deferred_fraction",
    "no_plausible_parent_fraction",
    "selected_parent_fraction",
    "copy_up_fraction",
    "non_same_state_share_of_selected_attachments",
)


def _fraction(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator / denominator) if denominator else None


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number.")
    return result


def _nonnegative_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer.")
    return value


def derive_attachment_measures(audit: Mapping[str, Any]) -> dict[str, int | float | None]:
    """Validate one deferred-tie audit and derive prespecified burden measures."""

    if audit.get("schema_version") != BIOPSY_GUIDED_AUDIT_SCHEMA_VERSION:
        raise ValueError("Unknown biopsy-layer decision-audit schema.")
    if (
        audit.get("parent_eligibility_policy")
        != FROZEN_TRANSITION_PARENT_ELIGIBILITY_POLICY
    ):
        raise ValueError("Unknown biopsy-layer parent-eligibility policy.")
    missing = set(BIOPSY_GUIDED_AUDIT_COUNTERS) - set(audit)
    if missing:
        raise ValueError(
            f"Biopsy-layer decision audit is missing counters: {sorted(missing)}."
        )
    counters = {
        field: _nonnegative_integer(audit[field], field)
        for field in BIOPSY_GUIDED_AUDIT_COUNTERS
    }
    if counters["child_decision_count"] <= 0:
        raise ValueError("The full attachment audit requires at least one child decision.")
    if counters["child_decision_count"] != (
        counters["selected_parent_count"] + counters["copy_up_count"]
    ):
        raise ValueError("Selected-parent and copy-up counts do not cover child decisions.")
    if counters["selected_parent_count"] != (
        counters["same_state_selected_count"]
        + counters["one_plausible_parent_count"]
        + counters["unique_minimum_parent_count"]
        + counters["tie_parent_selected_count"]
    ):
        raise ValueError("Selected-parent decision categories are inconsistent.")
    if counters["multiple_plausible_parent_count"] != (
        counters["unique_minimum_parent_count"]
        + counters["minimum_distance_tie_count"]
    ):
        raise ValueError("Multiple-plausible-parent categories are inconsistent.")
    if counters["minimum_distance_tie_count"] != (
        counters["tie_parent_selected_count"] + counters["tie_deferred_count"]
    ):
        raise ValueError("Minimum-distance-tie categories are inconsistent.")
    if counters["copy_up_count"] != (
        counters["no_plausible_parent_count"] + counters["tie_deferred_count"]
    ):
        raise ValueError("Copy-up categories are inconsistent for deferred tie handling.")
    if counters["tie_parent_selected_count"] != 0:
        raise ValueError("The ordered deferred-tie arm unexpectedly selected a tied parent.")

    child_count = counters["child_decision_count"]
    selected_count = counters["selected_parent_count"]
    hard_count = (
        counters["one_plausible_parent_count"]
        + counters["unique_minimum_parent_count"]
    )
    return {
        **counters,
        "non_same_state_hard_attachment_count": hard_count,
        "same_state_attachment_fraction": _fraction(
            counters["same_state_selected_count"], child_count
        ),
        "one_plausible_attachment_fraction": _fraction(
            counters["one_plausible_parent_count"], child_count
        ),
        "unique_minimum_attachment_fraction": _fraction(
            counters["unique_minimum_parent_count"], child_count
        ),
        "non_same_state_hard_attachment_fraction": _fraction(
            hard_count, child_count
        ),
        "minimum_distance_tie_fraction": _fraction(
            counters["minimum_distance_tie_count"], child_count
        ),
        "tie_deferred_fraction": _fraction(
            counters["tie_deferred_count"], child_count
        ),
        "no_plausible_parent_fraction": _fraction(
            counters["no_plausible_parent_count"], child_count
        ),
        "selected_parent_fraction": _fraction(selected_count, child_count),
        "copy_up_fraction": _fraction(counters["copy_up_count"], child_count),
        "non_same_state_share_of_selected_attachments": _fraction(
            hard_count, selected_count
        ),
    }


def _result_path(path: Path | str) -> Path:
    root = Path(path).expanduser().resolve()
    return root / RUN_RESULT_NAME if root.is_dir() else root


def _load_complete_run(
    path: Path | str,
    *,
    expected_block_count: int,
) -> tuple[Path, dict[str, Any]]:
    result_path = _result_path(path)
    result = read_json(result_path)
    if result.get("schema_version") not in {
        RUN_SCHEMA_VERSION,
        LEGACY_RUN_SCHEMA_VERSION,
    }:
        raise ValueError(f"Unknown development-run schema in {result_path}.")
    if result.get("status") != "complete":
        raise ValueError(f"Development run is not complete: {result_path}.")
    if result.get("block_count") != expected_block_count:
        raise ValueError(f"Unexpected block count in {result_path}.")
    expected_conditions = expected_block_count * len(HEIGHT_SCHEDULES)
    if result.get("condition_count") != expected_conditions:
        raise ValueError(f"Unexpected condition count in {result_path}.")
    records = result.get("records")
    if not isinstance(records, list):
        raise ValueError(f"Development run has no record list: {result_path}.")
    expected_records = result.get("expected_record_count")
    if (
        isinstance(expected_records, bool)
        or not isinstance(expected_records, int)
        or expected_records != len(records)
        or result.get("completed_record_count") != len(records)
    ):
        raise ValueError(f"Development run inventory is incomplete: {result_path}.")
    if not isinstance(result.get("arm_specs"), list):
        raise ValueError(f"Development run has no arm declarations: {result_path}.")
    return result_path, result


def _validate_arm_declaration(result: Mapping[str, Any], arm_id: str) -> None:
    declarations = [
        declaration
        for declaration in result["arm_specs"]
        if declaration.get("arm_id") == arm_id
    ]
    if len(declarations) != 1:
        raise ValueError(f"Run must declare arm {arm_id!r} exactly once.")
    if declarations[0] != ARM_SPEC_BY_ID[arm_id].as_record():
        raise ValueError(f"Stored declaration for arm {arm_id!r} changed.")


def _target_records(
    result: Mapping[str, Any],
    arm_id: str,
) -> dict[str, Mapping[str, Any]]:
    _validate_arm_declaration(result, arm_id)
    records = [record for record in result["records"] if record.get("arm_id") == arm_id]
    expected_count = int(result["condition_count"])
    if len(records) != expected_count:
        raise ValueError(f"Arm {arm_id!r} does not cover every condition once.")
    by_case: dict[str, Mapping[str, Any]] = {}
    by_block: dict[int, set[int]] = {}
    for record in records:
        case_id = str(record.get("case_id"))
        if case_id in by_case:
            raise ValueError(f"Duplicate target record for {case_id!r}/{arm_id!r}.")
        if record.get("status") != "success":
            raise ValueError(
                f"Target record {case_id!r}/{arm_id!r} is not successful; "
                "the audit never imputes failed comparisons."
            )
        metrics = record.get("metrics")
        if not isinstance(metrics, Mapping) or "ad_f1" not in metrics:
            raise ValueError(f"Target record {case_id!r}/{arm_id!r} lacks AD-F1.")
        _finite_number(metrics["ad_f1"], "ad_f1")
        block_index = _nonnegative_integer(record.get("block_index"), "block_index")
        height = _nonnegative_integer(record.get("height"), "height")
        if height not in HEIGHT_SCHEDULES:
            raise ValueError(f"Unexpected height H{height} in target records.")
        by_block.setdefault(block_index, set()).add(height)
        by_case[case_id] = record
    expected_heights = set(HEIGHT_SCHEDULES)
    expected_blocks = set(range(int(result["block_count"])))
    if set(by_block) != expected_blocks or any(
        heights != expected_heights for heights in by_block.values()
    ):
        raise ValueError("Target records do not form complete paired height blocks.")
    return by_case


def _average_ranks(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=float)
    start = 0
    while start < array.size:
        end = start + 1
        while end < array.size and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def _pearson(x_values: Sequence[float], y_values: Sequence[float]) -> float | None:
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    if x.size < 2 or y.size != x.size:
        return None
    x = x - np.mean(x)
    y = y - np.mean(y)
    denominator = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    if denominator == 0.0:
        return None
    value = float(np.sum(x * y) / denominator)
    return max(-1.0, min(1.0, value))


def association_summary(
    predictor: Sequence[float],
    outcome: Sequence[float],
) -> dict[str, Any]:
    """Return descriptive Pearson and tie-aware Spearman correlations."""

    if len(predictor) != len(outcome):
        raise ValueError("Association vectors must have equal lengths.")
    x = [_finite_number(value, "predictor") for value in predictor]
    y = [_finite_number(value, "outcome") for value in outcome]
    return {
        "case_count": len(x),
        "pearson_r": _pearson(x, y),
        "spearman_r": _pearson(_average_ranks(x), _average_ranks(y)),
        "formal_significance_test_run": False,
    }


def _available_numeric_summary(values: Iterable[float | None]) -> dict[str, Any] | None:
    available = [float(value) for value in values if value is not None]
    return numeric_summary(available)


def _sign_counts(values: Sequence[float]) -> dict[str, int]:
    return {
        "positive_count": sum(value > TIE_TOLERANCE for value in values),
        "tie_count": sum(abs(value) <= TIE_TOLERANCE for value in values),
        "negative_count": sum(value < -TIE_TOLERANCE for value in values),
    }


def _effect_summary(values: Sequence[float]) -> dict[str, Any]:
    return {
        **(numeric_summary(values) or {}),
        "ordered_win_count": sum(value > TIE_TOLERANCE for value in values),
        "tie_count": sum(abs(value) <= TIE_TOLERANCE for value in values),
        "ordered_loss_count": sum(value < -TIE_TOLERANCE for value in values),
    }


def _outcome_strata(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    predicates = {
        "ordered_win": lambda value: value > TIE_TOLERANCE,
        "tie": lambda value: abs(value) <= TIE_TOLERANCE,
        "ordered_loss": lambda value: value < -TIE_TOLERANCE,
    }
    result = {}
    for label, predicate in predicates.items():
        selected = [
            row
            for row in rows
            if predicate(float(row["ad_f1_delta_ordered_minus_pooled"]))
        ]
        result[label] = {
            "case_count": len(selected),
            "non_same_state_hard_attachment_count": _available_numeric_summary(
                row["non_same_state_hard_attachment_count"] for row in selected
            ),
            "non_same_state_hard_attachment_fraction": _available_numeric_summary(
                row["non_same_state_hard_attachment_fraction"] for row in selected
            ),
        }
    return result


def _summarize_height(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    deltas = [float(row["ad_f1_delta_ordered_minus_pooled"]) for row in rows]
    hard_counts = [float(row["non_same_state_hard_attachment_count"]) for row in rows]
    hard_fractions = [
        float(row["non_same_state_hard_attachment_fraction"]) for row in rows
    ]
    return {
        "case_count": len(rows),
        "count_summaries": {
            field: _available_numeric_summary(row[field] for row in rows)
            for field in COUNT_FIELDS
        },
        "fraction_summaries": {
            field: _available_numeric_summary(row[field] for row in rows)
            for field in FRACTION_FIELDS
        },
        "ad_f1_effect_ordered_minus_pooled": _effect_summary(deltas),
        "association_with_ad_f1_effect": {
            "primary_fraction_predictor": association_summary(
                hard_fractions, deltas
            ),
            "size_sensitive_count_predictor": association_summary(
                hard_counts, deltas
            ),
        },
        "burden_by_ad_f1_outcome": _outcome_strata(rows),
    }


def _transition_summary(
    rows_by_block_height: Mapping[tuple[int, int], Mapping[str, Any]],
    low_height: int,
    high_height: int,
    block_count: int,
) -> dict[str, Any]:
    records = []
    for block_index in range(block_count):
        low = rows_by_block_height[(block_index, low_height)]
        high = rows_by_block_height[(block_index, high_height)]
        records.append(
            {
                "hard_fraction_change": float(
                    high["non_same_state_hard_attachment_fraction"]
                )
                - float(low["non_same_state_hard_attachment_fraction"]),
                "hard_count_change": float(
                    high["non_same_state_hard_attachment_count"]
                )
                - float(low["non_same_state_hard_attachment_count"]),
                "ad_f1_effect_change": float(
                    high["ad_f1_delta_ordered_minus_pooled"]
                )
                - float(low["ad_f1_delta_ordered_minus_pooled"]),
            }
        )
    fraction_changes = [row["hard_fraction_change"] for row in records]
    count_changes = [row["hard_count_change"] for row in records]
    effect_changes = [row["ad_f1_effect_change"] for row in records]
    return {
        "from_height": low_height,
        "to_height": high_height,
        "independent_block_count": block_count,
        "non_same_state_hard_attachment_fraction_change": {
            **(numeric_summary(fraction_changes) or {}),
            **_sign_counts(fraction_changes),
        },
        "non_same_state_hard_attachment_count_change": {
            **(numeric_summary(count_changes) or {}),
            **_sign_counts(count_changes),
        },
        "ad_f1_effect_change": {
            **(numeric_summary(effect_changes) or {}),
            **_sign_counts(effect_changes),
        },
        "association_of_burden_change_with_ad_f1_effect_change": {
            "primary_fraction_change": association_summary(
                fraction_changes, effect_changes
            ),
            "size_sensitive_count_change": association_summary(
                count_changes, effect_changes
            ),
        },
    }


def build_full_attachment_audit(
    *,
    ordered_result_root: Path | str,
    pooled_result_root: Path | str,
    expected_block_count: int = DEFAULT_BLOCK_COUNT,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    if (
        isinstance(expected_block_count, bool)
        or not isinstance(expected_block_count, int)
        or expected_block_count <= 0
    ):
        raise ValueError("expected_block_count must be a positive integer.")
    ordered_path, ordered_result = _load_complete_run(
        ordered_result_root,
        expected_block_count=expected_block_count,
    )
    pooled_path, pooled_result = _load_complete_run(
        pooled_result_root,
        expected_block_count=expected_block_count,
    )
    if ordered_path == pooled_path:
        raise ValueError("Ordered and pooled inputs must be distinct result roots.")
    for field in ("bank_id", "bank_root", "block_count", "condition_count"):
        if ordered_result.get(field) != pooled_result.get(field):
            raise ValueError(f"Ordered and pooled runs disagree on {field}.")

    ordered_records = _target_records(ordered_result, ORDERED_ARM_ID)
    pooled_records = _target_records(pooled_result, POOLED_ARM_ID)
    if set(ordered_records) != set(pooled_records):
        raise ValueError("Ordered and pooled target arms cover different cases.")

    condition_records = []
    for case_id in sorted(
        ordered_records,
        key=lambda value: (
            int(ordered_records[value]["block_index"]),
            int(ordered_records[value]["height"]),
        ),
    ):
        ordered = ordered_records[case_id]
        pooled = pooled_records[case_id]
        descriptor = (int(ordered["block_index"]), int(ordered["height"]))
        if descriptor != (int(pooled["block_index"]), int(pooled["height"])):
            raise ValueError(f"Case descriptor changed between runs for {case_id!r}.")
        metadata = ordered.get("reconstruction_metadata")
        decision_audit = (
            metadata.get("biopsy_layer_decision_audit")
            if isinstance(metadata, Mapping)
            else None
        )
        if not isinstance(decision_audit, Mapping):
            raise ValueError(
                f"Ordered target record {case_id!r} lacks its decision audit."
            )
        measures = derive_attachment_measures(decision_audit)
        ordered_ad_f1 = _finite_number(ordered["metrics"]["ad_f1"], "ordered_ad_f1")
        pooled_ad_f1 = _finite_number(pooled["metrics"]["ad_f1"], "pooled_ad_f1")
        condition_records.append(
            {
                "case_id": case_id,
                "block_index": descriptor[0],
                "height": descriptor[1],
                "ordered_ad_f1": ordered_ad_f1,
                "pooled_ad_f1": pooled_ad_f1,
                "ad_f1_delta_ordered_minus_pooled": ordered_ad_f1 - pooled_ad_f1,
                **{field: measures[field] for field in COUNT_FIELDS},
                **{field: measures[field] for field in FRACTION_FIELDS},
            }
        )

    rows_by_block_height = {
        (int(row["block_index"]), int(row["height"])): row
        for row in condition_records
    }
    expected_pairs = {
        (block_index, height)
        for block_index in range(expected_block_count)
        for height in HEIGHT_SCHEDULES
    }
    if set(rows_by_block_height) != expected_pairs:
        raise ValueError("Paired audit records do not cover the expected block-height grid.")

    height_summaries = {
        str(height): _summarize_height(
            [row for row in condition_records if row["height"] == height]
        )
        for height in sorted(HEIGHT_SCHEDULES)
    }
    heights = sorted(HEIGHT_SCHEDULES)
    transition_pairs = [
        (heights[index], heights[index + 1])
        for index in range(len(heights) - 1)
    ]
    if len(heights) > 2:
        transition_pairs.append((heights[0], heights[-1]))
    transitions = {
        f"H{low}_to_H{high}": _transition_summary(
            rows_by_block_height,
            low,
            high,
            expected_block_count,
        )
        for low, high in transition_pairs
    }

    block_rows = []
    for block_index in range(expected_block_count):
        rows = [rows_by_block_height[(block_index, height)] for height in heights]
        block_rows.append(
            {
                "block_index": block_index,
                "mean_non_same_state_hard_attachment_fraction": float(
                    statistics.fmean(
                        float(row["non_same_state_hard_attachment_fraction"])
                        for row in rows
                    )
                ),
                "mean_non_same_state_hard_attachment_count": float(
                    statistics.fmean(
                        float(row["non_same_state_hard_attachment_count"])
                        for row in rows
                    )
                ),
                "mean_ad_f1_delta_ordered_minus_pooled": float(
                    statistics.fmean(
                        float(row["ad_f1_delta_ordered_minus_pooled"])
                        for row in rows
                    )
                ),
            }
        )

    return {
        "schema_version": FULL_ATTACHMENT_AUDIT_SCHEMA_VERSION,
        "status": "complete",
        "created_at_utc": created_at_utc or datetime.now(timezone.utc).isoformat(),
        "scientific_role": "adaptive_method_development_mechanism_diagnostic",
        "automatic_replacement_decision_declared": False,
        "formal_significance_tests_run": False,
        "hypothesis": (
            "Increasing evolutionary depth increases the burden of immediate "
            "non-same-state biopsy-layer attachments, and greater burden is "
            "associated with a lower ordered-minus-pooled AD-F1 effect."
        ),
        "burden_contract": {
            "hard_attachment_count": (
                "one_plausible_parent_count + unique_minimum_parent_count"
            ),
            "primary_predictor": (
                "non_same_state_hard_attachment_count / child_decision_count"
            ),
            "size_sensitive_secondary_predictor": (
                "non_same_state_hard_attachment_count"
            ),
            "same_state_semantics": (
                "Same state means the candidate and child share biological "
                "state_label/cell_id. Non-same-state does not by itself prove "
                "strictly positive CNP distance."
            ),
            "local_commitment_semantics": (
                "The deferred-tie arm commits single plausible parents and "
                "unique closest parents immediately; only exact closest-parent "
                "ties and absent plausible parents are copied upward."
            ),
        },
        "interpretation_contract": {
            "paired_effect": "ordered AD-F1 minus pooled AD-F1",
            "negative_association_direction": (
                "Greater hard-attachment burden accompanies a smaller or more "
                "negative ordered-method advantage."
            ),
            "dependence": (
                "The three heights share each truth block. Per-height and paired "
                "height-transition summaries retain 100 independent blocks; the "
                "300 condition rows are not 300 independent simulations."
            ),
            "causal_limit": (
                "Association can motivate one same-state-only attachment "
                "candidate but cannot establish that local attachments caused "
                "the AD-F1 difference."
            ),
        },
        "ordered_result": {
            "path": str(ordered_path),
            "run_id": ordered_result["run_id"],
            "arm_id": ORDERED_ARM_ID,
        },
        "pooled_result": {
            "path": str(pooled_path),
            "run_id": pooled_result["run_id"],
            "arm_id": POOLED_ARM_ID,
        },
        "bank_id": ordered_result["bank_id"],
        "bank_root": ordered_result["bank_root"],
        "independent_block_count": expected_block_count,
        "condition_count": len(condition_records),
        "paired_heights": heights,
        "height_summaries": height_summaries,
        "paired_height_transitions": transitions,
        "independent_block_summary": {
            "block_count": expected_block_count,
            "mean_hard_attachment_fraction": numeric_summary(
                row["mean_non_same_state_hard_attachment_fraction"]
                for row in block_rows
            ),
            "mean_hard_attachment_count": numeric_summary(
                row["mean_non_same_state_hard_attachment_count"]
                for row in block_rows
            ),
            "mean_ad_f1_effect": _effect_summary(
                [row["mean_ad_f1_delta_ordered_minus_pooled"] for row in block_rows]
            ),
            "association_with_mean_ad_f1_effect": {
                "primary_fraction_predictor": association_summary(
                    [
                        row["mean_non_same_state_hard_attachment_fraction"]
                        for row in block_rows
                    ],
                    [row["mean_ad_f1_delta_ordered_minus_pooled"] for row in block_rows],
                ),
                "size_sensitive_count_predictor": association_summary(
                    [
                        row["mean_non_same_state_hard_attachment_count"]
                        for row in block_rows
                    ],
                    [row["mean_ad_f1_delta_ordered_minus_pooled"] for row in block_rows],
                ),
            },
        },
        "condition_records": condition_records,
    }


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _format(value: Any, digits: int = 6) -> str:
    if value is None:
        return "NA"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def _render_report(result: Mapping[str, Any]) -> str:
    lines = [
        "# CTBF v5 full attachment mechanism audit",
        "",
        f"Status: `{result['status']}`.",
        "",
        "This diagnostic tests whether immediate non-same-state biopsy-layer "
        "attachments help explain the depth-dependent AD-F1 crossover. It makes "
        "no automatic algorithm-selection decision and runs no significance test.",
        "",
        "## Prespecified definitions",
        "",
        "- Hard attachment count = single plausible parent + unique closest parent "
        "among multiple plausible candidates.",
        "- Primary burden = hard attachment count divided by all biopsy-layer child "
        "decisions; the absolute count is the size-sensitive secondary view.",
        "- Effect = ordered full AD-F1 minus pooled plausible-parsimony AD-F1. "
        "A negative association is the hypothesized direction.",
        "- Non-same-state means different biological state labels; it does not prove "
        "strictly positive CNP distance.",
        "",
        "## Height summaries",
        "",
        "| Height | Cases | Child decisions mean | Hard count mean | Hard fraction mean | Same-state fraction mean | Tie-deferred fraction mean | Copy-up fraction mean | AD-F1 delta mean | AD-F1 delta median | W/T/L | Spearman hard fraction vs delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for height in result["paired_heights"]:
        summary = result["height_summaries"][str(height)]
        counts = summary["count_summaries"]
        fractions = summary["fraction_summaries"]
        effect = summary["ad_f1_effect_ordered_minus_pooled"]
        association = summary["association_with_ad_f1_effect"][
            "primary_fraction_predictor"
        ]
        lines.append(
            "| H{height} | {cases} | {children} | {hard_count} | {hard_fraction} | "
            "{same_state} | {tie_deferred} | {copy_up} | {effect_mean} | "
            "{effect_median} | {wins}/{ties}/{losses} | {spearman} |".format(
                height=height,
                cases=summary["case_count"],
                children=_format(counts["child_decision_count"]["mean"], 3),
                hard_count=_format(
                    counts["non_same_state_hard_attachment_count"]["mean"], 3
                ),
                hard_fraction=_format(
                    fractions["non_same_state_hard_attachment_fraction"]["mean"]
                ),
                same_state=_format(
                    fractions["same_state_attachment_fraction"]["mean"]
                ),
                tie_deferred=_format(fractions["tie_deferred_fraction"]["mean"]),
                copy_up=_format(fractions["copy_up_fraction"]["mean"]),
                effect_mean=_format(effect["mean"]),
                effect_median=_format(effect["median"]),
                wins=effect["ordered_win_count"],
                ties=effect["tie_count"],
                losses=effect["ordered_loss_count"],
                spearman=_format(association["spearman_r"]),
            )
        )

    lines.extend(
        [
            "",
            "## Burden by paired AD-F1 outcome",
            "",
            "| Height | Outcome | Cases | Hard count mean | Hard fraction mean |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for height in result["paired_heights"]:
        strata = result["height_summaries"][str(height)]["burden_by_ad_f1_outcome"]
        for outcome in ("ordered_win", "tie", "ordered_loss"):
            row = strata[outcome]
            count_summary = row["non_same_state_hard_attachment_count"]
            fraction_summary = row["non_same_state_hard_attachment_fraction"]
            lines.append(
                f"| H{height} | {outcome} | {row['case_count']} | "
                f"{_format(count_summary['mean'] if count_summary else None, 3)} | "
                f"{_format(fraction_summary['mean'] if fraction_summary else None)} |"
            )

    lines.extend(
        [
            "",
            "## Paired height transitions",
            "",
            "| Transition | Blocks | Hard-fraction change mean | Increase/tie/decrease | Hard-count change mean | AD-F1-effect change mean | Spearman fraction-change vs effect-change |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for label, transition in result["paired_height_transitions"].items():
        fraction_change = transition[
            "non_same_state_hard_attachment_fraction_change"
        ]
        association = transition[
            "association_of_burden_change_with_ad_f1_effect_change"
        ]["primary_fraction_change"]
        lines.append(
            "| {label} | {blocks} | {fraction_mean} | {positive}/{tie}/{negative} | "
            "{count_mean} | {effect_mean} | {spearman} |".format(
                label=label.replace("_", " "),
                blocks=transition["independent_block_count"],
                fraction_mean=_format(fraction_change["mean"]),
                positive=fraction_change["positive_count"],
                tie=fraction_change["tie_count"],
                negative=fraction_change["negative_count"],
                count_mean=_format(
                    transition["non_same_state_hard_attachment_count_change"]["mean"],
                    3,
                ),
                effect_mean=_format(transition["ad_f1_effect_change"]["mean"]),
                spearman=_format(association["spearman_r"]),
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "Evidence for the proposed mechanism would require a coherent pattern: "
            "hard-attachment burden rises with depth, ordered losses have greater "
            "burden than ordered wins within height, and burden/effect associations "
            "are predominantly negative. A mixed or absent pattern does not justify "
            "implementing the same-state-only candidate. Even a coherent association "
            "is adaptive development evidence rather than a causal result; a new "
            "algorithm still requires a separate owner decision.",
            "",
            "The JSON artifact retains the 300 compact derived condition rows for "
            "auditability. It contains no tree, CNP profile, distance matrix, or truth "
            "record.",
            "",
        ]
    )
    return "\n".join(lines)


def run_full_attachment_audit(
    *,
    ordered_result_root: Path | str,
    pooled_result_root: Path | str,
    output_root: Path | str,
    expected_block_count: int = DEFAULT_BLOCK_COUNT,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    ordered_path = _result_path(ordered_result_root)
    pooled_path = _result_path(pooled_result_root)
    planned_root = Path(output_root).expanduser().resolve()
    for source_path in (ordered_path, pooled_path):
        if _is_within(planned_root, source_path.parent):
            raise ValueError("Audit output must be outside both immutable result roots.")
    result = build_full_attachment_audit(
        ordered_result_root=ordered_path,
        pooled_result_root=pooled_path,
        expected_block_count=expected_block_count,
        created_at_utc=created_at_utc,
    )
    root = ensure_new_output_root(planned_root)
    write_json(root / RESULT_NAME, result)
    (root / REPORT_NAME).write_text(_render_report(result), encoding="utf-8")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ordered-result-root", type=Path, required=True)
    parser.add_argument("--pooled-result-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = run_full_attachment_audit(
        ordered_result_root=arguments.ordered_result_root,
        pooled_result_root=arguments.pooled_result_root,
        output_root=arguments.output_root,
    )
    print(
        f"complete: {result['condition_count']} paired condition records; "
        "no reconstruction reruns"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "FULL_ATTACHMENT_AUDIT_SCHEMA_VERSION",
    "association_summary",
    "build_full_attachment_audit",
    "derive_attachment_measures",
    "run_full_attachment_audit",
]
