"""Verify the frozen-transition correction on preserved development banks.

The report keeps partial and fully labeled reconstruction families separate.
It compares the corrected six-arm theory portfolio with the corresponding
pre-correction records on exactly the same stored conditions.  Historical
records are read only as explicitly superseded provenance; they are never
accepted as current reconstruction evidence.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import math
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

from algorithm_evaluation.paper_pipeline_contract import read_json
from algorithm_evaluation.v5_algorithm_development_common import ARM_SPEC_BY_ID
from algorithm_evaluation.v5_shortlist_robustness_common import (
    ADAPTIVE_A_PRIME_ID,
    ADAPTIVE_B_PRIME_ID,
    BASELINE_SIMULATOR_REGIME,
    CURRENT_PAPER_DEVELOPMENT_ARM_IDS,
    DEFAULT_BLOCK_COUNT,
    INTERMEDIATE_RUN_SCHEMA_VERSION,
    PARTIAL_ADAPTIVE_Y_PRIME_ID,
    PARTIAL_X_ID,
    PLACEMENT_POLICIES,
    POOLED_D_ID,
    POOLED_E_ID,
    PREVIOUS_RUN_SCHEMA_VERSION,
    RESULT_NAME,
    RUN_SCHEMA_VERSION,
    SHORT_LABEL_BY_ARM,
    V2A_SIMULATOR_REGIME,
    V2B_SIMULATOR_REGIME,
    V2C_SIMULATOR_REGIME,
    ensure_new_output_root,
    load_bank_manifest,
    write_json,
)
from algorithm_evaluation.v5_shortlist_robustness_report import (
    build_report as build_within_regime_report,
)
from reconstructor_biopsy_blocks import (
    BIOPSY_GUIDED_AUDIT_SCHEMA_VERSION,
    FROZEN_TRANSITION_PARENT_ELIGIBILITY_POLICY,
)


REPORT_SCHEMA_VERSION = (
    "ctbf-v5-frozen-transition-development-verification-v1"
)
CURRENT_ARM_SET = "current-paper-development"
TIE_TOLERANCE = 1e-12

REGIME_IDS = (
    BASELINE_SIMULATOR_REGIME,
    V2A_SIMULATOR_REGIME,
    V2B_SIMULATOR_REGIME,
    V2C_SIMULATOR_REGIME,
)
REGIME_LABELS = {
    BASELINE_SIMULATOR_REGIME: "v2",
    V2A_SIMULATOR_REGIME: "v2a",
    V2B_SIMULATOR_REGIME: "v2b",
    V2C_SIMULATOR_REGIME: "v2c",
}
AFFECTED_ARM_IDS = (
    ADAPTIVE_A_PRIME_ID,
    ADAPTIVE_B_PRIME_ID,
    PARTIAL_ADAPTIVE_Y_PRIME_ID,
)
UNAFFECTED_COMPARATOR_ARM_IDS = (
    POOLED_E_ID,
    POOLED_D_ID,
    PARTIAL_X_ID,
)


@dataclass(frozen=True)
class ContrastSpec:
    contrast_id: str
    family: str
    left_arm_id: str
    right_arm_id: str
    metrics: tuple[str, ...]
    primary_metric: str
    verification_contrast: bool = True


CONTRAST_SPECS = (
    ContrastSpec(
        "Y_prime_minus_X",
        "partial",
        PARTIAL_ADAPTIVE_Y_PRIME_ID,
        PARTIAL_X_ID,
        ("grf",),
        "grf",
    ),
    ContrastSpec(
        "A_prime_minus_E",
        "fully_labeled",
        ADAPTIVE_A_PRIME_ID,
        POOLED_E_ID,
        ("ad_f1", "grf"),
        "ad_f1",
    ),
    ContrastSpec(
        "A_prime_minus_D",
        "fully_labeled",
        ADAPTIVE_A_PRIME_ID,
        POOLED_D_ID,
        ("ad_f1", "grf"),
        "ad_f1",
    ),
    ContrastSpec(
        "B_prime_minus_E",
        "fully_labeled",
        ADAPTIVE_B_PRIME_ID,
        POOLED_E_ID,
        ("ad_f1", "grf"),
        "ad_f1",
    ),
    ContrastSpec(
        "B_prime_minus_D",
        "fully_labeled",
        ADAPTIVE_B_PRIME_ID,
        POOLED_D_ID,
        ("ad_f1", "grf"),
        "ad_f1",
    ),
    ContrastSpec(
        "A_prime_minus_B_prime",
        "fully_labeled",
        ADAPTIVE_A_PRIME_ID,
        ADAPTIVE_B_PRIME_ID,
        ("ad_f1", "grf"),
        "ad_f1",
        verification_contrast=False,
    ),
)


Coordinate = tuple[int, int, str]
RecordKey = tuple[int, int, str, str]


@dataclass(frozen=True)
class ResultBundle:
    regime_id: str
    result_roots: tuple[Path, ...]
    run_ids: tuple[str, ...]
    bank_root: Path
    bank: Mapping[str, Any]
    records: Mapping[RecordKey, Mapping[str, Any]]
    historical: bool

    @property
    def coordinates(self) -> tuple[Coordinate, ...]:
        return tuple(
            sorted(
                {
                    (key[0], key[1], key[2])
                    for key in self.records
                }
            )
        )


def _metric(record: Mapping[str, Any], metric: str) -> float:
    value = record.get("metrics", {}).get(metric)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Successful result record has invalid {metric}.")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"Successful result record has nonfinite {metric}.")
    return value


def _effect_summary(values: Sequence[float]) -> dict[str, Any]:
    normalized = [float(value) for value in values]
    if not normalized or any(not math.isfinite(value) for value in normalized):
        raise ValueError("Paired effects must be nonempty and finite.")
    ordered = sorted(normalized)
    positive = [value for value in ordered if value > TIE_TOLERANCE]
    negative = [value for value in ordered if value < -TIE_TOLERANCE]
    ties = len(ordered) - len(positive) - len(negative)
    return {
        "count": len(ordered),
        "minimum": ordered[0],
        "median": float(statistics.median(ordered)),
        "mean": float(statistics.fmean(ordered)),
        "maximum": ordered[-1],
        "wins": len(positive),
        "ties": ties,
        "losses": len(negative),
    }


def _result_records(
    result: Mapping[str, Any],
    *,
    required_arm_ids: Sequence[str],
) -> dict[RecordKey, Mapping[str, Any]]:
    required = set(required_arm_ids)
    records = result.get("records")
    if not isinstance(records, list):
        raise ValueError("Development result has no record inventory.")
    selected: dict[RecordKey, Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("Development result contains a malformed record.")
        arm_id = str(record.get("arm_id"))
        if arm_id not in required:
            continue
        if record.get("status") != "success":
            raise ValueError(
                f"Required arm {arm_id} contains a failed result record."
            )
        key = (
            int(record["block_index"]),
            int(record["height"]),
            str(record["placement_policy"]),
            arm_id,
        )
        if key in selected:
            raise ValueError(f"Duplicate development result record {key}.")
        selected[key] = record
    return selected


def _validate_current_audits(
    records: Mapping[RecordKey, Mapping[str, Any]],
) -> None:
    for key, record in records.items():
        if key[3] not in AFFECTED_ARM_IDS:
            continue
        metadata = record.get("reconstruction_metadata")
        audit = (
            metadata.get("biopsy_layer_decision_audit")
            if isinstance(metadata, Mapping)
            else None
        )
        if not isinstance(audit, Mapping):
            raise ValueError(f"Current affected record {key} has no decision audit.")
        if audit.get("schema_version") != BIOPSY_GUIDED_AUDIT_SCHEMA_VERSION:
            raise ValueError(f"Current affected record {key} has an old audit schema.")
        if (
            audit.get("parent_eligibility_policy")
            != FROZEN_TRANSITION_PARENT_ELIGIBILITY_POLICY
        ):
            raise ValueError(
                f"Current affected record {key} has the wrong parent policy."
            )


def _load_current_bundle(
    regime_id: str,
    result_root: Path | str,
) -> ResultBundle:
    root = Path(result_root).expanduser().resolve()
    validated = build_within_regime_report(
        result_root=root,
        expected_block_count=DEFAULT_BLOCK_COUNT,
        created_at_utc="validation-only",
    )
    if (
        int(validated["arm_count"]) != len(CURRENT_PAPER_DEVELOPMENT_ARM_IDS)
        or set(validated["shortlist_arm_ids"])
        != set(CURRENT_PAPER_DEVELOPMENT_ARM_IDS)
    ):
        raise ValueError("Current result does not contain the exact six-arm portfolio.")

    result = read_json(root / RESULT_NAME)
    if result.get("arm_set") != CURRENT_ARM_SET:
        raise ValueError("Current result has the wrong declared arm set.")
    if tuple(result.get("arm_ids", ())) != CURRENT_PAPER_DEVELOPMENT_ARM_IDS:
        raise ValueError("Current result arm order changed.")
    if result.get("failure_count") != 0:
        raise ValueError("Current result must contain zero failures.")
    gates = result.get("semantic_gate_by_arm")
    if not isinstance(gates, Mapping) or set(gates) != set(
        CURRENT_PAPER_DEVELOPMENT_ARM_IDS
    ):
        raise ValueError("Current result semantic-gate inventory is incomplete.")
    if any(
        not isinstance(gate, Mapping) or gate.get("status") != "passed"
        for gate in gates.values()
    ):
        raise ValueError("Current result contains a failed semantic gate.")

    bank_root, bank = load_bank_manifest(
        result["bank_root"],
        expected_block_count=DEFAULT_BLOCK_COUNT,
    )
    observed_regime = bank.get("simulator_regime_id")
    if observed_regime is None and regime_id == BASELINE_SIMULATOR_REGIME:
        observed_regime = BASELINE_SIMULATOR_REGIME
    if observed_regime != regime_id:
        raise ValueError(
            f"Expected current regime {regime_id}, observed {observed_regime}."
        )
    records = _result_records(
        result,
        required_arm_ids=CURRENT_PAPER_DEVELOPMENT_ARM_IDS,
    )
    expected_count = len(bank["cases"]) * len(CURRENT_PAPER_DEVELOPMENT_ARM_IDS)
    if len(records) != expected_count:
        raise ValueError("Current six-arm record inventory is incomplete.")
    _validate_current_audits(records)
    return ResultBundle(
        regime_id=regime_id,
        result_roots=(root,),
        run_ids=(str(result["run_id"]),),
        bank_root=bank_root,
        bank=bank,
        records=records,
        historical=False,
    )


def _load_historical_bundle(
    regime_id: str,
    result_roots: Sequence[Path | str],
    *,
    current: ResultBundle,
    required_arm_ids: Sequence[str] = CURRENT_PAPER_DEVELOPMENT_ARM_IDS,
    current_policy_arm_ids: Sequence[str] = AFFECTED_ARM_IDS,
) -> ResultBundle:
    roots = tuple(Path(root).expanduser().resolve() for root in result_roots)
    if not roots:
        raise ValueError(f"No historical result roots supplied for {regime_id}.")
    allowed_schemas = {
        RUN_SCHEMA_VERSION,
        PREVIOUS_RUN_SCHEMA_VERSION,
        INTERMEDIATE_RUN_SCHEMA_VERSION,
    }
    records: dict[RecordKey, Mapping[str, Any]] = {}
    run_ids = []
    observed_bank_roots = set()
    for root in roots:
        result = read_json(root / RESULT_NAME)
        if result.get("schema_version") not in allowed_schemas:
            raise ValueError("Unknown historical development-result schema.")
        if result.get("status") != "complete":
            raise ValueError("Historical development result is not complete.")
        run_ids.append(str(result["run_id"]))
        observed_bank_roots.add(
            str(Path(result["bank_root"]).expanduser().resolve())
        )
        for key, record in _result_records(
            result,
            required_arm_ids=required_arm_ids,
        ).items():
            if key in records:
                raise ValueError(f"Historical arm record occurs twice: {key}.")
            records[key] = record
    if observed_bank_roots != {str(current.bank_root)}:
        raise ValueError("Historical and current results do not use the same bank.")
    expected_count = len(current.bank["cases"]) * len(required_arm_ids)
    if len(records) != expected_count:
        missing_arms = set(required_arm_ids).difference(
            key[3] for key in records
        )
        raise ValueError(
            "Historical required-arm record inventory is incomplete; "
            "missing arms: "
            f"{sorted(missing_arms)}."
        )

    current_policy_records = []
    current_policy_arm_ids = set(current_policy_arm_ids)
    for key, record in records.items():
        if key[3] not in current_policy_arm_ids:
            continue
        metadata = record.get("reconstruction_metadata")
        audit = (
            metadata.get("biopsy_layer_decision_audit")
            if isinstance(metadata, Mapping)
            else None
        )
        if (
            isinstance(audit, Mapping)
            and audit.get("schema_version")
            == BIOPSY_GUIDED_AUDIT_SCHEMA_VERSION
            and audit.get("parent_eligibility_policy")
            == FROZEN_TRANSITION_PARENT_ELIGIBILITY_POLICY
        ):
            current_policy_records.append(key)
    if current_policy_records:
        raise ValueError(
            "A declared historical root already uses the frozen-transition policy."
        )
    return ResultBundle(
        regime_id=regime_id,
        result_roots=roots,
        run_ids=tuple(run_ids),
        bank_root=current.bank_root,
        bank=current.bank,
        records=records,
        historical=True,
    )


def _metric_values(
    bundle: ResultBundle,
    *,
    arm_id: str,
    metric: str,
) -> dict[Coordinate, float]:
    return {
        (key[0], key[1], key[2]): _metric(record, metric)
        for key, record in bundle.records.items()
        if key[3] == arm_id
    }


def _contrast_values(
    bundle: ResultBundle,
    *,
    spec: ContrastSpec,
    metric: str,
) -> dict[Coordinate, float]:
    left = _metric_values(bundle, arm_id=spec.left_arm_id, metric=metric)
    right = _metric_values(bundle, arm_id=spec.right_arm_id, metric=metric)
    if set(left) != set(right):
        raise ValueError(f"Contrast {spec.contrast_id} is not coordinate paired.")
    return {coordinate: left[coordinate] - right[coordinate] for coordinate in left}


def _height_values(
    values: Mapping[Coordinate, float],
    *,
    height: int,
    placement_policy: str | None = None,
) -> dict[Coordinate, float]:
    return {
        coordinate: value
        for coordinate, value in values.items()
        if coordinate[1] == int(height)
        and (placement_policy is None or coordinate[2] == placement_policy)
    }


def _complete_block_means(
    values: Mapping[Coordinate, float],
    *,
    height: int,
) -> dict[int, float]:
    by_block: dict[int, dict[str, float]] = {}
    for coordinate, value in _height_values(values, height=height).items():
        by_block.setdefault(coordinate[0], {})[coordinate[2]] = float(value)
    required = set(PLACEMENT_POLICIES)
    return {
        block_index: float(statistics.fmean(by_policy.values()))
        for block_index, by_policy in sorted(by_block.items())
        if set(by_policy) == required
    }


def _paired_shift(
    current: Mapping[Any, float],
    historical: Mapping[Any, float],
) -> dict[Any, float]:
    if set(current) != set(historical):
        raise ValueError("Current and historical effects are not exactly paired.")
    return {key: float(current[key]) - float(historical[key]) for key in current}


def _retention_status(historical_mean: float, current_mean: float) -> str:
    historical_positive = historical_mean > TIE_TOLERANCE
    current_positive = current_mean > TIE_TOLERANCE
    if historical_positive and current_positive:
        return "positive_contrast_retained"
    if historical_positive:
        return "historical_positive_contrast_not_retained"
    if current_positive:
        return "positive_only_after_correction"
    return "no_positive_mean_contrast_in_either_version"


def _contrast_rows(
    current: ResultBundle,
    historical: ResultBundle,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    block_rows = []
    placement_rows = []
    for spec in CONTRAST_SPECS:
        for metric in spec.metrics:
            current_values = _contrast_values(current, spec=spec, metric=metric)
            historical_values = _contrast_values(
                historical, spec=spec, metric=metric
            )
            for height in current.bank["heights"]:
                height = int(height)
                current_blocks = _complete_block_means(
                    current_values, height=height
                )
                historical_blocks = _complete_block_means(
                    historical_values, height=height
                )
                shifts = _paired_shift(current_blocks, historical_blocks)
                current_summary = _effect_summary(list(current_blocks.values()))
                historical_summary = _effect_summary(
                    list(historical_blocks.values())
                )
                block_rows.append(
                    {
                        "regime_id": current.regime_id,
                        "regime_label": REGIME_LABELS[current.regime_id],
                        "height": height,
                        "family": spec.family,
                        "contrast_id": spec.contrast_id,
                        "left_arm_id": spec.left_arm_id,
                        "left_label": SHORT_LABEL_BY_ARM[spec.left_arm_id],
                        "right_arm_id": spec.right_arm_id,
                        "right_label": SHORT_LABEL_BY_ARM[spec.right_arm_id],
                        "metric": metric,
                        "primary_metric": metric == spec.primary_metric,
                        "verification_contrast": spec.verification_contrast,
                        "paired_block_count": len(current_blocks),
                        "historical": historical_summary,
                        "current": current_summary,
                        "correction_shift": _effect_summary(list(shifts.values())),
                        "retention_status": _retention_status(
                            historical_summary["mean"], current_summary["mean"]
                        ),
                    }
                )
                for policy in PLACEMENT_POLICIES:
                    current_cell = _height_values(
                        current_values,
                        height=height,
                        placement_policy=policy,
                    )
                    historical_cell = _height_values(
                        historical_values,
                        height=height,
                        placement_policy=policy,
                    )
                    cell_shift = _paired_shift(current_cell, historical_cell)
                    placement_rows.append(
                        {
                            "regime_id": current.regime_id,
                            "regime_label": REGIME_LABELS[current.regime_id],
                            "height": height,
                            "placement_policy": policy,
                            "family": spec.family,
                            "contrast_id": spec.contrast_id,
                            "metric": metric,
                            "paired_condition_count": len(current_cell),
                            "historical": _effect_summary(
                                list(historical_cell.values())
                            ),
                            "current": _effect_summary(list(current_cell.values())),
                            "correction_shift": _effect_summary(
                                list(cell_shift.values())
                            ),
                        }
                    )
    return block_rows, placement_rows


def _score_change_rows(
    current: ResultBundle,
    historical: ResultBundle,
) -> list[dict[str, Any]]:
    rows = []
    for arm_id in AFFECTED_ARM_IDS:
        spec = ARM_SPEC_BY_ID[arm_id]
        metrics = (spec.primary_metric, *spec.complementary_metrics)
        for metric in metrics:
            current_values = _metric_values(current, arm_id=arm_id, metric=metric)
            historical_values = _metric_values(
                historical, arm_id=arm_id, metric=metric
            )
            condition_shifts = _paired_shift(current_values, historical_values)
            for height in current.bank["heights"]:
                height = int(height)
                block_shifts = _complete_block_means(
                    condition_shifts, height=height
                )
                rows.append(
                    {
                        "regime_id": current.regime_id,
                        "regime_label": REGIME_LABELS[current.regime_id],
                        "height": height,
                        "arm_id": arm_id,
                        "short_label": SHORT_LABEL_BY_ARM[arm_id],
                        "family": spec.family,
                        "metric": metric,
                        "primary_metric": metric == spec.primary_metric,
                        "paired_block_count": len(block_shifts),
                        "current_minus_historical": _effect_summary(
                            list(block_shifts.values())
                        ),
                    }
                )
    return rows


def _comparator_rows(
    current: ResultBundle,
    historical: ResultBundle,
) -> list[dict[str, Any]]:
    rows = []
    for arm_id in UNAFFECTED_COMPARATOR_ARM_IDS:
        spec = ARM_SPEC_BY_ID[arm_id]
        metrics = (spec.primary_metric, *spec.complementary_metrics)
        for metric in metrics:
            current_values = _metric_values(current, arm_id=arm_id, metric=metric)
            historical_values = _metric_values(
                historical, arm_id=arm_id, metric=metric
            )
            shifts = _paired_shift(current_values, historical_values)
            for height in current.bank["heights"]:
                height = int(height)
                height_shifts = list(
                    _height_values(shifts, height=height).values()
                )
                maximum_absolute_delta = max(abs(value) for value in height_shifts)
                rows.append(
                    {
                        "regime_id": current.regime_id,
                        "regime_label": REGIME_LABELS[current.regime_id],
                        "height": height,
                        "arm_id": arm_id,
                        "short_label": SHORT_LABEL_BY_ARM[arm_id],
                        "metric": metric,
                        "paired_condition_count": len(height_shifts),
                        "current_minus_historical": _effect_summary(height_shifts),
                        "maximum_absolute_delta": maximum_absolute_delta,
                        "exact_within_tolerance": (
                            maximum_absolute_delta <= TIE_TOLERANCE
                        ),
                    }
                )
    return rows


def build_report(
    *,
    current_result_roots: Mapping[str, Path | str],
    historical_result_roots: Mapping[str, Sequence[Path | str]],
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    if set(current_result_roots) != set(REGIME_IDS):
        raise ValueError("Current result roots must cover exactly v2/v2a/v2b/v2c.")
    if set(historical_result_roots) != set(REGIME_IDS):
        raise ValueError(
            "Historical result roots must cover exactly v2/v2a/v2b/v2c."
        )

    current_bundles = {}
    historical_bundles = {}
    source_inventory = []
    block_contrasts = []
    placement_contrasts = []
    score_changes = []
    comparator_rows = []
    for regime_id in REGIME_IDS:
        current = _load_current_bundle(
            regime_id,
            current_result_roots[regime_id],
        )
        historical = _load_historical_bundle(
            regime_id,
            historical_result_roots[regime_id],
            current=current,
        )
        if set(current.records) != set(historical.records):
            raise ValueError(
                f"Current and historical record keys differ for {regime_id}."
            )
        current_bundles[regime_id] = current
        historical_bundles[regime_id] = historical
        source_inventory.append(
            {
                "regime_id": regime_id,
                "regime_label": REGIME_LABELS[regime_id],
                "bank_root": str(current.bank_root),
                "available_condition_count": len(current.coordinates),
                "current_run_ids": list(current.run_ids),
                "current_result_roots": [
                    str(root) for root in current.result_roots
                ],
                "historical_run_ids": list(historical.run_ids),
                "historical_result_roots": [
                    str(root) for root in historical.result_roots
                ],
            }
        )
        regime_block_rows, regime_placement_rows = _contrast_rows(
            current, historical
        )
        block_contrasts.extend(regime_block_rows)
        placement_contrasts.extend(regime_placement_rows)
        score_changes.extend(_score_change_rows(current, historical))
        comparator_rows.extend(_comparator_rows(current, historical))

    primary_verification_rows = [
        row
        for row in block_contrasts
        if row["verification_contrast"] and row["primary_metric"]
    ]
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "complete",
        "created_at_utc": created_at_utc
        or datetime.now(timezone.utc).isoformat(),
        "scientific_role": (
            "post_correction_development_verification_not_held_out_"
            "paper_confirmation"
        ),
        "current_parent_eligibility_policy": (
            FROZEN_TRANSITION_PARENT_ELIGIBILITY_POLICY
        ),
        "current_decision_audit_schema": BIOPSY_GUIDED_AUDIT_SCHEMA_VERSION,
        "arm_ids": list(CURRENT_PAPER_DEVELOPMENT_ARM_IDS),
        "short_labels": {
            arm_id: SHORT_LABEL_BY_ARM[arm_id]
            for arm_id in CURRENT_PAPER_DEVELOPMENT_ARM_IDS
        },
        "source_inventory": source_inventory,
        "comparison_contract": {
            "independent_unit": "truth_block",
            "block_effect": "mean_of_three_placement_specific_paired_effects",
            "condition_wtl_is_descriptive": True,
            "partial_and_full_families_separate": True,
            "positive_contrast_means_left_arm_better": True,
            "historical_records_are_superseded_provenance_only": True,
            "automatic_winner_declared": False,
            "heights_pooled": False,
            "regimes_pooled": False,
        },
        "block_contrasts": block_contrasts,
        "placement_contrasts": placement_contrasts,
        "affected_arm_score_changes": score_changes,
        "unaffected_comparator_reproducibility": comparator_rows,
        "verification_inventory": {
            "primary_contrast_cell_count": len(primary_verification_rows),
            "historical_positive_contrast_cell_count": sum(
                row["historical"]["mean"] > TIE_TOLERANCE
                for row in primary_verification_rows
            ),
            "positive_contrast_retained_count": sum(
                row["retention_status"] == "positive_contrast_retained"
                for row in primary_verification_rows
            ),
            "historical_positive_not_retained_count": sum(
                row["retention_status"]
                == "historical_positive_contrast_not_retained"
                for row in primary_verification_rows
            ),
            "unaffected_comparator_cell_count": len(comparator_rows),
            "unaffected_comparator_exact_cell_count": sum(
                row["exact_within_tolerance"] for row in comparator_rows
            ),
        },
    }


def _flatten_effect(
    prefix: str,
    value: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        f"{prefix}_{field}": value[field]
        for field in (
            "count",
            "minimum",
            "median",
            "mean",
            "maximum",
            "wins",
            "ties",
            "losses",
        )
    }


def _flatten_rows(
    rows: Sequence[Mapping[str, Any]],
    effect_fields: Sequence[str],
) -> list[dict[str, Any]]:
    flattened = []
    for row in rows:
        flat = {
            key: value
            for key, value in row.items()
            if key not in effect_fields
        }
        for field in effect_fields:
            flat.update(_flatten_effect(field, row[field]))
        flattened.append(flat)
    return flattened


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table {path.name}.")
    fieldnames = list(rows[0])
    if any(list(row) != fieldnames for row in rows):
        raise ValueError(f"CSV row shape changed for {path.name}.")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _format_effect(effect: Mapping[str, Any]) -> str:
    return (
        f"{float(effect['mean']):+.4f} "
        f"({effect['wins']}/{effect['ties']}/{effect['losses']})"
    )


def _markdown(report: Mapping[str, Any]) -> str:
    inventory = report["verification_inventory"]
    lines = [
        "# Frozen-transition development verification",
        "",
        f"Schema: `{report['schema_version']}`  ",
        f"Decision audit: `{report['current_decision_audit_schema']}`  ",
        f"Parent policy: `{report['current_parent_eligibility_policy']}`",
        "",
        "This is paired development evidence on preserved banks, not held-out "
        "paper confirmation. Positive contrasts mean that the left method has "
        "the larger score. W/T/L is computed from one mean three-placement "
        "effect per complete truth block.",
        "",
        "## Source inventory",
        "",
        "| Regime | Available conditions | Current run | Historical runs |",
        "|---|---:|---|---|",
    ]
    for row in report["source_inventory"]:
        lines.append(
            f"| {row['regime_label']} | {row['available_condition_count']} | "
            f"{', '.join(row['current_run_ids'])} | "
            f"{', '.join(row['historical_run_ids'])} |"
        )
    lines.extend(
        [
            "",
            "## Primary paired contrasts",
            "",
            "| Regime | Height | Contrast | Metric | Historical | Corrected | "
            "Correction shift | Status |",
            "|---|---:|---|---|---:|---:|---:|---|",
        ]
    )
    for row in report["block_contrasts"]:
        if not row["primary_metric"]:
            continue
        lines.append(
            f"| {row['regime_label']} | H{row['height']} | "
            f"{row['left_label']}−{row['right_label']} | {row['metric']} | "
            f"{_format_effect(row['historical'])} | "
            f"{_format_effect(row['current'])} | "
            f"{_format_effect(row['correction_shift'])} | "
            f"{row['retention_status']} |"
        )
    lines.extend(
        [
            "",
            "## Affected-arm score changes",
            "",
            "Positive values mean that the corrected reconstruction scored "
            "higher than its pre-correction counterpart on the same block.",
            "",
            "| Regime | Height | Arm | Metric | Corrected−historical |",
            "|---|---:|---|---|---:|",
        ]
    )
    for row in report["affected_arm_score_changes"]:
        if row["primary_metric"]:
            lines.append(
                f"| {row['regime_label']} | H{row['height']} | "
                f"{row['short_label']} | {row['metric']} | "
                f"{_format_effect(row['current_minus_historical'])} |"
            )
    lines.extend(
        [
            "",
            "## Unaffected-comparator reproducibility",
            "",
            "| Regime | Height | Arm | Metric | Max absolute delta | Exact |",
            "|---|---:|---|---|---:|---|",
        ]
    )
    for row in report["unaffected_comparator_reproducibility"]:
        if row["metric"] != ARM_SPEC_BY_ID[row["arm_id"]].primary_metric:
            continue
        lines.append(
            f"| {row['regime_label']} | H{row['height']} | "
            f"{row['short_label']} | {row['metric']} | "
            f"{row['maximum_absolute_delta']:.3e} | "
            f"{'yes' if row['exact_within_tolerance'] else 'NO'} |"
        )
    lines.extend(
        [
            "",
            "## Verification inventory",
            "",
            f"- Primary contrast cells: "
            f"{inventory['primary_contrast_cell_count']}.",
            f"- Historical positive contrasts retained: "
            f"{inventory['positive_contrast_retained_count']}/"
            f"{inventory['historical_positive_contrast_cell_count']}.",
            f"- Historical positive contrasts not retained: "
            f"{inventory['historical_positive_not_retained_count']}.",
            f"- Unaffected comparator cells exact within tolerance: "
            f"{inventory['unaffected_comparator_exact_cell_count']}/"
            f"{inventory['unaffected_comparator_cell_count']}.",
            "",
            "Placement-specific contrasts, complementary full-tree GRF, and "
            "complete numeric summaries are stored in the adjacent CSV and "
            "JSON files. No automatic portfolio decision is made.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(
    *,
    current_result_roots: Mapping[str, Path | str],
    historical_result_roots: Mapping[str, Sequence[Path | str]],
    output_root: Path | str,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    report = build_report(
        current_result_roots=current_result_roots,
        historical_result_roots=historical_result_roots,
        created_at_utc=created_at_utc,
    )
    root = ensure_new_output_root(output_root)
    write_json(root / "summary.json", report)
    _write_csv(
        root / "block_contrasts.csv",
        _flatten_rows(
            report["block_contrasts"],
            ("historical", "current", "correction_shift"),
        ),
    )
    _write_csv(
        root / "placement_contrasts.csv",
        _flatten_rows(
            report["placement_contrasts"],
            ("historical", "current", "correction_shift"),
        ),
    )
    _write_csv(
        root / "affected_arm_score_changes.csv",
        _flatten_rows(
            report["affected_arm_score_changes"],
            ("current_minus_historical",),
        ),
    )
    _write_csv(
        root / "unaffected_comparator_reproducibility.csv",
        _flatten_rows(
            report["unaffected_comparator_reproducibility"],
            ("current_minus_historical",),
        ),
    )
    (root / "report.md").write_text(_markdown(report), encoding="utf-8")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for label in ("v2", "v2a", "v2b", "v2c"):
        parser.add_argument(
            f"--{label}-current-result-root",
            type=Path,
            required=True,
        )
        parser.add_argument(
            f"--{label}-historical-result-root",
            type=Path,
            action="append",
            required=True,
        )
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    current = {
        regime_id: getattr(
            arguments,
            f"{REGIME_LABELS[regime_id]}_current_result_root",
        )
        for regime_id in REGIME_IDS
    }
    historical = {
        regime_id: getattr(
            arguments,
            f"{REGIME_LABELS[regime_id]}_historical_result_root",
        )
        for regime_id in REGIME_IDS
    }
    report = write_report(
        current_result_roots=current,
        historical_result_roots=historical,
        output_root=arguments.output_root,
    )
    inventory = report["verification_inventory"]
    print(
        "complete: "
        f"{inventory['primary_contrast_cell_count']} primary contrast cells; "
        f"{inventory['unaffected_comparator_exact_cell_count']}/"
        f"{inventory['unaffected_comparator_cell_count']} comparator cells exact"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
