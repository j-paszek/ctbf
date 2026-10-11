"""Build the compact block-paired CTBF v5 v2/v2a/v2b/v2c report.

The three sensitivity regimes are compared directly with v2 on matching
truth-block, height, and placement coordinates.  Placement conditions remain
descriptive correlated observations; the independent effect is the mean of
the three placement deltas within each complete truth block.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
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
    DECLARED_METRICS,
    DEFAULT_BANK_ID,
    DEFAULT_BLOCK_COUNT,
    FULL_DEVELOPMENT_ARM_IDS,
    ORDERED_A_ID,
    ORDERED_B_ID,
    PARTIAL_ADAPTIVE_U_PRIME_ID,
    PARTIAL_ADAPTIVE_V_PRIME_ID,
    PARTIAL_ADAPTIVE_Y_PRIME_ID,
    PARTIAL_ADAPTIVE_Z_PRIME_ID,
    PARTIAL_DECLARED_METRICS,
    PARTIAL_DEVELOPMENT_ARM_IDS,
    PARTIAL_V2_ARM_IDS,
    PLACEMENT_POLICIES,
    POOLED_D_ID,
    POOLED_E_ID,
    SELECTED_V2_ARM_IDS,
    SHORT_LABEL_BY_ARM,
    V2A_SIMULATOR_REGIME,
    V2B_SIMULATOR_REGIME,
    V2C_SIMULATOR_REGIME,
    ensure_new_output_root,
    load_bank_manifest,
    write_json,
)
from reconstructor_biopsy_blocks import (
    ADAPTIVE_BIOPSY_GUIDED_AUDIT_SCHEMA_VERSION,
    FROZEN_TRANSITION_PARENT_ELIGIBILITY_POLICY,
)
from algorithm_evaluation.v5_shortlist_robustness_report import (
    build_report as build_within_regime_report,
)


REPORT_SCHEMA_VERSION = "ctbf-v5-shortlist-regime-comparison-v1"
TIE_TOLERANCE = 1e-12
WORST_COUNT = 5
SENSITIVITY_REGIMES = (
    V2A_SIMULATOR_REGIME,
    V2B_SIMULATOR_REGIME,
    V2C_SIMULATOR_REGIME,
)
REGIME_SHORT_LABEL = {
    BASELINE_SIMULATOR_REGIME: "v2",
    V2A_SIMULATOR_REGIME: "v2a",
    V2B_SIMULATOR_REGIME: "v2b",
    V2C_SIMULATOR_REGIME: "v2c",
}

FULL_CONTRASTS = (
    ("A_prime_minus_B_prime", ADAPTIVE_A_PRIME_ID, ADAPTIVE_B_PRIME_ID),
    ("A_prime_minus_D", ADAPTIVE_A_PRIME_ID, POOLED_D_ID),
    ("A_prime_minus_A", ADAPTIVE_A_PRIME_ID, ORDERED_A_ID),
    ("B_prime_minus_B", ADAPTIVE_B_PRIME_ID, ORDERED_B_ID),
)
PARTIAL_CONTRASTS = (
    (
        "Y_prime_minus_Z_prime",
        PARTIAL_ADAPTIVE_Y_PRIME_ID,
        PARTIAL_ADAPTIVE_Z_PRIME_ID,
    ),
    (
        "Y_prime_minus_Y",
        PARTIAL_ADAPTIVE_Y_PRIME_ID,
        PARTIAL_V2_ARM_IDS[1],
    ),
    (
        "Y_prime_minus_V_prime",
        PARTIAL_ADAPTIVE_Y_PRIME_ID,
        PARTIAL_ADAPTIVE_V_PRIME_ID,
    ),
    (
        "Y_prime_minus_X",
        PARTIAL_ADAPTIVE_Y_PRIME_ID,
        PARTIAL_V2_ARM_IDS[0],
    ),
    (
        "Z_prime_minus_U_prime",
        PARTIAL_ADAPTIVE_Z_PRIME_ID,
        PARTIAL_ADAPTIVE_U_PRIME_ID,
    ),
)


@dataclass(frozen=True)
class DiagnosticSpec:
    name: str
    label: str
    path: tuple[str, ...]
    boolean: bool = False


DIAGNOSTIC_SPECS = (
    DiagnosticSpec(
        "selected_occurrence_count",
        "selected occurrences",
        ("selected_occurrence_count",),
    ),
    DiagnosticSpec(
        "selected_unique_state_count",
        "selected unique states",
        ("selected_unique_state_count",),
    ),
    DiagnosticSpec(
        "mean_pairwise_cnp2cnp",
        "mean pairwise cnp2cnp",
        ("distance_summary", "off_diagonal", "mean"),
    ),
    DiagnosticSpec(
        "incomparable_pair_fraction",
        "incomparable-pair fraction",
        ("truth_sampling_diagnostics", "incomparable_pair_fraction"),
    ),
    DiagnosticSpec(
        "sampled_ancestor_coverage_fraction",
        "sampled-ancestor coverage",
        (
            "truth_sampling_diagnostics",
            "sampled_ancestor_coverage_fraction",
        ),
    ),
    DiagnosticSpec(
        "normalized_minimum_invented_edge_fraction",
        "minimum invented-edge fraction",
        (
            "truth_sampling_diagnostics",
            "normalized_minimum_invented_edge_fraction",
        ),
    ),
    DiagnosticSpec(
        "unique_state_label_coverage_fraction",
        "unique-state label coverage",
        (
            "truth_sampling_diagnostics",
            "unique_state_label_coverage_fraction",
        ),
    ),
    DiagnosticSpec(
        "observed_only_representable_fraction",
        "observed-only representable fraction",
        (
            "truth_sampling_diagnostics",
            "observed_only_occurrence_arborescence_representable",
        ),
        boolean=True,
    ),
    DiagnosticSpec(
        "effective_segmental_event_applications",
        "effective segmental events",
        ("simulator_event_totals", "effective_segmental_event_applications"),
    ),
    DiagnosticSpec(
        "retained_segmental_gain_events",
        "retained segmental gains",
        ("simulator_event_totals", "retained_segmental_gain_events"),
    ),
    DiagnosticSpec(
        "retained_segmental_loss_events",
        "retained segmental losses",
        ("simulator_event_totals", "retained_segmental_loss_events"),
    ),
    DiagnosticSpec(
        "viability_rejections",
        "viability rejections",
        ("simulator_event_totals", "viability_rejections"),
    ),
)

ADAPTIVE_FEATURE_LABELS = {
    "mean_effective_radius": "mean effective radius",
    "radius_gt1_transition_fraction": "transition fraction with radius > 1",
    "copy_up_fraction": "copy-up decision fraction",
    "frozen_candidate_coverage_fraction": "frozen candidate coverage",
}


Coordinate = tuple[int, int, str]
RecordKey = tuple[int, int, str, str]


@dataclass
class RegimeBundle:
    regime_id: str
    result_roots: tuple[Path, ...]
    run_ids: tuple[str, ...]
    bank_root: Path
    bank: dict[str, Any]
    records: dict[RecordKey, dict[str, Any]]
    cases: dict[Coordinate, Mapping[str, Any]]
    metadata_cache: dict[Coordinate, Mapping[str, Any]] = field(
        default_factory=dict
    )

    def metadata(self, coordinate: Coordinate) -> Mapping[str, Any]:
        cached = self.metadata_cache.get(coordinate)
        if cached is not None:
            return cached
        try:
            case = self.cases[coordinate]
        except KeyError as error:
            raise ValueError(
                f"{self.regime_id} has no available case at {coordinate}."
            ) from error
        path = self.bank_root / str(case["metadata_path"])
        value = read_json(path)
        expected = {
            "block_index": coordinate[0],
            "height": coordinate[1],
            "placement_policy": coordinate[2],
        }
        for name, target in expected.items():
            if value.get(name) != target:
                raise ValueError(
                    f"Case metadata coordinate changed at {coordinate}: {name}."
                )
        self.metadata_cache[coordinate] = value
        return value


def _numeric_summary(values: Sequence[float]) -> dict[str, Any] | None:
    if not values:
        return None
    normalized = [float(value) for value in values]
    if any(not math.isfinite(value) for value in normalized):
        raise ValueError("Summary values must be finite.")
    ordered = sorted(normalized)
    return {
        "count": len(ordered),
        "minimum": ordered[0],
        "median": float(statistics.median(ordered)),
        "mean": float(statistics.fmean(ordered)),
        "maximum": ordered[-1],
    }


def _effect_summary(values: Sequence[float]) -> dict[str, Any] | None:
    summary = _numeric_summary(values)
    if summary is None:
        return None
    normalized = [float(value) for value in values]
    positive = [value for value in normalized if value > TIE_TOLERANCE]
    negative = [value for value in normalized if value < -TIE_TOLERANCE]
    ties = len(normalized) - len(positive) - len(negative)
    ordered = sorted(normalized)
    return {
        **summary,
        "wins": len(positive),
        "ties": ties,
        "losses": len(negative),
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


def _summarize_paired_values(
    baseline_values: Mapping[Coordinate, float],
    sensitivity_values: Mapping[Coordinate, float],
    *,
    blocks: Sequence[int],
    height: int,
    policies: Sequence[str] = PLACEMENT_POLICIES,
    require_complete: bool,
) -> dict[str, Any]:
    paired_baseline: list[float] = []
    paired_sensitivity: list[float] = []
    condition_deltas: list[float] = []
    block_deltas: list[float] = []
    incomplete_blocks: list[int] = []
    for block_index in blocks:
        current_deltas = []
        for policy in policies:
            coordinate = (int(block_index), int(height), str(policy))
            baseline = baseline_values.get(coordinate)
            sensitivity = sensitivity_values.get(coordinate)
            if baseline is None or sensitivity is None:
                continue
            baseline = float(baseline)
            sensitivity = float(sensitivity)
            if not math.isfinite(baseline) or not math.isfinite(sensitivity):
                raise ValueError("Paired values must be finite.")
            paired_baseline.append(baseline)
            paired_sensitivity.append(sensitivity)
            delta = sensitivity - baseline
            condition_deltas.append(delta)
            current_deltas.append(delta)
        if len(current_deltas) == len(policies):
            block_deltas.append(float(statistics.fmean(current_deltas)))
        else:
            incomplete_blocks.append(int(block_index))
    if require_complete and incomplete_blocks:
        raise ValueError(
            "A declared complete paired effect has missing coordinates: "
            f"{incomplete_blocks}."
        )
    return {
        "paired_condition_count": len(condition_deltas),
        "paired_block_count": len(block_deltas),
        "incomplete_block_indices": incomplete_blocks,
        "baseline": _numeric_summary(paired_baseline),
        "sensitivity": _numeric_summary(paired_sensitivity),
        "condition_delta": _effect_summary(condition_deltas),
        "block_delta": _effect_summary(block_deltas),
    }


def _complete_blocks_from_inventory(
    inventory: Sequence[Mapping[str, Any]],
    *,
    height: int,
    policies: Sequence[str] = PLACEMENT_POLICIES,
) -> tuple[int, ...]:
    available_by_block: dict[int, set[str]] = {}
    for row in inventory:
        if int(row["height"]) != int(height) or row.get("status") != "available":
            continue
        available_by_block.setdefault(int(row["block_index"]), set()).add(
            str(row["placement_policy"])
        )
    target = set(policies)
    return tuple(
        block_index
        for block_index, available in sorted(available_by_block.items())
        if available == target
    )


def _adaptive_features(record: Mapping[str, Any]) -> dict[str, float]:
    metadata = record.get("reconstruction_metadata")
    audit = (
        metadata.get("biopsy_layer_decision_audit")
        if isinstance(metadata, Mapping)
        else None
    )
    if (
        not isinstance(audit, Mapping)
        or audit.get("schema_version")
        != ADAPTIVE_BIOPSY_GUIDED_AUDIT_SCHEMA_VERSION
        or audit.get("parent_eligibility_policy")
        != FROZEN_TRANSITION_PARENT_ELIGIBILITY_POLICY
    ):
        raise ValueError("A-prime adaptive decision-audit contract changed.")
    transitions = audit.get("transition_records") if isinstance(audit, Mapping) else None
    if not isinstance(transitions, list) or not transitions:
        raise ValueError("A-prime adaptive transition diagnostics are missing.")
    radii = [float(row["effective_radius"]) for row in transitions]
    counters = audit.get("child_decision_count"), audit.get("copy_up_count")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in counters):
        raise ValueError("A-prime adaptive decision counters are invalid.")
    child_count = float(counters[0])
    if child_count <= 0:
        raise ValueError("A-prime adaptive decision count must be positive.")
    frozen_children = sum(int(row["child_snapshot_count"]) for row in transitions)
    frozen_covered = sum(
        int(row["frozen_radius_candidate_covered_child_count"])
        for row in transitions
    )
    if frozen_children <= 0:
        raise ValueError("A-prime adaptive frozen child inventory is empty.")
    return {
        "mean_effective_radius": float(statistics.fmean(radii)),
        "radius_gt1_transition_fraction": (
            sum(radius > 1 for radius in radii) / len(radii)
        ),
        "copy_up_fraction": float(counters[1]) / child_count,
        "frozen_candidate_coverage_fraction": frozen_covered / frozen_children,
    }


def _load_bundle(
    regime_id: str,
    result_roots: Sequence[Path | str],
) -> RegimeBundle:
    normalized_roots = tuple(
        Path(root).expanduser().resolve() for root in result_roots
    )
    if not normalized_roots:
        raise ValueError(f"No result roots were provided for {regime_id}.")

    validated = build_within_regime_report(
        result_roots=normalized_roots,
        expected_block_count=DEFAULT_BLOCK_COUNT,
        created_at_utc="validation-only",
    )
    if tuple(validated["shortlist_arm_ids"]) != SELECTED_V2_ARM_IDS:
        raise ValueError(f"{regime_id} does not contain the exact 21-arm roster.")
    if validated["arm_count"] != len(SELECTED_V2_ARM_IDS):
        raise ValueError(f"{regime_id} arm count changed.")
    gates = validated.get("semantic_gate_by_arm")
    if not isinstance(gates, Mapping) or set(gates) != set(SELECTED_V2_ARM_IDS):
        raise ValueError(f"{regime_id} semantic-gate inventory is incomplete.")
    failed_gates = [
        arm_id
        for arm_id, gate in gates.items()
        if not isinstance(gate, Mapping) or gate.get("status") != "passed"
    ]
    if failed_gates:
        raise ValueError(f"{regime_id} has failed semantic gates: {failed_gates}.")

    bank_root, bank = load_bank_manifest(
        validated["bank_root"], expected_block_count=DEFAULT_BLOCK_COUNT
    )
    observed_regime = bank.get("simulator_regime_id")
    if observed_regime is None and bank.get("bank_id") == DEFAULT_BANK_ID:
        observed_regime = BASELINE_SIMULATOR_REGIME
    if observed_regime != regime_id:
        raise ValueError(
            f"Expected simulator regime {regime_id}, observed "
            f"{observed_regime}."
        )

    records: dict[RecordKey, dict[str, Any]] = {}
    run_ids = []
    for root in normalized_roots:
        result = read_json(root / "result.json")
        run_ids.append(str(result["run_id"]))
        for record in result["records"]:
            if record.get("status") != "success":
                raise ValueError(
                    f"The frozen {regime_id} report requires zero run failures."
                )
            arm_id = str(record["arm_id"])
            key = (
                int(record["block_index"]),
                int(record["height"]),
                str(record["placement_policy"]),
                arm_id,
            )
            if key in records:
                raise ValueError(f"Duplicate cross-regime record {key}.")
            spec = ARM_SPEC_BY_ID[arm_id]
            metrics = {}
            for metric in (spec.primary_metric, *spec.complementary_metrics):
                value = record.get("metrics", {}).get(metric)
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(f"Invalid {metric} in {regime_id} at {key}.")
                value = float(value)
                if not math.isfinite(value):
                    raise ValueError(f"Nonfinite {metric} in {regime_id} at {key}.")
                metrics[metric] = value
            records[key] = {
                "metrics": metrics,
                "adaptive_features": (
                    _adaptive_features(record)
                    if arm_id == ADAPTIVE_A_PRIME_ID
                    else None
                ),
            }

    expected_record_count = len(bank["cases"]) * len(SELECTED_V2_ARM_IDS)
    if len(records) != expected_record_count:
        raise ValueError(
            f"{regime_id} compact record count changed: expected "
            f"{expected_record_count}, observed {len(records)}."
        )
    cases = {
        (
            int(case["block_index"]),
            int(case["height"]),
            str(case["placement_policy"]),
        ): case
        for case in bank["cases"]
    }
    if len(cases) != len(bank["cases"]):
        raise ValueError(f"{regime_id} has duplicate available case coordinates.")
    return RegimeBundle(
        regime_id=regime_id,
        result_roots=normalized_roots,
        run_ids=tuple(run_ids),
        bank_root=bank_root,
        bank=bank,
        records=records,
        cases=cases,
    )


def load_regime_bundle(
    regime_id: str,
    result_roots: Sequence[Path | str],
) -> RegimeBundle:
    """Load one fully validated exact-21-arm shortlist regime bundle."""

    return _load_bundle(regime_id, result_roots)


def _paired_blocks(
    baseline: RegimeBundle,
    sensitivity: RegimeBundle,
    height: int,
) -> tuple[int, ...]:
    baseline_blocks = set(
        _complete_blocks_from_inventory(
            baseline.bank["condition_inventory"], height=height
        )
    )
    sensitivity_blocks = set(
        _complete_blocks_from_inventory(
            sensitivity.bank["condition_inventory"], height=height
        )
    )
    return tuple(sorted(baseline_blocks.intersection(sensitivity_blocks)))


def _validate_pairing_contract(
    baseline: RegimeBundle,
    sensitivity: RegimeBundle,
    *,
    height: int,
    blocks: Sequence[int],
) -> None:
    if sensitivity.bank.get("paired_seed_reference_bank_id") != DEFAULT_BANK_ID:
        raise ValueError("Sensitivity paired-seed reference changed.")
    for field_name in ("seed_namespace", "base_seed", "block_count"):
        if baseline.bank.get(field_name) != sensitivity.bank.get(field_name):
            raise ValueError(
                f"Paired bank field {field_name} differs for "
                f"{sensitivity.regime_id}."
            )
    for block_index in blocks:
        for policy in PLACEMENT_POLICIES:
            coordinate = (int(block_index), int(height), str(policy))
            baseline_metadata = baseline.metadata(coordinate)
            sensitivity_metadata = sensitivity.metadata(coordinate)
            for field_name in (
                "simulation_seed",
                "reconstruction_seed",
                "generations",
                "placement_schedule_seed",
            ):
                if baseline_metadata.get(field_name) != sensitivity_metadata.get(
                    field_name
                ):
                    raise ValueError(
                        f"Coordinate-pairing field {field_name} changed at "
                        f"{sensitivity.regime_id} {coordinate}."
                    )


def _unavailable_inventory(
    bundle: RegimeBundle,
    *,
    height: int,
) -> tuple[list[int], dict[str, int], int]:
    rows = [
        row
        for row in bundle.bank["condition_inventory"]
        if int(row["height"]) == int(height) and row.get("status") != "available"
    ]
    failure_types: dict[str, int] = {}
    for row in rows:
        failure = row.get("failure")
        failure_type = (
            str(failure.get("type", "unknown_failure"))
            if isinstance(failure, Mapping)
            else "unknown_failure"
        )
        failure_types[failure_type] = failure_types.get(failure_type, 0) + 1
    return (
        sorted({int(row["block_index"]) for row in rows}),
        failure_types,
        len(rows),
    )


def _metric_values(
    bundle: RegimeBundle,
    *,
    arm_id: str,
    metric: str,
    blocks: Sequence[int],
    height: int,
    policies: Sequence[str] = PLACEMENT_POLICIES,
) -> dict[Coordinate, float]:
    values = {}
    for block_index in blocks:
        for policy in policies:
            coordinate = (int(block_index), int(height), str(policy))
            values[coordinate] = float(
                bundle.records[(*coordinate, arm_id)]["metrics"][metric]
            )
    return values


def _paired_metric_rows(
    baseline: RegimeBundle,
    sensitivity: RegimeBundle,
    *,
    height: int,
    blocks: Sequence[int],
    by_placement: bool,
) -> list[dict[str, Any]]:
    rows = []
    group_specs = (
        ("fully_labeled", FULL_DEVELOPMENT_ARM_IDS, DECLARED_METRICS),
        ("partial", PARTIAL_DEVELOPMENT_ARM_IDS, PARTIAL_DECLARED_METRICS),
    )
    policy_sets = (
        tuple((policy,) for policy in PLACEMENT_POLICIES)
        if by_placement
        else (PLACEMENT_POLICIES,)
    )
    for family, arm_ids, metrics in group_specs:
        for arm_id in arm_ids:
            for metric in metrics:
                for policies in policy_sets:
                    baseline_values = _metric_values(
                        baseline,
                        arm_id=arm_id,
                        metric=metric,
                        blocks=blocks,
                        height=height,
                        policies=policies,
                    )
                    sensitivity_values = _metric_values(
                        sensitivity,
                        arm_id=arm_id,
                        metric=metric,
                        blocks=blocks,
                        height=height,
                        policies=policies,
                    )
                    rows.append(
                        {
                            "sensitivity_regime": sensitivity.regime_id,
                            "sensitivity_label": REGIME_SHORT_LABEL[
                                sensitivity.regime_id
                            ],
                            "height": int(height),
                            "placement_policy": (
                                policies[0] if by_placement else "all_three"
                            ),
                            "family": family,
                            "arm_id": arm_id,
                            "short_label": SHORT_LABEL_BY_ARM[arm_id],
                            "metric": metric,
                            "paired": _summarize_paired_values(
                                baseline_values,
                                sensitivity_values,
                                blocks=blocks,
                                height=height,
                                policies=policies,
                                require_complete=True,
                            ),
                        }
                    )
    return rows


def _contrast_rows(
    baseline: RegimeBundle,
    sensitivity: RegimeBundle,
    *,
    height: int,
    blocks: Sequence[int],
) -> list[dict[str, Any]]:
    rows = []
    groups = (
        ("fully_labeled", "ad_f1", FULL_CONTRASTS),
        ("partial", "grf", PARTIAL_CONTRASTS),
    )
    for family, metric, contrasts in groups:
        for contrast_id, left_arm, right_arm in contrasts:
            baseline_left = _metric_values(
                baseline,
                arm_id=left_arm,
                metric=metric,
                blocks=blocks,
                height=height,
            )
            baseline_right = _metric_values(
                baseline,
                arm_id=right_arm,
                metric=metric,
                blocks=blocks,
                height=height,
            )
            sensitivity_left = _metric_values(
                sensitivity,
                arm_id=left_arm,
                metric=metric,
                blocks=blocks,
                height=height,
            )
            sensitivity_right = _metric_values(
                sensitivity,
                arm_id=right_arm,
                metric=metric,
                blocks=blocks,
                height=height,
            )
            baseline_contrast = {
                coordinate: baseline_left[coordinate] - baseline_right[coordinate]
                for coordinate in baseline_left
            }
            sensitivity_contrast = {
                coordinate: sensitivity_left[coordinate]
                - sensitivity_right[coordinate]
                for coordinate in sensitivity_left
            }
            rows.append(
                {
                    "sensitivity_regime": sensitivity.regime_id,
                    "sensitivity_label": REGIME_SHORT_LABEL[sensitivity.regime_id],
                    "height": int(height),
                    "family": family,
                    "metric": metric,
                    "contrast_id": contrast_id,
                    "left_arm_id": left_arm,
                    "left_label": SHORT_LABEL_BY_ARM[left_arm],
                    "right_arm_id": right_arm,
                    "right_label": SHORT_LABEL_BY_ARM[right_arm],
                    "contrast_direction": "left_minus_right",
                    "paired": _summarize_paired_values(
                        baseline_contrast,
                        sensitivity_contrast,
                        blocks=blocks,
                        height=height,
                        require_complete=True,
                    ),
                }
            )
    return rows


def _path_value(
    metadata: Mapping[str, Any],
    spec: DiagnosticSpec,
) -> float | None:
    value: Any = metadata
    for key in spec.path:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    if value is None:
        return None
    if spec.boolean:
        if not isinstance(value, bool):
            raise ValueError(f"Diagnostic {spec.name} must be boolean.")
        return float(value)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Diagnostic {spec.name} must be numeric.")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"Diagnostic {spec.name} must be finite.")
    return value


def _diagnostic_rows(
    baseline: RegimeBundle,
    sensitivity: RegimeBundle,
    *,
    height: int,
    blocks: Sequence[int],
) -> list[dict[str, Any]]:
    rows = []
    coordinates = [
        (int(block_index), int(height), str(policy))
        for block_index in blocks
        for policy in PLACEMENT_POLICIES
    ]
    for spec in DIAGNOSTIC_SPECS:
        baseline_values = {}
        sensitivity_values = {}
        for coordinate in coordinates:
            baseline_value = _path_value(baseline.metadata(coordinate), spec)
            sensitivity_value = _path_value(sensitivity.metadata(coordinate), spec)
            if baseline_value is not None:
                baseline_values[coordinate] = baseline_value
            if sensitivity_value is not None:
                sensitivity_values[coordinate] = sensitivity_value
        rows.append(
            {
                "sensitivity_regime": sensitivity.regime_id,
                "sensitivity_label": REGIME_SHORT_LABEL[sensitivity.regime_id],
                "height": int(height),
                "diagnostic": spec.name,
                "diagnostic_label": spec.label,
                "paired": _summarize_paired_values(
                    baseline_values,
                    sensitivity_values,
                    blocks=blocks,
                    height=height,
                    require_complete=False,
                ),
            }
        )
    return rows


def _adaptive_rows(
    baseline: RegimeBundle,
    sensitivity: RegimeBundle,
    *,
    height: int,
    blocks: Sequence[int],
) -> list[dict[str, Any]]:
    coordinates = [
        (int(block_index), int(height), str(policy))
        for block_index in blocks
        for policy in PLACEMENT_POLICIES
    ]
    rows = []
    for feature, label in ADAPTIVE_FEATURE_LABELS.items():
        baseline_values = {
            coordinate: float(
                baseline.records[(*coordinate, ADAPTIVE_A_PRIME_ID)][
                    "adaptive_features"
                ][feature]
            )
            for coordinate in coordinates
        }
        sensitivity_values = {
            coordinate: float(
                sensitivity.records[(*coordinate, ADAPTIVE_A_PRIME_ID)][
                    "adaptive_features"
                ][feature]
            )
            for coordinate in coordinates
        }
        rows.append(
            {
                "sensitivity_regime": sensitivity.regime_id,
                "sensitivity_label": REGIME_SHORT_LABEL[sensitivity.regime_id],
                "height": int(height),
                "arm_id": ADAPTIVE_A_PRIME_ID,
                "short_label": SHORT_LABEL_BY_ARM[ADAPTIVE_A_PRIME_ID],
                "feature": feature,
                "feature_label": label,
                "paired": _summarize_paired_values(
                    baseline_values,
                    sensitivity_values,
                    blocks=blocks,
                    height=height,
                    require_complete=True,
                ),
            }
        )
    return rows


def build_report(
    *,
    v2_result_roots: Sequence[Path | str],
    v2a_result_roots: Sequence[Path | str],
    v2b_result_roots: Sequence[Path | str],
    v2c_result_roots: Sequence[Path | str],
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    baseline = load_regime_bundle(BASELINE_SIMULATOR_REGIME, v2_result_roots)
    sensitivities = (
        load_regime_bundle(V2A_SIMULATOR_REGIME, v2a_result_roots),
        load_regime_bundle(V2B_SIMULATOR_REGIME, v2b_result_roots),
        load_regime_bundle(V2C_SIMULATOR_REGIME, v2c_result_roots),
    )

    pairing_inventory = []
    metric_rows = []
    metric_by_placement_rows = []
    contrast_rows = []
    diagnostic_rows = []
    adaptive_rows = []
    for sensitivity in sensitivities:
        for height in sensitivity.bank["heights"]:
            height = int(height)
            baseline_complete = _complete_blocks_from_inventory(
                baseline.bank["condition_inventory"], height=height
            )
            sensitivity_complete = _complete_blocks_from_inventory(
                sensitivity.bank["condition_inventory"], height=height
            )
            blocks = _paired_blocks(baseline, sensitivity, height)
            _validate_pairing_contract(
                baseline, sensitivity, height=height, blocks=blocks
            )
            unavailable_blocks, failure_types, unavailable_count = (
                _unavailable_inventory(sensitivity, height=height)
            )
            excluded_blocks = sorted(
                set(range(DEFAULT_BLOCK_COUNT)).difference(blocks)
            )
            pairing_inventory.append(
                {
                    "sensitivity_regime": sensitivity.regime_id,
                    "sensitivity_label": REGIME_SHORT_LABEL[
                        sensitivity.regime_id
                    ],
                    "height": height,
                    "baseline_complete_block_count": len(baseline_complete),
                    "sensitivity_complete_block_count": len(
                        sensitivity_complete
                    ),
                    "paired_complete_block_count": len(blocks),
                    "paired_condition_count": len(blocks)
                    * len(PLACEMENT_POLICIES),
                    "paired_block_indices": list(blocks),
                    "excluded_block_indices": excluded_blocks,
                    "sensitivity_unavailable_block_indices": unavailable_blocks,
                    "sensitivity_unavailable_condition_count": unavailable_count,
                    "sensitivity_unavailable_failure_types": failure_types,
                }
            )
            metric_rows.extend(
                _paired_metric_rows(
                    baseline,
                    sensitivity,
                    height=height,
                    blocks=blocks,
                    by_placement=False,
                )
            )
            metric_by_placement_rows.extend(
                _paired_metric_rows(
                    baseline,
                    sensitivity,
                    height=height,
                    blocks=blocks,
                    by_placement=True,
                )
            )
            contrast_rows.extend(
                _contrast_rows(
                    baseline, sensitivity, height=height, blocks=blocks
                )
            )
            diagnostic_rows.extend(
                _diagnostic_rows(
                    baseline, sensitivity, height=height, blocks=blocks
                )
            )
            adaptive_rows.extend(
                _adaptive_rows(
                    baseline, sensitivity, height=height, blocks=blocks
                )
            )

    source_inventory = [
        {
            "regime_id": bundle.regime_id,
            "short_label": REGIME_SHORT_LABEL[bundle.regime_id],
            "bank_id": bundle.bank["bank_id"],
            "bank_root": str(bundle.bank_root),
            "run_ids": list(bundle.run_ids),
            "result_roots": [str(root) for root in bundle.result_roots],
            "available_condition_count": int(
                bundle.bank["available_condition_count"]
            ),
            "record_count": len(bundle.records),
        }
        for bundle in (baseline, *sensitivities)
    ]
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "complete",
        "created_at_utc": created_at_utc
        or datetime.now(timezone.utc).isoformat(),
        "scientific_role": (
            "paired_simulator_sensitivity_development_evidence_not_paper_"
            "confirmation"
        ),
        "baseline_regime": BASELINE_SIMULATOR_REGIME,
        "sensitivity_regimes": list(SENSITIVITY_REGIMES),
        "source_inventory": source_inventory,
        "arm_ids": list(SELECTED_V2_ARM_IDS),
        "short_labels": {
            arm_id: SHORT_LABEL_BY_ARM[arm_id]
            for arm_id in SELECTED_V2_ARM_IDS
        },
        "pairing_contract": {
            "reference_regime": BASELINE_SIMULATOR_REGIME,
            "comparison_direction": "sensitivity_minus_v2",
            "independent_unit": "truth_block",
            "within_block_placement_policies": list(PLACEMENT_POLICIES),
            "condition_wtl_is_descriptive": True,
            "block_effect": "mean_of_three_coordinate_paired_placement_deltas",
            "complete_block_requirement": (
                "all_three_placements_available_in_both_regimes"
            ),
            "sensitivity_to_sensitivity_comparisons_generated": False,
        },
        "interpretation_contract": {
            "automatic_winner_declared": False,
            "heights_pooled": False,
            "regimes_pooled": False,
            "output_families_combined": False,
            "p_values_generated": False,
            "v2a_role": "high_initiation_and_population_burden_stress",
            "v2b_role": "event_geometry_and_raw_cnp_severity_sensitivity",
            "v2c_role": "sparse_ongoing_or_late_wgd_sensitivity",
            "v2c_is_early_clonal_wgd_evidence": False,
        },
        "pairing_inventory": pairing_inventory,
        "paired_metric_by_height": metric_rows,
        "paired_metric_by_placement": metric_by_placement_rows,
        "contrast_shift_by_height": contrast_rows,
        "diagnostic_by_height": diagnostic_rows,
        "adaptive_radius_by_height": adaptive_rows,
    }


def _flatten_summary(
    prefix: str, value: Mapping[str, Any] | None
) -> dict[str, Any]:
    fields = ("count", "minimum", "median", "mean", "maximum")
    return {
        f"{prefix}_{field}": None if value is None else value.get(field)
        for field in fields
    }


def _flatten_effect(
    prefix: str, value: Mapping[str, Any] | None
) -> dict[str, Any]:
    fields = (
        "count",
        "minimum",
        "median",
        "mean",
        "maximum",
        "wins",
        "ties",
        "losses",
        "conditional_gain_mean",
        "conditional_loss_mean",
        "worst_five_or_available_mean",
    )
    return {
        f"{prefix}_{field}": None if value is None else value.get(field)
        for field in fields
    }


def _flatten_paired(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "paired_condition_count": value["paired_condition_count"],
        "paired_block_count": value["paired_block_count"],
        "incomplete_block_indices": ";".join(
            str(index) for index in value["incomplete_block_indices"]
        ),
        **_flatten_summary("v2", value["baseline"]),
        **_flatten_summary("sensitivity", value["sensitivity"]),
        **_flatten_effect("condition_delta", value["condition_delta"]),
        **_flatten_effect("block_delta", value["block_delta"]),
    }


def _flat_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    flattened = []
    for row in rows:
        flattened.append(
            {
                **{key: value for key, value in row.items() if key != "paired"},
                **_flatten_paired(row["paired"]),
            }
        )
    return flattened


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty report table {path.name}.")
    fieldnames = list(rows[0])
    if any(list(row) != fieldnames for row in rows):
        raise ValueError(f"CSV row shape changed for {path.name}.")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _format_number(value: float | None, *, digits: int = 4) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def _effect_cell(row: Mapping[str, Any]) -> str:
    effect = row["paired"]["block_delta"]
    if effect is None:
        return "NA"
    mean = float(effect["mean"])
    return (
        f"{mean:+.4f} "
        f"({effect['wins']}/{effect['ties']}/{effect['losses']})"
    )


def _transition_cell(row: Mapping[str, Any], *, digits: int = 3) -> str:
    paired = row["paired"]
    baseline = paired["baseline"]
    sensitivity = paired["sensitivity"]
    block = paired["block_delta"]
    if baseline is None or sensitivity is None or block is None:
        return "NA"
    return (
        f"{_format_number(baseline['mean'], digits=digits)}→"
        f"{_format_number(sensitivity['mean'], digits=digits)} "
        f"({float(block['mean']):+.{digits}f})"
    )


def _row_index(
    rows: Sequence[Mapping[str, Any]],
    *keys: str,
) -> dict[tuple[Any, ...], Mapping[str, Any]]:
    result = {}
    for row in rows:
        key = tuple(row[name] for name in keys)
        if key in result:
            raise ValueError(f"Duplicate report row {key}.")
        result[key] = row
    return result


def _markdown(report: Mapping[str, Any]) -> str:
    pairing = report["pairing_inventory"]
    metrics = _row_index(
        report["paired_metric_by_height"],
        "sensitivity_regime",
        "height",
        "arm_id",
        "metric",
    )
    contrasts = _row_index(
        report["contrast_shift_by_height"],
        "sensitivity_regime",
        "height",
        "contrast_id",
    )
    diagnostics = _row_index(
        report["diagnostic_by_height"],
        "sensitivity_regime",
        "height",
        "diagnostic",
    )
    adaptive = _row_index(
        report["adaptive_radius_by_height"],
        "sensitivity_regime",
        "height",
        "feature",
    )
    lines = [
        "# CTBF v5 block-paired v2/v2a/v2b/v2c report",
        "",
        "This is development sensitivity evidence, not final paper qualification. "
        "Every value compares one sensitivity directly with coordinate-matched v2; "
        "heights, regimes, and full/partial output families are not pooled.",
        "",
        "## Paired inventory",
        "",
        "| Sensitivity | Height | Complete paired blocks | Paired conditions | Excluded blocks | Typed unavailable conditions |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in pairing:
        failure_types = row["sensitivity_unavailable_failure_types"]
        failures = ", ".join(
            f"{name}: {count}" for name, count in sorted(failure_types.items())
        ) or "none"
        excluded = ", ".join(
            str(value) for value in row["excluded_block_indices"]
        ) or "none"
        lines.append(
            f"| {row['sensitivity_label']} | H{row['height']} | "
            f"{row['paired_complete_block_count']} | "
            f"{row['paired_condition_count']} | {excluded} | {failures} |"
        )

    burden_names = (
        "selected_occurrence_count",
        "selected_unique_state_count",
        "mean_pairwise_cnp2cnp",
        "incomparable_pair_fraction",
        "sampled_ancestor_coverage_fraction",
        "normalized_minimum_invented_edge_fraction",
    )
    lines.extend(
        [
            "",
            "## Input and truth burden",
            "",
            "Cells show paired v2→sensitivity means and the mean independent block "
            "change in parentheses.",
            "",
            "| Sensitivity | Height | Occurrences | Unique states | Mean cnp2cnp | Incomparable | Ancestor coverage | Minimum invented edges |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for inventory_row in pairing:
        regime = inventory_row["sensitivity_regime"]
        height = inventory_row["height"]
        cells = [
            _transition_cell(diagnostics[(regime, height, name)])
            for name in burden_names
        ]
        lines.append(
            f"| {inventory_row['sensitivity_label']} | H{height} | "
            + " | ".join(cells)
            + " |"
        )

    focused_full = (
        ADAPTIVE_A_PRIME_ID,
        ADAPTIVE_B_PRIME_ID,
        POOLED_D_ID,
        POOLED_E_ID,
    )
    lines.extend(
        [
            "",
            "## Fully labeled primary effects",
            "",
            "Each cell is the mean block-paired AD-F1 change, followed by independent "
            "block W/T/L. Positive means the method scores higher under the sensitivity "
            "than on paired v2.",
            "",
            "| Sensitivity | Height | A' | B' | D | E |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for inventory_row in pairing:
        regime = inventory_row["sensitivity_regime"]
        height = inventory_row["height"]
        cells = [
            _effect_cell(metrics[(regime, height, arm_id, "ad_f1")])
            for arm_id in focused_full
        ]
        lines.append(
            f"| {inventory_row['sensitivity_label']} | H{height} | "
            + " | ".join(cells)
            + " |"
        )

    lines.extend(
        [
            "",
            "## Fully labeled method-contrast shifts",
            "",
            "These are paired difference-in-differences: (left−right) in the "
            "sensitivity minus (left−right) in v2. W/T/L uses independent blocks.",
            "",
            "| Sensitivity | Height | A'−B' | A'−D | A'−A | B'−B |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for inventory_row in pairing:
        regime = inventory_row["sensitivity_regime"]
        height = inventory_row["height"]
        cells = [
            _effect_cell(contrasts[(regime, height, contrast_id)])
            for contrast_id, _left, _right in FULL_CONTRASTS
        ]
        lines.append(
            f"| {inventory_row['sensitivity_label']} | H{height} | "
            + " | ".join(cells)
            + " |"
        )

    focused_partial = (
        PARTIAL_V2_ARM_IDS[0],
        PARTIAL_V2_ARM_IDS[1],
        PARTIAL_ADAPTIVE_Y_PRIME_ID,
        PARTIAL_V2_ARM_IDS[2],
    )
    lines.extend(
        [
            "",
            "## Partial primary effects",
            "",
            "Each cell is the mean block-paired GRF change and independent block "
            "W/T/L.",
            "",
            "| Sensitivity | Height | X | Y | Y' | Z |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for inventory_row in pairing:
        regime = inventory_row["sensitivity_regime"]
        height = inventory_row["height"]
        cells = [
            _effect_cell(metrics[(regime, height, arm_id, "grf")])
            for arm_id in focused_partial
        ]
        lines.append(
            f"| {inventory_row['sensitivity_label']} | H{height} | "
            + " | ".join(cells)
            + " |"
        )

    lines.extend(
        [
            "",
            "## Partial method-contrast shifts",
            "",
            "| Sensitivity | Height | Y'−Z' | Y'−Y | Y'−V' | Y'−X | Z'−U' |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for inventory_row in pairing:
        regime = inventory_row["sensitivity_regime"]
        height = inventory_row["height"]
        cells = [
            _effect_cell(contrasts[(regime, height, contrast_id)])
            for contrast_id, _left, _right in PARTIAL_CONTRASTS
        ]
        lines.append(
            f"| {inventory_row['sensitivity_label']} | H{height} | "
            + " | ".join(cells)
            + " |"
        )

    lines.extend(
        [
            "",
            "## Adaptive-radius response (A')",
            "",
            "| Sensitivity | Height | Mean radius | Radius >1 transition fraction | Copy-up fraction | Candidate coverage |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for inventory_row in pairing:
        regime = inventory_row["sensitivity_regime"]
        height = inventory_row["height"]
        cells = [
            _transition_cell(adaptive[(regime, height, feature)])
            for feature in ADAPTIVE_FEATURE_LABELS
        ]
        lines.append(
            f"| {inventory_row['sensitivity_label']} | H{height} | "
            + " | ".join(cells)
            + " |"
        )

    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- v2a is a high-initiation and population-burden stress; it is not a "
            "clean distance-only perturbation. Its H24 evidence excludes blocks 65, "
            "87, and 91 after typed simulator guard failures, without replacement.",
            "- v2b tests event geometry and raw-CNP severity. cnp2cnp distance need "
            "not increase because event-cost semantics and viability pruning also "
            "change the retained population.",
            "- v2c tests sparse ongoing/late WGD. It does not establish robustness "
            "to an early clonal WGD.",
            "- Placement-level W/T/L and all 21-arm complementary metrics are in "
            "the CSV/JSON outputs. They are descriptive because the three placements "
            "share one truth block.",
            "- This report declares no automatic winner and performs no p-value, "
            "height pooling, regime pooling, or cross-family comparison.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(
    *,
    v2_result_roots: Sequence[Path | str],
    v2a_result_roots: Sequence[Path | str],
    v2b_result_roots: Sequence[Path | str],
    v2c_result_roots: Sequence[Path | str],
    output_root: Path | str,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    requested_root = Path(output_root).expanduser().resolve()
    if requested_root.exists():
        raise FileExistsError(f"Output root already exists: {requested_root}.")
    report = build_report(
        v2_result_roots=v2_result_roots,
        v2a_result_roots=v2a_result_roots,
        v2b_result_roots=v2b_result_roots,
        v2c_result_roots=v2c_result_roots,
        created_at_utc=created_at_utc,
    )
    root = ensure_new_output_root(requested_root)
    write_json(root / "summary.json", report)
    pairing_rows = []
    for row in report["pairing_inventory"]:
        pairing_rows.append(
            {
                **{
                    key: value
                    for key, value in row.items()
                    if key
                    not in {
                        "paired_block_indices",
                        "excluded_block_indices",
                        "sensitivity_unavailable_block_indices",
                        "sensitivity_unavailable_failure_types",
                    }
                },
                "paired_block_indices": ";".join(
                    str(value) for value in row["paired_block_indices"]
                ),
                "excluded_block_indices": ";".join(
                    str(value) for value in row["excluded_block_indices"]
                ),
                "sensitivity_unavailable_block_indices": ";".join(
                    str(value)
                    for value in row["sensitivity_unavailable_block_indices"]
                ),
                "sensitivity_unavailable_failure_types": ";".join(
                    f"{name}:{count}"
                    for name, count in sorted(
                        row["sensitivity_unavailable_failure_types"].items()
                    )
                ),
            }
        )
    _write_csv(root / "pairing_inventory.csv", pairing_rows)
    _write_csv(
        root / "paired_metric_by_height.csv",
        _flat_rows(report["paired_metric_by_height"]),
    )
    _write_csv(
        root / "paired_metric_by_placement.csv",
        _flat_rows(report["paired_metric_by_placement"]),
    )
    _write_csv(
        root / "contrast_shift_by_height.csv",
        _flat_rows(report["contrast_shift_by_height"]),
    )
    _write_csv(
        root / "diagnostic_by_height.csv",
        _flat_rows(report["diagnostic_by_height"]),
    )
    _write_csv(
        root / "adaptive_radius_by_height.csv",
        _flat_rows(report["adaptive_radius_by_height"]),
    )
    (root / "report.md").write_text(_markdown(report), encoding="utf-8")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-result-root", type=Path, action="append", required=True)
    parser.add_argument("--v2a-result-root", type=Path, action="append", required=True)
    parser.add_argument("--v2b-result-root", type=Path, action="append", required=True)
    parser.add_argument("--v2c-result-root", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    report = write_report(
        v2_result_roots=arguments.v2_result_root,
        v2a_result_roots=arguments.v2a_result_root,
        v2b_result_roots=arguments.v2b_result_root,
        v2c_result_roots=arguments.v2c_result_root,
        output_root=arguments.output_root,
    )
    paired_blocks = sum(
        row["paired_complete_block_count"]
        for row in report["pairing_inventory"]
    )
    print(
        f"complete: {len(report['arm_ids'])} algorithms, "
        f"{paired_blocks} regime-height paired blocks"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
