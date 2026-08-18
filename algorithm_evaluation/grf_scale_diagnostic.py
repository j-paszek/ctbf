"""Read-only, post-registration GRF scale diagnostics for a closed v2 run.

This module does not change the registered analysis or its decisions.  It
validates the complete checksum closure, reads native evaluation records, and
writes one compact derivative outside the immutable result root.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from evaluation_contract import validate_evaluation_result

from algorithm_evaluation.paper_pipeline_contract import (
    ANALYSIS_SCHEMA_VERSION,
    EXPECTED_INVENTORY_SCHEMA_VERSION,
    REGISTERED_ARM_SPECS,
    REGISTERED_CLEAN_EXPERIMENT,
    condition_id,
    file_sha256,
    read_json,
    validate_checksum_closure,
    write_json_atomic,
)


GRF_SCALE_DIAGNOSTIC_SCHEMA_VERSION = "ctbf-grf-scale-diagnostic-v1"
ANCHOR_CONDITION = condition_id(0.5, "L3")

# Every contrast stays inside one registered output problem.  The right arm is
# the baseline used by the relative reduction in remaining GRF distance.
CONTRAST_SPECS = (
    {
        "contrast_id": "biopsy_guided_classical_minus_classical_partial",
        "family": "partial",
        "left_arm": "biopsy_guided_classical",
        "right_arm": "classical_partial",
        "registered_role": "secondary_partial_anchor",
    },
    {
        "contrast_id": "temporal_minimum_minus_temporal_minimum_no_time",
        "family": "fully_labeled",
        "left_arm": "temporal_minimum",
        "right_arm": "temporal_minimum_no_time",
        "registered_role": "primary_paired_grf_diagnostic",
    },
    {
        "contrast_id": "temporal_minimum_minus_rooted_labeled_nj",
        "family": "fully_labeled",
        "left_arm": "temporal_minimum",
        "right_arm": "rooted_labeled_nj",
        "registered_role": "complementary_to_secondary_ad_f1",
    },
    {
        "contrast_id": "anticentral_parsimony_minus_rooted_labeled_nj",
        "family": "fully_labeled",
        "left_arm": "anticentral_parsimony",
        "right_arm": "rooted_labeled_nj",
        "registered_role": "complementary_to_secondary_ad_f1",
    },
    {
        "contrast_id": "temporal_minimum_minus_anticentral_parsimony",
        "family": "fully_labeled",
        "left_arm": "temporal_minimum",
        "right_arm": "anticentral_parsimony",
        "registered_role": "complementary_to_secondary_ad_f1",
    },
)


def _numeric_summary(values: Iterable[float | int | None]) -> dict[str, Any]:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "standard_deviation": None,
            "minimum": None,
            "maximum": None,
        }
    return {
        "count": len(numeric),
        "mean": statistics.fmean(numeric),
        "median": statistics.median(numeric),
        "standard_deviation": statistics.stdev(numeric) if len(numeric) > 1 else None,
        "minimum": min(numeric),
        "maximum": max(numeric),
    }


def _bootstrap_interval(
    values: Sequence[float],
    *,
    repetitions: int,
    seed: int,
    chunk_size: int = 10_000,
) -> dict[str, Any]:
    values_array = np.asarray(values, dtype=float)
    if not len(values_array):
        return {
            "status": "undefined_no_complete_blocks",
            "lower": None,
            "upper": None,
            "repetitions": repetitions,
        }
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    means = np.empty(repetitions, dtype=float)
    for start in range(0, repetitions, chunk_size):
        stop = min(start + chunk_size, repetitions)
        indices = rng.integers(
            0,
            len(values_array),
            size=(stop - start, len(values_array)),
        )
        means[start:stop] = values_array[indices].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return {
        "status": "success",
        "lower": float(lower),
        "upper": float(upper),
        "repetitions": repetitions,
        "interval": "two_sided_percentile_95_percent",
        "quantile_method": "numpy_default_linear",
    }


def _cohen_dz(values: Sequence[float]) -> dict[str, Any]:
    values = [float(value) for value in values]
    if len(values) < 2:
        return {"status": "undefined_fewer_than_two_blocks", "value": None}
    deviation = statistics.stdev(values)
    if deviation == 0:
        return {"status": "undefined_zero_standard_deviation", "value": None}
    return {"status": "success", "value": statistics.fmean(values) / deviation}


def _relative_distance_reduction(left_similarity: float, right_similarity: float):
    """Return the fraction of the right arm's remaining GRF distance removed."""
    baseline_distance = 1.0 - float(right_similarity)
    if math.isclose(baseline_distance, 0.0, rel_tol=0.0, abs_tol=1e-15):
        return None
    return (float(left_similarity) - float(right_similarity)) / baseline_distance


def _complete_block_values(
    rows: Sequence[Mapping[str, Any]],
    *,
    value_key: str,
    regimes: Sequence[str],
) -> list[float]:
    by_replicate: dict[int, dict[str, float]] = defaultdict(dict)
    for row in rows:
        value = row.get(value_key)
        if value is not None:
            by_replicate[int(row["replicate"])][str(row["regime_id"])] = float(value)
    return [
        statistics.fmean(values[regime] for regime in regimes)
        for _replicate, values in sorted(by_replicate.items())
        if set(values) == set(regimes)
    ]


def _rankdata(values: Sequence[float]) -> list[float]:
    """Return stable average ranks, including tied values, without SciPy."""
    indexed = sorted(enumerate(float(value) for value in values), key=lambda item: (item[1], item[0]))
    ranks = [0.0] * len(indexed)
    start = 0
    while start < len(indexed):
        stop = start + 1
        while stop < len(indexed) and indexed[stop][1] == indexed[start][1]:
            stop += 1
        average_rank = ((start + 1) + stop) / 2.0
        for position in range(start, stop):
            ranks[indexed[position][0]] = average_rank
        start = stop
    return ranks


def _correlation(
    rows: Sequence[Mapping[str, Any]],
    left_key: str,
    right_key: str,
    *,
    kind: str,
) -> dict[str, Any]:
    pairs = [
        (float(row[left_key]), float(row[right_key]))
        for row in rows
        if row.get(left_key) is not None and row.get(right_key) is not None
    ]
    if len(pairs) < 2:
        return {"status": "undefined_fewer_than_two_pairs", "value": None, "count": len(pairs)}
    left, right = zip(*pairs)
    if kind == "spearman":
        left = _rankdata(left)
        right = _rankdata(right)
    elif kind != "pearson":
        raise ValueError(f"Unknown correlation kind {kind!r}.")
    if len(set(left)) == 1 or len(set(right)) == 1:
        return {"status": "undefined_constant_input", "value": None, "count": len(pairs)}
    value = float(np.corrcoef(np.asarray(left), np.asarray(right))[0, 1])
    return {"status": "success", "value": value, "count": len(pairs)}


def _associations(
    rows: Sequence[Mapping[str, Any]],
    *,
    outcomes: Sequence[str],
    predictors: Sequence[str],
) -> dict[str, Any]:
    return {
        outcome: {
            predictor: {
                kind: _correlation(rows, outcome, predictor, kind=kind)
                for kind in ("spearman", "pearson")
            }
            for predictor in predictors
        }
        for outcome in outcomes
    }


def _equal_count_strata(
    rows: Sequence[Mapping[str, Any]],
    *,
    predictor: str,
    outcomes: Sequence[str],
    count: int = 4,
) -> list[dict[str, Any]]:
    eligible = [row for row in rows if row.get(predictor) is not None]
    ordered = sorted(
        eligible,
        key=lambda row: (
            float(row[predictor]),
            str(row.get("case_id", "")),
        ),
    )
    groups: list[list[Mapping[str, Any]]] = [[] for _ in range(count)]
    for index, row in enumerate(ordered):
        group_index = min(count - 1, index * count // len(ordered))
        groups[group_index].append(row)
    return [
        {
            "stratum": index + 1,
            "count": len(group),
            "predictor_minimum": min(float(row[predictor]) for row in group) if group else None,
            "predictor_maximum": max(float(row[predictor]) for row in group) if group else None,
            "outcomes": {
                outcome: _numeric_summary(row.get(outcome) for row in group)
                for outcome in outcomes
            },
        }
        for index, group in enumerate(groups)
    ]


def _wins(rows: Sequence[Mapping[str, Any]], *, tolerance: float) -> dict[str, Any]:
    effects = [float(row["raw_grf_gain"]) for row in rows]
    wins = sum(value > tolerance for value in effects)
    ties = sum(abs(value) <= tolerance for value in effects)
    losses = sum(value < -tolerance for value in effects)
    total = len(effects)
    return {
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "tolerance": tolerance,
        "win_probability_with_half_ties": (wins + 0.5 * ties) / total if total else None,
        "matched_rank_biserial": (wins - losses) / total if total else None,
    }


def _descriptive_contrast_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    tolerance: float,
) -> dict[str, Any]:
    return {
        "complete_pair_count": len(rows),
        "left_grf_similarity": _numeric_summary(row["left_grf"] for row in rows),
        "right_grf_similarity": _numeric_summary(row["right_grf"] for row in rows),
        "raw_paired_grf_gain": _numeric_summary(row["raw_grf_gain"] for row in rows),
        "relative_reduction_in_remaining_grf_distance": _numeric_summary(
            row["relative_distance_reduction"] for row in rows
        ),
        "relative_distance_reduction_undefined_count": sum(
            row["relative_distance_reduction"] is None for row in rows
        ),
        "wins_ties_losses": _wins(rows, tolerance=tolerance),
        "truth_node_count": _numeric_summary(row["truth_node_count"] for row in rows),
        "left_reconstructed_to_true_node_ratio": _numeric_summary(
            row["left_reconstructed_to_true_node_ratio"] for row in rows
        ),
        "right_reconstructed_to_true_node_ratio": _numeric_summary(
            row["right_reconstructed_to_true_node_ratio"] for row in rows
        ),
    }


def _paired_effect_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    regimes: Sequence[str],
    bootstrap_repetitions: int,
    bootstrap_seed: int,
    tolerance: float,
) -> dict[str, Any]:
    summary = _descriptive_contrast_summary(rows, tolerance=tolerance)
    raw_blocks = _complete_block_values(
        rows,
        value_key="raw_grf_gain",
        regimes=regimes,
    )
    relative_blocks = _complete_block_values(
        rows,
        value_key="relative_distance_reduction",
        regimes=regimes,
    )
    summary.update(
        {
            "raw_gain_block_analysis": {
                "complete_block_count": len(raw_blocks),
                "summary": _numeric_summary(raw_blocks),
                "bootstrap_interval": _bootstrap_interval(
                    raw_blocks,
                    repetitions=bootstrap_repetitions,
                    seed=bootstrap_seed,
                ),
                "cohen_dz": _cohen_dz(raw_blocks),
            },
            "relative_distance_reduction_block_analysis": {
                "complete_block_count": len(relative_blocks),
                "summary": _numeric_summary(relative_blocks),
                "bootstrap_interval": _bootstrap_interval(
                    relative_blocks,
                    repetitions=bootstrap_repetitions,
                    seed=bootstrap_seed,
                ),
                "cohen_dz": _cohen_dz(relative_blocks),
            },
            "materiality_status": "not_assessed_no_calibrated_scale_portable_threshold",
        }
    )
    return summary


def _evaluation_path(output_root: Path, case_id_value: str, condition: str, arm: str) -> Path:
    return (
        output_root
        / "cases"
        / case_id_value
        / "conditions"
        / condition
        / "arms"
        / arm
        / "evaluation.json"
    )


def _eligible_evaluation(result: Mapping[str, Any]) -> tuple[bool, str]:
    if result.get("status") != "success":
        failure = result.get("failure", {})
        return False, f"failure:{failure.get('code', 'unknown')}"
    coverage = result["inputs"]["observation_label_coverage"]["fraction"]
    if float(coverage) < 1.0:
        return False, "successful_below_full_observation_coverage"
    return True, "success"


def _absolute_row(
    *,
    case: Mapping[str, Any],
    condition: str,
    fraction: float,
    schedule_id: str,
    arm: str,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    true_nodes = int(result["inputs"]["true_tree"]["node_count"])
    reconstructed_nodes = int(result["inputs"]["reconstructed_tree"]["node_count"])
    return {
        "case_id": case["case_id"],
        "replicate": int(case["replicate"]),
        "regime_id": case["regime_id"],
        "condition_id": condition,
        "fraction": float(fraction),
        "schedule_id": schedule_id,
        "arm_id": arm,
        "grf_similarity": float(result["metrics"]["grf"]),
        "grf_distance": float(result["metrics"]["ext_grf"]),
        "truth_node_count": true_nodes,
        "reconstructed_node_count": reconstructed_nodes,
        "reconstructed_to_true_node_ratio": reconstructed_nodes / true_nodes,
        "true_minus_reconstructed_node_count": true_nodes - reconstructed_nodes,
        "observation_unique_label_count": int(
            result["inputs"]["observation_label_coverage"]["required_unique_label_count"]
        ),
    }


def _contrast_row(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> dict[str, Any]:
    shared_fields = ("case_id", "replicate", "regime_id", "condition_id", "fraction", "schedule_id")
    if any(left[field] != right[field] for field in shared_fields):
        raise ValueError("Paired GRF rows do not describe the same case and condition.")
    if left["truth_node_count"] != right["truth_node_count"]:
        raise ValueError("Paired evaluations disagree on true-tree node count.")
    left_similarity = float(left["grf_similarity"])
    right_similarity = float(right["grf_similarity"])
    return {
        **{field: left[field] for field in shared_fields},
        "truth_node_count": left["truth_node_count"],
        "left_grf": left_similarity,
        "right_grf": right_similarity,
        "raw_grf_gain": left_similarity - right_similarity,
        "relative_distance_reduction": _relative_distance_reduction(
            left_similarity,
            right_similarity,
        ),
        "left_reconstructed_node_count": left["reconstructed_node_count"],
        "right_reconstructed_node_count": right["reconstructed_node_count"],
        "left_reconstructed_to_true_node_ratio": left["reconstructed_to_true_node_ratio"],
        "right_reconstructed_to_true_node_ratio": right["reconstructed_to_true_node_ratio"],
        "reconstructed_to_true_ratio_difference": (
            left["reconstructed_to_true_node_ratio"]
            - right["reconstructed_to_true_node_ratio"]
        ),
        "observation_unique_label_count": left["observation_unique_label_count"],
    }


def _registered_decision_audit(output_root: Path) -> dict[str, Any]:
    summary_path = output_root / "analysis" / ANALYSIS_SCHEMA_VERSION / "summary.json"
    registered = read_json(summary_path)
    if registered.get("schema_version") != ANALYSIS_SCHEMA_VERSION:
        raise ValueError("Closed registered analysis has an unexpected schema version.")
    partial = registered["secondary_partial_anchor"]
    return {
        "registered_analysis_summary_sha256": file_sha256(summary_path),
        "contrast": "biopsy_guided_classical_minus_classical_partial",
        "condition": ANCHOR_CONDITION,
        "registered_material_threshold": partial["material_threshold"],
        "registered_unadjusted_statistical_support": partial[
            "unadjusted_statistical_support"
        ],
        "registered_raw_block_effect_summary": partial["block_effect_summary"],
        "registered_raw_bootstrap_interval": partial["bootstrap_interval"],
        "interpretation": (
            "The registered 0.01 rule remains an immutable audit fact but is retired "
            "as a future GRF materiality policy. This post-registration diagnostic "
            "cannot change the registered decision."
        ),
    }


def build_report(output_root: Path | str) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    validate_checksum_closure(
        output_root,
        "complete_checksums.sha256",
        include_analysis=True,
    )
    inventory = read_json(output_root / "expected_inventory.json")
    if inventory.get("schema_version") != EXPECTED_INVENTORY_SCHEMA_VERSION:
        raise ValueError("Unknown expected-inventory schema.")
    if inventory.get("experiment_id") != REGISTERED_CLEAN_EXPERIMENT:
        raise ValueError("GRF scale diagnostics require the registered clean experiment.")

    manifest = read_json(output_root / "design_manifest.snapshot.json")
    clean = manifest["experiments"]["clean_confirmation"]
    regimes = list(clean["regime_ids"])
    bootstrap_repetitions = int(clean["analysis"]["bootstrap_repetitions"])
    tolerance = float(clean["analysis"]["win_tie_loss_tolerance"])
    seed_record = next(
        record
        for record in manifest["seed_contract"]["experiments"]
        if record["experiment_id"] == REGISTERED_CLEAN_EXPERIMENT
    )
    bootstrap_seed = int(seed_record["analysis_seeds"]["block_bootstrap"])

    arm_ids = [arm_id for arm_id, _algorithm in REGISTERED_ARM_SPECS]
    absolute_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    contrast_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    eligibility_counts: Counter[str] = Counter()

    for case in inventory["cases"]:
        for condition in case["condition_ids"]:
            input_record = read_json(
                output_root
                / "cases"
                / case["case_id"]
                / "conditions"
                / condition
                / "input.json"
            )
            if input_record.get("condition_id") != condition:
                raise ValueError("Condition input id differs from the expected inventory.")
            fraction = float(input_record["fraction"])
            schedule_id = str(input_record["schedule_id"])
            by_arm = {}
            for arm in arm_ids:
                result = read_json(_evaluation_path(output_root, case["case_id"], condition, arm))
                validate_evaluation_result(result)
                eligible, code = _eligible_evaluation(result)
                eligibility_counts[f"{arm}:{code}"] += 1
                if not eligible:
                    continue
                row = _absolute_row(
                    case=case,
                    condition=condition,
                    fraction=fraction,
                    schedule_id=schedule_id,
                    arm=arm,
                    result=result,
                )
                absolute_rows[arm].append(row)
                by_arm[arm] = row
            for spec in CONTRAST_SPECS:
                left = by_arm.get(spec["left_arm"])
                right = by_arm.get(spec["right_arm"])
                if left is not None and right is not None:
                    contrast_rows[spec["contrast_id"]].append(_contrast_row(left, right))

    absolute_anchor = {}
    for arm in arm_ids:
        rows = [row for row in absolute_rows[arm] if row["condition_id"] == ANCHOR_CONDITION]
        absolute_anchor[arm] = {
            "count": len(rows),
            "grf_similarity": _numeric_summary(row["grf_similarity"] for row in rows),
            "tree_size": {
                "truth_node_count": _numeric_summary(row["truth_node_count"] for row in rows),
                "reconstructed_node_count": _numeric_summary(
                    row["reconstructed_node_count"] for row in rows
                ),
                "reconstructed_to_true_node_ratio": _numeric_summary(
                    row["reconstructed_to_true_node_ratio"] for row in rows
                ),
            },
            "score_size_associations": _associations(
                rows,
                outcomes=("grf_similarity",),
                predictors=(
                    "truth_node_count",
                    "reconstructed_node_count",
                    "reconstructed_to_true_node_ratio",
                    "true_minus_reconstructed_node_count",
                ),
            ),
            "score_by_truth_size_rank_quartile": _equal_count_strata(
                rows,
                predictor="truth_node_count",
                outcomes=("grf_similarity", "reconstructed_to_true_node_ratio"),
            ),
            "score_by_reconstructed_to_true_ratio_rank_quartile": _equal_count_strata(
                rows,
                predictor="reconstructed_to_true_node_ratio",
                outcomes=("grf_similarity", "truth_node_count"),
            ),
        }

    contrasts = {}
    for spec in CONTRAST_SPECS:
        identifier = spec["contrast_id"]
        rows = contrast_rows[identifier]
        anchor_rows = [row for row in rows if row["condition_id"] == ANCHOR_CONDITION]
        condition_summaries = {}
        for condition in sorted({row["condition_id"] for row in rows}):
            condition_rows = [row for row in rows if row["condition_id"] == condition]
            description = _descriptive_contrast_summary(
                condition_rows,
                tolerance=tolerance,
            )
            description["fraction"] = condition_rows[0]["fraction"]
            description["schedule_id"] = condition_rows[0]["schedule_id"]
            condition_summaries[condition] = description
        contrasts[identifier] = {
            **spec,
            "anchor_condition": ANCHOR_CONDITION,
            "anchor": {
                **_paired_effect_summary(
                    anchor_rows,
                    regimes=regimes,
                    bootstrap_repetitions=bootstrap_repetitions,
                    bootstrap_seed=bootstrap_seed,
                    tolerance=tolerance,
                ),
                "effect_size_associations": _associations(
                    anchor_rows,
                    outcomes=("raw_grf_gain", "relative_distance_reduction"),
                    predictors=(
                        "truth_node_count",
                        "left_reconstructed_to_true_node_ratio",
                        "right_reconstructed_to_true_node_ratio",
                        "reconstructed_to_true_ratio_difference",
                    ),
                ),
                "effect_by_truth_size_rank_quartile": _equal_count_strata(
                    anchor_rows,
                    predictor="truth_node_count",
                    outcomes=(
                        "raw_grf_gain",
                        "relative_distance_reduction",
                        "left_grf",
                        "right_grf",
                    ),
                ),
                "effect_by_baseline_size_ratio_rank_quartile": _equal_count_strata(
                    anchor_rows,
                    predictor="right_reconstructed_to_true_node_ratio",
                    outcomes=(
                        "raw_grf_gain",
                        "relative_distance_reduction",
                        "left_grf",
                        "right_grf",
                    ),
                ),
            },
            "descriptive_by_observation_condition": condition_summaries,
        }

    return {
        "schema_version": GRF_SCALE_DIAGNOSTIC_SCHEMA_VERSION,
        "analysis_role": "post_registration_descriptive_grf_scale_diagnostic",
        "experiment_id": REGISTERED_CLEAN_EXPERIMENT,
        "integrity": {
            "complete_checksum_validated_before_metric_reads": True,
            "raw_checksum_file_sha256": file_sha256(output_root / "raw_checksums.sha256"),
            "complete_checksum_file_sha256": file_sha256(
                output_root / "complete_checksums.sha256"
            ),
            "source_file_sha256": file_sha256(Path(__file__)),
        },
        "metric_semantics": {
            "reported_native_score": "grf_similarity_equals_1_minus_ext_grf_distance",
            "raw_gain": "left_grf_similarity_minus_right_grf_similarity",
            "relative_distance_reduction": (
                "raw_gain_divided_by_right_ext_grf_distance; undefined when the "
                "right distance is zero"
            ),
            "relative_similarity_gain": "deliberately_not_reported_near_zero_denominator",
            "absolute_score_is_not_percentage_topology_recovered": True,
            "cross_output_problem_ranking_forbidden": True,
        },
        "materiality_policy": {
            "registered_0p01_rule": "preserved_only_as_historical_decision_audit",
            "future_fixed_raw_grf_threshold": "retired_not_scale_portable",
            "replacement_reporting": [
                "raw paired GRF gain with block-aware interval",
                "relative reduction in remaining GRF distance",
                "paired win probability and matched rank-biserial effect",
                "truth-size, observation-condition, and reconstructed/true-size diagnostics",
            ],
            "post_registration_results_cannot_override_registered_decision": True,
        },
        "resampling": {
            "unit": "replicate block averaged across all registered regimes",
            "regimes": regimes,
            "bootstrap_repetitions": bootstrap_repetitions,
            "bootstrap_seed": bootstrap_seed,
            "same_seed_reused_for_declared_diagnostic_summaries": True,
        },
        "eligibility_counts": dict(sorted(eligibility_counts.items())),
        "registered_decision_audit": _registered_decision_audit(output_root),
        "absolute_anchor_by_arm": absolute_anchor,
        "contrasts": contrasts,
        "case_rows_written": False,
        "interpretation_limits": [
            "Condition and size strata are descriptive post-registration analyses.",
            "Score-size correlations are descriptive and may be confounded by regime and observation design.",
            "Reconstructed node count is an algorithm outcome, so conditioning on it may change the estimand.",
            "Relative distance reduction improves scale interpretation but does not estimate an observation-constrained oracle ceiling.",
            "No new material-effect threshold is inferred from these observed results.",
        ],
    }


def _printed_summary(report: Mapping[str, Any], output_path: Path) -> dict[str, Any]:
    partial = report["contrasts"]["biopsy_guided_classical_minus_classical_partial"]
    anchor = partial["anchor"]
    return {
        "schema_version": report["schema_version"],
        "analysis_role": report["analysis_role"],
        "complete_checksum_validated": report["integrity"][
            "complete_checksum_validated_before_metric_reads"
        ],
        "output": str(output_path),
        "registered_decision_audit": report["registered_decision_audit"],
        "partial_anchor": {
            "complete_pair_count": anchor["complete_pair_count"],
            "left_grf_similarity": anchor["left_grf_similarity"],
            "right_grf_similarity": anchor["right_grf_similarity"],
            "raw_gain_block_analysis": anchor["raw_gain_block_analysis"],
            "relative_distance_reduction_block_analysis": anchor[
                "relative_distance_reduction_block_analysis"
            ],
            "wins_ties_losses": anchor["wins_ties_losses"],
            "materiality_status": anchor["materiality_status"],
            "effect_size_associations": anchor["effect_size_associations"],
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create a compact post-registration GRF scale diagnostic."
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    output_root = args.output_root.resolve()
    output_path = args.output.resolve()
    if output_path == output_root or output_root in output_path.parents:
        raise ValueError("Diagnostic output must be outside the immutable result root.")
    if output_path.exists():
        raise FileExistsError(f"Refusing to replace existing diagnostic: {output_path}")

    report = build_report(output_root)
    write_json_atomic(output_path, report)
    print(json.dumps(_printed_summary(report, output_path), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ANCHOR_CONDITION",
    "CONTRAST_SPECS",
    "GRF_SCALE_DIAGNOSTIC_SCHEMA_VERSION",
    "build_report",
    "main",
]
