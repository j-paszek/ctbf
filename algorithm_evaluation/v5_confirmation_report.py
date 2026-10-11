"""Create the preregistered block-paired CTBF v5 confirmation report."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import math
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

import numpy as np

from algorithm_evaluation.paper_pipeline_contract import read_json
from algorithm_evaluation.v5_confirmation_common import (
    BIOPSY_COUNT_CONDITION_IDS,
    DEFAULT_MANIFEST_PATH,
    M0_ID,
    M2_ID,
    M5_ID,
    M6_ID,
    M7_ID,
    PAPER_ARM_IDS,
    PAPER_ID_BY_ARM,
    PLACEMENT_CONDITION_IDS,
    POOLED_Q_ID,
    REPORT_SCHEMA_VERSION,
    STANDARD_CONDITION_IDS,
    derive_analysis_seed,
    ensure_new_output_root,
    validate_internal_runtime,
    validate_manifest,
    write_checksum_closure,
    write_json,
)
from algorithm_evaluation.v5_confirmation_run import load_confirmation_run
from algorithm_evaluation.v5_confirmation_medicc2 import (
    load_medicc2_confirmation,
)
from algorithm_evaluation.v5_medicc2_development_run import (
    MEDICC2_DEVELOPMENT_ARM_ID,
)


REPORT_CHECKSUM_NAME = "report_checksums.sha256"


def _numeric_summary(values: Sequence[float]) -> dict[str, Any]:
    normalized = [float(value) for value in values]
    ordered = sorted(normalized)
    negative = [value for value in normalized if value < 0]
    positive = [value for value in normalized if value > 0]
    return {
        "count": len(normalized),
        "mean": statistics.fmean(normalized) if normalized else None,
        "median": statistics.median(normalized) if normalized else None,
        "minimum": min(normalized) if normalized else None,
        "maximum": max(normalized) if normalized else None,
        "worst_five_or_available_mean": (
            statistics.fmean(ordered[:5]) if ordered else None
        ),
        "conditional_gain_mean": statistics.fmean(positive) if positive else None,
        "conditional_loss_mean": statistics.fmean(negative) if negative else None,
    }


def _bootstrap_interval(
    values: Sequence[float],
    *,
    repetitions: int,
    seed: int,
    chunk_size: int = 10_000,
) -> dict[str, Any]:
    data = np.asarray(values, dtype=float)
    if not len(data):
        return {
            "status": "undefined_no_complete_blocks",
            "lower": None,
            "upper": None,
            "repetitions": repetitions,
        }
    rng = np.random.Generator(np.random.PCG64(seed))
    means = np.empty(repetitions, dtype=float)
    for start in range(0, repetitions, chunk_size):
        stop = min(start + chunk_size, repetitions)
        indices = rng.integers(0, len(data), size=(stop - start, len(data)))
        means[start:stop] = data[indices].mean(axis=1)
    lower, upper = np.quantile(means, (0.025, 0.975))
    return {
        "status": "success",
        "lower": float(lower),
        "upper": float(upper),
        "repetitions": repetitions,
        "interval": "two_sided_percentile_95_percent",
        "quantile_method": "numpy_default_linear",
    }


def _one_sided_sign_flip(
    values: Sequence[float],
    *,
    repetitions: int,
    seed: int,
    chunk_size: int = 20_000,
) -> dict[str, Any]:
    data = np.asarray(values, dtype=float)
    if not len(data):
        return {
            "status": "undefined_no_complete_blocks",
            "p_value": None,
            "repetitions": repetitions,
        }
    observed = float(data.mean())
    rng = np.random.Generator(np.random.PCG64(seed))
    extreme = 0
    for start in range(0, repetitions, chunk_size):
        count = min(chunk_size, repetitions - start)
        signs = rng.integers(0, 2, size=(count, len(data)), dtype=np.int8)
        signs = signs * 2 - 1
        permuted = (signs * data).mean(axis=1)
        extreme += int(np.count_nonzero(permuted >= observed - 1e-15))
    return {
        "status": "success",
        "p_value": float((extreme + 1) / (repetitions + 1)),
        "extreme_draw_count": extreme,
        "repetitions": repetitions,
        "tail": "one_sided_positive_mean",
        "monte_carlo_correction": "plus_one_numerator_and_denominator",
    }


def _wins_ties_losses(values: Sequence[float], tolerance: float) -> dict[str, Any]:
    return {
        "wins": sum(value > tolerance for value in values),
        "ties": sum(abs(value) <= tolerance for value in values),
        "losses": sum(value < -tolerance for value in values),
        "tolerance": tolerance,
    }


def _effect_summary(
    values: Sequence[float],
    *,
    bootstrap_seed: int,
    bootstrap_repetitions: int = 100_000,
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    return {
        "effect_summary": _numeric_summary(values),
        "wins_ties_losses": _wins_ties_losses(values, tolerance),
        "bootstrap_interval": _bootstrap_interval(
            values,
            repetitions=bootstrap_repetitions,
            seed=bootstrap_seed,
        ),
    }


def _load_record_index(
    run_root: Path,
    run: Mapping[str, Any],
) -> dict[tuple[int, str, str], dict[str, Any]]:
    index = {}
    for inventory_record in run["record_inventory"]:
        record = read_json(run_root / str(inventory_record["path"]))
        key = (
            int(record["block_index"]),
            str(record["condition_id"]),
            str(record["arm_id"]),
        )
        if key in index:
            raise ValueError(f"Duplicate confirmation record: {key}.")
        index[key] = record
    return index


def _load_medicc2_index(
    run_root: Path,
    run: Mapping[str, Any],
) -> dict[tuple[int, str, str], dict[str, Any]]:
    index = {}
    for inventory_record in run["record_inventory"]:
        record = read_json(run_root / str(inventory_record["path"]))
        key = (
            int(record["block_index"]),
            str(record["condition_id"]),
            str(record["arm_id"]),
        )
        if key in index:
            raise ValueError(f"Duplicate MEDICC2 confirmation record: {key}.")
        index[key] = record
    return index


def _metric(
    index: Mapping[tuple[int, str, str], Mapping[str, Any]],
    block_index: int,
    condition_id: str,
    arm_id: str,
    metric: str,
) -> float | None:
    record = index[(block_index, condition_id, arm_id)]
    if record.get("status") != "success":
        return None
    value = record.get("metrics", {}).get(metric)
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)


def _fixed_ad_f1(
    index: Mapping[tuple[int, str, str], Mapping[str, Any]],
    block_index: int,
    condition_id: str,
    arm_id: str,
) -> float | None:
    record = index[(block_index, condition_id, arm_id)]
    if record.get("status") != "success":
        return None
    fixed = record.get("fixed_label_evaluation")
    if not isinstance(fixed, Mapping) or fixed.get("status") != "success":
        return None
    return float(fixed["metrics"]["ad_f1"])


def _standard_block_effects(
    index: Mapping[tuple[int, str, str], Mapping[str, Any]],
    *,
    block_count: int,
    left_arm: str,
    right_arm: str,
    metric: str,
) -> tuple[list[dict[str, Any]], list[float]]:
    rows = []
    complete = []
    for block_index in range(block_count):
        effects = []
        per_condition = {}
        for condition_id in STANDARD_CONDITION_IDS:
            left = _metric(index, block_index, condition_id, left_arm, metric)
            right = _metric(index, block_index, condition_id, right_arm, metric)
            effect = None if left is None or right is None else left - right
            per_condition[condition_id] = effect
            if effect is not None:
                effects.append(effect)
        block_effect = (
            statistics.fmean(effects)
            if len(effects) == len(STANDARD_CONDITION_IDS)
            else None
        )
        if block_effect is not None:
            complete.append(block_effect)
        rows.append(
            {
                "block_index": block_index,
                "effects_by_condition": per_condition,
                "block_effect": block_effect,
                "complete": block_effect is not None,
            }
        )
    return rows, complete


def _primary_analysis(
    manifest: Mapping[str, Any],
    index: Mapping[tuple[int, str, str], Mapping[str, Any]],
    block_count: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    contract = manifest["primary_hypothesis"]
    rows, block_effects = _standard_block_effects(
        index,
        block_count=block_count,
        left_arm=M2_ID,
        right_arm=M0_ID,
        metric="grf",
    )
    case_effects = [
        effect
        for row in rows
        for effect in row["effects_by_condition"].values()
        if effect is not None
    ]
    by_condition = {
        condition_id: [
            row["effects_by_condition"][condition_id]
            for row in rows
            if row["effects_by_condition"][condition_id] is not None
        ]
        for condition_id in STANDARD_CONDITION_IDS
    }
    failure_rates = {}
    for arm_id in (M0_ID, M2_ID):
        failure_rates[arm_id] = {}
        for condition_id in STANDARD_CONDITION_IDS:
            successful = sum(
                index[(block_index, condition_id, arm_id)].get("status") == "success"
                for block_index in range(block_count)
            )
            failure_rates[arm_id][condition_id] = 1.0 - successful / block_count
    maximum_failure = max(
        value for by_condition_rates in failure_rates.values()
        for value in by_condition_rates.values()
    )
    interval = _bootstrap_interval(
        block_effects,
        repetitions=int(contract["bootstrap_repetitions"]),
        seed=derive_analysis_seed("primary_bootstrap"),
    )
    sign_flip = _one_sided_sign_flip(
        block_effects,
        repetitions=int(contract["sign_flip_repetitions"]),
        seed=derive_analysis_seed("primary_sign_flip"),
    )
    mean = statistics.fmean(block_effects) if block_effects else None
    gates = {
        "minimum_complete_blocks": len(block_effects)
        >= int(contract["minimum_complete_blocks"]),
        "maximum_failure_rate": maximum_failure
        <= float(contract["maximum_arm_failure_rate_per_height"]),
        "minimum_material_mean_gain": mean is not None
        and mean >= float(contract["minimum_material_mean_gain"]),
        "bootstrap_lower_above_zero": interval.get("lower") is not None
        and float(interval["lower"]) > 0,
        "one_sided_sign_flip": sign_flip.get("p_value") is not None
        and float(sign_flip["p_value"]) < float(contract["alpha"]),
    }
    return {
        "contrast": "M2_minus_M0",
        "metric": "grf",
        "independent_unit": "truth_block",
        "complete_block_count": len(block_effects),
        "complete_condition_pair_count": len(case_effects),
        "condition_effect_summary": _numeric_summary(case_effects),
        "block_effect_summary": _numeric_summary(block_effects),
        "wins_ties_losses_by_condition": {
            condition_id: _wins_ties_losses(
                values,
                float(contract["tie_tolerance"]),
            )
            for condition_id, values in by_condition.items()
        },
        "combined_condition_wins_ties_losses": _wins_ties_losses(
            case_effects,
            float(contract["tie_tolerance"]),
        ),
        "block_wins_ties_losses": _wins_ties_losses(
            block_effects,
            float(contract["tie_tolerance"]),
        ),
        "bootstrap_interval": interval,
        "sign_flip": sign_flip,
        "failure_rates": failure_rates,
        "maximum_primary_arm_failure_rate": maximum_failure,
        "support_gates": gates,
        "registered_support": all(gates.values()),
        "contract": contract,
    }, rows


def _secondary_standard_contrasts(
    index: Mapping[tuple[int, str, str], Mapping[str, Any]],
    block_count: int,
) -> dict[str, Any]:
    specifications = {
        "M6_minus_M5": (M6_ID, M5_ID, "ad_f1"),
        "M7_minus_Pooled-Q": (M7_ID, POOLED_Q_ID, "ad_f1"),
    }
    result = {}
    base_seed = derive_analysis_seed("secondary_bootstrap")
    for offset, (name, (left, right, metric)) in enumerate(specifications.items()):
        rows, effects = _standard_block_effects(
            index,
            block_count=block_count,
            left_arm=left,
            right_arm=right,
            metric=metric,
        )
        condition_effects = [
            effect
            for row in rows
            for effect in row["effects_by_condition"].values()
            if effect is not None
        ]
        result[name] = {
            "metric": metric,
            "complete_block_count": len(effects),
            "block": _effect_summary(
                effects,
                bootstrap_seed=base_seed + offset,
            ),
            "combined_conditions": {
                "effect_summary": _numeric_summary(condition_effects),
                "wins_ties_losses": _wins_ties_losses(condition_effects, 1e-12),
            },
            "binary_support_rule": False,
        }
    return result


def _biopsy_count_analysis(
    index: Mapping[tuple[int, str, str], Mapping[str, Any]],
    block_count: int,
) -> dict[str, Any]:
    curves = {}
    for condition_id in BIOPSY_COUNT_CONDITION_IDS:
        values = [
            value
            for block_index in range(block_count)
            if (
                value := _fixed_ad_f1(index, block_index, condition_id, M6_ID)
            )
            is not None
        ]
        curves[condition_id] = _numeric_summary(values)
    endpoint = []
    for block_index in range(block_count):
        low = _fixed_ad_f1(index, block_index, "H24_L2", M6_ID)
        high = _fixed_ad_f1(index, block_index, "H24_L5", M6_ID)
        if low is not None and high is not None:
            endpoint.append(high - low)
    return {
        "method": "M6",
        "metric": "fixed_label_ad_f1_on_H24_L2_labels",
        "curves": curves,
        "endpoint_H24_L5_minus_H24_L2": _effect_summary(
            endpoint,
            bootstrap_seed=derive_analysis_seed("secondary_bootstrap") + 20,
        ),
        "binary_support_rule": False,
    }


def _placement_analysis(
    index: Mapping[tuple[int, str, str], Mapping[str, Any]],
    block_count: int,
) -> dict[str, Any]:
    result = {}
    seed = derive_analysis_seed("secondary_bootstrap") + 40
    for arm_offset, arm_id in enumerate(PAPER_ARM_IDS):
        metric = "grf" if arm_id in {M0_ID, M2_ID} else "ad_f1"
        comparisons = {}
        for comparison_offset, comparator in enumerate(("H24_late", "H24_random")):
            values = []
            for block_index in range(block_count):
                anchor = _metric(index, block_index, "H24_rule_y", arm_id, metric)
                other = _metric(index, block_index, comparator, arm_id, metric)
                if anchor is not None and other is not None:
                    values.append(anchor - other)
            comparisons[f"H24_rule_y_minus_{comparator}"] = _effect_summary(
                values,
                bootstrap_seed=seed + arm_offset * 2 + comparison_offset,
            )
        result[PAPER_ID_BY_ARM[arm_id]] = {
            "arm_id": arm_id,
            "metric": metric,
            "comparisons": comparisons,
        }
    return result


def _absolute_scores(
    index: Mapping[tuple[int, str, str], Mapping[str, Any]],
    block_count: int,
) -> dict[str, Any]:
    result = {}
    for arm_id in PAPER_ARM_IDS:
        result[PAPER_ID_BY_ARM[arm_id]] = {
            condition_id: {
                metric: _numeric_summary(
                    [
                        value
                        for block_index in range(block_count)
                        if (
                            value := _metric(
                                index,
                                block_index,
                                condition_id,
                                arm_id,
                                metric,
                            )
                        )
                        is not None
                    ]
                )
                for metric in (
                    ("grf",)
                    if arm_id in {M0_ID, M2_ID}
                    else ("ad_f1", "grf", "ad_precision", "ad_recall")
                )
            }
            for condition_id in STANDARD_CONDITION_IDS
        }
    return result


def _resource_and_failure_summary(
    index: Mapping[tuple[int, str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    result = {}
    for arm_id in PAPER_ARM_IDS:
        records = [record for key, record in index.items() if key[2] == arm_id]
        statuses: dict[str, int] = {}
        reconstruction_wall = []
        evaluation_wall = []
        reconstruction_rss = []
        evaluation_rss = []
        for record in records:
            status = str(record["status"])
            statuses[status] = statuses.get(status, 0) + 1
            for stage, wall_values, rss_values in (
                ("reconstruction", reconstruction_wall, reconstruction_rss),
                ("evaluation", evaluation_wall, evaluation_rss),
            ):
                resources = record.get("resources", {}).get(stage)
                if not isinstance(resources, Mapping):
                    continue
                if isinstance(resources.get("wall_time_ns"), int):
                    wall_values.append(float(resources["wall_time_ns"]) / 1e9)
                memory = resources.get("memory")
                if isinstance(memory, Mapping) and isinstance(
                    memory.get("peak_rss_bytes"), int
                ):
                    rss_values.append(float(memory["peak_rss_bytes"]))
        result[PAPER_ID_BY_ARM[arm_id]] = {
            "arm_id": arm_id,
            "status_counts": statuses,
            "reconstruction_wall_seconds": _numeric_summary(reconstruction_wall),
            "evaluation_wall_seconds": _numeric_summary(evaluation_wall),
            "reconstruction_peak_rss_bytes": _numeric_summary(reconstruction_rss),
            "evaluation_peak_rss_bytes": _numeric_summary(evaluation_rss),
        }
    return result


def build_report(
    *,
    run_root: Path | str,
    medicc2_root: Path | str | None = None,
    manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
    technical_smoke: bool = False,
    created_at_utc: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    validation = validate_manifest(
        manifest_path,
        require_approved=not technical_smoke,
    )
    internal_runtime = validate_internal_runtime(validation["manifest"])
    root, run = load_confirmation_run(
        run_root,
        manifest_path=manifest_path,
        allow_smoke=technical_smoke,
    )
    index = _load_record_index(root, run)
    block_count = 1 if technical_smoke else int(validation["block_count"])
    medicc2 = None
    if medicc2_root is None:
        if not technical_smoke:
            raise ValueError(
                "The production report requires the separately closed MEDICC2 run."
            )
    else:
        if technical_smoke:
            raise ValueError("A technical-smoke report cannot ingest MEDICC2 evidence.")
        medicc2_path, medicc2_run = load_medicc2_confirmation(
            medicc2_root,
            validation=validation,
        )
        external_index = _load_medicc2_index(medicc2_path, medicc2_run)
        combined_index = {**index, **external_index}
        rows, effects = _standard_block_effects(
            combined_index,
            block_count=block_count,
            left_arm=M2_ID,
            right_arm=MEDICC2_DEVELOPMENT_ARM_ID,
            metric="grf",
        )
        condition_effects = [
            effect
            for row in rows
            for effect in row["effects_by_condition"].values()
            if effect is not None
        ]
        medicc2 = {
            "contrast": "M2_minus_MEDICC2",
            "metric": "grf",
            "complete_block_count": len(effects),
            "block": _effect_summary(
                effects,
                bootstrap_seed=derive_analysis_seed("secondary_bootstrap") + 80,
            ),
            "combined_conditions": {
                "effect_summary": _numeric_summary(condition_effects),
                "wins_ties_losses": _wins_ties_losses(condition_effects, 1e-12),
            },
            "run_root": str(medicc2_path),
            "run_checksum_sha256": medicc2_run.get("checksum_closure", {}).get(
                "sha256"
            ),
            "status_counts": medicc2_run.get("status_counts"),
            "binary_support_rule": False,
            "role": "topology_only_contextual_comparator",
        }
    primary, primary_rows = _primary_analysis(
        validation["manifest"],
        index,
        block_count,
    )
    if technical_smoke:
        primary["registered_support"] = False
        primary["support_suppressed_reason"] = "technical_smoke_not_accuracy_evidence"
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "complete",
        "scientific_role": run["scientific_role"],
        "created_at_utc": created_at_utc or datetime.now(timezone.utc).isoformat(),
        "design_manifest_sha256": validation["manifest_sha256"],
        "protocol_sha256": validation["protocol_sha256"],
        "source_lock_sha256": validation["source_lock"].get("sha256"),
        "internal_runtime": internal_runtime,
        "run_root": str(root),
        "run_checksum_sha256": run["checksum_closure"]["sha256"],
        "block_count": block_count,
        "independent_unit": "truth_block",
        "primary": primary,
        "secondary_standard_contrasts": _secondary_standard_contrasts(
            index, block_count
        ),
        "secondary_biopsy_count": _biopsy_count_analysis(index, block_count),
        "secondary_placement": _placement_analysis(index, block_count),
        "secondary_medicc2": medicc2,
        "absolute_standard_scores": _absolute_scores(index, block_count),
        "resources_and_failures": _resource_and_failure_summary(index),
        "interpretation_contract": {
            "automatic_winner_declared": False,
            "partial_and_full_families_combined": False,
            "secondary_binary_support_declared": False,
            "medicc2_is_separate_topology_only_context": True,
            "model_mismatch_generalization": False,
            "ktn152_accuracy_claim": False,
        },
    }
    return report, primary_rows


def _write_primary_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "block_index",
        *STANDARD_CONDITION_IDS,
        "block_effect",
        "complete",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "block_index": int(row["block_index"]) + 1,
                    **row["effects_by_condition"],
                    "block_effect": row["block_effect"],
                    "complete": row["complete"],
                }
            )


def _markdown(report: Mapping[str, Any]) -> str:
    primary = report["primary"]
    summary = primary["block_effect_summary"]
    interval = primary["bootstrap_interval"]
    sign_flip = primary["sign_flip"]
    status = "supported" if primary["registered_support"] else "not supported / inconclusive"
    lines = [
        "# CTBF V5 Independent Confirmation",
        "",
        f"Scientific role: `{report['scientific_role']}`.",
        "",
        "## Primary result",
        "",
        "M2 minus M0 on rooted GRF, averaging H14/H24/H34 Rule-Y effects "
        "within each independent truth block.",
        "",
        f"Registered conclusion: **{status}**.",
        "",
        f"- Complete blocks: {primary['complete_block_count']}/{report['block_count']}.",
        f"- Mean block effect: {summary['mean']}.",
        f"- Median block effect: {summary['median']}.",
        f"- 95% block-bootstrap interval: [{interval.get('lower')}, {interval.get('upper')}].",
        f"- One-sided block sign-flip p-value: {sign_flip.get('p_value')}.",
        f"- Block W/T/L: {primary['block_wins_ties_losses']['wins']}/"
        f"{primary['block_wins_ties_losses']['ties']}/"
        f"{primary['block_wins_ties_losses']['losses']}.",
        "",
        "All secondary contrasts are descriptive and cannot replace the sole "
        "primary test. Partial and fully labeled methods remain separate.",
        "",
    ]
    return "\n".join(lines)


def write_report(
    *,
    run_root: Path | str,
    medicc2_root: Path | str | None = None,
    output_root: Path | str,
    manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
    technical_smoke: bool = False,
) -> dict[str, Any]:
    report, rows = build_report(
        run_root=run_root,
        medicc2_root=medicc2_root,
        manifest_path=manifest_path,
        technical_smoke=technical_smoke,
    )
    root = ensure_new_output_root(output_root)
    write_json(root / "summary.json", report)
    _write_primary_csv(root / "primary_block_effects.csv", rows)
    (root / "report.md").write_text(_markdown(report), encoding="utf-8")
    write_checksum_closure(root, REPORT_CHECKSUM_NAME)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--medicc2-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--technical-smoke", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    report = write_report(
        run_root=arguments.run_root,
        medicc2_root=arguments.medicc2_root,
        output_root=arguments.output_root,
        manifest_path=arguments.manifest,
        technical_smoke=arguments.technical_smoke,
    )
    primary = report["primary"]
    print(
        "complete: primary "
        f"{'supported' if primary['registered_support'] else 'not supported/inconclusive'}; "
        f"{primary['complete_block_count']}/{report['block_count']} complete blocks"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "REPORT_CHECKSUM_NAME",
    "build_report",
    "main",
    "write_report",
]
