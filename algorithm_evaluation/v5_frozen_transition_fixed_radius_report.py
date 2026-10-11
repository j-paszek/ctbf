"""Compare corrected adaptive methods with fixed-radius counterparts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from algorithm_evaluation.paper_pipeline_contract import read_json
from algorithm_evaluation.v5_frozen_transition_development_report import (
    AFFECTED_ARM_IDS,
    ContrastSpec,
    REGIME_IDS,
    REGIME_LABELS,
    ResultBundle,
    TIE_TOLERANCE,
    _complete_block_means,
    _effect_summary,
    _flatten_rows,
    _format_effect,
    _height_values,
    _load_current_bundle,
    _load_historical_bundle,
    _metric_values,
    _paired_shift,
    _result_records,
    _retention_status,
    _write_csv,
)
from algorithm_evaluation.v5_shortlist_robustness_common import (
    ADAPTIVE_A_PRIME_ID,
    ADAPTIVE_B_PRIME_ID,
    CURRENT_PAPER_DEVELOPMENT_ARM_IDS,
    DEFAULT_BLOCK_COUNT,
    FIXED_RADIUS_CORRECTION_ARM_IDS,
    ORDERED_A_ID,
    ORDERED_B_ID,
    PARTIAL_ADAPTIVE_Y_PRIME_ID,
    PARTIAL_Y_ID,
    PLACEMENT_POLICIES,
    RESULT_NAME,
    RUN_SCHEMA_VERSION,
    SHORT_LABEL_BY_ARM,
    ensure_new_output_root,
    write_json,
)
from reconstructor_biopsy_blocks import (
    BIOPSY_GUIDED_AUDIT_SCHEMA_VERSION,
    FROZEN_TRANSITION_PARENT_ELIGIBILITY_POLICY,
)


REPORT_SCHEMA_VERSION = (
    "ctbf-v5-frozen-transition-fixed-radius-verification-v1"
)
FIXED_RADIUS_ARM_SET = "fixed-radius-correction"
CURRENT_NINE_ARM_IDS = (
    *CURRENT_PAPER_DEVELOPMENT_ARM_IDS,
    *FIXED_RADIUS_CORRECTION_ARM_IDS,
)
if len(CURRENT_NINE_ARM_IDS) != 9 or len(set(CURRENT_NINE_ARM_IDS)) != 9:
    raise RuntimeError("The corrected counterpart roster must contain nine arms.")

CONTRAST_SPECS = (
    ContrastSpec(
        "A_prime_minus_A",
        "fully_labeled",
        ADAPTIVE_A_PRIME_ID,
        ORDERED_A_ID,
        ("ad_f1", "grf"),
        "ad_f1",
    ),
    ContrastSpec(
        "B_prime_minus_B",
        "fully_labeled",
        ADAPTIVE_B_PRIME_ID,
        ORDERED_B_ID,
        ("ad_f1", "grf"),
        "ad_f1",
    ),
    ContrastSpec(
        "Y_prime_minus_Y",
        "partial",
        PARTIAL_ADAPTIVE_Y_PRIME_ID,
        PARTIAL_Y_ID,
        ("grf",),
        "grf",
    ),
)


def _validate_fixed_radius_audits(
    records: Mapping[tuple[int, int, str, str], Mapping[str, Any]],
) -> None:
    for key, record in records.items():
        metadata = record.get("reconstruction_metadata")
        audit = (
            metadata.get("biopsy_layer_decision_audit")
            if isinstance(metadata, Mapping)
            else None
        )
        if not isinstance(audit, Mapping):
            raise ValueError(
                f"Corrected fixed-radius record {key} has no decision audit."
            )
        if audit.get("schema_version") != BIOPSY_GUIDED_AUDIT_SCHEMA_VERSION:
            raise ValueError(
                f"Corrected fixed-radius record {key} has an old audit schema."
            )
        if (
            audit.get("parent_eligibility_policy")
            != FROZEN_TRANSITION_PARENT_ELIGIBILITY_POLICY
        ):
            raise ValueError(
                f"Corrected fixed-radius record {key} has the wrong policy."
            )


def _load_fixed_radius_bundle(
    regime_id: str,
    result_root: Path | str,
    *,
    base_current: ResultBundle,
) -> ResultBundle:
    root = Path(result_root).expanduser().resolve()
    result = read_json(root / RESULT_NAME)
    if result.get("schema_version") != RUN_SCHEMA_VERSION:
        raise ValueError("Corrected fixed-radius run schema changed.")
    expected = {
        "status": "complete",
        "arm_set": FIXED_RADIUS_ARM_SET,
        "arm_ids": list(FIXED_RADIUS_CORRECTION_ARM_IDS),
        "bank_root": str(base_current.bank_root),
        "block_count": DEFAULT_BLOCK_COUNT,
        "failure_count": 0,
    }
    for field, value in expected.items():
        if result.get(field) != value:
            raise ValueError(
                f"Corrected fixed-radius result field {field} changed."
            )
    gates = result.get("semantic_gate_by_arm")
    if (
        not isinstance(gates, Mapping)
        or set(gates) != set(FIXED_RADIUS_CORRECTION_ARM_IDS)
        or any(
            not isinstance(gate, Mapping) or gate.get("status") != "passed"
            for gate in gates.values()
        )
    ):
        raise ValueError("Corrected fixed-radius semantic gate did not pass.")
    records = _result_records(
        result,
        required_arm_ids=FIXED_RADIUS_CORRECTION_ARM_IDS,
    )
    expected_count = (
        len(base_current.bank["cases"])
        * len(FIXED_RADIUS_CORRECTION_ARM_IDS)
    )
    if len(records) != expected_count:
        raise ValueError(
            "Corrected fixed-radius record inventory is incomplete."
        )
    _validate_fixed_radius_audits(records)
    return ResultBundle(
        regime_id=regime_id,
        result_roots=(root,),
        run_ids=(str(result["run_id"]),),
        bank_root=base_current.bank_root,
        bank=base_current.bank,
        records=records,
        historical=False,
    )


def _merge_current(
    base: ResultBundle,
    fixed_radius: ResultBundle,
) -> ResultBundle:
    if (
        base.regime_id != fixed_radius.regime_id
        or base.bank_root != fixed_radius.bank_root
    ):
        raise ValueError(
            "Corrected base and fixed-radius results use different banks."
        )
    records = dict(base.records)
    overlap = set(records).intersection(fixed_radius.records)
    if overlap:
        raise ValueError(
            "Corrected fixed-radius records overlap the six-arm base run."
        )
    records.update(fixed_radius.records)
    expected = len(base.bank["cases"]) * len(CURRENT_NINE_ARM_IDS)
    if len(records) != expected:
        raise ValueError("Corrected nine-arm record inventory is incomplete.")
    return ResultBundle(
        regime_id=base.regime_id,
        result_roots=(*base.result_roots, *fixed_radius.result_roots),
        run_ids=(*base.run_ids, *fixed_radius.run_ids),
        bank_root=base.bank_root,
        bank=base.bank,
        records=records,
        historical=False,
    )


def _contrast_values(
    bundle: ResultBundle,
    *,
    spec: ContrastSpec,
    metric: str,
) -> dict[tuple[int, int, str], float]:
    left = _metric_values(bundle, arm_id=spec.left_arm_id, metric=metric)
    right = _metric_values(bundle, arm_id=spec.right_arm_id, metric=metric)
    if set(left) != set(right):
        raise ValueError(
            f"Fixed-radius contrast {spec.contrast_id} is not paired."
        )
    return {
        coordinate: left[coordinate] - right[coordinate]
        for coordinate in left
    }


def _contrast_rows(
    current: ResultBundle,
    historical: ResultBundle,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    block_rows = []
    placement_rows = []
    for spec in CONTRAST_SPECS:
        for metric in spec.metrics:
            current_values = _contrast_values(
                current,
                spec=spec,
                metric=metric,
            )
            historical_values = _contrast_values(
                historical,
                spec=spec,
                metric=metric,
            )
            for raw_height in current.bank["heights"]:
                height = int(raw_height)
                current_blocks = _complete_block_means(
                    current_values,
                    height=height,
                )
                historical_blocks = _complete_block_means(
                    historical_values,
                    height=height,
                )
                shifts = _paired_shift(current_blocks, historical_blocks)
                current_summary = _effect_summary(
                    list(current_blocks.values())
                )
                historical_summary = _effect_summary(
                    list(historical_blocks.values())
                )
                common = {
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
                    "primary_metric": spec.primary_metric,
                    "is_primary_metric": metric == spec.primary_metric,
                }
                block_rows.append(
                    {
                        **common,
                        "paired_block_count": len(current_blocks),
                        "historical": historical_summary,
                        "current": current_summary,
                        "correction_shift": _effect_summary(
                            list(shifts.values())
                        ),
                        "retention_status": _retention_status(
                            historical_summary["mean"],
                            current_summary["mean"],
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
                    cell_shifts = _paired_shift(
                        current_cell,
                        historical_cell,
                    )
                    placement_rows.append(
                        {
                            **common,
                            "placement_policy": policy,
                            "paired_condition_count": len(current_cell),
                            "historical": _effect_summary(
                                list(historical_cell.values())
                            ),
                            "current": _effect_summary(
                                list(current_cell.values())
                            ),
                            "correction_shift": _effect_summary(
                                list(cell_shifts.values())
                            ),
                        }
                    )
    return block_rows, placement_rows


def _score_change_rows(
    current: ResultBundle,
    historical: ResultBundle,
) -> list[dict[str, Any]]:
    rows = []
    for spec in CONTRAST_SPECS:
        for arm_role, arm_id in (
            ("adaptive", spec.left_arm_id),
            ("fixed", spec.right_arm_id),
        ):
            for metric in spec.metrics:
                current_values = _metric_values(
                    current,
                    arm_id=arm_id,
                    metric=metric,
                )
                historical_values = _metric_values(
                    historical,
                    arm_id=arm_id,
                    metric=metric,
                )
                shifts = _paired_shift(current_values, historical_values)
                for raw_height in current.bank["heights"]:
                    height = int(raw_height)
                    block_shifts = _complete_block_means(
                        shifts,
                        height=height,
                    )
                    rows.append(
                        {
                            "regime_id": current.regime_id,
                            "regime_label": REGIME_LABELS[
                                current.regime_id
                            ],
                            "height": height,
                            "family": spec.family,
                            "contrast_id": spec.contrast_id,
                            "arm_role": arm_role,
                            "arm_id": arm_id,
                            "short_label": SHORT_LABEL_BY_ARM[arm_id],
                            "metric": metric,
                            "paired_block_count": len(block_shifts),
                            "current_minus_historical": _effect_summary(
                                list(block_shifts.values())
                            ),
                        }
                    )
    return rows


def build_report(
    *,
    base_current_result_roots: Mapping[str, Path | str],
    fixed_radius_result_roots: Mapping[str, Path | str],
    historical_result_roots: Mapping[str, Sequence[Path | str]],
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    for name, roots in (
        ("base current", base_current_result_roots),
        ("fixed radius", fixed_radius_result_roots),
        ("historical", historical_result_roots),
    ):
        if set(roots) != set(REGIME_IDS):
            raise ValueError(
                f"{name} roots must cover exactly v2/v2a/v2b/v2c."
            )

    source_inventory = []
    block_rows = []
    placement_rows = []
    score_rows = []
    for regime_id in REGIME_IDS:
        base = _load_current_bundle(
            regime_id,
            base_current_result_roots[regime_id],
        )
        fixed_radius = _load_fixed_radius_bundle(
            regime_id,
            fixed_radius_result_roots[regime_id],
            base_current=base,
        )
        current = _merge_current(base, fixed_radius)
        historical = _load_historical_bundle(
            regime_id,
            historical_result_roots[regime_id],
            current=current,
            required_arm_ids=CURRENT_NINE_ARM_IDS,
            current_policy_arm_ids=(
                *AFFECTED_ARM_IDS,
                *FIXED_RADIUS_CORRECTION_ARM_IDS,
            ),
        )
        if set(current.records) != set(historical.records):
            raise ValueError(
                "Current and historical fixed-radius keys differ for "
                f"{regime_id}."
            )
        source_inventory.append(
            {
                "regime_id": regime_id,
                "regime_label": REGIME_LABELS[regime_id],
                "bank_root": str(current.bank_root),
                "available_condition_count": len(current.coordinates),
                "base_current_result_roots": [
                    str(root) for root in base.result_roots
                ],
                "fixed_radius_result_roots": [
                    str(root) for root in fixed_radius.result_roots
                ],
                "historical_result_roots": [
                    str(root) for root in historical.result_roots
                ],
                "base_current_run_ids": list(base.run_ids),
                "fixed_radius_run_ids": list(fixed_radius.run_ids),
                "historical_run_ids": list(historical.run_ids),
            }
        )
        regime_blocks, regime_placements = _contrast_rows(
            current,
            historical,
        )
        block_rows.extend(regime_blocks)
        placement_rows.extend(regime_placements)
        score_rows.extend(_score_change_rows(current, historical))

    primary_rows = [row for row in block_rows if row["is_primary_metric"]]
    historical_positive_primary = [
        row
        for row in primary_rows
        if row["historical"]["mean"] > TIE_TOLERANCE
    ]
    current_positive_primary = [
        row for row in primary_rows if row["current"]["mean"] > TIE_TOLERANCE
    ]
    historical_positive_all = [
        row
        for row in block_rows
        if row["historical"]["mean"] > TIE_TOLERANCE
    ]
    current_positive_all = [
        row for row in block_rows if row["current"]["mean"] > TIE_TOLERANCE
    ]
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "complete",
        "created_at_utc": created_at_utc
        or datetime.now(timezone.utc).isoformat(),
        "scientific_role": (
            "post_correction_fixed_radius_development_verification_not_"
            "held_out_paper_confirmation"
        ),
        "decision_audit_schema": BIOPSY_GUIDED_AUDIT_SCHEMA_VERSION,
        "parent_eligibility_policy": (
            FROZEN_TRANSITION_PARENT_ELIGIBILITY_POLICY
        ),
        "source_inventory": source_inventory,
        "comparison_contract": {
            "contrast_directions": [
                spec.contrast_id for spec in CONTRAST_SPECS
            ],
            "positive_means_adaptive_radius_better": True,
            "primary_metric_by_family": {
                "fully_labeled": "ad_f1",
                "partial": "grf",
            },
            "independent_unit": "truth_block",
            "block_effect": (
                "mean_of_three_placement_specific_paired_effects"
            ),
            "condition_wtl_is_descriptive": True,
            "historical_records_are_superseded_provenance_only": True,
            "automatic_winner_declared": False,
            "output_families_pooled": False,
            "heights_pooled": False,
            "regimes_pooled": False,
        },
        "adaptive_minus_fixed_by_height": block_rows,
        "adaptive_minus_fixed_by_placement": placement_rows,
        "method_score_changes": score_rows,
        "verification_inventory": {
            "primary_contrast_cell_count": len(primary_rows),
            "all_metric_contrast_cell_count": len(block_rows),
            "historical_positive_primary_cell_count": len(
                historical_positive_primary
            ),
            "historical_positive_primary_retained_count": sum(
                row["current"]["mean"] > TIE_TOLERANCE
                for row in historical_positive_primary
            ),
            "historical_positive_primary_not_retained_count": sum(
                row["current"]["mean"] <= TIE_TOLERANCE
                for row in historical_positive_primary
            ),
            "current_positive_primary_cell_count": len(
                current_positive_primary
            ),
            "historical_positive_all_metric_cell_count": len(
                historical_positive_all
            ),
            "current_positive_all_metric_cell_count": len(
                current_positive_all
            ),
        },
    }


def _append_effect_table(
    lines: list[str],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    lines.extend(
        [
            "| Regime | Height | Family | Contrast | Metric | Historical | "
            "Corrected | Correction shift | Status |",
            "|---|---:|---|---|---|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['regime_label']} | H{row['height']} | "
            f"{row['family']} | {row['left_label']}−{row['right_label']} | "
            f"{row['metric']} | {_format_effect(row['historical'])} | "
            f"{_format_effect(row['current'])} | "
            f"{_format_effect(row['correction_shift'])} | "
            f"{row['retention_status']} |"
        )


def _markdown(report: Mapping[str, Any]) -> str:
    inventory = report["verification_inventory"]
    rows = report["adaptive_minus_fixed_by_height"]
    primary_rows = [row for row in rows if row["is_primary_metric"]]
    secondary_rows = [row for row in rows if not row["is_primary_metric"]]
    lines = [
        "# Corrected adaptive versus fixed-radius reconstruction",
        "",
        f"Schema: `{report['schema_version']}`  ",
        f"Decision audit: `{report['decision_audit_schema']}`  ",
        f"Parent policy: `{report['parent_eligibility_policy']}`",
        "",
        "Positive A'−A, B'−B, or Y'−Y means that the adaptive-radius "
        "method scores higher. W/T/L uses one mean three-placement effect "
        "per complete truth block. Fully labeled and partial outputs remain "
        "separate. This is paired development evidence, not held-out "
        "confirmation.",
        "",
        "## Primary paired effects",
        "",
    ]
    _append_effect_table(lines, primary_rows)
    lines.extend(
        [
            "",
            "## Secondary fully labeled GRF effects",
            "",
        ]
    )
    _append_effect_table(lines, secondary_rows)
    lines.extend(
        [
            "",
            "## Verification inventory",
            "",
            "- Primary regime-height contrast cells: "
            f"{inventory['primary_contrast_cell_count']}.",
            "- Historical positive primary cells retained: "
            f"{inventory['historical_positive_primary_retained_count']}/"
            f"{inventory['historical_positive_primary_cell_count']}.",
            "- Historical positive primary cells not retained: "
            f"{inventory['historical_positive_primary_not_retained_count']}.",
            "- Corrected positive primary cells: "
            f"{inventory['current_positive_primary_cell_count']}/"
            f"{inventory['primary_contrast_cell_count']}.",
            "",
            "Placement-specific effects and separate fixed/adaptive method "
            "score changes are stored in the adjacent CSV and JSON files. "
            "No pooled winner is declared.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(
    *,
    base_current_result_roots: Mapping[str, Path | str],
    fixed_radius_result_roots: Mapping[str, Path | str],
    historical_result_roots: Mapping[str, Sequence[Path | str]],
    output_root: Path | str,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    report = build_report(
        base_current_result_roots=base_current_result_roots,
        fixed_radius_result_roots=fixed_radius_result_roots,
        historical_result_roots=historical_result_roots,
        created_at_utc=created_at_utc,
    )
    root = ensure_new_output_root(output_root)
    write_json(root / "summary.json", report)
    _write_csv(
        root / "adaptive_minus_fixed_by_height.csv",
        _flatten_rows(
            report["adaptive_minus_fixed_by_height"],
            ("historical", "current", "correction_shift"),
        ),
    )
    _write_csv(
        root / "adaptive_minus_fixed_by_placement.csv",
        _flatten_rows(
            report["adaptive_minus_fixed_by_placement"],
            ("historical", "current", "correction_shift"),
        ),
    )
    _write_csv(
        root / "method_score_changes.csv",
        _flatten_rows(
            report["method_score_changes"],
            ("current_minus_historical",),
        ),
    )
    (root / "report.md").write_text(_markdown(report), encoding="utf-8")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for label in ("v2", "v2a", "v2b", "v2c"):
        parser.add_argument(
            f"--{label}-base-current-result-root",
            type=Path,
            required=True,
        )
        parser.add_argument(
            f"--{label}-fixed-radius-result-root",
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
    base_current = {
        regime_id: getattr(
            arguments,
            f"{REGIME_LABELS[regime_id]}_base_current_result_root",
        )
        for regime_id in REGIME_IDS
    }
    fixed_radius = {
        regime_id: getattr(
            arguments,
            f"{REGIME_LABELS[regime_id]}_fixed_radius_result_root",
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
        base_current_result_roots=base_current,
        fixed_radius_result_roots=fixed_radius,
        historical_result_roots=historical,
        output_root=arguments.output_root,
    )
    inventory = report["verification_inventory"]
    print(
        "complete: "
        f"{inventory['primary_contrast_cell_count']} primary "
        "adaptive-minus-fixed cells; "
        f"{inventory['historical_positive_primary_retained_count']}/"
        f"{inventory['historical_positive_primary_cell_count']} historical "
        "positive primary cells retained"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
