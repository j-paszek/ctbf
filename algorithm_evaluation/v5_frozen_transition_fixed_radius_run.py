"""Run the corrected fixed-radius A/B/Y extension on all preserved banks."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from algorithm_evaluation.paper_pipeline_contract import read_json
from algorithm_evaluation.v5_algorithm_development_run import (
    DEFAULT_EVALUATION_TIMEOUT_SECONDS,
    DEFAULT_RECONSTRUCTION_TIMEOUT_SECONDS,
    DEFAULT_RSS_LIMIT_BYTES,
)
from algorithm_evaluation.v5_frozen_transition_development_report import (
    AFFECTED_ARM_IDS,
    ResultBundle,
    _load_current_bundle,
    _load_historical_bundle,
)
from algorithm_evaluation.v5_frozen_transition_development_run import (
    BANK_RELATIVE_PATHS,
    DEFAULT_EXPERIMENT_ROOT,
    DEFAULT_OUTPUT_DIRECTORY,
    HISTORICAL_RESULT_RELATIVE_PATHS,
    REGIME_IDS,
    REGIME_LABELS,
)
from algorithm_evaluation.v5_frozen_transition_fixed_radius_report import (
    CURRENT_NINE_ARM_IDS,
    REPORT_SCHEMA_VERSION as FIXED_RADIUS_REPORT_SCHEMA_VERSION,
    _load_fixed_radius_bundle,
    write_report as write_fixed_radius_report,
)
from algorithm_evaluation.v5_shortlist_robustness_common import (
    DEFAULT_BLOCK_COUNT,
    FIXED_RADIUS_CORRECTION_ARM_IDS,
    RESULT_NAME,
    write_json,
)
from algorithm_evaluation.v5_shortlist_robustness_report import (
    REPORT_SCHEMA_VERSION as WITHIN_REPORT_SCHEMA_VERSION,
    write_report as write_within_regime_report,
)
from algorithm_evaluation.v5_shortlist_robustness_run import (
    MAX_RECORD_WORKERS,
    run_shortlist,
)
from reconstructor_biopsy_blocks import (
    BIOPSY_GUIDED_AUDIT_SCHEMA_VERSION,
    FROZEN_TRANSITION_PARENT_ELIGIBILITY_POLICY,
)


PIPELINE_SCHEMA_VERSION = (
    "ctbf-v5-frozen-transition-fixed-radius-pipeline-v1"
)
FIXED_RADIUS_ARM_SET = "fixed-radius-correction"
DEFAULT_EXTENSION_DIRECTORY = "fixed_radius"
PIPELINE_MANIFEST_NAME = "pipeline_manifest.json"


def _resolved_paths(
    experiment_root: Path | str,
    corrected_root: Path | str | None,
    output_root: Path | str | None,
) -> tuple[Path, Path, Path]:
    experiment = Path(experiment_root).expanduser().resolve()
    corrected = (
        experiment / DEFAULT_OUTPUT_DIRECTORY
        if corrected_root is None
        else Path(corrected_root).expanduser().resolve()
    )
    output = (
        corrected / DEFAULT_EXTENSION_DIRECTORY
        if output_root is None
        else Path(output_root).expanduser().resolve()
    )
    return experiment, corrected, output


def _source_inventory(
    *,
    experiment_root: Path,
    corrected_root: Path,
) -> tuple[
    list[dict[str, Any]],
    dict[str, ResultBundle],
    dict[str, tuple[Path, ...]],
]:
    inventory = []
    base_bundles = {}
    historical_roots = {}
    for regime_id in REGIME_IDS:
        label = REGIME_LABELS[regime_id]
        base_root = (corrected_root / label / "run").resolve()
        base = _load_current_bundle(regime_id, base_root)
        roots = tuple(
            (experiment_root / relative).resolve()
            for relative in HISTORICAL_RESULT_RELATIVE_PATHS[regime_id]
        )
        historical = _load_historical_bundle(
            regime_id,
            roots,
            current=base,
            required_arm_ids=CURRENT_NINE_ARM_IDS,
            current_policy_arm_ids=(
                *AFFECTED_ARM_IDS,
                *FIXED_RADIUS_CORRECTION_ARM_IDS,
            ),
        )
        base_bundles[regime_id] = base
        historical_roots[regime_id] = roots
        inventory.append(
            {
                "regime_id": regime_id,
                "regime_label": label,
                "bank_root": str(base.bank_root),
                "available_condition_count": len(base.coordinates),
                "expected_fixed_radius_record_count": (
                    len(base.coordinates)
                    * len(FIXED_RADIUS_CORRECTION_ARM_IDS)
                ),
                "base_current_run_ids": list(base.run_ids),
                "historical_run_ids": list(historical.run_ids),
            }
        )
    return inventory, base_bundles, historical_roots


def _manifest_contract(
    *,
    experiment_root: Path,
    corrected_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    return {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "scientific_role": (
            "post_correction_fixed_radius_development_verification_not_"
            "held_out_paper_confirmation"
        ),
        "experiment_root": str(experiment_root),
        "corrected_six_arm_root": str(corrected_root),
        "output_root": str(output_root),
        "arm_set": FIXED_RADIUS_ARM_SET,
        "arm_ids": list(FIXED_RADIUS_CORRECTION_ARM_IDS),
        "decision_audit_schema": BIOPSY_GUIDED_AUDIT_SCHEMA_VERSION,
        "parent_eligibility_policy": (
            FROZEN_TRANSITION_PARENT_ELIGIBILITY_POLICY
        ),
        "banks_reused_without_regeneration": True,
        "corrected_adaptive_counterparts_reused_without_reconstruction": True,
        "historical_results_read_only": True,
        "regimes": list(REGIME_IDS),
    }


def _load_or_create_manifest(
    *,
    experiment_root: Path,
    corrected_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    contract = _manifest_contract(
        experiment_root=experiment_root,
        corrected_root=corrected_root,
        output_root=output_root,
    )
    manifest_path = output_root / PIPELINE_MANIFEST_NAME
    if output_root.exists():
        if not output_root.is_dir() or not manifest_path.is_file():
            raise ValueError(
                "Existing fixed-radius root has no pipeline manifest."
            )
        manifest = read_json(manifest_path)
        for field, expected in contract.items():
            if manifest.get(field) != expected:
                raise ValueError(
                    f"Stored fixed-radius pipeline field {field} changed."
                )
        if manifest.get("status") not in {"in_progress", "complete", "failure"}:
            raise ValueError("Stored fixed-radius pipeline status is invalid.")
        return manifest
    output_root.mkdir(parents=True)
    manifest = {
        **contract,
        "status": "in_progress",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed_regimes": [],
    }
    write_json(manifest_path, manifest)
    return manifest


def _validate_completed_run(
    result: Mapping[str, Any],
    *,
    bank_root: Path,
    run_id: str,
) -> None:
    expected = {
        "status": "complete",
        "run_id": run_id,
        "arm_set": FIXED_RADIUS_ARM_SET,
        "arm_ids": list(FIXED_RADIUS_CORRECTION_ARM_IDS),
        "bank_root": str(bank_root),
        "block_count": DEFAULT_BLOCK_COUNT,
        "failure_count": 0,
    }
    for field, value in expected.items():
        if result.get(field) != value:
            raise ValueError(
                f"Completed fixed-radius run field {field} changed."
            )


def _run_or_resume(
    *,
    regime_id: str,
    bank_root: Path,
    run_root: Path,
    record_workers: int,
    reconstruction_timeout_seconds: int,
    evaluation_timeout_seconds: int,
    rss_limit_bytes: int,
    progress: bool,
) -> Mapping[str, Any]:
    run_id = (
        f"shortlist-{REGIME_LABELS[regime_id]}-fixed-radius-"
        "frozen-transition-v3"
    )
    result_path = run_root / RESULT_NAME
    resume = False
    if run_root.exists():
        if not run_root.is_dir() or not result_path.is_file():
            raise ValueError(
                f"Existing fixed-radius run root is incomplete: {run_root}."
            )
        result = read_json(result_path)
        if result.get("status") == "complete":
            _validate_completed_run(result, bank_root=bank_root, run_id=run_id)
            return result
        resume = True
    result = run_shortlist(
        bank_root=bank_root,
        output_root=run_root,
        run_id=run_id,
        reconstruction_timeout_seconds=reconstruction_timeout_seconds,
        evaluation_timeout_seconds=evaluation_timeout_seconds,
        rss_limit_bytes=rss_limit_bytes,
        record_workers=record_workers,
        arm_set=FIXED_RADIUS_ARM_SET,
        expected_block_count=DEFAULT_BLOCK_COUNT,
        progress=progress,
        resume=resume,
    )
    _validate_completed_run(result, bank_root=bank_root, run_id=run_id)
    return result


def _write_or_validate_within_report(
    *,
    base_root: Path,
    fixed_radius_root: Path,
    output_root: Path,
) -> Mapping[str, Any]:
    summary_path = output_root / "summary.json"
    if output_root.exists():
        required = (summary_path, output_root / "report.md")
        if not output_root.is_dir() or any(not path.is_file() for path in required):
            raise ValueError(
                f"Existing fixed-radius report is incomplete: {output_root}."
            )
        report = read_json(summary_path)
        if report.get("schema_version") != WITHIN_REPORT_SCHEMA_VERSION:
            raise ValueError(
                "Stored fixed-radius within-report schema changed."
            )
        if set(report.get("shortlist_arm_ids", ())) != set(
            CURRENT_NINE_ARM_IDS
        ):
            raise ValueError("Stored fixed-radius within-report roster changed.")
        if report.get("result_roots") != [
            str(base_root.resolve()),
            str(fixed_radius_root.resolve()),
        ]:
            raise ValueError(
                "Stored fixed-radius within-report sources changed."
            )
        if report.get("status") != "complete":
            raise ValueError(
                "Stored fixed-radius within-report is not complete."
            )
        return report
    return write_within_regime_report(
        result_roots=(base_root, fixed_radius_root),
        output_root=output_root,
        expected_block_count=DEFAULT_BLOCK_COUNT,
    )


def _write_or_validate_verification(
    *,
    base_roots: Mapping[str, Path],
    fixed_radius_roots: Mapping[str, Path],
    historical_roots: Mapping[str, Sequence[Path]],
    output_root: Path,
) -> Mapping[str, Any]:
    summary_path = output_root / "summary.json"
    if output_root.exists():
        required = (
            summary_path,
            output_root / "report.md",
            output_root / "adaptive_minus_fixed_by_height.csv",
            output_root / "adaptive_minus_fixed_by_placement.csv",
            output_root / "method_score_changes.csv",
        )
        if not output_root.is_dir() or any(not path.is_file() for path in required):
            raise ValueError(
                "Existing fixed-radius verification is incomplete: "
                f"{output_root}."
            )
        report = read_json(summary_path)
        if report.get("schema_version") != FIXED_RADIUS_REPORT_SCHEMA_VERSION:
            raise ValueError(
                "Stored fixed-radius verification schema changed."
            )
        if report.get("status") != "complete":
            raise ValueError(
                "Stored fixed-radius verification is not complete."
            )
        inventory = {
            str(row["regime_id"]): row
            for row in report.get("source_inventory", ())
            if isinstance(row, Mapping)
        }
        if set(inventory) != set(REGIME_IDS):
            raise ValueError(
                "Stored fixed-radius verification regimes changed."
            )
        for regime_id in REGIME_IDS:
            observed = inventory[regime_id]
            expected = {
                "base_current_result_roots": [
                    str(base_roots[regime_id].resolve())
                ],
                "fixed_radius_result_roots": [
                    str(fixed_radius_roots[regime_id].resolve())
                ],
                "historical_result_roots": [
                    str(root.resolve()) for root in historical_roots[regime_id]
                ],
            }
            for field, value in expected.items():
                if observed.get(field) != value:
                    raise ValueError(
                        f"Stored fixed-radius verification {field} changed."
                    )
        return report
    return write_fixed_radius_report(
        base_current_result_roots=base_roots,
        fixed_radius_result_roots=fixed_radius_roots,
        historical_result_roots=historical_roots,
        output_root=output_root,
    )


def run_pipeline(
    *,
    experiment_root: Path | str = DEFAULT_EXPERIMENT_ROOT,
    corrected_root: Path | str | None = None,
    output_root: Path | str | None = None,
    record_workers: int = MAX_RECORD_WORKERS,
    reconstruction_timeout_seconds: int = DEFAULT_RECONSTRUCTION_TIMEOUT_SECONDS,
    evaluation_timeout_seconds: int = DEFAULT_EVALUATION_TIMEOUT_SECONDS,
    rss_limit_bytes: int = DEFAULT_RSS_LIMIT_BYTES,
    progress: bool = False,
) -> dict[str, Any]:
    if (
        isinstance(record_workers, bool)
        or not 1 <= int(record_workers) <= MAX_RECORD_WORKERS
    ):
        raise ValueError(f"record_workers must be between 1 and {MAX_RECORD_WORKERS}.")
    experiment, corrected, output = _resolved_paths(
        experiment_root,
        corrected_root,
        output_root,
    )
    source_inventory, base_bundles, historical_roots = _source_inventory(
        experiment_root=experiment,
        corrected_root=corrected,
    )
    manifest = _load_or_create_manifest(
        experiment_root=experiment,
        corrected_root=corrected,
        output_root=output,
    )
    manifest_path = output / PIPELINE_MANIFEST_NAME
    manifest["status"] = "in_progress"
    manifest["source_inventory"] = source_inventory
    manifest.pop("failure", None)
    manifest.pop("completed_at_utc", None)
    write_json(manifest_path, manifest)

    base_roots = {}
    fixed_radius_roots = {}
    try:
        for regime_id in REGIME_IDS:
            label = REGIME_LABELS[regime_id]
            base_root = base_bundles[regime_id].result_roots[0]
            run_root = (output / label / "run").resolve()
            report_root = (output / label / "report").resolve()
            _run_or_resume(
                regime_id=regime_id,
                bank_root=(experiment / BANK_RELATIVE_PATHS[regime_id]).resolve(),
                run_root=run_root,
                record_workers=int(record_workers),
                reconstruction_timeout_seconds=reconstruction_timeout_seconds,
                evaluation_timeout_seconds=evaluation_timeout_seconds,
                rss_limit_bytes=rss_limit_bytes,
                progress=progress,
            )
            _load_fixed_radius_bundle(
                regime_id,
                run_root,
                base_current=base_bundles[regime_id],
            )
            _write_or_validate_within_report(
                base_root=base_root,
                fixed_radius_root=run_root,
                output_root=report_root,
            )
            base_roots[regime_id] = base_root
            fixed_radius_roots[regime_id] = run_root
            completed = manifest.setdefault("completed_regimes", [])
            if regime_id not in completed:
                completed.append(regime_id)
                write_json(manifest_path, manifest)

        verification_root = (output / "verification").resolve()
        verification = _write_or_validate_verification(
            base_roots=base_roots,
            fixed_radius_roots=fixed_radius_roots,
            historical_roots=historical_roots,
            output_root=verification_root,
        )
    except BaseException as error:
        manifest["status"] = "failure"
        manifest["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        write_json(manifest_path, manifest)
        raise

    manifest["status"] = "complete"
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["verification_report_root"] = str(verification_root)
    manifest["verification_inventory"] = verification["verification_inventory"]
    write_json(manifest_path, manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=DEFAULT_EXPERIMENT_ROOT,
    )
    parser.add_argument(
        "--corrected-root",
        type=Path,
        help="Defaults to frozen_transition_v3 below --experiment-root.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Defaults to fixed_radius below --corrected-root.",
    )
    parser.add_argument(
        "--record-workers",
        type=int,
        choices=range(1, MAX_RECORD_WORKERS + 1),
        default=MAX_RECORD_WORKERS,
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
        "--rss-limit-bytes",
        type=int,
        default=DEFAULT_RSS_LIMIT_BYTES,
    )
    parser.add_argument("--progress", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    manifest = run_pipeline(
        experiment_root=arguments.experiment_root,
        corrected_root=arguments.corrected_root,
        output_root=arguments.output_root,
        record_workers=arguments.record_workers,
        reconstruction_timeout_seconds=arguments.reconstruction_timeout_seconds,
        evaluation_timeout_seconds=arguments.evaluation_timeout_seconds,
        rss_limit_bytes=arguments.rss_limit_bytes,
        progress=arguments.progress,
    )
    inventory = manifest["verification_inventory"]
    record_count = sum(
        int(row["expected_fixed_radius_record_count"])
        for row in manifest["source_inventory"]
    )
    print(
        f"complete: {record_count:,} corrected fixed-radius records; "
        f"{inventory['primary_contrast_cell_count']} primary "
        "adaptive-minus-fixed cells; "
        f"{inventory['historical_positive_primary_retained_count']}/"
        f"{inventory['historical_positive_primary_cell_count']} historical "
        "positive primary cells retained"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
