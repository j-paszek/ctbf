"""Run the frozen-transition portfolio or its additive Pooled-Q extension.

The pipeline reuses the preserved v2/v2a/v2b/v2c banks, writes every current
run to a fresh versioned root, resumes an interrupted prefix automatically,
and finishes with within-regime plus historical-versus-current reports.
With ``--add-pooled-q`` it validates the completed six-arm roots, reconstructs
only Pooled-Q, and writes separate expanded reports.  Neither mode regenerates
simulation, biopsy, or distance assets.
"""

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
    REPORT_SCHEMA_VERSION as VERIFICATION_REPORT_SCHEMA_VERSION,
    ResultBundle,
    _load_historical_bundle,
    write_report as write_verification_report,
)
from algorithm_evaluation.v5_shortlist_robustness_common import (
    BASELINE_SIMULATOR_REGIME,
    CURRENT_PAPER_DEVELOPMENT_ARM_IDS,
    CURRENT_PAPER_DEVELOPMENT_WITH_POOLED_Q_ARM_IDS,
    DEFAULT_BLOCK_COUNT,
    POOLED_Q_EXTENSION_ARM_IDS,
    RESULT_NAME,
    V2A_SIMULATOR_REGIME,
    V2B_SIMULATOR_REGIME,
    V2C_SIMULATOR_REGIME,
    load_bank_manifest,
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


PIPELINE_SCHEMA_VERSION = "ctbf-v5-frozen-transition-development-pipeline-v1"
CURRENT_ARM_SET = "current-paper-development"
POOLED_Q_EXTENSION_SCHEMA_VERSION = (
    "ctbf-v5-frozen-transition-pooled-q-extension-v1"
)
POOLED_Q_ARM_SET = "pooled-q-extension"
DEFAULT_EXPERIMENT_ROOT = Path(
    "algorithm_evaluation/results/v5_development/shortlist_robustness"
)
DEFAULT_OUTPUT_DIRECTORY = "frozen_transition_v3"
DEFAULT_POOLED_Q_EXTENSION_DIRECTORY = "pooled_q_v1"
PIPELINE_MANIFEST_NAME = "pipeline_manifest.json"


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
BANK_RELATIVE_PATHS = {
    BASELINE_SIMULATOR_REGIME: Path("bank_v1"),
    V2A_SIMULATOR_REGIME: Path("v2a/bank_v1"),
    V2B_SIMULATOR_REGIME: Path("v2b/bank_v1"),
    V2C_SIMULATOR_REGIME: Path("v2c/bank_v1"),
}
HISTORICAL_RESULT_RELATIVE_PATHS = {
    BASELINE_SIMULATOR_REGIME: (
        Path("runs/abcd_v2"),
        Path("runs/v2_extensions_v1"),
        Path("runs/adaptive_median_q50_v1"),
        Path("runs/partial_adaptive_median_q50_v1"),
    ),
    V2A_SIMULATOR_REGIME: (Path("v2a/runs/selected_all_v1"),),
    V2B_SIMULATOR_REGIME: (Path("v2b/runs/selected_all_v1"),),
    V2C_SIMULATOR_REGIME: (Path("v2c/runs/selected_all_v1"),),
}


def _resolved_paths(
    experiment_root: Path | str,
    output_root: Path | str | None,
) -> tuple[Path, Path]:
    experiment = Path(experiment_root).expanduser().resolve()
    output = (
        experiment / DEFAULT_OUTPUT_DIRECTORY
        if output_root is None
        else Path(output_root).expanduser().resolve()
    )
    return experiment, output


def _manifest_contract(
    *,
    experiment_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    return {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "scientific_role": (
            "post_correction_development_verification_not_held_out_"
            "paper_confirmation"
        ),
        "experiment_root": str(experiment_root),
        "output_root": str(output_root),
        "arm_set": CURRENT_ARM_SET,
        "arm_ids": list(CURRENT_PAPER_DEVELOPMENT_ARM_IDS),
        "decision_audit_schema": BIOPSY_GUIDED_AUDIT_SCHEMA_VERSION,
        "parent_eligibility_policy": (
            FROZEN_TRANSITION_PARENT_ELIGIBILITY_POLICY
        ),
        "banks_reused_without_regeneration": True,
        "historical_results_read_only": True,
        "regimes": list(REGIME_IDS),
    }


def _load_or_create_manifest(
    *,
    experiment_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    contract = _manifest_contract(
        experiment_root=experiment_root,
        output_root=output_root,
    )
    manifest_path = output_root / PIPELINE_MANIFEST_NAME
    if output_root.exists():
        if not output_root.is_dir() or not manifest_path.is_file():
            raise ValueError(
                "Existing frozen-transition output root has no pipeline manifest."
            )
        manifest = read_json(manifest_path)
        for field, expected in contract.items():
            if manifest.get(field) != expected:
                raise ValueError(
                    f"Stored frozen-transition pipeline field {field} changed."
                )
        if manifest.get("status") not in {"in_progress", "complete", "failure"}:
            raise ValueError("Stored frozen-transition pipeline status is invalid.")
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
        "arm_set": CURRENT_ARM_SET,
        "arm_ids": list(CURRENT_PAPER_DEVELOPMENT_ARM_IDS),
        "bank_root": str(bank_root),
        "block_count": DEFAULT_BLOCK_COUNT,
    }
    for field, value in expected.items():
        if result.get(field) != value:
            raise ValueError(f"Completed current run field {field} changed.")
    if result.get("failure_count") != 0:
        raise RuntimeError(
            f"Completed current run {run_id} contains "
            f"{result.get('failure_count')} failures."
        )


def _validate_source_inventory(experiment_root: Path) -> list[dict[str, Any]]:
    inventory = []
    for regime_id in REGIME_IDS:
        bank_root = (experiment_root / BANK_RELATIVE_PATHS[regime_id]).resolve()
        resolved_bank_root, bank = load_bank_manifest(
            bank_root,
            expected_block_count=DEFAULT_BLOCK_COUNT,
        )
        observed_regime = bank.get("simulator_regime_id")
        if observed_regime is None and regime_id == BASELINE_SIMULATOR_REGIME:
            observed_regime = BASELINE_SIMULATOR_REGIME
        if observed_regime != regime_id:
            raise ValueError(
                f"Expected source regime {regime_id}, observed {observed_regime}."
            )
        historical_roots = tuple(
            (experiment_root / relative).resolve()
            for relative in HISTORICAL_RESULT_RELATIVE_PATHS[regime_id]
        )
        current_stub = ResultBundle(
            regime_id=regime_id,
            result_roots=(),
            run_ids=(),
            bank_root=resolved_bank_root,
            bank=bank,
            records={},
            historical=False,
        )
        historical = _load_historical_bundle(
            regime_id,
            historical_roots,
            current=current_stub,
        )
        inventory.append(
            {
                "regime_id": regime_id,
                "regime_label": REGIME_LABELS[regime_id],
                "bank_root": str(resolved_bank_root),
                "available_condition_count": int(
                    bank["available_condition_count"]
                ),
                "expected_current_record_count": int(
                    bank["available_condition_count"]
                )
                * len(CURRENT_PAPER_DEVELOPMENT_ARM_IDS),
                "historical_run_ids": list(historical.run_ids),
            }
        )
    return inventory


def _run_or_resume_regime(
    *,
    regime_id: str,
    bank_root: Path,
    run_root: Path,
    record_workers: int,
    reconstruction_timeout_seconds: int,
    evaluation_timeout_seconds: int,
    rss_limit_bytes: int,
    progress: bool,
) -> dict[str, Any]:
    run_id = f"shortlist-{REGIME_LABELS[regime_id]}-frozen-transition-v3"
    result_path = run_root / RESULT_NAME
    resume = False
    if run_root.exists():
        if not run_root.is_dir() or not result_path.is_file():
            raise ValueError(f"Existing run root is incomplete: {run_root}.")
        stored = read_json(result_path)
        if stored.get("status") == "complete":
            _validate_completed_run(stored, bank_root=bank_root, run_id=run_id)
            return stored
        resume = True

    result = run_shortlist(
        bank_root=bank_root,
        output_root=run_root,
        run_id=run_id,
        reconstruction_timeout_seconds=reconstruction_timeout_seconds,
        evaluation_timeout_seconds=evaluation_timeout_seconds,
        rss_limit_bytes=rss_limit_bytes,
        record_workers=record_workers,
        arm_set=CURRENT_ARM_SET,
        expected_block_count=DEFAULT_BLOCK_COUNT,
        progress=progress,
        resume=resume,
    )
    _validate_completed_run(result, bank_root=bank_root, run_id=run_id)
    return result


def _write_or_validate_within_report(
    *,
    run_root: Path,
    report_root: Path,
) -> Mapping[str, Any]:
    summary_path = report_root / "summary.json"
    if report_root.exists():
        required_paths = (summary_path, report_root / "report.md")
        if not report_root.is_dir() or any(
            not path.is_file() for path in required_paths
        ):
            raise ValueError(f"Existing report root is incomplete: {report_root}.")
        report = read_json(summary_path)
        if report.get("schema_version") != WITHIN_REPORT_SCHEMA_VERSION:
            raise ValueError("Stored within-regime report schema changed.")
        if set(report.get("shortlist_arm_ids", ())) != set(
            CURRENT_PAPER_DEVELOPMENT_ARM_IDS
        ):
            raise ValueError("Stored within-regime report arm roster changed.")
        if report.get("result_roots") != [str(run_root.resolve())]:
            raise ValueError("Stored within-regime report source run changed.")
        if report.get("status") != "complete":
            raise ValueError("Stored within-regime report is not complete.")
        return report
    return write_within_regime_report(
        result_root=run_root,
        output_root=report_root,
        expected_block_count=DEFAULT_BLOCK_COUNT,
    )


def _write_or_validate_verification_report(
    *,
    current_roots: Mapping[str, Path],
    historical_roots: Mapping[str, Sequence[Path]],
    report_root: Path,
) -> Mapping[str, Any]:
    summary_path = report_root / "summary.json"
    if report_root.exists():
        required_paths = (
            summary_path,
            report_root / "report.md",
            report_root / "block_contrasts.csv",
            report_root / "placement_contrasts.csv",
            report_root / "affected_arm_score_changes.csv",
            report_root / "unaffected_comparator_reproducibility.csv",
        )
        if not report_root.is_dir() or any(
            not path.is_file() for path in required_paths
        ):
            raise ValueError(
                f"Existing verification report root is incomplete: {report_root}."
            )
        report = read_json(summary_path)
        if report.get("schema_version") != VERIFICATION_REPORT_SCHEMA_VERSION:
            raise ValueError("Stored verification report schema changed.")
        if report.get("status") != "complete":
            raise ValueError("Stored verification report is not complete.")
        observed_current_roots = {
            str(row["regime_id"]): row.get("current_result_roots")
            for row in report.get("source_inventory", ())
            if isinstance(row, Mapping)
        }
        expected_current_roots = {
            regime_id: [str(root.resolve())]
            for regime_id, root in current_roots.items()
        }
        if observed_current_roots != expected_current_roots:
            raise ValueError("Stored verification report source runs changed.")
        return report
    return write_verification_report(
        current_result_roots=current_roots,
        historical_result_roots=historical_roots,
        output_root=report_root,
    )


def _validate_base_pipeline_manifest(
    *,
    experiment_root: Path,
    output_root: Path,
) -> Mapping[str, Any]:
    manifest_path = output_root / PIPELINE_MANIFEST_NAME
    if not manifest_path.is_file():
        raise ValueError(
            "The completed six-arm frozen-transition pipeline is missing."
        )
    manifest = read_json(manifest_path)
    for field, expected in _manifest_contract(
        experiment_root=experiment_root,
        output_root=output_root,
    ).items():
        if manifest.get(field) != expected:
            raise ValueError(
                f"Stored six-arm pipeline field {field} changed."
            )
    if manifest.get("status") != "complete":
        raise ValueError(
            "The six-arm frozen-transition pipeline must be complete before "
            "adding Pooled-Q."
        )
    return manifest


def _pooled_q_manifest_contract(
    *,
    experiment_root: Path,
    base_output_root: Path,
    extension_root: Path,
) -> dict[str, Any]:
    return {
        "schema_version": POOLED_Q_EXTENSION_SCHEMA_VERSION,
        "scientific_role": (
            "additive_pooled_q_development_comparison_not_held_out_"
            "paper_confirmation"
        ),
        "experiment_root": str(experiment_root),
        "base_six_arm_output_root": str(base_output_root),
        "output_root": str(extension_root),
        "arm_set": POOLED_Q_ARM_SET,
        "arm_ids": list(POOLED_Q_EXTENSION_ARM_IDS),
        "expanded_arm_ids": list(
            CURRENT_PAPER_DEVELOPMENT_WITH_POOLED_Q_ARM_IDS
        ),
        "base_runs_read_only": True,
        "banks_reused_without_regeneration": True,
        "regimes": list(REGIME_IDS),
    }


def _load_or_create_pooled_q_manifest(
    *,
    experiment_root: Path,
    base_output_root: Path,
    extension_root: Path,
) -> dict[str, Any]:
    contract = _pooled_q_manifest_contract(
        experiment_root=experiment_root,
        base_output_root=base_output_root,
        extension_root=extension_root,
    )
    manifest_path = extension_root / PIPELINE_MANIFEST_NAME
    if extension_root.exists():
        if not extension_root.is_dir() or not manifest_path.is_file():
            raise ValueError(
                "Existing Pooled-Q extension root has no pipeline manifest."
            )
        manifest = read_json(manifest_path)
        for field, expected in contract.items():
            if manifest.get(field) != expected:
                raise ValueError(
                    f"Stored Pooled-Q extension field {field} changed."
                )
        if manifest.get("status") not in {
            "in_progress",
            "complete",
            "failure",
        }:
            raise ValueError("Stored Pooled-Q extension status is invalid.")
        return manifest

    extension_root.mkdir(parents=True)
    manifest = {
        **contract,
        "status": "in_progress",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed_regimes": [],
    }
    write_json(manifest_path, manifest)
    return manifest


def _validate_pooled_q_run(
    result: Mapping[str, Any],
    *,
    bank_root: Path,
    run_id: str,
    expected_record_count: int,
) -> None:
    expected = {
        "status": "complete",
        "run_id": run_id,
        "arm_set": POOLED_Q_ARM_SET,
        "arm_ids": list(POOLED_Q_EXTENSION_ARM_IDS),
        "bank_root": str(bank_root),
        "block_count": DEFAULT_BLOCK_COUNT,
        "expected_record_count": expected_record_count,
        "completed_record_count": expected_record_count,
    }
    for field, value in expected.items():
        if result.get(field) != value:
            raise ValueError(
                f"Completed Pooled-Q run field {field} changed."
            )
    failure_count = result.get("failure_count")
    if (
        isinstance(failure_count, bool)
        or not isinstance(failure_count, int)
        or failure_count < 0
    ):
        raise ValueError("Completed Pooled-Q run has an invalid failure count.")


def _run_or_resume_pooled_q_regime(
    *,
    regime_id: str,
    bank_root: Path,
    run_root: Path,
    expected_record_count: int,
    record_workers: int,
    reconstruction_timeout_seconds: int,
    evaluation_timeout_seconds: int,
    rss_limit_bytes: int,
    progress: bool,
) -> Mapping[str, Any]:
    run_id = f"shortlist-{REGIME_LABELS[regime_id]}-pooled-q-v1"
    result_path = run_root / RESULT_NAME
    resume = False
    if run_root.exists():
        if not run_root.is_dir() or not result_path.is_file():
            raise ValueError(
                f"Existing Pooled-Q run root is incomplete: {run_root}."
            )
        stored = read_json(result_path)
        if stored.get("status") == "complete":
            _validate_pooled_q_run(
                stored,
                bank_root=bank_root,
                run_id=run_id,
                expected_record_count=expected_record_count,
            )
            return stored
        resume = True

    result = run_shortlist(
        bank_root=bank_root,
        output_root=run_root,
        run_id=run_id,
        reconstruction_timeout_seconds=reconstruction_timeout_seconds,
        evaluation_timeout_seconds=evaluation_timeout_seconds,
        rss_limit_bytes=rss_limit_bytes,
        record_workers=record_workers,
        arm_set=POOLED_Q_ARM_SET,
        expected_block_count=DEFAULT_BLOCK_COUNT,
        progress=progress,
        resume=resume,
    )
    _validate_pooled_q_run(
        result,
        bank_root=bank_root,
        run_id=run_id,
        expected_record_count=expected_record_count,
    )
    return result


def _assert_existing_report_values_preserved(
    base_report: Mapping[str, Any],
    expanded_report: Mapping[str, Any],
) -> None:
    row_fields = (
        "algorithm_cell_summaries",
        "pairwise_by_height_and_placement",
        "depth_interactions",
        "placement_interactions",
    )
    for group_name, base_group in base_report["comparison_groups"].items():
        expanded_group = expanded_report["comparison_groups"].get(group_name)
        if not isinstance(expanded_group, Mapping):
            raise ValueError(
                f"Expanded report lost the {group_name} comparison group."
            )
        base_ids = set(base_group["arm_ids"])
        for field in row_fields:
            base_rows = base_group[field]
            expanded_rows = expanded_group[field]
            if field == "algorithm_cell_summaries":
                preserved = [
                    row for row in expanded_rows if row["arm_id"] in base_ids
                ]
            else:
                preserved = [
                    row
                    for row in expanded_rows
                    if row["arm_a"] in base_ids and row["arm_b"] in base_ids
                ]
            if preserved != base_rows:
                raise ValueError(
                    "Adding Pooled-Q changed existing report values in "
                    f"{group_name}.{field}."
                )
    if (
        expanded_report["condition_diagnostics"]
        != base_report["condition_diagnostics"]
        or expanded_report["adaptive_radius_diagnostics"]
        != base_report["adaptive_radius_diagnostics"]
    ):
        raise ValueError(
            "Adding Pooled-Q changed existing condition or radius diagnostics."
        )


def _write_or_validate_pooled_q_report(
    *,
    base_run_root: Path,
    pooled_q_run_root: Path,
    base_report: Mapping[str, Any],
    report_root: Path,
) -> Mapping[str, Any]:
    expected_result_roots = [
        str(base_run_root.resolve()),
        str(pooled_q_run_root.resolve()),
    ]
    summary_path = report_root / "summary.json"
    if report_root.exists():
        required_paths = (summary_path, report_root / "report.md")
        if not report_root.is_dir() or any(
            not path.is_file() for path in required_paths
        ):
            raise ValueError(
                f"Existing expanded report root is incomplete: {report_root}."
            )
        report = read_json(summary_path)
    else:
        report = write_within_regime_report(
            result_roots=(base_run_root, pooled_q_run_root),
            output_root=report_root,
            expected_block_count=DEFAULT_BLOCK_COUNT,
        )

    if report.get("schema_version") != WITHIN_REPORT_SCHEMA_VERSION:
        raise ValueError("Stored expanded report schema changed.")
    if report.get("status") != "complete":
        raise ValueError("Stored expanded report is not complete.")
    if set(report.get("shortlist_arm_ids", ())) != set(
        CURRENT_PAPER_DEVELOPMENT_WITH_POOLED_Q_ARM_IDS
    ):
        raise ValueError("Stored expanded report arm roster changed.")
    if report.get("result_roots") != expected_result_roots:
        raise ValueError("Stored expanded report source runs changed.")
    _assert_existing_report_values_preserved(base_report, report)
    return report


def run_pooled_q_extension(
    *,
    experiment_root: Path | str = DEFAULT_EXPERIMENT_ROOT,
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
        raise ValueError(
            f"record_workers must be between 1 and {MAX_RECORD_WORKERS}."
        )
    experiment, base_output = _resolved_paths(experiment_root, output_root)
    _validate_base_pipeline_manifest(
        experiment_root=experiment,
        output_root=base_output,
    )
    extension = (
        base_output / DEFAULT_POOLED_Q_EXTENSION_DIRECTORY
    ).resolve()
    manifest = _load_or_create_pooled_q_manifest(
        experiment_root=experiment,
        base_output_root=base_output,
        extension_root=extension,
    )
    manifest_path = extension / PIPELINE_MANIFEST_NAME
    manifest["status"] = "in_progress"
    manifest.pop("failure", None)
    manifest.pop("completed_at_utc", None)
    write_json(manifest_path, manifest)

    source_inventory = []
    total_records = 0
    total_failures = 0
    try:
        for regime_id in REGIME_IDS:
            label = REGIME_LABELS[regime_id]
            bank_root = (experiment / BANK_RELATIVE_PATHS[regime_id]).resolve()
            resolved_bank_root, bank = load_bank_manifest(
                bank_root,
                expected_block_count=DEFAULT_BLOCK_COUNT,
            )
            expected_record_count = int(bank["available_condition_count"])
            base_run_root = (base_output / label / "run").resolve()
            base_result = read_json(base_run_root / RESULT_NAME)
            _validate_completed_run(
                base_result,
                bank_root=resolved_bank_root,
                run_id=f"shortlist-{label}-frozen-transition-v3",
            )
            base_report = _write_or_validate_within_report(
                run_root=base_run_root,
                report_root=(base_output / label / "report").resolve(),
            )

            pooled_q_run_root = (extension / label / "run").resolve()
            pooled_q_result = _run_or_resume_pooled_q_regime(
                regime_id=regime_id,
                bank_root=resolved_bank_root,
                run_root=pooled_q_run_root,
                expected_record_count=expected_record_count,
                record_workers=int(record_workers),
                reconstruction_timeout_seconds=reconstruction_timeout_seconds,
                evaluation_timeout_seconds=evaluation_timeout_seconds,
                rss_limit_bytes=rss_limit_bytes,
                progress=progress,
            )
            expanded_report_root = (extension / label / "report").resolve()
            _write_or_validate_pooled_q_report(
                base_run_root=base_run_root,
                pooled_q_run_root=pooled_q_run_root,
                base_report=base_report,
                report_root=expanded_report_root,
            )

            regime_failures = int(pooled_q_result["failure_count"])
            total_records += expected_record_count
            total_failures += regime_failures
            source_inventory.append(
                {
                    "regime_id": regime_id,
                    "regime_label": label,
                    "bank_root": str(resolved_bank_root),
                    "base_result_root": str(base_run_root),
                    "pooled_q_result_root": str(pooled_q_run_root),
                    "expanded_report_root": str(expanded_report_root),
                    "record_count": expected_record_count,
                    "failure_count": regime_failures,
                    "existing_report_values_preserved": True,
                }
            )
            completed = manifest.setdefault("completed_regimes", [])
            if regime_id not in completed:
                completed.append(regime_id)
            manifest["source_inventory"] = source_inventory
            manifest["record_count"] = total_records
            manifest["failure_count"] = total_failures
            write_json(manifest_path, manifest)
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
    manifest["source_inventory"] = source_inventory
    manifest["record_count"] = total_records
    manifest["failure_count"] = total_failures
    manifest["existing_report_values_preserved"] = True
    write_json(manifest_path, manifest)
    return manifest


def run_pipeline(
    *,
    experiment_root: Path | str = DEFAULT_EXPERIMENT_ROOT,
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
    experiment, output = _resolved_paths(experiment_root, output_root)
    source_inventory = _validate_source_inventory(experiment)
    manifest = _load_or_create_manifest(
        experiment_root=experiment,
        output_root=output,
    )
    manifest_path = output / PIPELINE_MANIFEST_NAME
    manifest["status"] = "in_progress"
    manifest["source_inventory"] = source_inventory
    manifest.pop("failure", None)
    manifest.pop("completed_at_utc", None)
    write_json(manifest_path, manifest)

    current_roots = {}
    historical_roots = {}
    try:
        for regime_id in REGIME_IDS:
            label = REGIME_LABELS[regime_id]
            bank_root = (experiment / BANK_RELATIVE_PATHS[regime_id]).resolve()
            run_root = (output / label / "run").resolve()
            report_root = (output / label / "report").resolve()
            _run_or_resume_regime(
                regime_id=regime_id,
                bank_root=bank_root,
                run_root=run_root,
                record_workers=int(record_workers),
                reconstruction_timeout_seconds=reconstruction_timeout_seconds,
                evaluation_timeout_seconds=evaluation_timeout_seconds,
                rss_limit_bytes=rss_limit_bytes,
                progress=progress,
            )
            _write_or_validate_within_report(
                run_root=run_root,
                report_root=report_root,
            )
            current_roots[regime_id] = run_root
            historical_roots[regime_id] = tuple(
                (experiment / relative).resolve()
                for relative in HISTORICAL_RESULT_RELATIVE_PATHS[regime_id]
            )
            completed = manifest.setdefault("completed_regimes", [])
            if regime_id not in completed:
                completed.append(regime_id)
                write_json(manifest_path, manifest)

        verification_root = (output / "verification").resolve()
        verification = _write_or_validate_verification_report(
            current_roots=current_roots,
            historical_roots=historical_roots,
            report_root=verification_root,
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
        help="Root containing the preserved shortlist banks and historical runs.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help=(
            "Six-arm pipeline root. Without --add-pooled-q it is created as "
            "the pipeline output; with --add-pooled-q it must already contain "
            "the completed six-arm results. Defaults to frozen_transition_v3 "
            "below --experiment-root."
        ),
    )
    parser.add_argument(
        "--add-pooled-q",
        action="store_true",
        help=(
            "Run only the additive Pooled-Q arm on the preserved banks and "
            "write new seven-arm reports without modifying the completed "
            "six-arm runs or reports."
        ),
    )
    parser.add_argument(
        "--record-workers",
        type=int,
        default=MAX_RECORD_WORKERS,
        choices=range(1, MAX_RECORD_WORKERS + 1),
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
    if arguments.add_pooled_q:
        manifest = run_pooled_q_extension(
            experiment_root=arguments.experiment_root,
            output_root=arguments.output_root,
            record_workers=arguments.record_workers,
            reconstruction_timeout_seconds=(
                arguments.reconstruction_timeout_seconds
            ),
            evaluation_timeout_seconds=arguments.evaluation_timeout_seconds,
            rss_limit_bytes=arguments.rss_limit_bytes,
            progress=arguments.progress,
        )
        print(
            "complete: Pooled-Q added to four development regimes; "
            f"{manifest['record_count']} arm records; "
            f"{manifest['failure_count']} failures; existing report values "
            "preserved"
        )
        return 0

    manifest = run_pipeline(
        experiment_root=arguments.experiment_root,
        output_root=arguments.output_root,
        record_workers=arguments.record_workers,
        reconstruction_timeout_seconds=arguments.reconstruction_timeout_seconds,
        evaluation_timeout_seconds=arguments.evaluation_timeout_seconds,
        rss_limit_bytes=arguments.rss_limit_bytes,
        progress=arguments.progress,
    )
    inventory = manifest["verification_inventory"]
    print(
        "complete: four corrected development regimes; "
        f"{inventory['primary_contrast_cell_count']} primary contrast cells; "
        f"{inventory['positive_contrast_retained_count']}/"
        f"{inventory['historical_positive_contrast_cell_count']} historical "
        "positive cells retained; "
        f"{inventory['unaffected_comparator_exact_cell_count']}/"
        f"{inventory['unaffected_comparator_cell_count']} comparator cells exact"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
