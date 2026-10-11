"""Audit condition-specific cnp2cnp plus M2 resources on frozen inputs.

This post-confirmation audit does not regenerate truth, observations, scores,
or registered results.  Each standard condition is executed in a fresh process.
Its distance matrix is constructed only from CNPs present in that condition and
must exactly equal the corresponding submatrix in the frozen confirmation bank.
The resulting M2 topology summary must also exactly equal its frozen record.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime, timezone
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Callable, Mapping, Sequence

import networkx as nx
import numpy as np

from algorithm_evaluation.paper_pipeline_contract import (
    aligned_distance_submatrix,
    canonical_json_sha256,
    file_sha256,
    read_json,
)
from algorithm_evaluation.paper_pipeline_runner import (
    measured_stage,
    validate_reconstruction_input,
)
from algorithm_evaluation.process_isolation import (
    FreshSpawnTaskPool,
    fresh_process_contract,
)
from algorithm_evaluation.v5_algorithm_development_common import (
    ARM_SPEC_BY_ID,
    reconstruct_development_arm,
    tree_summary,
)
from algorithm_evaluation.v5_confirmation_bank import (
    deserialize_confirmation_distance,
    load_confirmation_bank,
)
from algorithm_evaluation.v5_confirmation_common import (
    DEFAULT_MANIFEST_PATH,
    M2_ID,
    PROJECT_ROOT,
    STANDARD_CONDITION_IDS,
    ensure_new_output_root,
    record_path,
    validate_checksum_closure,
    validate_cnp2cnp_source,
    validate_internal_runtime,
    validate_manifest,
    write_checksum_closure,
    write_json,
)
from algorithm_evaluation.v5_confirmation_medicc2 import (
    RUN_CHECKSUM_NAME as MEDICC2_RUN_CHECKSUM_NAME,
    load_medicc2_confirmation,
)
from algorithm_evaluation.v5_confirmation_run import load_confirmation_run
from ctbs import (
    Cnp2CnpFileDistanceProvider,
    CtbsRuntimeConfig,
    DistanceMatrix,
    load_ctbs_runtime_config,
)
from distance_semantics import (
    CNP2CNP_SEMANTICS_VERSION,
    validate_distance_label_coverage,
    validate_distance_matrix,
)
from simulator import Genotype


DEFAULT_CONFIRMATION_ROOT = (
    PROJECT_ROOT
    / "algorithm_evaluation"
    / "results"
    / "v5_confirmation"
    / "confirmation_v1"
)

AUDIT_SCHEMA_VERSION = "ctbf-v5-condition-specific-resource-audit-v1"
RECORD_SCHEMA_VERSION = "ctbf-v5-condition-specific-resource-record-v1"
SUMMARY_SCHEMA_VERSION = "ctbf-v5-condition-specific-resource-summary-v1"
AUDIT_MANIFEST_NAME = "audit_manifest.json"
AUDIT_SUMMARY_NAME = "summary.json"
AUDIT_RESOURCE_TABLE_NAME = "resources.csv"
AUDIT_CHECKSUM_NAME = "audit_checksums.sha256"
AUDIT_WORKER_UNIT = "condition_specific_distance_and_m2_reconstruction"

DEFAULT_WORKERS = 4
MAX_WORKERS = 8
DEFAULT_DISTANCE_TIMEOUT_SECONDS = 1200
DEFAULT_RECONSTRUCTION_TIMEOUT_SECONDS = 300
DEFAULT_CNP2CNP_PROCESS_TIMEOUT_SECONDS = 540
DEFAULT_RSS_LIMIT_BYTES = 4 * 1024**3
DEFAULT_CAPTURE_LIMIT_BYTES = 1_048_576
WORKER_GRACE_SECONDS = 90


DistanceCompute = Callable[[Sequence[Genotype]], DistanceMatrix]
ReconstructionCompute = Callable[
    [Mapping[str, Any], DistanceMatrix, int],
    tuple[nx.DiGraph, Mapping[Any, Any], Any, Mapping[str, Any]],
]


def _typed_error(error: BaseException, stage: str) -> dict[str, Any]:
    return {
        "stage": stage,
        "type": type(error).__name__,
        "message": str(error)[:4096],
    }


def _auditable_resource(resources: Mapping[str, Any], stage: str) -> None:
    memory = resources.get("memory")
    if not isinstance(resources.get("wall_time_ns"), int):
        raise RuntimeError(f"{stage} has no auditable wall time.")
    if not isinstance(memory, Mapping) or not isinstance(
        memory.get("peak_rss_bytes"), int
    ):
        raise RuntimeError(f"{stage} has no auditable process-tree peak RSS.")


def _profile_by_label(payload: Mapping[str, Any]) -> dict[Any, tuple[int, ...]]:
    """Return the unique CNP assigned to every observed state label."""
    validate_reconstruction_input(payload)
    profiles: dict[Any, tuple[int, ...]] = {}
    for level in payload["levels"]:
        for state in level["states"]:
            label = state["state_label"]
            profile = tuple(int(value) for value in state["cnp"])
            previous = profiles.setdefault(label, profile)
            if previous != profile:
                raise ValueError("One observed state label maps to multiple CNPs.")
    if not profiles:
        raise ValueError("A resource-audit condition has no observed CNPs.")
    return profiles


def condition_cells(
    payload: Mapping[str, Any],
    ordered_labels: Sequence[Any],
) -> list[Genotype]:
    """Construct distance inputs from this condition and no other condition."""
    profiles = _profile_by_label(payload)
    if len(ordered_labels) != len(set(ordered_labels)):
        raise ValueError("Condition-specific distance labels are not unique.")
    if set(ordered_labels) != set(profiles):
        raise ValueError("Condition-specific labels differ from the observed CNP set.")
    return [
        Genotype(
            profiles[label],
            label,
            generation=0,
            cell_id=label,
        )
        for label in ordered_labels
    ]


def _resource_pair(resources: Mapping[str, Any], stage: str) -> tuple[int, int]:
    _auditable_resource(resources, stage)
    return (
        int(resources["wall_time_ns"]),
        int(resources["memory"]["peak_rss_bytes"]),
    )


def _pipeline_resources(
    distance: Mapping[str, Any],
    reconstruction: Mapping[str, Any],
) -> dict[str, Any]:
    distance_wall, distance_rss = _resource_pair(distance, "distance")
    reconstruction_wall, reconstruction_rss = _resource_pair(
        reconstruction, "reconstruction"
    )
    return {
        "wall_time_ns": distance_wall + reconstruction_wall,
        "memory": {
            "method": "maximum_of_sequential_stage_process_tree_rss",
            "peak_rss_bytes": max(distance_rss, reconstruction_rss),
        },
        "stage_order": ["condition_specific_distance", "m2_reconstruction"],
    }


def _failure_record(
    case: Mapping[str, Any],
    *,
    error: BaseException,
    stage: str,
    observed_state_count: int | None,
    frozen_block_state_count: int | None,
    distance_resources: Mapping[str, Any] | None,
    reconstruction_resources: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema_version": RECORD_SCHEMA_VERSION,
        "case_id": str(case["case_id"]),
        "block_index": int(case["block_index"]),
        "condition_id": str(case["condition_id"]),
        "height": int(case["height"]),
        "arm_id": M2_ID,
        "status": "failure",
        "failure": _typed_error(error, stage),
        "observed_unique_state_count": observed_state_count,
        "frozen_block_union_state_count": frozen_block_state_count,
        "equivalence": None,
        "resources": {
            "distance": distance_resources,
            "reconstruction": reconstruction_resources,
            "pipeline": None,
        },
    }


def _audit_case(
    bank_root: str,
    internal_root: str,
    case: Mapping[str, Any],
    runtime_config: CtbsRuntimeConfig | None,
    distance_timeout_seconds: int,
    reconstruction_timeout_seconds: int,
    rss_limit_bytes: int,
    distance_compute: DistanceCompute | None = None,
    reconstruction_compute: ReconstructionCompute | None = None,
) -> dict[str, Any]:
    bank_path = Path(bank_root)
    run_path = Path(internal_root)
    distance_resources = None
    reconstruction_resources = None
    observed_state_count = None
    frozen_block_state_count = None

    try:
        payload = read_json(bank_path / str(case["input_path"]))
        validate_reconstruction_input(payload)
        if payload.get("case_id") != case.get("case_id"):
            raise ValueError("Frozen reconstruction-input case id changed.")
        metadata = read_json(bank_path / str(case["metadata_path"]))
        if metadata.get("case_id") != case.get("case_id"):
            raise ValueError("Frozen case-metadata id changed.")
        frozen_block_distance = deserialize_confirmation_distance(
            read_json(bank_path / str(case["distance_path"]))
        )
        frozen_block_state_count = len(frozen_block_distance.ids)
        profiles = _profile_by_label(payload)
        observed_state_count = len(profiles)
        expected_ids, expected_matrix = aligned_distance_submatrix(
            frozen_block_distance.ids,
            frozen_block_distance.matrix,
            profiles,
        )
        validate_distance_matrix(expected_ids, expected_matrix)
        validate_distance_label_coverage(
            expected_ids,
            profiles,
            allow_extra=False,
        )
        cells = condition_cells(payload, expected_ids)
        frozen_record = read_json(
            run_path / record_path(str(case["case_id"]), M2_ID)
        )
        if (
            frozen_record.get("status") != "success"
            or frozen_record.get("case_id") != case.get("case_id")
            or frozen_record.get("arm_id") != M2_ID
        ):
            raise ValueError("Frozen M2 record is unavailable or misaligned.")
    except Exception as error:
        return _failure_record(
            case,
            error=error,
            stage="input_validation",
            observed_state_count=observed_state_count,
            frozen_block_state_count=frozen_block_state_count,
            distance_resources=None,
            reconstruction_resources=None,
        )

    compute_distance = distance_compute
    if compute_distance is None:
        if runtime_config is None:
            return _failure_record(
                case,
                error=RuntimeError("No production distance runtime was supplied."),
                stage="distance",
                observed_state_count=observed_state_count,
                frozen_block_state_count=frozen_block_state_count,
                distance_resources=None,
                reconstruction_resources=None,
            )
        compute_distance = Cnp2CnpFileDistanceProvider(runtime_config).compute

    direct_distance, distance_resources, distance_error = measured_stage(
        lambda: compute_distance(cells),
        timeout_seconds=distance_timeout_seconds,
        rss_limit_bytes=rss_limit_bytes,
    )
    try:
        _auditable_resource(distance_resources, "distance")
        if distance_error is not None or direct_distance is None:
            raise distance_error or RuntimeError("Distance computation returned no matrix.")
        validate_distance_matrix(direct_distance.ids, direct_distance.matrix)
        validate_distance_label_coverage(
            direct_distance.ids,
            expected_ids,
            allow_extra=False,
        )
        if list(direct_distance.ids) != list(expected_ids):
            raise ValueError("Condition-specific distance row order changed.")
        difference = np.asarray(direct_distance.matrix) - np.asarray(expected_matrix)
        max_abs_difference = float(np.max(np.abs(difference), initial=0.0))
        if not np.array_equal(direct_distance.matrix, expected_matrix):
            raise ValueError(
                "Condition-specific distance matrix differs from its frozen submatrix "
                f"(maximum absolute difference {max_abs_difference})."
            )
        provenance = direct_distance.provenance
        if not isinstance(provenance, Mapping):
            raise ValueError("Condition-specific distance matrix lacks provenance.")
        semantics = provenance.get(
            "semantics_version",
            provenance.get("semantic_version"),
        )
        if semantics != CNP2CNP_SEMANTICS_VERSION:
            raise ValueError("Condition-specific distance semantics changed.")
    except Exception as error:
        return _failure_record(
            case,
            error=error,
            stage="distance_equivalence",
            observed_state_count=observed_state_count,
            frozen_block_state_count=frozen_block_state_count,
            distance_resources=distance_resources,
            reconstruction_resources=None,
        )

    spec = ARM_SPEC_BY_ID[M2_ID]
    seed = int(metadata["reconstruction_seed"])
    reconstruct = reconstruction_compute
    if reconstruct is None:
        reconstruct = lambda input_payload, distance, reconstruction_seed: (
            reconstruct_development_arm(
                spec,
                input_payload,
                distance,
                reconstruction_seed=reconstruction_seed,
            )
        )
    reconstruction, reconstruction_resources, reconstruction_error = measured_stage(
        lambda: reconstruct(payload, direct_distance, seed),
        timeout_seconds=reconstruction_timeout_seconds,
        rss_limit_bytes=rss_limit_bytes,
    )
    try:
        _auditable_resource(reconstruction_resources, "reconstruction")
        if reconstruction_error is not None or reconstruction is None:
            raise reconstruction_error or RuntimeError("M2 returned no reconstruction.")
        tree, _levels, _returned_root, _reconstruction_metadata = reconstruction
        if not nx.is_arborescence(tree):
            raise ValueError("Condition-specific M2 output is not an arborescence.")
        direct_summary = tree_summary(tree)
        frozen_summary = frozen_record.get("tree_summary")
        if direct_summary != frozen_summary:
            raise ValueError("Condition-specific M2 topology summary changed.")
    except Exception as error:
        return _failure_record(
            case,
            error=error,
            stage="reconstruction_equivalence",
            observed_state_count=observed_state_count,
            frozen_block_state_count=frozen_block_state_count,
            distance_resources=distance_resources,
            reconstruction_resources=reconstruction_resources,
        )

    condition_pairs = observed_state_count * (observed_state_count - 1) // 2
    frozen_pairs = frozen_block_state_count * (frozen_block_state_count - 1) // 2
    return {
        "schema_version": RECORD_SCHEMA_VERSION,
        "case_id": str(case["case_id"]),
        "block_index": int(case["block_index"]),
        "condition_id": str(case["condition_id"]),
        "height": int(case["height"]),
        "arm_id": M2_ID,
        "status": "success",
        "failure": None,
        "observed_unique_state_count": observed_state_count,
        "frozen_block_union_state_count": frozen_block_state_count,
        "unordered_condition_pair_count": condition_pairs,
        "unordered_frozen_block_union_pair_count": frozen_pairs,
        "equivalence": {
            "distance_ids_exact": True,
            "distance_matrix_exact": True,
            "distance_maximum_absolute_difference": max_abs_difference,
            "distance_digest": canonical_json_sha256(
                {
                    "ids": list(direct_distance.ids),
                    "matrix": np.asarray(direct_distance.matrix).tolist(),
                }
            ),
            "tree_summary_exact": True,
            "canonical_topology_digest": direct_summary[
                "canonical_topology_digest"
            ],
        },
        "resources": {
            "distance": distance_resources,
            "reconstruction": reconstruction_resources,
            "pipeline": _pipeline_resources(
                distance_resources,
                reconstruction_resources,
            ),
        },
    }


def _execute_task(
    ordinal: int,
    bank_root: str,
    internal_root: str,
    case: Mapping[str, Any],
    runtime_config: CtbsRuntimeConfig,
    distance_timeout_seconds: int,
    reconstruction_timeout_seconds: int,
    rss_limit_bytes: int,
) -> tuple[int, dict[str, Any]]:
    return ordinal, _audit_case(
        bank_root,
        internal_root,
        case,
        runtime_config,
        distance_timeout_seconds,
        reconstruction_timeout_seconds,
        rss_limit_bytes,
    )


def _numeric_summary(values: Sequence[float | int]) -> dict[str, Any]:
    if not values:
        raise ValueError("A resource summary cannot be empty.")
    numeric = [float(value) for value in values]
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError("A resource summary contains a non-finite value.")
    return {
        "count": len(numeric),
        "mean": statistics.fmean(numeric),
        "median": statistics.median(numeric),
        "minimum": min(numeric),
        "maximum": max(numeric),
    }


def _stage_summary(
    records: Sequence[Mapping[str, Any]],
    stage: str,
) -> dict[str, Any]:
    wall: list[float] = []
    rss: list[int] = []
    for record in records:
        resources = record["resources"][stage]
        stage_wall, stage_rss = _resource_pair(resources, stage)
        wall.append(stage_wall / 1e9)
        rss.append(stage_rss)
    return {
        "count": len(records),
        "wall_seconds": _numeric_summary(wall),
        "peak_rss_bytes": _numeric_summary(rss),
    }


def build_resource_summary(
    audit_records: Sequence[Mapping[str, Any]],
    medicc2_records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize aligned successful M2 and MEDICC2 condition resources."""
    if not audit_records:
        raise ValueError("No condition-specific M2 records were supplied.")
    if any(record.get("status") != "success" for record in audit_records):
        raise ValueError("Resource summary requires successful audit records.")
    ordered_medicc2 = []
    for record in audit_records:
        case_id = str(record["case_id"])
        medicc2 = medicc2_records.get(case_id)
        if medicc2 is None or medicc2.get("status") != "success":
            raise ValueError(f"Missing successful MEDICC2 record for {case_id}.")
        ordered_medicc2.append(
            {
                "resources": {
                    "pipeline": medicc2["resources"]["reconstruction"]
                }
            }
        )

    m2_pipeline = _stage_summary(audit_records, "pipeline")
    medicc2_pipeline = _stage_summary(ordered_medicc2, "pipeline")
    m2_seconds = [
        _resource_pair(record["resources"]["pipeline"], "M2 pipeline")[0] / 1e9
        for record in audit_records
    ]
    medicc2_seconds = [
        _resource_pair(record["resources"]["pipeline"], "MEDICC2 pipeline")[0]
        / 1e9
        for record in ordered_medicc2
    ]
    differences = [
        external - internal
        for internal, external in zip(m2_seconds, medicc2_seconds, strict=True)
    ]
    ratios = [
        external / internal
        for internal, external in zip(m2_seconds, medicc2_seconds, strict=True)
    ]
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "case_count": len(audit_records),
        "m2_condition_specific_distance": _stage_summary(
            audit_records, "distance"
        ),
        "m2_reconstruction": _stage_summary(audit_records, "reconstruction"),
        "m2_condition_specific_pipeline": m2_pipeline,
        "medicc2_pipeline": medicc2_pipeline,
        "paired_medicc2_minus_m2_wall_seconds": _numeric_summary(differences),
        "paired_medicc2_over_m2_wall_ratio": _numeric_summary(ratios),
        "medicc2_slower_condition_count": sum(value > 0 for value in differences),
        "equal_wall_time_condition_count": sum(value == 0 for value in differences),
        "medicc2_faster_condition_count": sum(value < 0 for value in differences),
        "resource_interpretation": (
            "M2 time is condition-specific cnp2cnp construction plus M2 "
            "reconstruction. M2 sequential peak RSS is the maximum of those "
            "two stage peaks. MEDICC2 resources are its frozen external-process "
            "distance-and-topology measurements. Evaluation is excluded from both."
        ),
    }


def _resource_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for stage in (
        "m2_condition_specific_distance",
        "m2_reconstruction",
        "m2_condition_specific_pipeline",
        "medicc2_pipeline",
    ):
        values = summary[stage]
        rows.append(
            {
                "schema_version": SUMMARY_SCHEMA_VERSION,
                "stage": stage,
                "count": values["count"],
                "wall_mean_seconds": values["wall_seconds"]["mean"],
                "wall_median_seconds": values["wall_seconds"]["median"],
                "wall_minimum_seconds": values["wall_seconds"]["minimum"],
                "wall_maximum_seconds": values["wall_seconds"]["maximum"],
                "rss_mean_bytes": values["peak_rss_bytes"]["mean"],
                "rss_median_bytes": values["peak_rss_bytes"]["median"],
                "rss_minimum_bytes": values["peak_rss_bytes"]["minimum"],
                "rss_maximum_bytes": values["peak_rss_bytes"]["maximum"],
            }
        )
    return rows


def _write_resource_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = list(rows[0])
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _standard_cases(bank: Mapping[str, Any]) -> list[dict[str, Any]]:
    rank = {condition_id: index for index, condition_id in enumerate(STANDARD_CONDITION_IDS)}
    cases = [
        dict(case)
        for case in bank["cases"]
        if case.get("condition_id") in rank
    ]
    cases.sort(
        key=lambda case: (
            int(case["block_index"]),
            rank[str(case["condition_id"])],
        )
    )
    return cases


def _medicc2_records(
    medicc2_root: Path,
    medicc2_run: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    records = {}
    for entry in medicc2_run["record_inventory"]:
        condition_id = str(entry["condition_id"])
        if condition_id not in STANDARD_CONDITION_IDS:
            continue
        record = read_json(medicc2_root / str(entry["path"]))
        if record.get("case_id") != entry.get("case_id"):
            raise ValueError("MEDICC2 record inventory is misaligned.")
        case_id = str(entry["case_id"])
        if case_id in records:
            raise ValueError(f"Duplicate MEDICC2 record for {case_id}.")
        records[case_id] = record
    if len(records) != 300:
        raise ValueError("The frozen MEDICC2 run does not contain 300 standard records.")
    return records


def _record_relative(case_id: str) -> str:
    return (Path("records") / f"{case_id}.json").as_posix()


def _new_manifest(
    *,
    validation: Mapping[str, Any],
    bank_root: Path,
    bank: Mapping[str, Any],
    internal_root: Path,
    internal_run: Mapping[str, Any],
    medicc2_root: Path,
    medicc2_run: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
    technical_smoke: bool,
    resources: Mapping[str, Any],
    cnp2cnp_runtime: Mapping[str, Any],
    internal_runtime: Mapping[str, Any],
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "audit_id": (
            "ctbf-v5-condition-specific-resource-smoke-v1"
            if technical_smoke
            else "ctbf-v5-condition-specific-resource-audit-v1"
        ),
        "status": "in_progress",
        "scientific_role": (
            "technical_smoke_not_publication_evidence"
            if technical_smoke
            else "post_confirmation_resource_audit_no_accuracy_reanalysis"
        ),
        "created_at_utc": created_at_utc or datetime.now(timezone.utc).isoformat(),
        "input_evidence": {
            "design_manifest_sha256": validation["manifest_sha256"],
            "protocol_sha256": validation["protocol_sha256"],
            "source_lock_sha256": validation["source_lock"]["sha256"],
            "bank_root": str(bank_root),
            "bank_checksum_sha256": bank["checksum_closure"]["sha256"],
            "internal_root": str(internal_root),
            "internal_checksum_sha256": internal_run["checksum_closure"]["sha256"],
            "medicc2_root": str(medicc2_root),
            "medicc2_checksum_sha256": medicc2_run["checksum_closure"]["sha256"],
        },
        "condition_ids": list(STANDARD_CONDITION_IDS),
        "case_ids": [str(case["case_id"]) for case in cases],
        "expected_record_count": len(cases),
        "distance_scope": "one_matrix_over_distinct_states_in_each_condition",
        "distance_semantics": CNP2CNP_SEMANTICS_VERSION,
        "arm_id": M2_ID,
        "equivalence_requirements": {
            "distance_ids_exact": True,
            "distance_matrix_exact": True,
            "tree_summary_exact": True,
        },
        "truth_or_evaluation_loaded": False,
        "resource_contract": dict(resources),
        "cnp2cnp_runtime": dict(cnp2cnp_runtime),
        "internal_runtime": dict(internal_runtime),
        "execution": fresh_process_contract(AUDIT_WORKER_UNIT),
        "completed_record_count": 0,
        "record_inventory": [],
        "status_counts": {},
    }


def _resume_manifest(root: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    path = root / AUDIT_MANIFEST_NAME
    if not path.is_file():
        raise ValueError("A resumed resource audit requires audit_manifest.json.")
    stored = read_json(path)
    stable_fields = (
        "schema_version",
        "audit_id",
        "scientific_role",
        "input_evidence",
        "condition_ids",
        "case_ids",
        "expected_record_count",
        "distance_scope",
        "distance_semantics",
        "arm_id",
        "equivalence_requirements",
        "truth_or_evaluation_loaded",
        "resource_contract",
        "cnp2cnp_runtime",
        "internal_runtime",
        "execution",
    )
    for field in stable_fields:
        if stored.get(field) != expected.get(field):
            raise ValueError(f"Cannot resume because stored {field} changed.")
    if stored.get("status") == "complete":
        validate_checksum_closure(root, AUDIT_CHECKSUM_NAME)
        return stored
    if stored.get("status") not in {"in_progress", "interrupted", "failure"}:
        raise ValueError("Only an incomplete resource audit can resume.")
    inventory = stored.get("record_inventory")
    if not isinstance(inventory, list):
        raise ValueError("Stored resource-audit inventory is invalid.")
    for entry in inventory:
        record_file = root / str(entry["path"])
        if not record_file.is_file() or file_sha256(record_file) != entry.get("sha256"):
            raise ValueError("A stored resource-audit record changed before resume.")
    stored["status"] = "in_progress"
    stored.pop("runner_failure", None)
    stored.setdefault("resume_history", []).append(
        {
            "resumed_at_utc": datetime.now(timezone.utc).isoformat(),
            "preserved_record_count": len(inventory),
        }
    )
    write_json(path, stored)
    return stored


def run_resource_audit(
    *,
    bank_root: Path | str,
    internal_root: Path | str,
    medicc2_root: Path | str,
    output_root: Path | str,
    cnp2cnp_file: Path | str,
    manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
    workers: int = DEFAULT_WORKERS,
    distance_timeout_seconds: int = DEFAULT_DISTANCE_TIMEOUT_SECONDS,
    reconstruction_timeout_seconds: int = DEFAULT_RECONSTRUCTION_TIMEOUT_SECONDS,
    cnp2cnp_process_timeout_seconds: int = DEFAULT_CNP2CNP_PROCESS_TIMEOUT_SECONDS,
    rss_limit_bytes: int = DEFAULT_RSS_LIMIT_BYTES,
    capture_limit_bytes: int = DEFAULT_CAPTURE_LIMIT_BYTES,
    case_limit: int | None = None,
    resume: bool = False,
    progress: bool = False,
) -> dict[str, Any]:
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= MAX_WORKERS:
        raise ValueError(f"workers must be an integer in [1,{MAX_WORKERS}].")
    for name, value in (
        ("distance_timeout_seconds", distance_timeout_seconds),
        ("reconstruction_timeout_seconds", reconstruction_timeout_seconds),
        ("cnp2cnp_process_timeout_seconds", cnp2cnp_process_timeout_seconds),
        ("rss_limit_bytes", rss_limit_bytes),
        ("capture_limit_bytes", capture_limit_bytes),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    if case_limit is not None and (
        isinstance(case_limit, bool) or not isinstance(case_limit, int) or case_limit <= 0
    ):
        raise ValueError("case_limit must be a positive integer when supplied.")
    if case_limit is not None and case_limit > 300:
        raise ValueError("case_limit cannot exceed the 300 standard conditions.")

    validation = validate_manifest(manifest_path, require_approved=True)
    resolved_bank, bank = load_confirmation_bank(bank_root, manifest_path=manifest_path)
    resolved_internal, internal_run = load_confirmation_run(
        internal_root, manifest_path=manifest_path
    )
    resolved_medicc2, medicc2_run = load_medicc2_confirmation(
        medicc2_root, validation=validation
    )
    medicc2_run = dict(medicc2_run)
    medicc2_run["checksum_closure"] = validate_checksum_closure(
        resolved_medicc2,
        MEDICC2_RUN_CHECKSUM_NAME,
    )
    bank_checksum = bank["checksum_closure"]["sha256"]
    if internal_run.get("bank_checksum_sha256") != bank_checksum:
        raise ValueError("The frozen internal run does not belong to the supplied bank.")
    if medicc2_run.get("bank_checksum_sha256") != bank_checksum:
        raise ValueError("The frozen MEDICC2 run does not belong to the supplied bank.")
    standard_cases = _standard_cases(bank)
    if len(standard_cases) != 300:
        raise ValueError("The frozen confirmation bank does not contain 300 standard cases.")
    technical_smoke = case_limit is not None
    cases = standard_cases if case_limit is None else standard_cases[:case_limit]

    runtime_config = load_ctbs_runtime_config()
    executable = Path(cnp2cnp_file).expanduser().resolve()
    runtime_config = replace(
        runtime_config,
        cnp2cnp_folder=str(executable.parent),
        cnp2cnp_file=str(executable),
        cnp2cnp_timeout_seconds=float(cnp2cnp_process_timeout_seconds),
        cnp2cnp_capture_limit_bytes=int(capture_limit_bytes),
    )
    cnp2cnp_runtime = validate_cnp2cnp_source(
        validation["manifest"], runtime_config.cnp2cnp_file
    )
    internal_runtime = validate_internal_runtime(validation["manifest"])
    resource_contract = {
        "distance_timeout_seconds": distance_timeout_seconds,
        "reconstruction_timeout_seconds": reconstruction_timeout_seconds,
        "cnp2cnp_process_timeout_seconds": cnp2cnp_process_timeout_seconds,
        "rss_limit_bytes": rss_limit_bytes,
        "capture_limit_bytes": capture_limit_bytes,
        "requested_workers": workers,
        "evaluation_included": False,
    }
    expected = _new_manifest(
        validation=validation,
        bank_root=resolved_bank,
        bank=bank,
        internal_root=resolved_internal,
        internal_run=internal_run,
        medicc2_root=resolved_medicc2,
        medicc2_run=medicc2_run,
        cases=cases,
        technical_smoke=technical_smoke,
        resources=resource_contract,
        cnp2cnp_runtime=cnp2cnp_runtime,
        internal_runtime=internal_runtime,
    )

    root = Path(output_root).expanduser().resolve()
    if resume:
        if not root.is_dir():
            raise ValueError("A resumed resource audit requires an existing root.")
        audit = _resume_manifest(root, expected)
        if audit.get("status") == "complete":
            return read_json(root / AUDIT_SUMMARY_NAME)
    else:
        root = ensure_new_output_root(root)
        audit = expected
        write_json(root / AUDIT_MANIFEST_NAME, audit)

    expected_case_ids = [str(case["case_id"]) for case in cases]
    completed_case_ids = [str(entry["case_id"]) for entry in audit["record_inventory"]]
    if completed_case_ids != expected_case_ids[: len(completed_case_ids)]:
        raise ValueError("Stored resource-audit records are not the declared prefix.")
    start = len(completed_case_ids)

    def pending_arguments():
        for ordinal, case in enumerate(cases):
            if ordinal < start:
                continue
            yield (
                ordinal,
                str(resolved_bank),
                str(resolved_internal),
                case,
                runtime_config,
                distance_timeout_seconds,
                reconstruction_timeout_seconds,
                rss_limit_bytes,
            )

    try:
        if start < len(cases):
            with FreshSpawnTaskPool(workers, buffer_factor=2) as pool:
                for ordinal, record in pool.map_ordered(
                    _execute_task,
                    pending_arguments(),
                    timeout_seconds=(
                        distance_timeout_seconds
                        + reconstruction_timeout_seconds
                        + WORKER_GRACE_SECONDS
                    ),
                ):
                    if ordinal != len(audit["record_inventory"]):
                        raise RuntimeError("Parallel resource results changed declared order.")
                    relative = _record_relative(str(record["case_id"]))
                    write_json(root / relative, record)
                    audit["record_inventory"].append(
                        {
                            "case_id": record["case_id"],
                            "condition_id": record["condition_id"],
                            "status": record["status"],
                            "path": relative,
                            "sha256": file_sha256(root / relative),
                        }
                    )
                    audit["completed_record_count"] = len(audit["record_inventory"])
                    write_json(root / AUDIT_MANIFEST_NAME, audit)
                    if progress:
                        print(
                            f"{record['case_id']}: {record['status']}",
                            file=sys.stderr,
                            flush=True,
                        )
    except BaseException as error:
        audit["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failure"
        audit["runner_failure"] = _typed_error(error, "resource_audit_runner")
        write_json(root / AUDIT_MANIFEST_NAME, audit)
        raise

    records = [read_json(root / str(entry["path"])) for entry in audit["record_inventory"]]
    status_counts: dict[str, int] = {}
    for record in records:
        status = str(record["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    audit["status_counts"] = status_counts
    if status_counts != {"success": len(cases)}:
        audit["status"] = "complete_with_failures"
        write_json(root / AUDIT_MANIFEST_NAME, audit)
        write_checksum_closure(root, AUDIT_CHECKSUM_NAME)
        raise RuntimeError(
            f"Resource audit completed with failures: {status_counts}."
        )

    summary = build_resource_summary(
        records,
        _medicc2_records(resolved_medicc2, medicc2_run),
    )
    summary.update(
        {
            "scientific_role": audit["scientific_role"],
            "publication_eligible": not technical_smoke,
            "input_evidence": audit["input_evidence"],
            "cnp2cnp_runtime": cnp2cnp_runtime,
            "equivalence": {
                "distance_matrix_exact_count": len(records),
                "tree_summary_exact_count": len(records),
                "failure_count": 0,
            },
        }
    )
    final_cnp2cnp_runtime = validate_cnp2cnp_source(
        validation["manifest"], runtime_config.cnp2cnp_file
    )
    if final_cnp2cnp_runtime != cnp2cnp_runtime:
        raise RuntimeError("cnp2cnp source identity changed during resource audit.")
    if validate_internal_runtime(validation["manifest"]) != internal_runtime:
        raise RuntimeError("Internal runtime identity changed during resource audit.")
    load_confirmation_bank(resolved_bank, manifest_path=manifest_path)
    load_confirmation_run(resolved_internal, manifest_path=manifest_path)
    load_medicc2_confirmation(resolved_medicc2, validation=validation)

    write_json(root / AUDIT_SUMMARY_NAME, summary)
    _write_resource_csv(
        root / AUDIT_RESOURCE_TABLE_NAME,
        _resource_rows(summary),
    )

    audit["status"] = "complete"
    audit["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    audit.pop("runner_failure", None)
    write_json(root / AUDIT_MANIFEST_NAME, audit)
    write_checksum_closure(root, AUDIT_CHECKSUM_NAME)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--bank-root", type=Path, default=DEFAULT_CONFIRMATION_ROOT / "bank")
    parser.add_argument(
        "--internal-root", type=Path, default=DEFAULT_CONFIRMATION_ROOT / "internal"
    )
    parser.add_argument(
        "--medicc2-root", type=Path, default=DEFAULT_CONFIRMATION_ROOT / "medicc2"
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cnp2cnp-file", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--distance-timeout-seconds",
        type=int,
        default=DEFAULT_DISTANCE_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--reconstruction-timeout-seconds",
        type=int,
        default=DEFAULT_RECONSTRUCTION_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--cnp2cnp-process-timeout-seconds",
        type=int,
        default=DEFAULT_CNP2CNP_PROCESS_TIMEOUT_SECONDS,
    )
    parser.add_argument("--rss-limit-bytes", type=int, default=DEFAULT_RSS_LIMIT_BYTES)
    parser.add_argument(
        "--capture-limit-bytes", type=int, default=DEFAULT_CAPTURE_LIMIT_BYTES
    )
    parser.add_argument(
        "--case-limit",
        type=int,
        help="Run a non-publication technical prefix only.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    summary = run_resource_audit(
        bank_root=arguments.bank_root,
        internal_root=arguments.internal_root,
        medicc2_root=arguments.medicc2_root,
        output_root=arguments.output_root,
        cnp2cnp_file=arguments.cnp2cnp_file,
        manifest_path=arguments.manifest,
        workers=arguments.workers,
        distance_timeout_seconds=arguments.distance_timeout_seconds,
        reconstruction_timeout_seconds=arguments.reconstruction_timeout_seconds,
        cnp2cnp_process_timeout_seconds=arguments.cnp2cnp_process_timeout_seconds,
        rss_limit_bytes=arguments.rss_limit_bytes,
        capture_limit_bytes=arguments.capture_limit_bytes,
        case_limit=arguments.case_limit,
        resume=arguments.resume,
        progress=arguments.progress,
    )
    print(
        "complete: condition-specific M2 resource audit; "
        f"{summary['case_count']} cases; "
        f"{summary['equivalence']['distance_matrix_exact_count']} exact matrices; "
        f"{summary['equivalence']['tree_summary_exact_count']} exact reconstructions"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
