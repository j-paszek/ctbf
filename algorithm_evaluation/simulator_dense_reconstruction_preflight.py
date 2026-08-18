"""Single-case operational preflight for dense CTBF v5 reconstruction.

The registered production case is the largest 50%-sampling case observed in
the completed truth-only fraction probe: H34, zero-based replicate index 11.
The module re-simulates that deterministic truth, selects representative
genotype states with ``min(N, max(6, ceil(0.5*N)))``, validates the compact
selection against the completed truth and capped-six reconstruction reports,
then runs production cnp2cnp and all six established reconstruction arms.

This is a technical feasibility gate.  It writes compact summaries only and
does not provide paper evidence or an accuracy estimate.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from typing import Any, Callable, Iterable, Mapping, Sequence

from algorithm_evaluation.hypothesis_height_ambiguity_trend import (
    _distance_identity,
    _sha256_json,
    ambiguity_case_summary,
)
from algorithm_evaluation.hypothesis_trend_distance_preflight import (
    _validate_production_distance,
    distance_matrix_summary,
)
from algorithm_evaluation.paper_pipeline_contract import (
    PROJECT_ROOT,
    canonical_json_sha256,
    json_safe,
    read_json,
    write_json_atomic,
)
from algorithm_evaluation.paper_pipeline_runner import (
    RECONSTRUCTION_INPUT_SCHEMA_VERSION,
    measured_stage,
    validate_reconstruction_input,
)
from algorithm_evaluation.simulator_growth_probe import (
    _file_sha256,
    _validate_standard_base_config,
)
from algorithm_evaluation.simulator_reconstruction_intuition_probe import (
    ANALYSIS_ROLE as SPARSE_ANALYSIS_ROLE,
    APPROVED_SCHEDULES,
    ARM_IDS,
    DEFAULT_BASE_CONFIG,
    DEFAULT_BASE_SEED,
    DEFAULT_CAPTURE_LIMIT_BYTES,
    DEFAULT_FIXED_RADIUS,
    DEFAULT_RSS_LIMIT_BYTES,
    SCHEMA_VERSION as SPARSE_SCHEMA_VERSION,
    _canonical_cells_at_generation,
    _event_totals,
    _profile,
    _reconstruction_seed,
    _run_arm,
    _simulation_seed,
    _truth_prefix_sha256,
    validate_report as validate_sparse_report,
)
from algorithm_evaluation.simulator_sampling_fraction_truth_probe import (
    ANALYSIS_ROLE as FRACTION_TRUTH_ANALYSIS_ROLE,
    DEFAULT_LOWER_BOUND,
    SAMPLING_VERSION,
    SCHEMA_VERSION as FRACTION_TRUTH_SCHEMA_VERSION,
    hybrid_sample_size,
    select_nested_fraction_levels,
    validate_report as validate_fraction_truth_report,
)
from ctbs import (
    Cnp2CnpFileDistanceProvider,
    DistanceMatrix,
    load_ctbs_runtime_config,
    unique_cells_by_cell_id,
)
from distance_semantics import CNP2CNP_SEMANTICS_VERSION, stable_distance_label_key
from reconstructor_temporal import TEMPORAL_ARBORESCENCE_SOLVER_VERSION
from simulator import CancerCellEvolutionSimulator


SCHEMA_VERSION = "ctbf-v5-dense-reconstruction-preflight-v1"
ANALYSIS_ROLE = "nonpaper_dense_reconstruction_operational_preflight"
CONDITION_ID = "fraction_50"
TARGET_FRACTION = 0.5

REGISTERED_HEIGHT = 34
REGISTERED_REPLICATE_INDEX = 11
EXPECTED_UNIQUE_STATE_COUNT = 329
EXPECTED_ORDERED_PAIR_COUNT = 107_912

EXPECTED_FRACTION_TRUTH_SHA256 = (
    "1734a2fb6d098b4d56faedec580aabfd0c5da81b254137dbf42a054c2a6c8dfa"
)
EXPECTED_SPARSE_RECONSTRUCTION_SHA256 = (
    "7ec75b8bc6b86b045d759c38fd657dd0e6021e22dbc5a16ea49609a506511b08"
)

DEFAULT_SIMULATION_TIMEOUT_SECONDS = 300
DEFAULT_DISTANCE_TIMEOUT_SECONDS = 1200
DEFAULT_DIAGNOSTIC_TIMEOUT_SECONDS = 300
DEFAULT_RECONSTRUCTION_TIMEOUT_SECONDS = 600
DEFAULT_EVALUATION_TIMEOUT_SECONDS = 300
DEFAULT_CNP2CNP_PROCESS_TIMEOUT_SECONDS = 540


DistanceCompute = Callable[[Sequence[Any]], DistanceMatrix]


def _validate_positive_integer(value: Any, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer.")


def _index_cases(report: Mapping[str, Any]) -> dict[tuple[int, int], Mapping[str, Any]]:
    index = {}
    for case in report.get("cases", []):
        key = (int(case["replicate_index"]), int(case["height"]))
        if key in index:
            raise ValueError(f"Reference report contains duplicate case {key}.")
        index[key] = case
    return index


def _validate_report_source_identity(report: Mapping[str, Any]) -> None:
    provenance = report.get("provenance", {})
    relative = provenance.get("source_module")
    expected = provenance.get("source_module_sha256")
    if not isinstance(relative, str) or not isinstance(expected, str):
        raise ValueError("Reference report lacks source-module identity.")
    source = (PROJECT_ROOT / relative).resolve()
    if not source.is_relative_to(PROJECT_ROOT):
        raise ValueError("Reference report source path escapes the project root.")
    if not source.is_file() or _file_sha256(source) != expected:
        raise ValueError("Reference report source identity differs from current code.")


def load_references(
    *,
    fraction_truth_report_path: Path | str,
    sparse_reconstruction_report_path: Path | str,
    base_config_sha256: str,
    base_seed: int,
    enforce_registered_hashes: bool = True,
) -> tuple[
    dict[tuple[int, int], Mapping[str, Any]],
    dict[tuple[int, int], Mapping[str, Any]],
    dict[str, Any],
]:
    fraction_path = Path(fraction_truth_report_path).expanduser().resolve()
    sparse_path = Path(sparse_reconstruction_report_path).expanduser().resolve()
    for label, path in (
        ("fraction-truth", fraction_path),
        ("sparse-reconstruction", sparse_path),
    ):
        if not path.is_file():
            raise ValueError(f"{label} reference is not a file: {path}.")

    fraction_hash = _file_sha256(fraction_path)
    sparse_hash = _file_sha256(sparse_path)
    if enforce_registered_hashes and fraction_hash != EXPECTED_FRACTION_TRUTH_SHA256:
        raise ValueError("Fraction-truth reference is not the registered artifact.")
    if (
        enforce_registered_hashes
        and sparse_hash != EXPECTED_SPARSE_RECONSTRUCTION_SHA256
    ):
        raise ValueError("Sparse reconstruction reference is not registered.")

    fraction = read_json(fraction_path)
    sparse = read_json(sparse_path)
    validate_fraction_truth_report(fraction)
    validate_sparse_report(sparse)
    if (
        fraction.get("schema_version") != FRACTION_TRUTH_SCHEMA_VERSION
        or fraction.get("analysis_role") != FRACTION_TRUTH_ANALYSIS_ROLE
        or fraction.get("status") != "complete"
    ):
        raise ValueError("Fraction-truth reference does not have completed identity.")
    if (
        sparse.get("schema_version") != SPARSE_SCHEMA_VERSION
        or sparse.get("analysis_role") != SPARSE_ANALYSIS_ROLE
        or sparse.get("status") != "complete"
    ):
        raise ValueError("Sparse reference does not have completed identity.")
    for label, report in (("fraction", fraction), ("sparse", sparse)):
        if report["input"]["base_config_sha256"] != base_config_sha256:
            raise ValueError(f"{label} reference uses a different base config.")
        if report["input"]["base_seed"] != base_seed:
            raise ValueError(f"{label} reference uses a different base seed.")
        if report["input"]["heights"] != list(APPROVED_SCHEDULES):
            raise ValueError(f"{label} reference uses different heights.")
        _validate_report_source_identity(report)
    if not fraction["reference"]["all_performed_checks_passed"]:
        raise ValueError("Fraction-truth reference did not pass its sparse controls.")
    if not fraction["aggregate"]["common_seed_prefix_consistency"][
        "all_planned_common_seed_prefix_checks_available_and_passed"
    ]:
        raise ValueError("Fraction-truth reference failed common truth prefixes.")

    fraction_index = _index_cases(fraction)
    sparse_index = _index_cases(sparse)
    expected_keys = {
        (replicate_index, height)
        for replicate_index in range(12)
        for height in APPROVED_SCHEDULES
    }
    if set(fraction_index) != expected_keys or set(sparse_index) != expected_keys:
        raise ValueError("Reference reports do not contain the registered 36 cases.")
    if any(case["status"] != "complete" for case in fraction_index.values()):
        raise ValueError("Fraction-truth reference contains an incomplete case.")
    if any(case["status"] != "complete" for case in sparse_index.values()):
        raise ValueError("Sparse reconstruction reference contains an incomplete case.")

    dense_unique_counts = {
        key: case["simulation_summary"]["conditions"][CONDITION_ID]["summary"][
            "selected_unique_state_count"
        ]
        for key, case in fraction_index.items()
    }
    dense_ordered_pair_counts = {
        key: case["simulation_summary"]["conditions"][CONDITION_ID]["summary"][
            "projected_bidirectional_ordered_pair_count"
        ]
        for key, case in fraction_index.items()
    }
    registered_key = (REGISTERED_REPLICATE_INDEX, REGISTERED_HEIGHT)
    if dense_unique_counts[registered_key] != EXPECTED_UNIQUE_STATE_COUNT:
        raise ValueError("Registered preflight case no longer has 329 unique states.")
    if max(dense_unique_counts.values()) != EXPECTED_UNIQUE_STATE_COUNT:
        raise ValueError("Registered preflight case is no longer the projected maximum.")
    if dense_ordered_pair_counts[registered_key] != EXPECTED_ORDERED_PAIR_COUNT:
        raise ValueError("Registered preflight case has a different pair bound.")

    metadata = {
        "fraction_truth": {
            "path": str(fraction_path),
            "sha256": fraction_hash,
            "schema_version": fraction["schema_version"],
            "case_count": len(fraction_index),
        },
        "sparse_reconstruction": {
            "path": str(sparse_path),
            "sha256": sparse_hash,
            "schema_version": sparse["schema_version"],
            "case_count": len(sparse_index),
        },
        "current_source_identities_match": True,
        "same_base_config_and_seed": True,
    }
    return fraction_index, sparse_index, metadata


def _reconstruction_input(
    case_key: str,
    height: int,
    generations: Sequence[int],
    selected_levels: Sequence[Sequence[Any]],
) -> dict[str, Any]:
    payload = {
        "schema_version": RECONSTRUCTION_INPUT_SCHEMA_VERSION,
        "case_id": case_key,
        "condition_id": f"H{height}_rule_y_fraction50_lower6",
        "sampling_rule": SAMPLING_VERSION,
        "levels": [
            {
                "biopsy_level": level_index,
                "generation": int(generation),
                "states": [
                    {
                        "state_label": json_safe(cell.cell_id),
                        "cnp": list(_profile(cell)),
                    }
                    for cell in cells
                ],
            }
            for level_index, (generation, cells) in enumerate(
                zip(generations, selected_levels)
            )
        ],
    }
    validate_reconstruction_input(payload)
    return payload


def _reference_check(
    *,
    observed: Mapping[str, Any],
    fraction_case: Mapping[str, Any] | None,
    sparse_case: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if fraction_case is None or sparse_case is None:
        return {"performed": False, "passed": None, "mismatched_fields": []}
    fraction_summary = fraction_case["simulation_summary"]
    dense_expected = fraction_summary["conditions"][CONDITION_ID]["summary"]
    capped_expected = fraction_summary["conditions"]["capped_six_control"][
        "summary"
    ]
    sparse_expected = sparse_case["simulation_summary"]
    comparisons = {
        "truth_prefix_sha256_by_height": (
            observed["truth_prefix_sha256_by_height"],
            fraction_summary["truth_prefix_sha256_by_height"],
        ),
        "truth_node_count": (
            observed["truth_node_count"],
            fraction_summary["truth_node_count"],
        ),
        "truth_edge_count": (
            observed["truth_edge_count"],
            fraction_summary["truth_edge_count"],
        ),
        "available_distinct_state_count_by_generation": (
            observed["available_distinct_state_count_by_generation"],
            fraction_summary["available_distinct_state_count_by_generation"],
        ),
        "dense_sampling_rows": (observed["sampling"], dense_expected["sampling"]),
        "dense_selected_occurrence_count": (
            observed["selected_occurrence_count"],
            dense_expected["selected_occurrence_count"],
        ),
        "dense_selected_unique_state_count": (
            observed["selected_unique_state_count"],
            dense_expected["selected_unique_state_count"],
        ),
        "dense_projected_ordered_pair_count": (
            observed["bidirectional_ordered_pair_bound"],
            dense_expected["projected_bidirectional_ordered_pair_count"],
        ),
        "capped_realized_counts": (
            observed["capped_control_realized_counts"],
            [row["realized_occurrence_count"] for row in capped_expected["sampling"]],
        ),
        "capped_unique_state_count": (
            observed["capped_control_unique_state_count"],
            sparse_expected["selected_unique_state_count"],
        ),
        "sparse_truth_prefix_sha256_by_height": (
            observed["truth_prefix_sha256_by_height"],
            sparse_expected["truth_prefix_sha256_by_height"],
        ),
    }
    mismatches = [
        field for field, (actual, expected) in comparisons.items() if actual != expected
    ]
    return {
        "performed": True,
        "passed": not mismatches,
        "mismatched_fields": mismatches,
        "fraction_truth_condition_summary_sha256": canonical_json_sha256(
            dense_expected
        ),
        "sparse_case_summary_sha256": canonical_json_sha256(sparse_expected),
    }


def _prepare_case(
    *,
    config_path: Path,
    simulation_seed: int,
    base_seed: int,
    replicate_index: int,
    height: int,
    generations: Sequence[int],
    fraction_case: Mapping[str, Any] | None,
    sparse_case: Mapping[str, Any] | None,
) -> dict[str, Any]:
    simulator = CancerCellEvolutionSimulator(str(config_path), seed=simulation_seed)
    simulator.run_simulation()
    available_levels = [
        _canonical_cells_at_generation(simulator, generation)
        for generation in generations
    ]
    selected, sampling, nesting = select_nested_fraction_levels(
        available_levels,
        generations,
        base_seed=base_seed,
        replicate_index=replicate_index,
        height=height,
        lower_bound=DEFAULT_LOWER_BOUND,
    )
    selected_levels = selected[CONDITION_ID]
    capped_levels = selected["capped_six_control"]
    occurrences = [cell for level in selected_levels for cell in level]
    distance_cells = sorted(
        unique_cells_by_cell_id(occurrences),
        key=lambda cell: stable_distance_label_key(cell.cell_id),
    )
    capped_cells = unique_cells_by_cell_id(
        [cell for level in capped_levels for cell in level]
    )
    if not distance_cells:
        raise ValueError("Dense sampling produced no distance state.")
    if not all(row["nested"] for row in nesting):
        raise ValueError("Dense sampling violated the nested-prefix contract.")

    case_key = f"v5-dense-preflight-H{height}-R{replicate_index + 1:03d}"
    truth_tree = simulator.canonicalized_tree_by_genome()
    prefix_hashes = {
        str(prefix_height): _truth_prefix_sha256(simulator.tree, prefix_height)
        for prefix_height in APPROVED_SCHEDULES
        if prefix_height <= height
    }
    summary = {
        "truth_node_count": truth_tree.number_of_nodes(),
        "truth_edge_count": truth_tree.number_of_edges(),
        "available_distinct_state_count_by_generation": [
            {"generation": int(generation), "count": len(level)}
            for generation, level in zip(generations, available_levels)
        ],
        "sampling": sampling[CONDITION_ID],
        "selected_occurrence_count": len(occurrences),
        "selected_unique_state_count": len(distance_cells),
        "repeated_state_occurrence_count": len(occurrences) - len(distance_cells),
        "distance_matrix_cell_count": len(distance_cells) ** 2,
        "bidirectional_ordered_pair_bound": (
            len(distance_cells) * (len(distance_cells) - 1)
        ),
        "capped_control_realized_counts": [len(level) for level in capped_levels],
        "capped_control_unique_state_count": len(capped_cells),
        "truth_prefix_sha256_by_height": prefix_hashes,
        "simulator_event_totals": _event_totals(simulator),
        "all_nested_checks_passed": True,
    }
    reference_check = _reference_check(
        observed=summary,
        fraction_case=fraction_case,
        sparse_case=sparse_case,
    )
    if reference_check["performed"] and not reference_check["passed"]:
        raise ValueError(
            "Dense case differs from registered references: "
            + ", ".join(reference_check["mismatched_fields"])
        )
    return {
        "distance_cells": distance_cells,
        "levels": selected_levels,
        "reconstruction_input": _reconstruction_input(
            case_key,
            height,
            generations,
            selected_levels,
        ),
        "true_tree": truth_tree,
        "summary": summary,
        "reference_check": reference_check,
    }


def _typed_error(error: BaseException | None) -> dict[str, Any] | None:
    if error is None:
        return None
    message = str(error)
    return {
        "type": type(error).__name__,
        "message_character_count": len(message),
        "message_sha256": hashlib.sha256(
            message.encode("utf-8", errors="replace")
        ).hexdigest(),
    }


def _run_case(
    *,
    base_config: Mapping[str, Any],
    height: int,
    replicate_index: int,
    base_seed: int,
    fraction_case: Mapping[str, Any] | None,
    sparse_case: Mapping[str, Any] | None,
    distance_compute: DistanceCompute,
    injected_distance: bool,
    simulation_timeout_seconds: int,
    distance_timeout_seconds: int,
    diagnostic_timeout_seconds: int,
    reconstruction_timeout_seconds: int,
    evaluation_timeout_seconds: int,
    rss_limit_bytes: int,
    progress: bool,
) -> dict[str, Any]:
    generations = APPROVED_SCHEDULES[height]
    simulation_seed = _simulation_seed(base_seed, replicate_index)
    reconstruction_seed = _reconstruction_seed(base_seed, replicate_index, height)
    case_key = f"v5-dense-preflight-H{height}-R{replicate_index + 1:03d}"
    config = dict(base_config)
    config["NUMBER_OF_GENERATIONS"] = height
    with tempfile.TemporaryDirectory(prefix="ctbf-v5-dense-preflight-") as directory:
        config_path = Path(directory) / "case.json"
        write_json_atomic(config_path, config)
        prepared, simulation_runtime, simulation_error = measured_stage(
            lambda: _prepare_case(
                config_path=config_path,
                simulation_seed=simulation_seed,
                base_seed=base_seed,
                replicate_index=replicate_index,
                height=height,
                generations=generations,
                fraction_case=fraction_case,
                sparse_case=sparse_case,
            ),
            timeout_seconds=simulation_timeout_seconds,
            rss_limit_bytes=rss_limit_bytes,
        )

    record = {
        "case_key": case_key,
        "height": height,
        "replicate_index": replicate_index,
        "simulation_seed": simulation_seed,
        "reconstruction_seed": reconstruction_seed,
        "generations": list(generations),
        "status": None,
        "error": _typed_error(simulation_error),
        "simulation_runtime": simulation_runtime,
        "simulation_summary": None if prepared is None else prepared["summary"],
        "reference_check": None if prepared is None else prepared["reference_check"],
        "distance": None,
        "ambiguity": None,
        "diagnostic_runtime": None,
        "arms": [],
    }
    if simulation_error is not None:
        record["status"] = "simulation_or_reference_failure"
        return record
    if progress:
        print(
            json.dumps(
                {
                    "case_key": case_key,
                    "stage": "simulation_and_reference",
                    "status": "success",
                    "unique_states": prepared["summary"][
                        "selected_unique_state_count"
                    ],
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )

    distance, distance_runtime, distance_error = measured_stage(
        lambda: distance_compute(prepared["distance_cells"]),
        timeout_seconds=distance_timeout_seconds,
        rss_limit_bytes=rss_limit_bytes,
    )
    if distance_error is None:
        try:
            _validate_production_distance(distance)
        except Exception as error:
            distance_error = error
            distance = None
    if distance_error is not None:
        record["status"] = "distance_failure"
        record["error"] = _typed_error(distance_error)
        record["distance"] = {"runtime": distance_runtime}
        return record
    identity = _distance_identity(distance.provenance)
    record["distance"] = {
        "runtime": distance_runtime,
        "summary": distance_matrix_summary(distance),
        "identity": identity,
        "identity_sha256": _sha256_json(identity),
        "external_process_count": distance.provenance.get("external_process_count"),
        "injected_distance_for_test": injected_distance,
    }
    if progress:
        print(
            json.dumps(
                {"case_key": case_key, "stage": "distance", "status": "success"},
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )

    ambiguity, diagnostic_runtime, diagnostic_error = measured_stage(
        lambda: ambiguity_case_summary(
            prepared["levels"],
            generations,
            distance,
            fixed_radius=DEFAULT_FIXED_RADIUS,
        ),
        timeout_seconds=diagnostic_timeout_seconds,
        rss_limit_bytes=rss_limit_bytes,
    )
    record["diagnostic_runtime"] = diagnostic_runtime
    if diagnostic_error is not None:
        record["status"] = "diagnostic_failure"
        record["error"] = _typed_error(diagnostic_error)
        return record
    record["ambiguity"] = ambiguity

    for arm_id in ARM_IDS:
        arm = _run_arm(
            arm_id=arm_id,
            reconstruction_input=prepared["reconstruction_input"],
            distance=distance,
            true_tree=prepared["true_tree"],
            reconstruction_seed=reconstruction_seed,
            reconstruction_timeout_seconds=reconstruction_timeout_seconds,
            evaluation_timeout_seconds=evaluation_timeout_seconds,
            rss_limit_bytes=rss_limit_bytes,
        )
        record["arms"].append(arm)
        if progress:
            print(
                json.dumps(
                    {
                        "case_key": case_key,
                        "stage": "arm",
                        "arm_id": arm_id,
                        "status": arm["status"],
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
                flush=True,
            )
    record["status"] = "complete"
    del prepared
    gc.collect()
    return record


def _runtime_utilization(runtime: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if runtime is None:
        return None
    timeout = runtime.get("timeout_seconds")
    limit = runtime.get("rss_limit_bytes")
    peak = runtime.get("memory", {}).get("peak_rss_bytes")
    seconds = runtime.get("wall_time_ns", 0) / 1_000_000_000
    return {
        "wall_time_seconds": seconds,
        "timeout_utilization": None if timeout is None else seconds / timeout,
        "peak_rss_bytes": peak,
        "rss_limit_utilization": (
            None if peak is None or not limit else peak / limit
        ),
    }


def _preflight_verdict(record: Mapping[str, Any], injected_distance: bool) -> str:
    if injected_distance:
        if record["status"] != "complete":
            return f"technical_smoke_failed_{record['status']}"
        if any(arm["status"] != "success" for arm in record["arms"]):
            return "technical_smoke_failed_one_or_more_arms"
        return "technical_smoke_only_injected_distance"
    if record["status"] != "complete":
        return f"reject_dependency_{record['status']}"
    if any(arm["status"] != "success" for arm in record["arms"]):
        return "reject_one_or_more_reconstruction_arms_failed"
    return "pass_all_registered_stages_pending_owner_runtime_review"


def run_preflight(
    *,
    base_config_path: Path | str = DEFAULT_BASE_CONFIG,
    fraction_truth_report_path: Path | str | None = None,
    sparse_reconstruction_report_path: Path | str | None = None,
    base_seed: int = DEFAULT_BASE_SEED,
    height: int = REGISTERED_HEIGHT,
    replicate_index: int = REGISTERED_REPLICATE_INDEX,
    simulation_timeout_seconds: int = DEFAULT_SIMULATION_TIMEOUT_SECONDS,
    distance_timeout_seconds: int = DEFAULT_DISTANCE_TIMEOUT_SECONDS,
    diagnostic_timeout_seconds: int = DEFAULT_DIAGNOSTIC_TIMEOUT_SECONDS,
    reconstruction_timeout_seconds: int = DEFAULT_RECONSTRUCTION_TIMEOUT_SECONDS,
    evaluation_timeout_seconds: int = DEFAULT_EVALUATION_TIMEOUT_SECONDS,
    cnp2cnp_process_timeout_seconds: int = DEFAULT_CNP2CNP_PROCESS_TIMEOUT_SECONDS,
    rss_limit_bytes: int = DEFAULT_RSS_LIMIT_BYTES,
    capture_limit_bytes: int = DEFAULT_CAPTURE_LIMIT_BYTES,
    distance_compute: DistanceCompute | None = None,
    enforce_registered_references: bool = True,
    created_at_utc: str | None = None,
    progress: bool = False,
) -> dict[str, Any]:
    base_config_path = Path(base_config_path).expanduser().resolve()
    if not base_config_path.is_file():
        raise ValueError(f"Base config is not a file: {base_config_path}.")
    base_config = read_json(base_config_path)
    _validate_standard_base_config(base_config)
    if height not in APPROVED_SCHEDULES:
        raise ValueError("Preflight height must be one approved probe height.")
    if (
        isinstance(replicate_index, bool)
        or not isinstance(replicate_index, int)
        or not 0 <= replicate_index < 12
    ):
        raise ValueError("replicate_index must be in [0,11].")
    if isinstance(base_seed, bool) or not isinstance(base_seed, int) or base_seed < 0:
        raise ValueError("base_seed must be a nonnegative integer.")
    if distance_compute is None and (
        height != REGISTERED_HEIGHT
        or replicate_index != REGISTERED_REPLICATE_INDEX
        or base_seed != DEFAULT_BASE_SEED
    ):
        raise ValueError("Production preflight is frozen to H34 replicate index 11.")
    for field, value in (
        ("simulation_timeout_seconds", simulation_timeout_seconds),
        ("distance_timeout_seconds", distance_timeout_seconds),
        ("diagnostic_timeout_seconds", diagnostic_timeout_seconds),
        ("reconstruction_timeout_seconds", reconstruction_timeout_seconds),
        ("evaluation_timeout_seconds", evaluation_timeout_seconds),
        ("cnp2cnp_process_timeout_seconds", cnp2cnp_process_timeout_seconds),
        ("rss_limit_bytes", rss_limit_bytes),
        ("capture_limit_bytes", capture_limit_bytes),
    ):
        _validate_positive_integer(value, field)

    fraction_index: dict[tuple[int, int], Mapping[str, Any]] = {}
    sparse_index: dict[tuple[int, int], Mapping[str, Any]] = {}
    reference_metadata = {
        "provided": False,
        "registered_hashes_enforced": False,
    }
    if (fraction_truth_report_path is None) != (
        sparse_reconstruction_report_path is None
    ):
        raise ValueError("Both compact references must be provided together.")
    if fraction_truth_report_path is not None:
        fraction_index, sparse_index, loaded_metadata = load_references(
            fraction_truth_report_path=fraction_truth_report_path,
            sparse_reconstruction_report_path=sparse_reconstruction_report_path,
            base_config_sha256=_file_sha256(base_config_path),
            base_seed=base_seed,
            enforce_registered_hashes=enforce_registered_references,
        )
        reference_metadata = {
            "provided": True,
            "registered_hashes_enforced": enforce_registered_references,
            **loaded_metadata,
        }
    if distance_compute is None and not reference_metadata["provided"]:
        raise ValueError("Production preflight requires both registered references.")

    if distance_compute is None:
        runtime_config = replace(
            load_ctbs_runtime_config(),
            cnp2cnp_timeout_seconds=float(cnp2cnp_process_timeout_seconds),
            cnp2cnp_capture_limit_bytes=int(capture_limit_bytes),
        )
        compute = Cnp2CnpFileDistanceProvider(runtime_config).compute
    else:
        compute = distance_compute
    key = (replicate_index, height)
    record = _run_case(
        base_config=base_config,
        height=height,
        replicate_index=replicate_index,
        base_seed=base_seed,
        fraction_case=fraction_index.get(key),
        sparse_case=sparse_index.get(key),
        distance_compute=compute,
        injected_distance=distance_compute is not None,
        simulation_timeout_seconds=simulation_timeout_seconds,
        distance_timeout_seconds=distance_timeout_seconds,
        diagnostic_timeout_seconds=diagnostic_timeout_seconds,
        reconstruction_timeout_seconds=reconstruction_timeout_seconds,
        evaluation_timeout_seconds=evaluation_timeout_seconds,
        rss_limit_bytes=rss_limit_bytes,
        progress=progress,
    )
    verdict = _preflight_verdict(record, distance_compute is not None)
    arm_status_counts = dict(
        sorted(Counter(arm["status"] for arm in record["arms"]).items())
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "analysis_role": ANALYSIS_ROLE,
        "created_at_utc": created_at_utc or datetime.now(timezone.utc).isoformat(),
        "status": "complete" if record["status"] == "complete" else "failed",
        "preflight_verdict": verdict,
        "scientific_role": {
            "paper_evidence_allowed": False,
            "discovery_only": False,
            "technical_preflight_only": True,
            "simulation_run": True,
            "cnp2cnp_run": distance_compute is None,
            "injected_distance_for_test": distance_compute is not None,
            "reconstruction_run": True,
            "evaluation_run": True,
            "accuracy_interpretation_allowed": False,
            "authorizes_full_probe": False,
        },
        "question": {
            "primary": (
                "Can the projected largest 50%-sampling H34 case complete "
                "production cnp2cnp and all six registered arms within bounds?"
            ),
            "not_an_accuracy_test": True,
            "not_a_paper_height_or_sampling_freeze": True,
        },
        "input": {
            "base_config": (
                base_config_path.relative_to(PROJECT_ROOT).as_posix()
                if base_config_path.is_relative_to(PROJECT_ROOT)
                else str(base_config_path)
            ),
            "base_config_sha256": _file_sha256(base_config_path),
            "base_seed": base_seed,
            "height": height,
            "replicate_index": replicate_index,
            "generations": list(APPROVED_SCHEDULES[height]),
            "condition_id": CONDITION_ID,
            "target_representative_state_fraction": TARGET_FRACTION,
            "biopsy_lower_bound": DEFAULT_LOWER_BOUND,
            "sample_size_formula": "min(N,max(6,ceil(0.5*N)))",
            "sampling_version": SAMPLING_VERSION,
            "distance_semantics": CNP2CNP_SEMANTICS_VERSION,
            "arm_ids": list(ARM_IDS),
        },
        "registered_case": {
            "height": REGISTERED_HEIGHT,
            "replicate_index": REGISTERED_REPLICATE_INDEX,
            "expected_unique_state_count": EXPECTED_UNIQUE_STATE_COUNT,
            "expected_bidirectional_ordered_pair_count": (
                EXPECTED_ORDERED_PAIR_COUNT
            ),
            "selected_as_maximum_from_fraction_truth_artifact": True,
        },
        "references": reference_metadata,
        "resource_bound": {
            "case_count": 1,
            "sequential_execution": True,
            "simulation_timeout_seconds": simulation_timeout_seconds,
            "distance_timeout_seconds": distance_timeout_seconds,
            "diagnostic_timeout_seconds": diagnostic_timeout_seconds,
            "reconstruction_timeout_seconds_per_arm": (
                reconstruction_timeout_seconds
            ),
            "evaluation_timeout_seconds_per_arm": evaluation_timeout_seconds,
            "cnp2cnp_process_timeout_seconds": cnp2cnp_process_timeout_seconds,
            "rss_limit_bytes_per_stage": rss_limit_bytes,
            "capture_limit_bytes": capture_limit_bytes,
        },
        "case": record,
        "summary": {
            "case_status": record["status"],
            "arm_status_counts": arm_status_counts,
            "simulation_runtime": _runtime_utilization(
                record["simulation_runtime"]
            ),
            "distance_runtime": _runtime_utilization(
                None if record["distance"] is None else record["distance"]["runtime"]
            ),
            "diagnostic_runtime": _runtime_utilization(
                record["diagnostic_runtime"]
            ),
            "arms": {
                arm["arm_id"]: {
                    "status": arm["status"],
                    "error": arm["error"],
                    "reconstruction_runtime": _runtime_utilization(
                        arm["reconstruction_runtime"]
                    ),
                    "evaluation_runtime": _runtime_utilization(
                        arm["evaluation_runtime"]
                    ),
                }
                for arm in record["arms"]
            },
        },
        "provenance": {
            "source_module": Path(__file__).relative_to(PROJECT_ROOT).as_posix(),
            "source_module_sha256": _file_sha256(Path(__file__)),
            "temporal_solver": {
                "implementation_version": TEMPORAL_ARBORESCENCE_SOLVER_VERSION,
                "source_module": "reconstructor_temporal.py",
                "source_module_sha256": _file_sha256(
                    PROJECT_ROOT / "reconstructor_temporal.py"
                ),
            },
            "reads_completed_compact_reference_reports": (
                reference_metadata["provided"]
            ),
            "writes_raw_profiles": False,
            "writes_truth_or_reconstructed_trees": False,
            "writes_distance_matrices": False,
            "writes_simulator_node_identities": False,
        },
    }
    validate_report(report)
    return report


def _walk_keys(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key)
            yield from _walk_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_keys(item)


def validate_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unknown dense-reconstruction preflight schema.")
    if report.get("analysis_role") != ANALYSIS_ROLE:
        raise ValueError("Dense-reconstruction preflight has the wrong role.")
    if report.get("status") not in {"complete", "failed"}:
        raise ValueError("Dense-reconstruction preflight has unknown status.")
    role = report.get("scientific_role", {})
    expected_role = {
        "paper_evidence_allowed": False,
        "technical_preflight_only": True,
        "reconstruction_run": True,
        "evaluation_run": True,
        "accuracy_interpretation_allowed": False,
        "authorizes_full_probe": False,
    }
    for field, expected in expected_role.items():
        if role.get(field) is not expected:
            raise ValueError(f"Scientific role has invalid {field}.")
    case = report.get("case", {})
    if case.get("status") not in {
        "complete",
        "simulation_or_reference_failure",
        "distance_failure",
        "diagnostic_failure",
    }:
        raise ValueError("Dense-reconstruction preflight case has unknown status.")
    if case.get("status") == "complete":
        if tuple(arm.get("arm_id") for arm in case.get("arms", [])) != ARM_IDS:
            raise ValueError("Complete dense case has the wrong arm portfolio.")
        if case.get("simulation_summary") is None or case.get("distance") is None:
            raise ValueError("Complete dense case lacks required dependencies.")
        for row in case["simulation_summary"]["sampling"]:
            expected = hybrid_sample_size(
                row["available_distinct_state_count"],
                TARGET_FRACTION,
                DEFAULT_LOWER_BOUND,
            )
            if row["realized_occurrence_count"] != expected:
                raise ValueError("Dense sample size differs from its formula.")
        for arm in case["arms"]:
            if arm.get("status") not in {
                "success",
                "reconstruction_failure",
                "evaluation_failure",
            }:
                raise ValueError("Dense reconstruction arm has unknown status.")
    if (report["status"] == "complete") != (case.get("status") == "complete"):
        raise ValueError("Report and case completion statuses disagree.")
    if report["scientific_role"].get("cnp2cnp_run") is True:
        if (
            report["input"].get("height") != REGISTERED_HEIGHT
            or report["input"].get("replicate_index")
            != REGISTERED_REPLICATE_INDEX
        ):
            raise ValueError("Production preflight did not use the registered case.")
        references = report.get("references", {})
        if not references.get("provided") or not references.get(
            "registered_hashes_enforced"
        ):
            raise ValueError("Production preflight lacks registered references.")
        if case.get("status") != "simulation_or_reference_failure":
            reference_check = case.get("reference_check") or {}
            if not reference_check.get("performed") or not reference_check.get(
                "passed"
            ):
                raise ValueError("Production preflight lacks a passing case reference.")
            summary = case.get("simulation_summary") or {}
            if (
                summary.get("selected_unique_state_count")
                != EXPECTED_UNIQUE_STATE_COUNT
                or summary.get("bidirectional_ordered_pair_bound")
                != EXPECTED_ORDERED_PAIR_COUNT
            ):
                raise ValueError(
                    "Production preflight differs from its registered bound."
                )
    forbidden_raw_keys = {
        "cnp",
        "genome",
        "tree",
        "matrix",
        "node_id",
        "nodes",
        "links",
    }
    present = forbidden_raw_keys & set(_walk_keys(report))
    if present:
        raise ValueError(
            "Compact dense-preflight report contains forbidden raw fields: "
            + ", ".join(sorted(present))
        )
    temporal_solver = report.get("provenance", {}).get("temporal_solver", {})
    if (
        temporal_solver.get("implementation_version")
        != TEMPORAL_ARBORESCENCE_SOLVER_VERSION
        or temporal_solver.get("source_module") != "reconstructor_temporal.py"
        or not isinstance(temporal_solver.get("source_module_sha256"), str)
        or len(temporal_solver["source_module_sha256"]) != 64
    ):
        raise ValueError("Dense preflight lacks compact temporal-solver identity.")
    json.dumps(report, sort_keys=True, allow_nan=False)


def compact_summary(report: Mapping[str, Any], output: Path) -> dict[str, Any]:
    case = report["case"]
    return {
        "schema_version": report["schema_version"],
        "analysis_role": report["analysis_role"],
        "status": report["status"],
        "preflight_verdict": report["preflight_verdict"],
        "output": str(output.resolve()),
        "case_key": case["case_key"],
        "case_status": case["status"],
        "reference_check": case["reference_check"],
        "sampling": (
            None
            if case["simulation_summary"] is None
            else {
                field: case["simulation_summary"][field]
                for field in (
                    "selected_occurrence_count",
                    "selected_unique_state_count",
                    "distance_matrix_cell_count",
                    "bidirectional_ordered_pair_bound",
                )
            }
        ),
        "distance": (
            None
            if case["distance"] is None
            else {
                "summary": case["distance"].get("summary"),
                "runtime": report["summary"]["distance_runtime"],
                "identity_sha256": case["distance"].get("identity_sha256"),
            }
        ),
        "arms": report["summary"]["arms"],
        "next_stage": "owner_and_agent_review_before_any_full_dense_probe",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the single registered H34-R012 50%-sampling production "
            "cnp2cnp and six-arm reconstruction feasibility preflight."
        )
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--fraction-truth-report", type=Path, required=True)
    parser.add_argument(
        "--sparse-reconstruction-report",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--simulation-timeout-seconds",
        type=int,
        default=DEFAULT_SIMULATION_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--distance-timeout-seconds",
        type=int,
        default=DEFAULT_DISTANCE_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--diagnostic-timeout-seconds",
        type=int,
        default=DEFAULT_DIAGNOSTIC_TIMEOUT_SECONDS,
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
        "--cnp2cnp-process-timeout-seconds",
        type=int,
        default=DEFAULT_CNP2CNP_PROCESS_TIMEOUT_SECONDS,
    )
    parser.add_argument("--rss-limit-bytes", type=int, default=DEFAULT_RSS_LIMIT_BYTES)
    parser.add_argument(
        "--capture-limit-bytes",
        type=int,
        default=DEFAULT_CAPTURE_LIMIT_BYTES,
    )
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    output = arguments.output.expanduser().resolve()
    if output.exists() and not arguments.overwrite:
        raise FileExistsError(
            f"Output already exists: {output}. Choose a new path or pass --overwrite."
        )
    report = run_preflight(
        base_config_path=arguments.base_config,
        fraction_truth_report_path=arguments.fraction_truth_report,
        sparse_reconstruction_report_path=arguments.sparse_reconstruction_report,
        simulation_timeout_seconds=arguments.simulation_timeout_seconds,
        distance_timeout_seconds=arguments.distance_timeout_seconds,
        diagnostic_timeout_seconds=arguments.diagnostic_timeout_seconds,
        reconstruction_timeout_seconds=arguments.reconstruction_timeout_seconds,
        evaluation_timeout_seconds=arguments.evaluation_timeout_seconds,
        cnp2cnp_process_timeout_seconds=(
            arguments.cnp2cnp_process_timeout_seconds
        ),
        rss_limit_bytes=arguments.rss_limit_bytes,
        capture_limit_bytes=arguments.capture_limit_bytes,
        progress=arguments.progress,
    )
    write_json_atomic(output, report)
    print(json.dumps(compact_summary(report, output), indent=2, sort_keys=True))
    return 0 if report["preflight_verdict"].startswith("pass_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
