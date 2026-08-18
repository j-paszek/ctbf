"""One-case cnp2cnp and ambiguity preflight for the hypothesis trend program.

The command consumes only the compact population preflight, reproduces one
approved simulation case, samples a fixed number of states per level without
looking at reconstruction outcomes, and runs the production minimum-
bidirectional cnp2cnp provider.  It does not reconstruct or evaluate a tree and
does not serialize truth trees or CNP profiles.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import tempfile
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from algorithm_evaluation.hypothesis_trend_pilot import PROJECT_ROOT
from algorithm_evaluation.hypothesis_trend_population_preflight import (
    SCHEMA_VERSION as POPULATION_SCHEMA_VERSION,
    _case_config,
    _file_sha256,
    _profile,
    validate_population_report,
)
from algorithm_evaluation.paper_pipeline_contract import read_json, write_json_atomic
from algorithm_evaluation.paper_pipeline_runner import measured_stage
from ctbs import (
    Cnp2CnpFileDistanceProvider,
    DistanceMatrix,
    load_ctbs_runtime_config,
    unique_cells_by_cell_id,
)
from distance_semantics import (
    CNP2CNP_SEMANTICS_VERSION,
    DISTANCE_PROVENANCE_SCHEMA_VERSION,
    stable_distance_label_key,
)
from simulator import CancerCellEvolutionSimulator


SCHEMA_VERSION = "ctbf-hypothesis-trend-distance-preflight-v1"
ANALYSIS_ROLE = "discovery_cnp2cnp_runtime_and_ambiguity_preflight"
SAMPLING_VERSION = "canonical-order-seeded-level-prefix-v1"
SAMPLING_SEED_NAMESPACE = "ctbf-v5-hypothesis-trend-distance-preflight-v1"

DEFAULT_POPULATION_PREFLIGHT = (
    PROJECT_ROOT
    / "experimental_results"
    / "ctbf_v3_hypothesis_trend_population_preflight_v1.json"
)
DEFAULT_CASE_KEY = "population-L50-H16"
DEFAULT_STATES_PER_LEVEL = 6
DEFAULT_RADII = (2.0, 4.0)
DEFAULT_CNP2CNP_TIMEOUT_SECONDS = 300
DEFAULT_STAGE_TIMEOUT_SECONDS = 660
DEFAULT_RSS_LIMIT_BYTES = 4 * 1024**3
DEFAULT_CAPTURE_LIMIT_BYTES = 1_048_576


DistanceCompute = Callable[[Sequence[Any]], DistanceMatrix]


def _sampling_seed(
    population_sha256: str,
    case_key: str,
    generation: int,
) -> int:
    material = (
        f"{SAMPLING_SEED_NAMESPACE}\0{population_sha256}\0{case_key}\0{generation}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**63)


def _canonical_cells_at_generation(
    simulator: CancerCellEvolutionSimulator,
    generation: int,
) -> list[Any]:
    cells = [
        genotype
        for genotype in simulator.genotypes.values()
        if genotype.generation == generation
    ]
    canonical = simulator.canonicalize_biopsy_genotypes(cells)
    return sorted(
        canonical,
        key=lambda cell: (_profile(cell), stable_distance_label_key(cell.cell_id)),
    )


def _sample_fixed_levels(
    simulator: CancerCellEvolutionSimulator,
    *,
    generations: Sequence[int],
    states_per_level: int,
    population_sha256: str,
    case_key: str,
) -> tuple[list[list[Any]], list[dict[str, Any]]]:
    selected_levels = []
    records = []
    for generation in generations:
        canonical = _canonical_cells_at_generation(simulator, int(generation))
        if len(canonical) < states_per_level:
            raise ValueError(
                f"Generation {generation} has only {len(canonical)} unique states; "
                f"cannot sample {states_per_level}."
            )
        seed = _sampling_seed(population_sha256, case_key, int(generation))
        permutation = np.random.Generator(np.random.PCG64(seed)).permutation(
            len(canonical)
        )
        selected = [canonical[int(index)] for index in permutation[:states_per_level]]
        selected = sorted(
            selected,
            key=lambda cell: (_profile(cell), stable_distance_label_key(cell.cell_id)),
        )
        selected_levels.append(selected)
        records.append(
            {
                "generation": int(generation),
                "available_unique_state_count": len(canonical),
                "selected_occurrence_count": len(selected),
                "sampling_seed": seed,
            }
        )
    return selected_levels, records


def _prepare_selected_case(
    *,
    config_path: Path,
    simulation_seed: int,
    generations: Sequence[int],
    states_per_level: int,
    population_sha256: str,
    case_key: str,
) -> dict[str, Any]:
    simulator = CancerCellEvolutionSimulator(str(config_path), seed=simulation_seed)
    simulator.run_simulation()
    levels, sampling_records = _sample_fixed_levels(
        simulator,
        generations=generations,
        states_per_level=states_per_level,
        population_sha256=population_sha256,
        case_key=case_key,
    )
    distance_cells = unique_cells_by_cell_id(
        [cell for level in levels for cell in level]
    )
    profile_by_label = {}
    for cell in distance_cells:
        profile = _profile(cell)
        if cell.cell_id in profile_by_label and profile_by_label[cell.cell_id] != profile:
            raise ValueError("One sampled state label maps to multiple CNP profiles.")
        profile_by_label[cell.cell_id] = profile
    return {
        "levels": levels,
        "distance_cells": distance_cells,
        "summary": {
            "truth_node_count": simulator.tree.number_of_nodes(),
            "truth_edge_count": simulator.tree.number_of_edges(),
            "selected_level_count": len(levels),
            "selected_occurrence_count": sum(len(level) for level in levels),
            "selected_unique_profile_count": len(distance_cells),
            "sampling": sampling_records,
        },
    }


def _validate_production_distance(distance: DistanceMatrix) -> None:
    provenance = distance.provenance
    if not isinstance(provenance, Mapping):
        raise ValueError("cnp2cnp result lacks provenance.")
    if provenance.get("schema_version") != DISTANCE_PROVENANCE_SCHEMA_VERSION:
        raise ValueError("cnp2cnp result has the wrong provenance schema.")
    required = {
        "metric": "cnp2cnp",
        "distance_mode": "any",
        "symmetrization": "minimum_bidirectional",
        "formula": "min(d_any(u,v),d_any(v,u))",
    }
    mismatches = {
        field: {"expected": expected, "observed": provenance.get(field)}
        for field, expected in required.items()
        if provenance.get(field) != expected
    }
    semantics_version = provenance.get(
        "semantics_version",
        provenance.get("semantic_version"),
    )
    if semantics_version != CNP2CNP_SEMANTICS_VERSION:
        mismatches["semantics_version"] = {
            "expected": CNP2CNP_SEMANTICS_VERSION,
            "observed": semantics_version,
        }
    if provenance.get("construction") not in {
        "opposite_order_matrix_mode",
        "trivial_singleton",
    }:
        mismatches["construction"] = {
            "expected": "opposite_order_matrix_mode or trivial_singleton",
            "observed": provenance.get("construction"),
        }
    if mismatches:
        raise ValueError(f"cnp2cnp provenance mismatch: {mismatches}")


def _number_summary(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "minimum": None,
            "maximum": None,
            "mean": None,
            "median": None,
        }
    normalized = [float(value) for value in values]
    return {
        "count": len(normalized),
        "minimum": min(normalized),
        "maximum": max(normalized),
        "mean": statistics.fmean(normalized),
        "median": statistics.median(normalized),
    }


def _array_number_summary(values: np.ndarray) -> dict[str, Any]:
    """Vectorized counterpart of ``_number_summary`` for dense diagnostics."""
    array = np.asarray(values, dtype=float).reshape(-1)
    if not array.size:
        return _number_summary([])
    return {
        "count": int(array.size),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
    }


def distance_matrix_summary(distance: DistanceMatrix) -> dict[str, Any]:
    matrix = np.asarray(distance.matrix, dtype=float)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Distance matrix is not square.")
    upper = matrix[np.triu_indices(matrix.shape[0], k=1)]
    if upper.size:
        unique_values, unique_counts = np.unique(upper, return_counts=True)
        distance_frequency = [
            {"distance": float(value), "count": int(count)}
            for value, count in zip(unique_values, unique_counts)
        ]
        off_diagonal = _array_number_summary(upper)
    else:
        distance_frequency = []
        off_diagonal = _number_summary([])
    return {
        "profile_count": matrix.shape[0],
        "unordered_pair_count": int(upper.size),
        "off_diagonal": off_diagonal,
        "zero_distance_unordered_pair_count": int(np.sum(upper == 0)),
        "distinct_off_diagonal_distance_count": len(distance_frequency),
        "distance_frequency": distance_frequency,
    }


def _degree_summary(degrees: Sequence[int]) -> dict[str, Any]:
    values = np.asarray(degrees, dtype=np.int64).reshape(-1)
    if values.size:
        unique_values, unique_counts = np.unique(values, return_counts=True)
        counts = {
            int(value): int(count)
            for value, count in zip(unique_values, unique_counts)
        }
        number_summary = {
            "count": int(values.size),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
            "mean": float(np.sum(values, dtype=np.int64) / values.size),
            "median": float(np.median(values)),
        }
    else:
        counts = {}
        number_summary = _number_summary([])
    return {
        **number_summary,
        "zero_count": counts.get(0, 0),
        "one_count": counts.get(1, 0),
        "multiple_count": sum(count for value, count in counts.items() if value > 1),
        "histogram": {str(value): counts[value] for value in sorted(counts)},
    }


@dataclass(frozen=True)
class _TransitionArrays:
    """Reusable dense arrays for one ordered biopsy transition."""

    cross_distances: np.ndarray
    biologically_plausible: np.ndarray
    same_state: np.ndarray


def _transition_arrays(
    parents: Sequence[Any],
    children: Sequence[Any],
    distance: DistanceMatrix,
) -> _TransitionArrays:
    """Build distance/plausibility arrays once for a transition.

    Plausibility is the absence of a parent-CN0/child-positive conflict.  A
    matrix product over the 0/positive indicators computes the exact Boolean
    result without a Python call for every parent-child pair.
    """
    id_to_index = {label: index for index, label in enumerate(distance.ids)}
    matrix = np.asarray(distance.matrix, dtype=float)
    parent_indices = np.asarray(
        [id_to_index[parent.cell_id] for parent in parents],
        dtype=int,
    )
    child_indices = np.asarray(
        [id_to_index[child.cell_id] for child in children],
        dtype=int,
    )
    cross_distances = matrix[np.ix_(child_indices, parent_indices)]
    child_count, parent_count = cross_distances.shape

    biologically_plausible = np.zeros(
        (child_count, parent_count),
        dtype=bool,
    )
    if child_count and parent_count:
        parent_genomes = np.asarray(
            [np.asarray(parent.genome) for parent in parents]
        )
        child_genomes = np.asarray(
            [np.asarray(child.genome) for child in children]
        )
        if (
            parent_genomes.ndim != 2
            or child_genomes.ndim != 2
            or parent_genomes.shape[1] != child_genomes.shape[1]
        ):
            raise ValueError("Candidate transition CNP profiles do not align.")
        conflict_counts = (
            (child_genomes > 0).astype(np.uint32)
            @ (parent_genomes == 0).astype(np.uint32).T
        )
        biologically_plausible = conflict_counts == 0

    same_state = np.zeros((child_count, parent_count), dtype=bool)
    parent_offsets_by_label: dict[Any, list[int]] = {}
    for parent_offset, parent in enumerate(parents):
        parent_offsets_by_label.setdefault(parent.cell_id, []).append(parent_offset)
    for child_offset, child in enumerate(children):
        same_state[
            child_offset,
            parent_offsets_by_label.get(child.cell_id, ()),
        ] = True
    return _TransitionArrays(
        cross_distances=cross_distances,
        biologically_plausible=biologically_plausible,
        same_state=same_state,
    )


def _pair_codegree_summary(candidate_mask: np.ndarray) -> tuple[dict[str, Any], int]:
    """Summarize parent-pair codegrees with one dense matrix multiplication."""
    candidate_mask = np.asarray(candidate_mask, dtype=bool)
    if candidate_mask.ndim != 2:
        raise ValueError("Candidate mask must be two-dimensional.")
    parent_count = candidate_mask.shape[1]
    if parent_count < 2:
        codegrees = np.empty(0, dtype=np.int64)
    else:
        count_dtype = (
            np.uint16
            if candidate_mask.shape[0] <= np.iinfo(np.uint16).max
            else np.uint32
        )
        numeric = candidate_mask.astype(count_dtype)
        codegree_matrix = numeric.T @ numeric
        upper = np.triu_indices(parent_count, k=1)
        # The v5 per-generation guard is 2,000, so the normal path uses exact
        # uint16 sums; the wider fallback prevents silent reuse overflow.
        codegrees = codegree_matrix[upper].astype(np.int64)
    four_cycle_count = int(
        np.sum(codegrees * (codegrees - 1) // 2, dtype=np.int64)
    )
    return _degree_summary(codegrees), four_cycle_count


def _candidate_layer_summaries(
    parents: Sequence[Any],
    children: Sequence[Any],
    distance: DistanceMatrix,
    *,
    radii: Sequence[float],
) -> list[dict[str, Any]]:
    arrays = _transition_arrays(parents, children, distance)
    cross_distances = arrays.cross_distances
    child_count, parent_count = cross_distances.shape
    results = []
    for radius in radii:
        raw = cross_distances <= float(radius)
        plausible = raw & arrays.biologically_plausible
        same = raw & arrays.same_state
        raw_degrees = np.sum(raw, axis=1, dtype=np.int64)
        plausible_degrees = np.sum(plausible, axis=1, dtype=np.int64)
        same_degrees = np.sum(same, axis=1, dtype=np.int64)
        same_rows = same_degrees > 0
        plausible_rows = plausible_degrees > 0

        minimum = np.zeros((child_count, parent_count), dtype=bool)
        if child_count and parent_count and np.any(plausible_rows):
            plausible_distances = np.where(plausible, cross_distances, np.inf)
            minimum_distances = np.min(plausible_distances, axis=1)
            minimum = plausible & (
                cross_distances == minimum_distances[:, np.newaxis]
            )
        if np.any(same_rows):
            minimum[same_rows] = same[same_rows]
        minimum_degrees = np.sum(minimum, axis=1, dtype=np.int64)
        codegree_summary, four_cycle_count = _pair_codegree_summary(plausible)
        results.append(
            {
                "parent_count": len(parents),
                "child_count": len(children),
                "radius": float(radius),
                "raw_radius_child_degree": _degree_summary(raw_degrees),
                "plausible_radius_child_degree": _degree_summary(
                    plausible_degrees
                ),
                "minimum_parent_child_degree": _degree_summary(minimum_degrees),
                "same_state_priority_count": int(np.count_nonzero(same_rows)),
                "missing_parent_count": int(
                    np.count_nonzero(~same_rows & ~plausible_rows)
                ),
                "parent_pair_plausible_codegree": codegree_summary,
                "plausible_radius_four_cycle_count": four_cycle_count,
            }
        )
    return results


def _candidate_layer_summary(
    parents: Sequence[Any],
    children: Sequence[Any],
    distance: DistanceMatrix,
    *,
    radius: float,
) -> dict[str, Any]:
    return _candidate_layer_summaries(
        parents,
        children,
        distance,
        radii=(radius,),
    )[0]


def candidate_graph_summary(
    levels: Sequence[Sequence[Any]],
    distance: DistanceMatrix,
    *,
    radii: Sequence[float],
) -> dict[str, Any]:
    transitions = []
    for child_level in range(1, len(levels)):
        parents = levels[child_level - 1]
        children = levels[child_level]
        transitions.append(
            {
                "parent_level": child_level - 1,
                "child_level": child_level,
                "radii": _candidate_layer_summaries(
                    parents,
                    children,
                    distance,
                    radii=radii,
                ),
            }
        )
    return {"transition_count": len(transitions), "transitions": transitions}


def _population_case(
    population: Mapping[str, Any],
    case_key: str,
) -> Mapping[str, Any]:
    matches = [record for record in population.get("cases", []) if record.get("case_key") == case_key]
    if len(matches) != 1:
        raise ValueError(f"Population preflight must contain exactly one {case_key!r} case.")
    record = matches[0]
    if record.get("status") != "success" or not isinstance(record.get("summary"), Mapping):
        raise ValueError(f"Population case {case_key!r} is not successful.")
    return record


def run_distance_preflight(
    *,
    population_preflight_path: Path | str = DEFAULT_POPULATION_PREFLIGHT,
    case_key: str = DEFAULT_CASE_KEY,
    states_per_level: int = DEFAULT_STATES_PER_LEVEL,
    radii: Sequence[float] = DEFAULT_RADII,
    cnp2cnp_timeout_seconds: int = DEFAULT_CNP2CNP_TIMEOUT_SECONDS,
    stage_timeout_seconds: int = DEFAULT_STAGE_TIMEOUT_SECONDS,
    rss_limit_bytes: int = DEFAULT_RSS_LIMIT_BYTES,
    capture_limit_bytes: int = DEFAULT_CAPTURE_LIMIT_BYTES,
    distance_compute: DistanceCompute | None = None,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    population_path = Path(population_preflight_path).expanduser().resolve()
    population = read_json(population_path)
    if population.get("schema_version") != POPULATION_SCHEMA_VERSION:
        raise ValueError("Unknown population-preflight schema.")
    validate_population_report(population)
    role = population.get("scientific_role")
    if not isinstance(role, Mapping) or role.get("cnp2cnp_run") is not False:
        raise ValueError("Input population artifact has the wrong scientific role.")
    if isinstance(states_per_level, bool) or not isinstance(states_per_level, int) or states_per_level <= 0:
        raise ValueError("states_per_level must be a positive integer.")
    normalized_radii = tuple(sorted({float(value) for value in radii}))
    if not normalized_radii or any(not math.isfinite(value) or value < 0 for value in normalized_radii):
        raise ValueError("radii must contain finite nonnegative values.")

    population_sha256 = _file_sha256(population_path)
    case = _population_case(population, case_key)
    if states_per_level > int(case["summary"]["maximum_common_fixed_budget_per_level"]):
        raise ValueError(
            f"states_per_level={states_per_level} exceeds the population-preflight "
            f"limit {case['summary']['maximum_common_fixed_budget_per_level']}."
        )

    base_config_value = population["input"]["base_config"]
    base_config_path = Path(base_config_value)
    if not base_config_path.is_absolute():
        base_config_path = PROJECT_ROOT / base_config_path
    base_config_path = base_config_path.resolve()
    if _file_sha256(base_config_path) != population["input"]["base_config_sha256"]:
        raise ValueError("Base simulator config changed after the population preflight.")
    base_config = read_json(base_config_path)
    config = _case_config(
        base_config,
        length=int(case["genome_length"]),
        height=int(case["number_of_generations"]),
        expected_cna_starts=float(case["expected_cna_starts_per_attempted_child"]),
    )
    generations = tuple(int(value) for value in case["relative_biopsy_generations"])
    simulation_seed = int(case["simulation_seed"])

    with tempfile.TemporaryDirectory(prefix="ctbf-hypothesis-distance-") as directory:
        config_path = Path(directory) / "case.json"
        write_json_atomic(config_path, config)
        prepared, simulation_runtime, simulation_error = measured_stage(
            lambda: _prepare_selected_case(
                config_path=config_path,
                simulation_seed=simulation_seed,
                generations=generations,
                states_per_level=states_per_level,
                population_sha256=population_sha256,
                case_key=case_key,
            ),
            timeout_seconds=int(stage_timeout_seconds),
            rss_limit_bytes=int(rss_limit_bytes),
        )

    distance = None
    distance_runtime = None
    distance_error = None
    if simulation_error is None:
        if prepared["summary"]["truth_node_count"] != case["summary"]["truth_node_count"]:
            simulation_error = ValueError(
                "Simulation replay truth-node count differs from population preflight."
            )
        else:
            if distance_compute is None:
                runtime_config = replace(
                    load_ctbs_runtime_config(),
                    cnp2cnp_timeout_seconds=float(cnp2cnp_timeout_seconds),
                    cnp2cnp_capture_limit_bytes=int(capture_limit_bytes),
                )
                provider = Cnp2CnpFileDistanceProvider(runtime_config)
                compute = provider.compute
            else:
                compute = distance_compute
            distance, distance_runtime, distance_error = measured_stage(
                lambda: compute(prepared["distance_cells"]),
                timeout_seconds=int(stage_timeout_seconds),
                rss_limit_bytes=int(rss_limit_bytes),
            )
            if distance_error is None:
                try:
                    _validate_production_distance(distance)
                except Exception as error:
                    distance_error = error
                    distance = None

    if simulation_error is not None:
        status = "simulation_failure"
        error = simulation_error
    elif distance_error is not None:
        status = "distance_failure"
        error = distance_error
    else:
        status = "success"
        error = None

    report = {
        "schema_version": SCHEMA_VERSION,
        "analysis_role": ANALYSIS_ROLE,
        "created_at_utc": created_at_utc or datetime.now(timezone.utc).isoformat(),
        "status": status,
        "error": (
            {"type": type(error).__name__, "message": str(error)[:4096]}
            if error is not None
            else None
        ),
        "scientific_role": {
            "paper_evidence_allowed": False,
            "simulation_replay": True,
            "cnp2cnp_run": distance_compute is None and simulation_error is None,
            "injected_distance_for_test": distance_compute is not None,
            "reconstruction_run": False,
            "evaluation_run": False,
        },
        "input": {
            "population_preflight": population_path.relative_to(PROJECT_ROOT).as_posix()
            if population_path.is_relative_to(PROJECT_ROOT)
            else str(population_path),
            "population_preflight_sha256": population_sha256,
            "case_key": case_key,
            "genome_length": int(case["genome_length"]),
            "number_of_generations": int(case["number_of_generations"]),
            "generations": list(generations),
            "states_per_level": states_per_level,
            "maximum_unique_profile_count": states_per_level * len(generations),
            "maximum_bidirectional_pair_entry_count": (
                states_per_level
                * len(generations)
                * (states_per_level * len(generations) - 1)
            ),
            "radii": list(normalized_radii),
            "simulation_seed": simulation_seed,
            "sampling_version": SAMPLING_VERSION,
            "distance_semantics": CNP2CNP_SEMANTICS_VERSION,
            "cnp2cnp_timeout_seconds_per_process": int(cnp2cnp_timeout_seconds),
            "stage_timeout_seconds": int(stage_timeout_seconds),
            "rss_limit_bytes": int(rss_limit_bytes),
        },
        "simulation": {
            "runtime": simulation_runtime,
            "summary": prepared["summary"] if prepared is not None else None,
        },
        "distance": {
            "runtime": distance_runtime,
            "provenance": distance.provenance if distance is not None else None,
            "summary": distance_matrix_summary(distance) if distance is not None else None,
        },
        "ambiguity": (
            candidate_graph_summary(
                prepared["levels"],
                distance,
                radii=normalized_radii,
            )
            if distance is not None
            else None
        ),
        "provenance": {
            "source_module": Path(__file__).relative_to(PROJECT_ROOT).as_posix(),
            "source_module_sha256": _file_sha256(Path(__file__)),
            "reads_existing_result_corpus": False,
            "writes_truth_trees_or_profiles": False,
        },
    }
    validate_distance_preflight_report(report)
    return report


def validate_distance_preflight_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unknown distance-preflight schema.")
    if report.get("analysis_role") != ANALYSIS_ROLE:
        raise ValueError("Distance preflight has the wrong analysis role.")
    if report.get("status") not in {"success", "simulation_failure", "distance_failure"}:
        raise ValueError("Distance preflight has an unknown status.")
    role = report.get("scientific_role")
    if not isinstance(role, Mapping) or role.get("paper_evidence_allowed") is not False:
        raise ValueError("Distance preflight must remain discovery-only.")
    if role.get("reconstruction_run") is not False or role.get("evaluation_run") is not False:
        raise ValueError("Distance preflight must not reconstruct or evaluate.")
    if report["status"] == "success":
        if report.get("error") is not None:
            raise ValueError("Successful distance preflight contains an error.")
        if not isinstance(report.get("ambiguity"), Mapping):
            raise ValueError("Successful distance preflight lacks ambiguity output.")
    json.dumps(report, sort_keys=True, allow_nan=False)


def compact_summary(report: Mapping[str, Any], output: Path) -> dict[str, Any]:
    distance = report["distance"]
    transitions = []
    if report.get("ambiguity"):
        for transition in report["ambiguity"]["transitions"]:
            transitions.append(
                {
                    "parent_level": transition["parent_level"],
                    "child_level": transition["child_level"],
                    "radii": [
                        {
                            "radius": row["radius"],
                            "plausible_degree_mean": row[
                                "plausible_radius_child_degree"
                            ]["mean"],
                            "minimum_parent_tie_children": row[
                                "minimum_parent_child_degree"
                            ]["multiple_count"],
                            "missing_parent_count": row["missing_parent_count"],
                            "four_cycle_count": row[
                                "plausible_radius_four_cycle_count"
                            ],
                        }
                        for row in transition["radii"]
                    ],
                }
            )
    return {
        "analysis_role": report["analysis_role"],
        "schema_version": report["schema_version"],
        "status": report["status"],
        "error": report["error"],
        "output": str(output.resolve()),
        "selected_unique_profile_count": (
            report["simulation"]["summary"]["selected_unique_profile_count"]
            if report["simulation"]["summary"]
            else None
        ),
        "distance_wall_time_seconds": (
            distance["runtime"]["wall_time_ns"] / 1_000_000_000
            if distance["runtime"]
            else None
        ),
        "distance_summary": distance["summary"],
        "candidate_graph_transitions": transitions,
        "next_stage": (
            "review_runtime_and_ambiguity_before_any_multi_case_cnp2cnp_run"
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one fixed-budget production cnp2cnp case and write compact "
            "distance/candidate-graph diagnostics without reconstruction."
        )
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--population-preflight",
        type=Path,
        default=DEFAULT_POPULATION_PREFLIGHT,
    )
    parser.add_argument("--case-key", default=DEFAULT_CASE_KEY)
    parser.add_argument("--states-per-level", type=int, default=DEFAULT_STATES_PER_LEVEL)
    parser.add_argument("--radii", type=float, nargs="+", default=list(DEFAULT_RADII))
    parser.add_argument(
        "--cnp2cnp-timeout-seconds",
        type=int,
        default=DEFAULT_CNP2CNP_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--stage-timeout-seconds",
        type=int,
        default=DEFAULT_STAGE_TIMEOUT_SECONDS,
    )
    parser.add_argument("--rss-limit-bytes", type=int, default=DEFAULT_RSS_LIMIT_BYTES)
    parser.add_argument("--capture-limit-bytes", type=int, default=DEFAULT_CAPTURE_LIMIT_BYTES)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    output = arguments.output.expanduser().resolve()
    if output.exists() and not arguments.overwrite:
        raise FileExistsError(
            f"Output already exists: {output}. Choose a new versioned path or pass --overwrite."
        )
    report = run_distance_preflight(
        population_preflight_path=arguments.population_preflight,
        case_key=arguments.case_key,
        states_per_level=arguments.states_per_level,
        radii=arguments.radii,
        cnp2cnp_timeout_seconds=arguments.cnp2cnp_timeout_seconds,
        stage_timeout_seconds=arguments.stage_timeout_seconds,
        rss_limit_bytes=arguments.rss_limit_bytes,
        capture_limit_bytes=arguments.capture_limit_bytes,
    )
    write_json_atomic(output, report)
    print(json.dumps(compact_summary(report, output), indent=2, sort_keys=True))
    return 0 if report["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
