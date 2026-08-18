"""Bounded fixed-budget height trend for cnp2cnp ambiguity diagnostics.

This discovery-only runner varies latent height while holding CNP length,
expected CNA starts, three relative biopsy positions, and the absolute number
of sampled states fixed.  It records distance/candidate-graph behavior only;
it never reconstructs or evaluates a tree and never serializes profiles or a
truth tree.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import tempfile
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from algorithm_evaluation.hypothesis_trend_distance_preflight import (
    ANALYSIS_ROLE as DISTANCE_PREFLIGHT_ROLE,
    SCHEMA_VERSION as DISTANCE_PREFLIGHT_SCHEMA_VERSION,
    _array_number_summary,
    _degree_summary,
    _number_summary,
    _pair_codegree_summary,
    _transition_arrays,
    _validate_production_distance,
    candidate_graph_summary,
    distance_matrix_summary,
)
from algorithm_evaluation.hypothesis_trend_pilot import (
    PROJECT_ROOT,
    _relative_biopsy_generations,
)
from algorithm_evaluation.hypothesis_trend_population_preflight import (
    _case_config,
    _file_sha256,
    _profile,
)
from algorithm_evaluation.paper_pipeline_contract import (
    read_json,
    write_json_atomic,
)
from algorithm_evaluation.paper_pipeline_runner import measured_stage
from ctbs import (
    Cnp2CnpFileDistanceProvider,
    DistanceMatrix,
    load_ctbs_runtime_config,
    unique_cells_by_cell_id,
)
from distance_semantics import (
    CNP2CNP_SEMANTICS_VERSION,
    stable_distance_label_key,
)
from simulator import CancerCellEvolutionSimulator


SCHEMA_VERSION = "ctbf-hypothesis-height-ambiguity-trend-v1"
ANALYSIS_ROLE = "discovery_fixed_budget_height_ambiguity_trend"
SEED_NAMESPACE = "ctbf-v5-hypothesis-height-ambiguity-trend-v1"
SAMPLING_VERSION = "canonical-order-seeded-fixed-level-budget-v1"
PREDICTION_VERSION = "post-t2b-height-ambiguity-prediction-v1"

DEFAULT_RADIUS_SCALE_PREFLIGHT = (
    PROJECT_ROOT
    / "experimental_results"
    / "ctbf_v5_hypothesis_trend_cnp2cnp_radius_scale_preflight_v1.json"
)
DEFAULT_BASE_CONFIG = (
    PROJECT_ROOT / "simulator_examples" / "paper_v5" / "clean_balanced.json"
)
DEFAULT_HEIGHTS = (8, 12, 16)
DEFAULT_GENOME_LENGTH = 50
DEFAULT_EXPECTED_CNA_STARTS = 0.1
DEFAULT_STATES_PER_LEVEL = 6
DEFAULT_REPLICATES = 12
DEFAULT_BASE_SEED = 20260805
DEFAULT_FIXED_RADIUS = 4.0
DEFAULT_CNP2CNP_TIMEOUT_SECONDS = 120
DEFAULT_STAGE_TIMEOUT_SECONDS = 180
DEFAULT_RSS_LIMIT_BYTES = 4 * 1024**3
DEFAULT_CAPTURE_LIMIT_BYTES = 1_048_576
DEFAULT_MAX_FAILURES = 6
MAX_REPLICATES = 24
MAX_PROFILE_COUNT_PER_CASE = 18
STATIC_TRUTH_NODE_CAP = 150_000


DistanceCompute = Callable[[Sequence[Any]], DistanceMatrix]


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _simulation_seed(base_seed: int, replicate_index: int) -> int:
    material = (
        f"{SEED_NAMESPACE}\0simulation\0{base_seed}\0{replicate_index}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**63)


def _sampling_seed(
    base_seed: int,
    replicate_index: int,
    height: int,
    generation: int,
) -> int:
    material = (
        f"{SEED_NAMESPACE}\0sampling\0{base_seed}\0{replicate_index}"
        f"\0{height}\0{generation}"
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


def _simulate_and_select(
    *,
    config_path: Path,
    simulation_seed: int,
    base_seed: int,
    replicate_index: int,
    height: int,
    generations: Sequence[int],
    states_per_level: int,
) -> dict[str, Any]:
    simulator = CancerCellEvolutionSimulator(str(config_path), seed=simulation_seed)
    simulator.run_simulation()

    available_levels = [
        _canonical_cells_at_generation(simulator, int(generation))
        for generation in generations
    ]
    sampling_rows = []
    for generation, available in zip(generations, available_levels):
        sampling_rows.append(
            {
                "generation": int(generation),
                "available_unique_state_count": len(available),
                "sampling_seed": _sampling_seed(
                    base_seed,
                    replicate_index,
                    height,
                    int(generation),
                ),
            }
        )

    insufficient = [
        row
        for row in sampling_rows
        if row["available_unique_state_count"] < states_per_level
    ]
    public_summary = {
        "truth_node_count": simulator.tree.number_of_nodes(),
        "truth_edge_count": simulator.tree.number_of_edges(),
        "sampling": sampling_rows,
        "eligible_for_fixed_budget": not insufficient,
        "insufficient_generation_count": len(insufficient),
        "selected_occurrence_count": 0,
        "selected_unique_profile_count": 0,
        "repeated_profile_across_level_count": 0,
    }
    if insufficient:
        return {
            "levels": None,
            "distance_cells": None,
            "summary": public_summary,
        }

    selected_levels = []
    for available, row in zip(available_levels, sampling_rows):
        permutation = np.random.Generator(
            np.random.PCG64(row["sampling_seed"])
        ).permutation(len(available))
        selected = [available[int(index)] for index in permutation[:states_per_level]]
        selected_levels.append(
            sorted(
                selected,
                key=lambda cell: (
                    _profile(cell),
                    stable_distance_label_key(cell.cell_id),
                ),
            )
        )
        row["selected_occurrence_count"] = states_per_level

    occurrences = [cell for level in selected_levels for cell in level]
    distance_cells = unique_cells_by_cell_id(occurrences)
    profile_by_label = {}
    for cell in occurrences:
        profile = _profile(cell)
        if cell.cell_id in profile_by_label and profile_by_label[cell.cell_id] != profile:
            raise ValueError("One sampled state label maps to multiple CNP profiles.")
        profile_by_label[cell.cell_id] = profile
    public_summary.update(
        {
            "selected_occurrence_count": len(occurrences),
            "selected_unique_profile_count": len(distance_cells),
            "repeated_profile_across_level_count": (
                len(occurrences) - len(distance_cells)
            ),
        }
    )
    return {
        "levels": selected_levels,
        "distance_cells": distance_cells,
        "summary": public_summary,
    }


def _four_cycle_summary(candidate_sets: Sequence[set[int]], parent_count: int) -> dict[str, Any]:
    candidate_mask = np.zeros((len(candidate_sets), parent_count), dtype=bool)
    for child_offset, candidates in enumerate(candidate_sets):
        candidate_mask[child_offset, list(candidates)] = True
    codegree_summary, four_cycle_count = _pair_codegree_summary(candidate_mask)
    return {
        "parent_pair_codegree": codegree_summary,
        "four_cycle_count": four_cycle_count,
    }


def transition_scale_summary(
    parents: Sequence[Any],
    children: Sequence[Any],
    distance: DistanceMatrix,
) -> dict[str, Any]:
    """Summarize nearest plausible parents without choosing a radius."""
    arrays = _transition_arrays(parents, children, distance)
    cross_distances = arrays.cross_distances
    if cross_distances.shape[1] == 0:
        raise ValueError("Transition-scale diagnostics require at least one parent.")
    plausible = arrays.biologically_plausible | arrays.same_state
    plausible_degrees = np.sum(plausible, axis=1, dtype=np.int64)
    plausible_rows = plausible_degrees > 0
    raw_nearest_distances = np.min(cross_distances, axis=1)
    plausible_distances = np.where(plausible, cross_distances, np.inf)
    minimum_distances = np.min(plausible_distances, axis=1)
    nearest = plausible & (cross_distances == minimum_distances[:, np.newaxis])
    nearest_degrees = np.sum(nearest, axis=1, dtype=np.int64)
    nearest_distances = minimum_distances[plausible_rows]

    sorted_nearest = np.sort(nearest_distances)
    half_target = math.ceil(len(children) / 2)
    half_coverage_radius = (
        float(sorted_nearest[half_target - 1])
        if sorted_nearest.size >= half_target
        else None
    )
    full_coverage_radius = (
        float(sorted_nearest[-1])
        if sorted_nearest.size == len(children) and sorted_nearest.size
        else None
    )
    plausible_codegrees, plausible_four_cycles = _pair_codegree_summary(plausible)
    nearest_codegrees, nearest_four_cycles = _pair_codegree_summary(nearest)
    return {
        "parent_count": len(parents),
        "child_count": len(children),
        "cross_level_pair_distance": _array_number_summary(cross_distances),
        "raw_nearest_parent_distance": _array_number_summary(
            raw_nearest_distances
        ),
        "unrestricted_plausible_child_degree": _degree_summary(plausible_degrees),
        "unrestricted_missing_plausible_parent_count": int(
            np.count_nonzero(~plausible_rows)
        ),
        "nearest_plausible_parent_distance": _array_number_summary(
            nearest_distances
        ),
        "nearest_plausible_parent_child_degree": _degree_summary(nearest_degrees),
        "half_child_coverage_target": half_target,
        "half_child_coverage_radius": half_coverage_radius,
        "full_child_coverage_radius": full_coverage_radius,
        "plausible_radius_graph": {
            "parent_pair_codegree": plausible_codegrees,
            "four_cycle_count": plausible_four_cycles,
        },
        "nearest_plausible_parent_graph": {
            "parent_pair_codegree": nearest_codegrees,
            "four_cycle_count": nearest_four_cycles,
        },
    }


def ambiguity_case_summary(
    levels: Sequence[Sequence[Any]],
    generations: Sequence[int],
    distance: DistanceMatrix,
    *,
    fixed_radius: float,
    fixed_candidate_graph: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if fixed_candidate_graph is None:
        fixed = candidate_graph_summary(
            levels,
            distance,
            radii=[fixed_radius],
        )["transitions"]
    else:
        source_transitions = fixed_candidate_graph.get("transitions")
        if (
            not isinstance(source_transitions, Sequence)
            or len(source_transitions) != len(levels) - 1
        ):
            raise ValueError("Precomputed candidate graph does not match levels.")
        fixed = []
        for child_level, source in enumerate(source_transitions, start=1):
            matches = [
                row
                for row in source.get("radii", ())
                if float(row.get("radius")) == float(fixed_radius)
            ]
            if (
                source.get("parent_level") != child_level - 1
                or source.get("child_level") != child_level
                or len(matches) != 1
            ):
                raise ValueError(
                    "Precomputed candidate graph lacks the fixed-radius transition."
                )
            fixed.append(
                {
                    "parent_level": child_level - 1,
                    "child_level": child_level,
                    "radii": matches,
                }
            )
    transitions = []
    for child_level in range(1, len(levels)):
        transitions.append(
            {
                "parent_level": child_level - 1,
                "child_level": child_level,
                "parent_generation": int(generations[child_level - 1]),
                "child_generation": int(generations[child_level]),
                "generation_gap": int(
                    generations[child_level] - generations[child_level - 1]
                ),
                "fixed_radius": fixed[child_level - 1]["radii"][0],
                "scale_free": transition_scale_summary(
                    levels[child_level - 1],
                    levels[child_level],
                    distance,
                ),
            }
        )

    child_count = sum(row["scale_free"]["child_count"] for row in transitions)
    fixed_missing = sum(
        row["fixed_radius"]["missing_parent_count"] for row in transitions
    )
    fixed_ties = sum(
        row["fixed_radius"]["minimum_parent_child_degree"]["multiple_count"]
        for row in transitions
    )
    fixed_edge_count = sum(
        row["fixed_radius"]["plausible_radius_child_degree"]["mean"]
        * row["scale_free"]["child_count"]
        for row in transitions
    )
    nearest_missing = sum(
        row["scale_free"]["unrestricted_missing_plausible_parent_count"]
        for row in transitions
    )
    nearest_ties = sum(
        row["scale_free"]["nearest_plausible_parent_child_degree"]["multiple_count"]
        for row in transitions
    )
    nearest_distance_count = sum(
        row["scale_free"]["nearest_plausible_parent_distance"]["count"]
        for row in transitions
    )
    nearest_distance_sum = sum(
        row["scale_free"]["nearest_plausible_parent_distance"]["mean"]
        * row["scale_free"]["nearest_plausible_parent_distance"]["count"]
        for row in transitions
        if row["scale_free"]["nearest_plausible_parent_distance"]["count"]
    )
    half_radii = [
        row["scale_free"]["half_child_coverage_radius"] for row in transitions
    ]
    full_radii = [
        row["scale_free"]["full_child_coverage_radius"] for row in transitions
    ]
    return {
        "transition_count": len(transitions),
        "transitions": transitions,
        "case_metrics": {
            "child_decision_count": child_count,
            "fixed_r4_missing_fraction": fixed_missing / child_count,
            "fixed_r4_plausible_degree_mean": fixed_edge_count / child_count,
            "fixed_r4_tie_child_fraction": fixed_ties / child_count,
            "fixed_r4_four_cycle_count": sum(
                row["fixed_radius"]["plausible_radius_four_cycle_count"]
                for row in transitions
            ),
            "unrestricted_missing_plausible_parent_fraction": (
                nearest_missing / child_count
            ),
            "nearest_plausible_distance_mean": (
                nearest_distance_sum / nearest_distance_count
                if nearest_distance_count
                else None
            ),
            "half_coverage_radius_mean": (
                statistics.fmean(half_radii)
                if all(value is not None for value in half_radii)
                else None
            ),
            "full_coverage_radius_mean": (
                statistics.fmean(full_radii)
                if all(value is not None for value in full_radii)
                else None
            ),
            "unrestricted_nearest_tie_child_fraction": nearest_ties / child_count,
            "unrestricted_nearest_four_cycle_count": sum(
                row["scale_free"]["nearest_plausible_parent_graph"][
                    "four_cycle_count"
                ]
                for row in transitions
            ),
        },
    }


def _distance_identity(provenance: Mapping[str, Any]) -> dict[str, Any]:
    return {
        field: provenance.get(field)
        for field in (
            "schema_version",
            "semantics_version",
            "metric",
            "distance_mode",
            "symmetrization",
            "formula",
            "construction",
            "cnp2cnp_source_revision",
            "source_sha256",
            "python_executable",
            "cnp2cnp_executable",
            "command_template",
        )
    }


def _typed_error(error: BaseException | None) -> dict[str, str] | None:
    if error is None:
        return None
    return {
        "type": type(error).__name__,
        "message": str(error)[:4096],
    }


def _run_case(
    *,
    base_config: Mapping[str, Any],
    height: int,
    replicate_index: int,
    base_seed: int,
    states_per_level: int,
    expected_cna_starts: float,
    fixed_radius: float,
    distance_compute: DistanceCompute,
    injected_distance: bool,
    stage_timeout_seconds: int,
    rss_limit_bytes: int,
) -> dict[str, Any]:
    generations = tuple(_relative_biopsy_generations(height))
    simulation_seed = _simulation_seed(base_seed, replicate_index)
    config = _case_config(
        base_config,
        length=DEFAULT_GENOME_LENGTH,
        height=height,
        expected_cna_starts=expected_cna_starts,
    )
    with tempfile.TemporaryDirectory(prefix="ctbf-height-ambiguity-") as directory:
        config_path = Path(directory) / "case.json"
        write_json_atomic(config_path, config)
        prepared, simulation_runtime, simulation_error = measured_stage(
            lambda: _simulate_and_select(
                config_path=config_path,
                simulation_seed=simulation_seed,
                base_seed=base_seed,
                replicate_index=replicate_index,
                height=height,
                generations=generations,
                states_per_level=states_per_level,
            ),
            timeout_seconds=stage_timeout_seconds,
            rss_limit_bytes=rss_limit_bytes,
        )

    record = {
        "case_key": f"height-H{height}-R{replicate_index + 1:03d}",
        "height": height,
        "replicate_index": replicate_index,
        "simulation_seed": simulation_seed,
        "generations": list(generations),
        "status": None,
        "error": None,
        "simulation": {
            "runtime": simulation_runtime,
            "summary": prepared["summary"] if prepared is not None else None,
        },
        "distance": None,
        "ambiguity": None,
    }
    if simulation_error is not None:
        record["status"] = "simulation_failure"
        record["error"] = _typed_error(simulation_error)
        return record
    if not prepared["summary"]["eligible_for_fixed_budget"]:
        record["status"] = "insufficient_unique_states"
        record["error"] = {
            "type": "InsufficientUniqueStates",
            "message": (
                f"At least one biopsy generation has fewer than "
                f"{states_per_level} unique states."
            ),
        }
        return record

    distance, distance_runtime, distance_error = measured_stage(
        lambda: distance_compute(prepared["distance_cells"]),
        timeout_seconds=stage_timeout_seconds,
        rss_limit_bytes=rss_limit_bytes,
    )
    if distance_error is None:
        try:
            _validate_production_distance(distance)
        except Exception as error:  # explicit typed boundary
            distance_error = error
            distance = None
    if distance_error is not None:
        record["status"] = "distance_failure"
        record["error"] = _typed_error(distance_error)
        record["distance"] = {"runtime": distance_runtime}
        return record

    provenance_identity = _distance_identity(distance.provenance)
    record["status"] = "success"
    record["distance"] = {
        "runtime": distance_runtime,
        "summary": distance_matrix_summary(distance),
        "identity": provenance_identity,
        "identity_sha256": _sha256_json(provenance_identity),
        "external_process_count": distance.provenance.get("external_process_count"),
        "injected_distance_for_test": injected_distance,
    }
    record["ambiguity"] = ambiguity_case_summary(
        prepared["levels"],
        generations,
        distance,
        fixed_radius=fixed_radius,
    )
    return record


def _metric_summary(records: Sequence[Mapping[str, Any]], metric: str) -> dict[str, Any]:
    values = [
        float(record["ambiguity"]["case_metrics"][metric])
        for record in records
        if record.get("status") == "success"
        and record["ambiguity"]["case_metrics"].get(metric) is not None
    ]
    return _number_summary(values)


CASE_METRICS = (
    "fixed_r4_missing_fraction",
    "fixed_r4_plausible_degree_mean",
    "fixed_r4_tie_child_fraction",
    "fixed_r4_four_cycle_count",
    "unrestricted_missing_plausible_parent_fraction",
    "nearest_plausible_distance_mean",
    "half_coverage_radius_mean",
    "full_coverage_radius_mean",
    "unrestricted_nearest_tie_child_fraction",
    "unrestricted_nearest_four_cycle_count",
)
PRIMARY_PREDICTED_METRICS = (
    "fixed_r4_missing_fraction",
    "nearest_plausible_distance_mean",
    "half_coverage_radius_mean",
)


def aggregate_cases(
    cases: Sequence[Mapping[str, Any]],
    heights: Sequence[int],
) -> dict[str, Any]:
    by_height = {}
    for height in heights:
        records = [record for record in cases if record["height"] == height]
        truth_counts = [
            float(record["simulation"]["summary"]["truth_node_count"])
            for record in records
            if record["simulation"]["summary"] is not None
        ]
        distance_seconds = [
            record["distance"]["runtime"]["wall_time_ns"] / 1_000_000_000
            for record in records
            if record.get("status") == "success"
        ]
        by_height[str(height)] = {
            "planned_case_count": len(records),
            "status_counts": dict(sorted(Counter(row["status"] for row in records).items())),
            "truth_node_count": _number_summary(truth_counts),
            "distance_wall_time_seconds": _number_summary(distance_seconds),
            "case_metrics": {
                metric: _metric_summary(records, metric) for metric in CASE_METRICS
            },
        }

    by_replicate: dict[int, dict[int, Mapping[str, Any]]] = {}
    for record in cases:
        if record["status"] == "success":
            by_replicate.setdefault(record["replicate_index"], {})[
                record["height"]
            ] = record
    endpoint_differences = {}
    low_height = min(heights)
    high_height = max(heights)
    for metric in PRIMARY_PREDICTED_METRICS:
        differences = []
        monotonic_count = 0
        monotonic_eligible = 0
        for block in by_replicate.values():
            if low_height in block and high_height in block:
                low = block[low_height]["ambiguity"]["case_metrics"].get(metric)
                high = block[high_height]["ambiguity"]["case_metrics"].get(metric)
                if low is not None and high is not None:
                    differences.append(float(high) - float(low))
            if all(height in block for height in heights):
                values = [
                    block[height]["ambiguity"]["case_metrics"].get(metric)
                    for height in heights
                ]
                if all(value is not None for value in values):
                    monotonic_eligible += 1
                    monotonic_count += all(
                        float(left) <= float(right)
                        for left, right in zip(values, values[1:])
                    )
        wins = sum(value > 1e-12 for value in differences)
        losses = sum(value < -1e-12 for value in differences)
        ties = len(differences) - wins - losses
        summary = _number_summary(differences)
        if not differences:
            verdict = "not_assessed_no_complete_endpoint_pairs"
        elif summary["mean"] > 0:
            verdict = "predicted_direction_supported_descriptively"
        elif summary["mean"] < 0:
            verdict = "predicted_direction_contradicted_descriptively"
        else:
            verdict = "no_mean_direction"
        endpoint_differences[metric] = {
            "contrast": f"H{high_height}_minus_H{low_height}",
            "difference": summary,
            "wins_ties_losses": {
                "wins": wins,
                "ties": ties,
                "losses": losses,
            },
            "monotonic_non_decreasing_block_count": monotonic_count,
            "monotonic_eligible_block_count": monotonic_eligible,
            "verdict": verdict,
        }
    return {
        "by_height": by_height,
        "paired_endpoint_differences": endpoint_differences,
    }


def _validate_prior_radius_scale(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != DISTANCE_PREFLIGHT_SCHEMA_VERSION:
        raise ValueError("Radius-scale input has an unknown schema.")
    if report.get("analysis_role") != DISTANCE_PREFLIGHT_ROLE:
        raise ValueError("Radius-scale input has the wrong analysis role.")
    if report.get("status") != "success":
        raise ValueError("Radius-scale input is not successful.")
    role = report.get("scientific_role")
    if not isinstance(role, Mapping):
        raise ValueError("Radius-scale input lacks its scientific role.")
    if role.get("paper_evidence_allowed") is not False or role.get("cnp2cnp_run") is not True:
        raise ValueError("Radius-scale input has the wrong discovery/distance role.")
    if role.get("reconstruction_run") is not False or role.get("evaluation_run") is not False:
        raise ValueError("Radius-scale input must not contain reconstruction/evaluation.")
    prior_input = report.get("input", {})
    expected = {
        "case_key": "population-L50-H16",
        "genome_length": 50,
        "number_of_generations": 16,
        "states_per_level": 6,
        "distance_semantics": CNP2CNP_SEMANTICS_VERSION,
    }
    mismatches = {
        field: {"expected": value, "observed": prior_input.get(field)}
        for field, value in expected.items()
        if prior_input.get(field) != value
    }
    if mismatches:
        raise ValueError(f"Radius-scale input contract mismatch: {mismatches}")
    if prior_input.get("radii") != [float(value) for value in range(4, 19)]:
        raise ValueError("Radius-scale input does not contain the frozen 4--18 sweep.")


def _validated_population_dependency(prior: Mapping[str, Any]) -> Mapping[str, Any]:
    prior_input = prior.get("input", {})
    population_value = prior_input.get("population_preflight")
    population_sha256 = prior_input.get("population_preflight_sha256")
    if not isinstance(population_value, str) or not isinstance(population_sha256, str):
        raise ValueError("Radius-scale input lacks its population dependency.")
    population_path = Path(population_value)
    if not population_path.is_absolute():
        population_path = PROJECT_ROOT / population_path
    population_path = population_path.resolve()
    if _file_sha256(population_path) != population_sha256:
        raise ValueError("Population dependency changed after the radius-scale run.")
    population = read_json(population_path)
    if not isinstance(population.get("input"), Mapping):
        raise ValueError("Population dependency lacks its input contract.")
    if not isinstance(population["input"].get("base_config"), str):
        raise ValueError("Population dependency lacks its base config path.")
    if not isinstance(population["input"].get("base_config_sha256"), str):
        raise ValueError("Population dependency lacks its base config checksum.")
    return population


def run_height_ambiguity_trend(
    *,
    output_basis_path: Path | str = DEFAULT_RADIUS_SCALE_PREFLIGHT,
    base_config_path: Path | str = DEFAULT_BASE_CONFIG,
    heights: Sequence[int] = DEFAULT_HEIGHTS,
    genome_length: int = DEFAULT_GENOME_LENGTH,
    expected_cna_starts: float = DEFAULT_EXPECTED_CNA_STARTS,
    states_per_level: int = DEFAULT_STATES_PER_LEVEL,
    replicates: int = DEFAULT_REPLICATES,
    base_seed: int = DEFAULT_BASE_SEED,
    fixed_radius: float = DEFAULT_FIXED_RADIUS,
    cnp2cnp_timeout_seconds: int = DEFAULT_CNP2CNP_TIMEOUT_SECONDS,
    stage_timeout_seconds: int = DEFAULT_STAGE_TIMEOUT_SECONDS,
    rss_limit_bytes: int = DEFAULT_RSS_LIMIT_BYTES,
    capture_limit_bytes: int = DEFAULT_CAPTURE_LIMIT_BYTES,
    max_failures: int = DEFAULT_MAX_FAILURES,
    distance_compute: DistanceCompute | None = None,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    output_basis_path = Path(output_basis_path).expanduser().resolve()
    prior = read_json(output_basis_path)
    _validate_prior_radius_scale(prior)
    population_dependency = _validated_population_dependency(prior)
    base_config_path = Path(base_config_path).expanduser().resolve()
    base_config = read_json(base_config_path)
    population_base_config = Path(population_dependency["input"]["base_config"])
    if not population_base_config.is_absolute():
        population_base_config = PROJECT_ROOT / population_base_config
    if population_base_config.resolve() != base_config_path:
        raise ValueError("Base config path differs from the population preflight.")
    if (
        population_dependency["input"].get("base_config_sha256")
        != _file_sha256(base_config_path)
    ):
        raise ValueError("Base config changed after the population preflight.")

    normalized_heights = tuple(sorted({int(value) for value in heights}))
    if normalized_heights != DEFAULT_HEIGHTS:
        raise ValueError(f"This frozen discovery stage requires heights {DEFAULT_HEIGHTS}.")
    if genome_length != DEFAULT_GENOME_LENGTH:
        raise ValueError(f"This frozen discovery stage requires length {DEFAULT_GENOME_LENGTH}.")
    if expected_cna_starts != DEFAULT_EXPECTED_CNA_STARTS:
        raise ValueError(
            f"This frozen discovery stage requires expected CNA starts "
            f"{DEFAULT_EXPECTED_CNA_STARTS}."
        )
    if fixed_radius != DEFAULT_FIXED_RADIUS:
        raise ValueError(f"This frozen discovery stage requires radius {DEFAULT_FIXED_RADIUS}.")
    if isinstance(replicates, bool) or not isinstance(replicates, int):
        raise ValueError("replicates must be an integer.")
    if not 1 <= replicates <= MAX_REPLICATES:
        raise ValueError(f"replicates must be in [1,{MAX_REPLICATES}].")
    if isinstance(states_per_level, bool) or not isinstance(states_per_level, int):
        raise ValueError("states_per_level must be an integer.")
    if states_per_level <= 0 or states_per_level * 3 > MAX_PROFILE_COUNT_PER_CASE:
        raise ValueError("Fixed sampling exceeds the 18-profile per-case cap.")
    if isinstance(base_seed, bool) or not isinstance(base_seed, int) or base_seed < 0:
        raise ValueError("base_seed must be a nonnegative integer.")
    if isinstance(max_failures, bool) or not isinstance(max_failures, int) or max_failures <= 0:
        raise ValueError("max_failures must be positive.")
    for field, value in (
        ("cnp2cnp_timeout_seconds", cnp2cnp_timeout_seconds),
        ("stage_timeout_seconds", stage_timeout_seconds),
        ("rss_limit_bytes", rss_limit_bytes),
        ("capture_limit_bytes", capture_limit_bytes),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{field} must be a positive integer.")
    if any((2 ** (height + 1)) - 1 > STATIC_TRUTH_NODE_CAP for height in normalized_heights):
        raise ValueError("A requested height exceeds the static truth-node cap.")
    if base_config.get("SIMULATOR_SEMANTIC_VERSION") != "ctbf-cnp-state-simulator-v5":
        raise ValueError("Height ambiguity trend requires a CTBF v5 simulator config.")

    if distance_compute is None:
        runtime_config = replace(
            load_ctbs_runtime_config(),
            cnp2cnp_timeout_seconds=float(cnp2cnp_timeout_seconds),
            cnp2cnp_capture_limit_bytes=int(capture_limit_bytes),
        )
        compute = Cnp2CnpFileDistanceProvider(runtime_config).compute
    else:
        compute = distance_compute

    cases = []
    failure_count = 0
    stopped_early = False
    for replicate_index in range(replicates):
        for height in normalized_heights:
            record = _run_case(
                base_config=base_config,
                height=height,
                replicate_index=replicate_index,
                base_seed=base_seed,
                states_per_level=states_per_level,
                expected_cna_starts=expected_cna_starts,
                fixed_radius=fixed_radius,
                distance_compute=compute,
                injected_distance=distance_compute is not None,
                stage_timeout_seconds=stage_timeout_seconds,
                rss_limit_bytes=rss_limit_bytes,
            )
            cases.append(record)
            if record["status"] in {"simulation_failure", "distance_failure"}:
                failure_count += 1
                if failure_count >= max_failures:
                    stopped_early = True
                    break
        if stopped_early:
            break

    identities = {}
    for record in cases:
        if record["status"] == "success":
            identity = record["distance"]["identity"]
            identity_hash = record["distance"]["identity_sha256"]
            identities.setdefault(
                identity_hash,
                {"case_count": 0, "identity": identity},
            )["case_count"] += 1

    report = {
        "schema_version": SCHEMA_VERSION,
        "analysis_role": ANALYSIS_ROLE,
        "created_at_utc": created_at_utc or datetime.now(timezone.utc).isoformat(),
        "status": "stopped_at_failure_cap" if stopped_early else "complete",
        "scientific_role": {
            "paper_evidence_allowed": False,
            "discovery_only": True,
            "simulation_run": True,
            "cnp2cnp_run": distance_compute is None,
            "injected_distance_for_test": distance_compute is not None,
            "reconstruction_run": False,
            "evaluation_run": False,
            "adaptive_radius_run": False,
        },
        "prediction_audit": {
            "version": PREDICTION_VERSION,
            "written_after_t2b_before_multi_case_run": True,
            "primary_directional_predictions": {
                "fixed_r4_missing_fraction": "mean_H16_greater_than_mean_H8",
                "nearest_plausible_distance_mean": "mean_H16_greater_than_mean_H8",
                "half_coverage_radius_mean": "mean_H16_greater_than_mean_H8",
            },
            "secondary_no_direction": [
                "fixed_r4_tie_child_fraction",
                "unrestricted_nearest_tie_child_fraction",
                "four_cycle_counts",
            ],
            "reason": (
                "Wider temporal gaps with a fixed observed-state budget should "
                "increase distance-to-observed-parent and fixed-r4 copy-up exposure; "
                "exact ties need not increase because distance spreading is opposing."
            ),
        },
        "input": {
            "radius_scale_preflight": (
                output_basis_path.relative_to(PROJECT_ROOT).as_posix()
                if output_basis_path.is_relative_to(PROJECT_ROOT)
                else str(output_basis_path)
            ),
            "radius_scale_preflight_sha256": _file_sha256(output_basis_path),
            "base_config": (
                base_config_path.relative_to(PROJECT_ROOT).as_posix()
                if base_config_path.is_relative_to(PROJECT_ROOT)
                else str(base_config_path)
            ),
            "base_config_sha256": _file_sha256(base_config_path),
            "heights": list(normalized_heights),
            "genome_length": genome_length,
            "expected_cna_starts_per_attempted_child": expected_cna_starts,
            "relative_biopsy_positions": "3/7,5/7,1",
            "states_per_level": states_per_level,
            "replicates": replicates,
            "base_seed": base_seed,
            "seed_namespace": SEED_NAMESPACE,
            "common_simulation_seed_within_replicate_across_heights": True,
            "sampling_version": SAMPLING_VERSION,
            "fixed_reconstruction_radius_diagnostic": fixed_radius,
            "distance_semantics": CNP2CNP_SEMANTICS_VERSION,
            "cnp2cnp_timeout_seconds_per_process": cnp2cnp_timeout_seconds,
            "stage_timeout_seconds": stage_timeout_seconds,
            "rss_limit_bytes": rss_limit_bytes,
            "capture_limit_bytes": capture_limit_bytes,
            "max_failures": max_failures,
        },
        "resource_bound": {
            "planned_case_count": replicates * len(normalized_heights),
            "attempted_case_count": len(cases),
            "maximum_profile_count_per_case": states_per_level * 3,
            "maximum_directed_pair_entries_per_case": (
                states_per_level * 3 * (states_per_level * 3 - 1)
            ),
            "maximum_external_process_count_if_all_cases_eligible": (
                replicates * len(normalized_heights) * 2
            ),
            "sequential_execution": True,
        },
        "cases": cases,
        "aggregate": aggregate_cases(cases, normalized_heights),
        "distance_identity_counts": [
            {"identity_sha256": key, **identities[key]}
            for key in sorted(identities)
        ],
        "provenance": {
            "source_module": Path(__file__).relative_to(PROJECT_ROOT).as_posix(),
            "source_module_sha256": _file_sha256(Path(__file__)),
            "reads_existing_result_corpus": False,
            "writes_truth_trees_or_profiles": False,
        },
    }
    validate_height_ambiguity_report(report)
    return report


def validate_height_ambiguity_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unknown height-ambiguity schema.")
    if report.get("analysis_role") != ANALYSIS_ROLE:
        raise ValueError("Height-ambiguity report has the wrong role.")
    if report.get("status") not in {"complete", "stopped_at_failure_cap"}:
        raise ValueError("Height-ambiguity report has an unknown status.")
    role = report.get("scientific_role")
    if not isinstance(role, Mapping) or role.get("paper_evidence_allowed") is not False:
        raise ValueError("Height-ambiguity report must remain discovery-only.")
    for field in ("reconstruction_run", "evaluation_run", "adaptive_radius_run"):
        if role.get(field) is not False:
            raise ValueError(f"Height-ambiguity report must keep {field}=false.")
    allowed_statuses = {
        "success",
        "simulation_failure",
        "insufficient_unique_states",
        "distance_failure",
    }
    for record in report.get("cases", []):
        if record.get("status") not in allowed_statuses:
            raise ValueError("Height-ambiguity case has an unknown status.")
        if record["status"] == "success" and not isinstance(record.get("ambiguity"), Mapping):
            raise ValueError("Successful height-ambiguity case lacks diagnostics.")
    json.dumps(report, sort_keys=True, allow_nan=False)


def compact_summary(report: Mapping[str, Any], output: Path) -> dict[str, Any]:
    by_height = {}
    for height, block in report["aggregate"]["by_height"].items():
        metrics = block["case_metrics"]
        by_height[height] = {
            "status_counts": block["status_counts"],
            "truth_node_count_mean": block["truth_node_count"]["mean"],
            "distance_wall_time_seconds_mean": block[
                "distance_wall_time_seconds"
            ]["mean"],
            "fixed_r4_missing_fraction_mean": metrics[
                "fixed_r4_missing_fraction"
            ]["mean"],
            "nearest_plausible_distance_mean": metrics[
                "nearest_plausible_distance_mean"
            ]["mean"],
            "half_coverage_radius_mean": metrics[
                "half_coverage_radius_mean"
            ]["mean"],
            "fixed_r4_tie_child_fraction_mean": metrics[
                "fixed_r4_tie_child_fraction"
            ]["mean"],
            "unrestricted_nearest_tie_child_fraction_mean": metrics[
                "unrestricted_nearest_tie_child_fraction"
            ]["mean"],
        }
    return {
        "analysis_role": report["analysis_role"],
        "schema_version": report["schema_version"],
        "status": report["status"],
        "output": str(output.resolve()),
        "planned_case_count": report["resource_bound"]["planned_case_count"],
        "attempted_case_count": report["resource_bound"]["attempted_case_count"],
        "by_height": by_height,
        "paired_endpoint_differences": report["aggregate"][
            "paired_endpoint_differences"
        ],
        "distance_identity_count": len(report["distance_identity_counts"]),
        "next_stage": "review_height_ambiguity_before_any_reconstruction_run",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the bounded 12-block fixed-budget height ambiguity trend "
            "without reconstruction or evaluation."
        )
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--radius-scale-preflight",
        type=Path,
        default=DEFAULT_RADIUS_SCALE_PREFLIGHT,
    )
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
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
    parser.add_argument("--max-failures", type=int, default=DEFAULT_MAX_FAILURES)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    output = arguments.output.expanduser().resolve()
    if output.exists() and not arguments.overwrite:
        raise FileExistsError(
            f"Output already exists: {output}. Choose a new versioned path or pass --overwrite."
        )
    report = run_height_ambiguity_trend(
        output_basis_path=arguments.radius_scale_preflight,
        replicates=arguments.replicates,
        base_seed=arguments.base_seed,
        cnp2cnp_timeout_seconds=arguments.cnp2cnp_timeout_seconds,
        stage_timeout_seconds=arguments.stage_timeout_seconds,
        rss_limit_bytes=arguments.rss_limit_bytes,
        capture_limit_bytes=arguments.capture_limit_bytes,
        max_failures=arguments.max_failures,
    )
    write_json_atomic(output, report)
    print(json.dumps(compact_summary(report, output), indent=2, sort_keys=True))
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
