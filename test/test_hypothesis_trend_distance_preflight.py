import math

import numpy as np

from algorithm_evaluation import hypothesis_trend_distance_preflight as preflight
from algorithm_evaluation import hypothesis_trend_population_preflight as population
from algorithm_evaluation.paper_pipeline_contract import write_json_atomic
from ctbs import DistanceMatrix
from distance_semantics import (
    CNP2CNP_SEMANTICS_VERSION,
    DISTANCE_PROVENANCE_SCHEMA_VERSION,
)
from reconstructor_plausibility import is_biologically_plausible_ancestor
from simulator import Genotype


def _production_provenance():
    return {
        "schema_version": DISTANCE_PROVENANCE_SCHEMA_VERSION,
        "semantics_version": CNP2CNP_SEMANTICS_VERSION,
        "metric": "cnp2cnp",
        "distance_mode": "any",
        "symmetrization": "minimum_bidirectional",
        "formula": "min(d_any(u,v),d_any(v,u))",
        "construction": "opposite_order_matrix_mode",
    }


def _injected_l1(cells):
    ids = [cell.cell_id for cell in cells]
    profiles = np.asarray([cell.genome for cell in cells], dtype=float)
    matrix = np.abs(profiles[:, None, :] - profiles[None, :, :]).sum(axis=2)
    return DistanceMatrix(ids=ids, matrix=matrix, provenance=_production_provenance())


def test_candidate_graph_reports_exact_partial_ties_and_four_cycle():
    parents = [
        Genotype([2, 2], "A", cell_id="A"),
        Genotype([2, 1], "B", cell_id="B"),
    ]
    children = [
        Genotype([1, 2], "C", cell_id="C"),
        Genotype([1, 1], "D", cell_id="D"),
    ]
    ids = ["A", "B", "C", "D"]
    matrix = np.array(
        [
            [0, 1, 1, 1],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [1, 1, 1, 0],
        ],
        dtype=float,
    )
    distance = DistanceMatrix(ids=ids, matrix=matrix, provenance=_production_provenance())

    summary = preflight.candidate_graph_summary(
        [parents, children],
        distance,
        radii=[1],
    )["transitions"][0]["radii"][0]

    assert summary["plausible_radius_child_degree"]["histogram"] == {"2": 2}
    assert summary["minimum_parent_child_degree"]["multiple_count"] == 2
    assert summary["parent_pair_plausible_codegree"]["histogram"] == {"2": 1}
    assert summary["plausible_radius_four_cycle_count"] == 1


def _reference_candidate_layer(parents, children, distance, radius):
    id_to_index = {label: index for index, label in enumerate(distance.ids)}
    matrix = np.asarray(distance.matrix, dtype=float)
    raw_sets = []
    plausible_sets = []
    minimum_sets = []
    same_state_priority_count = 0
    missing_parent_count = 0
    for child in children:
        child_index = id_to_index[child.cell_id]
        raw = {
            offset
            for offset, parent in enumerate(parents)
            if matrix[child_index, id_to_index[parent.cell_id]] <= radius
        }
        plausible = {
            offset
            for offset in raw
            if is_biologically_plausible_ancestor(parents[offset], child)
        }
        same = {
            offset
            for offset in raw
            if parents[offset].cell_id == child.cell_id
        }
        if same:
            selected = same
            same_state_priority_count += 1
        elif plausible:
            minimum_distance = min(
                matrix[child_index, id_to_index[parents[offset].cell_id]]
                for offset in plausible
            )
            selected = {
                offset
                for offset in plausible
                if matrix[child_index, id_to_index[parents[offset].cell_id]]
                == minimum_distance
            }
        else:
            selected = set()
            missing_parent_count += 1
        raw_sets.append(raw)
        plausible_sets.append(plausible)
        minimum_sets.append(selected)

    codegrees = []
    four_cycles = 0
    for left in range(len(parents)):
        for right in range(left + 1, len(parents)):
            codegree = sum(
                left in candidates and right in candidates
                for candidates in plausible_sets
            )
            codegrees.append(codegree)
            four_cycles += math.comb(codegree, 2) if codegree >= 2 else 0
    return {
        "parent_count": len(parents),
        "child_count": len(children),
        "radius": float(radius),
        "raw_radius_child_degree": preflight._degree_summary(
            [len(value) for value in raw_sets]
        ),
        "plausible_radius_child_degree": preflight._degree_summary(
            [len(value) for value in plausible_sets]
        ),
        "minimum_parent_child_degree": preflight._degree_summary(
            [len(value) for value in minimum_sets]
        ),
        "same_state_priority_count": same_state_priority_count,
        "missing_parent_count": missing_parent_count,
        "parent_pair_plausible_codegree": preflight._degree_summary(codegrees),
        "plausible_radius_four_cycle_count": four_cycles,
    }


def test_vectorized_candidate_graph_matches_pair_loop_reference():
    rng = np.random.default_rng(73)
    parent_profiles = rng.integers(0, 4, size=(8, 12))
    parents = [
        Genotype(profile, f"p{index}", cell_id=f"p{index}")
        for index, profile in enumerate(parent_profiles)
    ]
    children = [
        Genotype(parent_profiles[index].copy(), f"c{index}", cell_id=f"p{index}")
        for index in range(2)
    ]
    children.extend(
        Genotype(profile, f"c{index + 2}", cell_id=f"c{index + 2}")
        for index, profile in enumerate(rng.integers(0, 4, size=(10, 12)))
    )
    cells_by_label = {}
    for cell in [*parents, *children]:
        cells_by_label.setdefault(cell.cell_id, cell)
    cells = list(cells_by_label.values())
    profiles = np.asarray([cell.genome for cell in cells], dtype=float)
    distance = DistanceMatrix(
        ids=[cell.cell_id for cell in cells],
        matrix=np.abs(profiles[:, np.newaxis] - profiles[np.newaxis, :]).sum(
            axis=2
        ),
        provenance=_production_provenance(),
    )

    radii = (0.0, 4.0, 12.0)
    observed = preflight.candidate_graph_summary(
        [parents, children],
        distance,
        radii=radii,
    )["transitions"][0]["radii"]
    expected = [
        _reference_candidate_layer(parents, children, distance, radius)
        for radius in radii
    ]
    assert observed == expected


def test_distance_preflight_replays_compact_population_case_with_injected_distance(tmp_path):
    population_report = population.run_population_preflight(
        lengths=[10],
        heights=[7],
        length_trend_height=7,
        height_trend_length=10,
        base_seed=19,
        timeout_seconds_per_case=30,
        rss_limit_bytes=1024**3,
        created_at_utc="2026-08-05T00:00:00+00:00",
    )
    population_path = tmp_path / "population.json"
    write_json_atomic(population_path, population_report)

    report = preflight.run_distance_preflight(
        population_preflight_path=population_path,
        case_key="population-L10-H7",
        states_per_level=1,
        radii=[1, 2],
        stage_timeout_seconds=30,
        rss_limit_bytes=1024**3,
        distance_compute=_injected_l1,
        created_at_utc="2026-08-05T00:00:00+00:00",
    )

    assert report["status"] == "success"
    assert report["scientific_role"]["cnp2cnp_run"] is False
    assert report["scientific_role"]["injected_distance_for_test"] is True
    assert report["scientific_role"]["reconstruction_run"] is False
    assert report["simulation"]["summary"]["selected_occurrence_count"] == 3
    assert report["distance"]["summary"]["profile_count"] <= 3
    assert report["ambiguity"]["transition_count"] == 2


def test_distance_preflight_rejects_budget_above_population_limit(tmp_path):
    population_report = population.run_population_preflight(
        lengths=[10],
        heights=[7],
        length_trend_height=7,
        height_trend_length=10,
        base_seed=23,
        timeout_seconds_per_case=30,
        rss_limit_bytes=1024**3,
        created_at_utc="2026-08-05T00:00:00+00:00",
    )
    population_path = tmp_path / "population.json"
    write_json_atomic(population_path, population_report)

    try:
        preflight.run_distance_preflight(
            population_preflight_path=population_path,
            case_key="population-L10-H7",
            states_per_level=999,
            distance_compute=_injected_l1,
        )
    except ValueError as error:
        assert "exceeds the population-preflight limit" in str(error)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("Expected an over-budget preflight rejection.")
