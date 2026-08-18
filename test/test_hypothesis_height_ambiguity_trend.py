import json

import numpy as np
import pytest

from algorithm_evaluation import hypothesis_height_ambiguity_trend as trend
from algorithm_evaluation.hypothesis_trend_distance_preflight import (
    ANALYSIS_ROLE as DISTANCE_ROLE,
    SCHEMA_VERSION as DISTANCE_SCHEMA,
)
from algorithm_evaluation.paper_pipeline_contract import write_json_atomic
from ctbs import DistanceMatrix
from distance_semantics import (
    CNP2CNP_SEMANTICS_VERSION,
    DISTANCE_PROVENANCE_SCHEMA_VERSION,
)
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


def _prior_radius_report(population_path=None, population_sha256=None):
    return {
        "schema_version": DISTANCE_SCHEMA,
        "analysis_role": DISTANCE_ROLE,
        "status": "success",
        "scientific_role": {
            "paper_evidence_allowed": False,
            "cnp2cnp_run": True,
            "reconstruction_run": False,
            "evaluation_run": False,
        },
        "input": {
            "case_key": "population-L50-H16",
            "genome_length": 50,
            "number_of_generations": 16,
            "states_per_level": 6,
            "distance_semantics": CNP2CNP_SEMANTICS_VERSION,
            "radii": [float(value) for value in range(4, 19)],
            "population_preflight": (
                str(population_path) if population_path is not None else "unused.json"
            ),
            "population_preflight_sha256": population_sha256 or ("0" * 64),
        },
    }


def test_transition_scale_separates_radius_graph_from_nearest_ties():
    parents = [
        Genotype([2, 2], "A", cell_id="A"),
        Genotype([2, 1], "B", cell_id="B"),
    ]
    children = [
        Genotype([1, 2], "C", cell_id="C"),
        Genotype([1, 1], "D", cell_id="D"),
    ]
    distance = DistanceMatrix(
        ids=["A", "B", "C", "D"],
        matrix=np.asarray(
            [
                [0, 1, 1, 1],
                [1, 0, 1, 1],
                [1, 1, 0, 1],
                [1, 1, 1, 0],
            ],
            dtype=float,
        ),
        provenance=_production_provenance(),
    )

    summary = trend.transition_scale_summary(parents, children, distance)

    assert summary["half_child_coverage_radius"] == 1
    assert summary["full_child_coverage_radius"] == 1
    assert summary["nearest_plausible_parent_child_degree"]["histogram"] == {
        "2": 2
    }
    assert summary["nearest_plausible_parent_graph"]["four_cycle_count"] == 1


def test_ambiguity_summary_reuses_fixed_candidate_graph_without_changing_values():
    parents = [
        Genotype([2, 2], "A", cell_id="A"),
        Genotype([2, 1], "B", cell_id="B"),
    ]
    children = [
        Genotype([1, 2], "C", cell_id="C"),
        Genotype([1, 1], "D", cell_id="D"),
    ]
    distance = DistanceMatrix(
        ids=["A", "B", "C", "D"],
        matrix=np.asarray(
            [
                [0, 1, 1, 1],
                [1, 0, 1, 1],
                [1, 1, 0, 1],
                [1, 1, 1, 0],
            ],
            dtype=float,
        ),
        provenance=_production_provenance(),
    )
    levels = [parents, children]
    candidate_graph = trend.candidate_graph_summary(
        levels,
        distance,
        radii=(1, 4, 8),
    )

    fresh = trend.ambiguity_case_summary(
        levels,
        (1, 2),
        distance,
        fixed_radius=4,
    )
    reused = trend.ambiguity_case_summary(
        levels,
        (1, 2),
        distance,
        fixed_radius=4,
        fixed_candidate_graph=candidate_graph,
    )

    assert reused == fresh


def test_height_trend_runs_three_bounded_cases_with_injected_distance(tmp_path):
    population_path = tmp_path / "population.json"
    write_json_atomic(
        population_path,
        {
            "input": {
                "base_config": str(trend.DEFAULT_BASE_CONFIG),
                "base_config_sha256": trend._file_sha256(trend.DEFAULT_BASE_CONFIG),
            }
        },
    )
    prior_path = tmp_path / "radius-scale.json"
    write_json_atomic(
        prior_path,
        _prior_radius_report(
            population_path,
            trend._file_sha256(population_path),
        ),
    )

    report = trend.run_height_ambiguity_trend(
        output_basis_path=prior_path,
        states_per_level=1,
        replicates=1,
        base_seed=31,
        stage_timeout_seconds=30,
        rss_limit_bytes=1024**3,
        max_failures=2,
        distance_compute=_injected_l1,
        created_at_utc="2026-08-05T00:00:00+00:00",
    )

    assert report["status"] == "complete"
    assert len(report["cases"]) == 3
    assert {record["height"] for record in report["cases"]} == {8, 12, 16}
    by_height = {record["height"]: record for record in report["cases"]}
    assert by_height[8]["status"] == "success"
    assert by_height[12]["status"] == "success"
    assert by_height[16]["status"] == "success"
    assert report["input"][
        "expected_cna_starts_per_attempted_child"
    ] == pytest.approx(0.1)
    assert report["scientific_role"]["reconstruction_run"] is False
    assert report["scientific_role"]["evaluation_run"] is False
    assert report["scientific_role"]["adaptive_radius_run"] is False
    assert report["resource_bound"]["maximum_profile_count_per_case"] == 3
    assert '"genome":' not in json.dumps(report)


def test_radius_scale_input_requires_complete_frozen_sweep():
    prior = _prior_radius_report()
    prior["input"]["radii"] = [4.0, 18.0]

    try:
        trend._validate_prior_radius_scale(prior)
    except ValueError as error:
        assert "frozen 4--18 sweep" in str(error)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("Expected an incomplete sweep rejection.")
