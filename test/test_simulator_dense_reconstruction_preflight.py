import json

import numpy as np
import pytest

from algorithm_evaluation import simulator_dense_reconstruction_preflight as probe
from ctbs import DistanceMatrix
from distance_semantics import (
    CNP2CNP_SEMANTICS_VERSION,
    DISTANCE_PROVENANCE_SCHEMA_VERSION,
)


def _injected_l1(cells):
    ids = [cell.cell_id for cell in cells]
    profiles = np.asarray([cell.genome for cell in cells], dtype=float)
    matrix = np.abs(profiles[:, None, :] - profiles[None, :, :]).sum(axis=2)
    return DistanceMatrix(
        ids=ids,
        matrix=matrix,
        provenance={
            "schema_version": DISTANCE_PROVENANCE_SCHEMA_VERSION,
            "semantics_version": CNP2CNP_SEMANTICS_VERSION,
            "metric": "cnp2cnp",
            "distance_mode": "any",
            "symmetrization": "minimum_bidirectional",
            "formula": "min(d_any(u,v),d_any(v,u))",
            "construction": "opposite_order_matrix_mode",
            "external_process_count": 0,
        },
    )


def test_registered_case_is_the_declared_329_state_bound():
    assert probe.REGISTERED_HEIGHT == 34
    assert probe.REGISTERED_REPLICATE_INDEX == 11
    assert probe.EXPECTED_UNIQUE_STATE_COUNT == 329
    assert probe.EXPECTED_ORDERED_PAIR_COUNT == 329 * 328


def test_reference_check_detects_dense_sampling_mismatch():
    prefix = {"14": "a" * 64}
    observed = {
        "truth_prefix_sha256_by_height": prefix,
        "truth_node_count": 10,
        "truth_edge_count": 9,
        "available_distinct_state_count_by_generation": [
            {"generation": 9, "count": 3}
        ],
        "sampling": [{"realized_occurrence_count": 3}],
        "selected_occurrence_count": 3,
        "selected_unique_state_count": 3,
        "bidirectional_ordered_pair_bound": 6,
        "capped_control_realized_counts": [3],
        "capped_control_unique_state_count": 3,
    }
    dense_summary = {
        "sampling": [{"realized_occurrence_count": 2}],
        "selected_occurrence_count": 3,
        "selected_unique_state_count": 3,
        "projected_bidirectional_ordered_pair_count": 6,
    }
    fraction_case = {
        "simulation_summary": {
            "truth_prefix_sha256_by_height": prefix,
            "truth_node_count": 10,
            "truth_edge_count": 9,
            "available_distinct_state_count_by_generation": [
                {"generation": 9, "count": 3}
            ],
                "conditions": {
                    "fraction_50": {"summary": dense_summary},
                    "capped_six_control": {
                        "summary": {
                            **dense_summary,
                            "sampling": [{"realized_occurrence_count": 3}],
                        }
                    },
            },
        }
    }
    sparse_case = {
        "simulation_summary": {
            "truth_prefix_sha256_by_height": prefix,
            "selected_unique_state_count": 3,
        }
    }

    result = probe._reference_check(
        observed=observed,
        fraction_case=fraction_case,
        sparse_case=sparse_case,
    )

    assert result["performed"] is True
    assert result["passed"] is False
    assert result["mismatched_fields"] == ["dense_sampling_rows"]


def test_small_injected_case_runs_all_six_arms_without_reference_corpus():
    report = probe.run_preflight(
        height=14,
        replicate_index=0,
        base_seed=31,
        distance_compute=_injected_l1,
        enforce_registered_references=False,
        simulation_timeout_seconds=60,
        distance_timeout_seconds=30,
        diagnostic_timeout_seconds=30,
        reconstruction_timeout_seconds=30,
        evaluation_timeout_seconds=30,
        created_at_utc="2026-08-13T00:00:00+00:00",
    )

    assert report["status"] == "complete"
    assert report["preflight_verdict"] == (
        "technical_smoke_only_injected_distance"
    )
    assert report["case"]["reference_check"]["performed"] is False
    assert tuple(arm["arm_id"] for arm in report["case"]["arms"]) == probe.ARM_IDS
    assert all(arm["status"] == "success" for arm in report["case"]["arms"])
    assert report["scientific_role"]["accuracy_interpretation_allowed"] is False
    assert report["provenance"]["temporal_solver"]["implementation_version"] == (
        "ctbf-compact-chu-liu-edmonds-v1"
    )
    serialized = json.dumps(report, sort_keys=True)
    for raw_key in ('"cnp":', '"genome":', '"tree":', '"matrix":', '"node_id":'):
        assert raw_key not in serialized
    probe.validate_report(json.loads(serialized))
    missing_solver_identity = json.loads(serialized)
    missing_solver_identity["provenance"].pop("temporal_solver")
    with pytest.raises(ValueError, match="temporal-solver identity"):
        probe.validate_report(missing_solver_identity)


def test_production_preflight_rejects_a_nonregistered_case_before_execution():
    with pytest.raises(ValueError, match="frozen to H34"):
        probe.run_preflight(height=14, replicate_index=0)


def test_report_validation_rejects_raw_profiles():
    report = {
        "schema_version": probe.SCHEMA_VERSION,
        "analysis_role": probe.ANALYSIS_ROLE,
        "status": "failed",
        "preflight_verdict": "reject_dependency_distance_failure",
        "scientific_role": {
            "paper_evidence_allowed": False,
            "technical_preflight_only": True,
            "reconstruction_run": True,
            "evaluation_run": True,
            "accuracy_interpretation_allowed": False,
            "authorizes_full_probe": False,
            "cnp2cnp_run": False,
        },
        "case": {"status": "distance_failure"},
        "leak": {"cnp": [2, 2]},
    }

    with pytest.raises(ValueError, match="forbidden raw fields"):
        probe.validate_report(report)
