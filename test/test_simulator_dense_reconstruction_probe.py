import json

import numpy as np
import pytest

from algorithm_evaluation import simulator_dense_reconstruction_probe as probe
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


def _successful_arm(arm_id, value):
    metrics = {
        metric: value
        for metric in probe.ARM_ENDPOINTS[arm_id]["declared_metrics"]
    }
    return {
        "arm_id": arm_id,
        "status": "success",
        "evaluation": {"metrics": metrics},
    }


def test_registered_design_matches_approved_dense_probe():
    assert probe.DEFAULT_REPLICATES == 12
    assert tuple(probe.APPROVED_SCHEDULES) == (14, 24, 34)
    assert probe.TARGET_FRACTION == 0.5
    assert probe.DEFAULT_LOWER_BOUND == 6
    assert probe.DEFAULT_RSS_LIMIT_BYTES == 4 * 1024**3
    assert probe.EXPECTED_MAX_SELECTED_OCCURRENCE_COUNT == 395
    assert probe.EXPECTED_MAX_UNIQUE_STATE_COUNT == 329
    assert probe.EXPECTED_MAX_DISTANCE_MATRIX_CELL_COUNT == 108_241
    assert probe.EXPECTED_MAX_ORDERED_PAIR_COUNT == 107_912
    assert probe.EXPECTED_PREFLIGHT_SHA256 == (
        "431fddca7d73d9da0ead156d1dd4cc1a1a2085d5b0a5b0bbf3a81fed8b7144e9"
    )


def test_truth_reference_preserves_paired_values_and_direction():
    dense_values = {
        metric: float(index + 2)
        for index, metric in enumerate(probe.TRUTH_COMPARISON_DIRECTIONS)
    }
    capped_values = {
        metric: float(index + 1)
        for index, metric in enumerate(probe.TRUTH_COMPARISON_DIRECTIONS)
    }
    fraction_case = {
        "simulation_summary": {
            "conditions": {
                "fraction_50": {
                    "summary": {
                        "scalar_metrics": dense_values,
                        "cross_biopsy_relation_diagnostics": {
                            "observed_only_occurrence_arborescence_representable": False
                        },
                    }
                },
                "capped_six_control": {
                    "summary": {
                        "scalar_metrics": capped_values,
                        "cross_biopsy_relation_diagnostics": {
                            "observed_only_occurrence_arborescence_representable": True
                        },
                    }
                },
            }
        }
    }

    reference = probe._truth_reference(fraction_case)

    row = reference["paired_truth_metrics"][
        "cross_biopsy_ancestor_pair_fraction"
    ]
    assert row == {
        "preferred_direction": "higher",
        "fraction50": dense_values["cross_biopsy_ancestor_pair_fraction"],
        "capped_six": capped_values["cross_biopsy_ancestor_pair_fraction"],
        "difference": 1.0,
    }
    assert reference["fraction50_observed_only_arborescence_representable"] is False
    assert reference["capped_six_observed_only_arborescence_representable"] is True


def test_reconstruction_comparison_is_within_arm_and_paired():
    dense_case = {
        "arms": [_successful_arm(arm_id, 0.75) for arm_id in probe.ARM_IDS]
    }
    capped_case = {
        "case_key": "capped-H14-R001",
        "simulation_summary": {"selected_unique_state_count": 12},
        "arms": [_successful_arm(arm_id, 0.50) for arm_id in probe.ARM_IDS],
    }

    comparison = probe._paired_reconstruction_comparison(dense_case, capped_case)

    assert set(comparison["arms"]) == set(probe.ARM_IDS)
    for arm_id, arm in comparison["arms"].items():
        assert arm["fraction50_status"] == "success"
        assert arm["capped_six_status"] == "success"
        for metric in probe.ARM_ENDPOINTS[arm_id]["declared_metrics"]:
            assert arm["paired_metrics"][metric] == {
                "preferred_direction": "higher",
                "fraction50": 0.75,
                "capped_six": 0.50,
                "difference": 0.25,
            }


def test_small_injected_case_runs_all_six_arms_without_reference_corpus():
    report = probe.run_probe(
        replicates=1,
        heights=(14,),
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
    assert report["scientific_role"]["cnp2cnp_run"] is False
    assert report["scientific_role"]["injected_distance_for_test"] is True
    assert report["resource_bound"]["attempted_case_count"] == 1
    assert set(report["aggregate"]["by_height"]) == {"14"}
    assert report["aggregate"]["paired_height_endpoint_differences"][
        "distinct_endpoint_heights"
    ] is False
    case = report["cases"][0]
    assert case["reference_check"]["performed"] is False
    assert case["truth_reference"] is None
    assert case["capped_comparison"] is None
    assert tuple(arm["arm_id"] for arm in case["arms"]) == probe.ARM_IDS
    assert all(arm["status"] == "success" for arm in case["arms"])

    serialized = json.dumps(report, sort_keys=True)
    for raw_key in ('"cnp":', '"genome":', '"tree":', '"matrix":', '"node_id":'):
        assert raw_key not in serialized
    probe.validate_report(json.loads(serialized))

    leaked = json.loads(serialized)
    leaked["leak"] = {"cnp": [2, 2]}
    with pytest.raises(ValueError, match="forbidden raw fields"):
        probe.validate_report(leaked)


def test_production_probe_requires_both_references_and_passing_preflight():
    with pytest.raises(ValueError, match="requires both references and preflight"):
        probe.run_probe()


def test_unregistered_preflight_artifact_cannot_authorize_execution(tmp_path):
    path = tmp_path / "unregistered-preflight.json"
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="owner-reviewed artifact"):
        probe._load_preflight_authorization(
            path,
            base_config_sha256="0" * 64,
            base_seed=probe.DEFAULT_BASE_SEED,
            enforce_registered_hash=True,
        )
