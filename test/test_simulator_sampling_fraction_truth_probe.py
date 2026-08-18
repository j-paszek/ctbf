import json

import networkx as nx
import pytest

from algorithm_evaluation import simulator_sampling_fraction_truth_probe as probe
from algorithm_evaluation.simulator_reconstruction_intuition_probe import (
    select_capped_levels,
)
from simulator import Genotype


def _cell(node_id, generation, cell_id=None):
    return Genotype(
        [node_id + 1],
        node_id,
        generation=generation,
        cell_id=node_id if cell_id is None else cell_id,
    )


def test_hybrid_sample_size_uses_fraction_with_available_lower_bound():
    assert probe.hybrid_sample_size(0, 0.25) == 0
    assert probe.hybrid_sample_size(1, 0.05) == 1
    assert probe.hybrid_sample_size(5, 0.50) == 5
    assert probe.hybrid_sample_size(20, 0.05) == 6
    assert probe.hybrid_sample_size(20, 0.25) == 6
    assert probe.hybrid_sample_size(20, 0.50) == 10
    assert probe.hybrid_sample_size(21, 0.25) == 6
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        probe.hybrid_sample_size(20, 1.1)


def test_nested_fraction_selection_reproduces_capped_six_control():
    levels = [
        [_cell(index, 9) for index in range(3)],
        [_cell(10 + index, 12) for index in range(8)],
        [_cell(30 + index, 14) for index in range(40)],
    ]
    selected, rows, nesting = probe.select_nested_fraction_levels(
        levels,
        [9, 12, 14],
        base_seed=17,
        replicate_index=0,
        height=14,
    )
    capped, _capped_rows = select_capped_levels(
        levels,
        [9, 12, 14],
        base_seed=17,
        replicate_index=0,
        height=14,
    )

    assert [len(level) for level in selected["capped_six_control"]] == [3, 6, 6]
    assert [
        [cell.node_id for cell in level]
        for level in selected["capped_six_control"]
    ] == [[cell.node_id for cell in level] for level in capped]
    assert [
        len(level) for level in selected["fraction_25"]
    ] == [3, 6, 10]
    assert [
        len(level) for level in selected["fraction_50"]
    ] == [3, 6, 20]
    assert all(row["nested"] for row in nesting)
    for generation_index in range(3):
        condition_sets = [
            {cell.node_id for cell in selected[condition_id][generation_index]}
            for condition_id in probe.CONDITION_IDS
        ]
        assert all(
            smaller <= larger
            for smaller, larger in zip(condition_sets, condition_sets[1:])
        )
    assert rows["fraction_25"][2]["sample_size_driver"] == "fraction"


def test_cross_biopsy_diagnostics_separate_adjacent_and_any_earlier_ancestry():
    tree = nx.DiGraph()
    for node, generation in ((0, 0), (1, 1), (2, 1), (3, 2), (4, 2), (5, 3)):
        tree.add_node(node, generation=generation, cell_id=node)
    tree.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 4), (3, 5)])
    selected = [
        [_cell(1, 1), _cell(2, 1)],
        [_cell(4, 2)],
        [_cell(5, 3)],
    ]

    summary = probe.cross_biopsy_relation_diagnostics(tree, selected)

    assert summary["cross_biopsy_unordered_pair_count"] == 5
    assert summary["cross_biopsy_ancestor_pair_count"] == 2
    assert summary["cross_biopsy_ancestor_pair_fraction"] == pytest.approx(0.4)
    assert summary["adjacent_sampled_ancestor_coverage_fraction"] == 0.5
    assert summary["any_earlier_sampled_ancestor_coverage_fraction"] == 1.0
    assert summary["minimal_sampled_occurrence_count"] == 2
    assert summary["minimum_invented_edges_for_observed_only_arborescence"] == 1
    assert summary["observed_only_occurrence_arborescence_representable"] is False


def test_one_small_seed_block_runs_truth_only_with_all_nested_conditions():
    report = probe.run_probe(
        replicates=1,
        base_seed=31,
        timeout_seconds=60,
        created_at_utc="2026-08-13T00:00:00+00:00",
    )

    assert report["status"] == "complete"
    assert report["scientific_role"]["cnp2cnp_run"] is False
    assert report["scientific_role"]["reconstruction_run"] is False
    assert report["scientific_role"]["evaluation_run"] is False
    assert [case["height"] for case in report["cases"]] == [14, 24, 34]
    for case in report["cases"]:
        assert case["status"] == "complete"
        assert set(case["simulation_summary"]["conditions"]) == set(
            probe.CONDITION_IDS
        )
        assert case["simulation_summary"]["all_nested_checks_passed"] is True
    serialized = json.dumps(report, sort_keys=True)
    for raw_key in ('"cnp":', '"genome":', '"tree":', '"matrix":', '"node_id":'):
        assert raw_key not in serialized
    probe.validate_report(json.loads(serialized))


def test_reference_check_detects_compact_control_mismatch():
    diagnostics = {"metric": 1}
    prepared = {
        "truth_prefix_sha256_by_height": {"14": "a" * 64},
        "available_distinct_state_count_by_generation": [
            {"generation": 9, "count": 3}
        ],
        "conditions": {
            "capped_six_control": {
                "summary": {
                    "sampling": [{"realized_occurrence_count": 3}],
                    "selected_occurrence_count": 3,
                    "selected_unique_state_count": 3,
                    "truth_sampling_diagnostics": diagnostics,
                }
            }
        },
    }
    reference = {
        "simulation_summary": {
            "truth_prefix_sha256_by_height": {"14": "b" * 64},
            "sampling": [
                {
                    "available_distinct_state_count": 3,
                    "realized_occurrence_count": 3,
                }
            ],
            "selected_occurrence_count": 3,
            "selected_unique_state_count": 3,
            "truth_sampling_diagnostics": diagnostics,
        }
    }

    result = probe._reference_case_check(prepared, reference)

    assert result["performed"] is True
    assert result["passed"] is False
    assert result["mismatched_fields"] == ["truth_prefix_sha256_by_height"]


def test_report_validation_rejects_raw_profiles():
    report = {
        "schema_version": probe.SCHEMA_VERSION,
        "analysis_role": probe.ANALYSIS_ROLE,
        "status": "complete",
        "scientific_role": {
            "paper_evidence_allowed": False,
            "discovery_only": True,
            "simulation_run": True,
            "truth_diagnostics_run": True,
            "cnp2cnp_run": False,
            "reconstruction_run": False,
            "evaluation_run": False,
            "selects_simulator_parameters_from_accuracy": False,
            "freezes_paper_sampling_design": False,
        },
        "cases": [],
        "leak": {"cnp": [2, 2]},
    }

    with pytest.raises(ValueError, match="forbidden raw fields"):
        probe.validate_report(report)
