import json
import itertools
import math
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from algorithm_evaluation import simulator_reconstruction_intuition_probe as probe
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
        "external_process_count": 0,
    }


def _injected_l1(cells):
    ids = [cell.cell_id for cell in cells]
    profiles = np.asarray([cell.genome for cell in cells], dtype=float)
    matrix = np.abs(profiles[:, None, :] - profiles[None, :, :]).sum(axis=2)
    return DistanceMatrix(ids=ids, matrix=matrix, provenance=_production_provenance())


def _cell(node_id, generation, cell_id, value):
    return Genotype([value], node_id, generation=generation, cell_id=cell_id)


def test_rule_y_uses_ceiling_at_the_approved_heights():
    assert probe.rule_y_schedule(14) == (9, 12, 14)
    assert probe.rule_y_schedule(24) == (15, 20, 24)
    assert probe.rule_y_schedule(34) == (21, 28, 34)


def test_capped_sampling_takes_all_below_cap_and_seeded_six_above_cap():
    first = [_cell(index, 9, index, index) for index in range(3)]
    second = [_cell(10 + index, 12, 10 + index, index) for index in range(8)]
    third = [_cell(30 + index, 14, 30 + index, index) for index in range(6)]

    selected, rows = probe.select_capped_levels(
        [first, second, third],
        [9, 12, 14],
        base_seed=17,
        replicate_index=0,
        height=14,
    )
    repeated, repeated_rows = probe.select_capped_levels(
        [first, second, third],
        [9, 12, 14],
        base_seed=17,
        replicate_index=0,
        height=14,
    )

    assert [len(level) for level in selected] == [3, 6, 6]
    assert [cell.node_id for cell in selected[1]] == [
        cell.node_id for cell in repeated[1]
    ]
    assert rows == repeated_rows
    assert rows[0]["selection_mode"] == "all_available"
    assert rows[0]["sampling_seed_used"] is False
    assert rows[1]["selection_mode"] == "seeded_without_replacement_to_cap"
    assert rows[1]["sampling_seed_used"] is True


def test_truth_diagnostics_separate_hidden_forks_and_hidden_paths():
    tree = nx.DiGraph()
    attributes = {
        0: {"generation": 0, "cell_id": "root"},
        1: {"generation": 1, "cell_id": "same"},
        2: {"generation": 1, "cell_id": "branch"},
        3: {"generation": 2, "cell_id": "middle"},
        4: {"generation": 2, "cell_id": "same"},
        5: {"generation": 3, "cell_id": "late"},
    }
    tree.add_nodes_from((node, value) for node, value in attributes.items())
    tree.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 4), (3, 5)])
    selected = [
        [_cell(1, 1, "same", 1)],
        [_cell(4, 2, "same", 1)],
        [_cell(5, 3, "late", 2)],
    ]

    summary = probe.truth_sampling_diagnostics(tree, selected)

    assert summary["selected_unordered_pair_count"] == 3
    assert summary["comparable_ancestor_descendant_pair_count"] == 1
    assert summary["incomparable_pair_count"] == 2
    assert summary["hidden_fork_pair_count"] == 2
    assert summary["incomparable_same_state_pair_count"] == 1
    assert summary["later_with_sampled_ancestor_count"] == 1
    assert summary["sampled_ancestor_coverage_fraction"] == pytest.approx(0.5)
    assert summary["hidden_internal_nodes_to_nearest_sampled_ancestor"]["mean"] == 1
    assert summary["minimal_sampled_occurrence_count"] == 2
    assert summary["minimum_invented_edges_for_observed_only_arborescence"] == 1
    assert summary["observed_only_occurrence_arborescence_representable"] is False


def test_truth_diagnostics_match_brute_force_pair_classification():
    rng = np.random.default_rng(41)
    tree = nx.DiGraph()
    depths = {0: 0}
    tree.add_node(0, generation=0, cell_id=0)
    for node in range(1, 80):
        parent = int(rng.integers(0, node))
        depths[node] = depths[parent] + 1
        tree.add_node(
            node,
            generation=depths[node],
            cell_id=node % 9,
        )
        tree.add_edge(parent, node)

    selected_nodes = sorted(
        rng.choice(np.arange(1, 80), size=30, replace=False).tolist(),
        key=lambda node: (depths[node], node),
    )
    selected = [
        [
            _cell(node, depths[node], node % 9, node % 4)
            for node in selected_nodes[start:end]
        ]
        for start, end in ((0, 10), (10, 20), (20, 30))
    ]
    summary = probe.truth_sampling_diagnostics(tree, selected)

    selected_set = set(selected_nodes)
    ancestors = {node: nx.ancestors(tree, node) for node in selected_nodes}
    comparable = 0
    hidden_forks = 0
    sampled_forks = 0
    same_state = 0
    comparable_same = 0
    for left, right in itertools.combinations(selected_nodes, 2):
        is_comparable = left in ancestors[right] or right in ancestors[left]
        is_same = left % 9 == right % 9
        comparable += int(is_comparable)
        same_state += int(is_same)
        comparable_same += int(is_comparable and is_same)
        if is_comparable:
            continue
        lca = nx.lowest_common_ancestor(tree, left, right)
        sampled_forks += int(lca in selected_set)
        hidden_forks += int(lca not in selected_set)

    pair_count = math.comb(len(selected_nodes), 2)
    assert summary["selected_unordered_pair_count"] == pair_count
    assert summary["comparable_ancestor_descendant_pair_count"] == comparable
    assert summary["incomparable_pair_count"] == pair_count - comparable
    assert summary["hidden_fork_pair_count"] == hidden_forks
    assert summary["sampled_lca_fork_pair_count"] == sampled_forks
    assert summary["same_state_pair_count"] == same_state
    assert summary["comparable_same_state_pair_count"] == comparable_same
    assert summary["incomparable_same_state_pair_count"] == (
        same_state - comparable_same
    )


def test_dense_truth_diagnostics_do_not_call_pairwise_networkx_helpers(monkeypatch):
    leaf_count = 3000
    tree = nx.DiGraph()
    tree.add_node(0, generation=0, cell_id="root")
    cells = []
    for node in range(1, leaf_count + 1):
        tree.add_node(node, generation=1, cell_id=node)
        tree.add_edge(0, node)
        cells.append(_cell(node, 1, node, 2))

    def forbidden(*_args, **_kwargs):
        raise AssertionError("pairwise NetworkX traversal was called")

    monkeypatch.setattr(nx, "lowest_common_ancestor", forbidden)
    monkeypatch.setattr(nx, "shortest_path_length", forbidden)
    summary = probe.truth_sampling_diagnostics(
        tree,
        [cells[:1000], cells[1000:2000], cells[2000:]],
    )

    assert summary["selected_unordered_pair_count"] == math.comb(leaf_count, 2)
    assert summary["hidden_fork_pair_count"] == math.comb(leaf_count, 2)
    assert summary["comparable_ancestor_descendant_pair_count"] == 0


def test_one_block_runs_all_six_arms_with_compact_injected_output():
    report = probe.run_probe(
        replicates=1,
        base_seed=31,
        simulation_timeout_seconds=60,
        distance_timeout_seconds=30,
        reconstruction_timeout_seconds=30,
        evaluation_timeout_seconds=30,
        rss_limit_bytes=2 * 1024**3,
        max_case_dependency_failures=3,
        distance_compute=_injected_l1,
        created_at_utc="2026-08-12T00:00:00+00:00",
    )

    assert report["status"] == "complete"
    assert report["scientific_role"]["paper_evidence_allowed"] is False
    assert report["scientific_role"]["reconstruction_run"] is True
    assert report["scientific_role"]["evaluation_run"] is True
    assert report["scientific_role"]["injected_distance_for_test"] is True
    assert [record["height"] for record in report["cases"]] == [14, 24, 34]
    assert report["aggregate"]["common_seed_prefix_consistency"][
        "all_planned_common_seed_prefix_checks_available_and_passed"
    ] is True
    h14 = report["aggregate"]["by_height"]["14"]
    assert [row["generation"] for row in h14["biopsy_levels"]] == [9, 12, 14]
    assert h14["sampling_metrics"]["bidirectional_ordered_pair_bound"][
        "count"
    ] == 1
    for record in report["cases"]:
        assert record["status"] == "complete"
        assert tuple(arm["arm_id"] for arm in record["arms"]) == probe.ARM_IDS
        partial = next(
            arm for arm in record["arms"] if arm["arm_id"] == "classical_partial"
        )
        assert set(partial["evaluation"]["metrics"]) == {"grf"}
        assert all(
            row["realized_occurrence_count"]
            == min(6, row["available_distinct_state_count"])
            for row in record["simulation_summary"]["sampling"]
        )
    serialized = json.dumps(report, sort_keys=True)
    for raw_key in ('"cnp":', '"genome":', '"tree":', '"matrix":', '"node_id":'):
        assert raw_key not in serialized
    printed = probe.compact_summary(report, Path("/tmp/probe.json"))
    assert printed["by_height"]["14"]["biopsy_levels"][0]["generation"] == 9
    assert set(printed["by_height"]["14"]["arms"]) == set(probe.ARM_IDS)


def test_unavailable_prefix_comparison_is_not_a_false_invariant_failure():
    digest = "a" * 64
    cases = [
        {
            "height": 14,
            "replicate_index": 0,
            "simulation_summary": {
                "truth_prefix_sha256_by_height": {"14": digest}
            },
        },
        {
            "height": 24,
            "replicate_index": 0,
            "simulation_summary": None,
        },
    ]

    summary = probe._prefix_consistency(cases)

    assert summary["fully_available_comparison_count"] == 0
    assert summary["evaluable_comparison_count"] == 0
    assert summary["evaluable_failure_count"] == 0
    assert summary["all_evaluable_common_seed_prefix_checks_passed"] is True
    assert (
        summary["all_planned_common_seed_prefix_checks_available_and_passed"]
        is False
    )


def test_prefix_mismatch_is_detected_from_two_available_horizons():
    cases = [
        {
            "height": 14,
            "replicate_index": 0,
            "simulation_summary": {
                "truth_prefix_sha256_by_height": {"14": "a" * 64}
            },
        },
        {
            "height": 24,
            "replicate_index": 0,
            "simulation_summary": {
                "truth_prefix_sha256_by_height": {
                    "14": "b" * 64,
                    "24": "c" * 64,
                }
            },
        },
        {
            "height": 34,
            "replicate_index": 0,
            "simulation_summary": None,
        },
    ]

    summary = probe._prefix_consistency(cases)

    assert summary["evaluable_comparison_count"] == 1
    assert summary["evaluable_failure_count"] == 1
    assert summary["all_evaluable_common_seed_prefix_checks_passed"] is False


def test_report_validation_rejects_raw_profile_fields():
    report = {
        "schema_version": probe.SCHEMA_VERSION,
        "analysis_role": probe.ANALYSIS_ROLE,
        "status": "complete",
        "scientific_role": {
            "paper_evidence_allowed": False,
            "discovery_only": True,
            "simulation_run": True,
            "reconstruction_run": True,
            "evaluation_run": True,
            "selects_simulator_parameters_from_accuracy": False,
            "freezes_paper_height_set": False,
        },
        "cases": [],
        "leak": {"cnp": [2, 2]},
    }

    with pytest.raises(ValueError, match="forbidden raw fields"):
        probe.validate_report(report)
