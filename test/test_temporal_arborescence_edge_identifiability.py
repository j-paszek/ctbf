from pathlib import Path
import sys

import networkx as nx
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithm_evaluation import temporal_arborescence_edge_identifiability as audit
from ctbs import DistanceMatrix
from distance_semantics import CNP2CNP_SEMANTICS_VERSION


def _distance(ids, matrix):
    return DistanceMatrix(
        ids=ids,
        matrix=np.asarray(matrix, dtype=float),
        provenance={"semantics_version": CNP2CNP_SEMANTICS_VERSION},
    )


def _tree(nodes, edges):
    tree = nx.DiGraph()
    for node_id, cell_id, level, genome in nodes:
        tree.add_node(
            node_id,
            cell_id=cell_id,
            biopsy_level=level,
            genome=np.asarray(genome),
        )
    for parent, child, weight in edges:
        tree.add_edge(parent, child, weight=float(weight))
    assert nx.is_arborescence(tree)
    return tree


def test_unique_scientific_optimum_forces_selected_edges_and_temporal_root():
    tree = _tree(
        [
            (0, "A", 0, [1]),
            (1, "B", 1, [2]),
            (2, "C", 2, [3]),
        ],
        [(0, 1, 1), (1, 2, 1)],
    )
    distance = _distance(
        ["A", "B", "C"],
        [[0, 1, 10], [1, 0, 1], [10, 1, 0]],
    )

    result = audit.audit_selected_tree(tree, distance, use_time=True)

    assert result["baseline_scientific_objective"]["values"] == [0, 2.0, 11.0]
    assert result["selected_edge_summary"] == {
        "scientifically_forced_count": 2,
        "non_forced_before_seeded_tier_count": 0,
        "non_forced_before_seeded_tier_fraction": 0.0,
    }
    assert result["selected_root"]["scientifically_forced"] is True
    assert all(record["scientifically_forced"] for record in result["selected_edges"])
    assert {
        record["first_deciding_tier"] for record in result["selected_edges"]
    } <= {"feasibility", "total_edge_distance"}


def test_equal_scientific_optima_leave_each_selected_edge_and_root_non_forced():
    tree = _tree(
        [
            (0, "A", 0, [1]),
            (1, "B", 0, [1]),
            (2, "C", 0, [1]),
        ],
        [(0, 1, 0), (0, 2, 0)],
    )
    distance = _distance(
        ["A", "B", "C"],
        np.zeros((3, 3), dtype=float),
    )

    result = audit.audit_selected_tree(tree, distance, use_time=False)

    assert result["selected_edge_summary"] == {
        "scientifically_forced_count": 0,
        "non_forced_before_seeded_tier_count": 2,
        "non_forced_before_seeded_tier_fraction": 1.0,
    }
    assert result["selected_root"]["non_forced_before_seeded_tier"] is True
    assert result["selected_root"]["first_deciding_tier"] == "seeded_final_tier"
    assert all(
        record["non_forced_before_seeded_tier"]
        for record in result["selected_edges"]
    )
    assert {
        record["first_deciding_tier"] for record in result["selected_edges"]
    } == {"seeded_final_tier"}


def test_global_root_score_can_force_an_edge_even_when_total_distance_is_tied():
    tree = _tree(
        [
            (0, "A", 0, [1]),
            (1, "B", 0, [2]),
            (2, "C", 0, [3]),
        ],
        [(1, 0, 1), (1, 2, 1)],
    )
    distance = _distance(
        ["A", "B", "C"],
        [[0, 1, 2], [1, 0, 1], [2, 1, 0]],
    )

    result = audit.audit_selected_tree(tree, distance, use_time=False)

    assert result["baseline_scientific_objective"]["values"] == [0, 2.0, 2.0]
    assert {
        record["first_deciding_tier"] for record in result["selected_edges"]
    } == {"root_score"}
    assert result["selected_root"]["first_deciding_tier"] == "root_score"


def test_audit_rejects_a_frozen_tree_that_is_not_scientifically_optimal():
    tree = _tree(
        [
            (0, "A", 0, [1]),
            (1, "B", 0, [2]),
            (2, "C", 0, [3]),
        ],
        [(0, 2, 10), (2, 1, 1)],
    )
    distance = _distance(
        ["A", "B", "C"],
        [[0, 1, 10], [1, 0, 1], [10, 1, 0]],
    )

    with pytest.raises(RuntimeError, match="not optimal"):
        audit.audit_selected_tree(tree, distance, use_time=False)


def test_summary_keeps_biological_edges_and_root_choices_separate():
    arm_record = {
        "selected_biological_edge_count": 2,
        "selected_edge_summary": {
            "non_forced_before_seeded_tier_count": 1,
            "non_forced_before_seeded_tier_fraction": 0.5,
        },
        "selected_edges": [
            {"first_deciding_tier": "total_edge_distance"},
            {"first_deciding_tier": "seeded_final_tier"},
        ],
        "selected_root": {"non_forced_before_seeded_tier": True},
    }
    records = [
        {
            "case_id": "synthetic-r001",
            "replicate": 1,
            "regime_id": "synthetic",
            "arms": {arm_id: dict(arm_record) for arm_id in audit.TEMPORAL_ARMS},
        }
    ]

    result = audit.summarize(records)["overall"]

    for arm_id in audit.TEMPORAL_ARMS:
        arm = result["arms"][arm_id]
        assert arm["selected_biological_edge_count"] == 2
        assert arm["scientifically_forced_selected_edge_count"] == 1
        assert arm["non_forced_before_seeded_tier_selected_edge_count"] == 1
        assert arm["non_forced_before_seeded_tier_selected_root"]["count"] == 1
        assert arm["selected_edge_first_deciding_tier_counts"] == {
            "feasibility": 0,
            "plausibility_violation_edge_count": 0,
            "total_edge_distance": 1,
            "root_score": 0,
            "seeded_final_tier": 1,
        }
