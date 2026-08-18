from hashlib import sha256
from pathlib import Path
import sys

import networkx as nx
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithm_evaluation import temporal_arborescence_diagnostic as diagnostic
from algorithm_evaluation.paper_pipeline_contract import status_record, write_json_atomic
from ctbs import DistanceMatrix
from distance_semantics import CNP2CNP_SEMANTICS_VERSION
from evaluation_contract import evaluate_tree_pair


def _tree(edges, *, levels=(0, 1, 2)):
    tree = nx.DiGraph()
    tree.add_node(0, cell_id="A", biopsy_level=levels[0], genome=np.array([1]))
    tree.add_node(1, cell_id="A", biopsy_level=levels[1], genome=np.array([1]))
    tree.add_node(2, cell_id="B", biopsy_level=levels[2], genome=np.array([2]))
    for parent, child in edges:
        distance = 0.0 if tree.nodes[parent]["cell_id"] == tree.nodes[child]["cell_id"] else 1.0
        tree.add_edge(parent, child, weight=distance)
    return tree


def _distance():
    return DistanceMatrix(
        ids=["A", "B"],
        matrix=np.array([[0.0, 1.0], [1.0, 0.0]]),
        provenance={"semantics_version": CNP2CNP_SEMANTICS_VERSION},
    )


def test_frozen_diagnostic_seeds_match_the_declared_sha256_derivation():
    expected = tuple(
        int.from_bytes(
            sha256(
                f"{diagnostic.DIAGNOSTIC_SEED_NAMESPACE}:seed:{index}".encode()
            ).digest()[:4],
            "big",
        )
        for index in range(1, 9)
    )

    assert diagnostic.DIAGNOSTIC_SEEDS == expected


def test_tree_diagnostics_detect_backward_edge_late_root_and_plausibility():
    backward = nx.DiGraph()
    backward.add_node(0, cell_id="A", biopsy_level=1, genome=np.array([1]))
    backward.add_node(1, cell_id="B", biopsy_level=0, genome=np.array([0]))
    backward.add_edge(0, 1, weight=1.0)

    result = diagnostic.tree_scientific_diagnostics(backward, _distance())

    assert result["late_root"] is True
    assert result["backward_edge_count"] == 1
    assert result["constraint_active"] is True
    assert result["plausibility_violation_edge_count"] == 0
    assert result["scientific_objective_tuple"] == [0, 1.0, 1.0]

    implausible = nx.DiGraph()
    implausible.add_node(0, cell_id="B", biopsy_level=0, genome=np.array([0]))
    implausible.add_node(1, cell_id="A", biopsy_level=1, genome=np.array([1]))
    implausible.add_edge(0, 1, weight=1.0)

    result = diagnostic.tree_scientific_diagnostics(implausible, _distance())

    assert result["constraint_active"] is False
    assert result["plausibility_violation_edge_count"] == 1


def test_different_occurrence_trees_can_have_one_unique_label_pair_set():
    first = _tree([(0, 1), (1, 2)])
    second = _tree([(1, 0), (0, 2)])
    labels = ["A", "B"]

    assert diagnostic.occurrence_tree_signature(first) != diagnostic.occurrence_tree_signature(second)
    assert diagnostic.ancestor_pair_set(first, labels) == diagnostic.ancestor_pair_set(second, labels)


def test_frozen_case_audit_separates_constraint_activation_from_ad_visibility():
    temporal = _tree([(0, 1), (1, 2)])
    no_time = _tree([(1, 0), (0, 2)])
    truth = _tree([(0, 1), (1, 2)])
    labels = ["A", "B"]
    case = diagnostic._CaseData(
        metadata={"case_id": "synthetic-r001", "replicate": 1, "regime_id": "synthetic"},
        input_payload={},
        distance=_distance(),
        observed_labels=labels,
        true_pairs=diagnostic.ancestor_pair_set(truth, labels),
        frozen_trees={
            "temporal_minimum": temporal,
            "temporal_minimum_no_time": no_time,
        },
        frozen_evaluations={
            "temporal_minimum": evaluate_tree_pair(truth, temporal, labels),
            "temporal_minimum_no_time": evaluate_tree_pair(truth, no_time, labels),
        },
    )

    result = diagnostic.frozen_case_audit(case)

    assert result["comparison"] == {
        "same_directed_occurrence_tree": False,
        "same_unique_observed_label_ancestor_pair_set": True,
        "same_ad_f1": True,
        "different_tree_same_pair_set": True,
        "different_pair_set_same_ad_f1": False,
        "temporal_minus_no_time_ad_f1": 0.0,
        "no_time_constraint_active": True,
    }


def test_seed_summary_requires_invariant_scientific_objective():
    base = {
        "tree_diagnostics": {
            "scientific_objective_tuple": [0, 2.0, 3.0],
            "root_node_id": 0,
            "root_cell_id": "A",
            "root_biopsy_level": 0,
        },
        "ancestor_pair_set_sha256": "pairs",
        "ad_f1": {"ad_f1": 0.5},
    }
    runs = [
        {"seed": 1, "occurrence_tree_sha256": "tree-a", **base},
        {"seed": 2, "occurrence_tree_sha256": "tree-b", **base},
    ]

    summary = diagnostic._seed_arm_summary(runs)

    assert summary["tree_seed_sensitive"] is True
    assert summary["tree_sensitive_but_pair_set_invariant"] is True
    assert summary["ad_f1_seed_sensitive"] is False

    changed = dict(runs[1])
    changed["tree_diagnostics"] = {
        **runs[1]["tree_diagnostics"],
        "scientific_objective_tuple": [0, 3.0, 3.0],
    }
    with pytest.raises(ValueError, match="higher-priority scientific objective"):
        diagnostic._seed_arm_summary([runs[0], changed])


def test_frozen_distance_loader_checks_condition_submatrix_digest(tmp_path):
    case_root = tmp_path / "cases" / "synthetic-r001"
    matrix = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
    digest = sha256(
        np.asarray(matrix, dtype="<f8", order="C").tobytes(order="C")
    ).hexdigest()
    distance_record = {
        "schema_version": diagnostic.DISTANCE_RECORD_SCHEMA_VERSION,
        "case_id": "synthetic-r001",
        "method": "minimum_bidirectional",
        "status_record": status_record(
            entity_type="distance_provider",
            entity_id="synthetic-r001",
            status="success",
            stage="distance",
            code="distance_complete",
        ),
        "ids": ["A", "B"],
        "matrix": matrix.tolist(),
        "provenance": {"semantics_version": CNP2CNP_SEMANTICS_VERSION},
    }
    write_json_atomic(
        case_root / "distances" / "minimum_bidirectional.json",
        distance_record,
    )
    input_payload = {
        "distance": {
            "ids": ["A", "B"],
            "matrix_shape": [2, 2],
            "matrix_sha256_float64_c_order": digest,
            "semantic_version": CNP2CNP_SEMANTICS_VERSION,
        }
    }

    loaded = diagnostic._load_distance(case_root, input_payload, "synthetic-r001")

    assert loaded.ids == ["A", "B"]
    assert np.array_equal(loaded.matrix, matrix)

    input_payload["distance"]["matrix_sha256_float64_c_order"] = "0" * 64
    with pytest.raises(ValueError, match="distance bytes"):
        diagnostic._load_distance(case_root, input_payload, "synthetic-r001")
