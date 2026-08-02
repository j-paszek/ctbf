from copy import deepcopy
import json

import networkx as nx
import pytest

from algorithm_evaluation.metric_integrity import audit_record, run_integrity
from evaluation_contract import (
    AD_F1_SEMANTICS_VERSION,
    EVALUATION_RESULT_SCHEMA_VERSION,
    EvaluationContractError,
    GRF_SEMANTICS_VERSION,
    evaluate_tree_pair,
    evaluate_tree_pair_result,
    validate_evaluation_result,
)
from evaluator import grf_tree
from evaluator_full import (
    RestrictedAdf1Cache,
    adf1_restricted_metrics_from_contexts,
    normalize_cell_labels,
)


def _tree(labels, edges):
    tree = nx.DiGraph()
    for node, label in labels.items():
        tree.add_node(node, cell_id=label)
    tree.add_edges_from(edges)
    return tree


def _frozen_tree(labels, edges):
    return {
        "directed": True,
        "nodes": [
            {"node_id": node, "attributes": {"cell_id": label}}
            for node, label in labels.items()
        ],
        "edges": [
            {"source": source, "target": target, "attributes": {}}
            for source, target in edges
        ],
    }


def _bounded_record():
    frozen = _frozen_tree({"root": 1, "leaf": 2}, [("root", "leaf")])
    return {
        "status": "complete",
        "replay_input": {
            "biopsies": [
                {
                    "level": 0,
                    "observations": [{"cell_id": 1}, {"cell_id": 2}],
                }
            ],
            "truth_tree": frozen,
        },
        "arms": {
            "example": {
                "status": "success",
                "tree": frozen,
                "metrics": {
                    "ad_f1": 1.0,
                    "grf": 1.0,
                    "ad_f1_counts": {
                        "TP": 1,
                        "FP": 0,
                        "FN": 0,
                        "num_unique_pairs_true": 1,
                        "num_unique_pairs_rec": 1,
                    },
                },
            }
        },
    }


def test_restriction_labels_use_the_same_canonicalization_as_tree_labels():
    truth = _tree({"root": 1, "child": " 2 "}, [("root", "child")])
    reconstructed = _tree({"root": " 1 ", "child": 2}, [("root", "child")])
    cache = RestrictedAdf1Cache()

    integer_result = adf1_restricted_metrics_from_contexts(
        truth,
        reconstructed,
        restrict_labels=[1, 2, None, " "],
        cache=cache,
    )
    string_result = adf1_restricted_metrics_from_contexts(
        truth,
        reconstructed,
        restrict_labels={"1", "2"},
        cache=cache,
    )

    assert normalize_cell_labels(" 12 ") == frozenset({"12"})
    assert integer_result == string_result
    assert integer_result["F1"] == 1.0
    assert len(cache.true_pair_ids_by_restricted_labels) == 1


def test_grf_canonicalizes_raw_graph_labels_as_context_evaluation_does():
    integer_tree = _tree({0: 1, 1: " 2 "}, [(0, 1)])
    string_tree = _tree({"a": "1", "b": 2}, [("a", "b")])

    assert grf_tree(integer_tree, 0, string_tree, "a") == 1.0


def test_success_result_freezes_repeated_label_and_metric_directions():
    truth = _tree(
        {"root": 1, "middle": 2, "leaf": 1},
        [("root", "middle"), ("middle", "leaf")],
    )
    reconstructed = _tree(
        {"root": " 1 ", "middle": "2", "leaf": "1"},
        [("root", "middle"), ("middle", "leaf")],
    )

    result = evaluate_tree_pair(truth, reconstructed, [2, " 1 ", 2])

    assert result["schema_version"] == EVALUATION_RESULT_SCHEMA_VERSION
    assert result["status"] == "success"
    assert result["observation_labels"] == ["1", "2"]
    assert result["metric_contract"]["ad_f1"]["semantics"] == AD_F1_SEMANTICS_VERSION
    assert result["metric_contract"]["grf"]["semantics"] == GRF_SEMANTICS_VERSION
    assert result["metric_contract"]["ad_f1"]["direction"] == "higher_is_better"
    assert result["metric_contract"]["ext_grf"]["direction"] == "lower_is_better"
    assert result["metrics"]["ad_counts"] == {
        "tp": 3,
        "fp": 0,
        "fn": 0,
        "true_unique_pair_count": 3,
        "reconstructed_unique_pair_count": 3,
    }
    assert result["metrics"]["ad_f1"] == 1.0
    assert result["metrics"]["grf"] == 1.0
    assert result["metrics"]["ext_grf"] == 0.0
    assert result["inputs"]["true_tree"]["repeated_label_occurrence_count"] == 1
    validate_evaluation_result(result)


def test_empty_pair_convention_is_zero_and_explicit_not_perfect():
    truth = _tree({"only": 7}, [])
    reconstructed = _tree({"only": "7"}, [])

    result = evaluate_tree_pair(truth, reconstructed, [7])

    assert result["metrics"]["ad_f1"] == 0.0
    assert result["metrics"]["ad_f1_degenerate"] is True
    assert result["metrics"]["ad_f1_degeneracy"] == "empty_truth_and_reconstruction"
    assert result["metrics"]["grf"] == 1.0


def test_missing_reconstructed_observation_is_allowed_but_never_silent():
    truth = _tree(
        {"root": "A", "middle": "B", "leaf": "C"},
        [("root", "middle"), ("middle", "leaf")],
    )
    reconstructed = _tree({"root": "A", "leaf": "C"}, [("root", "leaf")])

    result = evaluate_tree_pair(truth, reconstructed, ["A", "B", "C"])

    assert result["inputs"]["observation_label_coverage"] == {
        "required_unique_label_count": 3,
        "reconstructed_unique_label_count": 2,
        "fraction": pytest.approx(2 / 3),
        "missing_labels": ["B"],
    }
    assert result["metrics"]["ad_counts"] == {
        "tp": 1,
        "fp": 0,
        "fn": 2,
        "true_unique_pair_count": 3,
        "reconstructed_unique_pair_count": 1,
    }


def test_compatibility_adf1_keeps_truth_only_restriction():
    truth = _tree({"root": "A", "leaf": "B"}, [("root", "leaf")])
    reconstructed = _tree({"root": "A", "leaf": "X"}, [("root", "leaf")])

    metrics = adf1_restricted_metrics_from_contexts(
        truth,
        reconstructed,
        restrict_labels={"A", "B"},
    )

    assert metrics["TP"] == 0
    assert metrics["FP"] == 1
    assert metrics["FN"] == 1


def test_paper_result_rejects_outside_reconstructed_labels():
    truth = _tree({"root": "A", "leaf": "B"}, [("root", "leaf")])
    reconstructed = _tree({"root": "A", "leaf": "X"}, [("root", "leaf")])

    with pytest.raises(EvaluationContractError) as error:
        evaluate_tree_pair(truth, reconstructed, ["A", "B"])

    assert error.value.code == "reconstructed_labels_outside_observation_set"
    failure = evaluate_tree_pair_result(truth, reconstructed, ["A", "B"])
    assert failure["status"] == "failure"
    assert failure["failure"]["code"] == error.value.code
    validate_evaluation_result(failure)


def test_observation_labels_must_be_declared_and_present_in_truth():
    tree = _tree({"only": "A"}, [])

    missing = evaluate_tree_pair_result(tree, tree, None)
    absent = evaluate_tree_pair_result(tree, tree, ["A", "B"])

    assert missing["failure"]["code"] == "observation_labels_required"
    assert absent["failure"]["code"] == "observation_labels_missing_from_truth"


@pytest.mark.parametrize(
    ("tree", "expected_code"),
    [
        (_tree({"left": "A", "right": "B"}, []), "root_count"),
        (
            _tree(
                {"root": "A", "left": "B", "right": "C"},
                [("root", "right"), ("root", "left"), ("left", "right")],
            ),
            "multiple_parents",
        ),
        (nx.Graph([(0, 1)]), "tree_not_directed"),
    ],
)
def test_invalid_tree_policy_returns_typed_failure(tree, expected_code):
    for node in tree.nodes:
        tree.nodes[node].setdefault("cell_id", str(node))
    valid = _tree({"root": "A", "leaf": "B"}, [("root", "leaf")])

    result = evaluate_tree_pair_result(valid, tree, ["A", "B"])

    assert result["status"] == "failure"
    assert result["failure"]["code"] == expected_code


def test_node_link_duplicate_ids_fail_before_metric_computation():
    valid = _tree({"root": "A"}, [])
    duplicate = {
        "directed": True,
        "nodes": [{"id": 1, "cell_id": "A"}, {"id": 1, "cell_id": "A"}],
        "links": [],
    }

    result = evaluate_tree_pair_result(valid, duplicate, ["A"])

    assert result["failure"]["code"] == "duplicate_node_id"


def test_schema_validation_detects_metric_tampering():
    tree = _tree({"root": "A", "leaf": "B"}, [("root", "leaf")])
    result = evaluate_tree_pair(tree, tree, ["A", "B"])
    altered = deepcopy(result)
    altered["metrics"]["grf"] = 0.5

    with pytest.raises(EvaluationContractError) as error:
        validate_evaluation_result(altered)

    assert error.value.code == "invalid_evaluation_result"


def test_bounded_record_replay_matches_stored_metrics_and_detects_drift():
    record = _bounded_record()

    checked, issues = audit_record(record, "records/example.json")
    altered = deepcopy(record)
    altered["arms"]["example"]["metrics"]["grf"] = 0.75
    _, altered_issues = audit_record(altered, "records/example.json")

    assert checked == 1
    assert issues == []
    assert [issue["code"] for issue in altered_issues] == ["stored_metric_drift"]


def test_bounded_integrity_runner_is_read_only_and_bounded(tmp_path):
    record_path = tmp_path / "records" / "case" / "replicate.json"
    record_path.parent.mkdir(parents=True)
    record_path.write_text(json.dumps(_bounded_record()), encoding="utf-8")

    before = record_path.read_bytes()
    report = run_integrity(tmp_path, max_records=1, verify_checksums=False)

    assert report["status"] == "pass"
    assert report["checked_record_count"] == 1
    assert report["checked_arm_count"] == 1
    assert record_path.read_bytes() == before
