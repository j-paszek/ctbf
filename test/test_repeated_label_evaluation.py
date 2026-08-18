from copy import deepcopy
import sys

import pytest

from algorithm_evaluation.repeated_label_evaluator_suite import (
    DISCRIMINATIVE_SUITE_ID,
    build_discriminative_cases,
    build_suite_report,
)
import repeated_label_evaluation as repeated_label
from repeated_label_evaluation import (
    CANDIDATE_METRIC_IDS,
    EDIST_AUDITED_REVISION,
    EDIST_AUDITED_SOURCE_SHA256,
    EPS_AUDITED_REVISION,
    EPS_AUDITED_SOURCE_SHA256,
    REFERENCE_MAX_NODES_PER_TREE,
    REPEATED_LABEL_RESULT_SCHEMA_VERSION,
    UTED_AUDITED_REVISION,
    UTED_AUDITED_SOURCE_SHA256,
    RepeatedLabelEvaluationError,
    build_external_edist_unit_probe,
    candidate_metric_contract,
    evaluate_external_cuted_tree_pair_result,
    evaluate_external_eps_approx_tree_pair_result,
    evaluate_repeated_label_tree_pair_result,
    inspect_external_edist_source,
    inspect_external_eps_source,
    inspect_external_uted_source,
    probe_external_edist_semantics,
    probe_external_uted_semantics,
    validate_repeated_label_result,
)


def _tree(labels, edges):
    return {
        "directed": True,
        "multigraph": False,
        "nodes": [
            {"id": node, "cell_id": label}
            for node, label in labels.items()
        ],
        "links": [
            {"source": source, "target": target}
            for source, target in edges
        ],
    }


def _cases_by_id():
    return {case["case_id"]: case for case in build_discriminative_cases()}


def test_candidate_registry_keeps_exact_and_constrained_algorithms_distinct():
    assert CANDIDATE_METRIC_IDS == (
        "uted_exact_reference",
        "cuted_edist",
        "eps_exact_reference",
        "eps_approx_external",
    )
    uted = candidate_metric_contract("uted_exact_reference")
    cuted = candidate_metric_contract("cuted_edist")
    eps_exact = candidate_metric_contract("eps_exact_reference")
    eps_approx = candidate_metric_contract("eps_approx_external")

    assert uted["family"] == "uted"
    assert uted["exactness"] == "exact_within_declared_size_limit"
    assert cuted["family"] == "cuted"
    assert "constrained_uted" in cuted["exactness"]
    assert cuted["dependency_identity"]["isolated_semantic_probe"] == "passed"
    assert cuted["mapping_policy"]["root"].startswith("not_forced")
    assert cuted["implementation_status"] == "external_runner_available"
    assert cuted["cost_policy"]["node_insertion"] == 1.0
    assert cuted["normalization"] == (
        "raw_cost_divided_by_sum_of_input_node_counts"
    )
    assert eps_exact["exactness"] == "exact_within_declared_size_limit"
    assert eps_approx["exactness"] == "four_approximation"
    assert eps_approx["implementation_status"] == "external_runner_available"
    assert eps_approx["direction_combination"] == (
        "maximum_of_forward_and_reverse_as_upstream_cli"
    )
    assert uted["tree_policy"]["graph_node_id_policy"] == (
        "graph_local_only_never_cross_tree_identity"
    )
    assert uted["tree_policy"]["copied_or_inferred_node_policy"] == (
        "retain_every_vertex_no_contraction"
    )


def test_contract_descriptors_are_owned_copies():
    descriptor = candidate_metric_contract("uted_exact_reference")
    descriptor["tree_policy"]["tree_scope"] = "tampered"

    assert candidate_metric_contract("uted_exact_reference")["tree_policy"][
        "tree_scope"
    ] == "full_validated_rooted_directed_tree"


def test_external_eps_source_audit_is_read_only_and_detects_pinned_files(tmp_path):
    for relative, expected_hash in EPS_AUDITED_SOURCE_SHA256.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not the audited source")
        assert expected_hash
    (tmp_path / "requirements.txt").write_text("gurobipy\n", encoding="utf-8")

    result = inspect_external_eps_source(tmp_path)

    assert result["status"] == "source_drift"
    assert result["expected_revision"] == EPS_AUDITED_REVISION
    assert result["revision"] is None
    assert result["source_matches_audited_identity"] is False
    assert result["backend_executed"] is False


def test_external_eps_source_audit_reports_missing_checkout(tmp_path):
    result = inspect_external_eps_source(tmp_path / "absent")

    assert result["status"] == "source_unavailable"
    assert result["missing_paths"] == sorted(
        [*EPS_AUDITED_SOURCE_SHA256, "requirements.txt"]
    )


def test_external_uted_source_audit_is_read_only_and_detects_drift(tmp_path):
    for relative, expected_hash in UTED_AUDITED_SOURCE_SHA256.items():
        path = tmp_path / relative
        path.write_bytes(b"not the audited source")
        assert expected_hash

    result = inspect_external_uted_source(tmp_path)

    assert result["status"] == "source_drift"
    assert result["expected_revision"] == UTED_AUDITED_REVISION
    assert result["revision"] is None
    assert result["source_matches_audited_identity"] is False
    assert result["backend_executed"] is False


def test_external_uted_source_audit_reports_missing_checkout(tmp_path):
    result = inspect_external_uted_source(tmp_path / "absent")

    assert result["status"] == "source_unavailable"
    assert result["missing_paths"] == sorted(UTED_AUDITED_SOURCE_SHA256)


def test_external_uted_probe_refuses_unverified_source(tmp_path):
    result = probe_external_uted_semantics(tmp_path / "absent")

    assert result["status"] == "source_not_executable"
    assert result["backend_executed"] is False
    assert result["records"] == []


def test_external_uted_probe_detects_directional_failure(tmp_path, monkeypatch):
    (tmp_path / "uted.py").write_text(
        "def uted_astar(x_nodes, x_adj, y_nodes, y_adj):\n"
        "    distance = 2.0 if len(x_nodes) == 3 and len(y_nodes) == 4 else 1.0\n"
        "    return distance, [-1] * len(x_nodes), 1\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        repeated_label,
        "inspect_external_uted_source",
        lambda repository_root: {
            "status": "source_verified_backend_dependency_present_unexecuted",
        },
    )

    result = repeated_label.probe_external_uted_semantics(tmp_path)

    assert result["status"] == "semantic_probe_failed"
    assert result["backend_executed"] is True
    incident = next(
        record
        for record in result["records"]
        if record["probe_id"] == "insert_parent_of_two_siblings"
    )
    assert incident["forward"]["distance"] == 2.0
    assert incident["reverse"]["distance"] == 1.0
    assert incident["symmetric"] is False
    assert incident["matches_expected"] is False


def test_external_edist_source_audit_is_read_only_and_detects_drift(tmp_path):
    for relative, expected_hash in EDIST_AUDITED_SOURCE_SHA256.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not the audited source")
        assert expected_hash

    result = inspect_external_edist_source(tmp_path)

    assert result["status"] == "source_drift"
    assert result["expected_revision"] == EDIST_AUDITED_REVISION
    assert result["source_matches_audited_identity"] is False
    assert result["backend_executed"] is False


def test_external_edist_source_audit_reports_missing_checkout(tmp_path):
    result = inspect_external_edist_source(tmp_path / "absent")

    assert result["status"] == "source_unavailable"
    assert result["missing_paths"] == sorted(EDIST_AUDITED_SOURCE_SHA256)


def test_external_edist_probe_refuses_unbuilt_source(tmp_path):
    result = probe_external_edist_semantics(tmp_path / "absent")

    assert result["status"] == "source_not_executable"
    assert result["backend_executed"] is False


def test_external_edist_probe_checks_constrained_and_custom_costs(
    tmp_path,
    monkeypatch,
):
    package = tmp_path / "edist"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "uted.py").write_text(
        "__version__ = 'fake-test'\n"
        "def uted(x_nodes, x_adj, y_nodes, y_adj, delta=None):\n"
        "    if delta is not None:\n"
        "        return delta(x_nodes[0], y_nodes[0])\n"
        "    sizes = {len(x_nodes), len(y_nodes)}\n"
        "    if sizes == {4, 5}:\n"
        "        return 3.0\n"
        "    if sizes == {1, 2}:\n"
        "        return 1.0\n"
        "    if sizes == {3, 4}:\n"
        "        return 1.0\n"
        "    return 0.0\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        repeated_label,
        "inspect_external_edist_source",
        lambda repository_root: {
            "status": "source_verified_backend_build_present_unexecuted",
        },
    )
    interpreter_link = tmp_path / "fake_venv" / "bin" / "python"
    interpreter_link.parent.mkdir(parents=True)
    interpreter_link.symlink_to(sys.executable)

    case = _cases_by_id()["node_id_relabeling"]
    additional_probe = build_external_edist_unit_probe(
        "frozen_suite:node_id_relabeling",
        case["true_tree"],
        case["reconstructed_tree"],
    )
    result = repeated_label.probe_external_edist_semantics(
        tmp_path,
        python_executable=interpreter_link,
        additional_probes=[additional_probe],
    )

    assert result["status"] == "semantic_probe_passed"
    assert result["python_executable"] == str(interpreter_link)
    assert result["backend_version"] == "fake-test"
    custom = next(
        record
        for record in result["records"]
        if record["probe_id"] == "custom_quarter_substitution"
    )
    assert custom["forward"]["distance"] == 0.25
    assert custom["matches_expected"] is True
    frozen = next(
        record
        for record in result["records"]
        if record["probe_id"] == "frozen_suite:node_id_relabeling"
    )
    assert frozen["forward"]["distance"] == 0.0
    assert frozen["symmetric"] is True


def test_external_edist_probe_conversion_canonicalizes_node_ids_and_siblings():
    for case_id in ("node_id_relabeling", "sibling_permutation"):
        case = _cases_by_id()[case_id]
        probe = build_external_edist_unit_probe(
            case_id,
            case["true_tree"],
            case["reconstructed_tree"],
        )

        assert probe["left_nodes"] == probe["right_nodes"]
        assert probe["left_adjacency"] == probe["right_adjacency"]
        assert probe["expected_distance"] is None
        assert probe["cost_mode"] == "unit"


def test_external_cuted_runner_returns_strict_unit_cost_result(tmp_path):
    package = tmp_path / "edist"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "uted.py").write_text(
        "__version__ = 'fake-test'\n"
        "def uted(x_nodes, x_adj, y_nodes, y_adj):\n"
        "    size_cost = abs(len(x_nodes) - len(y_nodes))\n"
        "    label_cost = sum(a != b for a, b in zip(x_nodes, y_nodes))\n"
        "    return float(size_cost + label_cost)\n",
        encoding="utf-8",
    )
    interpreter_link = tmp_path / "fake_venv" / "bin" / "python"
    interpreter_link.parent.mkdir(parents=True)
    interpreter_link.symlink_to(sys.executable)
    source_audit = {
        "status": "source_verified_backend_build_present_unexecuted",
        "repository_root": str(tmp_path.resolve()),
        "revision": EDIST_AUDITED_REVISION,
        "source_matches_audited_identity": True,
    }

    result = evaluate_external_cuted_tree_pair_result(
        _tree({"root": "A"}, []),
        _tree({"different-root": "B"}, []),
        repository_root=tmp_path,
        python_executable=interpreter_link,
        source_audit=source_audit,
    )

    assert result["status"] == "success", result
    assert result["metric"] == {
        "raw_value": 1.0,
        "normalization_denominator": 2.0,
        "value": 0.5,
        "degeneracy": "none",
    }
    assert result["external_execution"]["python_executable"] == str(
        interpreter_link
    )
    assert result["external_execution"]["fallback_used"] is False
    validate_repeated_label_result(result)

    missing_provenance = deepcopy(result)
    del missing_provenance["external_execution"]
    with pytest.raises(RepeatedLabelEvaluationError):
        validate_repeated_label_result(missing_provenance)


def test_external_cuted_runner_returns_typed_source_failure(tmp_path):
    interpreter_link = tmp_path / "python"
    interpreter_link.symlink_to(sys.executable)

    result = evaluate_external_cuted_tree_pair_result(
        _tree({"root": "A"}, []),
        _tree({"root": "A"}, []),
        repository_root=tmp_path,
        python_executable=interpreter_link,
        source_audit={
            "status": "source_drift",
            "repository_root": str(tmp_path.resolve()),
            "revision": "wrong",
            "source_matches_audited_identity": False,
        },
    )

    assert result["status"] == "failure"
    assert result["failure"]["code"] == "external_backend_unavailable"
    assert result["failure"]["stage"] == "backend_setup"


def test_external_cuted_runner_returns_typed_timeout(tmp_path, monkeypatch):
    interpreter_link = tmp_path / "python"
    interpreter_link.symlink_to(sys.executable)

    def raise_timeout(command, **kwargs):
        raise repeated_label.subprocess.TimeoutExpired(
            command,
            kwargs["timeout"],
        )

    monkeypatch.setattr(
        repeated_label.subprocess,
        "run",
        raise_timeout,
    )

    result = evaluate_external_cuted_tree_pair_result(
        _tree({"root": "A"}, []),
        _tree({"root": "A"}, []),
        repository_root=tmp_path,
        python_executable=interpreter_link,
        timeout_seconds=0.25,
        source_audit={
            "status": "source_verified_backend_build_present_unexecuted",
            "repository_root": str(tmp_path.resolve()),
            "revision": EDIST_AUDITED_REVISION,
            "source_matches_audited_identity": True,
        },
    )

    assert result["status"] == "failure"
    assert result["failure"]["code"] == "external_backend_timeout"
    assert result["failure"]["stage"] == "backend_execution"


def test_external_eps_runner_uses_bidirectional_maximum(tmp_path, monkeypatch):
    package = tmp_path / "edge_preservation_similarity"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    backend_path = package / "compute_eps.py"
    backend_path.write_text(
        "def compute_similarity(algorithm, first, second, limit, normalize):\n"
        "    score = 1 if first.nodes[0]['lbl'] == 'A' else 0\n"
        "    return score, 0.01, False\n",
        encoding="utf-8",
    )
    (tmp_path / "gurobipy.py").write_text(
        "class gurobi:\n"
        "    @staticmethod\n"
        "    def version():\n"
        "        return (13, 0, 2)\n",
        encoding="utf-8",
    )
    monkeypatch.setitem(
        repeated_label.EPS_AUDITED_SOURCE_SHA256,
        "edge_preservation_similarity/compute_eps.py",
        repeated_label._sha256_file(backend_path),
    )
    source_audit = {
        "status": "source_verified_external_dependency_unchecked",
        "repository_root": str(tmp_path.resolve()),
        "revision": EPS_AUDITED_REVISION,
        "source_matches_audited_identity": True,
    }

    result = evaluate_external_eps_approx_tree_pair_result(
        _tree({"root": "A", "leaf": "B"}, [("root", "leaf")]),
        _tree(
            {"other-root": "X", "other-leaf": "B"},
            [("other-root", "other-leaf")],
        ),
        repository_root=tmp_path,
        python_executable=sys.executable,
        source_audit=source_audit,
    )

    assert result["status"] == "success", result
    assert result["metric"] == {
        "raw_value": 1.0,
        "normalization_denominator": 1.0,
        "value": 1.0,
        "degeneracy": "none",
    }
    assert result["external_execution"]["directional_raw_values"] == {
        "forward": 1.0,
        "reverse": 0.0,
    }
    assert result["external_execution"]["fallback_used"] is False
    validate_repeated_label_result(result)


def test_external_eps_runner_preserves_zero_edge_degeneracy(
    tmp_path,
    monkeypatch,
):
    package = tmp_path / "edge_preservation_similarity"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    backend_path = package / "compute_eps.py"
    backend_path.write_text(
        "def compute_similarity(algorithm, first, second, limit, normalize):\n"
        "    return 0, 0.0, False\n",
        encoding="utf-8",
    )
    (tmp_path / "gurobipy.py").write_text(
        "class gurobi:\n"
        "    @staticmethod\n"
        "    def version():\n"
        "        return (13, 0, 2)\n",
        encoding="utf-8",
    )
    monkeypatch.setitem(
        repeated_label.EPS_AUDITED_SOURCE_SHA256,
        "edge_preservation_similarity/compute_eps.py",
        repeated_label._sha256_file(backend_path),
    )
    result = evaluate_external_eps_approx_tree_pair_result(
        _tree({"root": "X"}, []),
        _tree({"other-root": "Y"}, []),
        repository_root=tmp_path,
        python_executable=sys.executable,
        source_audit={
            "status": "source_verified_external_dependency_unchecked",
            "repository_root": str(tmp_path.resolve()),
            "revision": EPS_AUDITED_REVISION,
            "source_matches_audited_identity": True,
        },
    )

    assert result["status"] == "success", result
    assert result["metric"] == {
        "raw_value": 0.0,
        "normalization_denominator": 0.0,
        "value": None,
        "degeneracy": "zero_edge_denominator",
    }
    validate_repeated_label_result(result)


@pytest.mark.parametrize("metric_id", CANDIDATE_METRIC_IDS)
def test_every_candidate_returns_a_strict_status_record(metric_id):
    tree = _tree({"root": "A", "leaf": "B"}, [("root", "leaf")])

    result = evaluate_repeated_label_tree_pair_result(tree, tree, metric_id)

    assert result["schema_version"] == REPEATED_LABEL_RESULT_SCHEMA_VERSION
    validate_repeated_label_result(result)
    if metric_id in {"uted_exact_reference", "eps_exact_reference"}:
        assert result["status"] == "success"
    else:
        assert result["status"] == "failure"
        assert result["failure"]["code"] == "external_execution_required"


def test_invalid_input_is_a_typed_failure_before_backend_availability():
    invalid = _tree({"a": "A", "b": "B"}, [])
    valid = _tree({"root": "A"}, [])

    result = evaluate_repeated_label_tree_pair_result(
        invalid,
        valid,
        "cuted_edist",
    )

    assert result["status"] == "failure"
    assert result["failure"]["code"] == "root_count"
    assert result["failure"]["stage"] == "true_tree"
    validate_repeated_label_result(result)


def test_node_ids_and_sibling_order_do_not_change_reference_results():
    cases = _cases_by_id()
    for case_id in ("node_id_relabeling", "sibling_permutation"):
        case = cases[case_id]
        for metric_id in ("uted_exact_reference", "eps_exact_reference"):
            identity = evaluate_repeated_label_tree_pair_result(
                case["true_tree"],
                case["true_tree"],
                metric_id,
            )
            relabeled = evaluate_repeated_label_tree_pair_result(
                case["true_tree"],
                case["reconstructed_tree"],
                metric_id,
            )
            assert relabeled == identity


def test_exact_uted_reference_matches_published_unrestricted_mapping_example():
    left = _tree(
        {"a": "a", "b": "b", "c": "c", "d": "d", "e": "e"},
        [("a", "b"), ("b", "c"), ("b", "d"), ("a", "e")],
    )
    right = _tree(
        {"a": "a", "e": "e", "d": "d", "c": "c"},
        [("a", "e"), ("a", "d"), ("a", "c")],
    )

    result = evaluate_repeated_label_tree_pair_result(
        left,
        right,
        "uted_exact_reference",
    )

    assert result["status"] == "success"
    assert result["metric"] == {
        "raw_value": 1.0,
        "normalization_denominator": 9.0,
        "value": pytest.approx(1 / 9),
        "degeneracy": "none",
    }


@pytest.mark.parametrize(
    ("case_id", "uted_raw", "eps_raw", "eps_value"),
    (
        ("node_id_relabeling", 0.0, 2.0, 1.0),
        ("sibling_permutation", 0.0, 3.0, 1.0),
        ("displaced_repeated_state", 2.0, 1.0, 1 / 3),
        ("copied_state_internal_node", 1.0, 1.0, 1 / 2),
        ("unlabeled_refinement", 1.0, 0.0, 0.0),
        ("missing_observation", 1.0, 0.0, 0.0),
        ("wrong_ancestry", 2.0, 0.0, 0.0),
        ("one_bin_cnp_perturbation", 1.0, 0.0, 0.0),
        ("same_cnp_path", 1.0, 1.0, 1 / 2),
        ("copied_state_incident_branches", 1.0, 2.0, 2 / 3),
    ),
)
def test_frozen_suite_has_hand_checked_reference_values(
    case_id,
    uted_raw,
    eps_raw,
    eps_value,
):
    case = _cases_by_id()[case_id]
    uted = evaluate_repeated_label_tree_pair_result(
        case["true_tree"],
        case["reconstructed_tree"],
        "uted_exact_reference",
    )
    eps = evaluate_repeated_label_tree_pair_result(
        case["true_tree"],
        case["reconstructed_tree"],
        "eps_exact_reference",
    )

    assert uted["metric"]["raw_value"] == uted_raw
    assert eps["metric"]["raw_value"] == eps_raw
    assert eps["metric"]["value"] == pytest.approx(eps_value)


def test_reference_metrics_are_symmetric_on_every_frozen_case():
    for case in build_discriminative_cases():
        for metric_id in ("uted_exact_reference", "eps_exact_reference"):
            forward = evaluate_repeated_label_tree_pair_result(
                case["true_tree"],
                case["reconstructed_tree"],
                metric_id,
            )
            reverse = evaluate_repeated_label_tree_pair_result(
                case["reconstructed_tree"],
                case["true_tree"],
                metric_id,
            )
            assert forward["metric"]["raw_value"] == reverse["metric"]["raw_value"]
            assert forward["metric"]["value"] == reverse["metric"]["value"]


def test_zero_edge_eps_is_null_and_explicit_instead_of_inventing_similarity():
    left = _tree({"left": "A"}, [])
    right = _tree({"right": "A"}, [])

    result = evaluate_repeated_label_tree_pair_result(
        left,
        right,
        "eps_exact_reference",
    )

    assert result["metric"] == {
        "raw_value": 0.0,
        "normalization_denominator": 0.0,
        "value": None,
        "degeneracy": "zero_edge_denominator",
    }


def test_reference_limit_is_typed_and_does_not_fall_back():
    labels = {node: f"L{node}" for node in range(REFERENCE_MAX_NODES_PER_TREE + 1)}
    edges = [(node, node + 1) for node in range(REFERENCE_MAX_NODES_PER_TREE)]
    oversized = _tree(labels, edges)

    result = evaluate_repeated_label_tree_pair_result(
        oversized,
        oversized,
        "uted_exact_reference",
    )

    assert result["status"] == "failure"
    assert result["failure"]["code"] == "reference_size_limit_exceeded"


def test_result_validator_rejects_metric_contract_drift():
    tree = _tree({"root": "A", "leaf": "B"}, [("root", "leaf")])
    result = evaluate_repeated_label_tree_pair_result(
        tree,
        tree,
        "uted_exact_reference",
    )
    tampered = deepcopy(result)
    tampered["metric_contract"]["exactness"] = "approximate"

    with pytest.raises(RepeatedLabelEvaluationError, match="contract"):
        validate_repeated_label_result(tampered)


def test_complete_tiny_suite_reports_only_the_declared_backend_gaps():
    report = build_suite_report()

    assert report["suite_id"] == DISCRIMINATIVE_SUITE_ID
    assert report["case_count"] == 10
    assert [record["case_id"] for record in report["records"]] == [
        "node_id_relabeling",
        "sibling_permutation",
        "displaced_repeated_state",
        "copied_state_internal_node",
        "unlabeled_refinement",
        "missing_observation",
        "wrong_ancestry",
        "one_bin_cnp_perturbation",
        "same_cnp_path",
        "copied_state_incident_branches",
    ]
    assert report["summary"]["native_status_counts"] == {
        "success": 9,
        "failure": 1,
    }
    assert report["summary"]["candidate_status_counts"] == {
        "uted_exact_reference": {"success": 10, "failure": 0},
        "cuted_edist": {"success": 0, "failure": 10},
        "eps_exact_reference": {"success": 10, "failure": 0},
        "eps_approx_external": {"success": 0, "failure": 10},
    }
    perturbation = next(
        record
        for record in report["records"]
        if record["case_id"] == "one_bin_cnp_perturbation"
    )
    assert perturbation["native_ad_f1_grf"]["failure"]["code"] == (
        "reconstructed_labels_outside_observation_set"
    )
