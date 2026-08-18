import json

import networkx as nx
import pytest

from algorithm_evaluation import repeated_label_runtime_preflight as preflight
from algorithm_evaluation.paper_pipeline_contract import (
    EXPECTED_INVENTORY_SCHEMA_VERSION,
    status_record,
    write_checksum_file,
    write_json_atomic,
)
from algorithm_evaluation.paper_pipeline_runner import serialize_tree
from evaluation_contract import evaluate_tree_pair_result
from repeated_label_evaluation import (
    EPS_AUDITED_REVISION,
    EPS_AUDITED_SOURCE_SHA256,
    REPEATED_LABEL_RESULT_SCHEMA_VERSION,
    candidate_metric_contract,
    evaluate_repeated_label_tree_pair_result,
)


def _tree():
    tree = nx.DiGraph()
    tree.add_node("root-local", cell_id="A")
    tree.add_node("leaf-local", cell_id="B")
    tree.add_edge("root-local", "leaf-local")
    return tree


def _closed_fixture(tmp_path):
    root = tmp_path / "closed"
    case_id = "clean-balanced-r001"
    condition_id = "f0p50_L3"
    arm_id = "arm_a"
    arm_base = root / "cases" / case_id / "conditions" / condition_id / "arms" / arm_id
    truth_path = root / "cases" / case_id / "truth.json"
    reconstruction_path = arm_base / "reconstruction.json"
    evaluation_path = arm_base / "evaluation.json"
    tree = _tree()

    write_json_atomic(
        root / "run_status.json",
        status_record(
            entity_type="experiment",
            entity_id="closed-test",
            status="success",
            stage="raw_close",
            code="raw_closed",
        ),
    )
    write_json_atomic(
        truth_path,
        {
            "schema_version": "test-truth-v1",
            "status_record": status_record(
                entity_type="truth",
                entity_id=case_id,
                status="success",
                stage="simulation",
                code="truth_complete",
            ),
            "tree": serialize_tree(tree),
        },
    )
    write_json_atomic(
        reconstruction_path,
        {
            "schema_version": "test-reconstruction-v1",
            "status": "success",
            "tree": serialize_tree(tree),
        },
    )
    write_json_atomic(
        evaluation_path,
        evaluate_tree_pair_result(tree, tree, ["A", "B"]),
    )

    raw_files = [
        "expected_inventory.json",
        "run_status.json",
        truth_path.relative_to(root).as_posix(),
        reconstruction_path.relative_to(root).as_posix(),
        evaluation_path.relative_to(root).as_posix(),
    ]
    write_json_atomic(
        root / "expected_inventory.json",
        {
            "schema_version": EXPECTED_INVENTORY_SCHEMA_VERSION,
            "experiment_id": "closed-test",
            "cases": [
                {
                    "case_id": case_id,
                    "regime_id": "clean_balanced",
                    "replicate": 1,
                    "condition_ids": [condition_id],
                }
            ],
            "raw_files": sorted(raw_files),
            "required_root_files": [
                "expected_inventory.json",
                "run_status.json",
                "raw_checksums.sha256",
                "complete_checksums.sha256",
            ],
        },
    )
    write_checksum_file(root, "raw_checksums.sha256", include_analysis=False)
    write_json_atomic(root / "analysis" / "summary.json", {"closed": True})
    write_checksum_file(root, "complete_checksums.sha256", include_analysis=True)
    return root


def _inventory_for_sampling():
    conditions = [f"condition-{index:02d}" for index in range(16)]
    cases = []
    for regime in ("regime-c", "regime-a", "regime-b"):
        for replicate in range(1, 13):
            cases.append(
                {
                    "case_id": f"{regime}-r{replicate:03d}",
                    "regime_id": regime,
                    "replicate": replicate,
                    "condition_ids": list(reversed(conditions)),
                }
            )
    return {"cases": list(reversed(cases))}, conditions


def test_stratified_plan_is_order_invariant_and_covers_registered_axes():
    inventory, conditions = _inventory_for_sampling()
    arm_specs = tuple((f"arm-{index}", f"algorithm-{index}") for index in range(6))

    plan = preflight.build_stratified_sample_plan(
        inventory,
        arm_specs=arm_specs,
    )
    reordered = preflight.build_stratified_sample_plan(
        {"cases": list(reversed(inventory["cases"]))},
        arm_specs=arm_specs,
    )

    assert plan == reordered
    assert len(plan) == 18
    assert len({(row["regime_id"], row["arm_id"]) for row in plan}) == 18
    assert {row["condition_id"] for row in plan} == set(conditions)


def test_completed_root_uses_one_complete_pass_and_checks_raw_subset(tmp_path):
    root = _closed_fixture(tmp_path)

    audited_root, inventory, raw_paths, closure = preflight.audit_completed_output_root(root)

    assert audited_root == root.resolve()
    assert inventory["experiment_id"] == "closed-test"
    assert "expected_inventory.json" in raw_paths
    assert closure["complete_closure_verified"] is True
    assert closure["raw_is_complete_non_analysis_subset"] is True

    entries = (root / "raw_checksums.sha256").read_text(encoding="utf-8").splitlines()
    entries[0] = "0" * 64 + entries[0][64:]
    (root / "raw_checksums.sha256").write_text(
        "\n".join(entries) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="non-analysis subset"):
        preflight.audit_completed_output_root(root)


def test_plan_only_run_reads_one_selected_pair_and_keeps_metrics_external(tmp_path):
    root = _closed_fixture(tmp_path)

    report = preflight.run_bounded_preflight(
        root,
        arm_specs=(("arm_a", "algorithm_a"),),
    )

    assert report["requested_metrics"] == []
    assert report["summary"]["sample_count"] == 1
    assert report["summary"]["native_status_counts"] == {"success": 1}
    assert report["records"][0]["candidate_metrics"] == {}
    assert report["records"][0]["tree_sizes"]["total_nodes"] == 4


def test_external_capacity_failure_is_compact_and_not_retried(tmp_path, monkeypatch):
    root = _closed_fixture(tmp_path)
    calls = []

    monkeypatch.setattr(
        preflight,
        "inspect_external_edist_source",
        lambda _root: {"status": "source-test", "revision": "revision-test"},
    )

    def fail_with_license_message(true_tree, reconstructed_tree, **options):
        calls.append((true_tree.number_of_nodes(), reconstructed_tree.number_of_nodes(), options))
        return {
            "schema_version": REPEATED_LABEL_RESULT_SCHEMA_VERSION,
            "status": "failure",
            "metric_id": "cuted_edist",
            "metric_contract": candidate_metric_contract("cuted_edist"),
            "inputs": {},
            "failure": {
                "code": "external_backend_failed",
                "stage": "backend_execution",
                "message": "External backend failed.",
                "details": {
                    "stderr_tail": "Model too large for size-limited license"
                },
            },
        }

    monkeypatch.setattr(
        preflight,
        "evaluate_external_cuted_tree_pair_result",
        fail_with_license_message,
    )
    report = preflight.run_bounded_preflight(
        root,
        run_external_cuted=True,
        edist_python=tmp_path / "external-python",
        arm_specs=(("arm_a", "algorithm_a"),),
    )

    summary = report["summary"]["candidate_metrics"]["cuted_edist"]
    assert len(calls) == 1
    assert summary["status_counts"] == {"failure": 1}
    assert summary["failure_code_counts"] == {"external_backend_failed": 1}
    assert summary["license_or_capacity_text_detected"] is True
    assert summary["first_failure_by_code"]["external_backend_failed"][
        "sample_id"
    ] == "runtime-001"
    diagnostic = report["records"][0]["candidate_metrics"]["cuted_edist"]
    assert len(json.dumps(diagnostic)) < 1000


def test_external_eps_success_is_summarized_without_full_provenance(tmp_path, monkeypatch):
    root = _closed_fixture(tmp_path)

    monkeypatch.setattr(
        preflight,
        "inspect_external_eps_source",
        lambda _root: {"status": "source-test", "revision": EPS_AUDITED_REVISION},
    )

    def successful_eps(true_tree, reconstructed_tree, **_options):
        result = evaluate_repeated_label_tree_pair_result(
            true_tree,
            reconstructed_tree,
            "eps_exact_reference",
        )
        result["metric_id"] = "eps_approx_external"
        result["metric_contract"] = candidate_metric_contract("eps_approx_external")
        result["external_execution"] = {
            "backend": "edge_preservation_similarity.compute_eps.compute_similarity",
            "algorithm": "EDGE-PRESERVATION-SIM-APPROX",
            "repository_root": "/external/eps",
            "python_executable": "/external/eps/python",
            "source_revision": EPS_AUDITED_REVISION,
            "backend_module_path": "/external/eps/compute_eps.py",
            "backend_module_sha256": EPS_AUDITED_SOURCE_SHA256[
                "edge_preservation_similarity/compute_eps.py"
            ],
            "gurobi_version": [13, 0, 2],
            "networkx_version": "3.4.2",
            "numpy_version": "2.2.6",
            "timeout_seconds": 30.0,
            "direction_combination": "maximum_of_forward_and_reverse",
            "directional_raw_values": {"forward": 1.0, "reverse": 1.0},
            "directional_duration_seconds": {"forward": 0.01, "reverse": 0.02},
            "fallback_used": False,
        }
        return result

    monkeypatch.setattr(
        preflight,
        "evaluate_external_eps_approx_tree_pair_result",
        successful_eps,
    )
    report = preflight.run_bounded_preflight(
        root,
        run_external_eps_approx=True,
        eps_python=tmp_path / "external-python",
        arm_specs=(("arm_a", "algorithm_a"),),
    )

    metric = report["records"][0]["candidate_metrics"]["eps_approx_external"]
    summary = report["summary"]["candidate_metrics"]["eps_approx_external"]
    assert metric["value"] == 1.0
    assert metric["directional_disagreement"] is False
    assert "external_execution" not in metric
    assert summary["status_counts"] == {"success": 1}
    assert summary["largest_successful_total_nodes"] == 4
    assert report["summary"]["descriptive_comparisons"][
        "eps_approx_external"
    ]["paired_success_count"] == 1


def test_report_destination_must_be_new_and_outside_closed_root(tmp_path):
    root = _closed_fixture(tmp_path).resolve()

    with pytest.raises(ValueError, match="outside"):
        preflight._output_destination(root, root / "report.json")

    existing = tmp_path / "existing.json"
    existing.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="overwrite"):
        preflight._output_destination(root, existing)

    assert preflight._output_destination(root, tmp_path / "new.json") == (
        tmp_path / "new.json"
    ).resolve()
