from pathlib import Path

import networkx as nx
import pytest

from algorithm_evaluation.paper_pipeline_contract import file_sha256, read_json, write_json_atomic
from algorithm_evaluation.paper_pipeline_runner import serialize_tree
from algorithm_evaluation.v5_confirmation_common import PAPER_ARM_IDS, write_checksum_closure
from algorithm_evaluation import v5_trimmed_truth_run as runner
from algorithm_evaluation.v5_trimmed_truth_report import build_report, case_scores
from algorithm_evaluation.v5_trimmed_truth_sources import collect_sources, safe_child
from evaluation_contract import evaluate_tree_pair


def fixture_sources(tmp_path, *, failed_arm=None):
    bank, internal, medicc, nested, original = [tmp_path / name for name in
                                               ("bank", "internal", "medicc", "nested", "nested_original")]
    case_id = "conf-b001-H14_rule_y"
    case = {"case_id": case_id, "block_index": 0, "condition_id": "H14_rule_y",
            "condition_ordinal": 0, "condition_family": "standard", "height": 14,
            "generations": [1, 2], "status": "available", "input_path": "input.json", "truth_path": "truth.json"}
    truth = nx.DiGraph()
    truth.add_nodes_from([(0, {"cell_id": "hidden", "generation": 0}),
                          (1, {"cell_id": "a", "generation": 1}),
                          (2, {"cell_id": "b", "generation": 2}),
                          (3, {"cell_id": "c", "generation": 1})])
    truth.add_edges_from([(0, 1), (1, 2), (0, 3)])
    rec = nx.DiGraph()
    rec.add_nodes_from([(10, {"cell_id": "a"}), (20, {"cell_id": "b"})])
    rec.add_edge(10, 20)
    native = evaluate_tree_pair(truth, rec, ["a", "b"])
    payload = {"schema_version": "ctbf-v5-reconstruction-input-v1", "case_id": case_id,
               "levels": [{"biopsy_level": i, "generation": i + 1,
                           "states": [{"state_label": label, "cnp": [2 + i]}]}
                          for i, label in enumerate(("a", "b"))]}
    write_json_atomic(bank / "input.json", payload)
    write_json_atomic(bank / "truth.json", {"schema_version": "ctbf-v5-confirmatory-truth-v1", "block_index": 0, "tree": serialize_tree(truth)})
    write_json_atomic(bank / "bank_manifest.json", {
        "schema_version": "ctbf-v5-confirmatory-bank-v1", "status": "complete",
        "design_manifest_sha256": "fixture-design", "scientific_role": "test_fixture",
        "declared_condition_count": 1, "condition_inventory": [case], "cases": [case]})
    write_checksum_closure(bank, "bank_checksums.sha256")
    for root, arms, schema, record_schema in (
        (internal, PAPER_ARM_IDS, "ctbf-v5-confirmatory-run-v1", "ctbf-v5-confirmatory-arm-record-v1"),
        (medicc, ("medicc2_total_cn_topology_partial",), "ctbf-v5-confirmatory-medicc2-run-v1", "ctbf-v5-confirmatory-medicc2-record-v1"),
    ):
        inventory = []
        for arm in arms:
            status = "failure" if arm == failed_arm else "success"
            record = {**case, "schema_version": record_schema, "arm_id": arm, "status": status,
                      "tree": serialize_tree(rec), "native_evaluation": native,
                      "metrics": {"grf": native["metrics"]["grf"]}, "failure": {"message": "fixture failure"} if status == "failure" else None}
            path = root / f"{arm}.json"
            write_json_atomic(path, record)
            inventory.append({"case_id": case_id, "arm_id": arm, "condition_id": case["condition_id"],
                              "path": path.name, "sha256": file_sha256(path), "status": status})
        write_json_atomic(root / "run_manifest.json", {"schema_version": schema, "status": "complete",
            "bank_root": str(bank), "design_manifest_sha256": "fixture-design",
            "expected_record_count": len(inventory), "record_inventory": inventory})
        write_checksum_closure(root, "run_checksums.sha256")
    rows, trusted, artifacts = [], {str(bank / "truth.json"): file_sha256(bank / "truth.json")}, {}
    for replicate, root in enumerate((original, nested), 1):
        attempt = root / f"summaries/seed{replicate}"
        row = {"case_id": case_id, "block_index": 0, "height": 14, "method": "NestedBD-Long",
               "replicate": replicate, "seed": 100 + replicate, "status": "success", "failure": None,
               "grf": native["metrics"]["grf"], "attempt": str(attempt)}
        for filename, content in (("tree.json", serialize_tree(rec)), ("evaluation.json", native), ("result.json", row)):
            write_json_atomic(attempt / filename, content)
        write_json_atomic(attempt / "_SEALED.json", {path.name: file_sha256(path) for path in attempt.iterdir()})
        target = trusted if replicate == 1 else artifacts
        target[str(attempt / "_SEALED.json")] = file_sha256(attempt / "_SEALED.json")
        rows.append(row)
    report = {"schema_version": "ctbf-v5-nestedbd-long-benchmark-report-recovery-v1",
              "status": "fixed_budget_benchmark_complete_not_convergence", "source_root": str(original),
              "cases": [{**case, "input_sha256": file_sha256(bank / "input.json"),
                         "seeds": [{"replicate": i, "seed": 100 + i} for i in (1, 2)]}],
              "records": rows, "planned_external_runs": 2, "artifact_sha256": artifacts}
    write_json_atomic(nested / "reports/report_001/report.json", report)
    write_json_atomic(nested / "_COMPLETE.json", {"report_directory": "reports/report_001",
                      "sha256": file_sha256(nested / "reports/report_001/report.json")})
    write_json_atomic(nested / "analysis_contract.json", {"source_root": str(original), "protected_sha256": trusted})
    return bank, internal, medicc, nested


@pytest.fixture
def inline_workers(monkeypatch):
    class TestOnlyPool:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def map_ordered(self, function, rows, **kwargs):
            for args in rows:
                yield function(*args)
    monkeypatch.setattr(runner, "FreshSpawnTaskPool", TestOnlyPool)
    monkeypatch.setattr(runner, "measured_stage", lambda function, **kwargs: (function(), {"test_only_not_resource_evidence": True}, None))


def test_complete_fixture_reuses_all_three_sources_and_freezes_old_scores(tmp_path, inline_workers):
    roots = fixture_sources(tmp_path)
    sources = collect_sources(*roots)
    assert sources["expected_record_count"] == 10
    before = dict(sources["protected_sha256"])
    output = tmp_path / "audit"
    report = runner.run_audit(sources, output, workers=2, bootstrap_draws=20)
    assert report["status_counts"] == {"success": 10}
    assert report["completed_condition_count"] == 1
    assert len(report["case_scores"]) == 9  # both external seeds form one input-level score
    result = next((output / "cases").glob("*.json"))
    payload = read_json(result)["case_result"]
    assert len(payload["trimmed_tree"]["nodes"]) == 2
    assert all(row["grf_trim"] == 1.0 for row in payload["records"])
    assert all(row["audit"]["ad_fields_match_frozen"] for row in payload["records"])
    assert all(row["grf_full"] == row["audit"]["grf_full"] for row in payload["records"])
    runner.verify_hashes(before)
    first_hashes = {path: file_sha256(path) for path in output.rglob("*") if path.is_file()}
    assert runner.run_audit(sources, output, resume=True, bootstrap_draws=20) == report
    assert first_hashes == {path: file_sha256(path) for path in first_hashes}
    assert (output / "report.md").is_file() and (output / "scores.csv").is_file()


def test_interrupted_reporting_resumes_saved_cases_without_evaluation(tmp_path, inline_workers, monkeypatch):
    sources = collect_sources(*fixture_sources(tmp_path))
    output = tmp_path / "audit"
    original = runner.build_report
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt
    monkeypatch.setattr(runner, "build_report", interrupt)
    with pytest.raises(KeyboardInterrupt):
        runner.run_audit(sources, output, bootstrap_draws=20)
    assert len(list((output / "cases").glob("*.json"))) == 1
    monkeypatch.setattr(runner, "build_report", original)
    monkeypatch.setattr(runner, "evaluate_case", lambda *args: pytest.fail("completed case evaluated twice"))
    assert runner.run_audit(sources, output, resume=True, report_only=True, bootstrap_draws=20)["raw_record_count"] == 10


def test_source_failures_remain_in_inventory_and_paired_denominators(tmp_path, inline_workers):
    sources = collect_sources(*fixture_sources(tmp_path, failed_arm="classical_partial"))
    report = runner.run_audit(sources, tmp_path / "audit", bootstrap_draws=20)
    assert report["status_counts"] == {"source_failure": 1, "success": 9}
    effect = next(row for row in report["paired_contrasts"] if row["left"] == "M2" and row["right"] == "M0")
    assert effect["complete_histories"] == 0 and effect["declared_histories"] == 1


@pytest.mark.parametrize("error_type", [runner.StageTimeoutError, MemoryError])
def test_case_resource_failure_stops_remaining_methods(tmp_path, monkeypatch, error_type):
    sources = collect_sources(*fixture_sources(tmp_path))
    calls = []

    def exhausted(*args):
        calls.append(True)
        raise error_type("fixture exhausted condition budget")

    def capture_stage(function, **kwargs):
        try:
            return function(), {"test_only_not_resource_evidence": True}, None
        except Exception as error:
            return None, {"test_only_not_resource_evidence": True}, error

    monkeypatch.setattr(runner, "evaluate_prepared_trimmed_truth", exhausted)
    monkeypatch.setattr(runner, "measured_stage", capture_stage)
    result = runner.evaluate_case(sources["tasks"][0], 300, 4 * 1024**3)
    assert calls == [True]
    assert len(result["records"]) == 10
    for row in result["records"]:
        assert row["status"] == "audit_failure"
        assert row["failure"]["code"] == error_type.__name__
        assert row["grf_full"] is not None and row["grf_trim"] is None
        assert "audit" not in row


def test_sources_drift_and_output_overlap_are_rejected(tmp_path, inline_workers):
    sources = collect_sources(*fixture_sources(tmp_path))
    with pytest.raises(ValueError, match="overlaps"):
        runner.run_audit(sources, Path(sources["bank_root"]) / "audit")
    ref = sources["tasks"][0]["records"][0]["record_ref"]
    Path(ref["path"]).write_text("{}")
    with pytest.raises(ValueError, match="Protected source changed"):
        runner.run_audit(sources, tmp_path / "audit")


def test_changed_selection_and_cached_corruption_are_rejected(tmp_path, inline_workers, monkeypatch):
    sources = collect_sources(*fixture_sources(tmp_path))
    output = tmp_path / "audit"
    runner.run_audit(sources, output, bootstrap_draws=20)
    with pytest.raises(ValueError, match="settings changed"):
        runner.run_audit(sources, output, resume=True, bootstrap_draws=21)
    path = next((output / "cases").glob("*.json"))
    data = read_json(path)
    data["case_result"]["records"][0]["grf_trim"] = 0.123
    write_json_atomic(path, data)
    with pytest.raises(ValueError, match="Protected source changed"):
        runner.run_audit(sources, output, resume=True, bootstrap_draws=20)


def test_report_only_never_starts_unfinished_work(tmp_path, inline_workers):
    sources = collect_sources(*fixture_sources(tmp_path))
    with pytest.raises(ValueError, match="existing audit"):
        runner.run_audit(sources, tmp_path / "missing", report_only=True)


def test_closed_source_path_escape_and_wrong_case_selection(tmp_path):
    with pytest.raises(ValueError, match="Unsafe"):
        safe_child(tmp_path, "../outside")
    roots = fixture_sources(tmp_path)
    with pytest.raises(ValueError, match="unknown case"):
        collect_sources(*roots, case_ids=["missing"])


def test_duplicate_or_missing_nested_replicates_are_rejected():
    rows = [score(0, "H14_rule_y", "NestedBD-Long", .1, .2, replicate=1)]
    with pytest.raises(ValueError, match="Missing or duplicate"):
        case_scores(rows)


def score(block, condition, method, full, trimmed, *, replicate=None, status="success"):
    return {"case_id": f"case-{block}-{condition}", "block_index": block, "condition_id": condition,
            "height": int(condition[1:3]), "method": method, "problem": "partial",
            "replicate": replicate, "seed": replicate, "source_status": "success", "status": status,
            "grf_full": full, "grf_trim": trimmed, "ad_f1": .5}


def test_pairing_bootstraps_histories_not_nested_heights_or_mcmc_seeds():
    records = []
    for block in (0, 1):
        for condition in ("H14_rule_y", "H24_rule_y"):
            records += [score(block, condition, "M2", .4, .6),
                        score(block, condition, "NestedBD-Long", .1, .2, replicate=1),
                        score(block, condition, "NestedBD-Long", .3, .4, replicate=2)]
    report = build_report(records, bootstrap_draws=20)
    paired = next(row for row in report["paired_contrasts"] if len(row["conditions"]) == 2)
    assert paired["complete_histories"] == 2
    assert paired["condition_pair_count"] == 4
    assert paired["effects"]["grf_full"]["mean"] == pytest.approx(.2)
    assert paired["effects"]["grf_trim"]["mean"] == pytest.approx(.3)
    assert paired["effects"]["grf_trim"]["wtl"]["wins"] == 2
    assert paired["condition_wtl"]["grf_trim"]["wins"] == 4
    records[-1]["status"] = "audit_failure"
    changed = build_report(records, bootstrap_draws=20)
    paired = next(row for row in changed["paired_contrasts"] if len(row["conditions"]) == 2)
    assert paired["complete_histories"] == 1 and paired["declared_histories"] == 2
    assert paired["condition_pair_count"] == 3


def test_sampling_effects_have_explicit_changing_target_caveat():
    rows = [score(0, cid, "M2", .2 + i / 10, .3 + i / 10)
            for i, cid in enumerate(("H24_L2", "H24_rule_y", "H24_L4", "H24_L5", "H24_late", "H24_random"))]
    report = build_report(rows, bootstrap_draws=20)
    assert len(report["sampling_contrasts"]) == 8
    assert "do not hold the target fixed" in report["sampling_caveat"]
    assert report["registered_primary_decision_recomputed"] is False
