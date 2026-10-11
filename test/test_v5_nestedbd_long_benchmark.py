from __future__ import annotations

import copy
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import networkx as nx
import pytest

from algorithm_evaluation import v5_nestedbd_long_benchmark as b
from algorithm_evaluation import v5_nestedbd_long_pilot as p
from algorithm_evaluation import v5_nestedbd_long_topology_comparison as t


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))


def closure(root, name):
    b.confirmation.write_checksum_closure(root, name)


def nexus(path, taxa, *, count=2001):
    def subtree(names):
        if len(names) == 1:
            return names[0] + ":1"
        split = len(names) // 2
        return f"({subtree(names[:split])},{subtree(names[split:])}):1"
    newick = subtree(taxa) + ";"
    path.write_text("#NEXUS\nBegin trees;\n" + "\n".join(
        f"tree STATE_{i * 1000} = [&R] {newick}" for i in range(count)) + "\nEnd;\n")
    return newick


@pytest.fixture
def fixture_run(tmp_path, monkeypatch):
    from test_v5_nestedbd_long_pilot import payload
    from test_v5_nestedbd_long_clock_calibration import calibration_backend
    from algorithm_evaluation import v5_algorithm_development_common as common
    from ctbs import DistanceMatrix
    monkeypatch.setattr(b, "BLOCKS", (0, 1))
    monkeypatch.setitem(b.RULES, "block_indices", [0, 1])
    bank, internal, bio, previous = [tmp_path / name for name in ("bank", "internal", "bio", "previous")]
    for root in (bank, internal, bio, previous):
        root.mkdir()
    approval_path = tmp_path / "approval.py"
    approval_path.write_text("toy approval\n")
    approval = {"status": "approved", "path": str(approval_path), "sha256": p.sha256(approval_path),
                "APPROVED_MANIFEST_SHA256": "a" * 64, "APPROVED_PROTOCOL_SHA256": "b" * 64,
                "APPROVED_SOURCE_LOCK_SHA256": "c" * 64}
    monkeypatch.setattr(b.confirmation, "read_owner_approval_lock", lambda: approval)
    (bank / "owner_approval.snapshot.py").write_bytes(approval_path.read_bytes())
    put(bank / "simulator_config.json", p.read_json(b.calibration.REFERENCE_CONFIG))
    cases, inventory = [], []
    for block in b.BLOCKS:
        truth = nx.DiGraph()
        truth.add_node("root", cell_id=3, generation=0)
        for index in range(3):
            truth.add_node(str(index), cell_id=index, generation=index + 1)
            truth.add_edge("root", str(index))
        truth_path = f"truths/block_{block + 1:03d}.json"
        put(bank / truth_path, {"schema_version": "ctbf-v5-confirmatory-truth-v1",
                               "block_index": block, "tree": b.serialize_tree(truth)})
        distance_path = f"distances/block_{block + 1:03d}.json"
        put(bank / distance_path, {"fixture": "never deserialized by this extension"})
        for condition, height, generations in zip(b.CONDITIONS, (14, 24), p.SCHEDULES[:2]):
            case_id = b.confirmation.confirmation_case_id(block, condition)
            values = payload(case_id, generations)
            values["condition_id"] = condition
            for level in values["levels"]:
                for state in level["states"]:
                    state["state_label"] = 3 if state["state_label"] == "a" else int(state["state_label"])
            case = {"case_id": case_id, "block_index": block, "height": height, "condition_id": condition,
                    "generations": list(generations), "status": "available", "failure": None,
                    "condition_family": "standard", "input_path": f"cases/{case_id}/input.json",
                    "metadata_path": f"cases/{case_id}/case.json", "truth_path": truth_path,
                    "distance_path": distance_path}
            cases.append(case)
            put(bank / case["input_path"], values)
            put(bank / case["metadata_path"], {"reconstruction_seed": 7})
            labels = common.observed_labels(values)
            distance = DistanceMatrix(labels, [[abs(i - j) for i in range(len(labels))] for j in range(len(labels))])
            for arm in t.CTBF_ARMS.values():
                tree, _, _, _ = common.reconstruct_development_arm(
                    common.resolve_arm_specs([arm])[0], values, distance, reconstruction_seed=7)
                native = b.evaluate_tree_pair_result(truth, tree, labels)
                record = {"schema_version": "ctbf-v5-confirmatory-arm-record-v1", "case_id": case_id,
                          "block_index": block, "condition_id": condition, "height": height,
                          "arm_id": arm, "status": "success", "failure": None,
                          "metrics": {"grf": native["metrics"]["grf"]}, "native_evaluation": native,
                          "tree": b.serialize_tree(tree), "tree_summary": common.tree_summary(tree)}
                relative = b.confirmation.record_path(case_id, arm)
                put(internal / relative, record)
                inventory.append({"case_id": case_id, "arm_id": arm, "path": relative,
                                  "sha256": p.sha256(internal / relative), "status": "success"})
    identity = {"status": "complete", "scientific_role": "prospectively_locked_independent_confirmation",
                "bank_id": "toy-bank", "design_manifest_sha256": "a" * 64,
                "protocol_sha256": "b" * 64, "source_lock_sha256": "c" * 64}
    bank_manifest = {**identity, "schema_version": b.confirmation.BANK_SCHEMA_VERSION,
                     "block_count": 100, "owner_approval_sha256": approval["sha256"],
                     "cases": cases, "condition_inventory": copy.deepcopy(cases)}
    put(bank / b.confirmation.BANK_MANIFEST_NAME, bank_manifest)
    closure(bank, b.confirmation.BANK_CHECKSUM_NAME)
    put(internal / b.confirmation.RUN_MANIFEST_NAME, {**identity,
        "schema_version": b.confirmation.RUN_SCHEMA_VERSION,
        "bank_checksum_sha256": p.sha256(bank / b.confirmation.BANK_CHECKSUM_NAME), "record_inventory": inventory})
    closure(internal, b.confirmation.RUN_CHECKSUM_NAME)
    backend = calibration_backend()
    backend.audit = SimpleNamespace(review=SimpleNamespace(compare_windows=lambda *_: {
        str(burn): {"shared_majority_clades": 1, "majority_clades_per_seed": [1, 1],
                    "maximum_clade_frequency_difference": 0} for burn in (10, 25, 50)}))
    monkeypatch.setattr(p, "load_backend", lambda _: backend)
    monkeypatch.setattr(p, "backend_snapshot", lambda _: {})
    monkeypatch.setattr(b, "prior_snapshot", lambda _: {})
    calls, newicks = [], {}
    failures = {}
    def execute(_backend, directory, case, phase, seed, clock, stop):
        calls.append((case["case_id"], phase, seed))
        directory.mkdir(parents=True)
        _, xml, _ = b.calibration.make_xml(backend, case, phase, clock)
        xml.write(directory / "input.xml")
        status = failures.get((case["case_id"], phase, seed), "technical_complete")
        if status == "interrupted":
            stop.set()
        record = {**b.calibration.run_identity(case, phase, seed, clock), "attempt": str(directory),
                  "status": status, "execution": {"wall_seconds": 1, "peak_rss_bytes": 100000,
                                                  "resource_scope": "test double, not resource evidence"}}
        if phase == "pilot" and status == "technical_complete":
            path = directory / "shared_pilot.trees"
            newicks[path] = nexus(path, [row["taxon"] for row in case["records"]])
            record["scalar_windows"] = {"10": {"official_beast_statistics": {
                key: {"ESS": 30} for key in ("posterior", "likelihood", "TreeHeight")},
                "posterior_first_half_mean": -11, "posterior_second_half_mean": -10}}
            put(directory / "clade_windows.json", {
                str(burn): {"counts": {"7": 10}, "samples": 10} for burn in (10, 25, 50)})
        p.write_json(directory / "report.json", record)
        t._seal(directory)
        return record
    monkeypatch.setattr(b.calibration, "execute_attempt", execute)
    def numeric(_backend, directory):
        directory.mkdir()
        result = {"passes_observed_range": True}
        p.write_json(directory / "report.json", result)
        t._seal(directory)
        return result
    monkeypatch.setattr(p, "numerical_control", numeric)
    def summary(_backend, path, directory):
        (directory / "mcc.tree").write_text("#NEXUS\nBegin trees;\ntree TREE1 = [&R] " + newicks[path] + "\nEnd;\n")
        return {"inference": False}
    monkeypatch.setattr(t, "native_summary", summary)
    return SimpleNamespace(bank=bank, internal=internal, bio=bio, previous=previous,
        output=tmp_path / "output", calls=calls, failures=failures, backend=backend)


def prepare(s, **kwargs):
    return b.prepare(s.bank, s.internal, s.bio, s.previous, s.output,
                     **{"parallel": 2, "resume": True, **kwargs})


def run(s):
    cases, contract, backend = prepare(s)
    return b.run_prepared(backend, s.bank, s.output, cases, contract)


def refresh_fixture_closures(s):
    closure(s.bank, b.confirmation.BANK_CHECKSUM_NAME)
    path = s.internal / b.confirmation.RUN_MANIFEST_NAME
    value = p.read_json(path)
    value["bank_checksum_sha256"] = p.sha256(s.bank / b.confirmation.BANK_CHECKSUM_NAME)
    put(path, value)
    closure(s.internal, b.confirmation.RUN_CHECKSUM_NAME)


def test_prespecified_twenty_cases_and_forty_unique_seeds():
    ids = b.selected_ids()
    assert len(ids) == 20
    assert ids[0] == "conf-b001-H14_rule_y" and ids[-1] == "conf-b010-H24_rule_y"
    seeds = [b.seed_for(case, replicate) for case in ids for replicate in (1, 2)]
    assert len(set(seeds)) == 40 and min(seeds) > 0 and max(seeds) < 2147483647
    assert b.CLOCK["mean_clock_rate"] == 0.001061875
    assert b.RULES["inference"]["pilot_iterations"] == 2000000
    assert t.RULES["cases"] == list(b.calibration.SELECTED_CASES)  # old contract was not mutated


def test_preparation_never_deserializes_truth_distance_or_comparator_scores(fixture_run, monkeypatch):
    s = fixture_run
    reader = p.read_json
    def guarded(path):
        if Path(path).parent.name in ("truths", "distances") or "records" in Path(path).parts:
            pytest.fail("outcome or latent information accessed during preparation")
        return reader(path)
    monkeypatch.setattr(p, "read_json", guarded)
    cases, contract, _ = prepare(s)
    assert len(cases) == 4 and all(case["eligibility"] == "compatible" for case in cases)
    assert contract["truth_or_scores_deserialized_during_preparation"] is False
    assert s.calls == []
    assert prepare(s)[1] == contract
    with pytest.raises(FileExistsError):
        prepare(s, resume=False)


def test_selection_is_original_block_order_not_input_size_or_manifest_order(fixture_run):
    s = fixture_run
    path = s.bank / b.confirmation.BANK_MANIFEST_NAME
    value = p.read_json(path)
    value["cases"].reverse()
    value["condition_inventory"].reverse()
    put(path, value)
    refresh_fixture_closures(s)
    cases, _ = b.load_selection(s.bank, s.internal)
    assert [case["case_id"] for case in cases] == b.selected_ids()


def test_unsupported_copy_value_is_kept_not_clipped_or_replaced(fixture_run):
    s = fixture_run
    path = s.bank / f"cases/{b.selected_ids()[0]}/input.json"
    value = p.read_json(path)
    value["levels"][0]["states"][1]["cnp"][37] = 10
    put(path, value)
    refresh_fixture_closures(s)
    cases, _, _ = prepare(s)
    assert len(cases) == 4 and cases[0]["eligibility"] == "blocked_copy_number_range"
    assert cases[0]["records"][1]["cnp"][37] == 10
    copied = p.read_json(s.output / "inputs" / cases[0]["case_id"] / "input.json")
    assert copied == value
    outcome = b.inference_task(s.backend, s.output, cases[0], cases[0]["seeds"][0]["seed"], threading.Event())
    assert outcome["status"] == "ineligible" and not s.calls


def test_changed_checksum_member_and_output_overlap_refused(fixture_run):
    s = fixture_run
    with pytest.raises(ValueError, match="overlap"):
        b.prepare(s.bank, s.internal, s.bio, s.previous, s.bank / "new", parallel=2, resume=True)
    (s.bank / f"cases/{b.selected_ids()[0]}/input.json").write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        b.load_selection(s.bank, s.internal)


def test_complete_toy_workflow_freezes_contract_replays_metrics_and_resumes_readonly(fixture_run):
    s = fixture_run
    result = run(s)
    assert result["external_status_counts"] == {"success": 8}
    assert len(result["records"]) == 16
    assert all(row["frozen_evaluator_replay_exact"] for row in result["records"] if row["method"] != "NestedBD-Long")
    assert len(s.calls) == 16  # one short check and one long-call test double per seed
    snapshots = {str(path): p.sha256(path) for path in s.output.rglob("*") if path.is_file()}
    assert run(s) == result
    assert len(s.calls) == 16
    assert snapshots == {str(path): p.sha256(path) for path in s.output.rglob("*") if path.is_file()}
    assert all(row["n"] == 2 for row in result["paired_effects"]["history_paired"])
    report = b.report_text(result)
    assert "Post-registration comparator" in report and "No p-values" in report
    assert "not trees or posterior samples" in report


@pytest.mark.parametrize("status", ["timeout", "rss_limit", "capture_limit", "failed"])
def test_terminal_failures_are_retained_without_retry_or_stopping_other_cases(fixture_run, status):
    s = fixture_run
    case = b.selected_ids()[0]
    s.failures[(case, "pilot", b.seed_for(case, 1))] = status
    result = run(s)
    assert result["external_status_counts"] == {"failure": 1, "success": 7}
    assert all(row["n"] == 1 and row["missing_histories"] == 1 for row in result["paired_effects"]["history_paired"])
    calls = list(s.calls)
    s.failures.clear()
    assert run(s) == result and s.calls == calls


def test_failed_short_check_blocks_only_that_seed(fixture_run):
    s = fixture_run
    case = b.selected_ids()[0]
    seed = b.seed_for(case, 1)
    s.failures[(case, "technical", seed)] = "failed"
    result = run(s)
    assert result["external_status_counts"] == {"failure": 1, "success": 7}
    assert (case, "pilot", seed) not in s.calls
    assert len(s.calls) == 15


def test_unavailable_bank_condition_remains_in_report_without_replacement(fixture_run):
    s = fixture_run
    case_id = b.selected_ids()[0]
    path = s.bank / b.confirmation.BANK_MANIFEST_NAME
    manifest = p.read_json(path)
    manifest["cases"] = [row for row in manifest["cases"] if row["case_id"] != case_id]
    declaration = next(row for row in manifest["condition_inventory"] if row["case_id"] == case_id)
    declaration.update(status="unavailable", failure={"code": "fixture_bank_failure"})
    put(path, manifest)
    run_path = s.internal / b.confirmation.RUN_MANIFEST_NAME
    run_manifest = p.read_json(run_path)
    for item in run_manifest["record_inventory"]:
        if item["case_id"] == case_id:
            record_path = s.internal / item["path"]
            record = p.read_json(record_path)
            record.update(status="dependency_failure", failure={"code": "fixture_bank_failure"})
            put(record_path, record)
            item.update(status="dependency_failure", sha256=p.sha256(record_path))
    put(run_path, run_manifest)
    refresh_fixture_closures(s)
    result = run(s)
    assert len(result["records"]) == 16
    assert result["external_status_counts"] == {"ineligible": 2, "success": 6}
    assert {row["case_id"] for row in result["records"]} == set(b.selected_ids())
    assert not any(row[0] == case_id for row in s.calls)
    assert all(row["missing_histories"] == 1 for row in result["paired_effects"]["history_paired"])


def test_interrupted_attempt_restarts_from_zero_and_preserves_old_files(fixture_run):
    s = fixture_run
    cases, _, backend = prepare(s)
    case, seed = cases[0], cases[0]["seeds"][0]["seed"]
    s.failures[(case["case_id"], "technical", seed)] = "interrupted"
    assert b.inference_task(backend, s.output, case, seed, threading.Event()) is None
    unit = s.output / "inference" / case["case_id"] / f"seed_{seed}" / "technical"
    first = unit / "attempt_001"
    digest = p.sha256(first / "report.json")
    s.failures.clear()
    outcome = b.inference_task(backend, s.output, case, seed, threading.Event())
    assert outcome["status"] == "success"
    assert p.sha256(first / "report.json") == digest and (unit / "attempt_002").is_dir()
    assert outcome["run"]["native_checkpoint_restoration"] is False


def test_orphan_beast_process_refused_before_restart(fixture_run):
    s = fixture_run
    cases, _, backend = prepare(s)
    case = cases[0]
    unit = s.output / "orphan"
    put(unit / "attempt_001/execution_started.json", {"pid": 123})
    backend.pilot.process_alive = lambda _: True
    with pytest.raises(ValueError, match="still running"):
        b.choose_inference_attempt(backend, unit, case, "technical", case["seeds"][0]["seed"])


def test_root_os_lock_prevents_concurrent_controllers(tmp_path):
    with b.owner_lock(tmp_path):
        with pytest.raises(ValueError, match="another"):
            with b.owner_lock(tmp_path):
                pass
    with b.owner_lock(tmp_path):
        pass


def test_bounded_check_never_requests_pilot_or_scores(fixture_run, monkeypatch):
    s = fixture_run
    cases, contract, backend = prepare(s)
    monkeypatch.setattr(b, "finish_report", lambda *_: pytest.fail("bounded check tried scoring"))
    result = b.run_prepared(backend, s.bank, s.output, cases, contract, check=True)
    assert result["success"] and len(s.calls) <= 2
    assert {row[1] for row in s.calls} == {"technical"}
    calls = list(s.calls)
    assert b.run_prepared(backend, s.bank, s.output, cases, contract, check=True) == result
    assert s.calls == calls


def test_clade_columns_follow_replicates_not_parallel_completion_order(tmp_path):
    case_id = b.selected_ids()[0]
    outcomes = []
    for replicate in (1, 2):
        directory = tmp_path / str(replicate)
        put(directory / "clade_windows.json", {"10": {"marker": replicate}})
        outcomes.append({"case_id": case_id, "seed": b.seed_for(case_id, replicate),
                         "status": "success", "run": {"attempt": str(directory)}})
    backend = SimpleNamespace(audit=SimpleNamespace(review=SimpleNamespace(
        compare_windows=lambda left, right: [left[10]["marker"], right[10]["marker"]])))
    assert b.clade_agreement(backend, list(reversed(outcomes)))[case_id] == [1, 2]


def test_numeric_gate_failure_prevents_inference(fixture_run, monkeypatch):
    s = fixture_run
    cases, contract, backend = prepare(s)
    monkeypatch.setattr(b, "numeric_gate", lambda *_: {"passes_observed_range": False})
    with pytest.raises(ValueError, match="arithmetic"):
        b.run_prepared(backend, s.bank, s.output, cases, contract)
    assert not s.calls


def test_fatal_controller_error_signals_workers_before_executor_shutdown(fixture_run, monkeypatch):
    s = fixture_run
    cases, contract, backend = prepare(s)
    first_seed = cases[0]["seeds"][0]["seed"]
    def inference(_backend, _output, _case, seed, stop):
        if seed == first_seed:
            raise ValueError("fixture integrity failure")
        assert stop.wait(2), "controller waited for workers before requesting stop"
        return None
    monkeypatch.setattr(b, "inference_task", inference)
    with pytest.raises(ValueError, match="fixture integrity"):
        b.run_prepared(backend, s.bank, s.output, cases, contract)
    assert not (s.output / "_COMPLETE.json").exists()


def test_frozen_ctbf_record_replay_detects_changed_metric(fixture_run):
    s = fixture_run
    cases, _, _ = prepare(s)
    path = Path(cases[0]["internal_records"]["M2"])
    value = p.read_json(path)
    value["metrics"]["grf"] += .01
    put(path, value)
    with pytest.raises(ValueError, match="replay"):
        b.frozen_comparator(s.bank, cases[0], "M2")


def effect_records():
    rows = []
    for block in b.BLOCKS:
        for height in (14, 24):
            for method, replicate, score in (("M0", None, .1), ("M2", None, .5),
                                             ("NestedBD-Long", 1, .2), ("NestedBD-Long", 2, .4)):
                rows.append({"block_index": block, "height": height, "method": method,
                             "replicate": replicate, "status": "success", "grf": score})
    return rows


def test_paired_effects_average_within_history_before_bootstrap_and_wtl():
    result = b.paired_effects(effect_records())
    m2 = next(row for row in result["history_paired"] if row["method"] == "M2")
    assert m2["n"] == 10 and m2["wins"] == 10 and m2["mean"] == pytest.approx(.2)
    assert m2["descriptive_bootstrap_95_percent_interval"] == pytest.approx([.2, .2])
    assert all(row["n"] == 10 for row in result["per_height_and_seed"])
    assert next(row for row in result["history_paired"] if row["method"] == "M0")["losses"] == 10


def test_missing_seed_is_not_imputed_as_loss_or_replaced_by_other_seed():
    rows = effect_records()
    target = next(row for row in rows if row["method"] == "NestedBD-Long")
    target.update(status="failure", grf=None)
    result = b.paired_effects(rows)
    assert all(row["n"] == 9 and row["missing_histories"] == 1 for row in result["history_paired"])
    assert sum(row["missing_pairs"] for row in result["per_height_and_seed"]) == 2


def test_duplicate_pair_cell_is_rejected():
    rows = effect_records()
    with pytest.raises(ValueError, match="duplicate"):
        b.paired_effects(rows + [rows[0]])
