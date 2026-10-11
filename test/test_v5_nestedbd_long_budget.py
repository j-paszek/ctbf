from __future__ import annotations

import copy
import json
from pathlib import Path
import threading
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest

from algorithm_evaluation import v5_nestedbd_long_budget as budget
from algorithm_evaluation import v5_nestedbd_long_budget_report as reporting
from test_v5_nestedbd_long_clock_calibration import calibration_backend
from test_v5_nestedbd_long_pilot import payload


p, b, t = budget.pilot, budget.benchmark, budget.topology


def put(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    p.write_json(path, data)


def stats(ess):
    return {str(burn): {"official_beast_statistics": {
        key: {"ESS": ess} for key in ("posterior", "likelihood", "TreeHeight")},
        "posterior_first_half_mean": -20, "posterior_second_half_mean": -19}
        for burn in (10, 25, 50)}


def windows(samples=10):
    return {str(burn): {"samples": samples, "counts": {"7": samples}} for burn in (10, 25, 50)}


def compare(first, second):
    return {str(burn): {"majority_clades_per_seed": [1, 1], "shared_majority_clades": 1,
            "maximum_clade_frequency_difference": 0,
            "retained_per_seed": [first[burn]["samples"], second[burn]["samples"]]}
            for burn in (10, 25, 50)}


@pytest.fixture
def fixture_budget(tmp_path, monkeypatch):
    source, recovered, bank, bio, output = [tmp_path / name for name in ("source", "recovered", "bank", "bio", "output")]
    source.mkdir()
    (source / ".runner.lock").touch()
    backend = calibration_backend()
    corrected = backend.corrected_xml

    def configured(sequences, config, store):
        baseline, tree, proof = corrected(sequences, config, store)
        for value in (baseline, tree):
            value.getroot().find("./run").set("chainLength", str(config["chain_length"]))
        return baseline, tree, proof

    backend.corrected_xml = configured
    backend.audit = SimpleNamespace(review=SimpleNamespace(compare_windows=compare))
    monkeypatch.setattr(p, "load_backend", lambda _: backend)
    monkeypatch.setattr(p, "backend_snapshot", lambda _: {})
    cases, original_protected, recovered_protected = [], {}, {}
    for block in (0, 1):
        for height, generations in ((14, [9, 12, 14]), (24, [15, 20, 24])):
            condition = f"H{height}_rule_y"
            case_id = b.confirmation.confirmation_case_id(block, condition)
            data = payload(case_id, generations)
            data["condition_id"] = condition
            path = bank / case_id / "input.json"
            put(path, data)
            original_protected[str(path)] = p.sha256(path)
            recovered_protected[str(path)] = p.sha256(path)
            records = p.input_records(data)
            case = {"case_id": case_id, "block_index": block, "height": height, "condition_id": condition,
                    "generations": generations, "source_case": {"never_read_truth": True}, "eligibility": "compatible",
                    "input_path": str(path), "input_sha256": p.sha256(path), "input_summary": p.compatibility(records),
                    "records": records, "seeds": [{"replicate": rep, "seed": b.seed_for(case_id, rep)} for rep in (1, 2)]}
            cases.append(case)
            for seed in case["seeds"]:
                directory = source / "inference" / case_id / f"seed_{seed['seed']}" / "pilot" / "attempt_001"
                directory.mkdir(parents=True)
                _, tree, _ = b.calibration.make_xml(backend, case, "pilot", b.CLOCK)
                tree.write(directory / "input.xml")
                row = {**b.calibration.run_identity(case, "pilot", seed["seed"], b.CLOCK),
                       "attempt": str(directory), "status": "technical_complete", "scalar_windows": stats(30),
                       "execution": {"wall_seconds": 60, "peak_rss_bytes": 1234}}
                put(directory / "report.json", row)
                put(directory / "clade_windows.json", windows())
                t._seal(directory)
                recovered_protected[str(directory / "_SEALED.json")] = p.sha256(directory / "_SEALED.json")
    contract_path = source / "analysis_contract.json"
    put(contract_path, {"schema_version": b.SCHEMA, "rules": b.RULES,
                        "bio_root": str(bio), "bank_root": str(bank), "protected_sha256": original_protected,
                        "cases": [{key: value for key, value in case.items() if key != "records"} for case in cases]})
    recovered_protected[str(contract_path)] = p.sha256(contract_path)
    # It is deliberately not JSON: the budget loader must only hash this
    # accuracy report, not deserialize it or use it for selection.
    report_path = recovered / "reports/report_001/report.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text("accuracy forbidden to this diagnostic\n")
    put(recovered / "_COMPLETE.json", {"report_directory": "reports/report_001", "sha256": p.sha256(report_path)})
    put(recovered / "analysis_contract.json", {"source_root": str(source), "protected_sha256": recovered_protected})
    calls, failures = [], {}

    def execute(_backend, directory, case, phase, seed, stop):
        calls.append((case["case_id"], phase, seed))
        directory.mkdir(parents=True)
        _, tree, _ = budget.make_xml(backend, case, phase)
        tree.write(directory / "input.xml")
        status = failures.get((case["case_id"], phase, seed), "technical_complete")
        row = {**budget.run_identity(case, phase, seed), "attempt": str(directory), "status": status,
               "execution": {"wall_seconds": 300, "peak_rss_bytes": 2345, "test_only_not_resource_evidence": True}}
        if phase == "long" and status == "technical_complete":
            row["scalar_windows"] = stats(70)
            put(directory / "clade_windows.json", windows(50))
        put(directory / "report.json", row)
        t._seal(directory)
        if status == "interrupted":
            stop.set()
        return row

    def numeric(_backend, directory):
        directory.mkdir()
        row = {"passes_observed_range": True}
        put(directory / "report.json", row)
        t._seal(directory)
        return row

    monkeypatch.setattr(budget, "execute_attempt", execute)
    monkeypatch.setattr(p, "numerical_control", numeric)
    return SimpleNamespace(source=source, recovered=recovered, bank=bank, bio=bio, output=output,
                           cases=cases, backend=backend, calls=calls, failures=failures, execute=execute)


def prepare(s):
    return budget.prepare(s.recovered, s.bio, s.output, resume=True)


def test_frozen_selection_and_original_seed_identity(fixture_budget):
    s = fixture_budget
    cases, contract, _ = prepare(s)
    assert [case["case_id"] for case in cases] == budget.selected_ids()
    assert len(cases) == 4 and len(contract["baseline_runs"]) == 8
    assert all(case["records"] == original["records"] for case, original in zip(cases, s.cases))
    assert contract["truth_or_accuracy_deserialized"] is False
    assert all(row["seed"] == b.seed_for(row["case_id"], row["replicate"]) for row in contract["baseline_runs"])
    assert budget.RULES["main_benchmark_iterations"] == 2000000


@pytest.mark.parametrize("phase,length,timeout", [("technical", 10000, 300), ("long", 10000000, 18000)])
def test_exact_xml_change_is_only_chain_length(fixture_budget, phase, length, timeout):
    s = fixture_budget
    original, new, proof = budget.make_xml(s.backend, s.cases[0], phase)
    assert new.getroot().find("./run").get("chainLength") == str(length)
    assert original.getroot().find("./run").get("chainLength") == "2000000"
    restored = copy.deepcopy(new)
    restored.getroot().find("./run").set("chainLength", "2000000")
    assert ET.tostring(restored.getroot()) == ET.tostring(original.getroot())
    assert proof["all_other_xml_unchanged"]
    p.verify_xml_input(new, s.cases[0])
    config = budget.phase_config(s.cases[0], phase)
    assert config["timeout_seconds_per_run"] == timeout
    assert config["log_every"] == 1000 and config["burnin_percent"] == 10


def test_complete_run_resume_and_report_are_read_only_to_sources(fixture_budget, monkeypatch):
    s = fixture_budget
    cases, contract, backend = prepare(s)
    report = budget.run_prepared(backend, s.output, cases, contract)
    assert report["new_success_count"] == report["planned_new_runs"] == 8
    assert len(s.calls) == 16 and len(report["scalar_table"]) == 16
    assert len(report["paired_scalar_changes"]) == 8
    assert report["paired_scalar_changes"][0]["ten_million_minus_two_million"]["posterior_ess"] == 40
    assert report["scalar_table"][0]["posterior_ess_per_million"] == 15
    assert report["scalar_table"][8]["posterior_ess_per_million"] == 7
    assert report["within_seed_across_budget_clades"][0]["windows"]["10"]["retained_per_seed"] == [10, 50]
    p.verify_hashes(contract["protected_sha256"])
    saved = {path: p.sha256(path) for path in s.output.rglob("*") if path.is_file()}
    monkeypatch.setattr(budget, "inference_task", lambda *args: pytest.fail("completed work repeated"))
    assert budget.run_prepared(backend, s.output, cases, contract, parallel=1, report_only=True) == report
    assert {path: p.sha256(path) for path in saved} == saved
    text = budget.report_text(report)
    assert "main benchmark remains at 2M" in text and "not proof of convergence" in text


def test_check_only_runs_first_ten_thousand_steps(fixture_budget):
    s = fixture_budget
    cases, contract, backend = prepare(s)
    assert budget.run_prepared(backend, s.output, cases, contract, check=True)["success"]
    assert len(s.calls) == 1 and s.calls[0][1] == "technical"
    assert not (s.output / "_COMPLETE.json").exists()
    assert budget.run_prepared(backend, s.output, cases, contract, check=True)["success"]
    assert len(s.calls) == 1


@pytest.mark.parametrize("phase", ["technical", "long"])
def test_terminal_failure_retained_without_stopping_other_seeds_or_retry(fixture_budget, phase):
    s = fixture_budget
    case = s.cases[0]
    seed = case["seeds"][0]["seed"]
    s.failures[(case["case_id"], phase, seed)] = "timeout"
    cases, contract, backend = prepare(s)
    report = budget.run_prepared(backend, s.output, cases, contract)
    assert report["new_success_count"] == 7
    assert len(report["new_runs"]) == 8
    assert sum(row["complete"] for row in report["paired_scalar_changes"]) == 7
    assert report["between_seed_clades"][4]["status"] == "incomplete"
    count = len(s.calls)
    budget.run_prepared(backend, s.output, cases, contract)
    assert len(s.calls) == count


def test_interrupted_attempt_restarts_without_overwriting_checkpoint(fixture_budget):
    s = fixture_budget
    case = s.cases[0]
    seed = case["seeds"][0]["seed"]
    unit = s.output / "inference" / case["case_id"] / f"seed_{seed}" / "long"
    s.failures[(case["case_id"], "long", seed)] = "interrupted"
    old = s.execute(s.backend, unit / "attempt_001", case, "long", seed, threading.Event())
    before = p.sha256(Path(old["attempt"]) / "report.json")
    path, record = budget.choose_attempt(s.backend, unit, case, "long", seed)
    assert record is None and path.name == "attempt_002"
    assert p.sha256(Path(old["attempt"]) / "report.json") == before
    assert budget.RULES["native_checkpoint_restoration"] is False


def test_report_only_does_not_start_missing_work(fixture_budget):
    s = fixture_budget
    cases, contract, backend = prepare(s)
    with pytest.raises(ValueError, match="never starts inference"):
        budget.run_prepared(backend, s.output, cases, contract, report_only=True)
    assert not s.calls


def test_report_interruption_resumes_all_saved_seeds(fixture_budget, monkeypatch):
    s = fixture_budget
    cases, contract, backend = prepare(s)
    original = budget.make_report
    def interrupt(*args):
        raise KeyboardInterrupt
    monkeypatch.setattr(budget, "make_report", interrupt)
    with pytest.raises(KeyboardInterrupt):
        budget.run_prepared(backend, s.output, cases, contract)
    assert len(s.calls) == 16
    monkeypatch.setattr(budget, "make_report", original)
    result = budget.run_prepared(backend, s.output, cases, contract, report_only=True)
    assert result["new_success_count"] == 8 and len(s.calls) == 16


def test_changed_source_and_output_overlap_refused(fixture_budget):
    s = fixture_budget
    with pytest.raises(ValueError, match="overlaps"):
        budget.prepare(s.recovered, s.bio, s.source / "new", resume=True)
    path = Path(s.cases[0]["input_path"])
    path.write_text("changed")
    with pytest.raises(ValueError, match="Changed or unbound"):
        prepare(s)


def test_changed_contract_and_changed_cache_refused(fixture_budget):
    s = fixture_budget
    cases, contract, backend = prepare(s)
    path = s.output / "analysis_contract.json"
    changed = p.read_json(path)
    changed["rules"]["settings"]["long_iterations"] = 2000000
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="rules, code or runtime changed"):
        prepare(s)
    case = cases[0]
    seed = case["seeds"][0]["seed"]
    unit = s.output / "attempt_test"
    s.execute(backend, unit / "attempt_001", case, "long", seed, threading.Event())
    (unit / "attempt_001/input.xml").write_text("changed")
    with pytest.raises(ValueError, match="protected input or implementation changed"):
        budget.choose_attempt(backend, unit, case, "long", seed)


def test_missing_seed_is_not_silently_selected_away(fixture_budget):
    s = fixture_budget
    cases, contract, backend = prepare(s)
    with pytest.raises(ValueError, match="Missing or duplicate"):
        reporting.make_report(backend, cases, contract["baseline_runs"], contract["baseline_runs"][:-1])


def test_nonfinite_diagnostics_are_reported_as_missing():
    assert reporting.finite(float("nan")) is None
    assert reporting.finite(float("inf")) is None
    assert reporting.finite(None) is None
