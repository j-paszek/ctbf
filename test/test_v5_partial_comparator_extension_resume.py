"""Small synthetic resume regressions. Never launch BEAST or read owner results."""

from __future__ import annotations

import copy
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from algorithm_evaluation import v5_partial_comparator_extension as runner
from algorithm_evaluation import v5_partial_comparator_extension_common as common
from algorithm_evaluation import v5_partial_comparator_extension_2m_nested as nested
from test_v5_nestedbd_long_budget import fixture_budget, stats
from test_v5_partial_comparator_extension import case, put


@pytest.fixture
def two_million(fixture_budget, monkeypatch):
    s = fixture_budget

    def execute(backend, directory, data, phase, seed, clock, stop):
        s.calls.append((data["case_id"], phase, seed))
        directory.mkdir(parents=True)
        _, xml, _ = nested.calibration.make_xml(backend, data, phase, clock)
        xml.write(directory / "input.xml")
        status = s.failures.get((data["case_id"], phase, seed), "technical_complete")
        record = {**nested.calibration.run_identity(data, phase, seed, clock),
                  "status": status, "attempt": str(directory),
                  "scalar_windows": stats(30), "execution": {"fixture_only": True}}
        put(directory / "report.json", record)
        nested.topology._seal(directory)
        if status == "interrupted":
            stop.set()
        return record

    monkeypatch.setattr(nested.calibration, "execute_attempt", execute)
    monkeypatch.setattr(s.backend, "choose_attempt",
                        lambda *_: pytest.fail("Legacy failure/retry policy must not be used"))
    return s


def snapshot(root):
    return {str(p.relative_to(root)): nested.pilot.sha256(p)
            for p in root.rglob("*") if p.is_file()}


def unit(s, phase="pilot"):
    data = s.cases[0]
    seed = data["seeds"][0]["seed"]
    return s.output / "inference" / data["case_id"] / f"seed_{seed}" / phase


def infer(s):
    data = s.cases[0]
    return nested.inference_task(s.backend, s.output, data,
                                 data["seeds"][0]["seed"], threading.Event())


def cached(s):
    data = s.cases[0]
    return nested.cached_inference(s.backend, s.output, data, data["seeds"][0]["seed"])


def reseal(directory):
    # Only test fixtures are changed. Production attempts remain immutable.
    (directory / "_SEALED.json").unlink()
    nested.topology._seal(directory)


def test_completed_2m_inference_and_report_reuse_without_execution(two_million):
    s = two_million
    first = infer(s)
    assert first["status"] == "success"
    assert first["run"]["config"]["chain_length"] == 2_000_000
    saved = snapshot(s.output)
    assert infer(s) == cached(s) == first
    assert len(s.calls) == 2 and snapshot(s.output) == saved


@pytest.mark.parametrize("phase", ["technical", "pilot"])
@pytest.mark.parametrize("status", ["failed", "timeout", "rss_limit", "capture_limit"])
def test_terminal_failures_are_retained_without_retry(two_million, phase, status):
    s = two_million
    data, seed = s.cases[0], s.cases[0]["seeds"][0]["seed"]
    s.failures[(data["case_id"], phase, seed)] = status
    first = infer(s)
    assert (first["status"], first["stage"], first["reason"]) == ("failure", phase, status)
    saved, calls = snapshot(s.output), list(s.calls)
    assert infer(s) == cached(s) == first
    assert s.calls == calls and snapshot(s.output) == saved


def test_sealed_interruption_restarts_only_unfinished_phase(two_million):
    s = two_million
    data, seed = s.cases[0], s.cases[0]["seeds"][0]["seed"]
    s.failures[(data["case_id"], "pilot", seed)] = "interrupted"
    assert infer(s) is None
    saved = snapshot(s.output)
    with pytest.raises(ValueError, match="unfinished"):
        cached(s)
    assert snapshot(s.output) == saved and len(s.calls) == 2
    s.failures.clear()
    result = infer(s)
    assert len(s.calls) == 3 and s.calls[-1][1] == "pilot"
    assert result["run"]["attempt"].endswith("pilot/attempt_002")
    assert result["run"]["native_checkpoint_restoration"] is False
    assert all(nested.pilot.sha256(s.output / path) == digest for path, digest in saved.items())


def test_force_killed_attempt_is_preserved_and_never_restores_state(two_million):
    s = two_million
    previous = unit(s, "technical") / "attempt_004"
    previous.mkdir(parents=True)
    (previous / "input.xml.state").write_text("unfinished native state, not resumed")
    put(previous / "execution_started.json", {"fixture_process_is_dead": True})
    before = snapshot(previous)
    assert infer(s)["status"] == "success"
    assert snapshot(previous) == before
    assert (unit(s, "technical") / "attempt_005" / "report.json").is_file()
    assert len(s.calls) == 2


def test_report_only_cannot_create_missing_attempts(two_million):
    s = two_million
    with pytest.raises(ValueError, match="unfinished"):
        cached(s)
    assert not s.output.exists() and not s.calls


def test_live_jvm_and_unsealed_report_remain_blocking(two_million, monkeypatch):
    s = two_million
    directory = unit(s, "technical") / "attempt_001"
    put(directory / "execution_started.json", {})
    monkeypatch.setattr(s.backend.pilot, "process_alive", lambda _: True)
    with pytest.raises(ValueError, match="still running"):
        infer(s)
    monkeypatch.setattr(s.backend.pilot, "process_alive", lambda _: False)
    put(directory / "report.json", {"status": "failed"})
    with pytest.raises(ValueError, match="Unsealed"):
        infer(s)
    assert not s.calls


@pytest.mark.parametrize("change", ["seal", "budget", "seed", "input", "clock", "xml", "missing_checksum", "status"])
def test_cached_failed_attempt_cannot_bypass_identity_or_seal(two_million, change):
    s = two_million
    data, seed = s.cases[0], s.cases[0]["seeds"][0]["seed"]
    s.failures[(data["case_id"], "pilot", seed)] = "failed"
    first = infer(s)
    directory = Path(first["attempt"])
    record = nested.pilot.read_json(directory / "report.json")
    if change == "budget":
        record["config"]["chain_length"] = 10_000_000
    elif change == "seed":
        record["seed"] += 1
    elif change == "input":
        record["input_sha256"] = "b" * 64
    elif change == "clock":
        record["clock"]["mean_clock_rate"] = 1
    elif change == "xml":
        text = (directory / "input.xml").read_text().replace('chainLength="2000000"', 'chainLength="10000000"')
        (directory / "input.xml").write_text(text)
    elif change == "status":
        record["status"] = "unexpected"
    record["tampering_fixture"] = True
    runner.write_json_atomic(directory / "report.json", record)
    if change != "seal":
        reseal(directory)
    if change == "missing_checksum":
        seal = nested.pilot.read_json(directory / "_SEALED.json")
        del seal["report.json"]
        runner.write_json_atomic(directory / "_SEALED.json", seal)
    with pytest.raises(ValueError):
        infer(s)
    assert len(s.calls) == 2


def test_early_failure_without_xml_stays_terminal(two_million):
    s = two_million
    data, seed = s.cases[0], s.cases[0]["seeds"][0]["seed"]
    directory = unit(s, "technical") / "attempt_001"
    record = {**nested.calibration.run_identity(data, "technical", seed, nested.benchmark.CLOCK),
              "status": "failed", "attempt": str(directory), "error": "pre-execution fixture"}
    put(directory / "report.json", record)
    nested.topology._seal(directory)
    assert infer(s) == cached(s)
    assert not s.calls


def test_duplicate_terminal_outcomes_are_not_selected_by_success(two_million):
    s = two_million
    infer(s)
    old = unit(s) / "attempt_001"
    other = unit(s) / "attempt_002"
    record = nested.pilot.read_json(old / "report.json")
    record.update(status="failed", attempt=str(other))
    put(other / "report.json", record)
    (other / "input.xml").write_bytes((old / "input.xml").read_bytes())
    nested.topology._seal(other)
    with pytest.raises(ValueError, match="Multiple terminal"):
        infer(s)
    assert len(s.calls) == 2


def test_controller_resume_counts_saved_success_failure_and_runs_only_missing(two_million, monkeypatch):
    s = two_million
    common.configure_profile("2m")
    monkeypatch.setattr(runner, "nested10m", nested)
    monkeypatch.setattr(runner.benchmark, "numeric_gate", lambda *_: {"passes_observed_range": True})
    try:
        cases = s.cases[:2]
        first, second = cases[0]["seeds"]
        nested.inference_task(s.backend, s.output, cases[0], first["seed"], threading.Event())
        s.failures[(cases[0]["case_id"], "pilot", second["seed"])] = "failed"
        nested.inference_task(s.backend, s.output, cases[0], second["seed"], threading.Event())
        saved = snapshot(s.output)
        assert len(s.calls) == 4
        outcomes = runner.run_inference(s.backend, s.output, cases, parallel=2, reused=[])
        assert len(outcomes) == 4
        assert [r["status"] for r in outcomes].count("failure") == 1
        assert len(s.calls) == 8 and all(cid == cases[1]["case_id"] for cid, *_ in s.calls[4:])
        assert all(nested.pilot.sha256(s.output / path) == digest for path, digest in saved.items())
        assert runner.run_inference(s.backend, s.output, cases, parallel=1, reused=[]) == outcomes
        assert len(s.calls) == 8
    finally:
        common.configure_profile("10m")


def test_two_million_summary_rejects_wrong_budget_before_writing(two_million):
    s = two_million
    outcome = infer(s)
    outcome["run"]["config"]["chain_length"] = 10_000_000
    with pytest.raises(ValueError, match="2M"):
        nested.summarize_external(s.backend, s.bank, s.output, s.cases[0], 1, outcome)
    assert not (s.output / "summaries").exists()


def contracts():
    protected = {str(Path(runner.__file__).resolve().with_name(name)): digest
                 for name, digest in runner._PRE_RESUME_REPAIR.items()}
    old = {"rules": {"unchanged": True}, "protected_sha256": protected, "runtime": "frozen"}
    new = copy.deepcopy(old)
    new["protected_sha256"] = {path: nested.pilot.sha256(Path(path)) for path in protected}
    return old, new


def test_repair_accepts_only_exact_audited_predecessor_hashes():
    old, new = contracts()
    saved = copy.deepcopy(old)
    repair = runner.resume_repair(old, new)
    assert old == saved and len(repair["source_replacements"]) == 2
    assert runner.resume_repair(new, new) is None


@pytest.mark.parametrize("change", ["unknown_predecessor", "foreign_source", "source_set", "rules", "runtime"])
def test_repair_rejects_scientific_or_unknown_changes(change):
    old, new = contracts()
    if change == "unknown_predecessor":
        old["protected_sha256"][next(iter(old["protected_sha256"]))] = "a" * 64
    elif change == "foreign_source":
        old["protected_sha256"]["/elsewhere/source.py"] = "a" * 64
        new["protected_sha256"]["/elsewhere/source.py"] = "b" * 64
    elif change == "source_set":
        new["protected_sha256"]["/new/module.py"] = "b" * 64
    else:
        new[change] = "changed"
    with pytest.raises(ValueError):
        runner.resume_repair(old, new)


def test_prepare_preserves_old_contract_and_adds_exact_repair_record(tmp_path, monkeypatch):
    common.configure_profile("2m")
    try:
        old, current = contracts()
        data = {**case(), "records": [], "seeds": common.seeds_for(case()["case_id"]),
                "input_summary": {}, "eligibility": "compatible"}
        monkeypatch.setattr(common, "prepare_sources", lambda *_a, **_k: (
            [case()], [data], [], {"reused_outcomes": []}, None, current["protected_sha256"], []))
        args = SimpleNamespace(bank_root=tmp_path / "bank", trimmed_root=tmp_path / "trim",
            nested_root=tmp_path / "old", bio_root=tmp_path / "bio", output_root=tmp_path / "new",
            budget_root=tmp_path / "budget", resume=True)
        output, contract, *_ = runner.prepare(args)
        # Synthetic pre-repair contract with the exact audited predecessor pins.
        original = copy.deepcopy(contract)
        original["protected_sha256"] = old["protected_sha256"]
        runner.write_json_atomic(output / "analysis_contract.json", original)
        before = snapshot(output)
        _, resumed, *_ = runner.prepare(args)
        assert resumed == contract
        assert all(nested.pilot.sha256(output / path) == digest for path, digest in before.items())
        repair = nested.pilot.read_json(output / runner._RESUME_REPAIR_FILE)
        assert repair["original_contract_sha256"] == before["analysis_contract.json"]
        saved = snapshot(output)
        assert runner.prepare(args)[1] == contract and snapshot(output) == saved
        repair["source_replacements"][next(iter(repair["source_replacements"]))]["after"] = "changed"
        runner.write_json_atomic(output / runner._RESUME_REPAIR_FILE, repair)
        with pytest.raises(ValueError, match="recorded resume repair differs"):
            runner.prepare(args)
    finally:
        common.configure_profile("10m")


def test_completion_binds_2m_pilot_seal_and_repair_record(tmp_path, monkeypatch):
    common.configure_profile("2m")
    try:
        data = {**case(), "seeds": common.seeds_for(case()["case_id"])}
        contract = {"nested_cases": [data], "protected_sha256": {}}
        put(tmp_path / "analysis_contract.json", contract)
        put(tmp_path / runner._RESUME_REPAIR_FILE, {"fixture": True})
        directory = tmp_path / "inference" / data["case_id"] / f"seed_{data['seeds'][0]['seed']}" / "pilot" / "attempt_001"
        put(directory / "report.json", {"fixture_terminal_failure": True})
        nested.topology._seal(directory)
        monkeypatch.setattr(runner.reporting, "build_report", lambda *_a, **_k: {})
        monkeypatch.setattr(runner.reporting, "write_report",
                            lambda target, _rows, report: put(target / "report.json", report))
        report = runner.finish_report(tmp_path, contract, [], [], [], {})
        marker = nested.pilot.read_json(tmp_path / "_COMPLETE.json")
        relative = str((directory / "_SEALED.json").relative_to(tmp_path))
        assert relative in marker["artifacts"]
        assert runner._RESUME_REPAIR_FILE in marker["artifacts"]
        assert runner.completed(tmp_path, contract) == report
        (directory / "_SEALED.json").write_text("changed")
        with pytest.raises(ValueError, match="Changed source"):
            runner.completed(tmp_path, contract)
    finally:
        common.configure_profile("10m")


def _spawn_external_failure_task(bio, bank, output, data, replicate, outcome, profile):
    """Real fresh worker, mocked backend load only. No Java or external files."""
    assert common._PROFILE == "10m"
    runner.pilot.load_backend = lambda _: None
    return runner.external_score_task(bio, bank, output, data, replicate, outcome, profile)


@pytest.mark.parametrize("profile,iterations,samples", [("2m", 2_000_000, 2001), ("10m", 10_000_000, 10001)])
def test_real_spawn_summary_receives_selected_budget(tmp_path, profile, iterations, samples):
    common.configure_profile(profile)
    try:
        data = case()
        data.update(eligibility="compatible", input_sha256="a" * 64)
        seed = common.seeds_for(data["case_id"])[0]
        outcome = {"case_id": data["case_id"], "seed": seed["seed"], "status": "failure",
                   "stage": "pilot" if profile == "2m" else "long", "reason": "failed",
                   "provenance": common.nested10m.NEW_PROVENANCE, "attempt": "/unused/failed"}
        with runner.FreshSpawnTaskPool(1) as pool:
            bundle, = pool.map_ordered(_spawn_external_failure_task,
                [("/unused/bio", "/unused/bank", str(tmp_path), data, seed["replicate"], outcome, profile)],
                timeout_seconds=60)
        assert bundle["score"]["inference_budget_iterations"] == iterations
        assert bundle["external"]["inference_budget_iterations"] == iterations
        assert bundle["external"]["summary_rules"]["total_samples"] == samples
        assert bundle["score"]["status"] == "failure"
        # Saving, reading and report-only reuse keep exactly the same outcome.
        path = runner.score_path(tmp_path, data["case_id"], "NestedBD-Long", seed["seed"])
        common.bundle_write(path, bundle)
        assert runner.cached_score(path, data, "NestedBD-Long", seed["seed"], seed["replicate"]) == bundle
        assert common.nested10m.summarize_external(None, Path("/unused"), tmp_path,
                                                   data, seed["replicate"], outcome) == bundle["external"]
    finally:
        common.configure_profile("10m")


def test_summary_dispatch_passes_profile_instead_of_relying_on_parent_globals(tmp_path, monkeypatch):
    common.configure_profile("2m")
    try:
        data = case()
        seed = common.seeds_for(data["case_id"])[0]
        data["seeds"] = [seed]
        outcome = {"case_id": data["case_id"], "seed": seed["seed"], "status": "failure"}
        tasks = []

        class Pool:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def map_ordered(self, function, arguments, **kwargs):
                tasks.extend(arguments)
                return [{"score": {"status": "failure"}}]

        monkeypatch.setattr(runner, "FreshSpawnTaskPool", Pool)
        runner.summarize_new(tmp_path, {"bio_root": "/unused", "bank_root": "/unused"},
                             [data], [outcome], workers=1)
        assert tasks[0][-1] == "2m"
    finally:
        common.configure_profile("10m")
