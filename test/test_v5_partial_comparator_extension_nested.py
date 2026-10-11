from __future__ import annotations

import copy
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from algorithm_evaluation import v5_partial_comparator_extension_nested as n
from algorithm_evaluation import v5_partial_comparator_extension as runner
from test_v5_nestedbd_long_budget import fixture_budget, stats, windows
from test_v5_nestedbd_long_topology_comparison import nexus, records
from test_v5_partial_comparator_extension import put


@pytest.fixture
def reusable(fixture_budget):
    s = fixture_budget
    root = s.output / "reusable"
    root.mkdir(parents=True)
    (root / ".runner.lock").touch()
    runs, artifacts = [], {}
    for case in s.cases:
        for seed in case["seeds"]:
            directory = root / "inference" / case["case_id"] / f"seed_{seed['seed']}" / "long" / "attempt_001"
            directory.mkdir(parents=True)
            _, xml, _ = n.budget.make_xml(s.backend, case, "long")
            xml.write(directory / "input.xml")
            run = {**n.budget.run_identity(case, "long", seed["seed"]),
                   "status": "technical_complete", "attempt": str(directory),
                   "scalar_windows": stats(70), "execution": {"fixture_not_resource_evidence": True}}
            put(directory / "report.json", run)
            put(directory / "clade_windows.json", windows(9001))
            (directory / "shared_pilot.log").write_text("fixture scalar stream\n")
            (directory / "shared_pilot.trees").write_text("fixture tree stream\n")
            n.topology._seal(directory)
            artifacts.update({str(p): n.pilot.sha256(p) for p in directory.iterdir() if p.is_file()})
            runs.append(run)
    contract = {"rules": n.budget.RULES, "source_root": str(s.recovered), "bio_root": str(s.bio),
                "runtime": n.budget.recovery.runtime_identity(), "protected_sha256": {},
                "cases": [{k: v for k, v in case.items() if k != "records"} for case in s.cases]}
    put(root / "analysis_contract.json", contract)
    put(root / "reports/report_001/report.json", {"new_runs": runs})
    put(root / "_COMPLETE.json", {"schema_version": n.budget.SCHEMA,
        "contract_sha256": n.pilot.sha256(root / "analysis_contract.json"),
        "report_directory": "reports/report_001",
        "report_sha256": n.pilot.sha256(root / "reports/report_001/report.json"), "artifacts": artifacts})
    return SimpleNamespace(**vars(s), root=root, runs=runs)


def test_reuses_exactly_eight_sealed_10m_runs_without_writes_or_inference(reusable, monkeypatch):
    s = reusable
    before = {p: n.pilot.sha256(p) for p in s.root.rglob("*") if p.is_file()}
    monkeypatch.setattr(n.budget, "execute_attempt", lambda *_: pytest.fail("reused inference must not run"))
    protected = {}
    outcomes = n.load_reused(s.root, s.recovered, s.bio, s.cases, s.backend, protected)
    assert len(outcomes) == 8
    assert all(r["status"] == "success" and r["provenance"] == n.REUSED_PROVENANCE for r in outcomes)
    assert all(r["run"]["config"]["chain_length"] == 10000000 for r in outcomes)
    assert {p: n.pilot.sha256(p) for p in before} == before
    n.pilot.verify_hashes(protected)


@pytest.mark.parametrize("changed", ["trace", "input", "selection", "missing_seed"])
def test_reuse_refuses_tampering_or_different_selection(reusable, changed):
    s = reusable
    if changed == "trace":
        (Path(s.runs[0]["attempt"]) / "shared_pilot.trees").write_text("changed")
    elif changed == "input":
        s.cases[0]["input_sha256"] = "a" * 64
    elif changed == "selection":
        s.cases.pop()
    else:
        path = s.root / "reports/report_001/report.json"
        runner.write_json_atomic(path, {"new_runs": s.runs[:-1]})
        marker = n.pilot.read_json(s.root / "_COMPLETE.json")
        marker["report_sha256"] = n.pilot.sha256(path)
        runner.write_json_atomic(s.root / "_COMPLETE.json", marker)
    with pytest.raises(ValueError):
        n.load_reused(s.root, s.recovered, s.bio, s.cases, s.backend, {})


@pytest.mark.parametrize("field", ["budget", "seed", "clock", "input", "burnin"])
def test_mixed_or_changed_run_rejected_before_summary(reusable, field, tmp_path):
    s = reusable
    case, run = s.cases[0], copy.deepcopy(s.runs[0])
    if field == "budget":
        run["config"]["chain_length"] = 2000000
    elif field == "burnin":
        run["config"]["burnin_percent"] = 20
    elif field == "seed":
        run["seed"] += 1
    elif field == "clock":
        run["clock"]["mean_clock_rate"] = 1
    else:
        run["input_sha256"] = "b" * 64
    output = tmp_path / "must_not_create"
    with pytest.raises(ValueError, match="10M"):
        n.summarize_external(s.backend, s.bank, output, case, 1,
                            {"seed": s.runs[0]["seed"], "status": "success", "run": run,
                             "provenance": n.REUSED_PROVENANCE})
    assert not output.exists()


def test_summary_uses_10001_samples_and_9001_retained(tmp_path):
    burned = "((1:0,2:1):2,3:3):0;"
    retained = "((1:1,3:1):2,2:3):0;"
    source = nexus(tmp_path / "source.trees", [burned]*1000 + [retained]*9001)
    summary = nexus(tmp_path / "summary.tree", [retained], names=["TREE1"])
    audit = n.validate_sampled_summary(source, summary, records())
    assert (audit["samples"], audit["discarded"], audit["retained"]) == (10001, 1000, 9001)
    assert audit["matching_retained_topologies"] == 9001
    assert audit["first_retained_state_with_summary_topology"] == "STATE_1000000"
    nexus(summary, [burned], names=["TREE1"])
    with pytest.raises(ValueError, match="absent"):
        n.validate_sampled_summary(source, summary, records())


def test_summary_refuses_a_2m_tree_stream(tmp_path):
    tree = "((1:0,2:1):2,3:3):0;"
    source = nexus(tmp_path / "short.trees", [tree]*2001)
    summary = nexus(tmp_path / "summary.tree", [tree], names=["TREE1"])
    with pytest.raises(ValueError, match="complete retained 10M"):
        n.validate_sampled_summary(source, summary, records())


def test_native_summary_command_retains_semantics_and_scales_timeout(tmp_path, monkeypatch):
    calls = []
    backend = SimpleNamespace(runtime=SimpleNamespace(RUNTIME=tmp_path, java_prefix=lambda _: ["java", "-Xmx2g"]))
    def execute(command, **kwargs):
        calls.append((command, kwargs))
        Path(command[-1]).write_text("fixture summary")
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(n.subprocess, "run", execute)
    n.native_summary(backend, tmp_path / "cold.trees", tmp_path)
    command, options = calls[0]
    assert command[2:7] == ["beast.app.treeannotator.TreeAnnotator", "-burnin", "10", "-heights", "keep"]
    assert options["timeout"] == 900
    assert "beast.app.BeastMCMC" not in command


def test_adapter_inference_retains_failures_and_exact_long_identity(fixture_budget):
    s = fixture_budget
    case = s.cases[0]
    seed = case["seeds"][0]["seed"]
    first = n.inference_task(s.backend, s.output, case, seed, threading.Event())
    assert first["status"] == "success" and first["run"]["config"]["chain_length"] == 10000000
    assert len(s.calls) == 2
    assert n.cached_inference(s.backend, s.output, case, seed) == first
    assert n.inference_task(s.backend, s.output, case, seed, threading.Event()) == first
    assert len(s.calls) == 2
    other = case["seeds"][1]["seed"]
    s.failures[(case["case_id"], "long", other)] = "timeout"
    failure = n.inference_task(s.backend, s.output, case, other, threading.Event())
    assert failure["status"] == "failure" and failure["stage"] == "long"
    calls = len(s.calls)
    assert n.inference_task(s.backend, s.output, case, other, threading.Event()) == failure
    assert len(s.calls) == calls


def test_report_only_never_starts_missing_10m(fixture_budget):
    s = fixture_budget
    case = s.cases[0]
    with pytest.raises(ValueError, match="unfinished"):
        n.cached_inference(s.backend, s.output, case, case["seeds"][0]["seed"])
    assert not s.calls


def test_orchestrator_schedules_only_missing_10m_scope(reusable, monkeypatch):
    s = reusable
    reused = [n.outcome_for(s.cases[0], r["seed"], r, reused=True) for r in s.runs[:2]]
    calls = []
    def infer(_backend, _root, case, seed, _stop):
        calls.append((case["case_id"], seed))
        return {"case_id": case["case_id"], "seed": seed, "status": "failure", "provenance": n.NEW_PROVENANCE}
    monkeypatch.setattr(runner.benchmark, "numeric_gate", lambda *_: {"passes_observed_range": True})
    monkeypatch.setattr(n, "inference_task", infer)
    outcomes = runner.run_inference(s.backend, s.output, s.cases, parallel=2, reused=reused)
    assert len(outcomes) == 8 and len(calls) == 6
    assert all(cid != s.cases[0]["case_id"] for cid, _ in calls)
    assert [r for r in outcomes if r["provenance"] == n.REUSED_PROVENANCE] == sorted(reused, key=lambda r: (r["case_id"], r["seed"]))
