from __future__ import annotations

import copy
import csv
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import pytest

from algorithm_evaluation import v5_partial_comparator_extension as run
from algorithm_evaluation import v5_partial_comparator_extension_common as c
from algorithm_evaluation import v5_partial_comparator_extension_report as report
from algorithm_evaluation.paper_pipeline_runner import serialize_tree
from evaluation_contract import evaluate_tree_pair_result
from test_v5_nestedbd_long_pilot import payload


def put(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    c.pilot.write_json(path, data)


def case(block=10, condition="H14_rule_y"):
    height = int(condition.split("_")[0][1:])
    return {"case_id": c.confirmation.confirmation_case_id(block, condition),
            "block_index": block, "condition_id": condition, "condition_family": "standard",
            "condition_ordinal": c.CONDITIONS.index(condition), "height": height,
            "generations": [9, 12, 14] if height == 14 else [15, 20, 24],
            "status": "available", "input_path": "input.json", "metadata_path": "metadata.json",
            "truth_path": "truth.json", "distance_path": "distance.json", "failure": None}


def score(block, condition, method, value=.4, *, rep=None):
    data = case(block, condition)
    row = run.base_score(data, method, replicate=rep,
                         seed=c.benchmark.seed_for(data["case_id"], rep) if rep else None)
    row.update(status="success", source_status="success", grf_full=value, grf_trim=value + .1, ad_f1=.5)
    return row


@pytest.fixture
def tiny_scope(monkeypatch):
    monkeypatch.setattr(c, "ALL_BLOCKS", (0, 1, 2))
    monkeypatch.setattr(c, "REUSED_BLOCKS", (0,))
    monkeypatch.setattr(c, "NEW_BLOCKS", (1, 2))


def all_scores(*, include_new=True):
    rows = [score(b, cond, m, .4 if m == "M2" else .3)
            for b in c.ALL_BLOCKS for cond in c.CONDITIONS
            for m in (("M0", "M2", "M4") if include_new else ("M0", "M2"))]
    rows += [score(b, cond, "MEDICC2", .25) for b in c.ALL_BLOCKS for cond in c.confirmation.STANDARD_CONDITION_IDS]
    rows += [score(b, cond, "NestedBD-Long", .1 if rep == 1 else .5, rep=rep)
             for b in (c.ALL_BLOCKS if include_new else ())
             for cond in c.NESTED_CONDITIONS for rep in (1, 2)]
    return rows


def test_selection_all_800_and_uniform_400_seed_scope_with_eight_reused():
    declarations = [case(b, cond) for b in c.ALL_BLOCKS for cond in c.CONDITIONS]
    m4, nested = c.select_cases({"condition_inventory": declarations, "cases": declarations})
    assert len(m4) == 800 and len(nested) == 200
    assert nested[0]["case_id"] == "conf-b001-H14_rule_y"
    assert nested[-1]["case_id"] == "conf-b100-H24_rule_y"
    assert sum(r["block_index"] in c.REUSED_BLOCKS for r in nested)*2 == 8
    assert sum(r["block_index"] in c.NEW_BLOCKS for r in nested)*2 == 392
    assert {r["height"] for r in nested} == {14, 24}
    assert c.seeds_for("conf-b001-H14_rule_y") == [
        {"replicate": 1, "seed": 1946510383}, {"replicate": 2, "seed": 974924161}]
    m4_check, nested_check = c.select_cases({"condition_inventory": declarations, "cases": declarations}, check=True)
    assert m4_check == nested_check == nested[4:5]
    assert nested_check[0]["case_id"] == "conf-b003-H14_rule_y"


@pytest.mark.parametrize("alter", ["missing", "duplicate", "coordinate"])
def test_selection_rejects_changed_inventory(tiny_scope, alter):
    declarations = [case(b, cond) for b in c.ALL_BLOCKS for cond in c.CONDITIONS]
    if alter == "missing":
        declarations.pop()
    elif alter == "duplicate":
        declarations.append(declarations[0])
    else:
        declarations[0]["block_index"] = 55
    with pytest.raises(ValueError):
        c.select_cases({"condition_inventory": declarations, "cases": declarations})


def test_model_and_algorithm_are_existing_frozen_ones():
    spec = c.ARM_SPEC_BY_ID[c.M4_ARM]
    assert spec.problem == "partial"
    assert spec.algorithm_name == "neighbor_joining_classical"
    assert spec.biopsy_preset == "deferred_tie"
    assert spec.radius is None
    assert spec.radius_policy == "adaptive_median_prior_nn_q50_nearest_rank_min1"
    assert c.RULES["nested_inference"] == c.nested10m.budget.SETTINGS
    assert c.RULES["nested_inference"]["long_iterations"] == 10000000
    assert c.calibration.SETTINGS["pilot_iterations"] == 2000000
    assert c.topology.RULES["total_samples"] == 2001
    assert c.RULES["nested_summary"]["total_samples"] == 10001
    assert c.RULES["nested_summary"]["discarded_samples"] == 1000
    assert c.RULES["nested_summary"]["retained_samples"] == 9001
    changed = {"total_samples", "discarded_samples", "retained_samples", "treeannotator_timeout_seconds"}
    assert all(c.RULES["nested_summary"][key] == c.topology.RULES[key] for key in c.SUMMARY_KEYS if key not in changed)
    assert "cases" not in c.RULES["nested_summary"]
    assert c.RULES["nested_clock"] == c.benchmark.CLOCK
    assert c.M4_ARM not in c.confirmation.PAPER_ARM_IDS


def test_two_million_profile_is_separate_and_fresh():
    c.configure_profile("2m")
    try:
        assert c.SCHEMA == "ctbf-v5-partial-comparator-extension-2m-v1"
        assert c.nested10m.LABEL == "2M"
        assert c.nested10m.ITERATIONS == 2_000_000
        assert c.nested10m.SUMMARY["total_samples"] == 2001
        assert c.nested10m.SUMMARY["discarded_samples"] == 200
        assert c.nested10m.SUMMARY["retained_samples"] == 1801
        assert c.REUSED_BLOCKS == () and c.NEW_BLOCKS == c.ALL_BLOCKS
        assert c.RULES["nested_2m_scores_included"] is True
    finally:
        c.configure_profile("10m")


def test_nested_loader_preserves_all_bins_high_cn_and_recurrence(tmp_path):
    data = case()
    raw = payload(data["case_id"], data["generations"])
    raw["condition_id"] = data["condition_id"]
    # Change the same state consistently at every occurrence.
    target = raw["levels"][0]["states"][0]["state_label"]
    for level in raw["levels"]:
        for state in level["states"]:
            if state["state_label"] == target:
                state["cnp"][0] = 10
    put(tmp_path / "input.json", raw)
    source = SimpleNamespace(ref=lambda _: {"path": str(tmp_path / "input.json"), "sha256": c.pilot.sha256(tmp_path / "input.json")})
    loaded = c.nested_case(source, data)
    assert loaded["eligibility"] == "blocked_copy_number_range"
    assert len(loaded["records"]) == sum(len(level["states"]) for level in raw["levels"])
    assert all(len(row["cnp"]) == 100 for row in loaded["records"])
    assert max(max(row["cnp"]) for row in loaded["records"]) == 10
    assert c.pilot.read_json(tmp_path / "input.json") == raw


def test_unavailable_nested_case_remains_in_inventory():
    source_case = case()
    source_case["status"] = "unavailable"
    loaded = c.nested_case(SimpleNamespace(), source_case)
    assert loaded["eligibility"] == "bank_unavailable"
    assert len(loaded["seeds"]) == 2 and loaded["records"] == []


def test_score_bundle_immutable_and_tamper_detected(tmp_path):
    path = tmp_path / "nested/result.json"
    c.bundle_write(path, {"value": 42})
    assert c.bundle_read(path) == {"value": 42}
    with pytest.raises(FileExistsError):
        c.bundle_write(path, {"value": 42})
    changed = c.pilot.read_json(path)
    changed["payload"]["value"] = 43
    run.write_json_atomic(path, changed)
    with pytest.raises(ValueError, match="Changed extension bundle"):
        c.bundle_read(path)


@pytest.mark.parametrize("kind", ["bank", "bio", "old", "broad"])
def test_output_cannot_overlap_sources(tmp_path, kind):
    bank, bio, old = (tmp_path / name for name in ("bank", "bio", "old"))
    output = {"bank": bank / "output", "bio": bio, "old": old / "nested", "broad": tmp_path}[kind]
    with pytest.raises(ValueError):
        c.output_guard(output, bank, bio, [old])


def test_roster_and_both_seed_aggregation_no_independent_seed_inflation(tiny_scope):
    rows = all_scores()
    saved = copy.deepcopy(rows)
    result = report.build_report(rows, bootstrap_draws=100)
    assert rows == saved
    nested = [row for row in result["case_scores"] if row["method"] == "NestedBD-Long"]
    assert len(nested) == 6
    assert all(row["grf_full"] == pytest.approx(.3) and row["replicate_count"] == 2 for row in nested)
    combined = next(row for row in result["paired_contrasts"] if row["left"] == "M2" and row["right"] == "NestedBD-Long" and len(row["conditions"]) == 2)
    assert combined["complete_histories"] == 3
    assert combined["condition_pair_count"] == 6
    assert combined["effects"]["grf_full"]["mean"] == pytest.approx(.1)
    assert combined["effects"]["grf_full"]["wtl"]["wins"] == 3
    assert combined["condition_wtl"]["grf_full"]["wins"] == 6
    assert all(r["complete_histories"] == 3 for r in result["five_method_matched_scores"])
    assert {r["method"] for r in result["sampling_contrasts"]} == {"M0", "M2", "M4"}
    assert result["convergence_established"] is False and result["p_values"] is False


def test_one_failed_seed_excludes_input_not_choose_best_seed(tiny_scope):
    rows = all_scores()
    failed = next(row for row in rows if row["method"] == "NestedBD-Long" and row["block_index"] == 1 and row["height"] == 24 and row["replicate"] == 1)
    failed.update(status="failure", source_status="failure", grf_full=None, grf_trim=None)
    result = report.build_report(rows, bootstrap_draws=100)
    combined = next(row for row in result["paired_contrasts"] if row["left"] == "M2" and row["right"] == "NestedBD-Long" and len(row["conditions"]) == 2)
    assert combined["declared_histories"] == 3 and combined["complete_histories"] == 2
    assert combined["condition_pair_count"] == 5
    matched = {r["condition"]: r["complete_histories"] for r in result["five_method_matched_scores"]}
    assert matched == {"H14_rule_y": 3, "H24_rule_y": 2}


@pytest.mark.parametrize("change", ["drop_seed", "duplicate", "height", "nan", "foreign_method", "budget"])
def test_report_refuses_broken_roster(tiny_scope, change):
    rows = all_scores()
    if change == "drop_seed":
        rows.pop()
    elif change == "duplicate":
        rows.append(rows[0])
    elif change == "height":
        rows[0]["height"] = 38
    elif change == "nan":
        rows[0]["grf_full"] = float("nan")
    elif change == "foreign_method":
        rows[0]["method"] = "full_method"
    else:
        next(r for r in rows if r["method"] == "NestedBD-Long")["inference_budget_iterations"] = 2000000
    with pytest.raises(ValueError):
        report.build_report(rows, bootstrap_draws=10)


def test_report_writes_old_values_verbatim_and_both_wtl_counts(tmp_path, tiny_scope):
    rows = all_scores()
    result = report.build_report(rows, bootstrap_draws=100)
    report.write_report(tmp_path, rows, result)
    with (tmp_path / "scores.csv").open() as handle:
        saved = list(csv.DictReader(handle))
    assert [float(row["grf_full"]) for row in saved] == [row["grf_full"] for row in rows]
    text = (tmp_path / "report.md").read_text()
    assert "Full history W/T/L" in text and "Full condition W/T/L" in text
    assert "M4 is new on all 800 conditions" in text
    assert "same 10M budget" in text and "9,001" in text
    assert "No new inference" not in text


class InlinePool:
    """Test orchestration only, never represented as resource evidence."""
    def __init__(self, workers, **_):
        self.workers = workers
    def __enter__(self):
        return self
    def __exit__(self, *_):
        pass
    def map_ordered(self, fn, tasks, **_):
        return (fn(*args) for args in tasks)


def test_m4_uses_unchanged_executor_and_independent_trim(monkeypatch):
    graph = nx.DiGraph()
    graph.add_node(0, cell_id="a")
    graph.add_node(1, cell_id="b")
    graph.add_edge(0, 1)
    native = evaluate_tree_pair_result(graph, graph, ["a", "b"])
    record = {"status": "success", "native_evaluation": native, "tree": serialize_tree(graph)}
    calls = []
    def execute(bank, data, spec, reference, reconstruct_time, evaluate_time, rss):
        calls.append((bank, data, spec.arm_id, reference, reconstruct_time, evaluate_time, rss))
        return record
    monkeypatch.setattr(run.frozen, "_execute_record", execute)
    monkeypatch.setattr(run.frozen, "_load_truth_after_reconstruction", lambda *_: graph)
    monkeypatch.setattr(run, "measured_stage", lambda fn, **_: (fn(), {"memory": {"peak_rss_bytes": 10}}, None))
    result = run.m4_task("bank", case())
    assert calls[0][2:] == (c.M4_ARM, None, 300, 300, 4 * 1024**3)
    assert result["record"] is record
    assert result["score"]["status"] == "success"
    assert result["score"]["grf_full"] == native["metrics"]["grf"]
    assert result["trimmed"]["ad_relation_set_invariant"] is True
    assert native == evaluate_tree_pair_result(graph, graph, ["a", "b"])


def test_failed_reconstruction_never_reads_truth(monkeypatch):
    monkeypatch.setattr(run.frozen, "_execute_record", lambda *_: {"status": "failure", "failure": {"stage": "reconstruction"}})
    monkeypatch.setattr(run.frozen, "_load_truth_after_reconstruction", lambda *_: pytest.fail("truth must not be loaded"))
    result = run.m4_task("bank", case())
    assert result["score"]["status"] == "failure" and "trimmed" not in result


def test_m4_resume_reuses_success_and_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(run, "FreshSpawnTaskPool", InlinePool)
    calls = []
    def task(_bank, data):
        calls.append(data["case_id"])
        row = score(data["block_index"], data["condition_id"], "M4")
        if data["block_index"] == 11:
            row.update(status="failure", source_status="failure", grf_full=None, grf_trim=None)
        return {"score": row}
    monkeypatch.setattr(run, "m4_task", task)
    contract = {"m4_cases": [case(10), case(11)], "bank_root": "bank"}
    first = run.run_m4(tmp_path, contract, workers=4)
    second = run.run_m4(tmp_path, contract, workers=1)
    assert first == second and len(calls) == 2
    assert [r["status"] for r in second] == ["success", "failure"]


def test_cached_score_rejects_wrong_seed_or_case(tmp_path):
    data = case()
    path = tmp_path / "score.json"
    c.bundle_write(path, {"score": score(10, "H14_rule_y", "NestedBD-Long", rep=1)})
    with pytest.raises(ValueError, match="identity"):
        run.cached_score(path, data, "NestedBD-Long", seed=99, replicate=1)


def test_inference_handles_all_terminal_outcomes_and_only_new_scope(tmp_path, monkeypatch):
    cases = [{**case(b), "seeds": c.seeds_for(case(b)["case_id"])} for b in (10, 11)]
    calls = []
    monkeypatch.setattr(run.benchmark, "numeric_gate", lambda *_: {"passes_observed_range": True})
    def infer(_backend, _root, data, seed, _stop):
        calls.append((data["case_id"], seed))
        return {"case_id": data["case_id"], "seed": seed, "status": "failure" if data["block_index"] == 10 else "success"}
    monkeypatch.setattr(run.nested10m, "inference_task", infer)
    outcomes = run.run_inference(None, tmp_path, cases, parallel=2, reused=[])
    assert len(outcomes) == len(calls) == 4
    assert sum(r["status"] == "failure" for r in outcomes) == 2
    assert all("b011" in cid or "b012" in cid for cid, _ in calls)


def test_inference_interruption_does_not_create_final_report(tmp_path, monkeypatch):
    data = {**case(), "seeds": c.seeds_for(case()["case_id"])}
    monkeypatch.setattr(run.benchmark, "numeric_gate", lambda *_: {"passes_observed_range": True})
    def interrupt(_backend, _root, _case, _seed, stop):
        stop.set()
        return None
    monkeypatch.setattr(run.nested10m, "inference_task", interrupt)
    assert run.run_inference(None, tmp_path, [data], parallel=1, reused=[]) is None
    assert not (tmp_path / "_COMPLETE.json").exists()


def test_prepare_resume_operational_concurrency_can_change(tmp_path, monkeypatch):
    data = {**case(), "records": [], "seeds": c.seeds_for(case()["case_id"]),
            "input_summary": {}, "eligibility": "compatible"}
    monkeypatch.setattr(c, "prepare_sources", lambda *_a, **_k: ([case()], [data], [], {"reused_outcomes": []}, None, {}, []))
    args = SimpleNamespace(bank_root=tmp_path / "bank", trimmed_root=tmp_path / "trim", nested_root=tmp_path / "old",
                           bio_root=tmp_path / "bio", output_root=tmp_path / "new", budget_root=tmp_path / "budget",
                           resume=True, workers=4, parallel_runs=2)
    first = run.prepare(args)
    args.workers, args.parallel_runs = 1, 1
    second = run.prepare(args)
    assert first[1] == second[1]
    with pytest.raises(ValueError, match="mode"):
        run.prepare(args, check=True)


def test_report_refuses_missing_jobs_without_inference(tmp_path, monkeypatch):
    put(tmp_path / "analysis_contract.json", {})
    contract = {"m4_cases": [case()]}
    monkeypatch.setattr(run, "prepare", lambda *_a, **_k: (tmp_path, contract, [], [], {}, None))
    monkeypatch.setattr(run, "completed", lambda *_: None)
    monkeypatch.setattr(run, "run_inference", lambda *_a, **_k: pytest.fail("report cannot infer"))
    with pytest.raises(ValueError, match="M4 records are unfinished"):
        run.main(["report", "--bio-root", str(tmp_path / "bio"), "--output-root", str(tmp_path), "--resume"])


def test_check_requires_separate_root():
    with pytest.raises(SystemExit):
        run.main(["check", "--bio-root", "/unused"])


def test_complete_additive_run_resumes_without_reconstruction_or_mcmc(tmp_path, monkeypatch, tiny_scope):
    declarations = [case(b, cond) for b in c.ALL_BLOCKS for cond in c.CONDITIONS]
    new = [{**case(b, cond), "records": [], "seeds": c.seeds_for(case(b, cond)["case_id"]),
            "input_summary": {}, "eligibility": "compatible"} for b in c.ALL_BLOCKS for cond in c.NESTED_CONDITIONS]
    original = all_scores(include_new=False)
    original_saved = copy.deepcopy(original)
    reused = [{"case_id": data["case_id"], "seed": seed["seed"], "status": "success",
               "provenance": run.nested10m.REUSED_PROVENANCE}
              for data in new if data["block_index"] in c.REUSED_BLOCKS for seed in data["seeds"]]
    monkeypatch.setattr(c, "prepare_sources", lambda *_a, **_k: (
        declarations, new, original, {"reused_outcomes": reused}, None, {}, []))
    monkeypatch.setitem(c.RULES, "bootstrap_draws", 100)
    monkeypatch.setattr(run, "FreshSpawnTaskPool", InlinePool)
    calls = {"m4": 0, "nested": 0}
    def m4(_bank, data):
        calls["m4"] += 1
        return {"score": score(data["block_index"], data["condition_id"], "M4", .3)}
    def inference(_backend, _root, cases, *, reused, **_):
        calls["nested"] += len(cases)*2-len(reused)
        return reused + [{"case_id": data["case_id"], "seed": s["seed"], "status": "success",
                          "provenance": run.nested10m.NEW_PROVENANCE}
                         for data in cases if data["block_index"] in c.NEW_BLOCKS for s in data["seeds"]]
    def summarize(_bio, _bank, root, data, replicate, outcome, budget_profile):
        assert budget_profile == "10m"
        directory = Path(root) / "summaries" / data["case_id"] / f"seed_{outcome['seed']}" / "attempt_001"
        external = {"attempt": str(directory), "status": "success"}
        put(directory / "result.json", external)
        c.topology._seal(directory)
        scored = score(data["block_index"], data["condition_id"], "NestedBD-Long", .3, rep=replicate)
        scored["provenance"] = outcome["provenance"]
        return {"score": scored, "external": external}
    monkeypatch.setattr(run, "m4_task", m4)
    monkeypatch.setattr(run, "run_inference", inference)
    monkeypatch.setattr(run, "external_score_task", summarize)
    monkeypatch.setattr(run, "diagnostics", lambda *_: {"inference_outcomes": [], "convergence_established": False})
    output = tmp_path / "new"
    arguments = ["run", "--bio-root", str(tmp_path / "bio"), "--output-root", str(output), "--resume"]
    assert run.main(arguments) == 0
    assert calls == {"m4": 24, "nested": 8}
    assert original == original_saved
    result = c.pilot.read_json(output / "reports/report_001/report.json")
    assert result["raw_record_count"] == 93
    assert result["nested_new_seed_records"] == 8 and result["nested_reused_seed_records"] == 4
    assert run.main(arguments) == 0
    assert calls == {"m4": 24, "nested": 8}
    marker = c.pilot.read_json(output / "_COMPLETE.json")
    changed = output / next(iter(marker["artifacts"]))
    changed.write_text("tampered")
    with pytest.raises(ValueError, match="Changed source"):
        run.main(arguments)
