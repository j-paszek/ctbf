from __future__ import annotations

import copy
from pathlib import Path

import networkx as nx
import pytest

from algorithm_evaluation import v5_nestedbd_long_benchmark as b
from algorithm_evaluation import v5_nestedbd_long_benchmark_report as r
from test_v5_nestedbd_long_benchmark import fixture_run, run


def evaluation():
    truth, reconstructed = nx.DiGraph([(0, 1), (0, 2)]), nx.DiGraph([(0, 1), (1, 2)])
    for tree in (truth, reconstructed):
        nx.set_node_attributes(tree, {index: str(index) for index in range(3)}, "cell_id")
    return b.evaluate_tree_pair_result(truth, reconstructed, ["0", "1", "2"])


def snapshot(root):
    return {str(path.relative_to(root)): b.pilot.sha256(path)
            for path in root.rglob("*") if path.is_file()}


def prohibit_inference(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("report recovery attempted inference, reconstruction or distance loading")
    monkeypatch.setattr(b, "inference_task", forbidden)
    monkeypatch.setattr(b.calibration, "execute_attempt", forbidden)
    monkeypatch.setattr(b.pilot, "numerical_control", forbidden)
    from algorithm_evaluation import v5_algorithm_development_common as common
    from algorithm_evaluation import v5_confirmation_run as confirmation
    monkeypatch.setattr(common, "reconstruct_development_arm", forbidden)
    monkeypatch.setattr(confirmation, "deserialize_confirmation_distance", forbidden)


def test_exact_and_roundoff_only_replay_preserve_both_records():
    stored = evaluation()
    assert r.validate_replay(stored, stored)["bitwise_exact"]
    replay = copy.deepcopy(stored)
    replay["metrics"]["grf"] += 3e-16
    replay["metrics"]["ext_grf"] -= 3e-16
    before = copy.deepcopy((stored, replay))
    audit = r.validate_replay(stored, replay)
    assert not audit["bitwise_exact"] and audit["stored_scores_retained"]
    assert set(audit["roundoff"]) == {"grf", "ext_grf"}
    assert (stored, replay) == before
    assert r.REPLAY_ABSOLUTE_TOLERANCE < b.RULES["win_tie_loss_tolerance"]


def test_score_difference_beyond_roundoff_is_rejected():
    stored = evaluation()
    replay = copy.deepcopy(stored)
    replay["metrics"]["grf"] += 1e-10
    replay["metrics"]["ext_grf"] -= 1e-10
    with pytest.raises(ValueError, match="beyond roundoff"):
        r.validate_replay(stored, replay)


@pytest.mark.parametrize("field", ["ad_f1", "ad_precision", "ad_recall", "ad_iou"])
def test_non_grf_float_fields_are_not_fuzzed(field):
    stored = evaluation()
    replay = copy.deepcopy(stored)
    replay["metrics"][field] += 3e-16
    with pytest.raises(ValueError):
        r.validate_replay(stored, replay)


def test_changed_labels_are_rejected_even_with_identical_scores():
    stored = evaluation()
    replay = copy.deepcopy(stored)
    replay["observation_labels"] = ["a", "b", "c"]
    with pytest.raises(ValueError, match="outside GRF"):
        r.validate_replay(stored, replay)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_replay_is_rejected(value):
    stored = evaluation()
    replay = copy.deepcopy(stored)
    replay["metrics"]["grf"] = value
    with pytest.raises(ValueError):
        r.validate_replay(stored, replay)


def test_all_cached_summaries_and_scores_reused_with_readonly_resume(fixture_run, monkeypatch):
    s = fixture_run
    original = run(s)
    before = snapshot(s.output)
    prohibit_inference(monkeypatch)
    monkeypatch.setattr(b.topology, "native_summary", lambda *_: pytest.fail("cached summary recomputed"))
    output = s.output.with_name("recovery")
    result = r.recover(s.output, s.bio, output, resume=True)
    assert result["mcmc_runs_started"] == 0
    assert result["reused_sealed_source_summaries"] == 8
    assert result["paired_effects"] == original["paired_effects"]
    assert result["external_status_counts"] == {"success": 8}
    def scores(report):
        return {(row["case_id"], row["method"], row["seed"]): row["grf"] for row in report["records"]}
    assert scores(result) == scores(original)
    recovered_before = snapshot(output)
    assert r.recover(s.output, s.bio, output, resume=True) == result
    assert snapshot(output) == recovered_before and snapshot(s.output) == before


def test_recovery_after_partial_report_roundoff_crash_has_no_new_inference(fixture_run, monkeypatch):
    s = fixture_run
    def crash_after_two_summaries(backend, bank, output, cases, outcomes, contract):
        first = cases[0]
        for seed in first["seeds"]:
            outcome = next(row for row in outcomes
                           if row["case_id"] == first["case_id"] and row["seed"] == seed["seed"])
            b.summarize_external(backend, bank, output, first, seed["replicate"], outcome)
        raise ValueError("frozen CTBF evaluator replay differs")
    monkeypatch.setattr(b, "finish_report", crash_after_two_summaries)
    with pytest.raises(ValueError, match="replay differs"):
        run(s)
    assert len(s.calls) == 16
    before = snapshot(s.output)
    original_evaluate = b.evaluation_for
    def roundoff(bank, case, tree):
        native = original_evaluate(bank, case, tree)
        native["metrics"]["grf"] += 3e-16
        native["metrics"]["ext_grf"] -= 3e-16
        return native
    monkeypatch.setattr(b, "evaluation_for", roundoff)
    prohibit_inference(monkeypatch)
    result = r.recover(s.output, s.bio, s.output.with_name("recovery"), resume=True)
    assert result["external_status_counts"] == {"success": 8}
    assert result["reused_sealed_source_summaries"] == 2
    assert result["replay_audit"]["roundoff_only_comparators"] == 8
    assert all(row["frozen_evaluator_replay_exact"] is False for row in result["records"]
               if row["method"] in b.topology.CTBF_ARMS)
    for row in result["records"]:
        if row["method"] in b.topology.CTBF_ARMS:
            assert row["grf"] == b.pilot.read_json(Path(row["source_record"]))["metrics"]["grf"]
    assert len(s.calls) == 16 and snapshot(s.output) == before


def test_check_validates_sources_and_comparators_without_creating_output(fixture_run, monkeypatch):
    s = fixture_run
    run(s)
    prohibit_inference(monkeypatch)
    monkeypatch.setattr(b.topology, "native_summary", lambda *_: pytest.fail("check ran MCC"))
    output = s.output.with_name("check-output")
    result = r.recover(s.output, s.bio, output, resume=True, check=True)
    assert result["saved_external_outcomes"] == 8
    assert result["replay_audit"]["successful_comparators_checked"] == 8
    assert not output.exists()


def test_missing_or_interrupted_inference_is_not_restarted(fixture_run, monkeypatch):
    s = fixture_run
    first = b.selected_ids()[0]
    s.failures[(first, "technical", b.seed_for(first, 1))] = "interrupted"
    assert run(s) is None
    before = snapshot(s.output)
    prohibit_inference(monkeypatch)
    output = s.output.with_name("recovery")
    with pytest.raises(ValueError, match="report-only recovery never starts inference"):
        r.recover(s.output, s.bio, output, resume=True)
    assert snapshot(s.output) == before and not output.exists()


def test_terminal_failure_is_preserved_without_retry(fixture_run, monkeypatch):
    s = fixture_run
    first = b.selected_ids()[0]
    s.failures[(first, "pilot", b.seed_for(first, 1))] = "timeout"
    run(s)
    prohibit_inference(monkeypatch)
    result = r.recover(s.output, s.bio, s.output.with_name("recovery"), resume=True)
    assert result["external_status_counts"] == {"failure": 1, "success": 7}
    assert all(row["missing_histories"] == 1 for row in result["paired_effects"]["history_paired"])


def test_tampered_inference_is_rejected(fixture_run):
    s = fixture_run
    run(s)
    first = b.selected_ids()[0]
    path = s.output / "inference" / first / f"seed_{b.seed_for(first, 1)}" / "pilot/attempt_001/shared_pilot.trees"
    path.write_text("changed raw output\n")
    s.backend.pilot.verify_seal = b.topology._verify_seal
    with pytest.raises(ValueError):
        r.recover(s.output, s.bio, s.output.with_name("recovery"), resume=True)


def test_recovery_output_must_not_overlap_source(fixture_run):
    s = fixture_run
    run(s)
    with pytest.raises(ValueError, match="overlap"):
        r.recover(s.output, s.bio, s.output / "recovery", resume=True)


def test_source_controller_lock_prevents_concurrent_read(tmp_path):
    with b.owner_lock(tmp_path):
        with pytest.raises(ValueError, match="still running"):
            with r.source_read_lock(tmp_path):
                pytest.fail("lock was ignored")
