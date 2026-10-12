from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from algorithm_evaluation import v5_confirmation_run as confirmation_run
from algorithm_evaluation.paper_pipeline_contract import (
    PaperContractError,
    read_json,
    write_json_atomic,
)
from algorithm_evaluation.paper_pipeline_runner import serialize_tree
from algorithm_evaluation.v5_algorithm_development_common import ARM_SPEC_BY_ID
from algorithm_evaluation.v5_confirmation_bank import DISTANCE_SCHEMA_VERSION
from algorithm_evaluation.v5_confirmation_common import (
    CONDITIONS,
    M0_ID,
    M2_ID,
    PAPER_ARM_IDS,
    derive_analysis_seed,
    derive_seed,
    hybrid_sample_size,
    read_owner_approval_lock,
    resolve_schedule,
    validate_manifest,
)
from algorithm_evaluation.v5_confirmation_report import (
    _one_sided_sign_flip,
    _primary_analysis,
)
from algorithm_evaluation.v5_confirmation_run import _execute_record
from distance_semantics import CNP2CNP_SEMANTICS_VERSION


def test_confirmation_candidate_and_owner_lock_are_structurally_valid():
    report = validate_manifest(require_approved=False)

    assert report["status"] == "valid_frozen_candidate"
    assert report["block_count"] == 100
    assert report["condition_count"] == 8
    assert report["arm_count"] == 7
    assert report["source_lock"]["status"] == "frozen"
    assert report["source_lock"]["file_count"] == 47
    assert report["owner_approval"]["status"] == "approved"

    approved = validate_manifest(require_approved=True)
    assert approved["status"] == "valid_approved"
    assert approved["manifest_sha256"] == report["manifest_sha256"]


def test_owner_approval_file_is_parsed_as_restricted_data(tmp_path):
    path = tmp_path / "approval.py"
    path.write_text("raise RuntimeError('must not execute')\n", encoding="utf-8")

    with pytest.raises(PaperContractError, match="only its docstring and five data fields"):
        read_owner_approval_lock(path)


def test_confirmation_schedules_and_seeds_are_fixed_and_distinct():
    schedules = {
        condition.condition_id: resolve_schedule(condition, 0)[0]
        for condition in CONDITIONS
    }

    assert schedules["H14_rule_y"] == (9, 12, 14)
    assert schedules["H24_rule_y"] == (15, 20, 24)
    assert schedules["H34_rule_y"] == (21, 28, 34)
    assert schedules["H24_L2"] == (15, 24)
    assert set(schedules["H24_random"][:2]) <= set(range(15, 24))
    assert schedules["H24_random"][-1] == 24
    assert resolve_schedule(CONDITIONS[6], 0) == resolve_schedule(CONDITIONS[6], 0)
    assert derive_seed("simulation", 0) != derive_seed("simulation", 1)
    assert derive_seed("sampling", 0, 24) != derive_seed("reconstruction", 0, 0)
    assert derive_analysis_seed("primary_bootstrap") == 6917218322129967350


@pytest.mark.parametrize(
    ("available", "expected"),
    [(0, 0), (1, 1), (5, 5), (6, 6), (7, 6), (11, 6), (12, 6), (13, 7)],
)
def test_confirmation_hybrid_sample_size(available, expected):
    assert hybrid_sample_size(available) == expected


def test_confirmation_portfolio_resolves_to_current_v5_specs():
    assert len(PAPER_ARM_IDS) == len(set(PAPER_ARM_IDS)) == 7
    assert all(arm_id in ARM_SPEC_BY_ID for arm_id in PAPER_ARM_IDS)
    assert ARM_SPEC_BY_ID[M0_ID].primary_metric == "grf"
    assert ARM_SPEC_BY_ID[M2_ID].problem == "partial"


def _successful_record(block_index, condition_id, arm_id, value):
    return {
        "status": "success",
        "metrics": {"grf": value},
        "block_index": block_index,
        "condition_id": condition_id,
        "arm_id": arm_id,
    }


def test_primary_confirmation_uses_one_effect_per_truth_block():
    manifest = validate_manifest(require_approved=False)["manifest"]
    index = {}
    for block_index in range(100):
        for condition_id in ("H14_rule_y", "H24_rule_y", "H34_rule_y"):
            index[(block_index, condition_id, M0_ID)] = _successful_record(
                block_index, condition_id, M0_ID, 0.4
            )
            index[(block_index, condition_id, M2_ID)] = _successful_record(
                block_index, condition_id, M2_ID, 0.5
            )

    result, rows = _primary_analysis(manifest, index, 100)

    assert len(rows) == 100
    assert result["complete_condition_pair_count"] == 300
    assert result["complete_block_count"] == 100
    assert result["block_effect_summary"]["mean"] == pytest.approx(0.1)
    assert result["block_wins_ties_losses"] == {
        "wins": 100,
        "ties": 0,
        "losses": 0,
        "tolerance": 1e-12,
    }
    assert result["registered_support"] is True


def test_one_sided_sign_flip_is_deterministic():
    values = [0.1, 0.2, -0.01, 0.3]
    assert _one_sided_sign_flip(
        values, repetitions=2000, seed=17
    ) == _one_sided_sign_flip(values, repetitions=2000, seed=17)


@pytest.mark.parametrize(
    "over_limit_stage", [None, "reconstruction", "evaluation"],
    ids=["within-limit", "reconstruction-rss", "evaluation-rss"],
)
def test_confirmation_record_reconstructs_before_loading_truth(
    tmp_path, monkeypatch, controlled_stage_rss, over_limit_stage,
):
    stage_peaks = [64 * 1024**2, 96 * 1024**2]
    if over_limit_stage is not None:
        stage_peaks[["reconstruction", "evaluation"].index(over_limit_stage)] = (
            1024**3 + 1
        )
    controlled_stage_rss(*stage_peaks)
    case_root = tmp_path / "case"
    case_root.mkdir()
    profiles = {0: [2, 2, 2], 1: [2, 2, 1], 2: [2, 1, 1]}
    payload = {
        "schema_version": "ctbf-v5-reconstruction-input-v1",
        "case_id": "conf-toy",
        "condition_id": "toy",
        "levels": [
            {
                "biopsy_level": 0,
                "generation": 1,
                "states": [
                    {"state_label": 0, "cnp": profiles[0]},
                    {"state_label": 1, "cnp": profiles[1]},
                ],
            },
            {
                "biopsy_level": 1,
                "generation": 2,
                "states": [
                    {"state_label": 1, "cnp": profiles[1]},
                    {"state_label": 2, "cnp": profiles[2]},
                ],
            },
        ],
    }
    matrix = np.asarray([[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=float)
    truth = nx.DiGraph()
    truth.add_node("a", cell_id=0, generation=0)
    truth.add_node("b", cell_id=1, generation=1)
    truth.add_node("c", cell_id=2, generation=2)
    truth.add_edges_from([("a", "b"), ("b", "c")])
    write_json_atomic(case_root / "input.json", payload)
    write_json_atomic(case_root / "metadata.json", {
        "case_id": "conf-toy",
        "reconstruction_seed": 17,
    })
    write_json_atomic(case_root / "distance.json", {
        "schema_version": DISTANCE_SCHEMA_VERSION,
        "block_index": 0,
        "semantics_version": CNP2CNP_SEMANTICS_VERSION,
        "ids": [0, 1, 2],
        "matrix": matrix.tolist(),
        "provenance": {"semantics_version": CNP2CNP_SEMANTICS_VERSION},
        "resources": {},
    })
    write_json_atomic(case_root / "truth.json", {
        "schema_version": "ctbf-v5-confirmatory-truth-v1",
        "block_index": 0,
        "tree": serialize_tree(truth),
    })
    case = {
        "case_id": "conf-toy",
        "block_index": 0,
        "condition_id": "toy",
        "condition_family": "test",
        "height": 2,
        "generations": [1, 2],
        "input_path": "case/input.json",
        "metadata_path": "case/metadata.json",
        "distance_path": "case/distance.json",
        "truth_path": "case/truth.json",
    }

    events = []
    reconstruct = confirmation_run.reconstruct_development_arm
    read_asset = confirmation_run.read_json

    def observed_reconstruction(*args, **kwargs):
        events.append("reconstruction_started")
        result = reconstruct(*args, **kwargs)
        events.append("reconstruction_finished")
        return result

    def observed_read(path):
        if Path(path) == case_root / "truth.json":
            assert events == ["reconstruction_started", "reconstruction_finished"]
            events.append("truth_read")
        return read_asset(path)

    monkeypatch.setattr(
        confirmation_run, "reconstruct_development_arm", observed_reconstruction
    )
    monkeypatch.setattr(confirmation_run, "read_json", observed_read)

    record = _execute_record(
        str(tmp_path),
        case,
        ARM_SPEC_BY_ID[M0_ID],
        None,
        30,
        30,
        1024**3,
    )

    expected_events = ["reconstruction_started", "reconstruction_finished"]
    if over_limit_stage != "reconstruction":
        expected_events.append("truth_read")
    assert events == expected_events, record.get("failure")

    if over_limit_stage is not None:
        assert record["status"] == "failure", record.get("failure")
        assert record["failure"]["stage"] == over_limit_stage
        assert record["failure"]["type"] == "StageResourceError"
        assert "exceeded the registered limit" in record["failure"]["message"]
        return

    assert record["status"] == "success", record.get("failure")
    assert record["observation_coverage"]["fraction"] == 1.0
    assert record["reconstruction_metadata"][
        "truth_loaded_only_after_reconstruction"
    ] is True
    assert record["tree"] is not None
