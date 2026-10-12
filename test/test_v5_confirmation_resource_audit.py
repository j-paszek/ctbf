from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from algorithm_evaluation.paper_pipeline_contract import write_json_atomic
from algorithm_evaluation.v5_algorithm_development_common import tree_summary
from algorithm_evaluation.v5_confirmation_bank import DISTANCE_SCHEMA_VERSION
from algorithm_evaluation.v5_confirmation_common import M2_ID, record_path
from algorithm_evaluation.v5_confirmation_resource_audit import (
    _audit_case,
    _pipeline_resources,
    build_resource_summary,
    condition_cells,
)
from ctbs import DistanceMatrix
from distance_semantics import CNP2CNP_SEMANTICS_VERSION


def _resource(seconds, rss):
    return {
        "wall_time_ns": int(seconds * 1e9),
        "memory": {
            "method": "test_process_tree_rss",
            "peak_rss_bytes": rss,
        },
    }


def _payload():
    return {
        "schema_version": "ctbf-v5-reconstruction-input-v1",
        "case_id": "resource-toy",
        "condition_id": "H14_rule_y",
        "levels": [
            {
                "biopsy_level": 0,
                "generation": 1,
                "states": [
                    {"state_label": 0, "cnp": [2, 2, 2]},
                    {"state_label": 1, "cnp": [2, 2, 1]},
                ],
            },
            {
                "biopsy_level": 1,
                "generation": 2,
                "states": [
                    {"state_label": 1, "cnp": [2, 2, 1]},
                    {"state_label": 2, "cnp": [2, 1, 1]},
                ],
            },
        ],
    }


def test_condition_cells_contain_only_requested_condition_profiles():
    cells = condition_cells(_payload(), [2, 0, 1])

    assert [cell.get_id() for cell in cells] == [2, 0, 1]
    assert [cell.get_cnp() for cell in cells] == ["2,1,1", "2,2,2", "2,2,1"]


def test_sequential_pipeline_resource_uses_time_sum_and_rss_maximum():
    result = _pipeline_resources(_resource(2.5, 80), _resource(0.5, 120))

    assert result["wall_time_ns"] == 3_000_000_000
    assert result["memory"]["peak_rss_bytes"] == 120


def test_resource_summary_aligns_condition_specific_m2_and_medicc2():
    audit_records = [
        {
            "case_id": "a",
            "status": "success",
            "resources": {
                "distance": _resource(2, 80),
                "reconstruction": _resource(1, 100),
                "pipeline": _resource(3, 100),
            },
        },
        {
            "case_id": "b",
            "status": "success",
            "resources": {
                "distance": _resource(4, 90),
                "reconstruction": _resource(2, 120),
                "pipeline": _resource(6, 120),
            },
        },
    ]
    medicc2 = {
        "a": {
            "status": "success",
            "resources": {"reconstruction": _resource(9, 200)},
        },
        "b": {
            "status": "success",
            "resources": {"reconstruction": _resource(12, 240)},
        },
    }

    result = build_resource_summary(audit_records, medicc2)

    assert result["m2_condition_specific_distance"]["wall_seconds"]["mean"] == 3
    assert result["m2_condition_specific_pipeline"]["wall_seconds"]["mean"] == 4.5
    assert result["m2_condition_specific_pipeline"]["peak_rss_bytes"]["maximum"] == 120
    assert result["medicc2_pipeline"]["wall_seconds"]["mean"] == 10.5
    assert result["medicc2_slower_condition_count"] == 2
    assert result["paired_medicc2_over_m2_wall_ratio"]["median"] == pytest.approx(2.5)


@pytest.mark.parametrize(
    "change, expected_stage, expected_type, expected_message",
    [
        (None, None, None, None),
        (
            "distance", "distance_equivalence", "ValueError",
            "differs from its frozen submatrix",
        ),
        (
            "topology", "reconstruction_equivalence", "ValueError",
            "topology summary changed",
        ),
        (
            "distance_rss", "distance_equivalence", "StageResourceError",
            "exceeded the registered limit",
        ),
        (
            "reconstruction_rss", "reconstruction_equivalence", "StageResourceError",
            "exceeded the registered limit",
        ),
    ],
    ids=[
        "equivalent", "changed-distance", "changed-topology",
        "distance-rss", "reconstruction-rss",
    ],
)
def test_audit_case_requires_exact_distance_and_topology_equivalence(
    tmp_path: Path,
    controlled_stage_rss,
    change,
    expected_stage,
    expected_type,
    expected_message,
):
    bank_root = tmp_path / "bank"
    internal_root = tmp_path / "internal"
    payload = _payload()
    case = {
        "case_id": "resource-toy",
        "block_index": 0,
        "condition_id": "H14_rule_y",
        "height": 14,
        "input_path": "cases/resource-toy/input.json",
        "metadata_path": "cases/resource-toy/case.json",
        "distance_path": "distances/block_001.json",
    }
    condition_matrix = np.asarray(
        [[0, 1, 2], [1, 0, 1], [2, 1, 0]],
        dtype=float,
    )
    block_matrix = np.asarray(
        [
            [0, 1, 2, 8],
            [1, 0, 1, 7],
            [2, 1, 0, 6],
            [8, 7, 6, 0],
        ],
        dtype=float,
    )
    tree = nx.DiGraph()
    tree.add_node("root", cell_id=0)
    tree.add_node("middle", cell_id=1)
    tree.add_node("leaf", cell_id=2)
    tree.add_edges_from([("root", "middle"), ("middle", "leaf")])

    write_json_atomic(bank_root / case["input_path"], payload)
    write_json_atomic(
        bank_root / case["metadata_path"],
        {"case_id": "resource-toy", "reconstruction_seed": 17},
    )
    write_json_atomic(
        bank_root / case["distance_path"],
        {
            "schema_version": DISTANCE_SCHEMA_VERSION,
            "semantics_version": CNP2CNP_SEMANTICS_VERSION,
            "ids": [0, 1, 2, 99],
            "matrix": block_matrix.tolist(),
            "provenance": {"semantics_version": CNP2CNP_SEMANTICS_VERSION},
        },
    )
    write_json_atomic(
        internal_root / record_path("resource-toy", M2_ID),
        {
            "case_id": "resource-toy",
            "arm_id": M2_ID,
            "status": "success",
            "tree_summary": tree_summary(tree),
        },
    )

    def distance_compute(cells):
        assert [cell.get_id() for cell in cells] == [0, 1, 2]
        matrix = condition_matrix.copy()
        if change == "distance":
            matrix[0, 2] = matrix[2, 0] = 3
        return DistanceMatrix(
            ids=[0, 1, 2],
            matrix=matrix,
            provenance={"semantics_version": CNP2CNP_SEMANTICS_VERSION},
        )

    def reconstruction_compute(_payload, _distance, _seed):
        reconstructed = tree.copy()
        if change == "topology":
            reconstructed.remove_edge("middle", "leaf")
            reconstructed.add_edge("root", "leaf")
        return reconstructed, {}, "root", {}

    stage_peaks = [64 * 1024**2, 96 * 1024**2]
    if change == "distance_rss":
        stage_peaks[0] = 1024**3 + 1
    elif change == "reconstruction_rss":
        stage_peaks[1] = 1024**3 + 1
    controlled_stage_rss(*stage_peaks)

    record = _audit_case(
        str(bank_root),
        str(internal_root),
        case,
        None,
        30,
        30,
        1024**3,
        distance_compute,
        reconstruction_compute,
    )

    if change is not None:
        assert record["status"] == "failure", record
        assert record["failure"]["stage"] == expected_stage
        assert record["failure"]["type"] == expected_type
        assert expected_message in record["failure"]["message"]
        return

    assert record["status"] == "success", record
    assert record["observed_unique_state_count"] == 3
    assert record["frozen_block_union_state_count"] == 4
    assert record["unordered_condition_pair_count"] == 3
    assert record["unordered_frozen_block_union_pair_count"] == 6
    assert record["equivalence"]["distance_matrix_exact"] is True
    assert record["equivalence"]["tree_summary_exact"] is True
    assert record["resources"]["pipeline"]["wall_time_ns"] > 0
    assert record["resources"]["pipeline"]["memory"]["peak_rss_bytes"] == 96 * 1024**2

