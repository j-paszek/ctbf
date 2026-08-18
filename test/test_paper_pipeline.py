from collections import Counter
from copy import deepcopy
import random
import subprocess

import networkx as nx
import numpy as np
import pytest

import ctbs

from algorithm_evaluation.paper_pipeline_contract import (
    FIXED_LABEL_AD_F1_SCHEMA_VERSION,
    PaperContractError,
    aligned_distance_submatrix,
    condition_id,
    fixed_label_ad_f1,
    nested_positive_bin_dropout,
    read_json,
    sample_nested_observations,
    validate_checksum_closure,
    validate_manifest,
    validate_status_record,
    write_checksum_file,
    write_json_atomic,
)
from algorithm_evaluation.paper_pipeline_analysis import (
    _bootstrap_interval,
    _complete_block_values,
    _holm_adjust,
    _sign_flip_test,
)
from algorithm_evaluation.paper_pipeline_runner import (
    REGISTERED_ARM_SPECS,
    _write_condition_dependency,
    _distance_semantics_version,
    _validate_primary_distance_provenance,
    reconstruct_arm,
    serialize_tree,
    validate_reconstruction_input,
)
from ctbs import DistanceMatrix
from distance_semantics import stable_distance_label_key
from evaluation_contract import evaluate_tree_pair_result, validate_evaluation_result
from simulator import Genotype


def _tree(labels, edges):
    tree = nx.DiGraph()
    for node, label in labels.items():
        tree.add_node(node, cell_id=label)
    tree.add_edges_from(edges)
    return tree


def _toy_input():
    profiles = {
        0: [2, 2, 2, 2, 2],
        1: [2, 2, 2, 2, 1],
        2: [2, 2, 2, 1, 1],
        3: [2, 2, 1, 1, 1],
        4: [2, 1, 1, 1, 1],
    }
    levels = ([0, 1, 2], [1, 2, 3], [2, 3, 4])
    payload = {
        "schema_version": "ctbf-v5-reconstruction-input-v1",
        "case_id": "toy",
        "condition_id": "toy-L3",
        "fraction": 1.0,
        "schedule_id": "L3",
        "levels": [
            {
                "biopsy_level": level,
                "generation": generation,
                "states": [
                    {"state_label": label, "cnp": profiles[label]}
                    for label in labels
                ],
            }
            for level, (generation, labels) in enumerate(zip((3, 5, 7), levels))
        ],
    }
    validate_reconstruction_input(payload)
    ids = sorted(profiles, key=stable_distance_label_key)
    genomes = np.asarray([profiles[label] for label in ids], dtype=int)
    matrix = np.abs(genomes[:, None, :] - genomes[None, :, :]).sum(axis=2)
    distance = DistanceMatrix(
        ids=ids,
        matrix=matrix,
        provenance={"semantic_version": "toy-l1"},
    )
    return payload, distance


def _occurrence_signature(tree):
    return Counter(
        (attributes.get("biopsy_level"), attributes.get("cell_id"))
        for _node, attributes in tree.nodes(data=True)
        if "biopsy_level" in attributes
    )


def _expected_occurrence_signature(payload):
    return Counter(
        (level["biopsy_level"], state["state_label"])
        for level in payload["levels"]
        for state in level["states"]
    )


def _cluster_signature(tree, inverse_labels=None):
    inverse_labels = inverse_labels or {}
    signature = Counter()
    for node in tree.nodes:
        cluster = []
        for member in {node, *nx.descendants(tree, node)}:
            label = tree.nodes[member].get("cell_id")
            if label is not None:
                cluster.append(inverse_labels.get(label, label))
        if cluster:
            signature[tuple(sorted(cluster, key=stable_distance_label_key))] += 1
    return signature


def test_paper_execution_is_disabled_until_new_v5_bytes_are_approved(tmp_path):
    with pytest.raises(PaperContractError) as error:
        validate_manifest(tmp_path / "not-yet-approved-v5-manifest.json")

    assert error.value.code == "v5_preregistration_not_frozen"


def test_nested_observation_sampler_uses_fraction_prefixes_and_literal_schedules():
    contract = {
        "maximal_generations": [3, 4, 5, 6, 7],
        "fractions": [0.25, 0.5],
        "level_schedules": {
            "L2": [3, 7],
            "L5": [3, 4, 5, 6, 7],
        },
        "condition_count": 4,
    }
    generation_cells = {
        generation: [
            Genotype(
                [2, generation, index],
                node_id=1000 + generation * 10 + index,
                generation=generation,
                cell_id=generation * 10 + index,
            )
            for index in range(8)
        ]
        for generation in contract["maximal_generations"]
    }

    design = sample_nested_observations(
        generation_cells,
        experiment_id="toy-nesting",
        sampling_seed=12345,
        regime_id="toy-regime",
        observation_contract=contract,
    )

    for schedule_id, generations in contract["level_schedules"].items():
        previous = None
        for fraction in contract["fractions"]:
            condition = design.conditions[condition_id(fraction, schedule_id)]
            assert condition.generations == tuple(generations)
            memberships = [
                {cell.cell_id for cell in cells}
                for cells in condition.cells_by_generation
            ]
            if previous is not None:
                assert all(left <= right for left, right in zip(previous, memberships))
            previous = memberships

    l2 = design.conditions[condition_id(0.5, "L2")]
    l5 = design.conditions[condition_id(0.5, "L5")]
    l5_by_generation = dict(zip(l5.generations, l5.cells_by_generation))
    assert all(
        {cell.cell_id for cell in cells}
        == {cell.cell_id for cell in l5_by_generation[generation]}
        for generation, cells in zip(l2.generations, l2.cells_by_generation)
    )


def test_condition_distance_is_an_exact_id_aligned_maximal_submatrix():
    ids = [9, 3, 7, 2]
    matrix = np.asarray(
        [
            [0, 1, 2, 3],
            [1, 0, 4, 5],
            [2, 4, 0, 6],
            [3, 5, 6, 0],
        ],
        dtype=float,
    )

    sub_ids, submatrix = aligned_distance_submatrix(ids, matrix, [7, 9, 2])

    assert sub_ids == [2, 7, 9]
    assert np.array_equal(submatrix, matrix[np.ix_([3, 2, 0], [3, 2, 0])])


def test_temporal_and_no_time_receive_identical_occurrences_and_tie_seed():
    payload, distance = _toy_input()
    before = deepcopy(payload)
    seed = 87123

    temporal, _levels, _root, temporal_metadata = reconstruct_arm(
        "temporal_minimum", payload, distance, reconstruction_seed=seed
    )
    no_time, _levels, _root, no_time_metadata = reconstruct_arm(
        "temporal_minimum_no_time", payload, distance, reconstruction_seed=seed
    )

    assert payload == before
    assert temporal_metadata["reconstruction_seed"] == seed
    assert no_time_metadata["reconstruction_seed"] == seed
    expected = _expected_occurrence_signature(payload)
    assert _occurrence_signature(temporal) == expected
    assert _occurrence_signature(no_time) == expected


def test_reconstruction_input_rejects_truth_identity_fields():
    payload, _distance = _toy_input()
    validate_reconstruction_input(payload)
    serialized = str(payload)
    assert "node_id" not in serialized
    assert "sampling_rank" not in serialized
    assert "truth_tree" not in serialized

    invalid = deepcopy(payload)
    invalid["levels"][0]["states"][0]["node_id"] = 991
    with pytest.raises(ValueError, match="Truth-only fields"):
        validate_reconstruction_input(invalid)


def test_fixed_label_ad_f1_filters_both_pair_endpoints_without_tree_projection():
    truth = _tree(
        {"a": "A", "x": "X", "b": "B"},
        [("a", "x"), ("x", "b")],
    )
    reconstructed = _tree(
        {"a": "A", "y": "Y", "b": "B", "c": "C"},
        [("a", "y"), ("y", "b"), ("a", "c")],
    )
    before = (set(reconstructed.nodes), set(reconstructed.edges))

    result = fixed_label_ad_f1(truth, reconstructed, {"A", "B"})

    assert result["schema_version"] == FIXED_LABEL_AD_F1_SCHEMA_VERSION
    assert result["metrics"]["ad_f1"] == 1.0
    assert result["metrics"]["counts"] == {
        "tp": 1,
        "fp": 0,
        "fn": 0,
        "true_unique_pair_count": 1,
        "reconstructed_unique_pair_count": 1,
    }
    assert (set(reconstructed.nodes), set(reconstructed.edges)) == before

    degenerate = fixed_label_ad_f1(truth, reconstructed, {"A"})
    assert degenerate["metrics"]["ad_f1"] == 0.0
    assert degenerate["metrics"]["degeneracy"] == "empty_truth_and_reconstruction"


def test_all_registered_arms_are_independent_of_arm_order_and_global_rng_state():
    payload, distance = _toy_input()
    arm_ids = [arm_id for arm_id, _algorithm in REGISTERED_ARM_SPECS]
    random.seed(11)
    np.random.seed(12)
    forward = {
        arm_id: serialize_tree(
            reconstruct_arm(arm_id, payload, distance, reconstruction_seed=333)[0]
        )
        for arm_id in arm_ids
    }
    random.seed(9001)
    np.random.seed(9002)
    reverse = {
        arm_id: serialize_tree(
            reconstruct_arm(arm_id, payload, distance, reconstruction_seed=333)[0]
        )
        for arm_id in reversed(arm_ids)
    }

    assert forward == reverse


def test_dropout_masks_are_nested_collision_typed_and_label_opaque():
    profiles = {
        10: (1, 1, 1, 1, 1, 1),
        20: (2, 2, 2, 2, 2, 2),
        30: (3, 3, 3, 3, 3, 3),
    }
    rates = (0.0, 0.05, 0.1, 0.2)
    design = nested_positive_bin_dropout(profiles, dropout_seed=1984, rates=rates)
    relabel = {10: 701, 20: 103, 30: 509}
    relabeled = nested_positive_bin_dropout(
        {relabel[label]: profile for label, profile in profiles.items()},
        dropout_seed=1984,
        rates=rates,
    )

    for label in profiles:
        previous_zeroes = set()
        for rate in rates:
            result = design.profiles_by_rate[rate][label]
            zeroes = {index for index, value in enumerate(result) if value == 0}
            assert previous_zeroes <= zeroes
            previous_zeroes = zeroes
            assert result == relabeled.profiles_by_rate[rate][relabel[label]]

    with pytest.raises(PaperContractError) as error:
        nested_positive_bin_dropout(
            {1: (1, 0), 2: (1, 1)},
            dropout_seed=7,
            rates=(0.0, 1.0),
        )
    assert error.value.code == "perturbed_state_collision"


def test_fully_labeled_reconstruction_topologies_are_opaque_label_invariant():
    payload, distance = _toy_input()
    relabel = {0: 40, 1: 10, 2: 70, 3: 20, 4: 50}
    inverse = {new: old for old, new in relabel.items()}
    relabeled_payload = deepcopy(payload)
    for level in relabeled_payload["levels"]:
        for state in level["states"]:
            state["state_label"] = relabel[state["state_label"]]
    relabeled_ids = [relabel[label] for label in distance.ids]
    relabeled_distance = DistanceMatrix(
        ids=relabeled_ids,
        matrix=distance.matrix,
        provenance=distance.provenance,
    )

    for arm_id in (
        "rooted_labeled_nj",
        "temporal_minimum",
        "temporal_minimum_no_time",
        "anticentral_parsimony",
    ):
        original = reconstruct_arm(
            arm_id, payload, distance, reconstruction_seed=719
        )[0]
        transformed = reconstruct_arm(
            arm_id,
            relabeled_payload,
            relabeled_distance,
            reconstruction_seed=719,
        )[0]
        assert _cluster_signature(original) == _cluster_signature(transformed, inverse)


def test_dependency_failure_writes_every_arm_status_and_native_evaluation(tmp_path):
    condition_root = tmp_path / "condition"
    _write_condition_dependency(
        condition_root,
        condition_id_value="toy-condition",
        dependency="distance:toy",
        message="distance failed",
    )

    input_record = read_json(condition_root / "input.json")
    validate_status_record(input_record["status_record"])
    assert input_record["status_record"]["status"] == "not_run_dependency"
    for arm_id, _algorithm in REGISTERED_ARM_SPECS:
        arm_root = condition_root / "arms" / arm_id
        status = read_json(arm_root / "status.json")
        evaluation = read_json(arm_root / "evaluation.json")
        validate_status_record(status)
        validate_evaluation_result(evaluation)
        assert status["status"] == "not_run_dependency"
        assert evaluation["status"] == "failure"
        assert evaluation["failure"]["code"] == "not_run_dependency"


def test_raw_and_complete_checksum_closures_are_distinct_and_strict(tmp_path):
    write_json_atomic(tmp_path / "run_status.json", {"status": "raw_closed"})
    write_json_atomic(tmp_path / "cases" / "toy.json", {"value": 1})
    write_checksum_file(tmp_path, "raw_checksums.sha256", include_analysis=False)
    validate_checksum_closure(
        tmp_path, "raw_checksums.sha256", include_analysis=False
    )
    write_json_atomic(
        tmp_path / "analysis" / "ctbf-v5-paper-analysis-v1" / "summary.json",
        {"status": "complete"},
    )
    write_checksum_file(tmp_path, "complete_checksums.sha256", include_analysis=True)
    validate_checksum_closure(
        tmp_path, "complete_checksums.sha256", include_analysis=True
    )

    write_json_atomic(tmp_path / "cases" / "toy.json", {"value": 2})
    with pytest.raises(PaperContractError) as error:
        validate_checksum_closure(
            tmp_path, "raw_checksums.sha256", include_analysis=False
        )
    assert error.value.code == "checksum_closure"


def test_cnp2cnp_timeout_is_a_typed_bounded_external_failure(monkeypatch, tmp_path):
    def time_out(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(
            cmd=["cnp2cnp"],
            timeout=3,
            output=b"partial stdout",
            stderr=b"partial stderr",
        )

    monkeypatch.setattr(ctbs.subprocess, "run", time_out)
    with pytest.raises(ctbs.Cnp2CnpExecutionError) as error:
        ctbs._run_checked_cnp2cnp(
            ["cnp2cnp"],
            tmp_path,
            timeout_seconds=3,
            capture_limit_bytes=1024,
        )

    assert error.value.record["status"] == "timeout"
    assert error.value.record["timeout_seconds"] == 3.0
    assert error.value.record["stdout"]["preview"] == "partial stdout"
    assert error.value.record["stderr"]["preview"] == "partial stderr"


def test_native_evaluator_failure_and_subfull_coverage_remain_explicit():
    truth = _tree(
        {"root": "A", "middle": "B", "leaf": "C"},
        [("root", "middle"), ("middle", "leaf")],
    )
    missing_middle = _tree(
        {"root": "A", "leaf": "C"},
        [("root", "leaf")],
    )
    outside = _tree(
        {"root": "A", "leaf": "X"},
        [("root", "leaf")],
    )

    covered = evaluate_tree_pair_result(
        truth, missing_middle, ["A", "B", "C"]
    )
    failed = evaluate_tree_pair_result(truth, outside, ["A", "B", "C"])

    validate_evaluation_result(covered)
    validate_evaluation_result(failed)
    assert covered["status"] == "success"
    assert covered["inputs"]["observation_label_coverage"]["fraction"] == pytest.approx(2 / 3)
    assert failed["status"] == "failure"
    assert failed["failure"]["code"] == "reconstructed_labels_outside_observation_set"


def test_analysis_resampling_is_deterministic_and_blocks_by_replicate():
    values = [0.1, 0.2, -0.05, 0.4]
    assert _bootstrap_interval(values, repetitions=1000, seed=17) == _bootstrap_interval(
        values, repetitions=1000, seed=17
    )
    assert _sign_flip_test(values, repetitions=2000, seed=19) == _sign_flip_test(
        values, repetitions=2000, seed=19
    )
    assert _holm_adjust({"a": 0.01, "b": 0.04, "c": None}) == {
        "a": 0.03,
        "b": 0.08,
        "c": None,
    }

    rows = [
        {"replicate": 1, "regime_id": "a", "effect": 0.1},
        {"replicate": 1, "regime_id": "b", "effect": 0.2},
        {"replicate": 1, "regime_id": "c", "effect": 0.3},
        {"replicate": 2, "regime_id": "a", "effect": 100.0},
        {"replicate": 2, "regime_id": "b", "effect": 100.0},
    ]
    assert _complete_block_values(rows, ["a", "b", "c"]) == {
        1: pytest.approx(0.2)
    }


def test_runner_reads_the_established_plural_distance_semantics_key():
    provenance = {
        "schema_version": "ctbf-distance-provenance-v1",
        "semantics_version": "ctbf-cnp2cnp-any-min-bidirectional-v1",
        "metric": "cnp2cnp",
        "distance_mode": "any",
        "symmetrization": "minimum_bidirectional",
        "formula": "min(d_any(u,v),d_any(v,u))",
        "construction": "opposite_order_matrix_mode",
    }
    distance = DistanceMatrix(
        ids=[1, 2],
        matrix=[[0.0, 1.0], [1.0, 0.0]],
        provenance=provenance,
    )

    assert _distance_semantics_version(provenance) == provenance["semantics_version"]
    _validate_primary_distance_provenance(distance, smoke=False)
