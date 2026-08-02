from copy import deepcopy
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from algorithm_evaluation import bounded_discovery as discovery
from ctbs import DistanceMatrix
from distance_semantics import DirectedDistanceBundle
from simulator import Genotype


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = (
    PROJECT_ROOT
    / "experimental_description"
    / "g0_02_c_bounded_discovery_manifest.json"
)


def load_manifest():
    with MANIFEST_PATH.open("r", encoding="utf-8") as source:
        return json.load(source)


def test_frozen_seed_triples_are_namespace_derived_and_design_is_valid():
    manifest = discovery.validate_manifest(
        load_manifest(),
        require_sources=False,
    )
    namespace = manifest["seed_derivation"]["namespace"]
    for record in manifest["replicates"]:
        for stream in ("simulation", "biopsy", "reconstruction"):
            assert record[f"{stream}_seed"] == discovery.derive_seed(
                namespace,
                stream,
                record["replicate"],
            )
    assert len(manifest["cases"]) * len(manifest["replicates"]) == 30


def test_manifest_digest_ignores_only_its_execution_digest_field():
    manifest = load_manifest()
    before = discovery.manifest_contract_sha256(manifest)
    manifest["source_contract"]["manifest_sha256_at_execution"] = "ignored"
    assert discovery.manifest_contract_sha256(manifest) == before
    manifest["cases"][0]["r_dist"] += 1
    assert discovery.manifest_contract_sha256(manifest) != before


def test_truth_direction_selection_excludes_incomparable_and_marks_strict_time():
    truth = nx.DiGraph()
    truth.add_node(0, cell_id=0, genome=[2])
    truth.add_node(1, cell_id=1, genome=[2])
    truth.add_node(2, cell_id=2, genome=[1])
    truth.add_node(3, cell_id=3, genome=[3])
    truth.add_edges_from([(0, 1), (1, 2), (0, 3)])

    directions, time_decided, counts = discovery.derive_truth_directions(
        truth,
        [1, 2, 3],
        {1: {0}, 2: {2}, 3: {1}},
    )

    assert directions == [(1, 2)]
    assert time_decided == [(1, 2)]
    assert counts == {
        "unordered_observed_state_pairs": 3,
        "truth_unambiguous": 1,
        "strictly_time_decided": 1,
        "truth_incomparable": 2,
    }


def test_truth_direction_selection_excludes_recurrently_ambiguous_state_pair():
    truth = nx.DiGraph()
    truth.add_node(0, cell_id=1, genome=[2])
    truth.add_node(1, cell_id=2, genome=[1])
    truth.add_node(2, cell_id=1, genome=[2])
    truth.add_edges_from([(0, 1), (1, 2)])

    directions, time_decided, counts = discovery.derive_truth_directions(
        truth,
        [1, 2],
        {1: {0, 2}, 2: {1}},
    )

    assert directions == []
    assert time_decided == []
    assert counts["recurrently_ambiguous"] == 1


def test_biopsy_serialization_contains_observables_but_no_simulator_node_id():
    biopsies = [
        [Genotype([2, 2], 101, generation=4, cell_id=7)],
        [Genotype([2, 2], 909, generation=6, cell_id=7)],
    ]
    serialized = discovery.serialize_biopsies(biopsies, [4, 6])

    assert serialized[0]["observations"] == [{"cell_id": 7, "genome": [2, 2]}]
    assert "node_id" not in serialized[0]["observations"][0]
    rebuilt = discovery.deserialize_biopsies(serialized)
    assert [cells[0].cell_id for cells in rebuilt] == [7, 7]
    assert [cells[0].generation for cells in rebuilt] == [4, 6]


def test_every_frozen_arm_runs_on_one_tiny_observation_case():
    manifest = load_manifest()
    cell_lists = [
        [Genotype([2, 2], 1, generation=4, cell_id=1)],
        [Genotype([2, 1], 2, generation=6, cell_id=2)],
    ]
    minimum = DistanceMatrix(ids=[1, 2], matrix=[[0, 1], [1, 0]])
    fast = DistanceMatrix(ids=[1, 2], matrix=[[0, 1], [1, 0]])
    directed = DirectedDistanceBundle([1, 2], [[0, 1], [2, 0]])
    truth = nx.DiGraph()
    truth.add_node(10, cell_id=1, genome=[2, 2])
    truth.add_node(11, cell_id=2, genome=[2, 1])
    truth.add_edge(10, 11)

    for arm in manifest["portfolio_arms"]:
        result = discovery.run_arm(
            arm,
            cell_lists,
            {"minimum": minimum, "fast": fast, "directed": directed},
            reconstruction_seed=123,
            r_dist=2,
            true_tree=truth,
            true_root=10,
            observed_ids=[1, 2],
        )
        assert result["status"] == "success", (arm["id"], result)
        assert result["output"]["edge_count"] == result["output"]["node_count"] - 1
        if arm["id"] in {"temporal_minimum", "temporal_fast", "temporal_directed"}:
            assert result["metrics"]["ad_f1"] == 1.0


def test_tree_serialization_round_trip_preserves_labels_and_direction():
    tree = nx.DiGraph()
    tree.add_node(8, cell_id=1, genome=[2, 2])
    tree.add_node(9, cell_id=2, genome=[2, 1])
    tree.add_edge(8, 9, weight=1.0)

    rebuilt = discovery.deserialize_tree(discovery.serialize_tree(tree))

    assert list(rebuilt.edges(data=True)) == [(8, 9, {"weight": 1.0})]
    assert rebuilt.nodes[8]["cell_id"] == 1


def _fake_audit():
    return {
        "unordered_pair_count": 10,
        "asymmetric_pair_count": 6,
        "asymmetry_magnitude": {"minimum": 1, "median": 1, "maximum": 2},
        "plausibility_strata": {
            "both_plausible": {"pairs": 8, "asymmetric_pairs": 5},
            "left_only_plausible": {"pairs": 1, "asymmetric_pairs": 1},
            "right_only_plausible": {"pairs": 1, "asymmetric_pairs": 0},
            "neither_plausible": {"pairs": 0, "asymmetric_pairs": 0},
        },
        "profile_strata": {"all_positive": 8, "contains_zero": 2},
        "truth_direction": {
            "provided": 8,
            "excluded_time_decided": 1,
            "excluded_plausibility_decided": 1,
            "excluded_neither_plausible": 0,
            "eligible_both_plausible": 6,
            "ties": 1,
            "sign_informative": 5,
            "correct": 4,
            "incorrect": 1,
            "accuracy_by_absolute_difference": {
                "1.0": {"pairs": 5, "correct": 4, "accuracy": 0.8}
            },
        },
    }


def _fake_record(manifest, replicate, case_id, offset=0.0):
    values = {
        "classical_partial": (0.40, 0.50),
        "biopsy_guided_classical": (0.45, 0.55),
        "rooted_labeled_nj": (0.50, 0.50),
        "temporal_minimum": (0.60, 0.60),
        "temporal_minimum_no_time": (0.55, 0.58),
        "legacy_closest_pair": (0.48, 0.49),
        "anticentral_parsimony": (0.54, 0.50),
        "temporal_fast": (0.595, 0.595),
        "temporal_directed": (0.62, 0.60),
        "temporal_directed_no_time": (0.56, 0.58),
    }
    arms = {
        arm_id: {
            "status": "success",
            "metrics": {"ad_f1": ad_f1 + offset, "grf": grf + offset},
        }
        for arm_id, (ad_f1, grf) in values.items()
    }
    distance = {
        "ordered_triangle_fast": {
            "wall_time_ns": 100,
            "memory": {"peak_rss_bytes": 1000},
            "external_process_count": 1,
            "directional_transformation_count": 10,
        },
        "minimum_bidirectional": {
            "wall_time_ns": 200,
            "memory": {"peak_rss_bytes": 1200},
            "external_process_count": 2,
            "directional_transformation_count": 20,
        },
        "minimum_with_directed": {
            "wall_time_ns": 210,
            "memory": {"peak_rss_bytes": 1250},
            "external_process_count": 2,
            "directional_transformation_count": 20,
        },
    }
    return {
        "status": "complete",
        "case": {"id": case_id},
        "replicate": {"replicate": replicate},
        "arms": arms,
        "direction_audit": _fake_audit(),
        "distance": distance,
    }


def test_summary_uses_paired_replicate_blocks_and_applies_frozen_gates():
    manifest = load_manifest()
    records = [
        _fake_record(manifest, 1, "case_a"),
        _fake_record(manifest, 1, "case_b", 0.01),
        _fake_record(manifest, 2, "case_a", -0.01),
        _fake_record(manifest, 2, "case_b"),
    ]

    summary = discovery.summarize_records(records, manifest)

    temporal = summary["comparisons"]["temporal_vs_no_time_ad_f1"]
    assert temporal["complete_pair_count"] == 4
    assert temporal["summary"]["mean"] == pytest.approx(0.05)
    assert temporal["block_bootstrap_mean_95"]["block_count"] == 2
    assert summary["distance"]["minimum_over_fast_speedup"]["median"] == 2.0
    assert summary["direction_audit"]["sign_informative"] == 20
    assert not summary["promotion_gates"]["directed_to_secondary_confirmation"]["passed"]


def test_manifest_validation_detects_seed_and_config_changes():
    manifest = load_manifest()
    bad_seed = deepcopy(manifest)
    bad_seed["replicates"][0]["simulation_seed"] += 1
    try:
        discovery.validate_manifest(bad_seed, require_sources=False)
    except ValueError as exc:
        assert "simulation_seed" in str(exc)
    else:
        raise AssertionError("Changed seed should be rejected.")

    bad_hash = deepcopy(manifest)
    bad_hash["cases"][0]["config_sha256"] = "0" * 64
    try:
        discovery.validate_manifest(bad_hash, require_sources=False)
    except ValueError as exc:
        assert "Config hash mismatch" in str(exc)
    else:
        raise AssertionError("Changed config hash should be rejected.")
