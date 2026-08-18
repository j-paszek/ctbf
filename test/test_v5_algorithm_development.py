from __future__ import annotations

import copy

import networkx as nx
import numpy as np
import pytest

from algorithm_evaluation.process_isolation import (
    CASE_ARM_WORKER_UNIT,
    fresh_process_contract,
)
from algorithm_evaluation.v5_algorithm_development_bank import generate_bank
from algorithm_evaluation.v5_algorithm_development_common import (
    ARM_SPEC_BY_ID,
    BIOPSY_GUIDED_FULL_COUNTERPART_BY_PARTIAL_ID,
    BIOPSY_GUIDED_FULL_DEFAULT_ID,
    BIOPSY_GUIDED_FULL_BOTTOM_TOP_EXTENSION_ARM_SPECS,
    BIOPSY_GUIDED_FULL_BOTTOM_TOP_INTERACTION_ROLE,
    BIOPSY_GUIDED_FULL_EXTENSION_ARM_SPECS,
    BIOPSY_GUIDED_FULL_FAMILY,
    BIOPSY_GUIDED_FULL_INCUMBENT_ID,
    BIOPSY_GUIDED_FULL_ROOTED_Q_DEFERRED_ID,
    CONTEXT_FAMILY,
    DEFAULT_BLOCK_COUNT,
    DEVELOPMENT_EXTENSION_ARM_SPECS,
    DEVELOPMENT_NAMESPACE,
    HEIGHT_SCHEDULES,
    INFERRED_COPY_INCUMBENT_ID,
    INITIAL_ARM_SPECS,
    INFERRED_COPY_FAMILY,
    LEGACY_RUN_SCHEMA_VERSION,
    PARTIAL_BOTTOM_CANDIDATE_ROLE,
    PARTIAL_BOTTOM_CONTROL_ID,
    PARTIAL_BOTTOM_EXTENSION_ARM_SPECS,
    PARTIAL_BOTTOM_TOP_EXTENSION_ARM_SPECS,
    PARTIAL_BOTTOM_TOP_INTERACTION_ARM_ID,
    PARTIAL_BOTTOM_TOP_INTERACTION_ROLE,
    PARTIAL_FAMILY,
    PARTIAL_TOP_EXTENSION_ARM_SPECS,
    REPORT_SCHEMA_VERSION,
    TOP_OUTPUT_PROJECTION_CREATED_NODES_UNLABELED,
    TOP_OUTPUT_PROJECTION_NONE,
    canonical_topology_digest,
    derived_seed,
    load_bank_manifest,
    reconstruct_development_arm,
    resolve_arm_specs,
    write_json,
)
from algorithm_evaluation.v5_algorithm_development_full_attachment_audit import (
    FULL_ATTACHMENT_AUDIT_SCHEMA_VERSION,
    association_summary,
    derive_attachment_measures,
    run_full_attachment_audit,
)
from algorithm_evaluation.v5_algorithm_development_mechanism_audit import (
    reconstruct_incumbent_with_orientation_trace,
    run_mechanism_audit,
)
from algorithm_evaluation.v5_algorithm_development_report import (
    bottom_top_factorial_interaction_comparison,
    build_report,
    factorial_interaction_comparison,
    pairwise_comparison,
    write_report,
)
from algorithm_evaluation.v5_algorithm_development_run import (
    run_algorithms,
    run_semantic_gate,
)
from ctbs import DistanceMatrix
from distance_semantics import cnp2cnp_provenance
from reconstructor_biopsy_blocks import (
    BIOPSY_GUIDED_AUDIT_COUNTERS,
    BIOPSY_GUIDED_AUDIT_SCHEMA_VERSION,
)


def _input_payload() -> dict:
    return {
        "schema_version": "ctbf-v5-reconstruction-input-v1",
        "case_id": "fixture-H14",
        "condition_id": "fixture",
        "levels": [
            {
                "biopsy_level": 0,
                "generation": 9,
                "states": [
                    {"state_label": 0, "cnp": [2, 2, 2]},
                    {"state_label": 1, "cnp": [2, 2, 3]},
                ],
            },
            {
                "biopsy_level": 1,
                "generation": 12,
                "states": [
                    {"state_label": 2, "cnp": [2, 3, 3]},
                    {"state_label": 3, "cnp": [1, 2, 3]},
                ],
            },
            {
                "biopsy_level": 2,
                "generation": 14,
                "states": [
                    {"state_label": 4, "cnp": [1, 3, 3]},
                    {"state_label": 5, "cnp": [1, 3, 4]},
                ],
            },
        ],
    }


def _distance(payload: dict) -> DistanceMatrix:
    states = [state for level in payload["levels"] for state in level["states"]]
    ids = [state["state_label"] for state in states]
    profiles = np.asarray([state["cnp"] for state in states], dtype=float)
    matrix = np.abs(profiles[:, None, :] - profiles[None, :, :]).sum(axis=2)
    return DistanceMatrix(ids=ids, matrix=matrix)


def _injected_cnp2cnp_distance(cells) -> DistanceMatrix:
    ids = [cell.cell_id for cell in cells]
    profiles = np.asarray([cell.genome for cell in cells], dtype=float)
    matrix = np.abs(profiles[:, None, :] - profiles[None, :, :]).sum(axis=2)
    return DistanceMatrix(
        ids=ids,
        matrix=matrix,
        provenance=cnp2cnp_provenance(
            construction="opposite_order_matrix_mode",
            profile_count=len(ids),
        ),
    )


def _success_record(case: dict, arm_id: str, family: str, score: float) -> dict:
    return {
        **case,
        "arm_id": arm_id,
        "family": family,
        "problem": "partial",
        "primary_metric": "grf",
        "status": "success",
        "failure": None,
        "metrics": {"grf": score},
        "observation_coverage": {"fraction": 1.0},
        "tree_summary": {
            "node_count": 4,
            "inferred_copy_occurrence_count": 0,
        },
        "resources": {
            "reconstruction": {
                "wall_time_ns": 1_000_000,
                "memory": {"peak_rss_bytes": 1000},
            },
            "evaluation": {
                "wall_time_ns": 1_000_000,
                "memory": {"peak_rss_bytes": 1200},
            },
        },
    }


def test_full_attachment_audit_derives_the_prespecified_hard_burden():
    audit = {
        "schema_version": BIOPSY_GUIDED_AUDIT_SCHEMA_VERSION,
        **{field: 0 for field in BIOPSY_GUIDED_AUDIT_COUNTERS},
        "child_decision_count": 10,
        "raw_radius_candidate_total": 20,
        "plausible_candidate_total": 14,
        "same_state_selected_count": 2,
        "no_plausible_parent_count": 1,
        "one_plausible_parent_count": 3,
        "multiple_plausible_parent_count": 4,
        "unique_minimum_parent_count": 2,
        "minimum_distance_tie_count": 2,
        "tie_parent_selected_count": 0,
        "tie_deferred_count": 2,
        "selected_parent_count": 7,
        "copy_up_count": 3,
    }
    measures = derive_attachment_measures(audit)
    assert measures["non_same_state_hard_attachment_count"] == 5
    assert measures["non_same_state_hard_attachment_fraction"] == 0.5
    assert measures["non_same_state_share_of_selected_attachments"] == pytest.approx(
        5 / 7
    )
    association = association_summary([0.0, 0.5, 1.0], [1.0, 0.5, 0.0])
    assert association["pearson_r"] == pytest.approx(-1.0)
    assert association["spearman_r"] == pytest.approx(-1.0)
    assert association["formal_significance_test_run"] is False


def test_full_attachment_audit_rejects_inconsistent_decision_categories():
    audit = {
        "schema_version": BIOPSY_GUIDED_AUDIT_SCHEMA_VERSION,
        **{field: 0 for field in BIOPSY_GUIDED_AUDIT_COUNTERS},
        "child_decision_count": 1,
        "selected_parent_count": 1,
    }
    with pytest.raises(ValueError, match="Selected-parent decision categories"):
        derive_attachment_measures(audit)


def test_initial_roster_is_the_approved_32_arm_contract():
    counts = {
        family: sum(spec.family == family for spec in INITIAL_ARM_SPECS)
        for family in (PARTIAL_FAMILY, INFERRED_COPY_FAMILY, CONTEXT_FAMILY)
    }
    assert counts == {
        PARTIAL_FAMILY: 7,
        INFERRED_COPY_FAMILY: 22,
        CONTEXT_FAMILY: 3,
    }
    assert len({spec.arm_id for spec in INITIAL_ARM_SPECS}) == 32


def test_partial_top_extensions_include_only_the_approved_three_r2_and_one_r4_arms():
    assert [spec.arm_id for spec in PARTIAL_TOP_EXTENSION_ARM_SPECS] == [
        "biopsy_guided_top_rooted_labeled_q_r2",
        "biopsy_guided_top_anticentral_binary_r2",
        "biopsy_guided_top_anticentral_parent_reuse_r2",
        "biopsy_guided_top_anticentral_binary_r4",
    ]
    assert all(spec.family == PARTIAL_FAMILY for spec in PARTIAL_TOP_EXTENSION_ARM_SPECS)
    assert all(spec.input_mode == "ordered" for spec in PARTIAL_TOP_EXTENSION_ARM_SPECS)
    assert [spec.radius for spec in PARTIAL_TOP_EXTENSION_ARM_SPECS] == [2, 2, 2, 4]
    assert all(spec.biopsy_preset == "default" for spec in PARTIAL_TOP_EXTENSION_ARM_SPECS)
    assert all(
        spec.top_output_projection
        == TOP_OUTPUT_PROJECTION_CREATED_NODES_UNLABELED
        for spec in PARTIAL_TOP_EXTENSION_ARM_SPECS
    )
    r4 = ARM_SPEC_BY_ID["biopsy_guided_top_anticentral_binary_r4"]
    assert r4.algorithm_name == INFERRED_COPY_INCUMBENT_ID
    assert r4.role == "top_radius_interaction_candidate"


def test_partial_bottom_extensions_are_the_six_missing_fixed_r2_binary_top_rows():
    assert [spec.arm_id for spec in PARTIAL_BOTTOM_EXTENSION_ARM_SPECS] == [
        "biopsy_guided_top_anticentral_binary_r2_bottom_anticentral_tie",
        "biopsy_guided_top_anticentral_binary_r2_bottom_binarized",
        "biopsy_guided_top_anticentral_binary_r2_bottom_anticentral_tie_binarized",
        "biopsy_guided_top_anticentral_binary_r2_bottom_deferred_tie",
        "biopsy_guided_top_anticentral_binary_r2_bottom_central_tie",
        "biopsy_guided_top_anticentral_binary_r2_bottom_diploid_parsimony_tie",
    ]
    assert [spec.biopsy_preset for spec in PARTIAL_BOTTOM_EXTENSION_ARM_SPECS] == [
        "anticentral_tie",
        "binarized",
        "anticentral_tie_binarized",
        "deferred_tie",
        "central_tie",
        "diploid_parsimony_tie",
    ]
    assert all(spec.radius == 2 for spec in PARTIAL_BOTTOM_EXTENSION_ARM_SPECS)
    assert all(
        spec.role == PARTIAL_BOTTOM_CANDIDATE_ROLE
        for spec in PARTIAL_BOTTOM_EXTENSION_ARM_SPECS
    )
    assert all(
        spec.algorithm_name == INFERRED_COPY_INCUMBENT_ID
        for spec in PARTIAL_BOTTOM_EXTENSION_ARM_SPECS
    )
    assert all(
        spec.top_output_projection
        == TOP_OUTPUT_PROJECTION_CREATED_NODES_UNLABELED
        for spec in PARTIAL_BOTTOM_EXTENSION_ARM_SPECS
    )
    assert DEVELOPMENT_EXTENSION_ARM_SPECS == (
        PARTIAL_TOP_EXTENSION_ARM_SPECS
        + PARTIAL_BOTTOM_EXTENSION_ARM_SPECS
        + PARTIAL_BOTTOM_TOP_EXTENSION_ARM_SPECS
        + BIOPSY_GUIDED_FULL_EXTENSION_ARM_SPECS
        + BIOPSY_GUIDED_FULL_BOTTOM_TOP_EXTENSION_ARM_SPECS
    )
    assert PARTIAL_BOTTOM_CONTROL_ID == (
        "biopsy_guided_top_anticentral_binary_r2"
    )


def test_bottom_top_factorial_extension_is_only_the_missing_classical_cell():
    assert len(PARTIAL_BOTTOM_TOP_EXTENSION_ARM_SPECS) == 1
    spec = PARTIAL_BOTTOM_TOP_EXTENSION_ARM_SPECS[0]
    assert spec.arm_id == PARTIAL_BOTTOM_TOP_INTERACTION_ARM_ID
    assert spec.arm_id == "biopsy_guided_classical_r2_bottom_deferred_tie"
    assert spec.algorithm_name == "neighbor_joining_classical"
    assert spec.radius == 2
    assert spec.biopsy_preset == "deferred_tie"
    assert spec.role == PARTIAL_BOTTOM_TOP_INTERACTION_ROLE
    assert spec.top_output_projection == "none"


def test_fully_labeled_biopsy_guided_roster_covers_every_projected_arm_once():
    projected_ids = {
        spec.arm_id
        for spec in (
            PARTIAL_TOP_EXTENSION_ARM_SPECS
            + PARTIAL_BOTTOM_EXTENSION_ARM_SPECS
        )
    }
    assert set(BIOPSY_GUIDED_FULL_COUNTERPART_BY_PARTIAL_ID) == projected_ids
    assert set(BIOPSY_GUIDED_FULL_COUNTERPART_BY_PARTIAL_ID.values()) == {
        spec.arm_id for spec in BIOPSY_GUIDED_FULL_EXTENSION_ARM_SPECS
    }
    assert len(BIOPSY_GUIDED_FULL_EXTENSION_ARM_SPECS) == 10
    assert BIOPSY_GUIDED_FULL_DEFAULT_ID in {
        spec.arm_id for spec in BIOPSY_GUIDED_FULL_EXTENSION_ARM_SPECS
    }
    assert BIOPSY_GUIDED_FULL_INCUMBENT_ID == (
        "biopsy_guided_full_anticentral_binary_r2_bottom_deferred_tie"
    )
    assert all(
        spec.family == BIOPSY_GUIDED_FULL_FAMILY
        and spec.problem
        == "biopsy_guided_occurrence_aware_fully_labeled_closed_state"
        and spec.input_mode == "ordered"
        and not spec.only_nj
        and spec.primary_metric == "ad_f1"
        and spec.complementary_metrics
        == ("grf", "ad_precision", "ad_recall")
        and spec.top_output_projection == TOP_OUTPUT_PROJECTION_NONE
        and spec.algorithm_name != "neighbor_joining_classical"
        for spec in BIOPSY_GUIDED_FULL_EXTENSION_ARM_SPECS
    )
    assert resolve_arm_specs((BIOPSY_GUIDED_FULL_FAMILY,)) == (
        BIOPSY_GUIDED_FULL_EXTENSION_ARM_SPECS
    )


def test_full_bottom_top_extension_is_only_the_missing_rooted_q_cell():
    assert len(BIOPSY_GUIDED_FULL_BOTTOM_TOP_EXTENSION_ARM_SPECS) == 1
    spec = BIOPSY_GUIDED_FULL_BOTTOM_TOP_EXTENSION_ARM_SPECS[0]
    assert spec.arm_id == BIOPSY_GUIDED_FULL_ROOTED_Q_DEFERRED_ID
    assert spec.arm_id == (
        "biopsy_guided_full_rooted_labeled_q_r2_bottom_deferred_tie"
    )
    assert spec.algorithm_name == "rooted_labeled_nj"
    assert spec.radius == 2
    assert spec.biopsy_preset == "deferred_tie"
    assert spec.role == BIOPSY_GUIDED_FULL_BOTTOM_TOP_INTERACTION_ROLE
    assert spec.family == BIOPSY_GUIDED_FULL_FAMILY
    assert spec.primary_metric == "ad_f1"
    assert spec.top_output_projection == TOP_OUTPUT_PROJECTION_NONE
    assert resolve_arm_specs((spec.arm_id,)) == (spec,)


def test_projected_parent_reuse_preserves_copy_up_labels_and_permits_polytomy():
    payload = {
        "schema_version": "ctbf-v5-reconstruction-input-v1",
        "case_id": "projection-copy-up",
        "condition_id": "projection-copy-up",
        "levels": [
            {
                "biopsy_level": 0,
                "generation": 9,
                "states": [
                    {"state_label": 0, "cnp": [2, 2]},
                    {"state_label": 1, "cnp": [2, 3]},
                ],
            },
            {
                "biopsy_level": 1,
                "generation": 14,
                "states": [
                    {"state_label": 2, "cnp": [5, 5]},
                ],
            },
        ],
    }
    distance = _distance(payload)
    binary_tree, _levels, binary_root, binary_metadata = reconstruct_development_arm(
        ARM_SPEC_BY_ID["biopsy_guided_top_anticentral_binary_r2"],
        payload,
        distance,
        reconstruction_seed=7,
    )
    reuse_tree, _levels, reuse_root, reuse_metadata = reconstruct_development_arm(
        ARM_SPEC_BY_ID["biopsy_guided_top_anticentral_parent_reuse_r2"],
        payload,
        distance,
        reconstruction_seed=7,
    )

    assert nx.is_arborescence(binary_tree)
    assert nx.is_arborescence(reuse_tree)
    assert binary_tree.nodes[binary_root]["cell_id"] is None
    assert reuse_tree.nodes[reuse_root]["cell_id"] is None
    assert binary_metadata["top_created_node_count"] == 2
    assert binary_metadata["top_labels_cleared_count"] == 2
    assert reuse_metadata["top_created_node_count"] == 1
    assert reuse_metadata["top_labels_cleared_count"] == 1
    assert reuse_tree.number_of_nodes() + 1 == binary_tree.number_of_nodes()
    assert max(dict(binary_tree.out_degree()).values()) == 2
    assert max(dict(reuse_tree.out_degree()).values()) == 3

    # The lower observation and its radius-triggered upper copy both remain
    # labeled. Projection is restricted to nodes created by the top solver.
    assert [
        attributes["cell_id"]
        for _node, attributes in reuse_tree.nodes(data=True)
    ].count(2) == 2
    assert all(
        attributes.get("genome") is None
        for _node, attributes in reuse_tree.nodes(data=True)
        if attributes.get("cell_id") is None
    )


@pytest.mark.parametrize(
    ("partial_id", "full_id"),
    BIOPSY_GUIDED_FULL_COUNTERPART_BY_PARTIAL_ID.items(),
)
def test_full_counterpart_changes_only_projected_top_labels(
    partial_id,
    full_id,
):
    payload = _input_payload()
    distance = _distance(payload)
    partial_tree, partial_levels, partial_root, partial_metadata = (
        reconstruct_development_arm(
            ARM_SPEC_BY_ID[partial_id],
            payload,
            distance,
            reconstruction_seed=991,
        )
    )
    full_tree, full_levels, full_root, full_metadata = reconstruct_development_arm(
        ARM_SPEC_BY_ID[full_id],
        payload,
        distance,
        reconstruction_seed=991,
    )

    assert partial_root == full_root
    assert partial_levels == full_levels
    assert set(partial_tree.nodes) == set(full_tree.nodes)
    assert set(partial_tree.edges) == set(full_tree.edges)
    assert {
        edge: partial_tree.edges[edge]["weight"] for edge in partial_tree.edges
    } == {edge: full_tree.edges[edge]["weight"] for edge in full_tree.edges}
    assert all(
        attributes.get("cell_id") is not None
        and attributes.get("genome") is not None
        for _node, attributes in full_tree.nodes(data=True)
    )
    for node_id, attributes in partial_tree.nodes(data=True):
        if attributes.get("cell_id") is None:
            continue
        assert full_tree.nodes[node_id]["cell_id"] == attributes["cell_id"]
        assert np.array_equal(
            full_tree.nodes[node_id]["genome"],
            attributes["genome"],
        )
    assert full_metadata["top_output_projection"] == TOP_OUTPUT_PROJECTION_NONE
    assert full_metadata["top_created_node_count"] == (
        partial_metadata["top_created_node_count"]
    )
    assert full_metadata["top_labels_cleared_count"] == 0
    assert full_metadata["top_labels_retained_count"] == (
        partial_metadata["top_labels_cleared_count"]
    )
    assert full_metadata["top_genomes_retained_count"] == (
        partial_metadata["top_genomes_cleared_count"]
    )


def test_development_seeds_are_reproducible_and_stream_separated():
    first = derived_seed("simulation", 20260813, 0)
    assert first == derived_seed("simulation", 20260813, 0)
    assert first != derived_seed("simulation", 20260813, 1)
    assert first != derived_seed("sampling", 20260813, 0, 14, 9)


def test_v2_bank_contract_extends_to_100_three_height_seed_blocks():
    assert DEFAULT_BLOCK_COUNT == 100
    assert HEIGHT_SCHEDULES == {
        14: (9, 12, 14),
        24: (15, 20, 24),
        34: (21, 28, 34),
    }
    assert DEVELOPMENT_NAMESPACE == "ctbf-v5-g1-06-existing-screen-v1"
    simulation_seeds = {
        derived_seed("simulation", 20260813, block_index)
        for block_index in range(DEFAULT_BLOCK_COUNT)
    }
    assert len(simulation_seeds) == DEFAULT_BLOCK_COUNT


def test_one_block_bank_smoke_writes_paired_h14_h24_h34_inputs(tmp_path):
    output_root = tmp_path / "bank"
    manifest = generate_bank(
        output_root=output_root,
        block_count=1,
        allow_nonproduction_size=True,
        distance_compute=_injected_cnp2cnp_distance,
        base_seed=20260813,
        created_at_utc="fixture",
    )
    assert manifest["status"] == "complete"
    assert manifest["block_count"] == 1
    assert [case["height"] for case in manifest["cases"]] == [14, 24, 34]
    assert manifest["simulation_height"] == 34
    assert manifest["paired_condition_heights"] == [14, 24, 34]
    _root, loaded = load_bank_manifest(output_root, expected_block_count=1)
    assert loaded["completed_condition_count"] == 3

    run_root = tmp_path / "run"
    run = run_algorithms(
        bank_root=output_root,
        output_root=run_root,
        run_id="fixture-run",
        arm_ids=(
            "classical_partial",
            "biopsy_guided_classical_r4",
            "biopsy_guided_classical_r2",
            "neighbor_joining_baseline",
            "neighbor_joining_hybrid_anticentral_adaptive_v3_plausible_parsimony",
            "temporal_minimum",
            "temporal_minimum_no_time",
        ),
        expected_block_count=1,
        created_at_utc="fixture",
    )
    assert run["status"] == "complete"
    assert run["completed_record_count"] == 21
    assert run["failure_count"] == 0
    assert run["resources"]["record_execution"] == fresh_process_contract(
        CASE_ARM_WORKER_UNIT
    )

    resumable_root = tmp_path / "resumable-run"
    completed = run_algorithms(
        bank_root=output_root,
        output_root=resumable_root,
        run_id="fixture-resumable",
        arm_ids=("classical_partial",),
        expected_block_count=1,
        created_at_utc="fixture",
    )
    interrupted = copy.deepcopy(completed)
    interrupted["status"] = "failure"
    interrupted["records"] = interrupted["records"][:1]
    interrupted["completed_record_count"] = 1
    interrupted["completed_condition_count"] = 0
    interrupted["runner_failure"] = {
        "stage": "runner",
        "type": "FixtureInterruption",
        "message": "fixture",
    }
    interrupted.pop("completed_at_utc")
    write_json(resumable_root / "result.json", interrupted)
    resumed = run_algorithms(
        bank_root=output_root,
        output_root=resumable_root,
        run_id="fixture-resumable",
        arm_ids=("classical_partial",),
        expected_block_count=1,
        resume=True,
    )
    assert resumed["status"] == "complete"
    assert resumed["completed_record_count"] == 3
    assert resumed["resume_history"][-1]["preserved_record_count"] == 1
    assert resumed["resume_history"][-1]["previous_runner_failure"]["type"] == (
        "FixtureInterruption"
    )
    with pytest.raises(ValueError, match="stored run_id"):
        run_algorithms(
            bank_root=output_root,
            output_root=resumable_root,
            run_id="different-run",
            arm_ids=("classical_partial",),
            expected_block_count=1,
            resume=True,
        )
    with pytest.raises(ValueError, match="completed development run"):
        run_algorithms(
            bank_root=output_root,
            output_root=resumable_root,
            run_id="fixture-resumable",
            arm_ids=("classical_partial",),
            expected_block_count=1,
            resume=True,
        )

    extension_root = tmp_path / "partial-top-extension"
    extension = run_algorithms(
        bank_root=output_root,
        output_root=extension_root,
        run_id="fixture-partial-top-extension",
        arm_ids=tuple(spec.arm_id for spec in DEVELOPMENT_EXTENSION_ARM_SPECS),
        expected_block_count=1,
        created_at_utc="fixture",
    )
    assert extension["status"] == "complete"
    assert extension["completed_record_count"] == 66
    assert extension["failure_count"] == 0

    # The production control row was completed before the optional decision
    # audit existed. Ensure the new report accepts that stored run unchanged.
    for record in extension["records"]:
        if record["arm_id"] == PARTIAL_BOTTOM_CONTROL_ID:
            record["reconstruction_metadata"].pop(
                "biopsy_layer_decision_audit",
                None,
            )
    write_json(extension_root / "result.json", extension)

    report_root = tmp_path / "report"
    report = write_report(
        result_roots=[run_root, extension_root],
        output_root=report_root,
        expected_block_count=1,
    )
    assert report["block_count"] == 1
    assert report["schema_version"] == REPORT_SCHEMA_VERSION
    assert report["condition_count"] == 3
    assert report["resource_interpretation"][
        "all_runs_fresh_process_qualified"
    ] is True
    assert report["dependence_contract"]["paired_heights_per_block"] == [14, 24, 34]
    assert len(report["contextual_matched_comparisons"]) == 1
    assert report["families"][PARTIAL_FAMILY]["top_layer_comparison"][
        "control_id"
    ] == "biopsy_guided_classical_r2"
    interaction = report["families"][PARTIAL_FAMILY]["top_radius_interaction"]
    assert interaction["arm_ids"]["binary_by_radius"]["4"] == (
        "biopsy_guided_top_anticentral_binary_r4"
    )
    assert interaction["difference_in_differences"]["complete_case_count"] == 3
    assert interaction["difference_in_differences"]["independent_block_effect"][
        "complete_block_count"
    ] == 1
    bottom = report["families"][PARTIAL_FAMILY]["bottom_layer_comparison"]
    assert bottom["control_id"] == PARTIAL_BOTTOM_CONTROL_ID
    assert bottom["fixed_radius"] == 2
    assert len(bottom["candidate_ids"]) == 6
    assert len(bottom["pairwise_records"]) == 6
    assert len(bottom["mechanism_rows"]) == 7
    control_mechanism = next(
        row
        for row in bottom["mechanism_rows"]
        if row["arm_id"] == PARTIAL_BOTTOM_CONTROL_ID
    )
    assert control_mechanism["mean_child_decision_count"] is None
    assert all(
        row["mean_child_decision_count"] is not None
        for row in bottom["mechanism_rows"]
        if row["arm_id"] != PARTIAL_BOTTOM_CONTROL_ID
    )
    bottom_top = report["families"][PARTIAL_FAMILY]["bottom_top_factorial"]
    assert bottom_top["arm_ids"]["deferred"]["classical"] == (
        PARTIAL_BOTTOM_TOP_INTERACTION_ARM_ID
    )
    assert set(bottom_top["top_effect_by_bottom"]) == {
        "default",
        "deferred",
    }
    assert set(bottom_top["bottom_effect_by_top"]) == {
        "classical",
        "binary_anticentral",
    }
    assert bottom_top["difference_in_differences"]["complete_case_count"] == 3
    full_family = report["families"][BIOPSY_GUIDED_FULL_FAMILY]
    assert full_family["primary_metric"] == "ad_f1"
    assert full_family["incumbent_id"] == BIOPSY_GUIDED_FULL_INCUMBENT_ID
    assert len(full_family["arm_ids"]) == 11
    full_bottom_top = full_family["full_bottom_top_factorial"]
    assert full_bottom_top["arm_ids"]["deferred"]["rooted_labeled_q"] == (
        BIOPSY_GUIDED_FULL_ROOTED_Q_DEFERRED_ID
    )
    assert set(full_bottom_top["top_effect_by_bottom"]) == {
        "default",
        "deferred",
    }
    assert set(full_bottom_top["bottom_effect_by_top"]) == {
        "rooted_labeled_q",
        "binary_anticentral",
    }
    assert full_bottom_top["difference_in_differences"][
        "complete_case_count"
    ] == 3
    full_cross = report["biopsy_guided_full_vs_pooled_incumbent"]
    assert full_cross["pooled_incumbent_id"] == INFERRED_COPY_INCUMBENT_ID
    assert full_cross["principal_biopsy_guided_id"] == (
        BIOPSY_GUIDED_FULL_INCUMBENT_ID
    )
    assert len(full_cross["counterparts"]) == 10
    assert len(full_cross["pairwise_records"]) == 11
    principal_cross = next(
        comparison
        for comparison in full_cross["pairwise_records"]
        if comparison["arm_a"] == BIOPSY_GUIDED_FULL_INCUMBENT_ID
    )
    assert principal_cross["arm_b"] == INFERRED_COPY_INCUMBENT_ID
    assert principal_cross["joint_success_count"] == 3
    assert (report_root / "matrix_partial_combined_wins.csv").is_file()
    assert (
        report_root / "leaderboard_biopsy_guided_full.csv"
    ).is_file()
    assert (
        report_root / "biopsy_guided_full_vs_pooled_incumbent.csv"
    ).is_file()
    assert (report_root / "biopsy_guided_full_counterparts.csv").is_file()
    assert (
        report_root / "biopsy_guided_full_binary_top_effect_by_bottom.csv"
    ).is_file()
    assert (
        report_root / "biopsy_guided_full_deferred_bottom_effect_by_top.csv"
    ).is_file()
    assert (
        report_root
        / "biopsy_guided_full_bottom_top_difference_in_differences.csv"
    ).is_file()
    assert (report_root / "partial_top_layer_vs_r2_classical.csv").is_file()
    assert (report_root / "partial_binary_top_effect_by_radius.csv").is_file()
    assert (report_root / "partial_radius_effect_by_top.csv").is_file()
    assert (
        report_root / "partial_radius_top_difference_in_differences.csv"
    ).is_file()
    assert (report_root / "partial_bottom_layer_vs_default_r2.csv").is_file()
    assert (report_root / "partial_bottom_mechanism_summary.csv").is_file()
    assert (report_root / "partial_binary_top_effect_by_bottom.csv").is_file()
    assert (report_root / "partial_deferred_bottom_effect_by_top.csv").is_file()
    assert (
        report_root / "partial_bottom_top_difference_in_differences.csv"
    ).is_file()
    report_text = (report_root / "report.md").read_text(encoding="utf-8")
    assert "### Partial top-layer screen" in report_text
    assert "H34 mean" in report_text
    assert "H34 W/T/L" in report_text
    assert "### Partial radius x top-reconstruction check" in report_text
    assert "### Partial bottom-layer screen" in report_text
    assert "### Partial bottom x top-reconstruction check" in report_text
    assert "### Fully labeled bottom x top-reconstruction check" in report_text
    assert "## Fully labeled biopsy-guided versus pooled incumbent" in report_text
    assert BIOPSY_GUIDED_FULL_INCUMBENT_ID in report_text
    pairwise_header = (report_root / "pairwise_partial.csv").read_text(
        encoding="utf-8"
    ).splitlines()[0]
    assert "h34_wins" in pairwise_header

    # The documented production audit reads completed pre-isolation runs.
    # Their accuracy/counter records remain valid even though their resource
    # measurements are historical context only.
    extension["schema_version"] = LEGACY_RUN_SCHEMA_VERSION
    run["schema_version"] = LEGACY_RUN_SCHEMA_VERSION
    write_json(extension_root / "result.json", extension)
    write_json(run_root / "result.json", run)

    attachment_audit_root = tmp_path / "full-attachment-audit"
    attachment_audit = run_full_attachment_audit(
        ordered_result_root=extension_root,
        pooled_result_root=run_root,
        output_root=attachment_audit_root,
        expected_block_count=1,
        created_at_utc="fixture",
    )
    assert attachment_audit["schema_version"] == (
        FULL_ATTACHMENT_AUDIT_SCHEMA_VERSION
    )
    assert attachment_audit["condition_count"] == 3
    assert attachment_audit["paired_heights"] == [14, 24, 34]
    assert set(attachment_audit["paired_height_transitions"]) == {
        "H14_to_H24",
        "H24_to_H34",
        "H14_to_H34",
    }
    assert attachment_audit["automatic_replacement_decision_declared"] is False
    assert (attachment_audit_root / "full_attachment_audit.json").is_file()
    attachment_report = (attachment_audit_root / "report.md").read_text(
        encoding="utf-8"
    )
    assert "## Height summaries" in attachment_report
    assert "## Paired height transitions" in attachment_report

    manifest_before_audit = (output_root / "bank_manifest.json").read_bytes()
    audit_root = tmp_path / "mechanism-audit"
    audit = run_mechanism_audit(
        bank_root=output_root,
        output_root=audit_root,
        expected_block_count=1,
        created_at_utc="fixture",
    )
    assert audit["status"] == "complete"
    assert audit["completed_case_count"] == 3
    assert audit["orientation_success_count"] == 3
    assert audit["temporal_pair_success_count"] == 3
    assert audit["failure_count"] == 0
    assert audit["radius_candidate_graph"]["case_count"] == 3
    assert (
        audit["incumbent_orientation_trace"]["combined"]["counts"][
            "merge_count"
        ]
        > 0
    )
    assert audit["temporal_no_time_diagnostic"]["combined"]["case_count"] == 3
    assert (audit_root / "mechanism_audit.json").is_file()
    assert (audit_root / "report.md").is_file()
    assert (output_root / "bank_manifest.json").read_bytes() == manifest_before_audit

    with pytest.raises(ValueError, match="outside the immutable bank"):
        run_mechanism_audit(
            bank_root=output_root,
            output_root=output_root / "forbidden-audit",
            expected_block_count=1,
            created_at_utc="fixture",
        )


def test_incumbent_orientation_trace_preserves_exact_reconstruction():
    payload = _input_payload()
    distance = _distance(payload)
    incumbent = ARM_SPEC_BY_ID[INFERRED_COPY_INCUMBENT_ID]
    expected, _levels, _root, _metadata = reconstruct_development_arm(
        incumbent,
        payload,
        distance,
        reconstruction_seed=991,
    )
    observed, trace = reconstruct_incumbent_with_orientation_trace(
        payload,
        distance,
        reconstruction_seed=991,
    )
    assert canonical_topology_digest(observed) == canonical_topology_digest(expected)
    assert trace["merge_count"] == len(distance.ids) - 1
    assert trace["merge_count"] == (
        trace["one_way_plausible_count"]
        + trace["both_plausible_count"]
        + trace["neither_plausible_count"]
    )
    assert trace["centrality_fallback_count"] == (
        trace["both_plausible_parsimony_tied_count"]
        + trace["neither_plausible_count"]
    )


def test_canonical_topology_digest_is_byte_stable_and_handles_deep_trees():
    fixture = nx.DiGraph()
    fixture.add_nodes_from(
        [
            ("r", {"cell_id": 0}),
            ("a", {"cell_id": 2}),
            ("b", {"cell_id": 1}),
            ("c", {"cell_id": None}),
        ]
    )
    fixture.add_edges_from([("r", "a"), ("r", "b"), ("a", "c")])
    assert canonical_topology_digest(fixture) == (
        "37e9ce784a6fd6f9bfb018cc2f842fb75ad057b4dda179fbdbbce4bfee3d1666"
    )

    deep = nx.DiGraph()
    deep.add_nodes_from(
        (node, {"cell_id": node})
        for node in range(1_500)
    )
    deep.add_edges_from((node, node + 1) for node in range(1_499))
    digest = canonical_topology_digest(deep)
    assert len(digest) == 64
    assert digest == canonical_topology_digest(deep)


def test_incumbent_orientation_trace_exposes_neither_plausible_fallback():
    payload = {
        "schema_version": "ctbf-v5-reconstruction-input-v1",
        "case_id": "neither-plausible",
        "condition_id": "fixture",
        "levels": [
            {
                "biopsy_level": 0,
                "generation": 9,
                "states": [
                    {"state_label": 0, "cnp": [0, 2, 2]},
                ],
            },
            {
                "biopsy_level": 1,
                "generation": 14,
                "states": [
                    {"state_label": 1, "cnp": [2, 0, 0]},
                ],
            }
        ],
    }
    distance = _distance(payload)
    expected, _levels, _root, _metadata = reconstruct_development_arm(
        ARM_SPEC_BY_ID[INFERRED_COPY_INCUMBENT_ID],
        payload,
        distance,
        reconstruction_seed=991,
    )
    observed, trace = reconstruct_incumbent_with_orientation_trace(
        payload,
        distance,
        reconstruction_seed=991,
    )
    assert canonical_topology_digest(observed) == canonical_topology_digest(expected)
    assert trace["merge_count"] == 1
    assert trace["neither_plausible_count"] == 1
    assert trace["neither_plausible_parsimony_would_decide_count"] == 1
    assert trace["centrality_fallback_count"] == 1
    assert trace["centrality_fallback_exact_tie_count"] == 1


@pytest.mark.parametrize("spec", INITIAL_ARM_SPECS, ids=lambda spec: spec.arm_id)
def test_every_initial_arm_is_deterministic_on_the_semantic_fixture(spec):
    payload = _input_payload()
    distance = _distance(payload)
    before_payload = copy.deepcopy(payload)
    before_matrix = np.array(distance.matrix, copy=True)
    first, _levels, _root, _metadata = reconstruct_development_arm(
        spec,
        payload,
        distance,
        reconstruction_seed=991,
    )
    second, _levels, _root, _metadata = reconstruct_development_arm(
        spec,
        payload,
        distance,
        reconstruction_seed=991,
    )
    assert nx.is_arborescence(first)
    assert canonical_topology_digest(first) == canonical_topology_digest(second)
    assert payload == before_payload
    assert np.array_equal(distance.matrix, before_matrix)


@pytest.mark.parametrize(
    "spec",
    DEVELOPMENT_EXTENSION_ARM_SPECS,
    ids=lambda spec: spec.arm_id,
)
def test_every_development_extension_is_deterministic_on_the_semantic_fixture(spec):
    payload = _input_payload()
    distance = _distance(payload)
    first, _levels, _root, first_metadata = reconstruct_development_arm(
        spec,
        payload,
        distance,
        reconstruction_seed=991,
    )
    second, _levels, _root, second_metadata = reconstruct_development_arm(
        spec,
        payload,
        distance,
        reconstruction_seed=991,
    )
    assert nx.is_arborescence(first)
    assert canonical_topology_digest(first) == canonical_topology_digest(second)
    assert first_metadata == second_metadata
    audit = first_metadata["biopsy_layer_decision_audit"]
    assert audit["schema_version"] == "ctbf-biopsy-guided-decision-audit-v1"
    assert (
        audit["selected_parent_count"] + audit["copy_up_count"]
        == audit["child_decision_count"]
    )
    assert (
        audit["tie_parent_selected_count"] + audit["tie_deferred_count"]
        == audit["minimum_distance_tie_count"]
    )
    if (
        spec.top_output_projection
        == TOP_OUTPUT_PROJECTION_CREATED_NODES_UNLABELED
    ):
        assert all(
            attributes.get("genome") is None
            for _node, attributes in first.nodes(data=True)
            if attributes.get("cell_id") is None
        )


def test_every_development_extension_passes_the_complete_d0_gate():
    outcomes = run_semantic_gate(
        DEVELOPMENT_EXTENSION_ARM_SPECS,
        timeout_seconds=300,
        rss_limit_bytes=4 * 1024**3,
    )
    assert {arm_id: outcome["status"] for arm_id, outcome in outcomes.items()} == {
        spec.arm_id: "passed" for spec in DEVELOPMENT_EXTENSION_ARM_SPECS
    }


def test_every_initial_arm_passes_the_complete_d0_gate():
    outcomes = run_semantic_gate(
        INITIAL_ARM_SPECS,
        timeout_seconds=300,
        rss_limit_bytes=4 * 1024**3,
    )
    assert {arm_id: outcome["status"] for arm_id, outcome in outcomes.items()} == {
        spec.arm_id: "passed" for spec in INITIAL_ARM_SPECS
    }


def test_pairwise_report_keeps_frequency_and_rare_severe_loss_separate():
    cases = [
        {
            "case_id": f"b{block:03d}-H{height}",
            "block_index": block,
            "height": height,
        }
        for block in range(100)
        for height in (14, 24, 34)
    ]
    records = {}
    for index, case in enumerate(cases):
        a_score = 0.901 if index < 299 else 0.0
        records[(case["case_id"], "a")] = _success_record(
            case, "a", PARTIAL_FAMILY, a_score
        )
        records[(case["case_id"], "b")] = _success_record(
            case, "b", PARTIAL_FAMILY, 0.9
        )
    comparison = pairwise_comparison(
        "a",
        "b",
        records,
        cases,
        primary_metric="grf",
        complementary_metrics=(),
    )
    combined = comparison["combined_conditions"]
    assert combined["wtl"] == {
        "wins": 299,
        "ties": 0,
        "losses": 1,
        "eligible": 300,
        "win_score": 299 / 300,
    }
    assert combined["effect"]["mean"] < 0
    assert comparison["independent_block_effect"]["complete_block_count"] == 100
    assert (
        comparison["independent_block_effect"]["effect"]["minimum"] < -0.2
    )
    assert comparison["worst_five_or_available_conditions"][0]["case_id"] == (
        cases[-1]["case_id"]
    )


def test_factorial_interaction_compares_binary_top_effect_between_radii():
    cases = [
        {
            "case_id": f"b001-H{height}",
            "block_index": 0,
            "height": height,
        }
        for height in (14, 24, 34)
    ]
    scores = {
        "biopsy_guided_classical_r2": 0.4,
        "biopsy_guided_top_anticentral_binary_r2": 0.6,
        "biopsy_guided_classical_r4": 0.5,
        "biopsy_guided_top_anticentral_binary_r4": 0.6,
    }
    records = {
        (case["case_id"], arm_id): _success_record(
            case,
            arm_id,
            PARTIAL_FAMILY,
            score,
        )
        for case in cases
        for arm_id, score in scores.items()
    }
    interaction = factorial_interaction_comparison(
        records,
        cases,
        primary_metric="grf",
    )
    assert interaction["combined_conditions"]["wtl"] == {
        "wins": 3,
        "ties": 0,
        "losses": 0,
        "eligible": 3,
        "win_score": 1.0,
    }
    assert interaction["independent_block_effect"]["effect"]["mean"] == (
        pytest.approx(0.1)
    )


def test_bottom_top_factorial_compares_binary_effect_between_bottom_policies():
    cases = [
        {
            "case_id": f"b001-H{height}",
            "block_index": 0,
            "height": height,
        }
        for height in (14, 24, 34)
    ]
    scores = {
        "biopsy_guided_classical_r2": 0.4,
        "biopsy_guided_top_anticentral_binary_r2": 0.5,
        PARTIAL_BOTTOM_TOP_INTERACTION_ARM_ID: 0.6,
        "biopsy_guided_top_anticentral_binary_r2_bottom_deferred_tie": 0.8,
    }
    records = {
        (case["case_id"], arm_id): _success_record(
            case,
            arm_id,
            PARTIAL_FAMILY,
            score,
        )
        for case in cases
        for arm_id, score in scores.items()
    }

    interaction = bottom_top_factorial_interaction_comparison(
        records,
        cases,
        primary_metric="grf",
    )

    assert interaction["combined_conditions"]["wtl"] == {
        "wins": 3,
        "ties": 0,
        "losses": 0,
        "eligible": 3,
        "win_score": 1.0,
    }
    assert interaction["independent_block_effect"]["effect"]["mean"] == (
        pytest.approx(0.1)
    )


def test_build_report_treats_300_conditions_as_100_independent_blocks():
    cases = [
        {
            "case_id": f"b{block:03d}-H{height}",
            "block_index": block,
            "height": height,
        }
        for block in range(100)
        for height in (14, 24, 34)
    ]
    records = []
    for case in cases:
        records.append(
            _success_record(case, "classical_partial", PARTIAL_FAMILY, 0.4)
        )
        records.append(
            _success_record(
                case,
                "biopsy_guided_classical_r4",
                PARTIAL_FAMILY,
                0.5,
            )
        )
        records.append(_success_record(case, "candidate", PARTIAL_FAMILY, 0.6))
    result = {
        "run_id": "fixture",
        "bank_id": "fixture-bank",
        "bank_root": "/fixed/bank",
        "block_count": 100,
        "condition_count": 300,
        "arm_specs": [
            {
                "arm_id": "classical_partial",
                "family": PARTIAL_FAMILY,
                "problem": "partial",
                "role": "baseline",
                "primary_metric": "grf",
                "complementary_metrics": [],
            },
            {
                "arm_id": "biopsy_guided_classical_r4",
                "family": PARTIAL_FAMILY,
                "problem": "partial",
                "role": "incumbent",
                "primary_metric": "grf",
                "complementary_metrics": [],
            },
            {
                "arm_id": "candidate",
                "family": PARTIAL_FAMILY,
                "problem": "partial",
                "role": "candidate",
                "primary_metric": "grf",
                "complementary_metrics": [],
            },
        ],
        "records": records,
    }
    report = build_report([result], created_at_utc="fixture")
    assert report["condition_count"] == 300
    assert report["dependence_contract"]["independent_truth_block_count"] == 100
    candidate = next(
        row
        for row in report["families"][PARTIAL_FAMILY]["leaderboard"]
        if row["arm_id"] == "candidate"
    )
    assert candidate["vs_incumbent_wins"] == 300
    assert candidate["vs_incumbent_mean_block_delta"] == pytest.approx(0.1)
