import networkx as nx
import pytest

from evaluation_contract import evaluate_tree_pair

from algorithm_evaluation.grf_scale_diagnostic import (
    ANCHOR_CONDITION,
    GRF_SCALE_DIAGNOSTIC_SCHEMA_VERSION,
    _complete_block_values,
    _correlation,
    _equal_count_strata,
    _paired_effect_summary,
    _rankdata,
    _relative_distance_reduction,
    build_report,
)
from algorithm_evaluation.paper_pipeline_contract import (
    ANALYSIS_SCHEMA_VERSION,
    EXPECTED_INVENTORY_SCHEMA_VERSION,
    REGISTERED_ARM_SPECS,
    REGISTERED_CLEAN_EXPERIMENT,
    write_checksum_file,
    write_json_atomic,
)


def _paired_rows(raw_gain=0.1, relative_reduction=0.125):
    rows = []
    for replicate in (1, 2):
        for regime_index, regime in enumerate(("a", "b", "c")):
            rows.append(
                {
                    "case_id": f"r{replicate}-{regime}",
                    "replicate": replicate,
                    "regime_id": regime,
                    "raw_grf_gain": raw_gain + regime_index * 0.01,
                    "relative_distance_reduction": relative_reduction,
                    "left_grf": 0.3,
                    "right_grf": 0.2,
                    "truth_node_count": 10 + regime_index,
                    "left_reconstructed_to_true_node_ratio": 0.5,
                    "right_reconstructed_to_true_node_ratio": 0.4,
                }
            )
    return rows


def test_relative_distance_reduction_uses_remaining_distance_not_small_similarity():
    result = _relative_distance_reduction(0.0067, 0.0065)

    assert result == pytest.approx(0.0002 / 0.9935)
    assert result != pytest.approx(0.0002 / 0.0065)
    assert _relative_distance_reduction(1.0, 1.0) is None


def test_complete_blocks_require_every_regime_and_average_within_replicate():
    rows = _paired_rows()

    assert _complete_block_values(
        rows,
        value_key="raw_grf_gain",
        regimes=("a", "b", "c"),
    ) == pytest.approx([0.11, 0.11])
    assert _complete_block_values(
        rows[:-1],
        value_key="raw_grf_gain",
        regimes=("a", "b", "c"),
    ) == pytest.approx([0.11])


def test_paired_summary_keeps_magnitude_ordering_and_block_units_separate():
    summary = _paired_effect_summary(
        _paired_rows(),
        regimes=("a", "b", "c"),
        bootstrap_repetitions=1000,
        bootstrap_seed=17,
        tolerance=1e-12,
    )

    assert summary["complete_pair_count"] == 6
    assert summary["raw_gain_block_analysis"]["complete_block_count"] == 2
    assert summary["raw_gain_block_analysis"]["summary"]["mean"] == pytest.approx(0.11)
    assert summary["relative_distance_reduction_block_analysis"]["summary"]["mean"] == pytest.approx(0.125)
    assert summary["wins_ties_losses"]["win_probability_with_half_ties"] == 1.0
    assert summary["wins_ties_losses"]["matched_rank_biserial"] == 1.0
    assert summary["materiality_status"] == "not_assessed_no_calibrated_scale_portable_threshold"


def test_rank_correlation_handles_ties_and_constant_inputs_explicitly():
    assert _rankdata([30, 10, 10, 20]) == pytest.approx([4.0, 1.5, 1.5, 3.0])
    rows = [
        {"x": 1, "y": 3},
        {"x": 2, "y": 2},
        {"x": 3, "y": 1},
    ]
    assert _correlation(rows, "x", "y", kind="spearman")["value"] == pytest.approx(-1.0)
    constant = [{"x": 1, "y": 1}, {"x": 1, "y": 2}]
    assert _correlation(constant, "x", "y", kind="pearson")["status"] == "undefined_constant_input"


def test_equal_count_strata_are_descriptive_and_cover_every_row_once():
    rows = [
        {"case_id": str(index), "size": index, "effect": index / 10}
        for index in range(1, 10)
    ]

    strata = _equal_count_strata(
        rows,
        predictor="size",
        outcomes=("effect",),
    )

    assert len(strata) == 4
    assert sum(stratum["count"] for stratum in strata) == len(rows)
    assert strata[0]["predictor_minimum"] == 1
    assert strata[-1]["predictor_maximum"] == 9


def _tree():
    tree = nx.DiGraph()
    tree.add_node("root", cell_id="A")
    tree.add_node("leaf", cell_id="B")
    tree.add_edge("root", "leaf")
    return tree


def test_closed_root_report_is_read_only_compact_and_keeps_registered_audit(tmp_path):
    regimes = ("a", "b", "c")
    cases = []
    evaluation = evaluate_tree_pair(_tree(), _tree(), ["A", "B"])
    for regime in regimes:
        case_id = f"case-{regime}"
        cases.append(
            {
                "case_id": case_id,
                "regime_id": regime,
                "replicate": 1,
                "condition_ids": [ANCHOR_CONDITION],
            }
        )
        condition_root = tmp_path / "cases" / case_id / "conditions" / ANCHOR_CONDITION
        write_json_atomic(
            condition_root / "input.json",
            {
                "schema_version": "ctbf-v5-reconstruction-input-v1",
                "case_id": case_id,
                "condition_id": ANCHOR_CONDITION,
                "fraction": 0.5,
                "schedule_id": "L3",
                "levels": [],
            },
        )
        for arm, _algorithm in REGISTERED_ARM_SPECS:
            write_json_atomic(condition_root / "arms" / arm / "evaluation.json", evaluation)

    write_json_atomic(
        tmp_path / "expected_inventory.json",
        {
            "schema_version": EXPECTED_INVENTORY_SCHEMA_VERSION,
            "experiment_id": REGISTERED_CLEAN_EXPERIMENT,
            "cases": cases,
        },
    )
    write_json_atomic(
        tmp_path / "design_manifest.snapshot.json",
        {
            "experiments": {
                "clean_confirmation": {
                    "regime_ids": list(regimes),
                    "analysis": {
                        "bootstrap_repetitions": 100,
                        "win_tie_loss_tolerance": 1e-12,
                    },
                }
            },
            "seed_contract": {
                "experiments": [
                    {
                        "experiment_id": REGISTERED_CLEAN_EXPERIMENT,
                        "analysis_seeds": {"block_bootstrap": 19},
                    }
                ]
            },
        },
    )
    write_json_atomic(
        tmp_path / "analysis" / ANALYSIS_SCHEMA_VERSION / "summary.json",
        {
            "schema_version": ANALYSIS_SCHEMA_VERSION,
            "secondary_partial_anchor": {
                "material_threshold": 0.01,
                "unadjusted_statistical_support": False,
                "block_effect_summary": {"count": 1, "mean": 0.0},
                "bootstrap_interval": {"status": "success", "lower": 0.0, "upper": 0.0},
            },
        },
    )
    write_checksum_file(tmp_path, "raw_checksums.sha256", include_analysis=False)
    write_checksum_file(tmp_path, "complete_checksums.sha256", include_analysis=True)

    report = build_report(tmp_path)

    assert report["schema_version"] == GRF_SCALE_DIAGNOSTIC_SCHEMA_VERSION
    assert report["registered_decision_audit"]["registered_material_threshold"] == 0.01
    assert report["materiality_policy"]["future_fixed_raw_grf_threshold"] == "retired_not_scale_portable"
    partial = report["contrasts"]["biopsy_guided_classical_minus_classical_partial"]
    assert partial["anchor"]["complete_pair_count"] == 3
    assert partial["anchor"]["raw_paired_grf_gain"]["mean"] == 0.0
    assert report["case_rows_written"] is False
