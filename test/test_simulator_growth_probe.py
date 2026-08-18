import pytest

from algorithm_evaluation.simulator_growth_probe import (
    DEFAULT_BASE_CONFIG,
    _histogram_number_summary,
    _replicate_seed,
    format_generation_table,
    run_growth_probe,
)


def test_growth_probe_is_deterministic_compact_and_simulation_only():
    kwargs = {
        "base_config_path": DEFAULT_BASE_CONFIG,
        "generations": 3,
        "replicates": 2,
        "base_seed": 17,
        "created_at_utc": "2026-08-12T00:00:00+00:00",
    }
    first = run_growth_probe(**kwargs)
    second = run_growth_probe(**kwargs)

    assert first == second
    assert first["scientific_role"] == {
        "paper_evidence_allowed": False,
        "simulation_only": True,
        "truth_trees_serialized": False,
        "biopsies_sampled": False,
        "cnp2cnp_run": False,
        "reconstruction_run": False,
        "evaluation_run": False,
    }
    assert first["input"]["effective_config"]["NUMBER_OF_GENERATIONS"] == 3
    assert first["summary"]["outcome_counts"] == {"completed": 2}
    assert len(first["summary"]["by_generation"]) == 4
    assert all(
        row["complete_run_count"] == 2
        for row in first["summary"]["by_generation"]
    )
    assert all("tree" not in key and "profile" not in key for key in first["runs"][0])
    zero_rows = first["summary"]["zero_burden_by_generation"]
    assert len(zero_rows) == 4
    assert zero_rows[0]["profile_count"] == 2
    assert zero_rows[0]["zero_bin_fraction"]["maximum"] == 0.0
    assert all(row["all_zero_profile_count"] == 0 for row in zero_rows)
    assert set(first["summary"]["event_selection"]) == {
        "segmental_gain_loss_balance",
        "interval_footprint_length",
        "interval_footprint_count",
        "viability",
    }


def test_growth_probe_reports_the_two_prediction_curves_and_full_table():
    report = run_growth_probe(
        generations=8,
        replicates=1,
        base_seed=23,
        created_at_utc="2026-08-12T00:00:00+00:00",
    )
    prediction = report["summary"]["prediction"]

    assert prediction[
        "probability_at_least_one_segmental_start_per_attempt"
    ] == pytest.approx(1.0 - 0.999**100)
    assert prediction["earlier_first_order_growth_factor"] == pytest.approx(
        1.0 + 2.0 * (1.0 - 0.999**100)
    )
    assert prediction[
        "distinct_child_no_recurrence_growth_factor"
    ] < prediction["earlier_first_order_growth_factor"]
    table = format_generation_table(report)
    assert "| 0 |" in table
    assert "| 8 |" in table
    assert "earlier prediction" in table
    assert "CN0 burden" in table
    assert "viability rejections" in table


def test_growth_probe_seed_namespace_is_stable_and_index_specific():
    assert _replicate_seed(11, 0) == _replicate_seed(11, 0)
    assert _replicate_seed(11, 0) != _replicate_seed(11, 1)
    with pytest.raises(ValueError, match="nonnegative"):
        _replicate_seed(-1, 0)


def test_weighted_histogram_summary_matches_linear_quantiles():
    summary = _histogram_number_summary({0: 1, 10: 2, 50: 1}, scale=0.01)

    assert summary == pytest.approx(
        {
            "observation_count": 4,
            "minimum": 0.0,
            "q25": 0.075,
            "median": 0.1,
            "mean": 0.175,
            "q75": 0.2,
            "p95": 0.44,
            "maximum": 0.5,
        }
    )
