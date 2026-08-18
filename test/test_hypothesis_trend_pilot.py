import json

import pytest

from algorithm_evaluation import hypothesis_trend_pilot as pilot


def test_integer_l1_ball_formula_matches_hand_checkable_counts():
    assert pilot.integer_l1_shell_size(1, 0) == 1
    assert pilot.integer_l1_shell_size(1, 1) == 2
    assert pilot.integer_l1_shell_size(1, 2) == 2
    assert pilot.integer_l1_ball_size(1, 2) == 5

    assert pilot.integer_l1_shell_size(2, 1) == 4
    assert pilot.integer_l1_shell_size(2, 2) == 8
    assert pilot.integer_l1_ball_size(2, 2) == 13

    assert pilot.integer_l1_shell_size(10, 1) == 20
    assert pilot.integer_l1_shell_size(10, 2) == 200
    assert pilot.integer_l1_ball_size(10, 2) == 221
    assert pilot.interval_event_description_count(10) == 110


def test_default_report_separates_exact_theory_from_untested_predictions():
    report = pilot.build_pilot_report(created_at_utc="2026-08-05T00:00:00+00:00")

    assert report["schema_version"] == pilot.SCHEMA_VERSION
    assert report["analysis_role"] == "discovery_hypothesis_trend"
    assert report["theory"]["paper_evidence_allowed"] is False
    assert report["prediction_audit"]["result_status"] == {
        "H1a": "exact_theory_calculated",
        "H1b": "not_tested_requires_simulation",
        "H1c": "not_tested_requires_candidate_graph_instrumentation",
        "H2": "not_tested_requires_instrumented_reconstruction",
        "H3": "not_tested_requires_simulation",
        "H4": "not_tested_requires_evaluator_fixture",
        "H5": "not_tested_requires_simulation",
        "H6": "not_tested_requires_resolution_projection",
    }
    assert report["provenance"]["reads_existing_result_corpus"] is False
    assert report["provenance"]["runs_cnp2cnp"] is False

    rows = {
        row["genome_length"]: row
        for row in report["theory"]["rows"]
    }
    assert rows[50]["exact_shell_size_by_radius"]["2"] == 5000
    assert rows[50]["closed_ball_size_by_radius"]["2"] == 5101
    assert rows[100]["exact_shell_size_by_radius"]["2"] == 20000
    assert rows[100]["closed_ball_size_by_radius"]["2"] == 20201


def test_provisional_plan_scales_event_probability_and_separates_trends():
    report = pilot.build_pilot_report(created_at_utc="2026-08-05T00:00:00+00:00")
    plan = report["prospective_plan"]

    assert plan["paper_factor_levels_frozen"] is False
    assert plan["seed_count_frozen"] is False
    assert [row["case_key"] for row in plan["length_trend"]] == [
        "length-L10-H8",
        "length-L50-H8",
        "length-L100-H8",
    ]
    assert [row["cna_event_probability"] for row in plan["length_trend"]] == [
        0.01,
        0.002,
        0.001,
    ]
    assert [row["relative_biopsy_generations"] for row in plan["height_trend"]] == [
        [3, 6, 8],
        [5, 9, 12],
        [7, 11, 16],
    ]
    assert [
        row["unreduced_binary_total_node_bound"]
        for row in plan["height_trend"]
    ] == [511, 8191, 131071]
    assert plan["observation_arms"]["fixed_fraction"]["status"] == (
        "blocked_until_population_and_profile_cap_preflight"
    )
    assert len(plan["endpoint_interaction"]) == 4


def test_cli_writes_compact_versioned_json_and_refuses_silent_overwrite(tmp_path, capsys):
    output = tmp_path / "pilot.json"

    assert pilot.main(["--output", str(output)]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    printed = json.loads(capsys.readouterr().out)

    assert payload["schema_version"] == pilot.SCHEMA_VERSION
    assert printed["H1a_status"] == "exact_theory_calculated"
    assert printed["point_event_counts"]["10"]["radius_two_ball"] == 221
    assert printed["height_unreduced_total_node_bounds"] == {
        "8": 511,
        "12": 8191,
        "16": 131071,
    }
    assert printed["simulation_status"] == "not_run_by_T0"

    with pytest.raises(FileExistsError, match="Output already exists"):
        pilot.main(["--output", str(output)])


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"lengths": [10, 10]}, "must not contain duplicates"),
        ({"heights": [0]}, "positive integers"),
        ({"max_radius": 13}, "outside this bounded theory pilot"),
        ({"expected_cna_starts": 11, "lengths": [10]}, "impossible"),
    ],
)
def test_invalid_pilot_design_is_rejected(arguments, message):
    with pytest.raises(ValueError, match=message):
        pilot.build_pilot_report(**arguments)
