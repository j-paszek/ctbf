import copy
import json

from algorithm_evaluation import hypothesis_height_schedule_preflight as preflight


def _fake_summary(_config_path, _seed, schedule, states_per_level):
    rows = [
        {
            "generation": int(generation),
            "occurrence_count": states_per_level + generation,
            "unique_state_count": states_per_level + 1,
            "fixed_budget_shortfall": 0,
            "eligible_for_fixed_budget": True,
        }
        for generation in schedule
    ]
    return {
        "truth_node_count": 100 + int(schedule[-1]),
        "truth_edge_count": 99 + int(schedule[-1]),
        "realized_max_generation": int(schedule[-1]),
        "schedule": rows,
        "minimum_unique_state_count": states_per_level + 1,
        "eligible_for_fixed_budget": True,
        "insufficient_generation_count": 0,
    }


def _fake_summary_with_h8_shortfall(config_path, seed, schedule, states_per_level):
    summary = _fake_summary(config_path, seed, schedule, states_per_level)
    if schedule[-1] == 8:
        summary["schedule"][0].update(
            {
                "unique_state_count": states_per_level - 1,
                "fixed_budget_shortfall": 1,
                "eligible_for_fixed_budget": False,
            }
        )
        summary["minimum_unique_state_count"] = states_per_level - 1
        summary["eligible_for_fixed_budget"] = False
        summary["insufficient_generation_count"] = 1
    return summary


def test_preflight_freezes_six_unique_states_at_half_three_quarters_and_full_height():
    report = preflight.run_height_schedule_preflight(
        replicates=2,
        base_seed=41,
        simulation_compute=_fake_summary,
        created_at_utc="2026-08-10T00:00:00+00:00",
    )

    assert report["status"] == "complete"
    assert report["input"]["sampling_unit"] == (
        "canonical_unique_cnp_genotype_state"
    )
    assert report["input"]["states_per_level"] == 6
    assert report["input"]["relative_biopsy_positions"] == [0.5, 0.75, 1.0]
    assert report["input"]["generation_schedule_by_height"] == {
        "8": [4, 6, 8],
        "12": [6, 9, 12],
        "16": [8, 12, 16],
    }
    assert report["input"]["percentage_sampling"] is False
    assert report["input"]["population_abundance_weighted"] is False
    assert report["scientific_role"]["profile_selection_run"] is False
    assert report["scientific_role"]["cnp2cnp_run"] is False
    assert report["scientific_role"]["reconstruction_run"] is False
    assert report["scientific_role"]["evaluation_run"] is False
    assert report["resource_bound"]["planned_case_count"] == 6
    assert report["resource_bound"]["maximum_observation_count_if_later_sampled"] == 18
    assert report["resource_bound"]["profiles_selected_in_this_preflight"] == 0
    assert report["aggregate"]["all_planned_cases_eligible"] is True
    assert report["aggregate"]["complete_eligible_replicate_block_count"] == 2

    by_replicate = {}
    for case in report["cases"]:
        by_replicate.setdefault(case["replicate_index"], set()).add(
            case["simulation_seed"]
        )
    assert all(len(seeds) == 1 for seeds in by_replicate.values())
    assert len({next(iter(seeds)) for seeds in by_replicate.values()}) == 2
    assert '"genome":' not in json.dumps(report)


def test_insufficient_states_are_a_retained_design_outcome_not_an_operational_failure():
    report = preflight.run_height_schedule_preflight(
        replicates=1,
        base_seed=43,
        simulation_compute=_fake_summary_with_h8_shortfall,
        created_at_utc="2026-08-10T00:00:00+00:00",
    )

    assert report["status"] == "complete"
    assert [case["status"] for case in report["cases"]] == [
        "insufficient_unique_states",
        "eligible",
        "eligible",
    ]
    assert report["aggregate"]["all_planned_cases_eligible"] is False
    assert report["aggregate"]["availability_supported_for_owner_review"] is False
    assert report["aggregate"]["complete_eligible_replicate_block_count"] == 0


def test_validator_rejects_percentage_sampling_or_a_changed_schedule():
    report = preflight.run_height_schedule_preflight(
        replicates=1,
        simulation_compute=_fake_summary,
        created_at_utc="2026-08-10T00:00:00+00:00",
    )

    percentage = copy.deepcopy(report)
    percentage["input"]["percentage_sampling"] = True
    try:
        preflight.validate_height_schedule_report(percentage)
    except ValueError as error:
        assert "input contract mismatch" in str(error)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("Expected percentage sampling to be rejected.")

    changed_schedule = copy.deepcopy(report)
    changed_schedule["cases"][0]["schedule"] = [3, 6, 8]
    try:
        preflight.validate_height_schedule_report(changed_schedule)
    except ValueError as error:
        assert "unapproved schedule" in str(error)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("Expected the changed schedule to be rejected.")
