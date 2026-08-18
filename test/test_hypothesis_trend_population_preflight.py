from algorithm_evaluation import hypothesis_trend_population_preflight as preflight


def test_population_case_plan_deduplicates_shared_length_height_cell():
    plan = preflight.build_population_case_plan(
        lengths=[10, 20],
        heights=[8, 9],
        length_trend_height=8,
        height_trend_length=10,
    )

    assert [row["case_key"] for row in plan] == [
        "length-L10-H8",
        "length-L20-H8",
        "height-L10-H9",
    ]
    shared = plan[0]
    assert shared["trend_membership"] == ["length_trend", "height_trend"]
    assert shared["unreduced_binary_total_node_bound"] == 511


def test_small_population_preflight_runs_simulation_only_and_writes_compact_counts():
    report = preflight.run_population_preflight(
        lengths=[10],
        heights=[7, 8],
        length_trend_height=7,
        height_trend_length=10,
        base_seed=11,
        timeout_seconds_per_case=30,
        rss_limit_bytes=1024**3,
        created_at_utc="2026-08-05T00:00:00+00:00",
    )

    assert report["schema_version"] == preflight.SCHEMA_VERSION
    assert report["analysis_role"] == preflight.PREFLIGHT_ROLE
    assert report["scientific_role"] == {
        "parent_program_role": "discovery_hypothesis_trend",
        "paper_evidence_allowed": False,
        "simulation_only": True,
        "cnp2cnp_run": False,
        "reconstruction_run": False,
        "evaluation_run": False,
    }
    assert report["summary"]["status_counts"] == {"success": 2}
    assert len(report["cases"]) == 2
    for record in report["cases"]:
        assert record["status"] == "success"
        summary = record["summary"]
        assert 1 <= summary["truth_node_count"] <= record[
            "unreduced_binary_total_node_bound"
        ]
        assert summary["fixed_fraction_profile_union_upper_bound"] >= (
            summary["fixed_fraction_canonical_prefix_profile_union_count"]
        )
        assert summary["fixed_fraction_bidirectional_cnp2cnp_call_bound"] == (
            summary["fixed_fraction_profile_union_upper_bound"]
            * (summary["fixed_fraction_profile_union_upper_bound"] - 1)
        )


def test_static_bound_skips_cases_before_simulation():
    report = preflight.run_population_preflight(
        lengths=[10],
        heights=[8, 9],
        length_trend_height=8,
        height_trend_length=10,
        static_node_cap=500,
        created_at_utc="2026-08-05T00:00:00+00:00",
    )

    assert report["summary"]["status_counts"] == {"not_run_static_bound": 2}
    assert all(record["runtime"] is None for record in report["cases"])
    assert all(record["summary"] is None for record in report["cases"])
